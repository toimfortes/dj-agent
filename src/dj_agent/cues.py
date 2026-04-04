"""Phrase-aware cue point detection.

Improvements over the original:
- PSSI-first: reads Rekordbox song structure if available
- Adaptive k based on track duration
- Dual thresholds (hysteresis) instead of a single 0.6
- Phrase-level snapping (8-bar boundaries, not random beats)
- Smart intro (not always at 0:00)
"""

from __future__ import annotations

from pathlib import Path

from pathlib import Path

import librosa
import numpy as np

from .config import CueConfig
from .types import CuePoint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cue_points(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    duration: float,
    config: CueConfig | None = None,
    anlz_path: str | Path | None = None,
) -> list[CuePoint]:
    """Detect structural cue points from audio.

    If *anlz_path* is provided and ``config.use_pssi`` is True, tries to
    read Rekordbox's PSSI (song structure) tag first.  Falls back to
    audio-based detection if PSSI is unavailable.

    Returns a list of :class:`CuePoint` sorted by position.
    """
    if config is None:
        from .config import get_config
        config = get_config().cues

    # PSSI-first: read Rekordbox song structure if available
    if config.use_pssi and anlz_path:
        pssi_cues = detect_cue_points_from_pssi(anlz_path)
        if pssi_cues:
            return pssi_cues[:8]

    k = _adaptive_k(duration, config.min_segments, config.max_segments)

    # Mel spectrogram for segmentation
    S = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)

    # RMS energy per frame
    rms = librosa.feature.rms(y=audio, hop_length=512)[0]

    # Agglomerative clustering
    try:
        bounds = librosa.segment.agglomerative(S_db, k=k)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        try:
            bounds = librosa.segment.agglomerative(S_db, k=max(4, k // 2))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return []

    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=512)

    # Snap to phrase boundaries
    bpm = bpm if bpm and bpm > 0 else 120.0  # default to 120 if missing
    beat_sec = 60.0 / bpm
    phrase_sec = config.phrase_length_bars * beat_sec * 4  # 4 beats/bar
    bound_times = np.array([_snap_to_phrase(t, phrase_sec) for t in bound_times])

    # Build segments with energy
    segments = _build_segments(bound_times, rms, sr, duration)
    if not segments:
        return []

    # Normalise segment energy
    max_rms = max(s["energy"] for s in segments)
    if max_rms == 0:
        return []
    for s in segments:
        s["energy"] /= max_rms

    # Classify transitions using dual thresholds (hysteresis)
    cues = _classify_transitions(
        segments, duration, config, phrase_sec,
    )

    # Deduplicate cues that are too close
    cues = _deduplicate(cues, config.min_cue_distance_sec)

    # Limit to 8 (Rekordbox hot-cue max)
    return cues[:8]


def detect_cue_points_from_pssi(anlz_path: str | Path) -> list[CuePoint] | None:
    """Try to read cue points from Rekordbox PSSI (song structure) tag.

    Returns ``None`` if the PSSI tag is not present or pyrekordbox cannot
    parse it.
    """
    try:
        from pyrekordbox.anlz import AnlzFile  # type: ignore[import-untyped]

        anlz = AnlzFile.parse_file(str(anlz_path))
        pssi = None
        for tag in anlz.tags:
            if tag.type == "PSSI":
                pssi = tag
                break
        if pssi is None:
            return None

        label_map = {
            "intro": ("Intro", "green"),
            "verse": ("Verse", "green"),
            "chorus": ("Drop", "red"),
            "drop": ("Drop", "red"),
            "bridge": ("Breakdown", "blue"),
            "breakdown": ("Breakdown", "blue"),
            "outro": ("Outro", "yellow"),
        }

        cues: list[CuePoint] = []
        for entry in pssi.entries:
            label = (getattr(entry, "mood", "") or "").lower()
            for key, (name, colour) in label_map.items():
                if key in label:
                    # PSSI entries may have 'beat' (beat index) or 'start'
                    # (time in ms). Try time-based first, fall back to beat.
                    if hasattr(entry, "start") and entry.start is not None:
                        pos_ms = int(entry.start)
                    elif hasattr(entry, "beat") and entry.beat is not None:
                        # beat is a beat index — this is approximate without BPM
                        pos_ms = int(entry.beat * 500)  # ~120 BPM default
                    else:
                        pos_ms = 0
                    cues.append(CuePoint(
                        position_ms=pos_ms,
                        name=name,
                        colour=colour,
                    ))
                    break

        return cues if cues else None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _adaptive_k(duration: float, min_k: int, max_k: int) -> int:
    """Scale segment count with duration (~1 segment per 30 s)."""
    k = int(duration / 30)
    return max(min_k, min(max_k, k))


def _snap_to_phrase(time_sec: float, phrase_sec: float) -> float:
    """Snap a time to the nearest phrase boundary."""
    if phrase_sec <= 0:
        return time_sec
    return round(time_sec / phrase_sec) * phrase_sec


def _build_segments(
    bound_times: np.ndarray,
    rms: np.ndarray,
    sr: int,
    duration: float,
) -> list[dict]:
    segments: list[dict] = []
    for i in range(len(bound_times)):
        start = float(bound_times[i])
        end = float(bound_times[i + 1]) if i + 1 < len(bound_times) else duration
        sf = librosa.time_to_frames(start, sr=sr, hop_length=512)
        ef = min(librosa.time_to_frames(end, sr=sr, hop_length=512), len(rms))
        seg_rms = float(np.mean(rms[sf:ef])) if ef > sf else 0.0
        segments.append({"start": start, "end": end, "energy": seg_rms})
    return segments


def _classify_transitions(
    segments: list[dict],
    duration: float,
    config: CueConfig,
    phrase_sec: float,
) -> list[CuePoint]:
    """Use dual thresholds to classify transitions."""
    lo = config.energy_threshold_low
    hi = config.energy_threshold_high

    cues: list[CuePoint] = []

    # Smart intro: first segment above low threshold
    for s in segments:
        if s["energy"] >= lo:
            pos = int(s["start"] * 1000)
            cues.append(CuePoint(position_ms=pos, name="Intro", colour="green"))
            break
    else:
        # Fallback: start of track
        cues.append(CuePoint(position_ms=0, name="Intro", colour="green"))

    # Drops and breakdowns
    for i in range(1, len(segments)):
        prev_e = segments[i - 1]["energy"]
        curr_e = segments[i]["energy"]
        pos = int(segments[i]["start"] * 1000)

        if prev_e < lo and curr_e >= hi:
            cues.append(CuePoint(position_ms=pos, name="Drop", colour="red"))
        elif prev_e >= hi and curr_e < lo:
            cues.append(CuePoint(position_ms=pos, name="Breakdown", colour="blue"))

    # Outro: last low-energy segment after 60% of track
    for s in reversed(segments):
        if s["energy"] < lo and s["start"] > duration * 0.60:
            pos = int(s["start"] * 1000)
            cues.append(CuePoint(position_ms=pos, name="Outro", colour="yellow"))
            break

    return cues


def _deduplicate(cues: list[CuePoint], min_dist_sec: float) -> list[CuePoint]:
    """Remove cues that are too close together, keeping earlier ones."""
    if not cues:
        return cues
    cues.sort(key=lambda c: c.position_ms)
    result = [cues[0]]
    for cue in cues[1:]:
        if (cue.position_ms - result[-1].position_ms) >= min_dist_sec * 1000:
            result.append(cue)
    return result
