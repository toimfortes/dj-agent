"""Full analysis pipeline — analyze a track and store ALL results to memory.

This is the central function that wires every module together.
It stores results so they persist across sessions and can be
written to Rekordbox DB/XML when ready.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .audio import load_audio, measure_loudness
from .calibration import apply_calibration
from .cleanup import (
    cleanup_artist,
    cleanup_title,
    extract_featured_artists,
    normalize_for_split,
    smart_title_case,
    split_artist_from_title,
)
from .config import get_config
from .cues import detect_cue_points
from .energy import calculate_energy, energy_to_colour, energy_to_colour_id
from .health import generate_health_report
from .keydetect import detect_key, detect_tuning
from .memory import load_memory, save_memory, store_track_analysis, get_track_analysis
from .quality import check_audio_quality

_log = logging.getLogger(__name__)


def analyse_track_full(
    path: str | Path,
    memory: dict[str, Any] | None = None,
    skip_if_cached: bool = True,
    include_vocals: bool = True,
    include_mood: bool = True,
    include_reasoning: bool = False,
    reasoning_backend: str = "auto",
) -> dict[str, Any]:
    """Run the FULL analysis pipeline on one track and store to memory.

    Returns a dict with ALL analysis results. If skip_if_cached is True
    and the track was already analysed, returns the cached result.

    Capabilities exercised:
    - LUFS measurement (integrated, peak, LRA)
    - Energy scoring (1-10, LUFS-based with 6 features)
    - Key detection (Essentia EDMA or librosa fallback)
    - Tuning offset detection (432Hz detection)
    - Cue point detection (phrase-aware, adaptive k, dual thresholds)
    - BPM detection (librosa beat tracking)
    - Title cleanup (12-step pipeline: HTML entities, watermarks, store IDs, etc.)
    - Artist/title splitting (delimiter + known-artist matching)
    - Featured artist extraction (feat., vs., b2b, & — with duo protection)
    - Smart title casing (DJ-aware: preserves DJ, SOS, UK, VIP, etc.)
    - Genre casing (always title case)
    - Audio quality validation (fake lossless, clipping, silence, format)
    - Vocal detection (Essentia or librosa heuristic)
    - Mood classification (Essentia 5-mood or librosa heuristic)
    - AI vibe analysis (Gemini/Flamingo — optional)
    - Energy-to-colour mapping (Rekordbox colour IDs)
    """
    path = Path(path)
    config = get_config()

    # Check cache
    if skip_if_cached and memory:
        cached = get_track_analysis(memory, path)
        if cached:
            _log.debug("Cache hit for %s", path.name)
            return cached

    result: dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
    }
    start = time.time()

    # ── LUFS ──────────────────────────────────────────────────────
    try:
        loud = measure_loudness(path)
        result["lufs_integrated"] = round(loud.integrated_lufs, 1)
        result["sample_peak_dbfs"] = round(loud.sample_peak_dbfs, 1)
        result["loudness_range_lu"] = round(loud.loudness_range_lu, 1)
        result["short_term_max_lufs"] = round(loud.short_term_max_lufs, 1)
    except Exception as e:
        _log.warning("LUFS failed for %s: %s", path.name, e)
        result["lufs_integrated"] = -100.0

    # ── Audio load + BPM ──────────────────────────────────────────
    try:
        y, sr = load_audio(path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo = float(np.asarray(librosa.beat.beat_track(y=y, sr=sr)[0]).item())
        result["duration_sec"] = round(duration, 1)
        result["bpm"] = round(tempo, 1)
    except Exception as e:
        _log.warning("Audio load failed for %s: %s", path.name, e)
        result["duration_sec"] = 0
        result["bpm"] = 0
        y, sr, duration, tempo = None, 22050, 0, 0

    # ── Energy ────────────────────────────────────────────────────
    if y is not None:
        try:
            energy = calculate_energy(
                y, sr, bpm=tempo,
                loudness_lufs=result.get("lufs_integrated", -20),
            )
            result["energy"] = energy.calibrated_score
            result["energy_raw"] = round(energy.raw_score, 3)
            result["energy_colour"] = energy_to_colour(energy.calibrated_score)
            result["energy_colour_id"] = energy_to_colour_id(energy.calibrated_score)
            result["spectral_centroid"] = round(energy.spectral_centroid_mean, 1)
            result["onset_density"] = round(energy.onset_density, 2)
            result["bass_ratio"] = round(energy.bass_ratio, 3)
            result["dynamic_range"] = round(energy.dynamic_range, 1)
        except Exception as e:
            _log.warning("Energy failed for %s: %s", path.name, e)

    # ── Key detection ─────────────────────────────────────────────
    try:
        key = detect_key(path)
        result["key"] = key.key
        result["camelot"] = key.camelot
        result["key_confidence"] = round(key.confidence, 2)
        result["key_method"] = key.method
    except Exception as e:
        _log.warning("Key failed for %s: %s", path.name, e)

    # ── Tuning ────────────────────────────────────────────────────
    try:
        result["tuning_offset"] = round(detect_tuning(path), 3)
    except Exception:
        pass

    # ── Cue points ────────────────────────────────────────────────
    if y is not None and duration > 0 and tempo > 0:
        try:
            cues = detect_cue_points(y, sr, bpm=tempo, duration=duration)
            result["cues"] = [
                {"name": c.name, "position_ms": c.position_ms,
                 "colour": c.colour, "confidence": c.confidence}
                for c in cues
            ]
        except Exception as e:
            _log.warning("Cues failed for %s: %s", path.name, e)

    # ── Title cleanup ─────────────────────────────────────────────
    try:
        raw_name = path.stem

        # Pre-split: only normalize underscores so the delimiter becomes
        # visible. Title-specific rules must NOT run here.
        normalized_raw = normalize_for_split(raw_name)

        # Split BEFORE title cleanup so rules like "strip trailing year"
        # don't eat into the artist half or into short titles that happen
        # to be numeric (e.g. "Binary Finary - 1998" must keep "1998").
        artist, title_part = split_artist_from_title(normalized_raw)
        if title_part is None:
            title_part = normalized_raw

        # Clean each half with the appropriate ruleset
        cleaned_title, title_changes = cleanup_title(title_part)
        if artist:
            artist, artist_changes = cleanup_artist(artist)
        else:
            artist_changes = []

        # Featured artist extraction
        featured: list[str] = []
        if artist:
            artist, featured = extract_featured_artists(artist)

        # Smart title casing
        if cleaned_title:
            cleaned_title = smart_title_case(cleaned_title)
        if artist:
            artist = smart_title_case(artist)

        result["raw_filename"] = raw_name
        result["cleaned_title"] = cleaned_title
        result["artist"] = artist or ""
        result["title"] = cleaned_title
        result["featured_artists"] = featured
        result["cleanup_changes"] = title_changes + artist_changes
    except Exception as e:
        _log.warning("Cleanup failed for %s: %s", path.name, e)

    # ── Quality validation ────────────────────────────────────────
    try:
        q = check_audio_quality(path)
        result["format"] = q.format
        result["sample_rate"] = q.sample_rate
        result["bitrate"] = q.bitrate
        result["bits_per_sample"] = q.bits_per_sample
        result["is_fake_lossless"] = q.is_fake_lossless
        result["fake_lossless_confidence"] = round(q.fake_lossless_confidence, 2)
        result["clipping_count"] = q.clipping_count
        result["leading_silence_ms"] = q.leading_silence_ms
        result["trailing_silence_ms"] = q.trailing_silence_ms
        result["quality_warnings"] = q.warnings
    except Exception as e:
        _log.warning("Quality failed for %s: %s", path.name, e)

    # ── Vocal detection ───────────────────────────────────────────
    if include_vocals:
        try:
            from .vocals import detect_vocals_fast
            v = detect_vocals_fast(path)
            result["vocal_classification"] = v.classification
            result["vocal_probability"] = round(v.vocal_probability, 2)
            result["vocal_method"] = v.method
        except Exception as e:
            _log.warning("Vocal detection failed for %s: %s", path.name, e)

    # ── Mood classification ───────────────────────────────────────
    if include_mood:
        try:
            from .mood import classify_mood_essentia
            m = classify_mood_essentia(path)
            result["mood"] = m.primary_mood
            result["mood_scores"] = {k: round(v, 2) for k, v in m.mood_scores.items()}
            result["arousal"] = round(m.arousal, 2)
            result["valence"] = round(m.valence, 2)
            result["mood_method"] = m.method
        except Exception as e:
            _log.warning("Mood failed for %s: %s", path.name, e)

    # ── AI Reasoning (optional) ───────────────────────────────────
    if include_reasoning:
        try:
            from .reasoning import analyze_vibe, classify_nuance
            result["vibe"] = analyze_vibe(path, backend=reasoning_backend)[:200]
            result["nuance"] = classify_nuance(path, backend=reasoning_backend)
        except Exception as e:
            _log.warning("Reasoning failed for %s: %s", path.name, e)

    result["analysis_time_sec"] = round(time.time() - start, 1)

    # ── Store to memory ───────────────────────────────────────────
    if memory is not None:
        store_track_analysis(memory, path, result)

    return result
