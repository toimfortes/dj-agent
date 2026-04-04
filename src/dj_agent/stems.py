"""Stem separation & export — vocals, drums, bass, other.

V2: Uses audio-separator with Roformer models (SOTA vocal quality) when
available.  Falls back to Demucs for general stem separation.

Model hierarchy for vocals:
1. MelBand-Roformer (best vocal clarity, near-zero artifacts)
2. BS-Roformer (best instrumental extraction)
3. HTDemucs (general 4/6-stem separation, Demucs fallback)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

def _has_audio_separator() -> bool:
    try:
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def _has_demucs() -> bool:
    try:
        import demucs.api  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Roformer separation (preferred for vocals)
# ---------------------------------------------------------------------------

# Known good models — auto-downloads on first use via audio-separator
ROFORMER_VOCAL_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
ROFORMER_INSTRUM_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _separate_roformer(
    path: Path,
    model_filename: str = ROFORMER_VOCAL_MODEL,
) -> dict[str, np.ndarray]:
    """Separate using audio-separator with a Roformer model."""
    from audio_separator.separator import Separator  # type: ignore[import-untyped]

    separator = Separator()
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(str(path))

    # audio-separator returns list of output file paths
    stems: dict[str, np.ndarray] = {}
    for fpath in output_files:
        p = Path(fpath)
        name = p.stem.split("_")[-1].lower()  # e.g., "track_(Vocals)" → "vocals"
        # Normalise stem names
        if "vocal" in name or "voice" in name:
            name = "vocals"
        elif "instrument" in name or "no_vocal" in name:
            name = "other"
        data, sr = sf.read(str(p))
        if data.ndim == 1:
            data = data[:, np.newaxis]
        stems[name] = data

    return stems


# ---------------------------------------------------------------------------
# Demucs separation (fallback / multi-stem)
# ---------------------------------------------------------------------------

def _separate_demucs(
    path: Path,
    model: str = "htdemucs",
) -> dict[str, np.ndarray]:
    """Separate using Demucs (4 or 6 stems)."""
    import demucs.api  # type: ignore[import-untyped]

    separator = demucs.api.Separator(model=model)
    origin, separated = separator.separate_audio_file(str(path))

    stems: dict[str, np.ndarray] = {}
    for name, tensor in separated.items():
        arr = tensor.numpy()
        if arr.ndim == 2:
            arr = arr.T  # (channels, samples) → (samples, channels)
        stems[name] = arr

    return stems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_stems(
    path: str | Path,
    model: str = "auto",
) -> dict[str, np.ndarray]:
    """Separate a track into stems.

    Parameters
    ----------
    model : "auto" (best available), "roformer", "demucs", "htdemucs",
            "htdemucs_ft", "htdemucs_6s", or a specific model filename.

    Returns dict of stem_name → numpy array (samples, channels).
    """
    path = Path(path)

    if model == "auto":
        if _has_audio_separator():
            return _separate_roformer(path)
        elif _has_demucs():
            return _separate_demucs(path)
        else:
            raise ImportError(
                "No stem separation engine found. Install: "
                "pip install audio-separator[cpu]  OR  pip install demucs"
            )

    if model == "roformer":
        return _separate_roformer(path, ROFORMER_VOCAL_MODEL)

    if model.endswith(".ckpt") or model.endswith(".pth"):
        return _separate_roformer(path, model)

    # Demucs models
    return _separate_demucs(path, model)


def export_stems(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
) -> list[Path]:
    """Separate and save all stems as WAV files.

    Returns list of paths to the saved stem files.
    """
    path = Path(path)
    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_stems"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = separate_stems(path, model=model)
    saved: list[Path] = []

    for name, audio in stems.items():
        out_path = output_dir / f"{name}.wav"
        sf.write(str(out_path), audio, sr, subtype="PCM_24")
        saved.append(out_path)

    return saved


def create_instrumental(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
) -> Path:
    """Create an instrumental version (everything minus vocals).

    For best quality, uses BS-Roformer (instrumental specialist) if available.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_instrumental")
    output_path = Path(output_path)

    # BS-Roformer is the instrumental specialist
    if model == "auto" and _has_audio_separator():
        stems = _separate_roformer(path, ROFORMER_INSTRUM_MODEL)
        instrumental = stems.get("other") or stems.get("instrumental")
        if instrumental is not None:
            peak = np.max(np.abs(instrumental))
            if peak > 0.99:
                instrumental = instrumental * (0.99 / peak)
            sf.write(str(output_path), instrumental, sr, subtype="PCM_24")
            return output_path

    # Fallback: sum all non-vocal stems
    stems = separate_stems(path, model=model)
    non_vocal = [audio for name, audio in stems.items() if name != "vocals"]
    if not non_vocal:
        raise ValueError("No non-vocal stems found")

    instrumental = sum(non_vocal)
    peak = np.max(np.abs(instrumental))
    if peak > 0.99:
        instrumental = instrumental * (0.99 / peak)

    sf.write(str(output_path), instrumental, sr, subtype="PCM_24")
    return output_path


def create_acapella(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
) -> Path:
    """Create a vocal-only version (acapella).

    For best quality, uses MelBand-Roformer (vocal specialist) if available.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_acapella")
    output_path = Path(output_path)

    stems = separate_stems(path, model=model)
    vocals = stems.get("vocals") or stems.get("voice")
    if vocals is None:
        raise ValueError("No vocal stem found")

    sf.write(str(output_path), vocals, sr, subtype="PCM_24")
    return output_path
