"""Stem separation & export — vocals, drums, bass, other.

Wraps Demucs for full stem export, instrumental/acapella generation,
and stem folder management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def separate_stems(
    path: str | Path,
    model: str = "htdemucs",
) -> dict[str, np.ndarray]:
    """Separate a track into stems using Demucs.

    Returns a dict of stem_name → numpy array (samples, channels).
    Standard stems: vocals, drums, bass, other.
    With htdemucs_6s: + guitar, piano.
    """
    import demucs.api  # type: ignore[import-untyped]

    path = Path(path)
    separator = demucs.api.Separator(model=model)
    origin, separated = separator.separate_audio_file(str(path))

    # Convert tensors to numpy (samples, channels)
    stems: dict[str, np.ndarray] = {}
    for name, tensor in separated.items():
        arr = tensor.numpy()
        if arr.ndim == 2:
            arr = arr.T  # (channels, samples) → (samples, channels)
        stems[name] = arr

    return stems


def export_stems(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "htdemucs",
    sr: int = 44100,
) -> list[Path]:
    """Separate and save all stems as WAV files.

    Returns list of paths to the saved stem files.
    Folder structure: ``{track_name}_stems/vocals.wav``, etc.
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
    model: str = "htdemucs",
    sr: int = 44100,
) -> Path:
    """Create an instrumental version (everything minus vocals)."""
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_instrumental")
    output_path = Path(output_path)

    stems = separate_stems(path, model=model)

    # Sum all non-vocal stems
    non_vocal = [audio for name, audio in stems.items() if name != "vocals"]
    if not non_vocal:
        raise ValueError("No non-vocal stems found")

    instrumental = sum(non_vocal)
    # Normalize to avoid clipping
    peak = np.max(np.abs(instrumental))
    if peak > 0.99:
        instrumental = instrumental * (0.99 / peak)

    sf.write(str(output_path), instrumental, sr, subtype="PCM_24")
    return output_path


def create_acapella(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "htdemucs",
    sr: int = 44100,
) -> Path:
    """Create a vocal-only version (acapella)."""
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_acapella")
    output_path = Path(output_path)

    stems = separate_stems(path, model=model)

    vocals = stems.get("vocals")
    if vocals is None:
        raise ValueError("No vocal stem found")

    sf.write(str(output_path), vocals, sr, subtype="PCM_24")
    return output_path
