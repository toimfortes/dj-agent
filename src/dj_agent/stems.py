"""Stem separation & export — vocals, drums, bass, other.

Model selection (2026):
- **BS-RoFormer (12.97 SDR)** — top single-file vocal model on MVSep/MUSDB18
  leaderboards. 2-stem (vocals + instrumental) in one pass, which beats
  summing 4 Demucs stems (leakage compounds on sum).
- **Mel-Band RoFormer (11.44 SDR)** — cleaner on exposed pop vocals; a bit
  less robust on buried/electronic vocals. Kept as an alternative.
- **HTDemucs** — general 4/6-stem separator. Good for drums/bass but its
  vocal SDR (8.4) is now significantly behind RoFormer.

Quality presets:
- ``fast``     : HTDemucs 4-stem (~10 s/track on RTX 3090)
- ``balanced`` : BS-RoFormer 2-stem (~25 s/track) — DEFAULT
- ``best``     : BS-RoFormer + Mel-Band ensemble (~50 s/track, ~+0.3 dB)

audio-separator auto-downloads model weights on first use.
"""

from __future__ import annotations
import logging

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

# Known good models — auto-downloaded by audio-separator on first use.
#
# BS-RoFormer (ep 317, SDR 12.9755) is the top single-file vocal/instrumental
# model as of 2026 — a 2-stem model whose single inference yields BOTH vocals
# and instrumental. That's why it's our default for both create_acapella and
# create_instrumental.
ROFORMER_BS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_MEL_BAND_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"

# Backwards-compat aliases for callers that imported the old names.
ROFORMER_VOCAL_MODEL = ROFORMER_BS_MODEL
ROFORMER_INSTRUM_MODEL = ROFORMER_BS_MODEL


_roformer_cache: dict[str, Any] = {}
_roformer_lock = __import__("threading").Lock()


def _unload_roformer() -> None:
    """Unload all Roformer models from memory."""
    _roformer_cache.clear()


def _unload_demucs() -> None:
    """Unload all Demucs models from memory."""
    _demucs_cache.clear()


# Register with GPU manager
try:
    from .gpu import gpu_manager as _gpu_stems
    _gpu_stems.register_unloader("roformer", _unload_roformer)
    _gpu_stems.register_unloader("demucs", _unload_demucs)
except Exception:
    pass


def _get_roformer_separator(model_filename: str):
    """Cached Roformer separator (model loaded once per model name)."""
    if model_filename in _roformer_cache:
        return _roformer_cache[model_filename]
    with _roformer_lock:
        if model_filename in _roformer_cache:
            return _roformer_cache[model_filename]
        from audio_separator.separator import Separator  # type: ignore[import-untyped]
        sep = Separator()
        sep.load_model(model_filename=model_filename)
        _roformer_cache[model_filename] = sep
        return sep


def _separate_roformer(
    path: Path,
    model_filename: str = ROFORMER_BS_MODEL,
) -> dict[str, np.ndarray]:
    """Separate using audio-separator with a Roformer model."""
    try:
        from .gpu import gpu_manager
        gpu_manager._ensure_owner("roformer")
    except Exception:
        pass
    separator = _get_roformer_separator(model_filename)
    output_files = separator.separate(str(path))

    # audio-separator returns list of output file paths — read then clean up
    stems: dict[str, np.ndarray] = {}
    try:
        for fpath in output_files:
            p = Path(fpath)
            name = _parse_stem_name(p.stem)
            data, sr = sf.read(str(p))
            if data.ndim == 1:
                data = data[:, np.newaxis]
            stems[name] = data
    finally:
        # Clean up temp files written by audio-separator
        for fpath in output_files:
            try:
                Path(fpath).unlink(missing_ok=True)
            except OSError:
                pass

    return stems


def _parse_stem_name(filename_stem: str) -> str:
    """Extract a normalised stem name from an audio-separator output filename.

    audio-separator writes files like ``track_(Vocals)`` or ``track_(Instrumental)``
    so we look for the parenthesised suffix first. Falls back to substring
    matching if the convention changes.
    """
    import re
    match = re.search(r"\(([^)]+)\)\s*$", filename_stem)
    label = (match.group(1) if match else filename_stem).lower()
    if "vocal" in label or "voice" in label:
        return "vocals"
    if "instrument" in label or "no_vocal" in label or "no vocal" in label:
        return "instrumental"
    if "drum" in label:
        return "drums"
    if "bass" in label:
        return "bass"
    return label


# ---------------------------------------------------------------------------
# Demucs separation (fallback / multi-stem)
# ---------------------------------------------------------------------------

_demucs_cache: dict[str, Any] = {}
_demucs_lock = __import__("threading").Lock()


def _get_demucs_separator(model: str):
    """Cached Demucs separator (loaded once per model name)."""
    if model in _demucs_cache:
        return _demucs_cache[model]
    with _demucs_lock:
        if model in _demucs_cache:
            return _demucs_cache[model]
        import demucs.api  # type: ignore[import-untyped]
        sep = demucs.api.Separator(model=model)
        _demucs_cache[model] = sep
        return sep


def _separate_demucs(
    path: Path,
    model: str = "htdemucs",
) -> dict[str, np.ndarray]:
    """Separate using Demucs (4 or 6 stems)."""
    separator = _get_demucs_separator(model)
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
    quality: str = "balanced",
) -> dict[str, np.ndarray]:
    """Separate a track into stems.

    Parameters
    ----------
    model : str
        - ``"auto"``  — pick best engine by *quality* preset (default)
        - ``"roformer"``, ``"bs_roformer"``, ``"mel_band"`` — specific Roformer variant
        - ``"demucs"``, ``"htdemucs"``, ``"htdemucs_ft"``, ``"htdemucs_6s"`` — Demucs variant
        - a full ``.ckpt`` / ``.pth`` filename — passed through to audio-separator
    quality : str
        Used only when ``model="auto"``:
        - ``"fast"``     : HTDemucs 4-stem (~10 s/track on RTX 3090)
        - ``"balanced"`` : BS-RoFormer 2-stem (default, ~25 s/track)
        - ``"best"``     : BS-RoFormer + Mel-Band ensemble (~50 s/track, +0.3 dB)

    Returns dict of stem_name → numpy array (samples, channels).
    """
    path = Path(path)

    if model == "auto":
        if quality == "fast" and _has_demucs():
            try:
                return _separate_demucs(path, "htdemucs")
            except Exception:
                pass
        if quality == "best" and _has_audio_separator():
            try:
                return _separate_ensemble(path)
            except Exception:
                pass
        # balanced (default)
        if _has_audio_separator():
            try:
                return _separate_roformer(path, ROFORMER_BS_MODEL)
            except Exception:
                pass  # fall through
        if _has_demucs():
            try:
                return _separate_demucs(path)
            except Exception:
                pass
        raise ImportError(
            "No stem separation engine found or all engines failed. Install: "
            "pip install audio-separator[cpu]  OR  pip install demucs"
        )

    # Explicit model selection
    if model in ("roformer", "bs_roformer"):
        return _separate_roformer(path, ROFORMER_BS_MODEL)
    if model == "mel_band":
        return _separate_roformer(path, ROFORMER_MEL_BAND_MODEL)

    if model.endswith(".ckpt") or model.endswith(".pth"):
        return _separate_roformer(path, model)

    # Demucs models (htdemucs, htdemucs_ft, htdemucs_6s)
    demucs_model = "htdemucs" if model == "demucs" else model
    return _separate_demucs(path, demucs_model)


def _separate_ensemble(path: Path) -> dict[str, np.ndarray]:
    """Run BS-RoFormer + Mel-Band RoFormer and average the outputs.

    Yields ~0.3 dB higher SDR than either model alone at 2× the runtime cost.
    Use only when the extra quality is worth the time (single-track mashup
    workflows, not 1000+ batch jobs).
    """
    bs = _separate_roformer(path, ROFORMER_BS_MODEL)
    mb = _separate_roformer(path, ROFORMER_MEL_BAND_MODEL)

    # Align length (models may produce slightly different tail lengths)
    result: dict[str, np.ndarray] = {}
    for key in bs.keys():
        if key not in mb:
            result[key] = bs[key]
            continue
        a, b = bs[key], mb[key]
        n = min(len(a), len(b))
        result[key] = (a[:n] + b[:n]) / 2.0
    return result


def export_stems(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> list[Path]:
    """Separate and save all stems as WAV files.

    Returns list of paths to the saved stem files.
    """
    path = Path(path)
    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_stems"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = separate_stems(path, model=model, quality=quality)
    saved: list[Path] = []

    for name, audio in stems.items():
        out_path = output_dir / f"{name}.wav"
        sf.write(str(out_path), audio, sr, subtype="PCM_24")
        saved.append(out_path)

    return saved


def _split_vocal_instrumental(
    path: Path, model: str, quality: str
) -> dict[str, np.ndarray]:
    """Run one stem-separation pass and return both vocals and instrumental.

    BS-RoFormer is a 2-stem model — a single inference yields both halves,
    so splitting the acapella and the instrumental from the same call
    halves the work vs running the model twice.
    """
    stems = separate_stems(path, model=model, quality=quality)
    vocals = stems.get("vocals") or stems.get("voice")
    inst = (
        stems.get("instrumental")
        or stems.get("other")
        or stems.get("no_vocals")
    )
    # If we got a 4-stem output (Demucs), sum the non-vocal stems ourselves
    if inst is None:
        non_vocal = [a for k, a in stems.items() if k != "vocals"]
        if not non_vocal:
            raise ValueError("No non-vocal stems found")
        # Align lengths before summing
        n = min(len(a) for a in non_vocal)
        inst = sum(a[:n] for a in non_vocal)
    if vocals is None:
        raise ValueError("No vocal stem found")
    return {"vocals": vocals, "instrumental": inst}


def _write_normalized(audio: np.ndarray, out_path: Path, sr: int) -> Path:
    """Write audio to WAV with peak normalisation to avoid clipping."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(str(out_path), audio, sr, subtype="PCM_24")
    return out_path


def create_instrumental(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> Path:
    """Create an instrumental version (everything minus vocals).

    Default ``balanced`` quality uses BS-RoFormer 2-stem which yields a
    cleaner instrumental than summing 4 Demucs stems (where per-stem
    leakage compounds on the sum).
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_instrumental")
    return _write_normalized(
        _split_vocal_instrumental(path, model, quality)["instrumental"],
        Path(output_path),
        sr,
    )


def create_acapella(
    path: str | Path,
    output_path: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> Path:
    """Create a vocal-only version (acapella).

    Default ``balanced`` uses BS-RoFormer — top-scoring vocal model on the
    2026 MVSep multisong leaderboard.
    """
    path = Path(path)
    if output_path is None:
        output_path = path.with_stem(path.stem + "_acapella")
    return _write_normalized(
        _split_vocal_instrumental(path, model, quality)["vocals"],
        Path(output_path),
        sr,
    )


def create_acapella_and_instrumental(
    path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "auto",
    sr: int = 44100,
    quality: str = "balanced",
) -> tuple[Path, Path]:
    """Produce both acapella and instrumental from a single separation pass.

    Prefer this over calling :func:`create_acapella` and
    :func:`create_instrumental` back-to-back — both share one BS-RoFormer
    inference so this is ~2× faster.
    """
    path = Path(path)
    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = _split_vocal_instrumental(path, model, quality)
    aca = _write_normalized(
        split["vocals"], output_dir / f"{path.stem}_acapella.wav", sr,
    )
    inst = _write_normalized(
        split["instrumental"], output_dir / f"{path.stem}_instrumental.wav", sr,
    )
    return aca, inst
