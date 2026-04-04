"""AI-powered musical reasoning — vibe analysis, transition advice, nuance tagging.

Uses a multimodal LLM (local Ollama or cloud Gemini) to provide high-level
musical understanding that numerical features alone cannot capture.

Backend priority:
1. Ollama (local, free, private) — if running at localhost:11434
2. Gemini Flash (cloud, cheap, 1M context) — if GOOGLE_API_KEY is set
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _gemini_available() -> bool:
    """Check if Gemini API key is configured."""
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def get_backend() -> str:
    """Return the best available backend: 'ollama', 'gemini', or 'none'."""
    if _ollama_available():
        return "ollama"
    if _gemini_available():
        return "gemini"
    return "none"


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def _extract_snippet(
    path: str | Path,
    duration_sec: float = 30.0,
    offset_pct: float = 0.25,
    sr: int = 16000,
) -> Path:
    """Extract a representative snippet from a track.

    Takes a 30-second clip starting at 25% of the track (avoids intros).
    Downsamples to 16kHz mono WAV (standard for audio LLMs).
    """
    import soundfile as sf
    import librosa

    path = Path(path)
    y, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    total = len(y) / sr

    start = int(total * offset_pct * sr)
    end = start + int(duration_sec * sr)
    snippet = y[start:min(end, len(y))]

    tmp = Path(tempfile.mktemp(suffix=".wav", prefix="dj_reason_"))
    sf.write(str(tmp), snippet, sr)
    return tmp


def _audio_to_base64(path: Path) -> str:
    """Encode audio file as base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen3.5:27b"  # override with OLLAMA_MODEL env var


def _ollama_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to local Ollama."""
    import requests

    model = os.environ.get("OLLAMA_MODEL", _OLLAMA_MODEL)
    audio_b64 = _audio_to_base64(audio_path)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [audio_b64],  # Ollama multimodal uses "images" for all media
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _gemini_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to Gemini API."""
    try:
        from google import genai  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    audio_bytes = audio_path.read_bytes()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
    )
    return response.text


# ---------------------------------------------------------------------------
# Unified query
# ---------------------------------------------------------------------------

def _query(audio_path: Path, prompt: str, backend: str = "auto") -> str:
    """Send audio + prompt to the best available backend."""
    if backend == "auto":
        backend = get_backend()

    if backend == "ollama":
        return _ollama_query(audio_path, prompt)
    elif backend == "gemini":
        return _gemini_query(audio_path, prompt)
    else:
        raise RuntimeError(
            "No reasoning backend available. "
            "Start Ollama (ollama serve) or set GOOGLE_API_KEY for Gemini."
        )


# ---------------------------------------------------------------------------
# Public API — Musical reasoning functions
# ---------------------------------------------------------------------------

DJ_PERSONA = (
    "You are a world-class DJ and music analyst with 20 years of experience "
    "in electronic music. Be concise and specific. "
)


def analyze_vibe(path: str | Path, backend: str = "auto") -> str:
    """Describe the 'vibe' of a track — texture, setting, energy.

    Returns a short text description like "Dark, hypnotic warehouse techno.
    Peak-time energy with rolling bass and chopped vocal stabs."
    """
    snippet = _extract_snippet(path)
    try:
        return _query(snippet, DJ_PERSONA + (
            "Analyze this audio clip. Describe the 'vibe' in 2-3 sentences. "
            "Focus on: texture, aggression level, and the ideal dancefloor "
            "setting (e.g., 'warm-up', 'peak-time', 'after-hours')."
        ), backend)
    finally:
        snippet.unlink(missing_ok=True)


def get_energy_arc(path: str | Path, backend: str = "auto") -> str:
    """Describe how energy changes through the clip.

    Returns labels like "Rising", "Driving", "Floating", "Dropping".
    """
    snippet = _extract_snippet(path)
    try:
        return _query(snippet, DJ_PERSONA + (
            "Listen to this clip. How does the energy change? "
            "Classify as one of: 'Driving' (constant high), 'Rising' (building), "
            "'Floating' (ambient/sustained), 'Dropping' (energy decreasing). "
            "Explain in one sentence why."
        ), backend)
    finally:
        snippet.unlink(missing_ok=True)


def suggest_transition(
    path_a: str | Path,
    path_b: str | Path,
    backend: str = "auto",
) -> str:
    """Suggest the best transition technique between two tracks."""
    snip_a = _extract_snippet(path_a, offset_pct=0.75)  # end of track A
    snip_b = _extract_snippet(path_b, offset_pct=0.0)   # start of track B
    try:
        # Combine into one file for models that take single audio
        import soundfile as sf
        a_data, sr = sf.read(str(snip_a))
        b_data, _ = sf.read(str(snip_b))
        combined = np.concatenate([a_data, np.zeros(sr), b_data])  # 1s gap
        combined_path = Path(tempfile.mktemp(suffix=".wav", prefix="dj_mix_"))
        sf.write(str(combined_path), combined, sr)

        result = _query(combined_path, DJ_PERSONA + (
            "This clip contains the end of Track A, a brief silence, then the "
            "start of Track B. Suggest the best DJ transition technique "
            "(e.g., 'smooth EQ blend over 16 bars', 'hard cut on the 1', "
            "'filter sweep into drop'). Explain why in 2 sentences."
        ), backend)

        combined_path.unlink(missing_ok=True)
        return result
    finally:
        snip_a.unlink(missing_ok=True)
        snip_b.unlink(missing_ok=True)


def classify_nuance(path: str | Path, backend: str = "auto") -> dict[str, str]:
    """Extract nuanced musical tags that audio features can't capture.

    Returns tags like bassline_type, vocal_style, rhythm_feel, mood_detail.
    """
    snippet = _extract_snippet(path)
    try:
        raw = _query(snippet, DJ_PERSONA + (
            "Analyze this audio. Return ONLY a JSON object with these keys:\n"
            '- "bassline": one of "rolling", "off-beat", "syncopated", "minimal", "heavy"\n'
            '- "vocals": one of "none", "chopped", "soulful", "atmospheric", "MC", "rap"\n'
            '- "rhythm": one of "four-on-the-floor", "broken-beat", "shuffle", "polyrhythmic"\n'
            '- "mood": one of "dark", "euphoric", "melancholic", "aggressive", "groovy", "hypnotic"\n'
            '- "setting": one of "warm-up", "peak-time", "after-hours", "festival", "underground"\n'
            "Respond with ONLY the JSON, no explanation."
        ), backend)

        # Parse JSON from response (handle markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}
    finally:
        snippet.unlink(missing_ok=True)
