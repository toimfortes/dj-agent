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
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import get_config


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama is running with a multimodal model.

    Just having Ollama running is not enough — the configured model must
    support audio/image input. Text-only models will 500 on audio.
    """
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False

        # Check if any model in the list is known-multimodal
        models = r.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        config = get_config().reasoning
        target = os.environ.get("OLLAMA_MODEL", config.ollama_model)

        # Verify the target model exists locally
        if not any(target in name for name in model_names):
            return False

        # Known multimodal model families that accept audio/images
        _MULTIMODAL_PATTERNS = (
            "llava", "bakllava", "moondream", "qwen2-audio", "qwen2.5-omni",
            "gemma3", "llama3.2-vision", "minicpm-v",
        )
        # qwen3.5 is text-only — do NOT send audio to it
        if any(pat in target.lower() for pat in _MULTIMODAL_PATTERNS):
            return True

        # If unsure, don't risk a 500 — prefer Flamingo or Gemini
        return False

    except Exception:
        return False


_flamingo_verified: bool | None = None  # cache: True/False/None=untested


def _flamingo_available() -> bool:
    """Check if Audio Flamingo is actually operational — runs a smoke test.

    Returns True only if we can instantiate the processor (not just import).
    Caches the result so we only pay the check cost once per process.
    """
    global _flamingo_verified
    if _flamingo_verified is not None:
        return _flamingo_verified

    try:
        import torch
        if not torch.cuda.is_available():
            _flamingo_verified = False
            return False

        # Actually try to load the processor — this verifies model access,
        # HF auth, and that the class exists in this transformers version
        config = get_config().reasoning
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
        _flamingo_verified = True
        return True
    except Exception:
        _flamingo_verified = False
        return False


def _gemini_available() -> bool:
    """Check if any Gemini auth method is available.

    Checks (in order): API key, OAuth creds from Gemini CLI, gcloud ADC.
    """
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    if (Path.home() / ".gemini" / "oauth_creds.json").exists():
        return True
    if (Path.home() / ".config" / "gcloud" / "application_default_credentials.json").exists():
        return True
    return False


def get_backend() -> str:
    """Return the best available backend: 'flamingo', 'ollama', 'gemini', or 'none'."""
    if _flamingo_available():
        # Only suggest flamingo if CUDA is available, otherwise it's too slow
        import torch
        if torch.cuda.is_available():
            return "flamingo"
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
    duration_sec: float | None = None,
    offset_pct: float = 0.25,
    sr: int = 16000,
) -> Path:
    """Extract a representative snippet from a track.

    Samples from TWO points (25% and 50%) and concatenates them to avoid
    the "snippet bias" problem where progressive tracks are judged by
    their intro only.  Downsamples to 16kHz mono WAV.
    """
    import soundfile as sf
    import librosa

    if duration_sec is None:
        duration_sec = get_config().reasoning.snippet_duration_sec

    path = Path(path)
    # Get duration without loading full file
    import subprocess as _sp
    try:
        probe = _sp.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        total = float(probe.stdout.strip()) if probe.returncode == 0 else 300.0
    except Exception:
        total = 300.0

    half_dur = duration_sec / 2.0

    # Sample from two points to capture both intro-section and peak-section
    offsets = [total * 0.25, total * 0.50]
    segments = []
    for off in offsets:
        y, _ = librosa.load(str(path), sr=sr, mono=True,
                             offset=off, duration=half_dur)
        segments.append(y)

    snippet = np.concatenate(segments) if len(segments) > 1 else segments[0]

    tmp = Path(tempfile.NamedTemporaryFile(suffix=".wav", prefix="dj_reason_",
                                            delete=False).name)
    sf.write(str(tmp), snippet, sr)
    return tmp


def _audio_to_base64(path: Path) -> str:
    """Encode audio file as base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Audio Flamingo backend (Local transformers)
# ---------------------------------------------------------------------------

_FLAMINGO_MODEL = None
_FLAMINGO_PROCESSOR = None
_FLAMINGO_LOCK = threading.Lock()


def _load_flamingo() -> tuple[Any, Any]:
    """Lazy-load Audio Flamingo model and processor."""
    global _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR
    if _FLAMINGO_MODEL is not None:
        return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR

    with _FLAMINGO_LOCK:
        if _FLAMINGO_MODEL is not None:
            return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR

        import torch
        from transformers import (
            AutoProcessor,
            AudioFlamingo3ForConditionalGeneration,
            BitsAndBytesConfig,
        )

        config = get_config().reasoning
        model_id = config.model_id

        # Quantization settings
        bnb_config = None
        if config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        _FLAMINGO_PROCESSOR = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        _FLAMINGO_MODEL = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if bnb_config else torch.float32,
        )
        return _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR


def _unload_flamingo() -> None:
    """Unload Flamingo from GPU to free VRAM."""
    global _FLAMINGO_MODEL, _FLAMINGO_PROCESSOR
    _FLAMINGO_MODEL = None
    _FLAMINGO_PROCESSOR = None


# Register with GPU manager
try:
    from .gpu import gpu_manager as _gpu
    _gpu.register_unloader("flamingo", _unload_flamingo)
except Exception:
    pass


def _flamingo_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to local Audio Flamingo."""
    from .gpu import gpu_manager
    gpu_manager._ensure_owner("flamingo")
    model, processor = _load_flamingo()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    
    # Cast float32 inputs to model's dtype (e.g. float16) to avoid mismatch
    import torch
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            inputs[k] = v.to(model.dtype)
        inputs[k] = inputs[k].to(model.device)

    config = get_config().reasoning
    outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)

    # Decode only the generated part
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0].strip()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

_OLLAMA_MODEL = "qwen3.5:27b"  # override with OLLAMA_MODEL env var


def _ollama_query(audio_path: Path, prompt: str) -> str:
    """Send audio + prompt to local Ollama."""
    import requests

    config = get_config().reasoning
    model = os.environ.get("OLLAMA_MODEL", config.ollama_model)
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

# Model tiers for different use cases
GEMINI_MODELS = {
    "lite": "gemini-3.1-flash-lite-preview",   # $0.25/M — batch tagging
    "flash": "gemini-3-flash-preview",          # $0.50/M — default analysis
    "pro": "gemini-3.1-pro-preview",            # $2.00/M — deep reasoning
}


def setup_gemini(api_key: str) -> bool:
    """Store Gemini API key for this session (only if verification succeeds).

    Users can get a free key at https://aistudio.google.com/apikey
    """
    # Verify BEFORE storing — don't poison env with bad keys
    try:
        from google import genai  # type: ignore[import-untyped]
        client = genai.Client(api_key=api_key)
        client.models.list()
        os.environ["GOOGLE_API_KEY"] = api_key
        return True
    except Exception:
        return False


def _gemini_query(
    audio_path: Path,
    prompt: str,
    model_tier: str = "flash",
) -> str:
    """Send audio + prompt to Gemini via the 'gemini' CLI.

    This leverages the CLI's native ability to 'hear' files using the @path syntax.
    It's the leanest way to perform multimodal analysis without extra SDK logic.

    Parameters
    ----------
    model_tier : "lite", "flash", "pro"
    """
    try:
        # 1. Try the CLI first (most direct, handles Files API internally)
        return _gemini_cli_query(audio_path, prompt, model_tier)
    except Exception as e:
        # 2. Fallback to SDK if CLI isn't installed or fails
        try:
            return _gemini_sdk_query(audio_path, prompt, model_tier)
        except Exception as sdk_e:
            raise RuntimeError(f"Gemini CLI failed: {e}\nGemini SDK fallback failed: {sdk_e}")


def _gemini_cli_query(
    audio_path: Path,
    prompt: str,
    model_tier: str,
) -> str:
    """Query via Gemini CLI subprocess using existing OAuth session.

    The CLI uses its own OAuth credentials (~/.gemini/oauth_creds.json)
    and handles multimodal files automatically via the @path syntax.
    """
    import shutil
    import subprocess
    import json

    gemini_exe = shutil.which("gemini")
    if not gemini_exe:
        raise RuntimeError("Gemini CLI ('gemini') not found in PATH. Run 'npm install -g @google/gemini-cli'.")

    model = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["flash"])

    # The magic: @path tells the CLI to treat this as a multimodal input part
    # We use .resolve() to ensure the CLI can find the file from any CWD
    full_prompt = f"{prompt} @{audio_path.resolve()}"

    # Clean env to avoid nested CLI detection issues
    env = os.environ.copy()
    for key in ("CLAUDECODE", "CLAUDE_CODE", "CODEX_SANDBOX"):
        env.pop(key, None)

    cmd = [
        gemini_exe,
        "-p", full_prompt,
        "-m", model,
        "--yolo",  # Auto-approve the file reading tool
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    if result.returncode != 0:
        # If it fails, maybe it's an auth issue
        if "login" in result.stderr.lower() or "auth" in result.stderr.lower() or "unauthenticated" in result.stderr.lower():
            raise RuntimeError("Gemini CLI requires authentication. Run 'gemini auth' first.")
        raise RuntimeError(f"Gemini CLI failed: {result.stderr.strip()}")

    return result.stdout.strip()


def _gemini_sdk_query(
    audio_path: Path,
    prompt: str,
    model_tier: str,
) -> str:
    """Query via google-genai SDK (API key or OAuth/ADC auto-discovery)."""
    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import types
    except ImportError:
        raise ImportError(
            "pip install google-genai\n"
            "Then authenticate via one of:\n"
            "  1. Set GOOGLE_API_KEY (get free key at https://aistudio.google.com/apikey)\n"
            "  2. Run 'gemini auth' for Google account OAuth login\n"
            "  3. Run 'gcloud auth application-default login' for ADC"
        )

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    # Try to load from .env file if not in environment
    if not api_key:
        for env_path in [
            Path.cwd() / ".env",
            Path.home() / ".env",
            Path.home() / ".dj-agent" / ".env",
        ]:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().split("#")[0].strip()
                        break
            if api_key:
                break

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        # Check if we can use ADC
        try:
            client = genai.Client()
        except Exception:
            raise RuntimeError(
                "Gemini API key required for audio analysis fallback.\n"
                "Set GOOGLE_API_KEY in environment or .env file.\n"
                "Get a free key at https://aistudio.google.com/apikey"
            )

    model = GEMINI_MODELS.get(model_tier, GEMINI_MODELS["flash"])

    # Snippets are always small (<5MB) so inline is fine.
    # If someone passes a large file directly, it will be rejected by the API
    # with a clear error — better than silently failing.
    mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mpeg"
    audio_bytes = audio_path.read_bytes()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                ]
            )
        ],
    )
    return response.text


# ---------------------------------------------------------------------------
# Unified query
# ---------------------------------------------------------------------------

def _query(
    audio_path: Path,
    prompt: str,
    backend: str = "auto",
    model_tier: str = "flash",
) -> str:
    """Send audio + prompt to the best available backend.

    Parameters
    ----------
    backend : "auto", "flamingo", "ollama", "gemini", "gemini-lite", "gemini-pro"
    model_tier : "lite", "flash", "pro" (Gemini only)
    """
    if backend == "auto":
        config_backend = get_config().reasoning.backend
        backend = config_backend if config_backend != "auto" else get_backend()

    # Parse backend shortcuts
    if backend.startswith("gemini-"):
        model_tier = backend.split("-", 1)[1]  # "gemini-pro" → "pro"
        backend = "gemini"

    if backend == "flamingo":
        return _flamingo_query(audio_path, prompt)
    elif backend == "ollama":
        return _ollama_query(audio_path, prompt)
    elif backend == "gemini":
        return _gemini_query(audio_path, prompt, model_tier=model_tier)
    else:
        raise RuntimeError(
            "No reasoning backend available. "
            "Install transformers for Flamingo, start Ollama, or set GOOGLE_API_KEY."
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
