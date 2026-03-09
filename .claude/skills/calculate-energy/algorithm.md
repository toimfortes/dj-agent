# Energy Algorithm — Full Implementation

```python
import librosa
import numpy as np

def analyse_track(file_path, rekordbox_bpm):
    """
    Analyse a single track for energy and audio features.
    BPM and key are read from Rekordbox — we never recalculate them.
    """
    results = {"bpm": rekordbox_bpm}

    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
    except Exception as e:
        return {"error": str(e), "path": file_path}

    duration = librosa.get_duration(y=y, sr=sr)
    results["duration"] = round(duration, 1)
    results["energy"] = calculate_energy(y, sr, rekordbox_bpm)

    return results


def calculate_energy(y, sr, bpm, genre=None):
    """
    Composite energy score 1-10.

    Factors:
      - RMS loudness (25%)
      - Spectral centroid (15%)
      - Onset density (20%)
      - BPM normalised (15%)
      - Dynamic range (10%)
      - Bass energy ratio (15%)
    """
    # RMS loudness
    rms = librosa.feature.rms(y=y)[0]
    rms_score = np.clip(np.mean(rms) / 0.12, 0, 1)

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_score = np.clip(np.mean(centroid) / 4500, 0, 1)

    # Onset density
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    duration = librosa.get_duration(y=y, sr=sr)
    onset_density = len(onsets) / max(duration, 1)
    onset_score = np.clip(onset_density / 7.0, 0, 1)

    # BPM (normalised to typical DJ range)
    genre_ranges = {
        "ambient": (60, 100), "downtempo": (80, 115),
        "deep house": (118, 125), "house": (120, 130),
        "tech house": (124, 132), "techno": (128, 150),
        "hard techno": (140, 160), "trance": (128, 145),
        "drum and bass": (160, 180),
    }
    if genre and genre.lower() in genre_ranges:
        lo, hi = genre_ranges[genre.lower()]
    else:
        lo, hi = 100, 150
    bpm_score = np.clip((bpm - lo) / (hi - lo), 0, 1)

    # Dynamic range (inverted)
    rms_db = librosa.amplitude_to_db(rms)
    dynamic_range = np.std(rms_db)
    dyn_score = 1.0 - np.clip(dynamic_range / 20.0, 0, 1)

    # Bass energy ratio (below 150Hz vs full spectrum)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs <= 150
    bass_energy = np.mean(S[bass_mask, :]) if bass_mask.any() else 0
    total_energy = np.mean(S)
    bass_score = np.clip((bass_energy / (total_energy + 1e-8)) / 0.4, 0, 1)

    # Weighted sum
    raw = (0.25 * rms_score +
           0.15 * centroid_score +
           0.20 * onset_score +
           0.15 * bpm_score +
           0.10 * dyn_score +
           0.15 * bass_score)

    return int(np.clip(np.round(raw * 9 + 1), 1, 10))
```

## Energy-to-Rekordbox Mappings

```python
def energy_to_rating(energy):
    """Map energy 1-10 to Rekordbox rating 0-255."""
    if not energy:
        return 0
    return int((energy / 10) * 255)

def energy_to_colour(energy):
    """Map energy to Rekordbox colour ID."""
    if energy <= 2: return 1    # Blue
    if energy <= 4: return 3    # Green
    if energy <= 6: return 5    # Yellow
    if energy <= 8: return 7    # Orange
    return 9                    # Red
```
