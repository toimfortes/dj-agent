
import os
import sys
import json
from pathlib import Path

# Add src to sys.path to find dj_agent
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dj_agent import energy, mood, reasoning, beatgrid, types

def analyze_vibe(file_path):
    print(f"Analysing {file_path}...")
    
    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} does not exist.")
        return

    # 1. BPM Detection
    print("Detecting BPM...")
    try:
        import numpy as np
        import librosa
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            detected_bpm = float(np.mean(tempo))
        else:
            detected_bpm = float(tempo)
        print(f"Detected BPM: {detected_bpm:.1f}")
    except Exception as e:
        print(f"BPM detection failed: {e}")
        detected_bpm = 124.0

    # 2. Energy Analysis
    print("Analysing energy...")
    try:
        energy_res = energy.analyse_track(path, rekordbox_bpm=detected_bpm)
    except Exception as e:
        import traceback
        print(f"Energy analysis failed: {e}")
        traceback.print_exc()
        energy_res = None

    # 3. Mood Analysis (Essentia/Librosa fallback)
    print("Analysing mood (traditional)...")
    try:
        mood_res = mood.classify_mood_essentia(path)
    except Exception as e:
        print(f"Mood analysis failed: {e}")
        mood_res = None

    # 4. Commercial vs Underground
    print("Checking commercial factor...")
    try:
        commercial_factor = mood.classify_commercial(path)
    except Exception as e:
        print(f"Commercial check failed: {e}")
        commercial_factor = 0.5

    # 5. AI Reasoning (Vibe & Nuance)
    print("Analysing vibe with AI (auto)...")
    vibe_desc = "Unknown"
    nuance_tags = {}
    try:
        vibe_desc = reasoning.analyze_vibe(path, backend="auto")
        nuance_tags = reasoning.classify_nuance(path, backend="auto")
    except Exception as e:
        import traceback
        print(f"AI Reasoning failed: {e}")
        traceback.print_exc()

    # 6. Hardness
    hardness = 5
    if energy_res and mood_res:
        hardness = mood.calculate_hardness(
            energy_res.raw_score, 
            detected_bpm, 
            mood_res.primary_mood
        )

    # Summary
    print("\n" + "="*60)
    print(f"  VIBE ANALYSIS: {path.name}")
    print("="*60)
    
    if energy_res:
        color = energy.energy_to_colour(energy_res.calibrated_score)
        print(f"Energy:    {energy_res.calibrated_score}/10 ({color.upper()})")
        print(f"LUFS:      {energy_res.integrated_lufs:.1f} dB")
        print(f"Intensity: Bass={energy_res.bass_ratio:.2f}, Onsets={energy_res.onset_density:.1f}")
    
    if mood_res:
        print(f"Mood:      {mood_res.primary_mood.upper()} (via {mood_res.method})")
    
    print(f"Hardness:  {hardness}/10")
    print(f"Underground Factor: {1.0 - commercial_factor:.2f}")
    
    print("\nAI VIBE DESCRIPTION:")
    print(f"  {vibe_desc}")
    
    print("\nNUANCE TAGS:")
    for k, v in nuance_tags.items():
        print(f"  - {k}: {v}")
    
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vibe_analysis.py <file_path>")
    else:
        analyze_vibe(sys.argv[1])
