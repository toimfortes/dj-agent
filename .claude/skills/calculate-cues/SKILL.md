---
name: calculate-cues
description: Detect structural sections (intro, drop, breakdown, outro) and set hot cue points. Never overrides existing cues.
---

# Calculate Cues

Detect structural sections and set hot cues (coloured lettered pads A-H). **Never override cue points that already exist.**

## How It Works

Uses librosa's segmentation to find energy transitions:
1. Compute mel spectrogram and RMS energy
2. Run agglomerative clustering to find segment boundaries
3. Classify transitions by energy level (threshold: 0.6)
4. Snap all positions to nearest beat using Rekordbox BPM

See [algorithm.md](algorithm.md) for the full implementation.

## Cue Types

| Section | Colour | Trigger |
|---------|--------|---------|
| Intro | Green | Start of track (always at 0:00) |
| Drop | Red | Low-to-high energy transition |
| Breakdown | Blue | High-to-low energy transition |
| Outro | Yellow | Last low-energy segment after 70% of track |

## Rules

- Always use **hot cues** (`Type="0"`, `Num="0"` through `"7"` for pads A-H)
- **NEVER remove or overwrite existing POSITION_MARK elements**
- If a track already has cue points, skip it entirely unless user explicitly asks
- Maximum 8 hot cues per track (Rekordbox limit)
- Snap all positions to nearest beat using Rekordbox BPM

## Writing to Rekordbox XML

Cue points are `POSITION_MARK` elements inside each `TRACK`:

```xml
<POSITION_MARK Name="Drop" Type="0" Start="32.456" Num="1" Red="230" Green="30" Blue="30"/>
```

## Workflow

1. Load library from Rekordbox.
2. For each track in scope, check for existing `POSITION_MARK` elements — skip if any exist.
3. Run `detect_cue_points()` on tracks without cues.
4. Show proposed cue points and ask for confirmation before writing.
5. Write hot cues to the XML export.
