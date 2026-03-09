---
name: calculate-energy
description: Calculate energy ratings (1-10) for tracks using audio analysis. BPM and key always come from Rekordbox.
---

# Calculate Energy

Rate tracks 1-10 based on audio features. BPM and key are always read from Rekordbox — never recalculated.

## Algorithm

See [algorithm.md](algorithm.md) for the full implementation.

Composite energy score using weighted factors:
- RMS loudness (25%) — how loud overall
- Spectral centroid (15%) — how bright/aggressive
- Onset density (20%) — how busy/driving
- BPM normalised (15%) — faster = more energy (genre-aware ranges)
- Dynamic range (10%) — less dynamic = more compressed = more energy
- Bass energy ratio (15%) — more bass = more physical energy

## Calibration

Apply calibration offsets from `memory.json` → `energy_calibration`:
- `global_offset` — mean of all user corrections
- `genre_offsets` — per-genre adjustment (only if 3+ corrections exist for that genre)

```python
def calibrated_energy(raw_energy, genre, memory):
    cal = memory.get("energy_calibration", {})
    offset = cal.get("global_offset", 0)
    genre_offset = cal.get("genre_offsets", {}).get(genre.lower(), 0)
    adjusted = raw_energy + offset + genre_offset
    return int(np.clip(np.round(adjusted), 1, 10))
```

## Energy-to-Colour Mapping

| Energy | Colour | Use |
|--------|--------|-----|
| 1-2 | Blue | Ambient / warm-up |
| 3-4 | Green | Low energy / chill |
| 5-6 | Yellow | Mid energy |
| 7-8 | Orange | Peak time |
| 9-10 | Red | Maximum intensity |

## Output Format

```
Track                                    BPM    Energy  Colour
-----------------------------------------------------------------
Bicep - Glue                            130.0   6       Yellow
Charlotte de Witte - Overdrive          140.2   9       Red
```

## Workflow

1. Load library from Rekordbox (BPM, key from Rekordbox).
2. Load memory, check for already-processed tracks and manual overrides.
3. For each unprocessed track: `librosa.load()` → `calculate_energy()` → apply calibration.
4. Show results table.
5. Save to memory, offer to sync.
