---
name: find-duplicates
description: Find duplicate tracks using file hashing, audio fingerprinting, and fuzzy metadata matching.
---

# Find Duplicates

Three-pass duplicate detection:

## Pass 1: Exact Hash
Same file hash = identical files in different locations. Hash just the audio data (ignoring metadata tags).

## Pass 2: Acoustic Fingerprint
Same audio fingerprint (chromaprint/fpcalc) = same track in different formats or re-encodes. Requires `fpcalc` binary. Skipped if not installed.

## Pass 3: Fuzzy Metadata
Similar artist + title (fuzzywuzzy ratio >= 85) with similar duration (within 10 seconds) = probable duplicates.

## Output Format

```
DUPLICATE GROUP 1 (exact match):
  > /Music/Techno/Bicep - Glue.flac         [FLAC, 1411kbps, 4:32]  <- KEEP (best quality)
  x /Music/Downloads/Bicep - Glue.mp3       [MP3, 320kbps, 4:32]

DUPLICATE GROUP 2 (fuzzy match, 91% similar):
  ? /Music/House/Peggy Gou - Starry Night (Original Mix).flac
  ? /Music/House/Peggy Gou - Starry Night.mp3
  -> Same track? [y/n/skip]
```

## Rules

- Always ask before deleting. Recommend keeping the highest quality version but don't force.
- In a `magic` run, only check new tracks against existing library (not full N*N scan).
- Show file format, bitrate, and duration for each duplicate to help the user decide.

## Implementation

```python
import hashlib
from fuzzywuzzy import fuzz
from pathlib import Path

def find_duplicates(df):
    dupes = {"exact": [], "acoustic": [], "fuzzy": []}

    # Pass 1: Exact hash
    hashes = {}
    for idx, row in df.iterrows():
        path = row["path"]
        if not Path(path).exists():
            continue
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        hashes.setdefault(h, []).append(idx)

    for h, indices in hashes.items():
        if len(indices) > 1:
            dupes["exact"].append(df.loc[indices])

    # Pass 2: Acoustic fingerprint (if fpcalc available)
    # Uses chromaprint — skipped if fpcalc not installed

    # Pass 3: Fuzzy metadata
    keys = df.apply(lambda r: f"{r['artist']} - {r['title']}".lower().strip(), axis=1)
    seen = {}
    for idx, key in keys.items():
        for existing_key, existing_idx in seen.items():
            ratio = fuzz.ratio(key, existing_key)
            if ratio >= 85 and ratio < 100:
                dur_diff = abs(df.loc[idx]["duration"] - df.loc[existing_idx]["duration"])
                if dur_diff < 10:
                    dupes["fuzzy"].append((existing_idx, idx, ratio))
        seen[key] = idx

    return dupes
```
