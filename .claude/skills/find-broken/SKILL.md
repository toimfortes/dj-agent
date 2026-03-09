---
name: find-broken
description: Find missing/broken track files and offer to relocate them by searching specified directories.
---

# Find Broken Tracks

Check every track path in the library — does the file actually exist?

## Detection

```python
from pathlib import Path
from urllib.parse import unquote, urlparse

def find_broken_tracks(df):
    broken, ok = [], 0
    for idx, row in df.iterrows():
        path = row["path"]
        if path.startswith("file://"):
            path = unquote(urlparse(path).path)
        if not Path(path).exists():
            broken.append({"index": idx, "path": path,
                           "artist": row["artist"], "title": row["title"]})
        else:
            ok += 1
    return broken, ok
```

## Relocation

When files are missing, offer to search for them:
1. Ask user for directories to search
2. Index all audio files (`.mp3`, `.flac`, `.wav`, `.aiff`, `.m4a`, `.aac`, `.ogg`)
3. Match by: exact filename > fuzzy filename (fuzzywuzzy >= 85) > artist-title in metadata

## Output

```
BROKEN TRACKS: 8 found

Auto-relocated (6):
  > Bicep - Glue.flac -> /Volumes/DJ-SSD/Techno/Bicep - Glue.flac  [high confidence]
  > Peggy Gou - Starry Night.mp3 -> /Music/House/Peggy Gou - Starry Night.mp3  [fuzzy 92%]

Unresolved (2):
  x old_mix_final_v2.mp3 — cannot find anywhere
  x track_from_soundcloud.wav — cannot find anywhere

Apply relocations? [y/n]
```

Always ask before applying path changes.
