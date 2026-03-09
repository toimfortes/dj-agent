---
name: health
description: Library health report - file existence, metadata completeness, quality stats, genre/BPM distribution. No modifications.
---

# Library Health Report

Read-only summary of the library state. No modifications.

## What to Report

```python
def library_health(df):
    total = len(df)

    # File existence
    missing = sum(1 for _, r in df.iterrows() if not Path(r["path"]).exists())

    # Metadata completeness
    no_genre = (df["genre"].isna() | (df["genre"] == "")).sum()
    no_key = (df["key"].isna() | (df["key"] == "")).sum()
    no_artist = (df["artist"].isna() | (df["artist"] == "")).sum()

    # Quality
    low_quality = (df["bitrate"] < 320).sum() if "bitrate" in df.columns else 0

    # BPM sanity (flag for user review in Rekordbox)
    weird_bpm = ((df["bpm"] < 60) | (df["bpm"] > 200)).sum()

    # Genre distribution (top 10, with bar chart)
    # BPM range and average
    # Possible duplicates (same artist+title)
```

## Output Format

```
==================================================
  LIBRARY HEALTH REPORT - 3,247 tracks
==================================================

Files:  3,239 found, 8 missing
Missing genre: 112
Missing key: 47 - analyse these in Rekordbox
Missing artist: 23

Below 320kbps: 31
Suspicious BPM (<60 or >200): 2 - check in Rekordbox

Top genres:
   Tech House               ████████████ 842
   Techno                   ██████████ 731
   House                    ███████ 523

BPM range: 98-174, Average: 132

Possible duplicates (same artist+title): 14 tracks
```
