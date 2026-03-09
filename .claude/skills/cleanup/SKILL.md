---
name: cleanup
description: Clean up metadata - titles (HTML entities, watermarks, junk suffixes), artist names, artist/title splitting, genre casing.
---

# Metadata Cleanup

Interactive cleanup of titles, artist names, and genres. Always show proposed changes and ask for confirmation before applying.

## Pipeline

1. **Title cleanup** — see [title-rules.md](title-rules.md) for all rules
2. **Artist/title splitting** — see [artist-split.md](artist-split.md) for logic
3. **Title casing** — smart title case (capitalise words, preserve acronyms like DJ/SOS, keep connectors like feat./vs lowercase)
4. **Genre casing** — always title case (e.g. "Bounce" not "BOUNCE")
5. **Artist name deduplication** — find similar artist names (fuzzywuzzy >= 90) and normalise

## Output Format

Group changes by type:

```
TITLE CLEANUP — 47 changes proposed

HTML entities (14 tracks):
  Bassjackers - Beethoven&#39;s Aria -> Beethoven's Aria Fur Elise (Extended Mix)

Website watermarks (3 tracks):
  Sandstorm (2024 Extended Mix) djsoundtop.com -> Sandstorm (2024 Extended Mix)

ARTIST SPLIT — 45 tracks with artist in title field
  Title: "Charlotte de Witte - Overdrive (Original Mix)"
    -> Artist: Charlotte de Witte
    -> Title: Overdrive (Original Mix)

Apply all changes? [y/n/review each]
```

Always offer to review each change individually.
