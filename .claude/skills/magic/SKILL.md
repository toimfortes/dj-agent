---
name: magic
description: Full DJ library workflow - load library, analyse energy, detect cues, auto-tag, find duplicates, find broken tracks, health report, sync back to Rekordbox.
---

# Magic — Full Workflow

When the user says "magic", "do your thing", "run everything", or "full workflow", execute this pipeline in order:

## Pipeline

### STEP 1: Load library from Rekordbox
- Read all tracks from XML (or DB), including BPM and key from Rekordbox
- Load memory file (`~/.dj-agent/memory.json`)
- Show library summary (track count, BPM range, genre distribution, missing fields)

### STEP 2: Identify what needs work
- Compare library against memory to find new/unprocessed tracks
- Show: "X new tracks, Y flagged for re-analysis, Z already processed"
- If user specifies a playlist, scope to that playlist only

### STEP 3: Analyse new/flagged tracks
- Energy rating (see `/calculate-energy` for algorithm)
- Cue point detection — only for tracks with NO existing cues (see `/calculate-cues`)
- Show progress bar
- Apply energy calibration offsets from memory

### STEP 4: Auto-tag
- Write energy to My Tag via direct DB (see `/calculate-tags`)

### STEP 5: Find duplicates
- Only check new tracks against existing library (not full N*N scan)
- Show duplicate groups if any found, ask user what to do
- See `/find-duplicates` for details

### STEP 6: Find broken tracks
- Quick path existence check on full library
- Report any missing files
- See `/find-broken` for details

### STEP 7: Health report
- Summary of everything found (see `/health`)

### STEP 8: Sync
- **CRITICAL: Before DB writes, always ask the user to confirm Rekordbox is closed.** Do NOT proceed until the user confirms. Rekordbox overwrites the DB while running, so changes will be lost.
- **DB direct write:** energy My Tags, title/artist cleanup, genre casing, comments cleanup
- **XML export (only if new cues were generated):** hot cue points via `POSITION_MARK` elements — tell user to import via Rekordbox XML import
- If no cues were generated, no XML needed — user can just open Rekordbox
- See `/sync` for details

Each sub-command (e.g. `/calculate-energy`) runs only its specific step — it still loads the library and memory file first, but skips everything else.
