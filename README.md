# DJ Librarian

An AI agent that enriches your Rekordbox library — energy ratings, cue points, metadata cleanup, duplicate detection — powered by Claude Code.

No app to install. Just talk to the agent and it operates directly on your Rekordbox DB and XML.

## Setup

```bash
pip install pyrekordbox mutagen librosa numpy
pip install pyacoustid fuzzywuzzy python-Levenshtein
```

Requires Rekordbox 6 or 7.

## Commands

| Say this | What it does |
|---|---|
| `magic` / `do your thing` | Full pipeline — energy, cues, tags, cleanup, sync |
| `calculate energy` | Analyse tracks and assign energy ratings (1-10) |
| `calculate cues` | Detect intro, drop, breakdown, outro and set hot cues |
| `calculate tags` | Write energy ratings to Rekordbox My Tag system |
| `find duplicates` | Find dupes via file hashing, fingerprinting, fuzzy metadata |
| `find broken` | Find missing/moved files and offer to relocate them |
| `cleanup` | Clean up titles, artist names, genre casing |
| `health` | Library health report — no modifications |
| `sync` / `write back` | Write results back to Rekordbox (XML + DB) |

All commands can be run on the full library or scoped to a specific playlist (e.g. "calculate energy for Disco").

## How it works

1. You import tracks into Rekordbox (BPM, beat grid, key detection)
2. You ask the agent to enrich — it analyses audio and metadata
3. Results are written back via XML export (cues, titles, artists) and direct DB writes (My Tag energy)

BPM and Key are never touched — Rekordbox is the source of truth.

## What this agent doesn't do

- **BPM detection** — Rekordbox handles this
- **Key detection** — Rekordbox handles this
- **Beat grid analysis** — Rekordbox handles this
- **Waveform generation** — Rekordbox handles this

Import your tracks into Rekordbox first and let it do the audio analysis. This agent picks up where Rekordbox leaves off — the enrichment that Rekordbox can't do on its own.

## Energy calibration

The agent uses `energy_references.json` — user-provided energy ratings for sample tracks across each playlist. This calibrates the audio analysis to match your perception of energy, not just loudness.

## Memory

The agent remembers what it's done in `~/.dj-agent/memory.json` (snapshot: `memory.snapshot.json`). It tracks processed tracks, corrections, calibration offsets, and artist fixes so it doesn't redo work or repeat mistakes.
