---
name: sync
description: Write results back to Rekordbox - XML export for cues/titles/artists, direct DB for My Tags and Comments cleanup.
---

# Sync to Rekordbox

Two sync methods, used together:

## 1. XML Export (cues, titles, artists, genres, colours)

Generates a Rekordbox-compatible XML that can be imported via Bridge.

- Filename: `rekordbox_YYYY-MM-DD_HHMMSS.xml`
- Output directory: `~/Documents/DJ/dj-agent/`
- **Never write energy or tags to Comments or Label in XML** — energy goes to My Tag via DB
- Pass through BPM and key unchanged
- Set colour based on energy rating

### Import Steps

1. Open Rekordbox
2. Preferences > Advanced > Database > rekordbox xml
3. Set "Imported Library" path to the generated XML file
4. Click OK
5. In sidebar, expand "rekordbox xml"
6. Find your playlist > select all > right-click > Import to Collection

### What gets updated on import
- Track title, artist, genre
- Rating and colour (mapped from energy)
- Hot cue points (POSITION_MARK elements)

### What Rekordbox preserves
- Existing cue points, beat grids
- BPM and key (passed through unchanged)
- Playlists and playlist order

## 2. Direct DB Write (My Tags, Comments cleanup)

**Requires Rekordbox to be closed.**

- Energy ratings > My Tag system (`DjmdMyTag` / `DjmdSongMyTag` tables)
- Clear old agent tags from Comments field
- Genre casing fixes (e.g. "BOUNCE" > "Bounce")

```python
from pyrekordbox import Rekordbox6Database
db = Rekordbox6Database()
# ... write My Tags, clear comments, fix genres ...
db.session.commit()
db.close()
```

## Workflow

1. Ask user which method: XML only, DB only, or both
2. If DB: confirm Rekordbox is closed
3. Generate XML and/or write to DB
4. Show summary of what was written
5. Save memory file
