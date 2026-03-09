---
name: sync
description: Write results back to Rekordbox - XML export for cues/titles/artists, direct DB for My Tags and Comments cleanup.
---

# Sync to Rekordbox

Not everything can be synced the same way. Some data goes via direct DB write, some must be imported via XML. Both methods are used together in a single sync.

## Why two methods?

Cue points in Rekordbox's DB are stored across multiple linked tables with complex relationships, beat grid references, and internal checksums. Writing them directly risks corruption. XML import is Rekordbox's official supported way to bring in cues — it handles all the internal linking safely.

## What goes where

| Data | Method | Why |
|------|--------|-----|
| Hot cue points | **XML import** | Complex DB structure, XML is the safe route |
| Track title & artist | **XML import** | Works reliably via import |
| Genre | **Both** | XML import for new values, DB for casing fixes |
| Rating & colour | **XML import** | Mapped from energy |
| Energy tags (My Tag) | **DB only** | My Tags don't exist in XML format |
| Comments cleanup | **DB only** | Clearing old agent tags |

## Step 1: Direct DB Write (do this FIRST, with Rekordbox closed)

**Requires Rekordbox to be closed.**

Writes:
- Energy ratings → My Tag system (`DjmdMyTag` / `DjmdSongMyTag` tables)
- Clear old agent tags from Comments field
- Genre casing fixes (e.g. "BOUNCE" → "Bounce")

```python
from pyrekordbox import Rekordbox6Database
db = Rekordbox6Database()
# ... write My Tags, clear comments, fix genres ...
db.session.commit()
db.close()
```

## Step 2: XML Export (generate the file)

Generates a Rekordbox-compatible XML for manual import.

- Filename: `rekordbox_YYYY-MM-DD_HHMMSS.xml`
- Output directory: `~/Documents/DJ/dj-agent/`
- **Never write energy or tags to Comments or Label in XML**
- Pass through BPM and key unchanged
- Set colour based on energy rating

## Step 3: XML Import (user does this manually in Rekordbox)

1. Open Rekordbox
2. Preferences → Advanced → Database → rekordbox xml
3. Set "Imported Library" path to the generated XML file
4. Click OK
5. In sidebar, expand "rekordbox xml"
6. Find your playlist → select all → right-click → Import to Collection

### What gets updated on import
- Track title, artist, genre
- Rating and colour (mapped from energy)
- Hot cue points (POSITION_MARK elements)

### What Rekordbox preserves
- Existing cue points and beat grids
- BPM and key (passed through unchanged)
- Playlists and playlist order

## Agent Workflow

1. Do the DB write first (Rekordbox must be closed)
2. Generate the XML export
3. Tell user to open Rekordbox and import the XML
4. Save memory file
