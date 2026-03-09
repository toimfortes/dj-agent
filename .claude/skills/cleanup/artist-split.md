# Artist/Title Splitting

For tracks with empty artist fields where the artist is baked into the title.

## Pass 1: Delimiter-based split

Look for "Artist - Track Name" or "Artist -- Track Name" pattern:

```python
import re

def split_artist_from_title(title, existing_artist=""):
    if existing_artist.strip():
        return None, title

    match = re.match(r'^(.+?)\s*[-\u2013]\s+(.+)$', title)
    if match:
        potential_artist = match.group(1).strip()
        potential_title = match.group(2).strip()
        if (len(potential_artist) < 80 and potential_title and
            not re.search(r'(remix|edit|mix|version|bootleg|rework)', potential_artist, re.IGNORECASE)):
            return potential_artist, potential_title

    return None, title
```

## Pass 2: Known-artist matching

For titles without a delimiter (e.g. "Gwen Stefani What You Waiting For Edit"):

```python
def match_artist_from_library(title, known_artists):
    title_lower = title.lower()
    for artist in sorted(known_artists, key=len, reverse=True):
        if not artist or len(artist) < 3:
            continue
        if title_lower.startswith(artist.lower() + " "):
            remaining = title[len(artist):].strip()
            if remaining:
                return artist, remaining
    return None, title
```

Build `known_artists` from all tracks that DO have an artist field. If "Funk Tribu" appears as an artist on other tracks, it can be recognised in "Funk Tribu All Of It Healy Edit Master".

## No Match

If no match is found, show the track to the user and ask them to manually specify. Save manual mappings to `memory.json` under `artist_corrections` for future sessions.
