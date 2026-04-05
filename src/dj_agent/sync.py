"""Sync results back to Rekordbox — DB writes + XML export.

Fixes:
- Artist is set via ArtistID (not the read-only ArtistName proxy)
- XML uses manual ElementTree (never pyrekordbox.RekordboxXml)
"""

from __future__ import annotations

import random
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RekordboxConfig


# Path substrings that identify Rekordbox built-in tracks (sampler loops,
# demo tracks) which must never be matched by filename-based sync, because
# they use generic filenames like "House1.wav", "Techno1.wav", "Demo Track
# 1.mp3" that collide with real user library entries. Anything whose
# FolderPath contains one of these substrings should be skipped by the
# caller's matching loop.
BUILTIN_PATH_MARKERS: tuple[str, ...] = (
    "rekordbox/Sampler",
    "PioneerDJ/Demo",
)


def is_builtin_rekordbox_path(folder_path: str) -> bool:
    """Return True if the given FolderPath is a Rekordbox built-in track.

    Filename-based sync must skip these — they use generic filenames that
    collide with real library tracks (e.g. House1.wav, Techno1.wav).
    """
    normalized = (folder_path or "").replace("\\", "/")
    return any(marker in normalized for marker in BUILTIN_PATH_MARKERS)


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def set_artist(db: Any, content: Any, artist_name: str) -> None:
    """Set an artist on a track via ArtistID (not the read-only proxy).

    Creates the DjmdArtist row if it doesn't exist yet.
    """
    from pyrekordbox.db6 import tables  # type: ignore[import-untyped]

    existing = (
        db.session.query(tables.DjmdArtist)
        .filter_by(Name=artist_name)
        .first()
    )
    if not existing:
        now = datetime.now(timezone.utc)
        existing = tables.DjmdArtist(
            ID=str(random.randint(100_000_000, 4_294_967_295)),
            Name=artist_name,
            UUID=str(uuid.uuid4()),
            rb_data_status=0,
            rb_local_data_status=0,
            rb_local_deleted=0,
            rb_local_synced=0,
            rb_local_usn=100,
            created_at=now,
            updated_at=now,
        )
        db.session.add(existing)
        db.session.flush()

    content.ArtistID = existing.ID
    content.updated_at = datetime.now(timezone.utc)


def update_title(content: Any, title: str) -> None:
    """Safely set the track title (direct column write)."""
    content.Title = title
    content.updated_at = datetime.now(timezone.utc)


def update_genre(content: Any, genre: str) -> None:
    """Set genre (always title-cased)."""
    content.GenreName = genre.title()
    content.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# XML export (hot cues only)
# ---------------------------------------------------------------------------

COLOUR_MAP = {
    "green": {"Red": "40", "Green": "226", "Blue": "20"},
    "red": {"Red": "230", "Green": "50", "Blue": "50"},
    "blue": {"Red": "50", "Green": "100", "Blue": "230"},
    "yellow": {"Red": "230", "Green": "200", "Blue": "30"},
}


def rb_url_encode(path: str) -> str:
    """Encode a file path the way Rekordbox does in its XML.

    Rekordbox encodes spaces, ampersands, and other URI-unsafe characters
    but leaves parentheses, commas, hash signs, and drive-letter colons
    literal.

    Output format: ``file://localhost/<abs path>`` — the leading slash
    after ``localhost`` is always required. On Unix the path already
    starts with ``/``; on Windows the path starts with ``C:/...`` and
    we must prepend a ``/`` so the final URL is
    ``file://localhost/C:/...`` (with the colon kept literal).
    """
    from urllib.parse import quote
    if not path.startswith("/"):
        path = "/" + path
    # safe list includes ":" so Windows drive letters stay literal.
    return "file://localhost" + quote(path, safe="/:(),#;!~")


def generate_cue_xml(
    tracks: list[dict[str, Any]],
    config: RekordboxConfig | None = None,
) -> Path:
    """Generate a Rekordbox-compatible XML with hot cues.

    Parameters
    ----------
    tracks : list[dict]
        Each dict needs: ``path``, ``title``, ``artist``, ``db_content_id``,
        ``cues`` (list of :class:`CuePoint`-like dicts).

    Returns
    -------
    Path to the generated XML file.
    """
    if config is None:
        from .config import get_config
        config = get_config().rekordbox

    output_dir = Path(config.xml_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    xml_path = output_dir / f"rekordbox_{stamp}.xml"

    # Filter to tracks that have cues (so Entries count matches actual TRACK elements)
    tracks_with_cues = [t for t in tracks if t.get("cues")]

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="rekordbox", Version="7.2.8", Company="AlphaTheta")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks_with_cues)))

    for t in tracks_with_cues:

        raw_path = t.get("path", "")
        # Strip file://localhost prefix to get absolute path for encoding
        if raw_path.startswith("file://localhost"):
            raw_path = raw_path[len("file://localhost"):]

        track_el = ET.SubElement(
            collection,
            "TRACK",
            TrackID=str(t.get("db_content_id", "0")),
            Name=t.get("title", ""),
            Artist=t.get("artist", ""),
            Location=rb_url_encode(raw_path),
        )

        for idx, cue in enumerate(t.get("cues", [])[:8]):
            rgb = COLOUR_MAP.get(cue.get("colour", "green"), COLOUR_MAP["green"])
            pos_sec = cue.get("position_ms", 0) / 1000.0
            ET.SubElement(
                track_el,
                "POSITION_MARK",
                Name=cue.get("name", "Cue"),
                Type="0",
                Start=f"{pos_sec:.3f}",
                Num=str(idx),
                **rgb,
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)

    return xml_path
