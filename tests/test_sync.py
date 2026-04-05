"""Tests for sync — XML generation and URL encoding."""

from pathlib import Path

from dj_agent.sync import generate_cue_xml, rb_url_encode


def test_rb_url_encode_spaces():
    assert rb_url_encode("/music/My Track.flac") == "file://localhost/music/My%20Track.flac"


def test_rb_url_encode_ampersand():
    assert rb_url_encode("/music/A & B.mp3") == "file://localhost/music/A%20%26%20B.mp3"


def test_rb_url_encode_no_special():
    assert rb_url_encode("/music/track.flac") == "file://localhost/music/track.flac"


def test_rb_url_encode_parentheses_preserved():
    result = rb_url_encode("/music/Track (Remix).flac")
    assert "(Remix)" in result  # parens should NOT be encoded


def test_rb_url_encode_windows_drive_letter():
    # Windows paths begin with "C:/..." (no leading slash). The encoder
    # must prepend a slash after "localhost" and keep the colon literal
    # to match Rekordbox's own XML export format:
    #   file://localhost/C:/Users/.../File.mp3
    result = rb_url_encode("C:/Users/Antonio Fortes/Music/File.mp3")
    assert result == "file://localhost/C:/Users/Antonio%20Fortes/Music/File.mp3"


def test_rb_url_encode_windows_unchanged_colon():
    # The colon in the drive letter must not be percent-encoded.
    result = rb_url_encode("D:/Library/Track.flac")
    assert "D:/" in result
    assert "%3A" not in result


def test_is_builtin_rekordbox_path():
    from dj_agent.sync import is_builtin_rekordbox_path

    # Built-in sampler loops — filename-based sync must skip these
    assert is_builtin_rekordbox_path(
        "C:/Users/Foo/Music/rekordbox/Sampler/GROOVE CIRCUIT/PRESET/House1.wav"
    )
    assert is_builtin_rekordbox_path(
        "C:\\Users\\Foo\\Music\\rekordbox\\Sampler\\OSC_SAMPLER\\NOISE.wav"
    )
    # Demo tracks — also built-in
    assert is_builtin_rekordbox_path(
        "C:/Users/Foo/Music/PioneerDJ/Demo Tracks/Demo Track 1.mp3"
    )
    # Real user tracks — must NOT be flagged
    assert not is_builtin_rekordbox_path("C:/Users/Foo/Music/PioneerDJ/my_track.mp3")
    assert not is_builtin_rekordbox_path("/home/foo/Music/Track.flac")
    assert not is_builtin_rekordbox_path("")


def test_generate_cue_xml(tmp_path: Path):
    from dj_agent.config import RekordboxConfig

    config = RekordboxConfig(
        check_process=False,
        backup_before_write=False,
        xml_output_dir=str(tmp_path),
    )
    tracks = [
        {
            "path": "file://localhost/music/test.flac",
            "title": "Test Track",
            "artist": "Test Artist",
            "db_content_id": "12345",
            "cues": [
                {"position_ms": 0, "name": "Intro", "colour": "green"},
                {"position_ms": 32000, "name": "Drop", "colour": "red"},
            ],
        }
    ]
    xml_path = generate_cue_xml(tracks, config)
    assert xml_path.exists()
    content = xml_path.read_text()
    assert "DJ_PLAYLISTS" in content
    assert "POSITION_MARK" in content
    assert "Intro" in content
    assert "Drop" in content


def test_generate_cue_xml_writes_hot_and_memory_cues(tmp_path: Path):
    """Each cue must be written twice: Num=0..7 (hot cue slot) AND Num=-1
    (memory cue). Rekordbox silently drops the rest if only the slot
    entry is present — only Hot Cue A survives.
    """
    import xml.etree.ElementTree as ET
    from dj_agent.config import RekordboxConfig

    config = RekordboxConfig(xml_output_dir=str(tmp_path))
    tracks = [
        {
            "path": "/music/test.flac",
            "title": "Test",
            "artist": "A",
            "db_content_id": "1",
            "cues": [
                {"position_ms": 0, "name": "Intro", "colour": "green"},
                {"position_ms": 32000, "name": "Drop", "colour": "red"},
                {"position_ms": 64000, "name": "Outro", "colour": "blue"},
            ],
        }
    ]
    xml_path = generate_cue_xml(tracks, config)
    tree = ET.parse(str(xml_path))
    track = tree.getroot().find("COLLECTION/TRACK")
    assert track is not None
    marks = track.findall("POSITION_MARK")

    # 3 cues × 2 entries each = 6 POSITION_MARK elements
    assert len(marks) == 6

    # First 3 are hot-cue slots with Num 0, 1, 2
    slots = [m for m in marks if m.get("Num") != "-1"]
    mems = [m for m in marks if m.get("Num") == "-1"]
    assert len(slots) == 3
    assert len(mems) == 3
    assert [m.get("Num") for m in slots] == ["0", "1", "2"]

    # Slot and memory entries must share Name + Start
    for s, m in zip(slots, mems):
        assert s.get("Name") == m.get("Name")
        assert s.get("Start") == m.get("Start")


def test_generate_cue_xml_explicit_output_path(tmp_path: Path):
    """An explicit output_path must override the default timestamped filename."""
    tracks = [
        {
            "path": "/music/test.flac",
            "title": "T",
            "artist": "A",
            "db_content_id": "1",
            "cues": [{"position_ms": 0, "name": "Cue", "colour": "green"}],
        }
    ]
    target = tmp_path / "rekordbox.xml"
    result = generate_cue_xml(tracks, output_path=target)
    assert result == target
    assert target.exists()
    # The parent dir must be created if missing
    nested = tmp_path / "nested" / "deep" / "out.xml"
    generate_cue_xml(tracks, output_path=nested)
    assert nested.exists()


def test_generate_cue_xml_skips_trackless(tmp_path: Path):
    from dj_agent.config import RekordboxConfig

    config = RekordboxConfig(xml_output_dir=str(tmp_path))
    tracks = [{"path": "/a.mp3", "title": "No Cues", "artist": "A", "db_content_id": "1", "cues": []}]
    xml_path = generate_cue_xml(tracks, config)
    content = xml_path.read_text()
    assert "POSITION_MARK" not in content  # no cues = no POSITION_MARK
