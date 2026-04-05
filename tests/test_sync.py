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


def test_generate_cue_xml_skips_trackless(tmp_path: Path):
    from dj_agent.config import RekordboxConfig

    config = RekordboxConfig(xml_output_dir=str(tmp_path))
    tracks = [{"path": "/a.mp3", "title": "No Cues", "artist": "A", "db_content_id": "1", "cues": []}]
    xml_path = generate_cue_xml(tracks, config)
    content = xml_path.read_text()
    assert "POSITION_MARK" not in content  # no cues = no POSITION_MARK
