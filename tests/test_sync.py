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
