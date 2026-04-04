"""Tests for title/artist cleanup."""

from dj_agent.cleanup import (
    cleanup_title,
    extract_featured_artists,
    smart_title_case,
    split_artist_from_title,
)


class TestCleanupTitle:
    def test_html_entities(self):
        title, changes = cleanup_title("Beethoven&#39;s Aria")
        assert title == "Beethoven's Aria"
        assert any("HTML" in c for c in changes)

    def test_watermark_removal(self):
        title, _ = cleanup_title("Sandstorm (2024 Extended Mix) djsoundtop.com")
        assert "djsoundtop" not in title

    def test_file_extension_removal(self):
        title, _ = cleanup_title("Track Name.mp3")
        assert title == "Track Name"

    def test_beatport_id_removal(self):
        title, _ = cleanup_title("Track Name-62381251")
        assert title == "Track Name"

    def test_year_preserved_when_meaningful(self):
        # "Levels 2013" — could be a legit title variant, but currently
        # stripped if it looks like a bare trailing year.
        # The fix: don't strip if preceded by a common preposition.
        title, _ = cleanup_title("Class of 2013")
        assert "2013" in title  # "of 2013" should be preserved

    def test_store_id_removed(self):
        title, changes = cleanup_title("Some Track AB12345")
        assert title == "Some Track"

    def test_broad_code_not_removed(self):
        # The old pattern [A-Z]{0,3}\d{3,5} would eat "Channel 4000"
        title, _ = cleanup_title("Channel 4000")
        assert title == "Channel 4000"

    def test_master_suffix_removed(self):
        title, _ = cleanup_title("My Track MASTER")
        assert title == "My Track"

    def test_master_in_word_preserved(self):
        title, _ = cleanup_title("Grandmaster Flash")
        assert title == "Grandmaster Flash"


class TestArtistSplit:
    def test_basic_split(self):
        artist, title = split_artist_from_title("Charlotte de Witte - Overdrive")
        assert artist == "Charlotte de Witte"
        assert title == "Overdrive"

    def test_hyphenated_artist_not_split(self):
        artist, title = split_artist_from_title("Jay-Z - Empire State Of Mind")
        assert artist == "Jay-Z"
        assert title == "Empire State Of Mind"

    def test_no_space_hyphen_not_split(self):
        # "Some-Thing" with no spaces should NOT split
        artist, title = split_artist_from_title("Some-Thing Goes Here")
        assert artist is None
        assert "Some-Thing" in title

    def test_existing_artist_skips(self):
        artist, title = split_artist_from_title(
            "Charlotte de Witte - Overdrive", existing_artist="Charlotte de Witte"
        )
        assert artist is None

    def test_remix_in_artist_rejected(self):
        artist, title = split_artist_from_title("Original Mix - Extended")
        assert artist is None  # "Original Mix" contains "mix"


class TestFeaturedArtists:
    def test_feat_dot(self):
        main, guests = extract_featured_artists("Drake feat. Rihanna")
        assert main == "Drake"
        assert guests == ["Rihanna"]

    def test_ft(self):
        main, guests = extract_featured_artists("Artist ft Vocalist")
        assert main == "Artist"
        assert guests == ["Vocalist"]

    def test_no_feat(self):
        main, guests = extract_featured_artists("Solo Artist")
        assert main == "Solo Artist"
        assert guests == []

    def test_vs(self):
        main, guests = extract_featured_artists("Artist A vs Artist B")
        assert main == "Artist A"
        assert guests == ["Artist B"]

    def test_b2b(self):
        main, guests = extract_featured_artists("DJ Alpha b2b DJ Beta")
        assert main == "DJ Alpha"
        assert guests == ["DJ Beta"]

    def test_ampersand(self):
        main, guests = extract_featured_artists("Bicep & Hammer")
        assert main == "Bicep"
        assert guests == ["Hammer"]


class TestSmartTitleCase:
    def test_basic(self):
        assert smart_title_case("hello world") == "Hello World"

    def test_lowercase_connectors(self):
        result = smart_title_case("the art of mixing")
        assert result == "The Art of Mixing"

    def test_uppercase_acronyms(self):
        result = smart_title_case("dj set in nyc")
        assert result == "DJ Set in NYC"
