"""Unit tests for lm19/import_helpers.py — pure, Qt-free import helpers.

Previously untested (blind zone, audit-tests).
"""
from lm19.import_helpers import (
    build_csv_description,
    build_utd_description,
    csv_comment_lines,
    first_non_comment_csv_header,
    import_topology_payload,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


class TestImportTopologyPayload:
    def test_triode(self):
        p = import_topology_payload("triode")
        assert p["topology"] == TOPOLOGY_TRIODE
        assert p["scan"]["ug2_mode"] == TOPOLOGY_TRIODE
        assert p["scan"]["ug2_track_ua"] is False

    def test_triode_connected_is_pentode_topology_tracking_ua(self):
        p = import_topology_payload("triode_connected")
        assert p["topology"] == TOPOLOGY_PENTODE
        assert p["scan"]["ug2_mode"] == TOPOLOGY_TRIODE_CONNECTED
        assert p["scan"]["ug2_track_ua"] is True

    def test_pentode(self):
        p = import_topology_payload("pentode")
        assert p["topology"] == TOPOLOGY_PENTODE
        assert p["scan"]["ug2_track_ua"] is False

    def test_unknown_mode_defaults_to_pentode(self):
        p = import_topology_payload("garbage")
        assert p["topology"] == TOPOLOGY_PENTODE
        assert p["scan"]["ug2_mode"] == TOPOLOGY_PENTODE


class TestCsvHelpers:
    def test_first_header_skips_comments_and_blanks(self):
        text = "# comment\n\n#another\nUa,Ug1,Ia\n1,2,3"
        assert first_non_comment_csv_header(text) == "Ua,Ug1,Ia"

    def test_first_header_empty_when_all_comments(self):
        assert first_non_comment_csv_header("# only\n# comments") == ""

    def test_comment_lines_extracted_and_stripped(self):
        text = "# Tube: EL84\n#  Date: 2024\nUa,Ia\n1,2"
        assert csv_comment_lines(text) == ["Tube: EL84", "Date: 2024"]

    def test_comment_lines_stop_at_first_data_line(self):
        text = "# a\nUa,Ia\n# b"  # comment after data must not be collected
        assert csv_comment_lines(text) == ["a"]

    def test_comment_lines_respects_max(self):
        text = "\n".join(f"# c{i}" for i in range(10))
        assert len(csv_comment_lines(text, max_lines=3)) == 3


class TestDescriptions:
    def test_utd_description_includes_fields(self):
        d = build_utd_description(
            "/x/foo.utd",
            {"format": "pentode", "has_is": True,
             "x_values": [1, 2], "step_values": [1]},
            250.0,
        )
        assert "foo.utd" in d
        assert "pentode" in d
        assert "Has Is/Ig2: yes" in d
        assert "Guessed Vs from filename: 250 V" in d

    def test_utd_description_omits_vs_when_zero(self):
        d = build_utd_description("/x/foo.utd", {}, 0.0)
        assert "Guessed Vs" not in d
        assert "Has Is/Ig2: no" in d  # missing has_is → falsy → "no"

    def test_csv_description_header_and_comments(self):
        d = build_csv_description("/x/data.csv", "# Tube: 6L6\nUa,Ia\n1,2")
        assert "data.csv" in d
        assert "Header: Ua,Ia" in d
        assert "Tube: 6L6" in d
