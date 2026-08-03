"""Unit tests for app/amplifier_report.py pure formatting helpers.

Previously a blind zone (audit-tests). Covers the i18n-independent helpers;
the full HTML assembly is exercised via the amplifier-tab smoke tests.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.amplifier_report import _fmt_pout_w, dist_error_html


class TestFmtPoutW:
    def test_converts_mw_to_w(self):
        assert float(_fmt_pout_w(5000.0)) == 5.0
        assert float(_fmt_pout_w(0.0)) == 0.0
        assert float(_fmt_pout_w(12000.0)) == 12.0


class TestDistErrorHtml:
    def test_none_and_empty_give_generic(self):
        assert dist_error_html(None) == dist_error_html("")

    def test_unknown_code_falls_back_to_generic(self):
        # An unrecognised code must produce the same generic message as no
        # code — never the raw 'amp.dist_err_<code>' key.
        generic = dist_error_html(None)
        assert dist_error_html("totally_unknown_code_xyz") == generic
        assert "dist_err_totally" not in dist_error_html("totally_unknown_code_xyz")

    def test_bias_outside_with_params_names_the_range(self):
        """The diagnostic must tell the user WHAT range is valid, not just
        that theirs is not."""
        html = dist_error_html(
            "bias_outside_data",
            {"lo": -20.0, "hi": -2.0, "bias": -25.0})
        assert "-20.0" in html and "-2.0" in html and "-25.0" in html
        assert "amp.dist_err" not in html      # no raw keys

    def test_bias_at_edge_with_params_names_the_range(self):
        html = dist_error_html(
            "bias_at_data_edge",
            {"lo": -20.0, "hi": -2.0, "bias": -19.5})
        assert "-19.5" in html and "-20.0" in html

    def test_params_without_range_key_fall_back(self):
        """A code with params but no `_range` locale key keeps the plain
        message — never leaks raw placeholders."""
        plain = dist_error_html("few_intersections")
        assert dist_error_html("few_intersections",
                               {"lo": 1.0, "hi": 2.0}) == plain

    def test_no_params_keeps_legacy_message(self):
        html = dist_error_html("bias_outside_data")
        assert "outside the measured" in html
        assert "%{" not in html

    def test_no_signal_with_params_shows_flat_window(self):
        html = dist_error_html(
            "no_signal", {"imin": 5.02, "imax": 5.04, "b1": "0.003"})
        assert "5.0" in html and "0.003" in html   # b1 string not mangled
        assert "%{" not in html


class TestWindowB1Probe:
    @staticmethod
    def _isects(ias):
        return [{"ua": 100.0 + 10 * i, "ug1": -12.0 + 2.0 * i, "ia": ia}
                for i, ia in enumerate(ias)]

    def test_flat_window_diagnosed_as_no_signal(self):
        from lm19.amplifier import diagnose_distortion, window_b1_probe
        flat = self._isects([5.0] * 6)
        probe = window_b1_probe(flat, ug1_bias=-7.0)
        assert probe is not None and probe["b1"] <= 0.01
        assert diagnose_distortion(flat, ug1_bias=-7.0) == "no_signal"

    def test_sloped_window_stays_healthy(self):
        from lm19.amplifier import diagnose_distortion, window_b1_probe
        sloped = self._isects([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        probe = window_b1_probe(sloped, ug1_bias=-7.0)
        assert probe is not None and probe["b1"] > 0.01
        # a healthy window must NOT be misdiagnosed as no_signal
        assert diagnose_distortion(sloped, ug1_bias=-7.0) != "no_signal"

    def test_m_shaped_window_uses_half_points(self):
        """Window edges carry EQUAL current but the half-points differ:
        the fundamental is NOT zero — a probe that only looks at the
        edges would falsely report no_signal."""
        from lm19.amplifier import window_b1_probe
        # window is -12…-4 (center snaps to -8): edges ia 5.0 == 5.0,
        # half-points ia 3.0 vs 6.0 → true b1 = 1.0, edges-only b1 = 0
        m_shaped = self._isects([5.0, 3.0, 4.0, 6.0, 5.0, 9.0])
        probe = window_b1_probe(m_shaped, ug1_bias=-7.0)
        assert probe is not None and probe["b1"] > 0.01
