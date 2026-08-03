"""Pins for the standalone emission verdict and the reference-based
emission score.

Two distinct things are pinned here because they can fail independently:

* ``emission_score`` — the *relative* metric that feeds the weighted
  index. It must normalize against the active reference's own
  ``emission_ratio`` and fall back to the config nominal only when the
  reference carries none.
* ``emission_verdict`` — the *absolute* cathode-reserve scale. It must
  never be divided by a reference, otherwise a reference taken from a
  worn tube would declare the next worn tube healthy.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from i18n_setup import available_locales, t  # noqa: E402
from lm19.config import LampConfig, LampRange  # noqa: E402
from lm19.constants import TOPOLOGY_PENTODE, TOPOLOGY_TRIODE  # noqa: E402
from lm19.health import (
    EMISSION_VERDICT_EXHAUSTED,
    EMISSION_VERDICT_NA,
    EMISSION_VERDICT_NORMAL,
    EMISSION_VERDICT_WEAKENED,
    EMISSION_VERDICTS,
    _compute_scores,
    _emission_verdict,
    _extract_refs,
    run_health_test,
)

LOCALES_DIR = Path(__file__).parent.parent / "locales"

# Deliberately NOT the shipped 0.90/0.70/0.50 triple: a pin fed the
# shipped values cannot tell "reads config" from "hardcodes the shipped
# number".
CFG_NOMINAL = 0.86
CFG_GOOD_MIN = 0.66
CFG_WEAK_MIN = 0.44


def _cfg(**over):
    base = dict(
        health_emission_ratio_nominal=CFG_NOMINAL,
        health_emission_ratio_good_min=CFG_GOOD_MIN,
        health_emission_ratio_weak_min=CFG_WEAK_MIN,
        health_weight_ia=0.35,
        health_weight_s=0.40,
        health_weight_rh=0.10,
        health_weight_screen=0.0,
        health_weight_emission=0.15,
        health_renormalize_weights_if_metric_missing=True,
        health_verdict_strong_min=90.0,
        health_verdict_good_min=75.0,
        health_verdict_weak_min=55.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _lamp(topology=TOPOLOGY_PENTODE):
    return LampConfig(
        tube_type="EL84", socket="B", anodes=1, warmup_s=120,
        topology=topology, uh=6.3, ih=0.76, ug1=-7.3, ua=250.0,
        ia=48.0, ug2=250.0 if topology != TOPOLOGY_TRIODE else 0.0,
        ig2=5.5 if topology != TOPOLOGY_TRIODE else 0.0,
        s=11.0, r=40.0, k=19.0,
        ranges={"ua": LampRange(0, 250, 10), "ug1": LampRange(-20, 0, 1),
                "ug2": LampRange(0, 250, 10)},
        limits={},
    )


def _scores(refs, *, ia80, ia100, emission_enabled=True, cfg=None):
    return _compute_scores(
        ia_op=48.0, s=11.0, r=40.0, k=19.0,
        point_op={"ia": 48.0, "ig2": 5.5, "uh": 6.3, "ih": 0.0},
        lamp=_lamp(), cfg=cfg or _cfg(),
        refs=refs, emission_enabled=emission_enabled,
        ia80=ia80, ia100=ia100,
    )


def _base_refs(**over):
    refs = {"ia": 48.0, "s": 11.0, "r": 40.0, "k": 19.0,
            "rh": None, "screen_ratio": 5.5 / 48.0}
    refs.update(over)
    return refs


def _translating_locale(code, key):
    """A locale that actually renders *code* differently.

    Hardcoding a locale would be wrong (a locale may be added or fall
    back to en); running only under the session locale makes the pin
    degenerate whenever that locale renders the code verbatim.
    """
    from i18n_setup import translator_for
    for loc in available_locales():
        if translator_for(loc)(key) != code:
            return loc
    return None


class TestEmissionVerdictScale:
    """Absolute scale, both ends of both boundaries."""

    def test_above_good_min_is_normal(self):
        assert _emission_verdict(CFG_GOOD_MIN + 0.05, _cfg()) == EMISSION_VERDICT_NORMAL

    def test_exactly_good_min_is_normal(self):
        # `>=` vs `>` at the boundary is a one-character mutation.
        assert _emission_verdict(CFG_GOOD_MIN, _cfg()) == EMISSION_VERDICT_NORMAL

    def test_just_below_good_min_is_weakened(self):
        assert _emission_verdict(CFG_GOOD_MIN - 0.01, _cfg()) == EMISSION_VERDICT_WEAKENED

    def test_exactly_weak_min_is_weakened(self):
        assert _emission_verdict(CFG_WEAK_MIN, _cfg()) == EMISSION_VERDICT_WEAKENED

    def test_just_below_weak_min_is_exhausted(self):
        assert _emission_verdict(CFG_WEAK_MIN - 0.01, _cfg()) == EMISSION_VERDICT_EXHAUSTED

    def test_none_is_na(self):
        assert _emission_verdict(None, _cfg()) == EMISSION_VERDICT_NA

    def test_thresholds_come_from_cfg_not_constants(self):
        # A ratio that is "weakened" under the shipped 0.70/0.50 config
        # must be "normal" under a config whose good_min is lower.
        loose = _cfg(health_emission_ratio_good_min=0.30,
                     health_emission_ratio_weak_min=0.20)
        assert _emission_verdict(0.55, _cfg()) == EMISSION_VERDICT_WEAKENED
        assert _emission_verdict(0.55, loose) == EMISSION_VERDICT_NORMAL

    def test_every_code_is_registered(self):
        produced = {
            _emission_verdict(None, _cfg()),
            _emission_verdict(0.95, _cfg()),
            _emission_verdict(CFG_GOOD_MIN - 0.01, _cfg()),
            _emission_verdict(0.0, _cfg()),
        }
        assert produced == set(EMISSION_VERDICTS)


class TestEmissionScoreReference:
    """emission_score must honour the reference's own ratio."""

    def test_uses_reference_ratio_when_present(self):
        # ref 0.70 and cfg nominal 0.86 give clearly different scores —
        # ignoring refs (the pre-fix behaviour) fails this.
        scores = _scores(_base_refs(emission_ratio=0.70), ia80=7.0, ia100=10.0)
        assert scores["emission_ratio"] == pytest.approx(0.70)
        assert scores["emission_score"] == pytest.approx(100.0)
        assert scores["emission_ratio_ref"] == pytest.approx(0.70)

    def test_falls_back_to_cfg_nominal_without_reference(self):
        scores = _scores(_base_refs(), ia80=7.0, ia100=10.0)
        assert scores["emission_ratio_ref"] == pytest.approx(CFG_NOMINAL)
        assert scores["emission_score"] == pytest.approx(100.0 * 0.70 / CFG_NOMINAL)

    def test_real_extract_refs_path_datasheet_reference(self):
        # The datasheet reference carries no emission_ratio, so the whole
        # _extract_refs -> _compute_scores chain must land on the nominal.
        refs = _extract_refs({"reference": {"ia": 48.0, "s": 11.0}}, _lamp())
        scores = _scores(refs, ia80=8.0, ia100=10.0)
        assert scores["emission_ratio_ref"] == pytest.approx(CFG_NOMINAL)

    def test_real_extract_refs_path_personal_baseline(self):
        refs = _extract_refs(
            {"reference": {"ia": 48.0, "s": 11.0, "emission_ratio": 0.80}}, _lamp())
        scores = _scores(refs, ia80=8.0, ia100=10.0)
        assert scores["emission_ratio_ref"] == pytest.approx(0.80)
        assert scores["emission_score"] == pytest.approx(100.0)

    @pytest.mark.parametrize("bad", [0.0, -0.5, None, "0.8"])
    def test_unusable_reference_ratio_falls_back(self, bad):
        # Zero would divide by ~EPS and produce a nonsense score; a
        # string would raise. Both must degrade to the config nominal.
        scores = _scores(_base_refs(emission_ratio=bad), ia80=8.0, ia100=10.0)
        assert scores["emission_ratio_ref"] == pytest.approx(CFG_NOMINAL)
        assert scores["emission_score"] == pytest.approx(100.0 * 0.80 / CFG_NOMINAL)

    def test_verdict_is_not_scaled_by_reference(self):
        # A worn reference (0.55) must not turn a worn tube into "normal".
        scores = _scores(_base_refs(emission_ratio=0.55), ia80=5.5, ia100=10.0)
        assert scores["emission_score"] == pytest.approx(100.0)
        assert scores["emission_verdict"] == EMISSION_VERDICT_WEAKENED

    def test_disabled_emission_yields_na_verdict(self):
        scores = _scores(_base_refs(), ia80=9.0, ia100=10.0, emission_enabled=False)
        assert scores["emission_verdict"] == EMISSION_VERDICT_NA

    def test_disabled_emission_excluded_from_index(self):
        on = _scores(_base_refs(), ia80=3.0, ia100=10.0, emission_enabled=True)
        off = _scores(_base_refs(), ia80=3.0, ia100=10.0, emission_enabled=False)
        assert on["index"] < off["index"]

    def test_missing_ia100_yields_na(self):
        scores = _scores(_base_refs(), ia80=None, ia100=None)
        assert scores["emission_ratio"] is None
        assert scores["emission_score"] is None
        assert scores["emission_verdict"] == EMISSION_VERDICT_NA


class TestExtractRefsEmissionRatio:
    def test_passthrough_from_reference_block(self):
        refs = _extract_refs({"reference": {"ia": 40.0, "emission_ratio": 0.77}}, _lamp())
        assert refs["emission_ratio"] == pytest.approx(0.77)

    def test_passthrough_from_flat_dict(self):
        refs = _extract_refs({"ia": 40.0, "emission_ratio": 0.77}, _lamp())
        assert refs["emission_ratio"] == pytest.approx(0.77)

    def test_absent_is_none(self):
        assert _extract_refs({"reference": {"ia": 40.0}}, _lamp())["emission_ratio"] is None

    def test_non_numeric_is_none(self):
        refs = _extract_refs({"reference": {"emission_ratio": "high"}}, _lamp())
        assert refs["emission_ratio"] is None


class TestResultContract:
    """The persisted ``metrics`` block is the contract seen by history,
    CSV export and the reference builder — a scores-level pin does not
    prove the orchestrator copies the keys across."""

    def test_scores_expose_both_keys(self):
        scores = _scores(_base_refs(emission_ratio=0.70), ia80=7.0, ia100=10.0)
        assert {"emission_ratio_ref", "emission_verdict"} <= set(scores)

    def test_measurement_metrics_carry_both_keys(self):
        from lm19.calibration import CalibrationData
        from test_health_logic import _FakeClient, _cfg as _full_cfg, _lamp as _full_lamp

        m = run_health_test(
            client=_FakeClient(), lamp=_full_lamp("triode"), cfg=_full_cfg(),
            calibration=CalibrationData(), lamp_id="L-em", name="contract",
            reference_mode="datasheet", emission_enabled=False, warmup_s=60,
        )
        metrics = m["health"]["metrics"]
        assert "emission_ratio_ref" in metrics
        assert metrics["emission_verdict"] == EMISSION_VERDICT_NA


class TestEmissionVerdictI18n:
    """Registry <-> locales bijection, read from the files directly.

    ``translator_for`` silently falls back to en, which would hide a key
    missing from another locale.
    """

    @pytest.fixture(params=[f"{loc}.json" for loc in available_locales()])
    def locale_data(self, request):
        with open(LOCALES_DIR / request.param, encoding="utf-8") as f:
            return json.load(f), request.param

    def test_every_code_has_a_key(self):
        from app.health_tab import EMISSION_VERDICT_KEYS
        assert set(EMISSION_VERDICT_KEYS) == set(EMISSION_VERDICTS)

    def test_keys_present_in_every_locale(self, locale_data):
        from app.health_tab import EMISSION_VERDICT_KEYS
        data, name = locale_data
        health = data.get("health", {})
        for code, key in sorted(EMISSION_VERDICT_KEYS.items()):
            short = key.split(".", 1)[1]
            assert short in health, f"Missing health.{short} ({code}) in {name}"
            assert health[short].strip(), f"Empty health.{short} in {name}"

    def test_result_line_key_present_in_every_locale(self, locale_data):
        data, name = locale_data
        line = data.get("health", {}).get("Result_em_line")
        assert line, f"Missing health.Result_em_line in {name}"
        assert "%{emission}" in line and "%{verdict}" in line, name
        # format_label("emission") already carries the "Em:" prefix — a
        # prefix in the template doubles it on screen (user-reported).
        assert "Em" not in line, f"Duplicated Em prefix in {name}"

    def test_codes_render_distinctly(self):
        from app.health_tab import emission_verdict_text
        rendered = {emission_verdict_text(c) for c in EMISSION_VERDICTS}
        assert len(rendered) == len(EMISSION_VERDICTS)

    def test_unknown_code_degrades_to_na(self):
        from app.health_tab import EMISSION_VERDICT_KEYS, emission_verdict_text
        assert emission_verdict_text("bogus") == t(EMISSION_VERDICT_KEYS[EMISSION_VERDICT_NA])


class TestResultPanelCallSite:
    """Call-site spy: the panel must render the *translated* verdict.

    A unit pin on ``emission_verdict_text`` does not prove the panel
    calls it — the pre-fix panel printed the ratio alone.
    """

    def _stub(self):
        from app.health_tab import HealthTab

        class _Stub:
            _fmt_pct = HealthTab._fmt_pct
            _fmt_err = HealthTab._fmt_err
            _update_result = HealthTab._update_result

        stub = _Stub()
        for name in ("result_index", "result_verdict", "result_delta",
                     "result_pct", "result_ia_abs", "result_bias",
                     "result_emission", "result_srk", "result_sg2"):
            setattr(stub, name, MagicMock())
        return stub

    def _measurement(self, verdict=EMISSION_VERDICT_WEAKENED, ratio=0.62):
        return {
            "health": {
                "index": 78.0, "verdict": "Good",
                "metrics": {"ia_pct": 73.0, "s_pct": 90.0, "r_pct": 95.0,
                            "k_pct": 92.0, "emission_ratio": ratio,
                            "emission_verdict": verdict},
                "raw": {"ia_op": 35.0},
            },
            "srk": {"s": 9.9, "r": 41.0, "k": 19.0, "sg2": None,
                    "mu_g1g2": None, "uncertainty": {}},
        }

    def test_panel_shows_translated_verdict(self):
        from app.health_tab import EMISSION_VERDICT_KEYS
        stub = self._stub()
        stub._update_result(self._measurement())
        text = stub.result_emission.setText.call_args[0][0]
        expected = t(EMISSION_VERDICT_KEYS[EMISSION_VERDICT_WEAKENED])
        assert expected != EMISSION_VERDICT_WEAKENED, "degenerate translation"
        assert expected in text
        assert "0.62" in text
        # Substring pins missed a doubled prefix once (user-reported
        # "Em: Em: 0.620"): pin the exact shape of the whole line.
        assert text.count("Em") == 1
        assert text.startswith("Em: 0.620")

    def test_panel_distinguishes_verdicts(self):
        # Same ratio string, different verdict -> different rendered line.
        stub_a, stub_b = self._stub(), self._stub()
        stub_a._update_result(self._measurement(EMISSION_VERDICT_NORMAL, 0.62))
        stub_b._update_result(self._measurement(EMISSION_VERDICT_EXHAUSTED, 0.62))
        assert (stub_a.result_emission.setText.call_args[0][0]
                != stub_b.result_emission.setText.call_args[0][0])

    def test_panel_falls_back_when_ratio_missing(self):
        stub = self._stub()
        m = self._measurement()
        m["health"]["metrics"]["emission_ratio"] = None
        stub._update_result(m)
        assert stub.result_emission.setText.call_args[0][0] == t("health.Result_em_none")


class TestCompositeVerdictLocalization:
    """The composite verdict is a persisted contract code; only its
    rendering is localized. Before this, the panel printed the raw code
    while the emission line next to it was translated."""

    def test_verdict_codes_come_from_the_registry(self):
        from lm19.health import HEALTH_VERDICTS, _verdict
        produced = {
            _verdict(None, _cfg()),
            _verdict(95.0, _cfg()),
            _verdict(80.0, _cfg()),
            _verdict(60.0, _cfg()),
            _verdict(10.0, _cfg()),
        }
        assert produced == set(HEALTH_VERDICTS)

    def test_key_map_is_a_bijection(self):
        from app.health_tab import HEALTH_VERDICT_KEYS
        from lm19.health import HEALTH_VERDICTS
        assert set(HEALTH_VERDICT_KEYS) == set(HEALTH_VERDICTS)

    def test_keys_present_in_every_locale(self):
        from app.health_tab import HEALTH_VERDICT_KEYS
        for name in [f"{loc}.json" for loc in available_locales()]:
            with open(LOCALES_DIR / name, encoding="utf-8") as f:
                data = json.load(f)
            for code, key in sorted(HEALTH_VERDICT_KEYS.items()):
                section, short = key.split(".", 1)
                assert short in data.get(section, {}), f"Missing {key} ({code}) in {name}"
                assert data[section][short].strip(), f"Empty {key} in {name}"

    def test_codes_render_distinctly(self):
        from app.health_tab import health_verdict_text
        from lm19.health import HEALTH_VERDICTS
        rendered = {health_verdict_text(c) for c in HEALTH_VERDICTS}
        assert len(rendered) == len(HEALTH_VERDICTS)

    def test_unknown_code_degrades_to_na(self):
        from app.health_tab import HEALTH_VERDICT_KEYS, health_verdict_text
        from lm19.health import HEALTH_VERDICT_NA
        assert health_verdict_text("Superb") == t(HEALTH_VERDICT_KEYS[HEALTH_VERDICT_NA])

    def test_panel_renders_the_translated_verdict(self):
        from app.health_tab import HEALTH_VERDICT_KEYS, HealthTab
        from lm19.health import HEALTH_VERDICT_GOOD

        class _Stub:
            _fmt_pct = HealthTab._fmt_pct
            _fmt_err = HealthTab._fmt_err
            _fmt_num = HealthTab._fmt_num
            _update_result = HealthTab._update_result

        stub = _Stub()
        for name in ("result_index", "result_verdict", "result_delta",
                     "result_pct", "result_ia_abs", "result_bias",
                     "result_emission", "result_srk", "result_sg2"):
            setattr(stub, name, MagicMock())
        stub._update_result({
            "health": {"index": 78.0, "verdict": HEALTH_VERDICT_GOOD,
                       "metrics": {"emission_ratio": None},
                       "raw": {"ia_op": 35.0}},
            "srk": {"s": 9.9, "r": 41.0, "k": 19.0, "sg2": None,
                    "mu_g1g2": None, "uncertainty": {}},
        })
        text = stub.result_verdict.setText.call_args[0][0]
        assert t(HEALTH_VERDICT_KEYS[HEALTH_VERDICT_GOOD]) in text

    def test_panel_verdict_is_localized_not_raw(self):
        # Discriminating version of the pin above: run under a locale that
        # renders the code differently, so "print the raw code" fails.
        from i18n_setup import locale_override
        from app.health_tab import HEALTH_VERDICT_KEYS, HealthTab
        from lm19.health import HEALTH_VERDICT_GOOD

        loc = _translating_locale(HEALTH_VERDICT_GOOD,
                                       HEALTH_VERDICT_KEYS[HEALTH_VERDICT_GOOD])
        if loc is None:
            pytest.skip("no locale translates the verdict away from its code")

        class _Stub:
            _fmt_pct = HealthTab._fmt_pct
            _fmt_err = HealthTab._fmt_err
            _fmt_num = HealthTab._fmt_num
            _update_result = HealthTab._update_result

        stub = _Stub()
        for name in ("result_index", "result_verdict", "result_delta",
                     "result_pct", "result_ia_abs", "result_bias",
                     "result_emission", "result_srk", "result_sg2"):
            setattr(stub, name, MagicMock())
        with locale_override(loc):
            stub._update_result({
                "health": {"index": 78.0, "verdict": HEALTH_VERDICT_GOOD,
                           "metrics": {"emission_ratio": None},
                           "raw": {"ia_op": 35.0}},
                "srk": {"s": 9.9, "r": 41.0, "k": 19.0, "sg2": None,
                        "mu_g1g2": None, "uncertainty": {}},
            })
            expected = t(HEALTH_VERDICT_KEYS[HEALTH_VERDICT_GOOD])
        text = stub.result_verdict.setText.call_args[0][0]
        assert expected in text
        assert HEALTH_VERDICT_GOOD not in text

    def test_history_filter_uses_the_same_codes(self):
        # The filter recomputes the verdict from the index; a drift
        # between its literals and the registry silently hides rows.
        from app.health_history import entry_matches_filter
        from lm19.health import (HEALTH_VERDICT_GOOD, HEALTH_VERDICT_STRONG,
                                 HEALTH_VERDICT_WEAK)
        thr = {"strong": 85, "good": 65, "weak": 40}
        entry = {"health": {"index": 70.0}}
        assert entry_matches_filter(entry, verdict_filter=HEALTH_VERDICT_GOOD,
                                    verdict_thresholds=thr) is True
        assert entry_matches_filter(entry, verdict_filter=HEALTH_VERDICT_STRONG,
                                    verdict_thresholds=thr) is False
        assert entry_matches_filter({"health": {"index": 50.0}},
                                    verdict_filter=HEALTH_VERDICT_WEAK,
                                    verdict_thresholds=thr) is True


class TestBiasServoPlanDefault:
    """A plan silent about the servo must fall back to the config, the
    same way the emission mode does."""

    def test_config_default_applies_when_plan_omits_the_key(self):
        from lm19.health import _parse_health_targets
        from test_health_logic import _cfg as _full_cfg, _lamp as _full_lamp
        lamp = _full_lamp()
        plan = {"op": {"ua": 250.0, "ug1": -7.0}}
        on = _parse_health_targets(plan, lamp,
                                   _full_cfg(health_bias_servo_enabled_default=True))
        off = _parse_health_targets(plan, lamp,
                                    _full_cfg(health_bias_servo_enabled_default=False))
        assert on.bias_servo_enabled is True
        assert off.bias_servo_enabled is False

    def test_explicit_plan_value_wins_over_config(self):
        from lm19.health import _parse_health_targets
        from test_health_logic import _cfg as _full_cfg, _lamp as _full_lamp
        targets = _parse_health_targets(
            {"bias_servo": {"enabled": False}}, _full_lamp(),
            _full_cfg(health_bias_servo_enabled_default=True))
        assert targets.bias_servo_enabled is False

    def test_no_cfg_defaults_to_off(self):
        from lm19.health import _parse_health_targets
        from test_health_logic import _lamp as _full_lamp
        assert _parse_health_targets({}, _full_lamp()).bias_servo_enabled is False


class TestVerdictFilterCombo:
    """The filter combo is the only place the codes meet the user as a
    choice: labels must be translated, data must stay raw codes, and the
    list must cover the whole registry."""

    def _combo(self):
        from PySide6.QtWidgets import QApplication, QComboBox
        from app.health_tab import populate_verdict_filter_combo
        QApplication.instance() or QApplication([])
        combo = QComboBox()
        populate_verdict_filter_combo(combo)
        return combo

    def test_covers_every_verdict_plus_all(self):
        from app.health_history import FILTER_ALL
        from lm19.health import HEALTH_VERDICT_ORDER
        combo = self._combo()
        data = [combo.itemData(i) for i in range(combo.count())]
        assert data[0] == FILTER_ALL
        assert set(data[1:]) == set(HEALTH_VERDICT_ORDER)

    def test_data_is_the_raw_code(self):
        from lm19.health import HEALTH_VERDICT_STRONG
        combo = self._combo()
        idx = combo.findData(HEALTH_VERDICT_STRONG)
        assert idx >= 0, "filter cannot select Strong by its code"

    def test_labels_are_localized(self):
        from i18n_setup import locale_override, translator_for
        from app.health_tab import HEALTH_VERDICT_KEYS
        from lm19.health import HEALTH_VERDICT_STRONG
        key = HEALTH_VERDICT_KEYS[HEALTH_VERDICT_STRONG]
        loc = next((l for l in available_locales()
                    if translator_for(l)(key) != HEALTH_VERDICT_STRONG), None)
        if loc is None:
            pytest.skip("no locale translates the verdict away from its code")
        with locale_override(loc):
            combo = self._combo()
            expected = translator_for(loc)(key)
        idx = combo.findData(HEALTH_VERDICT_STRONG)
        assert combo.itemText(idx) == expected

    def test_order_is_best_first(self):
        from lm19.health import HEALTH_VERDICT_ORDER
        combo = self._combo()
        data = [combo.itemData(i) for i in range(1, combo.count())]
        assert data == list(reversed(HEALTH_VERDICT_ORDER))
