"""Unit tests for Tube Health core logic."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import dataclasses
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm19.config import LampConfig, LampRange  # noqa: E402
from lm19.health import (  # noqa: E402
    run_health_test,
    _safe_pct, _clamp_score, _weighted_index, _verdict,
    _ug2_for_ua, _extract_refs, _build_ug1_sweep, _compute_scores,
    clamp_delta_ua, clamp_delta_ug1, clamp_delta_ug2,
    compute_shifted_r_center, compute_shifted_sg2_center,
    _raise_on_hw_error,
    _setup_op, _ramp_ug1_to_op, _run_stabilized_ia80, _HwState,
)
from lm19.analysis import (  # noqa: E402
    compute_srk_direct, estimate_srk_uncertainty,
)
from lm19.protocol import (  # noqa: E402
    decode_ug1, encode_ih, encode_uh, encode_ug1,
    decode_ig2, decode_ih, decode_uh,
    DELTA_UG1_MIN_V,
)
from lm19.calibration import CalibrationData, IA_HW_SCALE  # noqa: E402
from lm19.scan.exceptions import HealthProtectionError  # noqa: E402
from scan_test_helpers import _make_cal  # noqa: E402
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


class _FakeClient:
    def __init__(self):
        self.state = {
            "Ua": 0,
            "Ug1": 0,  # raw (abs*100), decode_ug1 returns negative
            "Ug2": 0,
            "Uh": encode_uh(6.3),
            "Ih": encode_ih(0.76),
            "An": 1,
        }

    def set_param(self, name, value):
        self.state[name] = int(value)

    def get_param(self, name, real=False):
        if name in self.state:
            return self.state[name]
        if name == "Ia":
            ua = float(self.state["Ua"])
            ug1 = decode_ug1(int(self.state["Ug1"]))
            uh = self.state["Uh"] / 10.0
            # Simple smooth synthetic model in mA.
            ia_ma = max(0.0, ua * 0.04 + (ug1 + 10.0) * 6.0) * (uh / 6.3)
            return int(round(ia_ma / IA_HW_SCALE))
        if name == "Ig2":
            ia_raw = float(self.get_param("Ia", real=True))
            ia_ma = ia_raw * IA_HW_SCALE
            ig2_ma = max(0.0, ia_ma * 0.12)
            return int(round(ig2_ma * 100.0))
        return 0

    def is_open(self):
        return True


def _cfg(**overrides):
    base = dict(
        health_ua_settle_per_volt_s=0.0,
        health_ua_settle_base_s=0.0,
        health_ug1_settle_per_volt_s=0.0,
        health_ug1_settle_base_s=0.0,
        health_ug2_settle_per_volt_s=0.0,
        health_ug2_settle_base_s=0.0,
        health_ua_tolerance_v=2.0,
        health_ug1_tolerance_v=0.3,
        health_ug2_tolerance_v=2.0,
        health_ua_retries=1,
        health_ug1_retries=1,
        health_ug2_retries=1,
        health_emission_enabled_default=True,
        health_emission_uh_ratio=0.8,
        health_emission_stable_warmup_ratio=0.01,
        health_emission_stable_min_s=1,
        health_emission_stable_max_s=2,
        health_emission_sample_period_s=0.01,
        health_emission_stable_window_points=3,
        health_emission_stable_slope_threshold_ma_per_s=1000.0,
        health_weight_ia=0.35,
        health_weight_s=0.40,
        health_weight_rh=0.10,
        health_weight_screen=0.0,
        health_weight_emission=0.15,
        health_renormalize_weights_if_metric_missing=True,
        health_verdict_strong_min=90.0,
        health_verdict_good_min=75.0,
        health_verdict_weak_min=55.0,
        health_emission_ratio_nominal=0.90,
        health_emission_ratio_weak_min=0.50,
        health_emission_min_ik_ratio=0.30,
        health_emission_mode_default="single",
        health_emission_uh_sweep_steps=5,
        health_emission_uh_sweep_min_ratio=0.70,
        health_emission_uh_sweep_abs_min_ratio=0.50,
        health_emission_knee_drop_pct=10.0,
        health_emission_sweep_max_total_s=600.0,
        health_bias_servo_enabled_default=False,
        health_bias_servo_tol_ma=0.5,
        health_bias_servo_max_shift_v=3.0,
        health_bias_servo_max_iter=8,
        health_bias_servo_step_v=0.5, health_bias_servo_pa_ceiling_pct=90.0,
        health_bias_servo_shift_margin=2.0,
        health_bias_servo_ug1_floor_v=0.1,
        health_emission_ratio_good_min=0.70,
        health_ia_samples=3,
        health_ia_sample_delay_ms=0,
        health_delta_ug2_pct=5,
        health_op_ramp_enabled=True,
        health_op_ug1_ramp_step_v=1.0,
        health_pa_safety_pct=120.0,
        health_pig2_safety_pct=120.0,
        ug1_after_stop=-24.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lamp(topology=TOPOLOGY_PENTODE):
    return LampConfig(
        tube_type="EL84",
        socket="B",
        anodes=1,
        warmup_s=120,
        topology=topology,
        uh=6.3,
        ih=0.76,
        ug1=-7.3,
        ua=250.0,
        ia=48.0,
        ug2=250.0 if topology != TOPOLOGY_TRIODE else 0.0,
        ig2=5.5 if topology != TOPOLOGY_TRIODE else 0.0,
        s=11.0,
        r=40.0,
        k=19.0,
        ranges={"ua": LampRange(0, 250, 10), "ug1": LampRange(-20, 0, 1), "ug2": LampRange(0, 250, 10)},
        limits={},
    )


@pytest.mark.smoke
class TestRunHealthTest:
    """End-to-end smoke tests for run_health_test()."""

    def test_without_emission_renormalizes_weights(self):
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("triode")

        m = run_health_test(
            client=client,
            lamp=lamp,
            cfg=cfg,
            calibration=CalibrationData(),
            lamp_id="L1",
            name="test",
            reference_mode="datasheet",
            reference=None,
            emission_enabled=False,
            warmup_s=lamp.warmup_s,
            measurement_plan={
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3},
            },
        )

        assert m["health"]["emission_enabled"] is False
        assert m["health"]["metrics"]["emission_ratio"] is None
        assert isinstance(m["health"]["index"], float)
        assert m["health"]["verdict"] in {"Strong", "Good", "Weak", "Replace"}
        assert isinstance(m.get("measurement_points"), list)
        assert len(m.get("measurement_points") or []) > 0

    def test_with_emission_produces_ratio(self):
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("pentode")

        m = run_health_test(
            client=client,
            lamp=lamp,
            cfg=cfg,
            calibration=CalibrationData(),
            lamp_id="L2",
            name="test2",
            reference_mode="datasheet",
            reference=None,
            emission_enabled=True,
            warmup_s=lamp.warmup_s,
            measurement_plan={
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3},
            },
        )

        er = m["health"]["metrics"]["emission_ratio"]
        assert er is not None
        assert er > 0
        assert "uh80_stabilization" in m["health"]["raw"]
        assert m["health"]["raw"]["uh80_stabilization"]["t_max_s"] is not None
        assert any(p.get("step") == "emission_80" for p in (m.get("measurement_points") or []))

    def test_conditions_have_nominal_uh_and_ih(self):
        """conditions carries the NOMINAL heater for BOTH channels (ih was
        previously missing — only uh was stored). The ACTUAL applied heater is
        NOT duplicated into conditions; it is preserved in the OP measurement
        point, so a stuck-reduced heater is detectable as
        measurement_points[op]["uh"] != conditions["uh"]."""
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=cfg, calibration=CalibrationData(),
            lamp_id="LC", name="cond", reference_mode="datasheet", reference=None,
            emission_enabled=False, warmup_s=lamp.warmup_s,
            measurement_plan={
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3},
            },
        )
        cond = m["conditions"]
        assert "uh" in cond and "ih" in cond                       # nominal, both channels
        assert "uh_actual" not in cond and "ih_actual" not in cond  # not duplicated
        # The actual OP heater IS preserved in the OP measurement point — the
        # OP point is dict(point_op, step="op"), so point_op's uh/ih are copied.
        op_pt = next(p for p in (m.get("measurement_points") or [])
                     if p.get("step") == "op")
        assert "uh" in op_pt and isinstance(op_pt["uh"], (int, float))
        assert "ih" in op_pt and isinstance(op_pt["ih"], (int, float))

    def test_computes_rk_without_last_srk(self):
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("pentode")

        m = run_health_test(
            client=client,
            lamp=lamp,
            cfg=cfg,
            calibration=CalibrationData(),
            lamp_id="L3",
            name="test3",
            reference_mode="datasheet",
            reference=None,
            emission_enabled=False,
            warmup_s=lamp.warmup_s,
            measurement_plan={
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 7, "repeats": 3},
            },
        )

        srk = m.get("srk", {})
        assert isinstance(srk.get("s"), (int, float))
        assert isinstance(srk.get("r"), (int, float))
        assert isinstance(srk.get("k"), (int, float))


# ── Pure helpers in lm19.health ──────────────────────────────────────

class TestSafePct:
    def test_normal(self):
        assert _safe_pct(50.0, 100.0) == pytest.approx(50.0)

    def test_none_value(self):
        assert _safe_pct(None, 100.0) is None

    def test_none_ref(self):
        assert _safe_pct(50.0, None) is None

    def test_zero_ref(self):
        assert _safe_pct(50.0, 0.0) is None

    def test_over_100(self):
        assert _safe_pct(120.0, 100.0) == pytest.approx(120.0)


class TestClampScore:
    def test_normal(self):
        assert _clamp_score(80.0) == pytest.approx(80.0)

    def test_none(self):
        assert _clamp_score(None) is None

    def test_below_zero(self):
        assert _clamp_score(-5.0) == pytest.approx(0.0)

    def test_above_max(self):
        assert _clamp_score(200.0) == pytest.approx(130.0)

    def test_custom_max(self):
        assert _clamp_score(200.0, max_pct=150.0) == pytest.approx(150.0)


class TestWeightedIndex:
    def test_simple(self):
        metrics = {"ia": 100.0, "s": 80.0}
        weights = {"ia": 0.5, "s": 0.5}
        assert _weighted_index(metrics, weights, False) == pytest.approx(90.0)

    def test_renormalize_missing(self):
        metrics = {"ia": 100.0, "s": None}
        weights = {"ia": 0.5, "s": 0.5}
        assert _weighted_index(metrics, weights, True) == pytest.approx(100.0)

    def test_all_none(self):
        metrics = {"ia": None, "s": None}
        weights = {"ia": 0.5, "s": 0.5}
        assert _weighted_index(metrics, weights, False) is None


class TestVerdict:
    def test_strong(self):
        cfg = _cfg()
        assert _verdict(95.0, cfg) == "Strong"

    def test_good(self):
        assert _verdict(80.0, _cfg()) == "Good"

    def test_weak(self):
        assert _verdict(60.0, _cfg()) == "Weak"

    def test_replace(self):
        assert _verdict(30.0, _cfg()) == "Replace"

    def test_none(self):
        assert _verdict(None, _cfg()) == "N/A"


class TestUg2ForUa:
    def test_triode(self):
        lamp = _lamp("triode")
        assert _ug2_for_ua(lamp, 200.0, False, 0.0, 100.0) == 0.0

    def test_pentode_fixed(self):
        lamp = _lamp("pentode")
        assert _ug2_for_ua(lamp, 200.0, False, 0.0, 250.0) == 250.0

    def test_pentode_track(self):
        lamp = _lamp("pentode")
        assert _ug2_for_ua(lamp, 200.0, True, 10.0, 250.0) == 210.0

    def test_pentode_track_clamp_zero(self):
        lamp = _lamp("pentode")
        assert _ug2_for_ua(lamp, 5.0, True, -20.0, 250.0) == 0.0


class TestExtractRefs:
    def test_from_lamp_defaults(self):
        lamp = _lamp("pentode")
        refs = _extract_refs(None, lamp)
        assert refs["ia"] is None
        assert refs["s"] is None
        assert refs["rh"] == pytest.approx(lamp.uh / lamp.ih)
        assert refs["screen_ratio"] == pytest.approx(lamp.ig2 / lamp.ia)

    def test_from_reference_dict(self):
        lamp = _lamp("pentode")
        ref = {"reference": {"ia": 50.0, "s": 12.0, "r": 35.0, "k": 20.0}}
        refs = _extract_refs(ref, lamp)
        assert refs["ia"] == pytest.approx(50.0)
        assert refs["s"] == pytest.approx(12.0)
        assert refs["r"] == pytest.approx(35.0)


class TestBuildUg1Sweep:
    def test_basic_2_point(self):
        sweep = _build_ug1_sweep(-5.0, 0.5, 2)
        assert len(sweep) == 2
        assert sweep[0] == pytest.approx(-5.5)
        assert sweep[1] == pytest.approx(-4.5)

    def test_multi_point_excludes_op(self):
        sweep = _build_ug1_sweep(-5.0, 1.0, 5)
        for v in sweep:
            assert abs(v - (-5.0)) > 1e-6

    def test_n_ug1_3_gives_2_points(self):
        sweep = _build_ug1_sweep(-5.0, 1.0, 3)
        assert len(sweep) == 2


class TestComputeSrkDirect:
    def test_linear_data(self):
        s_ug1 = [-6.0, -5.0, -4.0]
        s_ia = [10.0, 12.0, 14.0]
        r_ua = [190.0, 210.0]
        r_ia = [9.5, 10.5]
        s, r, k = compute_srk_direct(s_ug1, s_ia, r_ua, r_ia)
        assert s == pytest.approx(2.0)
        assert r == pytest.approx(20.0)
        assert k == pytest.approx(40.0)

    def test_insufficient_r_data(self):
        s, r, k = compute_srk_direct([-5.0, -4.0], [10.0, 12.0], [200.0], [10.0])
        assert s is not None
        assert r is None
        assert k is None


class TestEstimateSrkUncertainty:
    def test_returns_all_keys(self):
        unc = estimate_srk_uncertainty(2.0, 20.0, 25.0, 0.8, 5)
        assert "s_rel" in unc
        assert "r_rel" in unc
        assert "k_rel" in unc
        assert all(v is not None and v > 0 for v in unc.values())

    def test_more_repeats_lower_error(self):
        unc5 = estimate_srk_uncertainty(2.0, 20.0, 25.0, 0.8, 5)
        unc50 = estimate_srk_uncertainty(2.0, 20.0, 25.0, 0.8, 50)
        assert unc50["s_rel"] < unc5["s_rel"]
        assert unc50["r_rel"] < unc5["r_rel"]

    def test_none_s(self):
        unc = estimate_srk_uncertainty(None, 20.0, 25.0, 0.8)
        assert unc["s_rel"] is None
        assert unc["r_rel"] is not None
        assert unc["k_rel"] is None


class TestComputeScores:
    def test_all_matching_refs(self):
        lamp = _lamp("pentode")
        cfg = _cfg()
        point_op = {"ia": 48.0, "ig2": 5.5, "uh": 6.3, "ih": 0.76}
        refs = {"ia": 48.0, "s": 11.0, "r": 40.0, "k": 19.0,
                "rh": 6.3 / 0.76, "screen_ratio": 5.5 / 48.0}
        scores = _compute_scores(
            ia_op=48.0, s=11.0, r=40.0, k=19.0,
            point_op=point_op, lamp=lamp, cfg=cfg,
            refs=refs, emission_enabled=False,
            ia80=None, ia100=None)
        assert scores["ia_pct"] == pytest.approx(100.0)
        assert scores["s_pct"] == pytest.approx(100.0)
        assert scores["r_pct"] == pytest.approx(100.0)
        assert scores["verdict"] == "Strong"

    def test_srk_uncertainty_in_result(self):
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=cfg,
            calibration=CalibrationData(),
            lamp_id="L4", name="unc_test", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3}},
        )
        unc = m["srk"].get("uncertainty")
        assert unc is not None
        assert isinstance(unc.get("s_rel"), float)
        assert isinstance(unc.get("r_rel"), float)
        assert isinstance(unc.get("k_rel"), float)


class TestClampDeltaUa:
    def test_normal_range(self):
        assert clamp_delta_ua(25.0) == 25.0

    def test_clamp_below_min(self):
        assert clamp_delta_ua(3.0) == 10.0

    def test_clamp_above_max(self):
        assert clamp_delta_ua(80.0) == 50.0

    def test_rounds_to_resolution(self):
        result = clamp_delta_ua(12.3)
        assert result == 12.0

    def test_percentage_for_typical_lamp(self):
        result = clamp_delta_ua(250.0 * 0.10)
        assert result == 25.0

    def test_percentage_for_low_ua(self):
        result = clamp_delta_ua(50.0 * 0.10)
        assert result == 10.0


class TestClampDeltaUg1:
    def test_normal_range(self):
        assert clamp_delta_ug1(1.0) == 1.0

    def test_clamp_below_min(self):
        assert clamp_delta_ug1(0.1) == pytest.approx(0.48, abs=0.04)

    def test_clamp_above_max(self):
        assert clamp_delta_ug1(5.0) == 2.0

    def test_rounds_to_resolution(self):
        result = clamp_delta_ug1(0.73)
        assert result == pytest.approx(0.72, abs=0.01)

    def test_small_ug1_gets_safe_minimum(self):
        result = clamp_delta_ug1(abs(-0.5) * 0.10)
        assert result >= DELTA_UG1_MIN_V - 0.04


class TestClampDeltaUg2:
    def test_normal_range(self):
        assert clamp_delta_ug2(13.0) == 13.0

    def test_clamp_below_min(self):
        assert clamp_delta_ug2(2.0) == 5.0

    def test_clamp_above_max(self):
        assert clamp_delta_ug2(80.0) == 50.0


class TestComputeShiftedRCenter:
    def test_no_shift_needed(self):
        center, method = compute_shifted_r_center(250.0, 25.0)
        assert center == 250.0
        assert method == "central"

    def test_shift_needed_upper_exceeds(self):
        center, method = compute_shifted_r_center(285.0, 29.0, ua_max=300.0)
        assert method == "shifted_op"
        assert center == 271.0
        assert center + 29.0 <= 300.0
        assert center - 29.0 >= 0.0

    def test_shift_near_max(self):
        center, method = compute_shifted_r_center(295.0, 15.0, ua_max=300.0)
        assert method == "shifted_op"
        assert center == 285.0
        assert center + 15.0 <= 300.0

    def test_no_shift_exact_fit(self):
        center, method = compute_shifted_r_center(275.0, 25.0, ua_max=300.0)
        assert center == 275.0
        assert method == "central"

    def test_shift_clamps_to_delta(self):
        center, method = compute_shifted_r_center(5.0, 25.0, ua_max=300.0)
        assert center >= 25.0
        assert center - 25.0 >= 0.0


class TestComputeShiftedSg2Center:
    def test_no_shift_needed(self):
        center, method = compute_shifted_sg2_center(250.0, 13.0)
        assert center == 250.0
        assert method == "central"

    def test_shift_needed(self):
        center, method = compute_shifted_sg2_center(295.0, 15.0, ug2_max=300.0)
        assert method == "shifted_op"
        assert center + 15.0 <= 300.0


class TestHealthResultContainsRMethod:
    def test_central_method_recorded(self):
        client = _FakeClient()
        cfg = _cfg()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=cfg,
            calibration=CalibrationData(),
            lamp_id="L5", name="method_test", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3}},
        )
        srk_plan = m["measurement_plan"]["srk"]
        assert "r_method" in srk_plan
        assert srk_plan["r_method"] in ("central", "shifted_op")
        assert "r_center" in srk_plan


# ── Integration: auto-calculated clamped deltas ──────────────────────

def _lamp_near_limit():
    """Lamp near device limit: Ua=285, Ug2=295 (295+delta over ug2_max=300 -> shifted_op; 285 was NOT near enough and the guarded body never ran).

    Ug1=-8 (not -20) so the fake client model produces non-zero Ia.
    """
    return LampConfig(
        tube_type="6F6S", socket="A", anodes=1, warmup_s=120,
        topology=TOPOLOGY_PENTODE,
        uh=6.3, ih=0.7, ug1=-8.0, ua=285.0, ia=38.0,
        ug2=295.0, ig2=7.0, s=2.55, r=50.0, k=0.0,
        ranges={"ua": LampRange(0, 300, 10), "ug1": LampRange(-24, 0, 1),
                "ug2": LampRange(0, 300, 10)},
        limits={},
    )


def _lamp_small_ug1():
    """Lamp with very small |Ug1|."""
    return LampConfig(
        tube_type="TEST_SMALL_UG1", socket="A", anodes=1, warmup_s=60,
        topology=TOPOLOGY_TRIODE,
        uh=6.3, ih=0.3, ug1=-0.5, ua=100.0, ia=5.0,
        ug2=0.0, ig2=0.0, s=3.0, r=20.0, k=60.0,
        ranges={"ua": LampRange(0, 300, 10), "ug1": LampRange(-5, 0, 0.1),
                "ug2": LampRange(0, 0, 0)},
        limits={},
    )


class TestAutoClampedDeltaIntegration:
    """run_health_test WITHOUT explicit delta — verifies auto-calculation."""

    def test_auto_delta_ua_is_clamped(self):
        client = _FakeClient()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="AC1", name="auto_clamp", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        delta_ua = m["measurement_plan"]["srk"]["delta_ua"]
        assert 10.0 <= delta_ua <= 50.0

    def test_auto_delta_ug1_is_clamped(self):
        client = _FakeClient()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="AC2", name="auto_clamp_ug1", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        delta_ug1 = m["measurement_plan"]["srk"]["delta_ug1"]
        assert 0.48 <= delta_ug1 <= 2.0

    def test_small_ug1_gets_safe_delta(self):
        client = _FakeClient()
        lamp = _lamp_small_ug1()
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="AC3", name="small_ug1", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        delta_ug1 = m["measurement_plan"]["srk"]["delta_ug1"]
        assert delta_ug1 >= 0.48, f"δUg1={delta_ug1} is too small for reliable gm"


class TestShiftedOpIntegration:
    """run_health_test with lamp near device limit — R-phase uses shifted OP."""

    def test_near_limit_uses_shifted_op(self):
        client = _FakeClient()
        lamp = _lamp_near_limit()
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="SH1", name="shifted", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        srk = m["measurement_plan"]["srk"]
        assert srk["r_method"] == "shifted_op"
        assert srk["r_center"] < lamp.ua

    def test_r_phase_points_within_device_limits(self):
        client = _FakeClient()
        lamp = _lamp_near_limit()
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="SH2", name="limits_check", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        r_points = [p for p in m["measurement_points"]
                     if p.get("step", "").startswith("srk_r")]
        assert len(r_points) == 2
        for p in r_points:
            assert p["ua"] <= 300.0, f"R-phase point Ua={p['ua']} exceeds 300V"
            assert p["ua"] >= 0.0

    def test_shifted_op_still_produces_valid_srk(self):
        client = _FakeClient()
        lamp = _lamp_near_limit()
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="SH3", name="valid_srk", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        srk = m["srk"]
        assert isinstance(srk["s"], (int, float)) and srk["s"] > 0
        assert isinstance(srk["r"], (int, float)) and srk["r"] > 0
        assert isinstance(srk["k"], (int, float))

    def test_pentode_sg2_shifted_when_ug2_near_limit(self):
        client = _FakeClient()
        lamp = _lamp_near_limit()
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="SH4", name="sg2_shifted", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        srk = m["measurement_plan"]["srk"]
        assert srk.get('sg2_method') == 'shifted_op', "guard de-vacuated 2026-07-12: value must be present"
        sg2_points = [p for p in m["measurement_points"]
                      if "sg2" in p.get("step", "")]
        for p in sg2_points:
            assert p["ug2"] <= 300.0, f"Sg2 point Ug2={p['ug2']} exceeds 300V"

    def test_normal_lamp_uses_central_method(self):
        client = _FakeClient()
        lamp = _lamp("triode")
        m = run_health_test(
            client=client, lamp=lamp, cfg=_cfg(),
            calibration=CalibrationData(),
            lamp_id="SH5", name="central", reference_mode="datasheet",
            emission_enabled=False, warmup_s=60,
            measurement_plan={"srk": {"points": 5, "repeats": 3}},
        )
        assert m["measurement_plan"]["srk"]["r_method"] == "central"
        assert m["measurement_plan"]["srk"]["r_center"] == lamp.ua


class TestClampDeltaCustomMinMax:
    """clamp_delta_* with custom min/max from config."""

    def test_ua_custom_min(self):
        assert clamp_delta_ua(3.0, min_v=5.0) == 5.0

    def test_ua_custom_max(self):
        assert clamp_delta_ua(40.0, max_v=30.0) == 30.0

    def test_ug1_custom_range(self):
        result = clamp_delta_ug1(0.3, min_v=0.2, max_v=1.0)
        assert 0.2 <= result <= 1.0

    def test_ug2_custom_min(self):
        assert clamp_delta_ug2(2.0, min_v=3.0) == 3.0


class TestRaiseOnHwError:
    """Tests for _raise_on_hw_error helper."""

    def test_zero_no_raise(self):
        """Er=0 → no exception."""
        _raise_on_hw_error(0)  # should not raise

    def test_overia_raises(self):
        """Er=0x02 (OVERIA) → RuntimeError with abbr."""
        with pytest.raises(RuntimeError, match="OC:Ia"):
            _raise_on_hw_error(0x02)

    def test_overig_raises(self):
        """Er=0x04 (OVERIG) → RuntimeError with abbr."""
        with pytest.raises(RuntimeError, match="OC:Ig2"):
            _raise_on_hw_error(0x04)

    def test_overih_raises(self):
        """Er=0x01 (OVERIH) → RuntimeError with abbr."""
        with pytest.raises(RuntimeError, match="OC:Ih"):
            _raise_on_hw_error(0x01)

    def test_overte_raises(self):
        """Er=0x08 (OVERTE) → RuntimeError with abbr."""
        with pytest.raises(RuntimeError, match="OT"):
            _raise_on_hw_error(0x08)

    def test_multiple_flags(self):
        """Er=0x06 (OVERIA+OVERIG) → both abbrs in message."""
        with pytest.raises(RuntimeError, match="OC:Ia.*OC:Ig2"):
            _raise_on_hw_error(0x06)

    def test_unknown_bits_raises(self):
        """Er with no known bits (e.g. 0x10) → still raises with hex value."""
        with pytest.raises(RuntimeError, match="Er=0x10"):
            _raise_on_hw_error(0x10)

    def test_false_value_no_raise(self):
        """Er=None or Er=False → no exception (falsy check)."""
        _raise_on_hw_error(None)
        _raise_on_hw_error(False)


# ── Plan B (docs/CALIBRATION_PLAN.md §5.5): calibration in health OP path ──


class _CalAwareClient(_FakeClient):
    """_FakeClient + set_param recording + linear hardware error models.

    ``read_cals = {"Ua": (gain, offset)}``: the device PHYSICALLY holds
    the commanded value, but its ADC reading r satisfies
    ``apply_read(r) == physical``, i.e. r = (phys - offset) / gain.
    Only raw==volts channels (Ua, Ug2) are supported by this branch.

    ``ug1_dac_gain != 1.0``: Ug1 DAC error — physical = decoded_cmd *
    ug1_dac_gain; readback encodes the physical value. A SET cal with
    gain = 1/ug1_dac_gain is then the exact feedforward correction.
    """

    def __init__(self, read_cals=None, ug1_dac_gain=1.0):
        super().__init__()
        self.read_cals = read_cals or {}
        self.ug1_dac_gain = ug1_dac_gain
        self.set_calls = []

    def set_param(self, name, value):
        self.set_calls.append((name, int(value)))
        super().set_param(name, value)

    def ug1_raws(self):
        return [v for n, v in self.set_calls if n == "Ug1"]

    def get_param(self, name, real=False):
        # real=False (setpoint readback in the protection check) must stay
        # raw and undistorted — fall through to the plain state lookup.
        if real and name == "Ug1" and self.ug1_dac_gain != 1.0:
            phys = decode_ug1(int(self.state["Ug1"])) * self.ug1_dac_gain
            return int(round(abs(phys) * 100.0))
        if real and name in self.read_cals:
            gain, offset = self.read_cals[name]
            # Real LM19Serial.get_param returns int (raw == volts for
            # Ua/Ug2) — the mock must match the wire interface.
            return int(round((float(self.state[name]) - offset) / gain))
        return super().get_param(name, real=real)


class TestSetupOpCalibrated(unittest.TestCase):
    """_setup_op verify must compare in the physical domain.

    Today (health.py:574) it compares the raw decoded actual against the
    physical target, so a non-default READ cal makes a perfectly healthy
    device fail verify. After plan B the adapter returns
    apply_read(decoded) and all 6 caller-side verifies pass.
    """

    @patch("time.sleep")
    def test_setup_op_passes_with_nondefault_read_cal(self, _sleep):
        read_gain = 1.02
        # Device physically reaches 200 V; ADC reading is 200/1.02 = 196.08.
        # Old code: |196.08 - 200| = 3.92 > tol 2.0 → RuntimeError.
        client = _CalAwareClient(read_cals={"Ua": (read_gain, 0.0)})
        cfg = _cfg()
        lamp = _lamp("triode")
        cal = _make_cal(channel="ua", read_gain=read_gain)
        hw = _HwState(prev_ua=0.0, prev_ug1=0.0, prev_ug2=0.0, prev_uh=6.3)

        point = _setup_op(
            client, cfg, cal, hw, lamp,
            200.0, -7.3, 0.0,
            ug2_mode=TOPOLOGY_TRIODE, lamp_id="CAL1",
            progress=None, stop=None,
        )

        # Caller domain is physical: actual equals the requested target
        # within wire quantization (int raw → ±read_gain/2 ≈ ±0.51 V).
        self.assertAlmostEqual(hw.actual_ua, 200.0, delta=0.6)
        self.assertAlmostEqual(point["ua"], 200.0, delta=0.6)
        # No SET cal on ua → the raw command stays the plain target
        # (guards against compound READ-into-SET correction).
        ua_raws = [v for n, v in client.set_calls if n == "Ua"]
        self.assertEqual(ua_raws, [200])

    @patch("time.sleep")
    def test_setup_op_pentode_ug2_verify_calibrated(self, _sleep):
        """Pentode path: the Ug2 verify (plan §5.5) must also hold with a
        non-default READ cal — same domain rule as Ua."""
        read_gain = 1.02
        client = _CalAwareClient(read_cals={"Ua": (read_gain, 0.0),
                                            "Ug2": (read_gain, 0.0)})
        cfg = _cfg()
        lamp = _lamp("pentode")
        cal = _make_cal(channel="ua", read_gain=read_gain)
        cal.set_channel("ug2", "read", read_gain, 0.0)
        hw = _HwState(prev_ua=0.0, prev_ug1=0.0, prev_ug2=0.0, prev_uh=6.3)

        _setup_op(
            client, cfg, cal, hw, lamp,
            250.0, -7.3, 250.0,
            ug2_mode=TOPOLOGY_PENTODE, lamp_id="CAL2",
            progress=None, stop=None,
        )

        self.assertAlmostEqual(hw.actual_ua, 250.0, delta=0.6)
        self.assertAlmostEqual(hw.actual_ug2, 250.0, delta=0.6)

    @patch("time.sleep")
    def test_setup_op_legacy_verifies_calibrated(self, _sleep):
        """No-ramp fallback (_setup_op_legacy, health_op_ramp_enabled=False):
        its 3 caller-side verifies (Ug1/Ua/Ug2) must hold with a
        non-default READ cal too."""
        read_gain = 1.02
        client = _CalAwareClient(read_cals={"Ua": (read_gain, 0.0),
                                            "Ug2": (read_gain, 0.0)})
        cfg = _cfg(health_op_ramp_enabled=False)
        lamp = _lamp("pentode")
        cal = _make_cal(channel="ua", read_gain=read_gain)
        cal.set_channel("ug2", "read", read_gain, 0.0)
        hw = _HwState(prev_ua=0.0, prev_ug1=0.0, prev_ug2=0.0, prev_uh=6.3)

        _setup_op(
            client, cfg, cal, hw, lamp,
            250.0, -7.3, 250.0,
            ug2_mode=TOPOLOGY_PENTODE, lamp_id="CAL3",
            progress=None, stop=None,
        )

        self.assertAlmostEqual(hw.actual_ua, 250.0, delta=0.6)
        self.assertAlmostEqual(hw.actual_ug2, 250.0, delta=0.6)
        self.assertAlmostEqual(hw.actual_ug1, -7.3, delta=0.1)


class TestRampUg1Feedforward(unittest.TestCase):
    """Ug1 ramp steps: feedforward-corrected commands, raw safe lock."""

    @patch("time.sleep")
    def test_ramp_ug1_steps_are_feedforward_corrected(self, _sleep):
        set_gain = 1.05
        # DAC is weak by 1/set_gain, so a SET cal with gain=set_gain is the
        # exact correction: commanded raw must be encode_ug1(step * 1.05).
        client = _CalAwareClient(ug1_dac_gain=1.0 / set_gain)
        cfg = _cfg(ug1_after_stop=-10.0)
        lamp = _lamp("triode")
        cal = _make_cal(channel="ug1", set_gain=set_gain)
        # Pre-condition of _ramp_ug1_to_op: Ug1 already at safe lock.
        client.state["Ug1"] = encode_ug1(-10.0)
        hw = _HwState(prev_ua=0.0, prev_ug1=-10.0, prev_ug2=0.0, prev_uh=6.3)

        _ramp_ug1_to_op(
            client, cfg, cal, lamp, hw,
            target_ug1=-6.0, lamp_id="CAL2", ug2_mode=TOPOLOGY_TRIODE,
            progress=None, stop=None,
        )

        steps = [-9.0, -8.0, -7.0, -6.0]
        # Expected raws computed by hand, NOT via cal.apply_set(): the
        # current SET_LIMITS["ug1"] = (0, 24) would clamp negative commands
        # to 0 V and bake the old bug into the expectation.
        expected = [int(round(abs(s * set_gain) * 100.0)) for s in steps]
        self.assertEqual(client.ug1_raws(), expected)
        # Bookkeeping stays physical: the tube really sits at -6.0 V.
        self.assertAlmostEqual(hw.prev_ug1, -6.0, places=6)

    @patch("time.sleep")
    def test_safe_lock_stays_raw(self, _sleep):
        """PIN (green now and after plan B): safe-lock restore is RAW.

        apply_set(safe_lock) could clamp/shift the bias toward 0 V =
        maximum anode current — the dangerous direction. The restore after
        a protection trip must keep sending encode_ug1(cfg.ug1_after_stop)
        unmodified, with only the bookkeeping in the physical domain.
        """
        client = _CalAwareClient()
        client.state["Ua"] = 250
        cfg = _cfg(ug1_after_stop=-12.0)
        lamp = _lamp("triode")
        # Pa limit 2.0 W × 120% safety = 2.4 W: the fake Ia model crosses
        # it mid-ramp (Ia=10 mA @ Ug1=-10 → Pa=2.5 W), before the target.
        lamp.pa_max = 2.0
        cal = _make_cal(channel="ug1", set_gain=1.05)
        client.state["Ug1"] = encode_ug1(-12.0)
        hw = _HwState(prev_ua=250.0, prev_ug1=-12.0, prev_ug2=0.0, prev_uh=6.3)

        with self.assertRaises(HealthProtectionError):
            _ramp_ug1_to_op(
                client, cfg, cal, lamp, hw,
                target_ug1=-7.3, lamp_id="CAL3", ug2_mode=TOPOLOGY_TRIODE,
                progress=None, stop=None,
            )

        # Last Ug1 command is the safe-lock restore: raw despite SET cal.
        self.assertEqual(client.ug1_raws()[-1], encode_ug1(-12.0))
        # Bookkeeping is physical (no READ cal on ug1 → identity).
        self.assertAlmostEqual(hw.prev_ug1, -12.0, places=6)


class TestHealthSrkSettleThreadsStop(unittest.TestCase):
    """SRK phase settles must thread ``stop`` into ``_set_param_calibrated`` so
    an in-flight multi-second settle (e.g. the Ug1 jump back to OP) is
    cancellable — not only the between-settle ``if stop(): break``.

    Audit finding #6: previously 15/16 health settle calls dropped stop,
    leaving each settle non-interruptible (and, after an emergency zero, able to
    re-assert the setpoint via verify-retry).
    """

    def _run_and_capture(self, call):
        import lm19.health as H
        captured = []

        def fake_calibrated(*a, stop=None, **kw):
            captured.append(stop)
            return 0.0

        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 200.0, "ia": 10.0, "er": 0}
        with patch.object(H, "_set_param_calibrated", side_effect=fake_calibrated), \
             patch.object(H, "_read_measurement_point", return_value=pt), \
             patch.object(H, "_raise_on_hw_error"), \
             patch("time.sleep"):
            call(H, captured)
        return captured

    def test_s_phase_threads_stop(self):
        marker = lambda: False
        def call(H, captured):
            hw = H._HwState(prev_ua=250.0, prev_ug1=-5.0, prev_ug2=0.0, prev_uh=6.3)
            H._measure_s_phase(
                object(), _cfg(), None, hw,
                ug1_sweep=[-6.0, -7.0], n_repeats=1,
                ia_op=10.0, target_ug1_op=-5.0,
                srk_idx=0, total_srk=2,
                measurement_points=[], progress=None, stop=marker,
            )
        captured = self._run_and_capture(call)
        self.assertTrue(captured, "s-phase performed no settle")
        self.assertTrue(all(s is marker for s in captured),
                        "every S-phase Ug1 settle must receive stop")

    def test_r_phase_threads_stop(self):
        marker = lambda: False
        def call(H, captured):
            hw = H._HwState(prev_ua=250.0, prev_ug1=-5.0, prev_ug2=0.0, prev_uh=6.3)
            H._measure_r_phase(
                object(), _lamp("triode"), _cfg(), None, hw,
                target_ua_op=250.0, target_ug1_op=-5.0, target_ug2_op=0.0,
                delta_ua=10.0, n_repeats=1,
                ug2_track_ua=False, ug2_offset=0.0,
                srk_idx=0, total_srk=2,
                measurement_points=[], progress=None, stop=marker,
            )
        captured = self._run_and_capture(call)
        self.assertTrue(captured, "r-phase performed no settle")
        self.assertTrue(all(s is marker for s in captured),
                        "every R-phase Ug1/Ua settle must receive stop")

    def test_sg2_phase_threads_stop(self):
        marker = lambda: False
        def call(H, captured):
            hw = H._HwState(prev_ua=250.0, prev_ug1=-5.0, prev_ug2=200.0, prev_uh=6.3)
            H._measure_sg2_phase(
                object(), _cfg(), None, hw,
                target_ua_op=250.0,
                target_ug2_op=200.0, delta_ug2=10.0, n_repeats=1,
                srk_idx=0, total_srk=2,
                measurement_points=[], progress=None, stop=marker,
            )
        captured = self._run_and_capture(call)
        self.assertTrue(captured, "sg2-phase performed no settle")
        self.assertTrue(all(s is marker for s in captured),
                        "every Sg2-phase Ug2 settle must receive stop")


class TestSg2PhaseAtOp(unittest.TestCase):
    """ML-060: the R-phase leaves Ua at its last sweep point
    (r_center + delta); the Sg2 phase must restore Ua to the OP BEFORE
    sweeping Ug2 - otherwise Sg2 and mu_g1g2 carry a systematic Ua bias."""

    def test_ua_restored_before_first_ug2_set(self):
        import lm19.health as H
        calls = []

        def fake_calibrated(client, name, chkey, target, prev, cal, **kw):
            calls.append((name, float(target)))
            return float(target)

        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 200.0, "ia": 10.0, "er": 0}
        with patch.object(H, "_set_param_calibrated", side_effect=fake_calibrated),              patch.object(H, "_read_measurement_point", return_value=pt),              patch.object(H, "_raise_on_hw_error"),              patch("time.sleep"):
            hw = H._HwState(prev_ua=260.0, prev_ug1=-5.0, prev_ug2=200.0,
                            prev_uh=6.3)
            H._measure_sg2_phase(
                object(), _cfg(), None, hw,
                target_ua_op=250.0,
                target_ug2_op=200.0, delta_ug2=10.0, n_repeats=1,
                srk_idx=0, total_srk=2,
                measurement_points=[], progress=None, stop=None,
            )
        self.assertTrue(calls, "sg2 phase made no set calls")
        self.assertEqual(calls[0], ("Ua", 250.0),
                         "Sg2 phase must restore Ua to OP before Ug2 sweep")
        self.assertEqual([t for n, t in calls if n == "Ug2"], [190.0, 210.0])
        # Mutation-audit: dropping the prev_ua bookkeeping after
        # the restore survived - the NEXT calibrated set would settle from
        # a stale voltage estimate.
        self.assertEqual(hw.prev_ua, 250.0)

    def test_stopped_sg2_phase_sends_nothing(self):
        """Mutation-audit: removing the stop-gate on the Ua
        restore survived - a cancelled worker must NOT re-assert a
        setpoint after the emergency zero (ML-131 class)."""
        import lm19.health as H
        calls = []

        def fake_calibrated(client, name, chkey, target, prev, cal, **kw):
            calls.append((name, float(target)))
            return float(target)

        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 200.0, "ia": 10.0, "er": 0}
        with patch.object(H, "_set_param_calibrated", side_effect=fake_calibrated),              patch.object(H, "_read_measurement_point", return_value=pt),              patch.object(H, "_raise_on_hw_error"),              patch("time.sleep"):
            hw = H._HwState(prev_ua=260.0, prev_ug1=-5.0, prev_ug2=200.0,
                            prev_uh=6.3)
            H._measure_sg2_phase(
                object(), _cfg(), None, hw,
                target_ua_op=250.0,
                target_ug2_op=200.0, delta_ug2=10.0, n_repeats=1,
                srk_idx=0, total_srk=2,
                measurement_points=[], progress=None,
                stop=lambda: True,
            )
        self.assertEqual(calls, [],
                         "stopped Sg2 phase must not command hardware")

    def test_run_srk_phase_passes_op_ua_not_r_center(self):
        """Mutation-audit: the call site passing r_center
        instead of targets.target_ua survived the unit pin (they are
        equal away from device limits). target_ua=298 with delta=10
        shifts r_center to 290 - the values must diverge here."""
        import lm19.health as H
        seen = {}

        def spy_sg2(client, cfg, calibration, hw, target_ua_op,
                    target_ug2_op, delta_ug2, n_repeats, **kw):
            seen["target_ua_op"] = target_ua_op
            return [190.0, 210.0], [9.0, 11.0], kw.get("srk_idx", 0)

        targets = H._HealthTargets(
            op_plan={}, srk_plan={"delta_ua": 10.0, "points": 4,
                                  "repeats": 1},
            emission_plan={}, an=1,
            ug2_track_ua=False, ug2_offset=0.0,
            target_ua=298.0, target_ug1=-5.0, target_ug2=200.0,
            is_pentode_mode=True,
        )
        r_center, _ = H.compute_shifted_r_center(298.0, 10.0)
        self.assertNotEqual(r_center, 298.0,
                            "precondition: r_center must differ from OP")
        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 200.0, "ia": 10.0, "er": 0}
        with patch.object(H, "_measure_s_phase",
                          return_value=([-6.0, -5.0, -4.0],
                                        [9.0, 10.0, 11.0], 3)),              patch.object(H, "_measure_r_phase",
                          return_value=([288.0, 292.0], [9.9, 10.1], 5)),              patch.object(H, "_measure_sg2_phase", side_effect=spy_sg2),              patch.object(H, "_restore_op"),              patch.object(H, "_set_param_calibrated",
                          side_effect=lambda *a, **k: float(a[3])),              patch.object(H, "_read_measurement_point", return_value=pt),              patch.object(H, "_raise_on_hw_error"),              patch("time.sleep"):
            hw = H._HwState(prev_ua=298.0, prev_ug1=-5.0, prev_ug2=200.0,
                            prev_uh=6.3)
            H._run_srk_phase(
                object(), _cfg(), None, _lamp("pentode"), hw, targets,
                ia_op=10.0, measurement_points=[], progress=None, stop=None,
            )
        self.assertEqual(seen.get("target_ua_op"), 298.0,
                         "Sg2 phase must receive the OP Ua, not r_center")


class TestUh80EventCalibrated(unittest.TestCase):
    """uh80_stabilizing progress event must carry calibrated readings.

    Today only ia_ma goes through apply_read (health.py:464); ig2/uh/ih
    are emitted as raw decodes (465-467) — the UI display lies when a
    READ cal is non-default.
    """

    @patch("time.sleep")
    def test_uh80_event_readings_calibrated(self, _sleep):
        client = _FakeClient()
        # Prime an OP so the fake Ia/Ig2 model yields non-zero currents.
        client.state["Ua"] = 200
        client.state["Ug1"] = encode_ug1(-7.3)
        cal = _make_cal(channel="ig2", read_gain=1.05)
        cal.set_channel("uh", "read", 1.02, 0.0)
        cal.set_channel("ih", "read", 0.98, 0.0)

        events = []
        # time.time is left real: the loop exits via the stability check
        # (slope threshold 1000 → stable at the 3rd sample), and even a
        # degenerate timeout exit still emits >= 1 event to assert on.
        _run_stabilized_ia80(
            client, _cfg(), cal, warmup_s=120,
            progress=events.append, stop=None,
        )

        uh80 = [e for e in events if e.get("event") == "uh80_stabilizing"]
        self.assertGreaterEqual(len(uh80), 1)
        ev = uh80[-1]
        # Expected = apply_read of the decoded device reading; the fake
        # client state is constant, so re-reading it here is exact.
        expected_ig2 = cal.apply_read(
            "ig2", decode_ig2(client.get_param("Ig2", real=True)))
        expected_uh = cal.apply_read(
            "uh", decode_uh(client.get_param("Uh", real=True)))
        expected_ih = cal.apply_read(
            "ih", decode_ih(client.get_param("Ih", real=True)))
        self.assertAlmostEqual(ev["ig2_ma"], expected_ig2, places=6)
        self.assertAlmostEqual(ev["uh"], expected_uh, places=6)
        self.assertAlmostEqual(ev["ih"], expected_ih, places=6)


class _RecordingFakeClient(_FakeClient):
    """_FakeClient that also records every set_param (name, value)."""

    def __init__(self):
        super().__init__()
        self.set_calls = []

    def set_param(self, name, value):
        self.set_calls.append((name, int(value)))
        super().set_param(name, value)


class TestEmissionHeaterMode(unittest.TestCase):
    """#814: emission drives the heater on the lamp's actual channel — a
    current-heater lamp (uh=0, ih>0) must NOT be sent Uh=0 (heater off)."""

    @patch("time.sleep")
    def test_heater_set_drives_ih_for_current_heater(self, _):
        import lm19.health as H
        from lm19.protocol import encode_ih
        client = MagicMock()
        client.get_param.return_value = encode_ih(0.3)
        H._set_heater_with_verify(client, False, 0.3, 0.0, CalibrationData())
        names = [c.args[0] for c in client.set_param.call_args_list]
        self.assertIn("Ih", names)
        self.assertNotIn("Uh", names)  # current-heater: never touch Uh

    @patch("time.sleep")
    def test_heater_set_drives_uh_for_voltage_heater(self, _):
        import lm19.health as H
        from lm19.protocol import encode_uh
        client = MagicMock()
        client.get_param.return_value = encode_uh(6.3)
        H._set_heater_with_verify(client, True, 6.3, 0.0, CalibrationData())
        names = [c.args[0] for c in client.set_param.call_args_list]
        self.assertIn("Uh", names)
        self.assertNotIn("Ih", names)

    @patch("time.sleep")
    def test_emission_current_heater_drives_ih_not_uh_zero(self, _):
        import lm19.health as H
        client = _RecordingFakeClient()
        hw = H._HwState(prev_ua=250.0, prev_ug1=-5.0, prev_ug2=0.0, prev_uh=0.0)
        lamp = dataclasses.replace(_lamp("triode"), uh=0.0, ih=0.3)
        op_plan = {"uh": 0.0, "ih": 0.3, "ua": 250.0, "ug1": -5.0, "ug2": 0.0}
        H._run_emission(client, _cfg(), CalibrationData(), hw, lamp,
                        op_plan, {}, 0, measurement_points=[],
                        progress=None, stop=None)
        names = [n for n, _ in client.set_calls]
        self.assertIn("Ih", names, "current-heater emission must drive Ih")
        self.assertNotIn(("Uh", 0), client.set_calls,
                         "current-heater emission must never send Uh=0")


class TestStopAbortsSrk(unittest.TestCase):
    """#669: a stop during the SRK phases must ABORT (raise), not produce a
    partial result that could be saved as a personal baseline."""

    @patch("time.sleep")
    def test_run_health_test_raises_when_stopped_after_srk(self, _):
        import lm19.health as H
        op_point = {"ua": 250.0, "ug1": -5.0, "ug2": 0.0, "ia": 10.0,
                    "ig2": 0.0, "uh": 6.3, "ih": 0.76, "er": 0}
        srk = {"s_measured": 1.0, "r_measured": 1.0, "k_measured": 1.0,
               "sg2_measured": None, "mu_g1g2": None, "srk_uncertainty": None,
               "sg2_rel": None, "delta_ua": 25.0, "delta_ug1": 0.8,
               "delta_ug2": 0.0, "n_points": 5, "n_repeats": 3,
               "r_center": 250.0, "r_method": "central",
               "sg2_center": 0.0, "sg2_method": "central"}
        with patch.object(H, "_setup_op", return_value=op_point), \
             patch.object(H, "_run_srk_phase", return_value=srk):
            with self.assertRaises(RuntimeError):
                run_health_test(
                    client=_FakeClient(), lamp=_lamp("triode"), cfg=_cfg(),
                    calibration=CalibrationData(), lamp_id="L", name="n",
                    reference_mode="datasheet", reference=None,
                    emission_enabled=False, warmup_s=0,
                    measurement_plan={"srk": {"delta_ua": 25, "delta_ug1": 0.84,
                                              "points": 5, "repeats": 3}},
                    stop=lambda: True)


if __name__ == "__main__":
    unittest.main()
