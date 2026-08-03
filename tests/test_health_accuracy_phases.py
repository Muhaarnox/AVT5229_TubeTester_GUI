"""Pins for the three accuracy phases added on top of the Quick test:

* A1 bias servo — drive Ia to the reference current before measuring S;
* A2 emission sensitivity — flag a reduced-heater probe run so far below
  the tube's current capability that it cannot see a depleted cathode;
* A3 deep emission — sweep Ia(Uh) and locate the space-charge knee.

Every phase writes to the hardware or changes what a later phase
measures, so the call-site pins matter as much as the unit pins.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm19.calibration import CalibrationData, IA_HW_SCALE  # noqa: E402
from lm19.constants import TOPOLOGY_PENTODE  # noqa: E402
from lm19.protocol import decode_ug1, encode_ih, encode_uh  # noqa: E402
from lm19.scan.exceptions import HealthProtectionError  # noqa: E402
from lm19.health import (  # noqa: E402
    BIAS_SERVO_DISABLED,
    BIAS_SERVO_NO_REFERENCE,
    BIAS_SERVO_OK,
    BIAS_SERVO_STATUSES,
    BIAS_SERVO_UNREACHABLE,
    EMISSION_MODE_SINGLE,
    EMISSION_MODE_SWEEP,
    EMISSION_MODES,
    _HwState,
    _emission_ratio_grid,
    _emission_sensitivity,
    _find_emission_knee,
    _run_emission,
    _servo_ug1_to_ref_ia,
    run_health_test,
)
from test_health_logic import _FakeClient, _cfg, _lamp  # noqa: E402


@pytest.fixture(autouse=True)
def _instant_heater_settle(monkeypatch):
    """Zero the heater settle pace for every test in this module.

    The heater settle is paced by module constants (not config keys), so
    ``_cfg()`` cannot zero it the way it zeroes the ua/ug1/ug2 settles —
    an emission sweep then sleeps ~0.55 s per heater step for real, and
    any test running two sweeps overruns the global per-test timeout.
    The swept clients are static in ``uh``, so settle time carries no
    information here."""
    import lm19.health as _health_mod
    monkeypatch.setattr(_health_mod, "_UH_SETTLE_BASE_S", 0.0)
    monkeypatch.setattr(_health_mod, "_UH_SETTLE_PER_VOLT_S", 0.0)


def _lamp_with(topology=TOPOLOGY_PENTODE, **over):
    """A LampConfig copy with fields the shared ``_lamp`` helper leaves unset."""
    import dataclasses
    return dataclasses.replace(_lamp(topology), **over)


def _hw(client, calibration=None):
    cal = calibration or CalibrationData()
    from lm19.protocol import decode_uh
    return _HwState(
        prev_ua=float(client.get_param("Ua", real=True)),
        prev_ug1=float(decode_ug1(client.get_param("Ug1", real=True))),
        prev_ug2=float(client.get_param("Ug2", real=True)),
        prev_uh=decode_uh(client.get_param("Uh", real=True)),
    )


def _servo(client, *, ref_ia, target_ug1, ia_at_target, cfg=None, lamp=None,
           enabled=True, points=None, progress=None, stop=None):
    cfg = cfg or _cfg()
    lamp = lamp or _lamp(TOPOLOGY_PENTODE)
    return _servo_ug1_to_ref_ia(
        client, cfg, CalibrationData(), lamp, _hw(client),
        ref_ia=ref_ia, target_ug1=target_ug1, ia_at_target=ia_at_target,
        point_at_target={"ia": ia_at_target, "ug1": target_ug1, "ua": 250.0},
        ug2_mode=TOPOLOGY_PENTODE, lamp_id="L1", enabled=enabled,
        measurement_points=points if points is not None else [],
        progress=progress, stop=stop,
    )


def _ia_of(client, ua, ug1, ug2=250.0):
    """Anode current the fake tube produces at a given bias (mA)."""
    client.state["Ua"] = int(ua)
    from lm19.protocol import encode_ug1
    client.state["Ug1"] = encode_ug1(ug1)
    return client.get_param("Ia", real=True) * IA_HW_SCALE


# ===========================================================================
# A1 — bias servo
# ===========================================================================


class TestBiasServoConvergence:
    def _prepared(self):
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        return client

    def test_converges_to_reference_current(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        ref_ia = ia_at_target + 6.0  # needs a less negative bias
        res = _servo(client, ref_ia=ref_ia, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert res["ia"] == pytest.approx(ref_ia, abs=1.0)
        assert res["bias_shift_v"] > 0.0  # less negative

    def test_converges_downward_when_current_is_high(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        ref_ia = ia_at_target - 6.0  # needs a more negative bias
        res = _servo(client, ref_ia=ref_ia, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert res["ia"] == pytest.approx(ref_ia, abs=1.0)
        assert res["bias_shift_v"] < 0.0  # more negative

    def test_already_on_reference_writes_nothing(self):
        # Negative space: no hardware write at all when the OP already
        # sits on the reference current.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        client.set_param = MagicMock(side_effect=client.set_param)
        res = _servo(client, ref_ia=ia_at_target, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert res["bias_shift_v"] == pytest.approx(0.0)
        assert client.set_param.call_count == 0

    def test_disabled_writes_nothing(self):
        client = self._prepared()
        client.set_param = MagicMock(side_effect=client.set_param)
        res = _servo(client, ref_ia=999.0, target_ug1=-7.0, ia_at_target=1.0,
                     enabled=False)
        assert res["status"] == BIAS_SERVO_DISABLED
        assert client.set_param.call_count == 0

    @pytest.mark.parametrize("ref", [None, 0.0, -3.0])
    def test_unusable_reference_writes_nothing(self, ref):
        client = self._prepared()
        client.set_param = MagicMock(side_effect=client.set_param)
        res = _servo(client, ref_ia=ref, target_ug1=-7.0, ia_at_target=20.0)
        assert res["status"] == BIAS_SERVO_NO_REFERENCE
        assert client.set_param.call_count == 0

    def test_unreachable_restores_plan_bias(self):
        # The reference current is far outside the allowed excursion. The
        # servo must report it AND put the bias back where the plan said,
        # otherwise SRK would silently run at the excursion edge.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 500.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     cfg=_cfg(health_bias_servo_max_shift_v=1.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(target_ug1, abs=0.3)

    def test_bracket_is_anchored_at_the_plan_bias(self):
        # The root can only lie between the plan bias and the excursion
        # edge; widening the bracket to the whole allowed range wastes
        # iterations and, on a tight budget, fails to converge at all.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     cfg=_cfg(health_bias_servo_max_iter=5,
                              health_bias_servo_tol_ma=0.5,
                              health_bias_servo_max_shift_v=3.0))
        assert res["status"] == BIAS_SERVO_OK

    def test_no_headroom_is_unreachable(self):
        client = self._prepared()
        res = _servo(client, ref_ia=500.0, target_ug1=0.0, ia_at_target=1.0)
        assert res["status"] == BIAS_SERVO_UNREACHABLE

    def test_status_always_from_registry(self):
        client = self._prepared()
        ia = _ia_of(client, 250.0, -7.0)
        seen = {
            _servo(client, ref_ia=ia, target_ug1=-7.0, ia_at_target=ia,
                   enabled=False)["status"],
            _servo(client, ref_ia=None, target_ug1=-7.0, ia_at_target=ia)["status"],
            _servo(client, ref_ia=ia, target_ug1=-7.0, ia_at_target=ia)["status"],
            _servo(client, ref_ia=ia + 500.0, target_ug1=-7.0, ia_at_target=ia,
                   cfg=_cfg(health_bias_servo_max_shift_v=1.0))["status"],
        }
        assert seen == set(BIAS_SERVO_STATUSES)

    def test_probes_are_recorded_and_reported(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points, events = [], []
        _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
               ia_at_target=ia_at_target, points=points,
               progress=events.append)
        assert points, "servo probes must land in measurement_points"
        assert all(str(p["step"]).startswith("bias_servo") for p in points)
        assert [e for e in events if e.get("event") == "bias_servo"]

    def test_protection_trip_restores_safe_lock(self):
        # Pa limit low enough that walking the bias up trips it. The tube
        # must be left cut off, not at the tripping bias.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        cfg = _cfg(health_pa_safety_pct=1.0, ug1_after_stop=-24.0)
        with pytest.raises(HealthProtectionError):
            _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                   ia_at_target=ia_at_target, cfg=cfg,
                   lamp=_lamp_with(pa_max=1.0))
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(-24.0, abs=0.3)


class TestBiasServoCallSite:
    """run_health_test must actually move the OP the servo found."""

    def _run(self, *, enabled, ref_ia):
        client = _FakeClient()
        written = []
        orig = client.set_param

        def spy(name, value):
            if name == "Ug1":
                written.append(decode_ug1(int(value)))
            orig(name, value)

        client.set_param = spy
        m = run_health_test(
            client=client, lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="servo",
            reference_mode="type",
            reference={"reference": {"ia": ref_ia, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "bias_servo": {"enabled": enabled},
            },
        )
        return m, written

    def test_disabled_by_default_status_in_result(self):
        m, _ = self._run(enabled=False, ref_ia=40.0)
        assert m["health"]["bias_servo"]["status"] == BIAS_SERVO_DISABLED
        assert m["health"]["metrics"]["bias_shift_v"] is None

    @staticmethod
    def _s_sweep_centre(measurement):
        # The S sweep is symmetric around its centre and excludes it, so
        # the mean of its points IS the centre the phase used.
        ug1s = [p["ug1"] for p in measurement["measurement_points"]
                if str(p.get("step", "")).startswith("srk_s")]
        assert ug1s, "no S-phase points recorded"
        return sum(ug1s) / len(ug1s)

    def test_enabled_moves_the_srk_centre(self):
        # With the servo on, the SRK sweep must be centred on the servo
        # bias, not on the plan bias — a unit pin on the servo alone
        # cannot prove the orchestrator threads the new Ug1 through.
        # Looking at raw Ug1 writes is NOT enough: the servo's own probes
        # are writes too, so they mask an SRK phase still using the plan.
        m_off, _ = self._run(enabled=False, ref_ia=40.0)
        m_on, _ = self._run(enabled=True, ref_ia=40.0)
        assert m_on["health"]["bias_servo"]["status"] == BIAS_SERVO_OK
        servo_ug1 = m_on["health"]["bias_servo"]["ug1"]
        assert servo_ug1 != pytest.approx(-7.0, abs=0.05)
        assert self._s_sweep_centre(m_off) == pytest.approx(-7.0, abs=0.1)
        assert self._s_sweep_centre(m_on) == pytest.approx(servo_ug1, abs=0.1)

    def test_enabled_reports_bias_shift(self):
        m, _ = self._run(enabled=True, ref_ia=40.0)
        shift = m["health"]["metrics"]["bias_shift_v"]
        assert isinstance(shift, float) and shift != 0.0
        assert m["health"]["bias_servo"]["plan_ug1"] == pytest.approx(-7.0)


# ===========================================================================
# A2 — emission sensitivity
# ===========================================================================


class TestEmissionSensitivity:
    def _lamp_with_ik_max(self, ik_max):
        return _lamp_with(ia_max_limit=ik_max)

    def test_ratio_uses_cathode_current(self):
        # Ik = Ia + Ig2, not Ia alone: on a pentode the screen carries a
        # real share of the cathode current.
        res = _emission_sensitivity({"ia": 40.0, "ig2": 10.0},
                                    self._lamp_with_ik_max(100.0), _cfg())
        assert res["ratio"] == pytest.approx(0.50)
        assert res["ik_op"] == pytest.approx(50.0)

    def test_low_below_threshold(self):
        res = _emission_sensitivity({"ia": 20.0, "ig2": 0.0},
                                    self._lamp_with_ik_max(100.0),
                                    _cfg(health_emission_min_ik_ratio=0.30))
        assert res["low"] is True

    def test_not_low_exactly_at_threshold(self):
        res = _emission_sensitivity({"ia": 30.0, "ig2": 0.0},
                                    self._lamp_with_ik_max(100.0),
                                    _cfg(health_emission_min_ik_ratio=0.30))
        assert res["low"] is False

    def test_not_low_above_threshold(self):
        res = _emission_sensitivity({"ia": 60.0, "ig2": 0.0},
                                    self._lamp_with_ik_max(100.0),
                                    _cfg(health_emission_min_ik_ratio=0.30))
        assert res["low"] is False

    def test_threshold_comes_from_config(self):
        point, lamp = {"ia": 40.0, "ig2": 0.0}, self._lamp_with_ik_max(100.0)
        assert _emission_sensitivity(point, lamp,
                                     _cfg(health_emission_min_ik_ratio=0.30))["low"] is False
        assert _emission_sensitivity(point, lamp,
                                     _cfg(health_emission_min_ik_ratio=0.60))["low"] is True

    def test_missing_ik_max_is_unknown_not_low(self):
        res = _emission_sensitivity({"ia": 1.0, "ig2": 0.0},
                                    self._lamp_with_ik_max(None), _cfg())
        assert res["ratio"] is None and res["low"] is False

    def test_flag_only_when_emission_ran(self):
        from lm19.health import _compute_scores
        lamp = _lamp_with(ia_max_limit=65.0)
        refs = {"ia": 48.0, "s": 11.0, "r": 40.0, "k": 19.0,
                "rh": None, "screen_ratio": 0.1, "emission_ratio": None}
        low_point = {"ia": 2.0, "ig2": 0.0, "uh": 6.3, "ih": 0.0}
        kw = dict(ia_op=2.0, s=11.0, r=40.0, k=19.0, point_op=low_point,
                  lamp=lamp, cfg=_cfg(), refs=refs)
        on = _compute_scores(**kw, emission_enabled=True, ia80=9.0, ia100=10.0)
        off = _compute_scores(**kw, emission_enabled=False, ia80=None, ia100=None)
        assert on["emission_low_sensitivity"] is True
        assert off["emission_low_sensitivity"] is False
        # The ratio itself is still reported either way.
        assert off["emission_sensitivity_ratio"] is not None


# ===========================================================================
# A3 — deep emission sweep
# ===========================================================================


class TestEmissionRatioGrid:
    def test_contains_endpoints_and_configured_ratio(self):
        grid = _emission_ratio_grid(
            _cfg(health_emission_uh_sweep_steps=5,
                 health_emission_uh_sweep_min_ratio=0.70), 0.83)
        assert grid[0] == pytest.approx(1.0)
        assert min(grid) == pytest.approx(0.70)
        assert any(abs(r - 0.83) < 1e-9 for r in grid), \
            "the single-point ratio must stay on the grid so emission_ratio " \
            "remains comparable between modes"

    def test_descending(self):
        grid = _emission_ratio_grid(_cfg(), 0.8)
        assert grid == sorted(grid, reverse=True)

    def test_step_count_from_config(self):
        few = _emission_ratio_grid(_cfg(health_emission_uh_sweep_steps=3), 0.8)
        many = _emission_ratio_grid(_cfg(health_emission_uh_sweep_steps=9), 0.8)
        assert len(many) > len(few)

    def test_min_ratio_from_config(self):
        grid = _emission_ratio_grid(
            _cfg(health_emission_uh_sweep_min_ratio=0.50), 0.8)
        assert min(grid) == pytest.approx(0.50)


class TestFindEmissionKnee:
    """Miram two-line knee fit — every outcome branch pinned."""

    def _curve(self, pairs):
        return [{"uh": uh, "ia": ia} for uh, ia in pairs]

    def test_flat_plateau_exact_corner(self):
        # Flat plateau at 100 mA, linear fall of 50 mA/V below 5.0 V:
        # the two-line intersection recovers the corner EXACTLY — a
        # threshold method would sit 0.2 V below it (at the 90 mA
        # crossing), outside this tolerance.
        from lm19.health import KNEE_CONF_OK
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.9, 100.0), (5.5, 100.0),
                         (4.8, 90.0), (4.4, 70.0)]),
            6.3, 10.0)
        assert res["uh_knee"] == pytest.approx(5.0, abs=0.02)
        assert res["knee_confidence"] == KNEE_CONF_OK
        assert res["below_range"] is False
        assert res["reserve_pct"] == pytest.approx(100.0 * 1.3 / 6.3, abs=0.5)

    def test_tilted_plateau_alone_is_not_a_knee(self):
        # The discriminator against the old flat threshold: a healthy
        # tube whose plateau merely sags ~0.5%/1%Uh reads 90.5 mA at 80%
        # heater — the old method called THAT the knee (reserve ~20%).
        # The tilted points lie on one straight line: no knee here.
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.83, 96.3), (5.36, 92.5),
                         (5.04, 90.0), (4.41, 85.0)]),
            6.3, 10.0)
        assert res["below_range"] is True
        assert res["uh_knee"] is None
        assert res["reserve_pct"] == pytest.approx(100.0 * (6.3 - 4.41) / 6.3)
        assert res["plateau_slope_ma_per_v"] == pytest.approx(7.9, abs=0.3)

    def test_tilted_plateau_with_real_knee(self):
        # Tilted plateau (slope 8 mA/V) breaking into a steep branch
        # (slope 50 mA/V) at 4.8 V — intersection lands on the corner
        # despite the tilt.
        from lm19.health import KNEE_CONF_OK
        plateau = [(6.3, 100.0), (5.8, 96.0), (5.3, 92.0), (4.8, 88.0)]
        steep = [(4.3, 63.0), (3.9, 43.0)]
        res = _find_emission_knee(self._curve(plateau + steep), 6.3, 10.0)
        assert res["uh_knee"] == pytest.approx(4.8, abs=0.1)
        assert res["knee_confidence"] == KNEE_CONF_OK

    def test_single_steep_point_gives_bracket_midpoint_low_conf(self):
        from lm19.health import KNEE_CONF_LOW
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.9, 100.0), (5.5, 100.0),
                         (4.4, 60.0)]),
            6.3, 10.0)
        assert res["uh_knee"] == pytest.approx(0.5 * (5.5 + 4.4), abs=1e-6)
        assert res["knee_confidence"] == KNEE_CONF_LOW
        assert res["below_range"] is False

    def test_parallel_offset_shelf_degenerates_to_midpoint(self):
        # Steep points parallel to the plateau line (offset shelf): the
        # intersection does not exist — the bracket midpoint stands in,
        # flagged low.
        from lm19.health import KNEE_CONF_LOW
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.9, 100.0), (5.5, 100.0),
                         (5.0, 70.0), (4.6, 70.0), (4.2, 70.0)]),
            6.3, 10.0)
        assert res["uh_knee"] == pytest.approx(0.5 * (5.5 + 5.0), abs=1e-6)
        assert res["knee_confidence"] == KNEE_CONF_LOW

    def test_no_plateau_means_zero_reserve(self):
        # Emission-limited at nominal already: the whole curve is one
        # steep line — knee at the top point, reserve 0, flagged low.
        from lm19.health import KNEE_CONF_LOW
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.9, 80.0), (5.5, 60.0),
                         (5.1, 40.0)]),
            6.3, 10.0)
        assert res["uh_knee"] == pytest.approx(6.3)
        assert res["reserve_pct"] == pytest.approx(0.0)
        assert res["knee_confidence"] == KNEE_CONF_LOW
        assert res["below_range"] is False

    def test_intersection_clamped_into_bracket(self):
        # A noisy plateau tilts the fit so the raw intersection escapes
        # the bracket — the estimate must stay inside the data bracket
        # and be flagged.
        from lm19.health import KNEE_CONF_LOW, KNEE_CONF_OK
        res = _find_emission_knee(
            self._curve([(6.3, 98.0), (5.9, 100.0), (5.5, 99.0),
                         (5.2, 90.5), (4.6, 88.0), (4.2, 86.0)]),
            6.3, 10.0)
        lo, hi = 5.2, 5.5  # [first steep point, last plateau point]
        assert lo - 1e-9 <= res["uh_knee"] <= hi + 1e-9
        assert res["knee_confidence"] in (KNEE_CONF_LOW, KNEE_CONF_OK)
        # This particular noisy plateau throws the raw intersection ABOVE
        # the bracket — the clamp must fire and demote confidence.
        assert res["knee_confidence"] == KNEE_CONF_LOW

    def test_drop_pct_moves_plateau_membership(self):
        # drop_pct now controls plateau membership: a point sagging 4%
        # below the line stays plateau at drop 10% (tol 5%) and turns
        # steep at drop 4% (tol 2%).
        curve = self._curve([(6.3, 100.0), (5.9, 100.0), (5.5, 100.0),
                             (5.0, 96.0), (4.4, 60.0)])
        wide = _find_emission_knee(curve, 6.3, 10.0)
        strict = _find_emission_knee(curve, 6.3, 4.0)
        # wide: only 4.4 is steep (midpoint 4.7, low conf); strict: two
        # steep points → a real intersection higher up.
        assert wide["uh_knee"] == pytest.approx(0.5 * (5.0 + 4.4), abs=1e-6)
        assert strict["uh_knee"] > wide["uh_knee"]

    def test_flat_curve_reports_below_range(self):
        res = _find_emission_knee(
            self._curve([(6.3, 100.0), (5.7, 99.0), (5.1, 98.0)]), 6.3, 10.0)
        assert res["uh_knee"] is None
        assert res["below_range"] is True
        assert res["reserve_pct"] == pytest.approx(100.0 * (6.3 - 5.1) / 6.3)
        assert res["knee_confidence"] is None

    def test_too_short_curve(self):
        assert _find_emission_knee(self._curve([(6.3, 100.0)]), 6.3, 10.0)["uh_knee"] is None

    def test_dead_tube_plateau(self):
        res = _find_emission_knee(self._curve([(6.3, 0.0), (5.1, 0.0)]), 6.3, 10.0)
        assert res["uh_knee"] is None and res["reserve_pct"] is None

    def test_confidence_registry(self):
        from lm19.health import KNEE_CONF_LOW, KNEE_CONF_OK, KNEE_CONFIDENCES
        assert KNEE_CONFIDENCES == {KNEE_CONF_OK, KNEE_CONF_LOW}


class _KneeClient(_FakeClient):
    """Fake tube whose emission collapses below a chosen heater voltage."""

    KNEE_UH = 5.4
    PLATEAU_MA = 50.0
    # Gentle enough that every swept ratio stays measurably above zero —
    # a curve that bottoms out early cannot discriminate "read Ia at the
    # configured ratio" from "read the last point".
    FALL_PER_V = 1.0

    def get_param(self, name, real=False):
        if name == "Ia":
            uh = self.state["Uh"] / 10.0
            if uh >= self.KNEE_UH:
                ia_ma = self.PLATEAU_MA
            else:
                ia_ma = max(0.0, self.PLATEAU_MA
                            * (1.0 - self.FALL_PER_V * (self.KNEE_UH - uh)))
            return int(round(ia_ma / IA_HW_SCALE))
        return super().get_param(name, real=real)


class TestRunEmissionSweep:
    def _run(self, mode, cfg=None, client=None, points=None, progress=None):
        client = client or _KneeClient()
        # min_s=0: the knee clients are static in uh, holding each swept
        # point for the production minimum only adds real sleep time.
        cfg = cfg or _cfg(health_emission_stable_min_s=0)
        hw = _hw(client)
        return client, _run_emission(
            client, cfg, CalibrationData(), hw, _lamp(TOPOLOGY_PENTODE),
            {"uh": 6.3, "ih": 0.0},
            {"uh_ratio": 0.8, "mode": mode},
            1, points if points is not None else [], progress, None,
        )

    def test_single_mode_measures_one_reduced_point(self):
        client, res = self._run(EMISSION_MODE_SINGLE)
        assert res["mode"] == EMISSION_MODE_SINGLE
        assert res["curve"] is None
        assert res["uh_knee"] is None
        assert res["ia80"] is not None

    def test_sweep_mode_builds_a_curve(self):
        client, res = self._run(EMISSION_MODE_SWEEP)
        assert res["mode"] == EMISSION_MODE_SWEEP
        assert len(res["curve"]) >= 4
        assert [p["uh"] for p in res["curve"]] == sorted(
            (p["uh"] for p in res["curve"]), reverse=True)

    def test_sweep_finds_the_knee(self):
        client, res = self._run(EMISSION_MODE_SWEEP)
        assert res["uh_knee"] == pytest.approx(_KneeClient.KNEE_UH, abs=0.35)
        assert res["reserve_pct"] == pytest.approx(
            100.0 * (6.3 - res["uh_knee"]) / 6.3, abs=1e-6)

    def test_ia80_taken_at_the_configured_ratio_not_the_last_point(self):
        # Ia at 0.8*6.3 = 5.04 V is far below the plateau, while the last
        # swept point (0.70 -> 4.41 V) is lower still: reading the last
        # point instead would corrupt emission_ratio.
        client, res = self._run(EMISSION_MODE_SWEEP)
        by_ratio = {round(p["ratio"], 4): p["ia"] for p in res["curve"]}
        assert res["ia80"] == pytest.approx(by_ratio[0.8])
        assert res["ia80"] != pytest.approx(res["curve"][-1]["ia"])
        assert res["uh80"] == pytest.approx(6.3 * 0.8, abs=0.15)

    def test_heater_restored_to_nominal(self):
        client, res = self._run(EMISSION_MODE_SWEEP)
        assert client.state["Uh"] == pytest.approx(encode_uh(6.3), abs=1)

    def test_heater_restored_even_on_stop(self):
        client = _KneeClient()
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 3

        with pytest.raises(RuntimeError):
            _run_emission(
                client, _cfg(), CalibrationData(), _hw(client), _lamp(TOPOLOGY_PENTODE),
                {"uh": 6.3, "ih": 0.0}, {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
                1, [], None, stop,
            )
        assert client.state["Uh"] == pytest.approx(encode_uh(6.3), abs=1)

    def test_sweep_emits_progress_events(self):
        events = []
        self._run(EMISSION_MODE_SWEEP, progress=events.append)
        sweep_events = [e for e in events if e.get("event") == "emission_sweep"]
        assert len(sweep_events) >= 3
        assert all(0.0 < e["ratio"] < 1.0 for e in sweep_events)

    def test_single_mode_emits_no_sweep_events(self):
        events = []
        self._run(EMISSION_MODE_SINGLE, progress=events.append)
        assert not [e for e in events if e.get("event") == "emission_sweep"]

    def test_unknown_mode_falls_back_to_single(self):
        client, res = self._run("deep-ultra")
        assert res["mode"] == EMISSION_MODE_SINGLE

    def test_mode_default_from_config_when_plan_is_silent(self):
        client = _KneeClient()
        res = _run_emission(
            client, _cfg(health_emission_mode_default=EMISSION_MODE_SWEEP,
                         health_emission_stable_min_s=0),
            CalibrationData(), _hw(client), _lamp(TOPOLOGY_PENTODE),
            {"uh": 6.3, "ih": 0.0}, {"uh_ratio": 0.8}, 1, [], None, None,
        )
        assert res["mode"] == EMISSION_MODE_SWEEP

    def test_budget_truncates_and_lowers_confidence(self):
        client, res = self._run(
            EMISSION_MODE_SWEEP,
            cfg=_cfg(health_emission_sweep_max_total_s=1e-9))
        assert res["sweep_truncated"] is True
        assert res["confidence"] == "low"

    def test_points_recorded_for_every_step(self):
        points = []
        self._run(EMISSION_MODE_SWEEP, points=points)
        steps = [p["step"] for p in points]
        assert steps.count("emission_100") == 1
        assert steps.count("emission_80") >= 3

    def test_modes_registry_covers_every_produced_mode(self):
        produced = {self._run(EMISSION_MODE_SINGLE)[1]["mode"],
                    self._run(EMISSION_MODE_SWEEP)[1]["mode"]}
        assert produced == set(EMISSION_MODES)


class _DeepKneeClient(_KneeClient):
    """Knee just below the configured grid bottom (70% of 6.3 = 4.41 V):
    only the adaptive descent can bracket it."""
    KNEE_UH = 4.2


class _AbyssKneeClient(_KneeClient):
    """Knee far below the absolute floor: the descent must stop at the
    floor and report below_range, not chase the knee into starvation."""
    KNEE_UH = 2.5

    def __init__(self):
        super().__init__()
        self.uh_commands: list = []

    def set_param(self, name, value):
        if name == "Uh":
            self.uh_commands.append(float(value) / 10.0)
        super().set_param(name, value)


class TestAdaptiveDescent:
    def _run(self, client, cfg=None, stop=None):
        cfg = cfg or _cfg(health_emission_stable_min_s=0)
        return _run_emission(
            client, cfg, CalibrationData(), _hw(client),
            _lamp(TOPOLOGY_PENTODE),
            {"uh": 6.3, "ih": 0.0}, {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
            1, [], None, stop,
        )

    def test_no_extension_when_knee_inside_the_grid(self):
        # KNEE_UH = 5.4 (86%): the fixed grid already brackets it with
        # two steep points — no ratio below the configured minimum.
        res = self._run(_KneeClient())
        assert min(p["ratio"] for p in res["curve"]) >= 0.70 - 1e-6

    def test_descends_past_the_grid_to_find_a_deep_knee(self):
        from lm19.health import KNEE_CONF_OK
        res = self._run(_DeepKneeClient())
        assert min(p["ratio"] for p in res["curve"]) < 0.70
        assert res["uh_knee"] == pytest.approx(4.2, abs=0.2)
        assert res["knee_below_range"] is False
        assert res["knee_confidence"] == KNEE_CONF_OK

    def test_descent_stops_at_two_steep_points_not_at_the_floor(self):
        # With the floor pushed far down (30%) a keep-descending mutant
        # would walk to 0.325; the real rule stops as soon as the steep
        # branch holds two points (ratios 0.625 and 0.55 for a 4.2 V
        # knee).
        res = self._run(_DeepKneeClient(),
                        cfg=_cfg(health_emission_stable_min_s=0,
                                 health_emission_uh_sweep_abs_min_ratio=0.30))
        assert min(p["ratio"] for p in res["curve"]) >= 0.55 - 1e-6

    def test_extension_steps_continue_the_grid_pace(self):
        res = self._run(_DeepKneeClient())
        below = sorted((p["ratio"] for p in res["curve"] if p["ratio"] < 0.70),
                       reverse=True)
        # Default grid pace is (1 - 0.70)/(5 - 1) = 0.075 per step.
        assert below and below[0] == pytest.approx(0.70 - 0.075, abs=1e-6)

    def test_knee_below_floor_stops_and_reports_below_range(self):
        client = _AbyssKneeClient()
        res = self._run(client)
        assert res["knee_below_range"] is True
        assert res["uh_knee"] is None
        assert res["reserve_pct"] is not None and res["reserve_pct"] > 45.0
        swept = [u for u in client.uh_commands if 0.0 < u < 6.3]
        assert min(swept) >= 0.50 * 6.3 - 0.2, \
            "the heater must never be commanded below the absolute floor"

    def test_stop_during_extension_restores_heater(self):
        client = _DeepKneeClient()
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            # Fixed grid runs 5 sweep iterations; fire on the first
            # extension iteration.
            return calls["n"] > 5

        with pytest.raises(RuntimeError):
            self._run(client, stop=stop)
        assert client.state["Uh"] == pytest.approx(encode_uh(6.3), abs=1)

    def test_no_descent_for_a_tube_with_no_plateau(self):
        # Emission-limited at nominal already (knee above nominal): the
        # whole curve is the falling branch — going deeper cannot add a
        # plateau, so the sweep must not burn floor-depth points.
        class _DyingKneeClient(_KneeClient):
            KNEE_UH = 7.0

        res = self._run(_DyingKneeClient())
        assert min(p["ratio"] for p in res["curve"]) >= 0.70 - 1e-6
        assert res["reserve_pct"] == pytest.approx(0.0)


class TestEmissionSweepResultContract:
    def test_result_carries_sweep_block(self):
        client = _KneeClient()
        m = run_health_test(
            client=client, lamp=_lamp(TOPOLOGY_PENTODE),
            cfg=_cfg(health_emission_stable_min_s=0),
            calibration=CalibrationData(), lamp_id="L1", name="sweep",
            reference_mode="datasheet", emission_enabled=True, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "emission": {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
            },
        )
        sweep = m["health"]["emission_sweep"]
        assert sweep["mode"] == EMISSION_MODE_SWEEP
        assert sweep["curve"] and sweep["uh_knee"] is not None
        assert m["health"]["metrics"]["emission_reserve_pct"] == pytest.approx(
            sweep["reserve_pct"])
        # Miram-fit fields reach the saved JSON and the UI-facing metrics.
        from lm19.health import KNEE_CONFIDENCES
        assert sweep["knee_confidence"] in KNEE_CONFIDENCES
        assert sweep["plateau_slope_ma_per_v"] is not None
        assert (m["health"]["metrics"]["emission_knee_confidence"]
                == sweep["knee_confidence"])


class TestServoEmissionSeam:
    """Seam of the two accuracy phases: with a successful bias servo the
    emission phase (incl. the Miram sweep) must run at the SERVO bias.

    Physics: the knee position depends on the current the anode DEMANDS
    from the cathode. At the servo point every tube is swept at the same
    reference demand, which is what makes knees comparable between
    differently worn tubes; a worn tube swept at the plan bias would
    under-demand its cathode and flatter its reserve."""

    def test_sweep_points_sit_at_the_servo_bias(self):
        client = _WornKorenClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        plan_ug1 = -7.3
        ref_ia = client.healthy_ia(250.0, plan_ug1, 250.0)
        m = run_health_test(
            client=client, lamp=_lamp_with(s=11.0), cfg=_cfg(
                health_emission_stable_min_s=0),
            calibration=CalibrationData(), lamp_id="L1", name="seam",
            reference_mode="type",
            reference={"reference": {"ia": ref_ia, "s": 11.0,
                                     "r": 40.0, "k": 19.0}},
            emission_enabled=True, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": plan_ug1, "ug2": 250.0,
                       "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5,
                        "repeats": 1},
                "emission": {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
                "bias_servo": {"enabled": True},
            },
        )
        servo = m["health"]["bias_servo"]
        assert servo["status"] == BIAS_SERVO_OK
        assert servo["ug1"] != pytest.approx(plan_ug1, abs=0.5)

        em_points = [p for p in m["measurement_points"]
                     if str(p.get("step", "")).startswith("emission")]
        assert em_points, "the sweep must have recorded emission points"
        for p in em_points:
            assert p["ug1"] == pytest.approx(servo["ug1"], abs=0.1), \
                "emission must be measured at the servo bias, not the plan"
        # The demand at nominal heater equals the reference current — the
        # comparability argument in one number.
        assert m["health"]["raw"]["ia100"] == pytest.approx(ref_ia, abs=1.6)


# ===========================================================================
# UI call sites
# ===========================================================================


class TestAccuracyPanelCallSites:
    def _stub(self):
        from app.health_tab import HealthTab

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
        return stub

    def _measurement(self, **over):
        metrics = {"ia_pct": 73.0, "s_pct": 90.0, "r_pct": 95.0, "k_pct": 92.0,
                   "emission_ratio": 0.62, "emission_verdict": "weakened",
                   "emission_reserve_pct": None, "emission_low_sensitivity": False,
                   "emission_sensitivity_ratio": 0.55, "bias_shift_v": None}
        metrics.update(over.pop("metrics", {}))
        health = {"index": 78.0, "verdict": "Good", "metrics": metrics,
                  "raw": {"ia_op": 35.0},
                  "bias_servo": {"status": BIAS_SERVO_DISABLED, "ug1": -7.0},
                  "emission_sweep": {"knee_below_range": False}}
        health.update(over)
        return {"health": health,
                "srk": {"s": 9.9, "r": 41.0, "k": 19.0, "sg2": None,
                        "mu_g1g2": None, "uncertainty": {}}}

    def test_bias_shift_shown_on_its_own_row(self):
        # Servo details live on a separate row; appended to the Ia
        # cell they visually merge with the S/R/K column.
        stub = self._stub()
        stub._update_result(self._measurement(
            metrics={"bias_shift_v": 0.85},
            bias_servo={"status": BIAS_SERVO_OK, "ug1": -6.15}))
        bias_text = stub.result_bias.setText.call_args[0][0]
        assert "0.85" in bias_text and "-6.15" in bias_text
        stub.result_bias.setVisible.assert_called_with(True)
        ia_text = stub.result_ia_abs.setText.call_args[0][0]
        assert "0.85" not in ia_text and "-6.15" not in ia_text

    def test_low_confidence_knee_marks_the_em_line(self):
        from i18n_setup import t
        from lm19.health import KNEE_CONF_LOW, KNEE_CONF_OK
        marker = t("health.Result_em_knee_lowconf")
        assert marker != "health.Result_em_knee_lowconf", "key must exist"
        stub = self._stub()
        stub._update_result(self._measurement(
            metrics={"emission_reserve_pct": 23.0,
                     "emission_knee_confidence": KNEE_CONF_LOW}))
        em_text = stub.result_emission.setText.call_args[0][0]
        assert marker in em_text
        # ...and a solid fit stays unmarked (twin).
        stub2 = self._stub()
        stub2._update_result(self._measurement(
            metrics={"emission_reserve_pct": 23.0,
                     "emission_knee_confidence": KNEE_CONF_OK}))
        assert marker not in stub2.result_emission.setText.call_args[0][0]

    def test_bias_row_hidden_when_servo_off(self):
        stub = self._stub()
        stub._update_result(self._measurement())
        assert stub.result_bias.setText.call_args[0][0] == ""
        stub.result_bias.setVisible.assert_called_with(False)

    def test_servo_failure_is_visible(self):
        from app.health_tab import BIAS_SERVO_STATUS_KEYS
        from i18n_setup import t
        stub = self._stub()
        stub._update_result(self._measurement(
            bias_servo={"status": BIAS_SERVO_UNREACHABLE, "ug1": -7.0}))
        text = stub.result_bias.setText.call_args[0][0]
        assert t(BIAS_SERVO_STATUS_KEYS[BIAS_SERVO_UNREACHABLE]) in text
        stub.result_bias.setVisible.assert_called_with(True)

    def test_reserve_shown_when_swept(self):
        stub = self._stub()
        stub._update_result(self._measurement(
            metrics={"emission_reserve_pct": 14.3}))
        assert "14" in stub.result_emission.setText.call_args[0][0]

    def test_low_sensitivity_marker(self):
        on, off = self._stub(), self._stub()
        on._update_result(self._measurement(
            metrics={"emission_low_sensitivity": True,
                     "emission_sensitivity_ratio": 0.12}))
        off._update_result(self._measurement())
        on_text = on.result_emission.setText.call_args[0][0]
        off_text = off.result_emission.setText.call_args[0][0]
        assert on_text != off_text
        assert "12" in on_text

    def test_status_registry_bijection(self):
        from app.health_tab import BIAS_SERVO_STATUS_KEYS
        assert set(BIAS_SERVO_STATUS_KEYS) == set(BIAS_SERVO_STATUSES)

    def test_status_keys_exist_in_every_locale(self):
        import json
        from i18n_setup import available_locales
        from app.health_tab import BIAS_SERVO_STATUS_KEYS
        from app.health_plan_builder import EMISSION_MODE_ITEMS
        keys = list(BIAS_SERVO_STATUS_KEYS.values()) + [k for _, k in EMISSION_MODE_ITEMS]
        root = Path(__file__).parent.parent / "locales"
        for loc in available_locales():
            data = json.loads((root / f"{loc}.json").read_text(encoding="utf-8"))
            for key in keys:
                section, short = key.split(".", 1)
                assert short in data.get(section, {}), f"Missing {key} in {loc}"

    def test_emission_mode_combo_covers_registry(self):
        from app.health_plan_builder import EMISSION_MODE_ITEMS
        assert {code for code, _ in EMISSION_MODE_ITEMS} == set(EMISSION_MODES)


class TestPlanCollectsAccuracyOptions:
    """The plan dict is what actually reaches run_health_test."""

    def test_plan_carries_mode_and_servo_flag(self):
        from app.health_tab import HealthTab
        stub = SimpleNamespace(
            lamp_panel=SimpleNamespace(anode=lambda: 1),
            ug2_track_radio=SimpleNamespace(isChecked=lambda: False),
            plan_ua_target=SimpleNamespace(value=lambda: 250.0),
            plan_ug1_target=SimpleNamespace(value=lambda: -7.0),
            plan_ug2_target=SimpleNamespace(value=lambda: 250.0),
            plan_delta_ua=SimpleNamespace(value=lambda: 25.0),
            plan_delta_ug1=SimpleNamespace(value=lambda: 0.84),
            plan_delta_ug2=SimpleNamespace(value=lambda: 12.0),
            plan_points=SimpleNamespace(value=lambda: 5),
            plan_repeats=SimpleNamespace(value=lambda: 3),
            plan_emission_ratio=SimpleNamespace(value=lambda: 0.8),
            plan_emission_mode=SimpleNamespace(currentData=lambda: EMISSION_MODE_SWEEP),
            plan_bias_servo=SimpleNamespace(isChecked=lambda: True),
            ug2_offset=SimpleNamespace(value=lambda: 0.0),
            tube_combo=SimpleNamespace(currentText=lambda: "EL84"),
            get_lamps=lambda: [_lamp(TOPOLOGY_PENTODE)],
        )
        plan = HealthTab._collect_measurement_plan(stub)
        assert plan["emission"]["mode"] == EMISSION_MODE_SWEEP
        assert plan["bias_servo"]["enabled"] is True


class TestBiasServoOrchestratorDefaults:
    """The parser only sees the config if the orchestrator passes it —
    a unit pin calling _parse_health_targets(.., cfg) directly cannot
    prove that call site does."""

    def _run(self, cfg, plan):
        return run_health_test(
            client=_FakeClient(), lamp=_lamp(TOPOLOGY_PENTODE), cfg=cfg,
            calibration=CalibrationData(), lamp_id="L1", name="default",
            reference_mode="type",
            reference={"reference": {"ia": 40.0, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1, measurement_plan=plan,
        )

    _PLAN_NO_SERVO = {
        "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
        "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
    }

    def test_config_default_reaches_the_run(self):
        m = self._run(_cfg(health_bias_servo_enabled_default=True),
                      self._PLAN_NO_SERVO)
        assert m["health"]["bias_servo"]["status"] != BIAS_SERVO_DISABLED

    def test_config_off_keeps_the_servo_off(self):
        m = self._run(_cfg(health_bias_servo_enabled_default=False),
                      self._PLAN_NO_SERVO)
        assert m["health"]["bias_servo"]["status"] == BIAS_SERVO_DISABLED


class TestBiasServoUnreachableRecording:
    def test_restore_probe_is_recorded(self):
        # Two probes touch the tube on the unreachable path: the
        # excursion edge and the restore. Both are real measurements at
        # real biases, so both belong in the saved point list.
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 500.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     cfg=_cfg(health_bias_servo_max_shift_v=1.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        servo_points = [p for p in points if str(p["step"]).startswith("bias_servo")]
        # 1.0 V of headroom at 0.5 V steps = two walk probes, then the
        # restore probe back at the plan bias.
        assert len(servo_points) == 3, "two walk probes and the restore probe"
        assert servo_points[-1]["ug1"] == pytest.approx(target_ug1, abs=0.3)


class _WornClient(_FakeClient):
    """Same tube model scaled down - a worn twin of _FakeClient."""

    GAIN = 0.75

    def get_param(self, name, real=False):
        if name == "Ia":
            return int(round(super().get_param(name, real=real) * self.GAIN))
        return super().get_param(name, real=real)


class TestServoMatchingCompatibility:
    """Servo runs must stay matchable: conditions carry the PLAN bias
    plus a servo flag, never the per-tube servo outcome."""

    # ref Ia reachable for BOTH the fresh tube (Ia@plan = 28 mA, shift
    # down) and the worn one (21 mA, shift up) — the shared-key pin
    # needs two OK servo runs at different biases, not one OK and one
    # unreachable. The lamp's datasheet S matches the fake tube's true
    # slope (6 mA/V), so the auto excursion estimate is honest; the
    # x0.75 worn twin then sits inside the x2 margin.
    REF_IA = 25.0

    def _run(self, client, *, servo=True):
        return run_health_test(
            client=client, lamp=_lamp_with(s=6.0), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="match",
            reference_mode="type",
            reference={"reference": {"ia": self.REF_IA, "s": 11.0,
                                     "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "bias_servo": {"enabled": servo},
            },
        )

    def test_conditions_carry_plan_bias_not_servo_outcome(self):
        m = self._run(_FakeClient())
        assert m["health"]["bias_servo"]["status"] == BIAS_SERVO_OK
        servo_ug1 = m["health"]["bias_servo"]["ug1"]
        assert servo_ug1 != pytest.approx(-7.0, abs=0.05)
        assert m["conditions"]["ug1"] == pytest.approx(-7.0)
        assert m["measurement_plan"]["op"]["ug1"] == pytest.approx(-7.0)

    def test_two_worn_differently_tubes_share_a_key(self):
        from lm19.tube_matching import _conditions_key
        m_fresh = self._run(_FakeClient())
        m_worn = self._run(_WornClient())
        assert m_fresh["health"]["bias_servo"]["status"] == BIAS_SERVO_OK
        assert m_worn["health"]["bias_servo"]["status"] == BIAS_SERVO_OK
        assert (m_fresh["health"]["bias_servo"]["ug1"]
                != pytest.approx(m_worn["health"]["bias_servo"]["ug1"], abs=0.2))
        assert _conditions_key(m_fresh) == _conditions_key(m_worn)

    def test_servo_and_fixed_bias_runs_never_share_a_key(self):
        from lm19.tube_matching import _conditions_key
        m_servo = self._run(_FakeClient(), servo=True)
        m_fixed = self._run(_FakeClient(), servo=False)
        assert _conditions_key(m_servo) != _conditions_key(m_fixed)

    def test_failed_servo_counts_as_fixed_bias(self):
        # Unreachable -> the tube was measured at the plan bias, so the
        # run IS a fixed-bias run for matching purposes.
        from lm19.tube_matching import _conditions_key
        client = _FakeClient()
        m_failed = run_health_test(
            client=client, lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="match",
            reference_mode="type",
            reference={"reference": {"ia": 500.0, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "bias_servo": {"enabled": True},
            },
        )
        assert m_failed["health"]["bias_servo"]["status"] == BIAS_SERVO_UNREACHABLE
        m_fixed = self._run(_FakeClient(), servo=False)
        assert _conditions_key(m_failed) == _conditions_key(m_fixed)

    def test_ui_conditions_tuple_matches_core_key(self):
        # build_matching_conditions output is compared against
        # _conditions_key with == : the shapes must never drift.
        from app.health_history import build_matching_conditions
        from lm19.tube_matching import _conditions_key
        m = self._run(_FakeClient())
        assert build_matching_conditions([m], "pentode") == _conditions_key(m)

    def test_real_dicts_feed_the_shared_protocol(self):
        # Producer→consumer key round-trip: _extract_record reads the
        # keys run_health_test ACTUALLY writes (raw.ia_plan_ma,
        # metrics.bias_shift_v, conditions.bias_servo). Synthetic
        # fixtures on both sides would stay green through a rename that
        # silently empties every shared-protocol pool on real data.
        from lm19.tube_matching import (
            _extract_record, match_tubes, MATCHING_PROTOCOL_SHARED)
        m_fresh = self._run(_FakeClient())
        m_worn = self._run(_WornClient())
        m_worn["lamp_id"] = "L2"

        rec = _extract_record(m_worn)
        assert rec.servo is True
        assert rec.ia_plan == pytest.approx(
            m_worn["health"]["raw"]["ia_plan_ma"])
        assert rec.bias_shift == pytest.approx(
            m_worn["health"]["metrics"]["bias_shift_v"])
        # The worn tube's plan-point current sits BELOW the reference it
        # was servoed to — the wear-honest figure survived extraction.
        assert rec.ia_plan < self.REF_IA

        result = match_tubes([m_fresh, m_worn], group_size=2,
                             protocol=MATCHING_PROTOCOL_SHARED,
                             max_iq_imbalance_pct=0.0)
        assert len(result.groups) == 1
        group = result.groups[0]
        assert group.iq_imbalance_ma == pytest.approx(
            abs(m_fresh["health"]["raw"]["ia_plan_ma"]
                - m_worn["health"]["raw"]["ia_plan_ma"]))

    def test_real_dicts_feed_the_individual_protocol(self):
        from lm19.tube_matching import (
            match_tubes, MATCHING_PROTOCOL_INDIVIDUAL)
        m_fresh = self._run(_FakeClient())
        m_worn = self._run(_WornClient())
        m_worn["lamp_id"] = "L2"
        result = match_tubes([m_fresh, m_worn], group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL,
                             bias_adjust_range_pct=50.0)
        assert len(result.groups) == 1


class TestServoNonConvergenceRestore:
    def test_bracketed_non_convergence_restores_plan_bias(self):
        # Bracket exists but the budget is too small to converge: the tube
        # must be left at the PLAN bias (same contract as the unbracketed
        # branch), and no shift reported - none was applied.
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        # ref lands strictly between two walk steps, so convergence needs
        # bisection — which the 2-probe budget does not leave room for.
        res = _servo(client, ref_ia=ia_at_target + 4.5, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     cfg=_cfg(health_bias_servo_max_iter=2,
                              health_bias_servo_tol_ma=0.1))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        assert res["bias_shift_v"] is None
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(target_ug1, abs=0.3)
        assert res["ug1"] == pytest.approx(target_ug1, abs=0.3)


class TestSweepUnityRatio:
    def test_sweep_with_ratio_one_still_reports_a_ratio(self):
        # uh_ratio=1.0 means "no reduction" - the plateau point is the
        # reference point. The sweep grid excludes 1.0, so without the
        # seed the ratio silently came back None in sweep mode only.
        client = _KneeClient()
        res = _run_emission(
            client, _cfg(), CalibrationData(), _hw(client), _lamp(TOPOLOGY_PENTODE),
            {"uh": 6.3, "ih": 0.0}, {"uh_ratio": 1.0, "mode": EMISSION_MODE_SWEEP},
            1, [], None, None,
        )
        assert res["ia80"] == pytest.approx(res["ia100"])
        assert res["uh_knee"] is not None  # sweep itself still ran


class TestProgressEventConsumers:
    """Every emitted event needs a UI consumer (project rule): these are
    call-site spies on the _on_progress branches."""

    def _stub(self):
        from app.health_tab import HealthTab

        class _Stub:
            _on_progress = HealthTab._on_progress

        stub = _Stub()
        stub.progress_label = MagicMock()
        stub.progress = MagicMock()
        stub.live_panel = MagicMock()
        return stub

    def test_bias_servo_event_updates_the_label(self):
        stub = self._stub()
        stub._on_progress({"event": "bias_servo", "iteration": 3,
                           "max_iterations": 8, "ug1": -6.5, "ia_ma": 41.0,
                           "ref_ia_ma": 40.0, "target_ug1": -7.0})
        text = stub.progress_label.setText.call_args[0][0]
        assert "3" in text and "8" in text

    def test_emission_sweep_event_updates_the_label(self):
        stub = self._stub()
        stub._on_progress({"event": "emission_sweep", "step_idx": 2,
                           "total_steps": 6, "uh": 5.8, "ratio": 0.925,
                           "ia_ma": 50.0, "ia100_ma": 50.0})
        text = stub.progress_label.setText.call_args[0][0]
        # 0.925 * 100 -> banker's rounding gives 92.
        assert "2" in text and "6" in text and "92" in text

    def test_servo_probe_emits_live_point(self):
        # The live panel is fed by live_point events - the servo probes
        # must emit one each, like every other measuring phase.
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        events = []
        _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
               ia_at_target=ia_at_target, progress=events.append)
        servo_events = [e for e in events if e.get("event") == "bias_servo"]
        live_events = [e for e in events if e.get("event") == "live_point"]
        assert servo_events
        assert len(live_events) >= len(servo_events)


class TestPlanRoundTrip:
    """measurement_plan must persist and re-apply the accuracy options."""

    def test_saved_plan_carries_mode_and_servo(self):
        client = _KneeClient()
        m = run_health_test(
            client=client, lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="rt",
            reference_mode="datasheet", emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "emission": {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
                "bias_servo": {"enabled": True},
            },
        )
        plan = m["measurement_plan"]
        # Emission disabled - the requested mode must still round-trip.
        assert plan["emission"]["mode"] == EMISSION_MODE_SWEEP
        assert plan["bias_servo"]["enabled"] is True

    def test_load_plan_restores_mode_and_servo(self):
        from app.health_tab import HealthTab
        stub = MagicMock()
        stub.plan_emission_mode.findData.return_value = 1
        HealthTab._load_plan_from_measurement(stub, {
            "measurement_plan": {
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0},
                "srk": {},
                "emission": {"uh_ratio": 0.8, "enabled": True,
                             "mode": EMISSION_MODE_SWEEP},
                "bias_servo": {"enabled": True},
            },
        })
        stub.plan_emission_mode.findData.assert_called_once_with(EMISSION_MODE_SWEEP)
        stub.plan_emission_mode.setCurrentIndex.assert_called_once_with(1)
        stub.plan_bias_servo.setChecked.assert_called_once_with(True)

    def test_plan_for_reference_carries_both(self):
        # _plan_for_reference reads only its argument; a MagicMock self
        # avoids constructing a QWidget.
        from app.health_tab import HealthTab
        out = HealthTab._plan_for_reference(MagicMock(), {
            "op": {}, "srk": {},
            "emission": {"enabled": True, "uh_ratio": 0.8,
                         "mode": EMISSION_MODE_SWEEP},
            "bias_servo": {"enabled": True},
        })
        assert out["emission"]["mode"] == EMISSION_MODE_SWEEP
        assert out["bias_servo"]["enabled"] is True


class TestServoSteppedApproach:
    """The servo must approach the reference in bounded steps and
    stop at the Pa ceiling BEFORE the trip fires — an edge-first
    probe over the full excursion drives a steep tube into the trip."""

    def _prepared(self):
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        return client

    def test_no_probe_overshoots_beyond_one_step(self):
        # ref needs a +1.0 V shift; with 0.5 V steps no probe may sit
        # further than one step past the crossing. The old edge-first
        # algorithm probed plan+3.0 V first and fails this immediately.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points)
        assert res["status"] == BIAS_SERVO_OK
        max_probe_ug1 = max(p["ug1"] for p in points if str(p["step"]).startswith("bias_servo"))
        assert max_probe_ug1 <= target_ug1 + 1.0 + 0.5 + 0.05

    def test_walk_is_monotonic_until_crossing(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
               ia_at_target=ia_at_target, points=points)
        ug1s = [p["ug1"] for p in points if str(p["step"]).startswith("bias_servo")]
        assert ug1s == sorted(ug1s), "walk must not jump around"

    def test_step_size_comes_from_config(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
               ia_at_target=ia_at_target, points=points,
               cfg=_cfg(health_bias_servo_step_v=0.25))
        first = [p["ug1"] for p in points if str(p["step"]).startswith("bias_servo")][0]
        assert first == pytest.approx(target_ug1 + 0.25, abs=0.05)

    def test_pa_ceiling_stops_before_the_trip(self):
        # pa_max=8 W, safety 120% -> trip at 9.6 W, ceiling 90% = 8.64 W.
        # Walking up crosses the ceiling (9.25 W at -5.5 V) while still
        # BELOW the trip limit: the servo must return unreachable and
        # restore the plan bias - no HealthProtectionError, no trip.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 500.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     lamp=_lamp_with(pa_max=8.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(target_ug1, abs=0.3)
        # Every probe stayed below the trip limit - protection never fired.
        trip_w = 8.0 * 120.0 / 100.0
        for p in points:
            if str(p["step"]).startswith("bias_servo"):
                assert p["ua"] * p["ia"] / 1000.0 < trip_w

    def test_ceiling_ignored_when_walking_down(self):
        # A hot tube that needs LESS current: walking down reduces Pa,
        # so the ceiling must not abort it. pa_max=5.7 W, safety 120% ->
        # trip 6.84 W, ceiling 90% = 6.156 W. The FIRST downward probe
        # (-7.5 V -> 25 mA -> 6.25 W) is still above the ceiling but
        # below the trip: a ceiling applied in both directions kills the
        # servo right there, the correct one converges.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)  # 28 mA -> 7.0 W
        res = _servo(client, ref_ia=ia_at_target - 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     lamp=_lamp_with(pa_max=5.7))
        assert res["status"] == BIAS_SERVO_OK
        assert res["bias_shift_v"] < 0.0

    def test_bisection_confined_to_the_last_step(self):
        # ref sits strictly inside the last walk step [-6.5, -6.0]; every
        # probe must stay inside it. A bisection over the whole excursion
        # (plan..limit) would probe -5.5 V first — a step-and-a-half past
        # the crossing, exactly the overshoot this design removes.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 4.2, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points)
        assert res["status"] == BIAS_SERVO_OK
        for p in points:
            if str(p["step"]).startswith("bias_servo"):
                assert -6.55 <= p["ug1"] <= -6.0 + 0.05

    def test_ug1_floor_accepts_closest_achievable_point(self):
        # With a tolerance no real tube can meet, the bisection stops at
        # the hardware floor and must ACCEPT the closest probe — the
        # closest achievable point IS the answer. Reporting unreachable
        # while a probe sits next to the reference is a false negative.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 4.4, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     cfg=_cfg(health_bias_servo_tol_ma=0.0001,
                              health_bias_servo_ug1_floor_v=0.2,
                              health_bias_servo_max_iter=12))
        servo_probes = [p for p in points if str(p["step"]).startswith("bias_servo")]
        assert len(servo_probes) < 8, "must not burn the budget below the floor"
        assert res["status"] == BIAS_SERVO_OK
        # Residual bounded by the floor quantum: S_real (6 mA/V) x 0.2 V.
        assert abs(res["ia"] - (ia_at_target + 4.4)) <= 6.0 * 0.2 + 0.1
        # The tube actually sits at the accepted point.
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(res["ug1"], abs=0.05)


class TestServoAutoExcursion:
    """The excursion limit scales with the tube's own current deficit
    (margin * |ref - Ia| / lamp.s), capped by bias_servo_max_shift_v.
    One park-wide constant was too small for a 6L6 and ~7x too wide for
    a 12AX7."""

    def _prepared(self):
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        return client

    def test_limit_scales_with_deficit_not_global(self):
        # Datasheet S inflated to 60 mA/V: a 6 mA deficit estimates a
        # 0.2 V shift, margin x2 = 0.4 V, floored to one step (0.5 V).
        # The real tube needs 1.0 V, so the servo must give up at ITS
        # limit — under the old global 3-6 V it would converge.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     lamp=_lamp_with(s=60.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        probes = [p["ug1"] for p in points if str(p["step"]).startswith("bias_servo")]
        assert max(probes) <= -6.5 + 0.05, "must not wander past its own limit"
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(target_ug1, abs=0.3)

    def test_margin_covers_sagged_transconductance(self):
        # Datasheet S=11, the fake tube's real slope is 6 mA/V — a worn
        # tube. The raw estimate (0.55 V) undershoots the needed 1.0 V;
        # the x2 margin is exactly what makes this converge.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert res["bias_shift_v"] == pytest.approx(1.0, abs=0.1)

    def test_result_carries_the_applied_limit(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        # margin 2 x (6 mA / 11 mA/V) = 1.09 V
        assert res["shift_limit_v"] == pytest.approx(2.0 * 6.0 / 11.0, abs=0.01)

    def test_global_key_still_caps_the_auto_limit(self):
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     cfg=_cfg(health_bias_servo_max_shift_v=0.6))
        assert res["shift_limit_v"] == pytest.approx(0.6)
        assert res["status"] == BIAS_SERVO_UNREACHABLE

    def test_missing_s_falls_back_to_global_ceiling(self):
        # No datasheet S: the estimate is impossible, so the global
        # ceiling applies (visible in the log). A dropped fallback
        # crashes on the zero division; a collapsed one (limit=step)
        # would return unreachable instead of converging.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     lamp=_lamp_with(s=0.0),
                     cfg=_cfg(health_bias_servo_max_shift_v=6.0))
        assert res["status"] == BIAS_SERVO_OK
        assert res["shift_limit_v"] == pytest.approx(6.0)

    def test_persisted_block_carries_the_limit(self):
        m = run_health_test(
            client=self._prepared(), lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="lim",
            reference_mode="type",
            reference={"reference": {"ia": 34.0, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "bias_servo": {"enabled": True},
            },
        )
        assert isinstance(m["health"]["bias_servo"]["shift_limit_v"], float)


class _DriftClient(_FakeClient):
    """Fake tube that heats up: every Ia read adds a cumulative drift.

    Models the field behaviour where later probes read systematically
    higher, so the LAST probe is not the CLOSEST one.
    """

    DRIFT_MA_PER_READ = 0.4

    def __init__(self):
        super().__init__()
        self._ia_reads = 0

    def get_param(self, name, real=False):
        if name == "Ia":
            base = super().get_param(name, real=real)
            self._ia_reads += 1
            return base + int(round(self.DRIFT_MA_PER_READ * self._ia_reads
                                    / IA_HW_SCALE))
        return super().get_param(name, real=real)


class TestServoBestNotLast:
    def test_accepts_the_closest_probe_not_the_last_one(self):
        # Under upward drift the first walk probe stays closest to the
        # reference while every later bisection probe reads higher. The
        # servo must move back to and accept the CLOSEST probe — taking
        # the last one silently discards the near-reference point.
        client = _DriftClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        target_ug1 = -7.0
        ia_plan = 28.0  # model value at the plan bias, drift-free
        res = _servo(client, ref_ia=ia_plan + 4.0, target_ug1=target_ug1,
                     ia_at_target=ia_plan,
                     cfg=_cfg(health_ia_samples=1,
                              health_bias_servo_tol_ma=0.05,
                              health_bias_servo_ug1_floor_v=0.13))
        assert res["status"] == BIAS_SERVO_OK
        # The first walk probe (plan + one step) was the closest — the
        # accepted bias must be that probe, not the last bisection one.
        assert res["ug1"] == pytest.approx(target_ug1 + 0.5, abs=0.06)
        # And the tube is really sitting there.
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(res["ug1"], abs=0.05)


class _KorenEl84Client(_FakeClient):
    """Fake serial backed by the production Koren EL84 model
    (config/tube_params.json, kvb=20 knee calibration) — real curvature
    and real local S (~10.8 mA/V at the datasheet OP), so the floor
    quantum S*0.1 V (~1.1 mA) emerges naturally instead of being an
    artifact of the linear stub."""

    WEAR = 1.0

    def __init__(self):
        super().__init__()
        from lm19.tube_params import lookup_tube
        from lm19.tube_sim import TubeModel
        rec = lookup_tube("EL84")
        self._model = TubeModel(name="EL84", topology=rec.topology,
                                koren=rec.koren)

    def healthy_ia(self, ua, ug1, ug2):
        return self._model.ia(ua, ug1, ug2)

    def get_param(self, name, real=False):
        if name == "Ia":
            ua = float(self.state["Ua"])
            ug1 = decode_ug1(int(self.state["Ug1"]))
            ug2 = float(self.state["Ug2"])
            ia_ma = max(0.0, self._model.ia(ua, ug1, ug2) * self.WEAR)
            return int(round(ia_ma / IA_HW_SCALE))
        if name == "Ig2":
            ia_ma = float(self.get_param("Ia", real=True)) * IA_HW_SCALE
            return int(round(ia_ma * 0.11 * 100.0))
        return super().get_param(name, real=real)


class _WornKorenClient(_KorenEl84Client):
    WEAR = 0.73  # the PSVANE field case: ~35 mA where the sheet says 48


class TestServoOnKorenModel:
    """End-to-end servo behaviour on a physically honest EL84 —
    a physics-honest complement to the linear stub."""

    def _prepared(self, cls):
        client = cls()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        return client

    def test_worn_el84_recovers_the_reference_current(self):
        # The exact field scenario: tube reads ~34 mA at the plan bias,
        # reference is the healthy ~46.9 mA. The servo must land within
        # the hardware resolution quantum (S*floor ~ 1.1 mA) of it.
        client = self._prepared(_WornKorenClient)
        target_ug1 = -7.3
        ref_ia = client.healthy_ia(250.0, target_ug1, 250.0)
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        assert ia_at_target == pytest.approx(34.2, abs=0.5)  # worn state
        res = _servo(client, ref_ia=ref_ia, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert abs(res["ia"] - ref_ia) <= 1.5  # ~S*floor + read quantum
        assert 1.0 <= res["bias_shift_v"] <= 2.5  # ~deficit/S_worn ~ 1.6 V

    def test_healthy_el84_needs_no_shift(self):
        client = self._prepared(_KorenEl84Client)
        target_ug1 = -7.3
        ref_ia = client.healthy_ia(250.0, target_ug1, 250.0)
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        client.set_param = MagicMock(side_effect=client.set_param)
        res = _servo(client, ref_ia=ref_ia, target_ug1=target_ug1,
                     ia_at_target=ia_at_target)
        assert res["status"] == BIAS_SERVO_OK
        assert res["bias_shift_v"] == pytest.approx(0.0)
        assert client.set_param.call_count == 0

    def test_strict_tolerance_still_converges_at_the_floor(self):
        # tol far below the S*floor quantum: the pre-fix code returned
        # a false "unreachable" on every EL84-class tube. Now the servo
        # accepts the closest achievable point.
        client = self._prepared(_WornKorenClient)
        target_ug1 = -7.3
        ref_ia = client.healthy_ia(250.0, target_ug1, 250.0)
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        res = _servo(client, ref_ia=ref_ia, target_ug1=target_ug1,
                     ia_at_target=ia_at_target,
                     cfg=_cfg(health_bias_servo_tol_ma=0.05))
        assert res["status"] == BIAS_SERVO_OK
        assert abs(res["ia"] - ref_ia) <= 1.5

    def test_pa_ceiling_respected_on_the_model(self):
        # Reference above what the Pa ceiling allows: the walk must stop
        # under the ceiling, never reach the trip limit, and restore.
        client = self._prepared(_KorenEl84Client)
        target_ug1 = -7.3
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=60.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     lamp=_lamp_with(pa_max=12.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        trip_w = 12.0 * 120.0 / 100.0  # _cfg pa_safety_pct = 120
        for p in points:
            if str(p["step"]).startswith("bias_servo"):
                assert p["ua"] * p["ia"] / 1000.0 < trip_w
        assert decode_ug1(client.state["Ug1"]) == pytest.approx(target_ug1, abs=0.3)


import math


class _MiramKorenClient(_KorenEl84Client):
    """Koren space-charge characteristic capped by a Richardson-style
    emission limit: Ia = min(Ia_koren, I_sat(Uh)) — a Miram curve with
    a hard knee (real knees are smoothed by work-function nonuniformity;
    the hard min() is the worst case for the interpolating knee finder).

    Physics: heater is radiation-dominated (T ~ Uh^0.6), saturation
    follows Richardson-Dushman J ~ T^2 exp(-b/T). SAT_MARGIN is the
    cathode reserve: I_sat at nominal heater over the space-charge
    demand at the OP — it shrinks as the emitter depletes, which is
    exactly what moves the knee toward nominal heater voltage.
    """

    T0_K = 1050.0
    B_K = 12800.0
    UH_NOM = 6.3
    SAT_MARGIN = 3.0
    # Space-charge plateau tilt: Ia_scl ~ (uh/uh_nom)^exp. 0 = the ideal
    # flat Child-Langmuir plateau; 0.5 reproduces the real-tube ~10% sag
    # at 80% heater (contact-potential drift with cathode temperature).
    SCL_UH_EXP = 0.0

    def __init__(self):
        super().__init__()
        self._ia_sc_op = self._model.ia(250.0, -7.3, 250.0)
        self.uh_commands: list = []

    def set_param(self, name, value):
        if name == "Uh":
            self.uh_commands.append(float(value) / 10.0)
        super().set_param(name, value)

    def sat_ma(self, uh: float) -> float:
        t = self.T0_K * (max(0.05, uh) / self.UH_NOM) ** 0.6
        j = t * t * math.exp(-self.B_K / t)
        j0 = self.T0_K ** 2 * math.exp(-self.B_K / self.T0_K)
        return self.SAT_MARGIN * self._ia_sc_op * j / j0

    def scl_ma(self, uh: float, ia_model_ma: float) -> float:
        return ia_model_ma * (max(0.05, uh) / self.UH_NOM) ** self.SCL_UH_EXP

    def get_param(self, name, real=False):
        if name == "Ia":
            ua = float(self.state["Ua"])
            ug1 = decode_ug1(int(self.state["Ug1"]))
            ug2 = float(self.state["Ug2"])
            uh = float(self.state["Uh"]) / 10.0
            scl = self.scl_ma(uh, max(0.0, self._model.ia(ua, ug1, ug2)))
            ia_ma = min(scl, self.sat_ma(uh))
            return int(round(ia_ma / IA_HW_SCALE))
        return super().get_param(name, real=real)


class _WornMiramClient(_MiramKorenClient):
    SAT_MARGIN = 1.3  # depleted emitter: barely above the OP demand


class _TiltedHealthyMiramClient(_MiramKorenClient):
    """The regression case behind the Miram rework: a tilted plateau
    (~10% sag at 80% heater) whose knee sits BELOW the sweep floor — the
    old flat threshold read the tilt itself as a knee near 80% heater.

    The toy Richardson branch e-folds every ~0.3-0.7 V, so pushing the
    corner below 50% heater takes a deliberately absurd margin; the test
    cares about the geometry (tilt without a reachable knee), not the
    margin's realism."""
    SCL_UH_EXP = 0.5
    SAT_MARGIN = 2000.0


class _TiltedWornMiramClient(_MiramKorenClient):
    """Tilted plateau AND a depleted emitter: a real knee mid-grid
    (~5.4 V for this margin), sitting on a sagging plateau."""
    SCL_UH_EXP = 0.5
    SAT_MARGIN = 3.6


class TestEmissionKneeOnModel:
    """Deep-emission sweep against the Miram-physics client — the knee
    estimate is cross-checked against the CLIENT's own saturation
    physics, not against a re-implementation of the finder."""

    def _run_sweep(self, cls):
        client = cls()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        from lm19.protocol import encode_ug1
        client.state["Ug1"] = encode_ug1(-7.3)
        res = _run_emission(
            client, _cfg(health_emission_stable_min_s=0), CalibrationData(),
            _hw(client), _lamp(TOPOLOGY_PENTODE),
            {"uh": 6.3, "ih": 0.0}, {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
            1, [], None, None,
        )
        return client, res

    @staticmethod
    def _expected_corner(client):
        # Bisect the client's own physics for the TRUE corner — the
        # heater voltage where saturation meets the (possibly tilted)
        # space-charge branch. Independent of the lm19 knee finder.
        demand = client._ia_sc_op
        lo, hi = 2.0, 6.3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if client.sat_ma(mid) < client.scl_ma(mid, demand):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def test_knee_matches_the_client_physics(self):
        client, res = self._run_sweep(_MiramKorenClient)
        assert res["uh_knee"] is not None
        expected = self._expected_corner(client)
        # Grid spacing is ~0.3-0.5 V and the saturation branch is steep;
        # the two-line intersection lands within a few tenths of a volt.
        assert res["uh_knee"] == pytest.approx(expected, abs=0.35)

    def test_tilted_healthy_plateau_is_not_a_knee(self):
        # THE regression case behind the Miram rework: ~10% sag at 80%
        # heater with a huge emission reserve. The old flat threshold
        # reported a knee near 80% (reserve ~20%); the two-line fit must
        # ride the tilted plateau to the floor and honestly say
        # "below range".
        client, res = self._run_sweep(_TiltedHealthyMiramClient)
        assert res["uh_knee"] is None
        assert res["knee_below_range"] is True
        assert res["reserve_pct"] is not None and res["reserve_pct"] > 45.0

    def test_tilted_worn_knee_matches_the_corner(self):
        client, res = self._run_sweep(_TiltedWornMiramClient)
        assert res["uh_knee"] is not None
        expected = self._expected_corner(client)
        assert res["uh_knee"] == pytest.approx(expected, abs=0.35)

    def test_descent_extends_but_never_below_the_floor(self):
        # Negative space of the adaptive descent: the healthy tilted
        # tube drives the sweep past the configured grid (below 70%)
        # yet the heater is never commanded below the absolute floor.
        client, res = self._run_sweep(_TiltedHealthyMiramClient)
        swept = [u for u in client.uh_commands if 0.0 < u < 6.3]
        assert swept, "the sweep must have commanded reduced heater"
        floor_v = 0.50 * 6.3
        assert min(swept) < 0.70 * 6.3 - 0.2, "descent below the grid"
        assert min(swept) >= floor_v - 0.2, "floor must hold"

    def test_knee_migrates_up_as_the_emitter_depletes(self):
        # The Miram wear signal (external_sources/theory, arXiv
        # 2202.08247): less reserve -> the knee moves toward nominal.
        _, healthy = self._run_sweep(_MiramKorenClient)
        _, worn = self._run_sweep(_WornMiramClient)
        assert healthy["uh_knee"] is not None
        assert worn["uh_knee"] is not None
        assert worn["uh_knee"] > healthy["uh_knee"] + 0.3

    def test_reserve_shrinks_with_wear(self):
        _, healthy = self._run_sweep(_MiramKorenClient)
        _, worn = self._run_sweep(_WornMiramClient)
        assert worn["reserve_pct"] < healthy["reserve_pct"]

    def test_single_point_ratio_tracks_saturation(self):
        # The classic Ia80/Ia100 probe on the same physics: the worn
        # cathode drops far deeper at 80% heater than the healthy one.
        _, healthy = self._run_sweep(_MiramKorenClient)
        _, worn = self._run_sweep(_WornMiramClient)
        r_healthy = healthy["ia80"] / healthy["ia100"]
        r_worn = worn["ia80"] / worn["ia100"]
        assert r_worn < r_healthy - 0.2


class TestServoStepTags:
    """Three step tags so the points table can tell them apart — with
    one shared tag the ACCEPTED operating point is indistinguishable
    from intermediate probes and the restore row."""

    def _prepared(self):
        client = _FakeClient()
        client.state["Ua"] = 250
        client.state["Ug2"] = 250
        return client

    @staticmethod
    def _tags(points):
        return [p["step"] for p in points
                if str(p["step"]).startswith("bias_servo")]

    def test_converged_run_marks_exactly_one_op_point(self):
        from lm19.health import STEP_BIAS_SERVO_OP
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points)
        tags = self._tags(points)
        assert tags.count(STEP_BIAS_SERVO_OP) == 1
        assert tags[-1] == STEP_BIAS_SERVO_OP, "accepted point is the last row"
        op_row = [p for p in points if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row["ug1"] == pytest.approx(res["ug1"], abs=0.05)

    def test_floor_accept_marks_the_best_probe_as_op(self):
        from lm19.health import STEP_BIAS_SERVO_OP
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 4.4, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     cfg=_cfg(health_bias_servo_tol_ma=0.0001,
                              health_bias_servo_ug1_floor_v=0.2,
                              health_bias_servo_max_iter=12))
        assert res["status"] == BIAS_SERVO_OK
        tags = self._tags(points)
        assert tags.count(STEP_BIAS_SERVO_OP) == 1
        op_row = [p for p in points if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row["ug1"] == pytest.approx(res["ug1"], abs=0.05)

    def test_floor_accept_tags_when_last_probe_is_the_best(self):
        # ref chosen so the bisection walks monotonically closer: the
        # LAST probe is also the closest, no re-probe happens, and the
        # in-place retag is the only thing that marks the OP row.
        from lm19.health import STEP_BIAS_SERVO_OP
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 3.9, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     cfg=_cfg(health_bias_servo_tol_ma=0.0001,
                              health_bias_servo_ug1_floor_v=0.2,
                              health_bias_servo_max_iter=12))
        assert res["status"] == BIAS_SERVO_OK
        tags = self._tags(points)
        assert tags.count(STEP_BIAS_SERVO_OP) == 1
        op_row = [p for p in points if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row["ug1"] == pytest.approx(res["ug1"], abs=0.05)
        # No re-probe: the accepted row IS the last bisection probe.
        assert tags[-1] == STEP_BIAS_SERVO_OP

    def test_failed_run_marks_restore_and_no_op(self):
        from lm19.health import STEP_BIAS_SERVO_OP, STEP_BIAS_SERVO_RESTORE
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 500.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     cfg=_cfg(health_bias_servo_max_shift_v=1.0))
        assert res["status"] == BIAS_SERVO_UNREACHABLE
        tags = self._tags(points)
        assert STEP_BIAS_SERVO_OP not in tags
        assert tags[-1] == STEP_BIAS_SERVO_RESTORE

    def test_bisect_accept_point_carries_the_bias_shift(self):
        # The bisect tol-hit is a separate accept site from the walk
        # tol-hit (textually identical blocks): ref +4 mA forces walk
        # overshoot (+3, +6 crossed) and a first-midpoint acceptance.
        from lm19.health import STEP_BIAS_SERVO_OP
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        # lamp.s must match the client's true slope (6 mA/V): with the
        # default 11 mS lamp the auto excursion limit clamps the second
        # walk step straight into tolerance and bisection never runs.
        res = _servo(client, ref_ia=ia_at_target + 4.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points,
                     lamp=_lamp_with(s=6.0))
        assert res["status"] == BIAS_SERVO_OK
        assert res["iterations"] >= 3, "must have entered the bisection"
        op_row = [p for p in points if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row["bias_shift_v"] == pytest.approx(res["bias_shift_v"])

    def test_accept_event_emitted_on_every_accept_path(self):
        # walk tol-hit / bisect tol-hit / floor-accept — each accept
        # site must announce the acceptance (the accepted probe's
        # live_point predates it, so without this event the live table
        # cannot mark the OP row until the run ends).
        scenarios = [
            ("walk", dict(ref_delta=6.0, cfg=None, lamp=None)),
            ("bisect", dict(ref_delta=4.0, cfg=None,
                            lamp=_lamp_with(s=6.0))),
            ("floor", dict(ref_delta=4.4, lamp=None,
                           cfg=_cfg(health_bias_servo_tol_ma=0.0001,
                                    health_bias_servo_ug1_floor_v=0.2,
                                    health_bias_servo_max_iter=12))),
        ]
        for name, sc in scenarios:
            client = self._prepared()
            target_ug1 = -7.0
            ia_at_target = _ia_of(client, 250.0, target_ug1)
            events = []
            res = _servo(client, ref_ia=ia_at_target + sc["ref_delta"],
                         target_ug1=target_ug1, ia_at_target=ia_at_target,
                         cfg=sc["cfg"], lamp=sc["lamp"],
                         progress=events.append)
            assert res["status"] == BIAS_SERVO_OK, name
            accepts = [e for e in events
                       if e.get("event") == "bias_servo_accept"]
            assert len(accepts) == 1, name
            assert accepts[0]["bias_shift_v"] == pytest.approx(
                res["bias_shift_v"]), name
            assert accepts[0]["ug1"] == pytest.approx(res["ug1"]), name

    def test_probe_points_carry_the_reference_current(self):
        # Structured field for the Details column: every servo point
        # carries the reference it converges to, so the per-probe
        # deviation is renderable from the point alone.
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        ref = ia_at_target + 6.0
        points = []
        _servo(client, ref_ia=ref, target_ug1=target_ug1,
               ia_at_target=ia_at_target, points=points)
        servo_pts = [p for p in points
                     if str(p["step"]).startswith("bias_servo")]
        assert servo_pts
        for p in servo_pts:
            assert p["ref_ia"] == pytest.approx(ref)

    def test_accepted_point_carries_the_bias_shift(self):
        # Both accept paths: tolerance hit on the walk/bisect...
        from lm19.health import STEP_BIAS_SERVO_OP
        client = self._prepared()
        target_ug1 = -7.0
        ia_at_target = _ia_of(client, 250.0, target_ug1)
        points = []
        res = _servo(client, ref_ia=ia_at_target + 6.0, target_ug1=target_ug1,
                     ia_at_target=ia_at_target, points=points)
        op_row = [p for p in points if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row["bias_shift_v"] == pytest.approx(res["bias_shift_v"])
        # ...and the floor-accept (closest achievable probe).
        client2 = self._prepared()
        points2 = []
        res2 = _servo(client2, ref_ia=ia_at_target + 4.4,
                      target_ug1=target_ug1,
                      ia_at_target=ia_at_target, points=points2,
                      cfg=_cfg(health_bias_servo_tol_ma=0.0001,
                               health_bias_servo_ug1_floor_v=0.2,
                               health_bias_servo_max_iter=12))
        op_row2 = [p for p in points2 if p["step"] == STEP_BIAS_SERVO_OP][0]
        assert op_row2["bias_shift_v"] == pytest.approx(res2["bias_shift_v"])

    def test_steps_table_colors_only_the_accepted_point(self):
        from app.health_tab import HealthTab
        from app.ui_theme import HEALTH_STEP_OP
        from lm19.health import (STEP_BIAS_SERVO, STEP_BIAS_SERVO_OP,
                                 STEP_BIAS_SERVO_RESTORE)
        assert HealthTab._step_color(STEP_BIAS_SERVO_OP) == HEALTH_STEP_OP
        assert HealthTab._step_color(STEP_BIAS_SERVO) is None
        assert HealthTab._step_color(STEP_BIAS_SERVO_RESTORE) is None


class TestPlanPointDeficitMetric:
    """metrics.ia_plan_pct / raw.ia_plan_ma — the wear figure ia_pct no
    longer carries once the servo moves the OP; shown in the Result
    line and in the history dbias tooltip."""

    def _run(self, *, servo, ref_ia=40.0):
        return run_health_test(
            client=_FakeClient(), lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="plan",
            reference_mode="type",
            reference={"reference": {"ia": ref_ia, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=False, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "bias_servo": {"enabled": servo},
            },
        )

    def test_deficit_measured_at_the_plan_point_not_the_servo_one(self):
        # Plan-point Ia is 28 mA, ref 40 -> 70%. Capturing it AFTER the
        # servo replacement would read ~100%.
        m = self._run(servo=True)
        assert m["health"]["bias_servo"]["status"] == BIAS_SERVO_OK
        assert m["health"]["metrics"]["ia_plan_pct"] == pytest.approx(70.0, abs=3.0)
        assert m["health"]["raw"]["ia_plan_ma"] == pytest.approx(28.0, abs=1.0)
        # And ia_pct is the at-reference figure, ~100.
        assert m["health"]["metrics"]["ia_pct"] == pytest.approx(100.0, abs=3.0)

    def test_absent_without_servo(self):
        m = self._run(servo=False)
        assert m["health"]["metrics"]["ia_plan_pct"] is None
        assert m["health"]["raw"]["ia_plan_ma"] is None


class TestPlanDeficitPanelAndTooltip:
    def _stub(self):
        from app.health_tab import HealthTab

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
        return stub

    def _measurement(self, plan_pct=73.0):
        metrics = {"ia_pct": 100.0, "s_pct": 90.0, "r_pct": 95.0, "k_pct": 92.0,
                   "emission_ratio": None, "bias_shift_v": 1.6,
                   "ia_plan_pct": plan_pct}
        return {"health": {"index": 89.0, "verdict": "Good", "metrics": metrics,
                           "raw": {"ia_op": 46.9},
                           "bias_servo": {"status": BIAS_SERVO_OK, "ug1": -5.7}},
                "srk": {"s": 9.9, "r": 41.0, "k": 19.0, "sg2": None,
                        "mu_g1g2": None, "uncertainty": {}}}

    def test_result_line_carries_the_plan_deficit(self):
        stub = self._stub()
        stub._update_result(self._measurement())
        text = stub.result_bias.setText.call_args[0][0]
        assert "73" in text

    def test_no_suffix_for_older_measurements(self):
        # Servo measurements saved before the metric existed must render
        # without the suffix, not with a bogus number.
        stub = self._stub()
        m = self._measurement()
        del m["health"]["metrics"]["ia_plan_pct"]
        stub._update_result(m)
        text = stub.result_bias.setText.call_args[0][0]
        assert "73" not in text and "None" not in text

    def test_tooltip_via_populate(self):
        from PySide6.QtWidgets import QApplication, QTableWidget
        from app.health_history import COL_DBIAS, populate_history_table
        QApplication.instance() or QApplication([])
        table = QTableWidget(0, 25)
        entry = {
            "timestamp": "t", "lamp_id": "L", "name": "",
            "conditions": {"ua": 250.0, "ug1": -7.3, "ug2": 250.0,
                           "bias_servo": True},
            "health": {"index": 89.0,
                       "metrics": {"bias_shift_v": 1.6, "ia_plan_pct": 73.0},
                       "raw": {"ia_op": 46.9, "ia_plan_ma": 35.0},
                       "bias_servo": {"status": "ok", "ug1": -5.7}},
            "srk": {},
        }
        populate_history_table(table, [entry])
        tip = table.item(0, COL_DBIAS).toolTip()
        assert "35.0" in tip and "73" in tip and "-5.70" in tip


class TestLivePointsCarryStepNames:
    """Unplanned rows (servo probes, sweep points) have no pre-created
    row in the live steps table to take a name from — every live_point
    payload must therefore carry its own step tag."""

    def test_every_live_point_is_named(self):
        events = []
        run_health_test(
            client=_KneeClient(), lamp=_lamp(TOPOLOGY_PENTODE), cfg=_cfg(),
            calibration=CalibrationData(), lamp_id="L1", name="live",
            reference_mode="type",
            reference={"reference": {"ia": 34.0, "s": 11.0, "r": 40.0, "k": 19.0}},
            emission_enabled=True, warmup_s=1,
            measurement_plan={
                "op": {"ua": 250.0, "ug1": -7.0, "ug2": 250.0, "uh": 6.3, "ih": 0.0},
                "srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 1},
                "emission": {"uh_ratio": 0.8, "mode": EMISSION_MODE_SWEEP},
                "bias_servo": {"enabled": True},
            },
            progress=events.append,
        )
        live = [e for e in events if e.get("event") == "live_point"]
        assert live, "no live points at all?"
        unnamed = [e for e in live if not str(e["point"].get("step", ""))]
        assert unnamed == [], f"{len(unnamed)} of {len(live)} live points unnamed"
        # The run covered every phase family, not just the planned ones.
        tags = {str(e["point"]["step"]) for e in live}
        assert any(t.startswith("bias_servo") for t in tags)
        assert "emission_80" in tags and "op" in tags
        assert any(t.startswith("srk_") for t in tags)

    def test_live_row_shows_the_tag_for_unplanned_points(self):
        # UI side of the same contract: an unplanned point (servo probe)
        # must render under its OWN step tag — a per-index writer that
        # keeps the PLANNED row name makes servo probes vanish from the
        # live table while they exist in the saved measurement.
        from app.health_tab import HealthTab
        from PySide6.QtWidgets import QApplication, QTableWidget
        from unittest.mock import MagicMock
        QApplication.instance() or QApplication([])

        class _Stub:
            _render_live_steps = HealthTab._render_live_steps
            _remaining_plan = HealthTab._remaining_plan
            _consumes_plan_slot = staticmethod(HealthTab._consumes_plan_slot)
            _point_row_vals = HealthTab._point_row_vals
            _point_details = HealthTab._point_details
            _point_float = HealthTab._point_float
            _fill_row = HealthTab._fill_row
            _fmt_cell = staticmethod(HealthTab._fmt_cell)
            _step_color = staticmethod(HealthTab._step_color)
            _plan_step_color = staticmethod(lambda name: None)

        stub = _Stub()
        stub.steps_table = QTableWidget(0, 11)
        stub.steps_live_btn = MagicMock()
        stub._planned_steps = [{"step": "OP", "ua": 250.0, "ug1": -7.3,
                                "ug2": 250.0}]
        stub._live_points = [{"ua": 250.0, "ug1": -6.5, "ug2": 250.0,
                              "ia": 41.0, "ig2": 4.1, "step": "bias_servo"}]
        stub._steps_view_live = True
        stub._render_live_steps()
        text = stub.steps_table.item(0, 0).text()
        assert "bias_servo" in text, f"unplanned live row unnamed: {text!r}"
        # The servo probe must NOT consume the plan slot: the planned OP
        # row is still previewed below it.
        assert stub.steps_table.rowCount() == 2
        assert "OP" in stub.steps_table.item(1, 0).text()
