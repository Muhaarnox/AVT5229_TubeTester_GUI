"""Unit tests for ``lm19/health_planning.py``.

Exercises ``compute_planned_steps`` directly — pure function, no Qt
required.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lm19.config import LampConfig, LampRange
from lm19.health_planning import compute_planned_steps
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _make_lamp(
    *,
    tube_type: str = "TestTube",
    topology: str = TOPOLOGY_PENTODE,
    ua: float = 250.0, ug1: float = -8.0, ug2: float = 200.0,
    uh: float = 6.3, ih: float = 0.0,
) -> LampConfig:
    """Build a LampConfig for tests with sensible defaults."""
    return LampConfig(
        tube_type=tube_type,
        socket="noval",
        anodes=1,
        warmup_s=10,
        topology=topology,
        uh=uh, ih=ih,
        ug1=ug1, ua=ua, ia=50.0, ug2=ug2, ig2=4.0,
        s=10.0, r=20.0, k=200.0,
        ranges={
            "ua": LampRange(0, 300, 10),
            "ug1": LampRange(-30, 0, 0.5),
            "ug2": LampRange(0, 300, 10),
        },
        limits={"ua_max": 300, "ug1_max": 24, "ug2_max": 300,
                "uh_max": 15, "ih_max": 2.5, "ia_max": 200, "ig2_max": 20},
    )


def _basic_plan(
    *,
    ua: float = 250.0, ug1: float = -8.0, ug2: float = 200.0,
    delta_ua: float = 25.0, delta_ug1: float = 0.84, delta_ug2: float = 13.0,
    points: int = 5, repeats: int = 5,
    ug2_track_ua: bool = False, ug2_offset: float = 0.0,
    uh_ratio: float = 0.8,
) -> Dict:
    return {
        "op": {"ua": ua, "ug1": ug1, "ug2": ug2},
        "srk": {
            "delta_ua": delta_ua, "delta_ug1": delta_ug1,
            "delta_ug2": delta_ug2,
            "points": points, "repeats": repeats,
        },
        "emission": {"uh_ratio": uh_ratio},
        "ug2_track_ua": ug2_track_ua,
        "ug2_offset": ug2_offset,
    }


def _step_labels(steps: List[Dict]) -> List[str]:
    return [s["step"] for s in steps]


# ─── Triode behavior ─────────────────────────────────────────────────

class TestTriodeMode:
    """``is_triode=True`` lamps skip Sg2 steps and force ug2=0 everywhere."""

    def test_no_sg2_steps(self):
        lamp = _make_lamp(topology=TOPOLOGY_TRIODE)
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=lamp,
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "Sg2-" not in labels
        assert "Sg2+" not in labels

    def test_all_steps_have_ug2_zero(self):
        lamp = _make_lamp(topology=TOPOLOGY_TRIODE)
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=lamp,
            emission_enabled=False, uh_ratio_default=0.8,
        )
        for s in steps:
            assert s["ug2"] == 0.0, f"Step {s['step']} has ug2={s['ug2']!r}, expected 0"


# ─── Pentode behavior ────────────────────────────────────────────────

class TestPentodeIndependentMode:
    """``is_triode=False`` + ``ug2_track_ua=False`` adds Sg2- / Sg2+ steps."""

    def test_includes_sg2_steps(self):
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "Sg2-" in labels
        assert "Sg2+" in labels

    def test_sg2_steps_use_target_ug2_with_delta(self):
        plan = _basic_plan(ug2=200.0, delta_ug2=10.0)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(ug2=200.0),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        sg2_minus = next(s for s in steps if s["step"] == "Sg2-")
        sg2_plus = next(s for s in steps if s["step"] == "Sg2+")
        # Plain centred sweep (centre 200V > 10V delta, no shift)
        assert sg2_minus["ug2"] == pytest.approx(190.0)
        assert sg2_plus["ug2"] == pytest.approx(210.0)


class TestPentodeTriodeConnectedMode:
    """``ug2_track_ua=True`` makes Ug2 follow Ua + offset, no Sg2 steps."""

    def test_no_sg2_steps_in_track_mode(self):
        plan = _basic_plan(ug2_track_ua=True, ug2_offset=0.0)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "Sg2-" not in labels
        assert "Sg2+" not in labels

    def test_ug2_tracks_ua_with_offset(self):
        """For OP-step Ug2 should equal Ua + offset (clamped to ≥ 0)."""
        plan = _basic_plan(ua=200.0, ug2_track_ua=True, ug2_offset=10.0)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        op_step = steps[0]
        assert op_step["ug2"] == pytest.approx(210.0)


# ─── Ug1 sweep guard ─────────────────────────────────────────────────

class TestBuildUg1SweepGuard:
    """``_build_ug1_sweep`` must fail loud rather than emit a positive-grid
    sweep (defense-in-depth behind the UI validation)."""

    def test_rejects_zero_crossing(self):
        from lm19.health import _build_ug1_sweep
        with pytest.raises(ValueError):
            _build_ug1_sweep(-3.0, 4.0, 5)   # -3 + 4 = +1 > 0

    def test_rejects_zero_endpoint(self):
        """Sweep top exactly 0 V is rejected (>= 0), matching the UI boundary."""
        from lm19.health import _build_ug1_sweep
        with pytest.raises(ValueError):
            _build_ug1_sweep(-2.0, 2.0, 5)   # -2 + 2 = 0

    def test_ok_within_bounds(self):
        from lm19.health import _build_ug1_sweep
        sweep = _build_ug1_sweep(-8.0, 2.0, 5)   # top -6 < 0
        assert sweep
        assert all(v < 0 for v in sweep)


# ─── S sweep modes ───────────────────────────────────────────────────

class TestSSweepShape:
    """S-sweep: ``points<=3`` yields S-/S+ pair; >3 yields S1..Sn."""

    def test_two_point_sweep_when_points_5(self):
        # points=5 → n_ug1=3 → S-/S+ branch
        steps = compute_planned_steps(
            plan=_basic_plan(points=5), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "S-" in labels
        assert "S+" in labels
        assert not any(label.startswith("S") and label[1:].isdigit() for label in labels)

    def test_multi_point_sweep_when_points_7(self):
        # points=7 → n_ug1=5 → S1..S5 branch (skipping centre)
        steps = compute_planned_steps(
            plan=_basic_plan(points=7), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        # Should have indexed S labels
        s_indexed = [l for l in labels if l.startswith("S") and len(l) > 1 and l[1:].isdigit()]
        # S1/S2/S4/S5 (S3 = centre, skipped)
        assert len(s_indexed) >= 4

    def test_centre_step_skipped_in_sweep(self):
        """In multi-point sweep, the step that equals ug1_0 is dropped."""
        plan = _basic_plan(ug1=-8.0, delta_ug1=2.0, points=7)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(ug1=-8.0),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        # Bias is -8V; centre would be at ug1=-8 — should not appear among
        # S-sweep steps (S1..Sn). Filter only the S-indexed labels (not Sg2*).
        s_steps = [s for s in steps
                   if s["step"].startswith("S")
                   and s["step"][1:].isdigit()]
        for s in s_steps:
            assert abs(s["ug1"] - (-8.0)) > 1e-3, f"Step {s['step']} has ug1=-8 (centre, should be skipped)"


# ─── R sweep ─────────────────────────────────────────────────────────

class TestRSweep:
    """R-sweep always generates R-/R+ at ua0 ± delta_ua."""

    def test_includes_r_minus_and_plus(self):
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "R-" in labels
        assert "R+" in labels

    def test_centred_r_sweep(self):
        """Plain ua0=250, delta=25 → R- at 225, R+ at 275 (no shift)."""
        plan = _basic_plan(ua=250.0, delta_ua=25.0)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        r_minus = next(s for s in steps if s["step"] == "R-")
        r_plus = next(s for s in steps if s["step"] == "R+")
        assert r_minus["ua"] == pytest.approx(225.0)
        assert r_plus["ua"] == pytest.approx(275.0)

    def test_r_sweep_ug1_unchanged(self):
        """R-sweep keeps ug1 = ug1_0 (only ua moves)."""
        steps = compute_planned_steps(
            plan=_basic_plan(ug1=-7.0), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        r_minus = next(s for s in steps if s["step"] == "R-")
        r_plus = next(s for s in steps if s["step"] == "R+")
        assert r_minus["ug1"] == pytest.approx(-7.0)
        assert r_plus["ug1"] == pytest.approx(-7.0)


# ─── Emission test ───────────────────────────────────────────────────

class TestEmissionSteps:
    """``emission_enabled`` toggles two extra steps at end of list."""

    def test_no_emission_steps_when_disabled(self):
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        # Last step should be Sg2+ for pentode (no emission)
        assert steps[-1]["step"] == "Sg2+"

    def test_emission_adds_two_steps(self):
        steps_off = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        steps_on = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(),
            emission_enabled=True, uh_ratio_default=0.8,
        )
        assert len(steps_on) == len(steps_off) + 2

    def test_emission_uses_explicit_uh_ratio(self):
        """``plan['emission']['uh_ratio']`` overrides ``uh_ratio_default``."""
        plan = _basic_plan(uh_ratio=0.7)
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(uh=6.3),
            emission_enabled=True, uh_ratio_default=0.8,
        )
        em_steps = [s for s in steps[-2:]]
        assert em_steps[0]["uh"] == pytest.approx(6.3)        # 100%
        assert em_steps[1]["uh"] == pytest.approx(6.3 * 0.7)  # 70%

    def test_emission_uh_ratio_default_when_plan_missing(self):
        """No ``uh_ratio`` in plan → uses ``uh_ratio_default``."""
        plan = _basic_plan()
        plan["emission"] = {}  # remove uh_ratio
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(uh=6.3),
            emission_enabled=True, uh_ratio_default=0.85,
        )
        em_steps = steps[-2:]
        assert em_steps[1]["uh"] == pytest.approx(6.3 * 0.85)


# ─── Heater handling ─────────────────────────────────────────────────

class TestHeaterFields:
    """Voltage- vs current-driven heater controls which field is set."""

    def test_voltage_heater_sets_uh_only(self):
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(uh=6.3, ih=0.0),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        for s in steps:
            assert s["uh"] == pytest.approx(6.3)
            assert s["ih"] is None

    def test_current_heater_sets_ih_only(self):
        steps = compute_planned_steps(
            plan=_basic_plan(), lamp=_make_lamp(uh=0.0, ih=0.3),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        for s in steps:
            assert s["uh"] is None
            assert s["ih"] == pytest.approx(0.3)


# ─── OP step ─────────────────────────────────────────────────────────

class TestOpStep:
    """First step is always the operating-point reference."""

    def test_op_is_first(self):
        steps = compute_planned_steps(
            plan=_basic_plan(ua=250, ug1=-8, ug2=200), lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        op = steps[0]
        assert op["ua"] == pytest.approx(250)
        assert op["ug1"] == pytest.approx(-8)
        assert op["ug2"] == pytest.approx(200)


# ─── Plan defaults ───────────────────────────────────────────────────

class TestPlanDefaults:
    """Missing plan fields fall back to lamp config / hard-coded defaults."""

    def test_missing_op_uses_lamp_defaults(self):
        plan = {"op": {}, "srk": {}, "emission": {}}
        lamp = _make_lamp(ua=180, ug1=-5, ug2=150)
        steps = compute_planned_steps(
            plan=plan, lamp=lamp,
            emission_enabled=False, uh_ratio_default=0.8,
        )
        op = steps[0]
        assert op["ua"] == pytest.approx(180)
        assert op["ug1"] == pytest.approx(-5)
        assert op["ug2"] == pytest.approx(150)

    def test_missing_points_default_5(self):
        """No ``points`` in srk → defaults to 5 → 2-point S-/S+ branch."""
        plan = {"op": {"ua": 250, "ug1": -8, "ug2": 200},
                "srk": {}, "emission": {}}
        steps = compute_planned_steps(
            plan=plan, lamp=_make_lamp(),
            emission_enabled=False, uh_ratio_default=0.8,
        )
        labels = _step_labels(steps)
        assert "S-" in labels and "S+" in labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
