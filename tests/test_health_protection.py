"""Tests for OP-approach Ug1 ramp + Pa/Pg2 safety protection.

Covers the soft-start path added to ``_setup_op``:
* Healthy tube: ramp reaches OP without tripping.
* Pa over-dissipation: ``HealthProtectionError(kind="pa")`` raised mid-ramp.
* Pg2 over-dissipation: ditto for ``kind="pg2"``.
* Triode (Ug2=0): Pg2 check is skipped even with high mock Ig2.
* Triode-connected pentode: both checks active.
* ``pa_max=None`` / ``pig2_max=None``: corresponding check skipped.
* safety_pct threshold respected.
* ``stop`` callback aborts the ramp without HealthProtectionError.
* On trip, Ug1 is restored to the safe-lock value.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.calibration import CalibrationData, IA_HW_SCALE  # noqa: E402
from lm19.config import LampConfig, LampRange  # noqa: E402
from lm19.health import (  # noqa: E402
    _check_pa_pg2_at_step, _ramp_ug1_to_op, _HwState,
)
from lm19.protocol import (  # noqa: E402
    decode_ih, decode_ug1, decode_uh, encode_ih, encode_uh,
)
from lm19.scan.exceptions import HealthProtectionError  # noqa: E402
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


# ── module local constants ──
_SAFE_LOCK_V = -24.0  # matches default cfg.ug1_after_stop
_RAMP_STEP_V = 1.0


class _FakeClient:
    """Minimal LM19Serial mock.

    ``ia_model(ua, ug1, ug2) -> ia_mA`` and ``ig2_model(ua, ug1, ug2) -> ig2_mA``
    let each test drive the protection logic with bespoke physics.
    """

    def __init__(self,
                 ia_model: Callable[[float, float, float], float],
                 ig2_model: Optional[Callable[[float, float, float], float]] = None):
        self.state = {
            "Ua": 0, "Ug1": 0, "Ug2": 0,
            "Uh": encode_uh(6.3), "Ih": encode_ih(0.76),
            "An": 1,
        }
        self._ia_model = ia_model
        self._ig2_model = ig2_model or (lambda ua, ug1, ug2: 0.0)

    def set_param(self, name, value):
        self.state[name] = int(value)

    def get_param(self, name, real=False):
        if name in self.state:
            return self.state[name]
        if name == "Ia":
            ua = float(self.state["Ua"])
            ug1 = decode_ug1(int(self.state["Ug1"]))
            ug2 = float(self.state["Ug2"])
            ia_ma = max(0.0, self._ia_model(ua, ug1, ug2))
            return int(round(ia_ma / IA_HW_SCALE))
        if name == "Ig2":
            ua = float(self.state["Ua"])
            ug1 = decode_ug1(int(self.state["Ug1"]))
            ug2 = float(self.state["Ug2"])
            ig2_ma = max(0.0, self._ig2_model(ua, ug1, ug2))
            return int(round(ig2_ma * 100.0))
        return 0

    def is_open(self):
        return True


def _cfg(**overrides):
    base = dict(
        health_op_ramp_enabled=True,
        health_op_ug1_ramp_step_v=_RAMP_STEP_V,
        health_emission_min_ik_ratio=0.30, health_emission_mode_default="single",
        health_emission_uh_sweep_steps=5, health_emission_uh_sweep_min_ratio=0.70,
        health_emission_knee_drop_pct=10.0, health_emission_sweep_max_total_s=600.0,
        health_bias_servo_enabled_default=False, health_bias_servo_tol_ma=0.5,
        health_bias_servo_max_shift_v=3.0, health_bias_servo_max_iter=8,
        health_bias_servo_step_v=0.5, health_bias_servo_pa_ceiling_pct=90.0,
        health_bias_servo_shift_margin=2.0,
        health_bias_servo_ug1_floor_v=0.1,
        health_pa_safety_pct=120.0,
        health_pig2_safety_pct=120.0,
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
        health_ia_samples=3,
        health_ia_sample_delay_ms=0,
        ug1_after_stop=_SAFE_LOCK_V,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lamp(*, topology=TOPOLOGY_PENTODE, pa_max=12.0, pig2_max=2.0):
    """Build an EL84-ish LampConfig with overridable limits."""
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
        ranges={
            "ua": LampRange(0, 250, 10),
            "ug1": LampRange(-20, 0, 1),
            "ug2": LampRange(0, 250, 10),
        },
        limits={},
        pa_max=pa_max,
        pig2_max=pig2_max,
    )


def _hw_at_safe_lock():
    """HwState pre-positioned at the safe-lock Ug1 (as _setup_op does)."""
    return _HwState(
        prev_ua=250.0, prev_ug1=_SAFE_LOCK_V, prev_ug2=250.0, prev_uh=6.3,
    )


def _set_hardware_at_op(client: _FakeClient, ua: float, ug2: float, ug1: float) -> None:
    """Place fake-client hardware state at the ramp entry point."""
    from lm19.protocol import encode_ug1
    client.state["Ua"] = int(round(ua))
    client.state["Ug2"] = int(round(ug2))
    client.state["Ug1"] = encode_ug1(ug1)


def _ramp(client, cfg, lamp, *, target_ug1, ug2_mode=TOPOLOGY_PENTODE,
          progress=None, stop=None, calibration=None):
    """Invoke _ramp_ug1_to_op with sensible defaults for tests."""
    hw = _hw_at_safe_lock()
    _set_hardware_at_op(client, ua=lamp.ua, ug2=lamp.ug2, ug1=_SAFE_LOCK_V)
    return _ramp_ug1_to_op(
        client, cfg, calibration or CalibrationData(), lamp, hw,
        target_ug1=target_ug1,
        lamp_id="L1",
        ug2_mode=ug2_mode,
        progress=progress,
        stop=stop,
    )


class TestCheckPaPg2AtStep:
    """Direct unit tests for the protection predicate."""

    def test_pa_within_limit_passes(self):
        cfg = _cfg()
        lamp = _lamp()
        pt = {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
              "ia": 50.0, "ig2": 5.0}  # Pa = 12.5 W, limit 14.4 W
        _check_pa_pg2_at_step(
            pt, lamp, cfg, step_idx=10, total_steps=17,
            start_ug1=_SAFE_LOCK_V, target_ug1=-7.0,
            lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
        )  # should not raise

    def test_pa_exceeds_limit_raises(self):
        cfg = _cfg()
        lamp = _lamp(pa_max=12.0)
        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 250.0,
              "ia": 60.0, "ig2": 5.0}  # Pa = 15.0 W > 14.4 W
        with pytest.raises(HealthProtectionError) as excinfo:
            _check_pa_pg2_at_step(
                pt, lamp, cfg, step_idx=10, total_steps=17,
                start_ug1=_SAFE_LOCK_V, target_ug1=-5.0,
                lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
            )
        payload = excinfo.value.payload
        assert payload.kind == "pa"
        assert payload.measured_w == pytest.approx(15.0, rel=1e-3)
        assert payload.limit_w == pytest.approx(14.4, rel=1e-3)
        assert payload.datasheet_max_w == 12.0
        assert payload.safety_pct == 120.0
        assert payload.step_idx == 10
        assert payload.total_steps == 17
        assert payload.tube_type == "EL84"
        assert payload.lamp_id == "L1"
        assert payload.ug2_mode == TOPOLOGY_PENTODE

    def test_pa_exact_120pct_threshold(self):
        """Trip at >120% × Pa_max=12 (i.e. > 14.4 W); 14.4 itself passes."""
        cfg = _cfg(health_pa_safety_pct=120.0)
        lamp = _lamp(pa_max=12.0)
        # exactly 14.4 W → does NOT trip (limit is strict ">", not ">=")
        pt_at_limit = {"ua": 240.0, "ug1": -7.0, "ug2": 250.0,
                       "ia": 60.0, "ig2": 5.0}
        _check_pa_pg2_at_step(
            pt_at_limit, lamp, cfg, step_idx=1, total_steps=2,
            start_ug1=_SAFE_LOCK_V, target_ug1=-5.0,
            lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
        )
        pt_over = {"ua": 250.0, "ug1": -7.0, "ug2": 250.0,
                   "ia": 58.0, "ig2": 5.0}  # 14.5 W > 14.4 W
        with pytest.raises(HealthProtectionError):
            _check_pa_pg2_at_step(
                pt_over, lamp, cfg, step_idx=1, total_steps=2,
                start_ug1=_SAFE_LOCK_V, target_ug1=-5.0,
                lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
            )

    def test_pg2_exceeds_limit_raises(self):
        cfg = _cfg()
        lamp = _lamp(pig2_max=2.0)
        # Pg2 limit = 2.0 × 1.2 = 2.4 W. Ug2=250, Ig2=12 → 3.0 W.
        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 250.0,
              "ia": 40.0, "ig2": 12.0}  # Pa = 10 W (under), Pg2 = 3 W (over)
        with pytest.raises(HealthProtectionError) as excinfo:
            _check_pa_pg2_at_step(
                pt, lamp, cfg, step_idx=5, total_steps=17,
                start_ug1=_SAFE_LOCK_V, target_ug1=-5.0,
                lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
            )
        assert excinfo.value.payload.kind == "pg2"

    def test_pa_max_none_skips_pa_check(self):
        cfg = _cfg()
        lamp = _lamp(pa_max=None, pig2_max=None)
        pt = {"ua": 250.0, "ug1": -5.0, "ug2": 250.0,
              "ia": 200.0, "ig2": 100.0}  # huge Pa & Pg2
        _check_pa_pg2_at_step(
            pt, lamp, cfg, step_idx=1, total_steps=2,
            start_ug1=_SAFE_LOCK_V, target_ug1=-5.0,
            lamp_id="L1", ug2_mode=TOPOLOGY_PENTODE,
        )  # should not raise — limits unknown

    def test_triode_skips_pg2_check(self):
        """True triode (Ug2=0): Pg2 limit can be huge mock Ig2, no trip."""
        cfg = _cfg()
        lamp = _lamp(topology=TOPOLOGY_TRIODE, pa_max=2.5, pig2_max=None)
        pt = {"ua": 100.0, "ug1": -2.0, "ug2": 0.0,
              "ia": 20.0, "ig2": 50.0}  # Pa = 2 W (ok), Ug2=0 → no Pg2
        _check_pa_pg2_at_step(
            pt, lamp, cfg, step_idx=1, total_steps=2,
            start_ug1=_SAFE_LOCK_V, target_ug1=-2.0,
            lamp_id="L1", ug2_mode=TOPOLOGY_TRIODE,
        )  # should not raise


class TestRampUg1ToOp:
    """End-to-end ramp behaviour with mock client."""

    def test_healthy_pentode_ramp_succeeds(self):
        """Linear model: Ia ≤ 60 mA at OP → Pa ≤ 15 W → under 120% limit."""
        cfg = _cfg()
        lamp = _lamp(pa_max=20.0, pig2_max=3.0)  # generous limits
        events: list = []

        def ia(ua, ug1, ug2):
            # Closed at -24 V, ~50 mA at -7 V → 50 / (24-7) = 2.94 mA / V open
            return max(0.0, (ug1 - (-24.0)) * 2.94)

        def ig2(ua, ug1, ug2):
            return ia(ua, ug1, ug2) * 0.1

        client = _FakeClient(ia, ig2)
        _ramp(client, cfg, lamp,
              target_ug1=-7.0, ug2_mode=TOPOLOGY_PENTODE,
              progress=events.append)

        ramp_events = [e for e in events if e.get("event") == "op_ramp"]
        assert len(ramp_events) == 17  # 24-7 / step=1 → 17 steps
        # Final step lands exactly on target
        assert ramp_events[-1]["ug1"] == pytest.approx(-7.0, abs=0.05)
        # Ia and Pa monotone non-decreasing (closing → opening)
        ia_seq = [e["ia_ma"] for e in ramp_events]
        assert ia_seq[0] < ia_seq[-1]
        assert ramp_events[-1]["pa_w"] > ramp_events[0]["pa_w"]

    def test_ramp_aborts_on_pa_exceed_mid_sweep(self):
        """High-emission tube: ramp trips before reaching nominal Ug1."""
        cfg = _cfg()
        lamp = _lamp(pa_max=12.0)  # limit 14.4 W
        # Linear Ia such that Pa hits 14.4 W at Ug1 ≈ -7.6 V (mid-ramp).
        # Slope ~9.8 mA/V from cutoff at -24 V.

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 9.8)

        client = _FakeClient(ia, lambda ua, ug1, ug2: 0.0)
        with pytest.raises(HealthProtectionError) as excinfo:
            _ramp(client, cfg, lamp,
                  target_ug1=-5.0, ug2_mode=TOPOLOGY_PENTODE)
        p = excinfo.value.payload
        assert p.kind == "pa"
        # Tripped before reaching target (still negative-side of OP).
        assert p.ug1 < -5.0
        assert p.step_idx < p.total_steps

    def test_ramp_aborts_on_pg2_exceed(self):
        """Screen-current trip independent of Pa."""
        cfg = _cfg()
        lamp = _lamp(pa_max=50.0, pig2_max=1.0)  # very lax Pa, tight Pg2 (1.2 W)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 1.0)  # tiny Ia → Pa always safe

        def ig2(ua, ug1, ug2):
            # Pg2_limit = 1.0 × 1.2 = 1.2 W. Ug2=250 → trip at Ig2 > 4.8 mA.
            return max(0.0, (ug1 - (-24.0)) * 0.5)

        client = _FakeClient(ia, ig2)
        with pytest.raises(HealthProtectionError) as excinfo:
            _ramp(client, cfg, lamp,
                  target_ug1=-5.0, ug2_mode=TOPOLOGY_PENTODE)
        assert excinfo.value.payload.kind == "pg2"

    def test_ramp_for_triode(self):
        """True triode mode: Pg2 not evaluated even with mock Ig2 set."""
        cfg = _cfg()
        lamp = _lamp(topology=TOPOLOGY_TRIODE, pa_max=5.0, pig2_max=None)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 1.0)

        client = _FakeClient(ia, lambda ua, ug1, ug2: 999.0)  # absurd Ig2, must be ignored
        _set_hardware_at_op(client, ua=200.0, ug2=0.0, ug1=_SAFE_LOCK_V)
        hw = _HwState(prev_ua=200.0, prev_ug1=_SAFE_LOCK_V,
                      prev_ug2=0.0, prev_uh=6.3)
        events: list = []
        _ramp_ug1_to_op(
            client, cfg, CalibrationData(), lamp, hw,
            target_ug1=-2.0, lamp_id="L1", ug2_mode=TOPOLOGY_TRIODE,
            progress=events.append, stop=None,
        )
        ramp = [e for e in events if e.get("event") == "op_ramp"]
        # ig2_ma reported as None for triode
        assert all(e.get("ig2_ma") is None for e in ramp)
        assert all(e.get("pg2_w") is None for e in ramp)

    def test_ramp_for_triode_connected_pentode(self):
        """Pentode in triode-connection: Ug2 ≈ Ua → both checks run."""
        cfg = _cfg()
        lamp = _lamp(pa_max=12.0, pig2_max=2.0)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 1.5)

        def ig2(ua, ug1, ug2):
            return ia(ua, ug1, ug2) * 0.1

        client = _FakeClient(ia, ig2)
        _ramp(client, cfg, lamp,
              target_ug1=-7.0, ug2_mode=TOPOLOGY_TRIODE_CONNECTED)
        # Ig2 reported (pentode topology), test should reach OP without trip

    def test_pa_max_none_allows_full_ramp(self):
        """Without datasheet Pa_max we cannot judge; ramp must proceed."""
        cfg = _cfg()
        lamp = _lamp(pa_max=None, pig2_max=None)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 10.0)  # would trip if pa_max set

        client = _FakeClient(ia, lambda *a: 0.0)
        events: list = []
        _ramp(client, cfg, lamp,
              target_ug1=-5.0, ug2_mode=TOPOLOGY_PENTODE,
              progress=events.append)
        assert any(e.get("event") == "op_ramp" for e in events)

    def test_ramp_event_carries_calibrated_heater(self):
        """Every op_ramp step must report the measured heater.

        Twin of the uh80_stabilizing pin: a live display fed a heater-less
        event renders "Uh = 0", which reads as a dead heater. Non-unity
        READ gains make a raw decode differ from the calibrated value, so
        this also discriminates an uncalibrated emission.
        """
        cfg = _cfg()
        lamp = _lamp(pa_max=20.0, pig2_max=3.0)
        cal = CalibrationData()
        cal.set_channel("uh", "read", 1.04, 0.0)
        cal.set_channel("ih", "read", 0.97, 0.0)

        client = _FakeClient(lambda ua, ug1, ug2: max(0.0, (ug1 + 24.0) * 2.0),
                             lambda ua, ug1, ug2: 0.0)
        events: list = []
        _ramp(client, cfg, lamp, target_ug1=-7.0,
              ug2_mode=TOPOLOGY_PENTODE, progress=events.append,
              calibration=cal)

        ramp = [e for e in events if e.get("event") == "op_ramp"]
        assert ramp, "ramp must emit at least one step event"
        # The fake client's heater state is constant, so the expectation is
        # exact: decode of the device reading through the READ calibration.
        expected_uh = cal.apply_read(
            "uh", decode_uh(client.get_param("Uh", real=True)))
        expected_ih = cal.apply_read(
            "ih", decode_ih(client.get_param("Ih", real=True)))
        assert expected_uh != pytest.approx(
            decode_uh(client.get_param("Uh", real=True)), rel=1e-6), \
            "test setup degenerate: calibrated value equals the raw decode"
        for ev in ramp:
            assert ev["uh"] == pytest.approx(expected_uh, rel=1e-6)
            assert ev["ih"] == pytest.approx(expected_ih, rel=1e-6)

    def test_stop_aborts_ramp_without_protection_error(self):
        """Cancel between steps: ramp exits via RuntimeError, not protection."""
        cfg = _cfg()
        lamp = _lamp()

        def ia(ua, ug1, ug2):
            return 1.0  # tiny, no trip

        client = _FakeClient(ia, lambda *a: 0.0)
        seen = {"n": 0}

        def stop():
            seen["n"] += 1
            return seen["n"] > 3  # cancel after a few iterations

        with pytest.raises(RuntimeError) as excinfo:
            _ramp(client, cfg, lamp,
                  target_ug1=-7.0, ug2_mode=TOPOLOGY_PENTODE, stop=stop)
        # Must not be the protection variant
        assert not isinstance(excinfo.value, HealthProtectionError)
        assert "stopped" in str(excinfo.value).lower()

    def test_protection_trip_restores_ug1_to_safe_lock(self):
        """After a Pa trip the worker must leave Ug1 at the safe-lock value."""
        cfg = _cfg()
        lamp = _lamp(pa_max=12.0)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 9.8)  # trips mid-ramp

        client = _FakeClient(ia, lambda *a: 0.0)
        with pytest.raises(HealthProtectionError):
            _ramp(client, cfg, lamp,
                  target_ug1=-5.0, ug2_mode=TOPOLOGY_PENTODE)
        # After raise, Ug1 must be back at safe-lock (-24 V).
        ug1_after = decode_ug1(client.state["Ug1"])
        assert ug1_after == pytest.approx(_SAFE_LOCK_V, abs=0.1)

    def test_safety_pct_relaxed_lets_higher_pa_pass(self):
        """200% safety_pct doubles the trip threshold."""
        cfg = _cfg(health_pa_safety_pct=200.0)  # limit = 24 W for pa_max=12
        lamp = _lamp(pa_max=12.0)

        def ia(ua, ug1, ug2):
            return max(0.0, (ug1 - (-24.0)) * 5.0)  # at -7 V: 85 mA, Pa = 21 W

        client = _FakeClient(ia, lambda *a: 0.0)
        # Pa stays under 24 W → ramp completes
        _ramp(client, cfg, lamp,
              target_ug1=-7.0, ug2_mode=TOPOLOGY_PENTODE)
