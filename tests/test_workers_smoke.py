"""Smoke tests for worker happy/error paths with mocks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import workers
from lm19.config import LampConfig, LampRange
from lm19.protocol import decode_ug1, encode_ih, encode_uh
from lm19.calibration import CalibrationData, IA_HW_SCALE
from lm19.scan import ScanRange, ScanSettings
from lm19.constants import (
    TOPOLOGY_PENTODE,
)

pytestmark = [pytest.mark.smoke_workers]


class _FakeClient:
    def __init__(self, fail_on_get: bool = False):
        self.fail_on_get = fail_on_get
        self.set_calls = []

    def get_param(self, name, real=False):
        if self.fail_on_get:
            raise TimeoutError("mock timeout")
        values = {"Ua": 120, "Ug1": 250, "Ug2": 100, "Uh": 63, "Ih": 80, "Ia": 10, "Ig2": 5}
        return values.get(name, 0)

    def set_param(self, name, value):
        self.set_calls.append((name, value))

    def is_open(self):
        return True


@pytest.mark.smoke
def test_scan_worker_happy_path_with_mock_protocol(monkeypatch):
    emitted_progress = []
    emitted_finished = []

    def _fake_run_scan(client, settings, progress, stop, on_comm_error=None):
        progress({"idx": 1, "total": 1})
        assert stop() is False
        return [{"ua": 100.0, "ug1": -2.0, "ug2": 100.0, "ia": 1.0, "ig2": 0.0, "uh": 6.3, "ih": 0.3}]

    monkeypatch.setattr(workers, "run_scan", _fake_run_scan)

    settings = ScanSettings(
        ua=ScanRange(100, 100, 10),
        ug1=ScanRange(-2, -2, 1),
        ug2=ScanRange(100, 100, 10),
        uh=6.3,
        ih=0.3,
    )
    worker = workers.ScanWorker(_FakeClient(), settings)
    worker.progress.connect(lambda payload: emitted_progress.append(payload))
    worker.finished.connect(lambda payload: emitted_finished.append(payload))

    worker._execute()

    assert len(emitted_progress) == 1
    assert len(emitted_finished) == 1
    assert emitted_finished[0][0]["ua"] == 100.0


@pytest.mark.smoke
def test_checkcom_worker_timeout_or_device_error_handled_gracefully():
    errors = []
    worker = workers.CheckComWorker(_FakeClient(fail_on_get=True))
    worker.failed.connect(lambda msg: errors.append(msg))

    # run() wraps _execute() and emits failed on exception.
    worker.run()

    assert errors
    assert "timeout" in errors[0].lower()


@pytest.mark.smoke
def test_reset_worker_order_smoke():
    client = _FakeClient()
    worker = workers.ResetWorker(client, ug1_value=-3.0, order=["Ug2", "Ug1", "Ua", "Uh"])

    worker._execute()

    names = [name for name, _ in client.set_calls]
    assert names == ["Ug2", "Ug1", "Ua", "Uh"]


@pytest.mark.smoke
def test_reset_worker_zeros_ih_when_resetting_heater():
    """Default reset must zero Ih too — a current-driven heater (Uh=0, Ih>0)
    is otherwise left powered after 'Reset All'."""
    client = _FakeClient()
    worker = workers.ResetWorker(client, ug1_value=-3.0, reset_heater=True)

    worker._execute()

    assert ("Ih", 0) in client.set_calls
    assert ("Uh", 0) in client.set_calls


@pytest.mark.smoke
def test_reset_worker_keeps_ih_when_heater_kept():
    """With reset_heater=False neither heater channel is touched."""
    client = _FakeClient()
    worker = workers.ResetWorker(client, ug1_value=-3.0, reset_heater=False)

    worker._execute()

    names = [name for name, _ in client.set_calls]
    assert "Ih" not in names
    assert "Uh" not in names


class _BrokenClient(_FakeClient):
    """set_param raises for the given params (or all when params=None)."""

    def __init__(self, exc: Exception, params=None):
        super().__init__()
        self._exc = exc
        self._fail_params = params

    def set_param(self, name, value):
        if self._fail_params is None or name in self._fail_params:
            raise self._exc
        super().set_param(name, value)


@pytest.mark.smoke
def test_reset_worker_all_writes_failed_emits_failed_not_finished():
    """ML-125: a reset where NO output was actually zeroed must surface as
    `failed` — the old code swallowed per-param SerialException and emitted
    `finished`, so the UI showed a successful reset with HV still on the
    tube."""
    from serial import SerialException
    client = _BrokenClient(SerialException("port broken"))
    worker = workers.ResetWorker(client, ug1_value=-3.0)
    finished, failed = [], []
    worker.finished.connect(lambda: finished.append(True))
    worker.failed.connect(failed.append)

    worker._execute()

    assert not finished, "finished emitted although nothing was zeroed"
    assert failed and "Ug2" in failed[0] and "Ua" in failed[0]


@pytest.mark.smoke
def test_reset_worker_continues_past_runtime_error():
    """ML-124: the closed-port preflight raises RuntimeError (not
    SerialException); the best-effort reset must keep zeroing the remaining
    outputs instead of dying on the first one — and still report failure."""
    client = _BrokenClient(RuntimeError("Serial port is not open"),
                           params={"Ug2"})
    worker = workers.ResetWorker(client, ug1_value=-3.0)
    failed = []
    worker.failed.connect(failed.append)

    worker._execute()

    names = [name for name, _ in client.set_calls]
    assert names == ["Ug1", "Ua", "Uh", "Ih"], \
        "remaining outputs were not zeroed after the Ug2 write failed"
    assert failed and "Ug2" in failed[0]


@pytest.mark.smoke
def test_reset_worker_clean_run_emits_finished_only():
    client = _FakeClient()
    worker = workers.ResetWorker(client, ug1_value=-3.0)
    finished, failed = [], []
    worker.finished.connect(lambda: finished.append(True))
    worker.failed.connect(failed.append)

    worker._execute()

    assert finished and not failed


@pytest.mark.smoke
def test_param_poller_emits_snapshot_smoke():
    updates = []
    worker = workers.ParamPoller(_FakeClient(), interval_ms=0)
    worker.updated.connect(lambda payload: (updates.append(payload), worker.stop()))

    worker._execute()

    assert len(updates) == 1
    assert {"ua", "ug1", "ug2", "uh", "ih", "ia", "ig2", "an"}.issubset(updates[0].keys())


@pytest.mark.smoke
def test_preheat_worker_ramp_and_finish_smoke(monkeypatch):
    finished = []
    progress = []
    client = _FakeClient()
    worker = workers.PreheatWorker(client, target_uh=6.3, target_ih=0.0, warmup_s=1)
    worker.finished.connect(lambda: finished.append(True))
    worker.progress.connect(lambda uh, ih, remaining: progress.append((uh, ih, remaining)))

    monkeypatch.setattr(workers.BaseWorker, "msleep", lambda self, ms: None)
    worker._execute()

    assert finished
    assert progress
    # Ramp path writes Uh multiple times before final emit.
    assert sum(1 for name, _ in client.set_calls if name == "Uh") >= 2


# ── HealthWorker ─────────────────────────────────────────────────────


class _HealthFakeClient:
    """Client with a smooth synthetic Ia model for health tests."""

    def __init__(self):
        self.state = {
            "Ua": 0, "Ug1": 0, "Ug2": 0,
            "Uh": encode_uh(6.3), "Ih": encode_ih(0.76), "An": 1,
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
            ia_ma = max(0.0, ua * 0.04 + (ug1 + 10.0) * 6.0) * (uh / 6.3)
            return int(round(ia_ma / IA_HW_SCALE))
        if name == "Ig2":
            ia_raw = float(self.get_param("Ia", real=True))
            ia_ma = ia_raw * IA_HW_SCALE
            return int(round(max(0.0, ia_ma * 0.12) * 100.0))
        return 0

    def is_open(self):
        return True


def _health_cfg():
    return SimpleNamespace(
        health_ua_settle_per_volt_s=0.0, health_ua_settle_base_s=0.0,
        health_ug1_settle_per_volt_s=0.0, health_ug1_settle_base_s=0.0,
        health_ug2_settle_per_volt_s=0.0, health_ug2_settle_base_s=0.0,
        health_ua_tolerance_v=2.0, health_ug1_tolerance_v=0.3, health_ug2_tolerance_v=2.0,
        health_ua_retries=1, health_ug1_retries=1, health_ug2_retries=1,
        health_emission_enabled_default=False,
        health_emission_uh_ratio=0.8,
        health_emission_stable_warmup_ratio=0.01,
        health_emission_stable_min_s=1, health_emission_stable_max_s=2,
        health_emission_sample_period_s=0.01,
        health_emission_stable_window_points=3,
        health_emission_stable_slope_threshold_ma_per_s=1000.0,
        health_weight_ia=0.35, health_weight_s=0.40,
        health_weight_rh=0.10, health_weight_screen=0.0, health_weight_emission=0.15,
        health_renormalize_weights_if_metric_missing=True,
        health_verdict_strong_min=90.0, health_verdict_good_min=75.0,
        health_verdict_weak_min=55.0, health_emission_ratio_nominal=0.90, health_emission_ratio_good_min=0.70,
        health_emission_min_ik_ratio=0.30, health_emission_mode_default="single",
        health_emission_uh_sweep_steps=5, health_emission_uh_sweep_min_ratio=0.70,
        health_emission_knee_drop_pct=10.0, health_emission_sweep_max_total_s=600.0,
        health_bias_servo_enabled_default=False, health_bias_servo_tol_ma=0.5,
        health_bias_servo_max_shift_v=3.0, health_bias_servo_max_iter=8,
        health_bias_servo_step_v=0.5, health_bias_servo_pa_ceiling_pct=90.0,
        health_bias_servo_shift_margin=2.0,
        health_bias_servo_ug1_floor_v=0.1,
        health_ia_samples=3,
        health_ia_sample_delay_ms=0,
        health_delta_ug2_pct=5,
        health_op_ramp_enabled=True,
        health_op_ug1_ramp_step_v=1.0,
        health_pa_safety_pct=120.0,
        health_pig2_safety_pct=120.0,
        ug1_after_stop=-24.0,
    )


def _health_lamp():
    return LampConfig(
        tube_type="EL84", socket="B", anodes=1, warmup_s=120,
        topology=TOPOLOGY_PENTODE, uh=6.3, ih=0.76, ug1=-7.3,
        ua=250.0, ia=48.0, ug2=250.0, ig2=5.5,
        s=11.0, r=40.0, k=19.0,
        ranges={"ua": LampRange(0, 250, 10), "ug1": LampRange(-20, 0, 1),
                "ug2": LampRange(0, 250, 10)},
        limits={},
    )


@pytest.mark.smoke
def test_health_worker_happy_path():
    emitted_progress = []
    emitted_finished = []

    worker = workers.HealthWorker(
        client=_HealthFakeClient(),
        lamp=_health_lamp(),
        app_config=_health_cfg(),
        calibration=CalibrationData(),
        lamp_id="L1",
        name="smoke",
        reference_mode="datasheet",
        reference=None,
        emission_enabled=False,
        measurement_plan={"srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3}},
        warmup_s=60,
    )
    worker.progress.connect(lambda payload: emitted_progress.append(payload))
    worker.finished.connect(lambda payload: emitted_finished.append(payload))

    worker._execute()

    assert len(emitted_finished) == 1
    m = emitted_finished[0]
    assert m["tube_type"] == "EL84"
    assert m["lamp_id"] == "L1"
    assert isinstance(m["health"]["index"], float)
    assert m["health"]["verdict"] in {"Strong", "Good", "Weak", "Replace"}
    assert isinstance(m.get("srk", {}).get("s"), (int, float))
    assert len(emitted_progress) > 0


@pytest.mark.smoke
def test_health_worker_error_emits_failed():
    errors = []
    worker = workers.HealthWorker(
        client=_FakeClient(fail_on_get=True),
        lamp=_health_lamp(),
        app_config=_health_cfg(),
        calibration=CalibrationData(),
        lamp_id="L1",
        name="fail",
        reference_mode="datasheet",
        reference=None,
        emission_enabled=False,
        measurement_plan={"srk": {"delta_ua": 25, "delta_ug1": 0.84, "points": 5, "repeats": 3}},
        warmup_s=60,
    )
    worker.failed.connect(lambda msg: errors.append(msg))

    worker.run()

    assert errors
    assert len(errors) == 1


@pytest.mark.smoke
def test_param_poller_failed_once_per_streak():
    """ML-108: a persistent comm error must emit `failed` ONCE per failure
    streak, not every 800 ms (the old code flashed the COM-error status/LED
    for the whole outage), and recover cleanly on the next good read."""
    calls = {"n": 0}

    class _FlakyClient(_FakeClient):
        def get_param(self, name, real=False):
            calls["n"] += 1
            if calls["n"] <= 20:      # first ~2 polls fail...
                raise TimeoutError("mock timeout")
            return super().get_param(name, real)

    worker = workers.ParamPoller(_FlakyClient(), interval_ms=0)
    failed, updated = [], []
    worker.failed.connect(failed.append)
    worker.updated.connect(
        lambda payload: (updated.append(payload), worker.stop()))

    worker._execute()

    assert len(updated) == 1          # recovered
    assert len(failed) == 1,         f"failed emitted {len(failed)} times for one streak"


@pytest.mark.smoke
def test_param_poller_programming_error_propagates():
    """ML-108: AttributeError (refactor regression) must NOT be swallowed
    into the failed-spam path — it propagates to BaseWorker.run."""
    class _BrokenAttrClient(_FakeClient):
        def get_param(self, name, real=False):
            raise AttributeError("api drift")

    worker = workers.ParamPoller(_BrokenAttrClient(), interval_ms=0)
    with pytest.raises(AttributeError):
        worker._execute()
