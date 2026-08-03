"""Scan/SRK core failure visibility.

Pins:
- ML-108: out-of-tolerance settle → warning + ``stats`` counter →
  ``scan_summary`` keys → summary dialog extras;
- ML-109: Ia outlier re-read → warning + counter;
- ML-112: Er flag / dead heater mid-SRK → typed error (not garbage S/R/K);
- ML-090: dropped ``reset_outputs`` returns a reason code;
- ML-091: SRK uncertainty reaches the label suffix and the saved dict;
- ML-083: preheat wait deadline formula (constants exist and are sane).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lm19.scan  # noqa: F401  (init package first — srk/scan import order)
from lm19.calibration import CalibrationData
from i18n_setup import translator_for
from lm19.scan.io import _read_measurement_point, _set_param_with_settle

IDENTITY_CAL = CalibrationData()


class _FakeClient:
    """Minimal LM19Serial stand-in: scripted get_param values."""

    def __init__(self, real_values: Dict[str, object],
                 set_values: Optional[Dict[str, float]] = None):
        self._real = dict(real_values)
        self._set = dict(set_values or {})
        self.set_calls: List = []

    def set_param(self, name, value):
        self.set_calls.append((name, value))

    def get_param(self, name, real=True):
        src = self._real if real else self._set
        # Er is read with real=False by _read_measurement_point — fall
        # back to the real-values dict for scripted flags.
        v = src.get(name, self._real.get(name, 0))
        if callable(v):
            return v()
        return v

    def is_open(self):
        return True


# ── ML-108: settle out-of-tolerance ──────────────────────────────────

class TestSettleOutOfTolerance:

    def test_counter_incremented_and_logged(self, caplog):
        # Readback stuck at 100 while target is 200 → all retries fail;
        # setpoint (real=False) intact → no ProtectionError path.
        client = _FakeClient({"Ua": 100}, {"Ua": 200})
        stats: Dict[str, int] = {}
        import logging
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            actual = _set_param_with_settle(
                client, "Ua", 200.0, 0.0, 0.0, 0.0,
                tolerance=2.0, max_retries=2, stats=stats)
        assert actual == 100.0
        assert stats == {"settle_out_of_tolerance": 1}
        assert any("failed to settle" in r.message for r in caplog.records)

    def test_in_tolerance_no_count(self):
        client = _FakeClient({"Ua": 200}, {"Ua": 200})
        stats: Dict[str, int] = {}
        _set_param_with_settle(client, "Ua", 200.0, 0.0, 0.0, 0.0,
                               tolerance=2.0, max_retries=2, stats=stats)
        assert stats == {}


# ── ML-109: outlier re-read ──────────────────────────────────────────

class TestOutlierReread:

    def _client(self, ia_seq: List[int]) -> _FakeClient:
        it = iter(ia_seq)
        return _FakeClient({
            "Ia": lambda: next(it),
            "Ig2": 0, "Ua": 100, "Ug1": 512, "Ug2": 0, "Uh": 63, "Ih": 0,
            "Er": 0,
        })

    def test_outlier_counts_and_warns(self, caplog):
        import logging
        # 3 initial samples with a big spread (min > floor) → extra batch.
        client = self._client([100, 400, 100, 110, 105, 100])
        stats: Dict[str, int] = {}
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                    ia_outlier_ratio=2.0, stats=stats)
        assert stats == {"ia_outlier_rereads": 1}
        assert any("outlier" in r.message.lower() for r in caplog.records)

    def test_warning_names_the_operating_point(self, caplog):
        import logging
        # Ua drifts across the samples, so the line must carry their mean
        # (240/250/260 → 250.0), not whichever sample happened to be first.
        # All three channels differ, so a swapped argument is visible too.
        ua_it = iter([240, 250, 260, 250, 250, 250])
        client = _FakeClient({
            "Ia": lambda: next(iter_ia),
            "Ig2": 0, "Ua": lambda: next(ua_it), "Ug1": 1210, "Ug2": 175,
            "Uh": 63, "Ih": 0, "Er": 0,
        })
        iter_ia = iter([100, 400, 100, 110, 105, 100])
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                    ia_outlier_ratio=2.0)
        msg = caplog.records[0].getMessage()   # detection line, not resolution
        assert "Ua=250.0 V" in msg, msg
        assert "Ug1=-12.10 V" in msg, msg   # decode_ug1(1210)
        assert "Ug2=175.0 V" in msg, msg

    def test_negative_reread_setting_reported_as_zero(self, caplog):
        import logging
        # range(-2) is empty either way; the clamp is what keeps the line
        # from promising a negative number of extra samples.
        client = self._client([100, 400, 100])
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                    ia_outlier_ratio=2.0,
                                    ia_outlier_reread_samples=-2)
        assert "re-reading 0 extra samples" in caplog.records[0].getMessage()

    def test_resolution_line_shows_pool_and_result(self, caplog):
        import logging
        # The pool must be the full 2N set (the three re-reads included),
        # and the reported result the value that goes into the point.
        client = self._client([100, 400, 100, 110, 105, 100])
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            point = _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                            ia_outlier_ratio=2.0)
        msg = caplog.records[-1].getMessage()
        for sample in ("1.000", "1.100", "1.050"):
            assert sample in msg, (sample, msg)
        # The spike is marked as rejected, the rest is not.
        assert "[4.000]" in msg, msg
        assert "[1.100]" not in msg, msg
        assert "mean of 5 kept" in msg, msg
        # mean(1.00, 1.00, 1.10, 1.05, 1.00) = 1.03
        assert f"Ia={point['ia']:.3f} mA" in msg, msg
        assert "1.030" in msg, msg

    def test_no_resolution_line_without_outlier(self, caplog):
        import logging
        client = self._client([100, 101, 100])
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                    ia_outlier_ratio=2.0)
        assert not caplog.records

    def test_collapsed_sample_counts_and_warns(self, caplog):
        import logging
        # Contact drops out on one sample (0 mA): the ratio is undefined,
        # and the point must still reach the user as a warning + counter.
        client = self._client([1000, 0, 1000, 1000, 1000, 1000])
        stats: Dict[str, int] = {}
        with caplog.at_level(logging.WARNING, logger="lm19.scan.io"):
            _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                    ia_outlier_ratio=2.0, stats=stats)
        assert stats == {"ia_outlier_rereads": 1}
        assert any("outlier" in r.message.lower() for r in caplog.records)

    def test_stable_readings_no_count(self):
        client = self._client([100, 101, 100])
        stats: Dict[str, int] = {}
        _read_measurement_point(client, IDENTITY_CAL, ia_samples=3,
                                ia_outlier_ratio=2.0, stats=stats)
        assert stats == {}


# ── scan_summary carries the counters ────────────────────────────────

class TestSummaryKeys:

    def test_typed_dict_has_counter_fields(self):
        from lm19.scan.events import _ScanSummary
        keys = _ScanSummary.__annotations__
        assert "settle_out_of_tolerance" in keys
        assert "ia_outlier_rereads" in keys

    def test_dialog_shows_extras(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMainWindow, QMessageBox
        from app.main_window_scan import MainWindowScan

        class Host(MainWindowScan, QMainWindow):
            pass

        captured = {}
        monkeypatch.setattr(
            QMessageBox, "information",
            staticmethod(lambda parent, title, text: captured.setdefault(
                "text", text)))
        host = Host()
        host._show_scan_summary_dialog({
            "curves": [], "total_points": 10, "duration_s": 5.0,
            "settle_out_of_tolerance": 2, "ia_outlier_rereads": 3,
            "ia_unstable_points": 4,
        })
        assert "⚠" in captured["text"]
        assert "2" in captured["text"] and "3" in captured["text"]
        # Points that rejection could not clean reach the user too — a
        # counter without a UI consumer does not satisfy failure-visibility.
        assert "4" in captured["text"]
        unstable_line = translator_for("en")("msg.Scan_summary_unstable_points",
                                             count=4)
        assert unstable_line in captured["text"]
        # clean scan → no extras block
        captured.clear()
        host._show_scan_summary_dialog({
            "curves": [], "total_points": 10, "duration_s": 5.0,
            "settle_out_of_tolerance": 0, "ia_outlier_rereads": 0,
            "ia_unstable_points": 0,
        })
        assert "⚠" not in captured["text"]


# ── ML-112: SRK er / heater checks ───────────────────────────────────

class TestSrkHardwareChecks:

    def _settings(self, **kw):
        from lm19.srk import SrkSettings
        base = dict(ua_min=100.0, ua_max=200.0, ug1_min=-10.0,
                    ug1_max=-6.0, ug2=0.0, samples=1, settle_s=0.0,
                    calibration=IDENTITY_CAL,
                    settle_per_volt_s=0.0, settle_base_s=0.0,
                    ug1_verify_tolerance=1000.0,  # drift check not under test
                    is_triode=True)
        base.update(kw)
        return SrkSettings(**base)

    def _client(self, er: int = 0, uh: int = 63) -> _FakeClient:
        values = {"Ia": 100, "Ig2": 0, "Ua": 150, "Ug1": 512, "Ug2": 0,
                  "Uh": uh, "Ih": 0, "Er": er}
        client = _FakeClient(values, dict(values))
        return client

    def test_er_flag_aborts_with_typed_error(self, monkeypatch):
        from lm19.srk import SrkVerifyError, measure_srk
        import lm19.srk as srk_mod
        # settle succeeds trivially (readback == whatever was asked)
        monkeypatch.setattr(
            srk_mod, "_set_param_calibrated",
            lambda client, name, ch, target, prev, cal, **kw: target)
        with pytest.raises(SrkVerifyError, match="Hardware protection"):
            measure_srk(self._client(er=0x04), self._settings())

    def test_dead_heater_aborts(self, monkeypatch):
        from lm19.scan.exceptions import HeaterLostError
        from lm19.srk import measure_srk
        import lm19.srk as srk_mod
        monkeypatch.setattr(
            srk_mod, "_set_param_calibrated",
            lambda client, name, ch, target, prev, cal, **kw: target)
        # uh raw 0 → decode ~0 V while settings expect 6.3 V
        with pytest.raises(HeaterLostError):
            measure_srk(self._client(uh=0), self._settings(uh=6.3))

    def test_clean_point_passes(self, monkeypatch):
        from lm19.srk import measure_srk
        import lm19.srk as srk_mod
        monkeypatch.setattr(
            srk_mod, "_set_param_calibrated",
            lambda client, name, ch, target, prev, cal, **kw: target)
        s, r, k, points, unc = measure_srk(
            self._client(), self._settings())
        assert len(points) == 4  # 2 Ug1 × 2 Ua corners measured


# ── ML-090: dropped reset returns a reason ───────────────────────────

class TestResetOutputsDropped:

    def _controller(self):
        from app.scan_controller import ScanController
        cfg = MagicMock(ug1_after_stop=-40.0, ug1_settle_s=0.1)
        return ScanController(cfg)

    def test_port_closed_reason(self, qapp):
        ctrl = self._controller()
        assert ctrl.reset_outputs(None) == "port_closed"
        dead = MagicMock()
        dead.is_open.return_value = False
        assert ctrl.reset_outputs(dead) == "port_closed"

    def test_busy_reason(self, qapp):
        ctrl = self._controller()
        running = MagicMock()
        running.isRunning.return_value = True
        ctrl._reset_worker = running
        client = MagicMock()
        client.is_open.return_value = True
        assert ctrl.reset_outputs(client) == "reset_busy"


# ── ML-091: uncertainty reaches label + saved dict ───────────────────

class TestSrkUncertaintyVisible:

    def test_average_uncertainty(self, qapp):
        from app.srk_widget import SrkController
        valid = [
            {"s": 5.0, "r": 8.0, "k": 40.0,
             "uncertainty": {"s": 0.02, "r": 0.04, "k": 0.05}},
            {"s": 5.2, "r": 8.2, "k": 42.0,
             "uncertainty": {"s": 0.04, "r": 0.06, "k": 0.07}},
        ]
        unc = SrkController._average_uncertainty(valid)
        assert unc["s"] == pytest.approx(0.03)
        assert unc["r"] == pytest.approx(0.05)
        assert unc["k"] == pytest.approx(0.06)
        assert SrkController._average_uncertainty(
            [{"s": 5.0, "r": 8.0, "k": 40.0, "uncertainty": None}]) == {}

    def test_saved_measurement_carries_uncertainty(self, qapp):
        from app.srk_widget import SrkController
        ctrl = SrkController()
        ctrl._after_scan = True
        ctrl._pending_points = [{"ua": 1.0}]
        ctrl._pending_meta = {"tube_type": "EL84"}
        saved = {}
        ctrl.measurement_ready.connect(
            lambda m, pts: saved.update(m))
        results = [{"s": 5.0, "r": 8.0, "k": 40.0, "points": [],
                    "uncertainty": {"s": 0.02, "r": 0.04, "k": 0.05}}]
        ctrl._on_finished(results)
        assert saved["srk"]["uncertainty"]["s"] == pytest.approx(0.02)
        assert saved["srk_results"][0]["uncertainty"]["k"] == pytest.approx(0.05)

    def test_label_suffix_contains_uncertainty(self, qapp):
        from app.srk_widget import SrkController
        ctrl = SrkController()
        labels: List[str] = []
        ctrl.label_changed.connect(labels.append)
        ctrl._on_finished([{"s": 5.0, "r": 8.0, "k": 40.0, "points": [],
                            "uncertainty": {"s": 0.02, "r": 0.04,
                                            "k": 0.05}}])
        assert labels and "±" in labels[-1]


# ── ML-083: preheat deadline constants ───────────────────────────────

class TestPreheatTimeoutConstants:

    def test_constants_sane(self):
        from app.ui_theme import (
            HEALTH_PREHEAT_TIMEOUT_FACTOR, HEALTH_PREHEAT_TIMEOUT_MARGIN_S,
        )
        assert HEALTH_PREHEAT_TIMEOUT_FACTOR >= 1.5
        assert HEALTH_PREHEAT_TIMEOUT_MARGIN_S > 0

    def test_timeout_aborts_and_notifies(self, qapp, monkeypatch):
        """Past the deadline the wait loop stops, cleans up, and shows
        both a progress-label error and a QMessageBox."""
        import time as time_mod
        from PySide6.QtWidgets import QMessageBox
        import app.health_tab as ht

        tab = ht.HealthTab.__new__(ht.HealthTab)  # no full UI construction
        tab._pending_start_after_preheat = True
        tab._preheat_start_ts = time_mod.monotonic() - 10_000.0
        tab._preheat_warmup_s = 10
        tab.get_preheat_done = lambda: False
        tab._preheat_wait_timer = None
        cleaned = {}
        monkeypatch.setattr(
            ht.HealthTab, "_cleanup_after_test",
            lambda self: cleaned.setdefault("yes", True))
        label = MagicMock()
        state = MagicMock()
        tab.progress_label = label
        tab.live_state = state
        boxes = {}
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a: boxes.setdefault("shown", True)))
        tab._check_preheat_ready()
        assert cleaned.get("yes") and boxes.get("shown")
        assert label.setText.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
