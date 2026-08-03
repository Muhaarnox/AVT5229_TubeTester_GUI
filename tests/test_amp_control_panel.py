"""Tests for AmpControlPanel and bidirectional spin sync."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from app.amp_control_panel import AmpControlPanel
from app.main_window import MainWindow
from lm19.amplifier.constants import (
    CIRCUIT_CF,
    CIRCUIT_PP,
    CIRCUIT_SE,
    CIRCUIT_SE_XFMR,
)

pytestmark = [pytest.mark.smoke_ui]


@pytest.fixture(autouse=True)
def _ensure_qapp():
    QApplication.instance() or QApplication([])


# ═══════════════════════════════════════════════════════════════════
# 1. AmpControlPanel creation
# ═══════════════════════════════════════════════════════════════════

class TestAmpControlPanelCreation:
    def test_creates_without_error(self):
        panel = AmpControlPanel()
        assert panel is not None
        panel.close()

    def test_has_circuit_combo(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "circuit_combo")
        panel.close()

    def test_has_data_source_combo(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "data_source_combo")
        panel.close()

    def test_has_hd_method_combo(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "hd_method_combo")
        panel.close()

    def test_has_parameter_spins(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "ub_spin")
        assert hasattr(panel, "ra_spin")
        assert hasattr(panel, "ug1_spin")
        assert hasattr(panel, "swing_spin")
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 2. params_snapshot
# ═══════════════════════════════════════════════════════════════════

class TestParamsSnapshot:
    def test_returns_amp_params(self):
        panel = AmpControlPanel()
        p = panel.params_snapshot()
        assert p.ub == panel.ub_spin.value()
        assert p.ra == panel.ra_spin.value()
        assert p.ug1_bias == panel.ug1_spin.value()
        panel.close()

    def test_reflects_changed_values(self):
        panel = AmpControlPanel()
        panel.ub_spin.setValue(300.0)
        panel.ra_spin.setValue(8.0)
        panel.ug1_spin.setValue(-10.0)
        panel.swing_spin.setValue(2.5)
        p = panel.params_snapshot()
        assert p.ub == 300.0
        assert p.ra == 8.0
        assert p.ug1_bias == -10.0
        assert p.half_swing == 2.5
        panel.close()

    def test_show_hd45_default_false(self):
        panel = AmpControlPanel()
        p = panel.params_snapshot()
        assert p.show_hd45 is False
        panel.close()

    def test_show_hd45_toggled(self):
        panel = AmpControlPanel()
        panel.show_hd45_cb.setChecked(True)
        p = panel.params_snapshot()
        assert p.show_hd45 is True
        panel.close()

    def test_show_gzp_default_false(self):
        panel = AmpControlPanel()
        p = panel.params_snapshot()
        assert p.show_gzp is False
        panel.close()

    def test_show_gzp_toggled(self):
        panel = AmpControlPanel()
        panel.show_gzp_cb.setChecked(True)
        p = panel.params_snapshot()
        assert p.show_gzp is True
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# Optimizer section
# ═══════════════════════════════════════════════════════════════════

class TestOptimizerSection:

    def test_optimizer_widgets_exist(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "opt_pareto_btn")
        assert hasattr(panel, "opt_target_combo")
        assert hasattr(panel, "opt_pout_min_spin")
        assert hasattr(panel, "opt_thd_max_spin")
        assert hasattr(panel, "opt_run_btn")
        panel.close()

    def test_no_class_combo_attr(self):
        """``AmpControlPanel`` exposes ``opt_class_a_mode_combo`` only
        (no ``opt_class_combo`` — the class-A power filter superseded
        the any/A/AB filter)."""
        panel = AmpControlPanel()
        assert not hasattr(panel, "opt_class_combo"), \
            "AmpControlPanel must not define opt_class_combo; use opt_class_a_mode_combo"
        panel.close()

    def test_optimizer_enabled_default_false(self):
        panel = AmpControlPanel()
        assert panel.optimizer_enabled is False
        panel.close()

    def test_optimizer_enabled_toggled(self):
        panel = AmpControlPanel()
        panel.opt_pareto_btn.setChecked(True)
        assert panel.optimizer_enabled is True
        panel.close()

    def test_optimizer_constraints_default(self):
        panel = AmpControlPanel()
        c = panel.optimizer_constraints()
        assert c.target == "min_thd"
        assert c.pout_min_w == 0.0
        assert c.pa_max_w == panel.pa_max_spin.value()
        panel.close()

    def test_optimizer_constraints_changed(self):
        panel = AmpControlPanel()
        panel.opt_target_combo.setCurrentIndex(1)  # max_pout
        panel.opt_pout_min_spin.setValue(1.0)
        panel.opt_thd_max_spin.setValue(5.0)
        c = panel.optimizer_constraints()
        assert c.target == "max_pout"
        assert c.pout_min_w == 1.0
        assert c.thd_max_pct == 5.0
        panel.close()

    # ── Class-A power threshold (PP filter) ─────────────────────

    def test_class_a_widgets_exist(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "opt_class_a_mode_combo")
        assert hasattr(panel, "opt_class_a_value_spin")
        # Default: off, spin disabled
        assert panel.opt_class_a_mode_combo.currentData() == "off"
        assert panel.opt_class_a_value_spin.isEnabled() is False
        panel.close()

    def test_class_a_mode_off_disables_spin(self):
        panel = AmpControlPanel()
        panel.opt_class_a_mode_combo.setCurrentIndex(1)  # absolute
        assert panel.opt_class_a_value_spin.isEnabled() is True
        panel.opt_class_a_mode_combo.setCurrentIndex(0)  # off
        assert panel.opt_class_a_value_spin.isEnabled() is False
        panel.close()

    def test_class_a_mode_absolute_suffix(self):
        panel = AmpControlPanel()
        panel.opt_class_a_mode_combo.setCurrentIndex(1)  # absolute
        assert panel.opt_class_a_value_spin.suffix().strip() == "W"
        panel.close()

    def test_class_a_mode_percent_suffix(self):
        panel = AmpControlPanel()
        panel.opt_class_a_mode_combo.setCurrentIndex(2)  # percent
        assert panel.opt_class_a_value_spin.suffix().strip() == "%"
        panel.close()

    def test_class_a_constraints_propagate_off(self):
        panel = AmpControlPanel()
        # Default state
        c = panel.optimizer_constraints()
        assert c.class_a_power_mode == "off"
        assert c.class_a_power_value == 0.0
        panel.close()

    def test_class_a_constraints_propagate_absolute(self):
        panel = AmpControlPanel()
        panel.opt_class_a_mode_combo.setCurrentIndex(1)  # absolute
        panel.opt_class_a_value_spin.setValue(3.5)
        c = panel.optimizer_constraints()
        assert c.class_a_power_mode == "absolute"
        assert c.class_a_power_value == 3.5
        panel.close()

    def test_class_a_constraints_propagate_percent(self):
        panel = AmpControlPanel()
        panel.opt_class_a_mode_combo.setCurrentIndex(2)  # percent
        panel.opt_class_a_value_spin.setValue(20.0)
        c = panel.optimizer_constraints()
        assert c.class_a_power_mode == "percent"
        assert c.class_a_power_value == 20.0
        panel.close()

    def test_hd_method_propagates_to_optimizer_constraints(self):
        """UI hd_method_combo flows into OptimizerConstraints.hd_method."""
        panel = AmpControlPanel()
        # auto/chebyshev/dft/5point are the 4 options
        for idx, expected in enumerate(["auto", "chebyshev", "dft", "5point"]):
            panel.hd_method_combo.setCurrentIndex(idx)
            c = panel.optimizer_constraints()
            assert c.hd_method == expected, f"index={idx}"
        panel.close()

    def test_apply_best_and_top_n_buttons_exist(self):
        """Post-run action buttons present and disabled by default."""
        panel = AmpControlPanel()
        assert hasattr(panel, "opt_apply_btn")
        assert hasattr(panel, "opt_top_n_btn")
        assert panel.opt_apply_btn.isEnabled() is False
        assert panel.opt_top_n_btn.isEnabled() is False
        panel.close()

    def test_apply_best_button_emits_signal(self):
        from PySide6.QtCore import Signal
        panel = AmpControlPanel()
        panel.set_optimizer_result_available(True)
        captured = []
        panel.optimizer_apply_best.connect(lambda: captured.append(True))
        panel.opt_apply_btn.click()
        assert captured == [True]
        panel.close()

    def test_top_n_button_emits_signal(self):
        panel = AmpControlPanel()
        panel.set_optimizer_result_available(True)
        captured = []
        panel.optimizer_show_top_n.connect(lambda: captured.append(True))
        panel.opt_top_n_btn.click()
        assert captured == [True]
        panel.close()

    def test_post_run_buttons_disabled_during_run(self):
        """Starting a new run disables Apply/Top-N (prevents stale-result clicks)."""
        panel = AmpControlPanel()
        panel.set_optimizer_result_available(True)
        assert panel.opt_apply_btn.isEnabled() is True
        panel.set_optimizer_running(True)
        assert panel.opt_apply_btn.isEnabled() is False
        assert panel.opt_top_n_btn.isEnabled() is False
        panel.close()

    def test_set_optimizer_result_available_toggles_both(self):
        panel = AmpControlPanel()
        panel.set_optimizer_result_available(True)
        assert panel.opt_apply_btn.isEnabled() is True
        assert panel.opt_top_n_btn.isEnabled() is True
        panel.set_optimizer_result_available(False)
        assert panel.opt_apply_btn.isEnabled() is False
        assert panel.opt_top_n_btn.isEnabled() is False
        panel.close()

    # ── Target tooltips + balanced_weight control ────────────────

    def test_target_combo_has_tooltip(self):
        panel = AmpControlPanel()
        tip = panel.opt_target_combo.toolTip()
        assert tip and "balanced" in tip.lower()
        panel.close()

    def test_target_combo_items_have_tooltips(self):
        from PySide6.QtCore import Qt
        panel = AmpControlPanel()
        # 3 items: min_thd / max_pout / balanced — each has its own tip
        for i in range(3):
            tip = panel.opt_target_combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
            assert tip, f"item {i} missing tooltip"
        panel.close()

    def test_balanced_weight_hidden_by_default(self):
        """min_thd is the default target → weight controls hidden."""
        panel = AmpControlPanel()
        assert panel.opt_balanced_weight_label.isHidden()
        assert panel.opt_balanced_weight_spin.isHidden()
        panel.close()

    def test_balanced_weight_visible_when_balanced_selected(self):
        panel = AmpControlPanel()
        # Find 'balanced' index
        for i in range(panel.opt_target_combo.count()):
            if panel.opt_target_combo.itemData(i) == "balanced":
                panel.opt_target_combo.setCurrentIndex(i)
                break
        panel.show()  # required so isHidden reflects layout state
        assert not panel.opt_balanced_weight_spin.isHidden()
        panel.close()

    def test_balanced_weight_hidden_again_when_other_target(self):
        panel = AmpControlPanel()
        panel.show()
        # Switch to balanced first
        for i in range(panel.opt_target_combo.count()):
            if panel.opt_target_combo.itemData(i) == "balanced":
                panel.opt_target_combo.setCurrentIndex(i)
                break
        assert not panel.opt_balanced_weight_spin.isHidden()
        # Switch back to min_thd
        for i in range(panel.opt_target_combo.count()):
            if panel.opt_target_combo.itemData(i) == "min_thd":
                panel.opt_target_combo.setCurrentIndex(i)
                break
        assert panel.opt_balanced_weight_spin.isHidden()
        panel.close()

    def test_balanced_weight_default_is_half(self):
        panel = AmpControlPanel()
        assert panel.opt_balanced_weight_spin.value() == 0.5
        panel.close()

    def test_balanced_weight_propagates_to_constraints(self):
        panel = AmpControlPanel()
        panel.opt_balanced_weight_spin.setValue(0.8)
        c = panel.optimizer_constraints()
        assert c.balanced_weight == 0.8
        panel.close()

    def test_balanced_weight_range(self):
        panel = AmpControlPanel()
        assert panel.opt_balanced_weight_spin.minimum() == 0.0
        # Range goes up to 20 — typical Pout-priority needs w≥5 to flip
        # the "best" away from the lowest-THD point.
        assert panel.opt_balanced_weight_spin.maximum() == 20.0
        panel.close()

    def test_balanced_weight_high_value_actually_changes_best(self):
        """Sanity: a high balanced_weight must shift optimizer's pick
        toward higher Pout. Default 0.5 is too small for visible flip."""
        from lm19.tube_sim import quick_triode
        from lm19.optimizer import OptimizerConstraints, optimize_measurements
        _, pts = quick_triode("12AU7")
        common = dict(
            target="balanced",
            ug1_range=(-15.0, -3.0), ra_range=(5.0, 30.0),
            ug1_steps=8, ra_steps=8, pa_max_w=2.75, pout_min_w=0.05,
        )
        r_low = optimize_measurements(
            pts, ub=250.0,
            constraints=OptimizerConstraints(balanced_weight=0.5, **common),
        )
        r_high = optimize_measurements(
            pts, ub=250.0,
            constraints=OptimizerConstraints(balanced_weight=15.0, **common),
        )
        assert r_low.best and r_high.best, "guard de-vacuated 2026-07-12: value must be present"
        assert r_high.best.pout_mw >= r_low.best.pout_mw, (
            f"high w={15} Pout={r_high.best.pout_mw} not >= "
            f"low w=0.5 Pout={r_low.best.pout_mw}"
        )

    def test_append_optimizer_status_preserves_existing(self):
        """append_optimizer_status keeps the prior text (e.g. HD method line)."""
        panel = AmpControlPanel()
        panel.set_optimizer_status("Grid: 100 pts\nHD method: dft")
        panel.append_optimizer_status("Applied: Ub=250V")
        assert "HD method: dft" in panel.opt_status_label.text()
        assert "Applied: Ub=250V" in panel.opt_status_label.text()
        panel.close()

    def test_append_optimizer_status_handles_empty(self):
        panel = AmpControlPanel()
        panel.set_optimizer_status("")
        panel.append_optimizer_status("Applied: foo")
        assert panel.opt_status_label.text() == "Applied: foo"
        panel.close()

    def test_pp_ra_dc_spin_exists_with_default(self):
        """PP half-primary winding DC resistance — new field for full
        transformer modelling parity with SE Transformer mode."""
        panel = AmpControlPanel()
        assert hasattr(panel, "pp_ra_dc_spin")
        assert panel.pp_ra_dc_spin.value() == 0.1
        assert panel.pp_ra_dc_spin.minimum() == 0.0
        assert panel.pp_ra_dc_spin.maximum() == 5.0
        panel.close()

    def test_pp_ra_dc_propagates_to_optimizer_constraints(self):
        panel = AmpControlPanel()
        panel.pp_ra_dc_spin.setValue(0.3)
        c = panel.optimizer_constraints()
        assert c.pp_ra_dc == 0.3
        panel.close()

    def test_pp_ra_dc_propagates_to_amp_params(self):
        panel = AmpControlPanel()
        panel.pp_ra_dc_spin.setValue(0.25)
        params = panel.params_snapshot()
        assert params.pp_ra_dc == 0.25
        panel.close()

    def test_optimizer_status_label_is_copyable(self):
        """Result field must allow text selection by mouse for copy/paste."""
        from PySide6.QtCore import Qt
        panel = AmpControlPanel()
        flags = panel.opt_status_label.textInteractionFlags()
        assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
        panel.close()

    def test_optimize_requested_signal(self):
        panel = AmpControlPanel()
        received = []
        panel.optimize_requested.connect(lambda: received.append(True))
        # Emit clicked directly (button hidden in collapsed section in offscreen)
        panel.opt_run_btn.clicked.emit()
        assert len(received) == 1
        panel.close()

    def test_set_optimizer_status(self):
        panel = AmpControlPanel()
        panel.set_optimizer_status("test status")
        assert panel.opt_status_label.text() == "test status"
        panel.close()

    def test_optimizer_has_cancel_and_progress(self):
        panel = AmpControlPanel()
        assert hasattr(panel, "opt_cancel_btn")
        assert hasattr(panel, "opt_progress")
        # Initially hidden (offscreen: use isHidden, not isVisible)
        assert panel.opt_cancel_btn.isHidden()
        assert panel.opt_progress.isHidden()
        panel.close()

    def test_optimizer_running_state(self):
        panel = AmpControlPanel()
        panel.set_optimizer_running(True)
        assert not panel.opt_cancel_btn.isHidden()
        assert not panel.opt_progress.isHidden()
        assert panel.opt_run_btn.isHidden()

        panel.set_optimizer_running(False)
        assert panel.opt_cancel_btn.isHidden()
        assert panel.opt_progress.isHidden()
        assert not panel.opt_run_btn.isHidden()
        panel.close()

    def test_optimizer_progress_update(self):
        panel = AmpControlPanel()
        panel.set_optimizer_running(True)
        panel.set_optimizer_progress(42, "Grid sweep...")
        assert panel.opt_progress.value() == 42
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 3. Bidirectional spin sync (_sync_spin_pair)
# ═══════════════════════════════════════════════════════════════════

class TestBidirectionalSync:
    """Test MainWindow._sync_spin_pair static method."""

    def test_a_to_b(self):
        a = QDoubleSpinBox()
        b = QDoubleSpinBox()
        a.setRange(0, 1000)
        b.setRange(0, 1000)
        MainWindow._sync_spin_pair(a, b)
        a.setValue(42.0)
        assert b.value() == 42.0

    def test_b_to_a(self):
        a = QDoubleSpinBox()
        b = QDoubleSpinBox()
        a.setRange(0, 1000)
        b.setRange(0, 1000)
        MainWindow._sync_spin_pair(a, b)
        b.setValue(77.0)
        assert a.value() == 77.0

    def test_no_infinite_loop(self):
        """Changing A sets B, which should NOT re-trigger A's signal."""
        a = QDoubleSpinBox()
        b = QDoubleSpinBox()
        a.setRange(0, 1000)
        b.setRange(0, 1000)
        call_count = [0]

        original_set = a.setValue

        def counting_set(val):
            call_count[0] += 1
            original_set(val)

        MainWindow._sync_spin_pair(a, b)
        a.setValue(50.0)
        # b should be 50, and no infinite loop
        assert b.value() == 50.0
        assert a.value() == 50.0

    def test_both_directions_sequential(self):
        a = QDoubleSpinBox()
        b = QDoubleSpinBox()
        a.setRange(0, 1000)
        b.setRange(0, 1000)
        MainWindow._sync_spin_pair(a, b)

        a.setValue(100.0)
        assert b.value() == 100.0

        b.setValue(200.0)
        assert a.value() == 200.0

        a.setValue(300.0)
        assert b.value() == 300.0


# ═══════════════════════════════════════════════════════════════════
# 4. settings_changed signal
# ═══════════════════════════════════════════════════════════════════

class TestSettingsSignal:
    def test_settings_changed_emitted_on_ub_change(self):
        panel = AmpControlPanel()
        received = []
        panel.settings_changed.connect(lambda: received.append(True))
        panel.ub_spin.setValue(300.0)
        assert len(received) > 0
        panel.close()

    def test_settings_changed_emitted_on_circuit_change(self):
        panel = AmpControlPanel()
        received = []
        panel.settings_changed.connect(lambda: received.append(True))
        panel.circuit_combo.setCurrentIndex(1)
        assert len(received) > 0
        panel.close()

    def test_settings_changed_emitted_on_ra_change(self):
        panel = AmpControlPanel()
        received = []
        panel.settings_changed.connect(lambda: received.append(True))
        panel.ra_spin.setValue(10.0)
        assert len(received) > 0
        panel.close()

    def test_settings_changed_emitted_on_nfb_toggle(self):
        panel = AmpControlPanel()
        received = []
        panel.settings_changed.connect(lambda: received.append(True))
        panel.nfb_check.setChecked(True)
        assert len(received) > 0
        panel.close()

    def test_no_signal_during_set_series_items(self):
        panel = AmpControlPanel()
        received = []
        panel.settings_changed.connect(lambda: received.append(True))
        panel.set_series_items({1: "Scan 1", 2: "Scan 2"})
        assert len(received) == 0
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 5. params_snapshot per circuit type
# ═══════════════════════════════════════════════════════════════════

class TestParamsSnapshotCircuits:
    def test_se_circuit(self):
        panel = AmpControlPanel()
        panel.circuit_combo.setCurrentIndex(0)  # SE
        p = panel.params_snapshot()
        assert p.circuit == CIRCUIT_SE
        panel.close()

    def test_se_xfmr_circuit_includes_ra_dc(self):
        panel = AmpControlPanel()
        idx = panel.circuit_combo.findData("se_xfmr")
        panel.circuit_combo.setCurrentIndex(idx)
        panel.xfmr_ra_dc_spin.setValue(0.1)
        p = panel.params_snapshot()
        assert p.circuit == CIRCUIT_SE_XFMR
        assert p.ra_dc == 0.1
        panel.close()

    def test_cf_circuit_includes_rk_rl(self):
        panel = AmpControlPanel()
        idx = panel.circuit_combo.findData("cf")
        panel.circuit_combo.setCurrentIndex(idx)
        panel.cf_rk_spin.setValue(15.0)
        panel.cf_rl_spin.setValue(20.0)
        p = panel.params_snapshot()
        assert p.circuit == CIRCUIT_CF
        assert p.cf_rk == 15.0
        assert p.cf_rl == 20.0
        panel.close()

    def test_pp_circuit_includes_raa(self):
        panel = AmpControlPanel()
        idx = panel.circuit_combo.findData("pp")
        panel.circuit_combo.setCurrentIndex(idx)
        panel.pp_raa_spin.setValue(10.0)
        p = panel.params_snapshot()
        assert p.circuit == CIRCUIT_PP
        assert p.pp_raa == 10.0
        panel.close()

    def test_pp_matched_by_default(self):
        panel = AmpControlPanel()
        p = panel.params_snapshot()
        assert p.pp_matched is True
        assert p.pp_tube_b_sid is None
        panel.close()

    def test_pp_unmatched_reads_tube_b(self):
        panel = AmpControlPanel()
        panel.pp_matched_btn.setChecked(False)
        panel.pp_tube_b_combo.addItem("Other tube", 42)
        panel.pp_tube_b_combo.setCurrentIndex(0)
        p = panel.params_snapshot()
        assert p.pp_matched is False
        assert p.pp_tube_b_sid == 42
        panel.close()

    def test_swing_zero_becomes_none(self):
        panel = AmpControlPanel()
        panel.swing_spin.setValue(0.0)
        p = panel.params_snapshot()
        assert p.half_swing is None
        panel.close()

    def test_nfb_disabled_returns_none(self):
        panel = AmpControlPanel()
        panel.nfb_check.setChecked(False)
        p = panel.params_snapshot()
        assert p.nfb_db is None
        panel.close()

    def test_nfb_enabled_returns_value(self):
        panel = AmpControlPanel()
        panel.nfb_check.setChecked(True)
        panel.nfb_spin.setValue(6.0)
        p = panel.params_snapshot()
        assert p.nfb_db == 6.0
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 6. show_results
# ═══════════════════════════════════════════════════════════════════

class TestShowResults:
    def test_sets_label_text(self):
        panel = AmpControlPanel()
        panel.show_results("<b>THD=3.2%</b>")
        assert "THD=3.2%" in panel.results_label.text()
        panel.close()

    def test_html_preserved(self):
        panel = AmpControlPanel()
        panel.show_results("<table><tr><td>x</td></tr></table>")
        assert "<table>" in panel.results_label.text()
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 7. set_series_items
# ═══════════════════════════════════════════════════════════════════

class TestSetSeriesItems:
    def test_populates_combo(self):
        panel = AmpControlPanel()
        panel.set_series_items({1: "Scan 1", 2: "Scan 2"})
        assert panel.source_combo.count() == 2
        panel.close()

    def test_selects_current_sid(self):
        panel = AmpControlPanel()
        panel.set_series_items({10: "A", 20: "B"}, current_sid=20)
        assert panel.source_combo.currentData() == 20
        panel.close()

    def test_empty_dict(self):
        panel = AmpControlPanel()
        panel.set_series_items({})
        assert panel.source_combo.count() == 0
        panel.close()

    def test_replaces_existing(self):
        panel = AmpControlPanel()
        panel.set_series_items({1: "First"})
        panel.set_series_items({2: "Second", 3: "Third"})
        assert panel.source_combo.count() == 2
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 8. set_pp_tube_b_items
# ═══════════════════════════════════════════════════════════════════

class TestSetPPTubeBItems:
    def test_populates_tube_b_combo(self):
        panel = AmpControlPanel()
        panel.set_pp_tube_b_items({1: "Tube A", 2: "Tube B"})
        assert panel.pp_tube_b_combo.count() == 2
        panel.close()

    def test_empty_dict(self):
        panel = AmpControlPanel()
        panel.set_pp_tube_b_items({})
        assert panel.pp_tube_b_combo.count() == 0
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 9. set_available_models
# ═══════════════════════════════════════════════════════════════════

class TestSetAvailableModels:
    def test_adds_models_to_data_source(self):
        panel = AmpControlPanel()
        panel.set_available_models({"koren": "Koren fit"})
        keys = panel.data_source_combo.checked_keys()
        assert "measurements" in keys
        assert "koren" in keys
        panel.close()

    def test_no_models_only_measurements(self):
        panel = AmpControlPanel()
        panel.set_available_models({})
        keys = panel.data_source_combo.checked_keys()
        assert keys == ["measurements"]
        panel.close()

    def test_preserves_unchecked_state(self):
        """Previously unchecked sources stay unchecked after model update."""
        panel = AmpControlPanel()
        panel.set_available_models({"koren": "Koren"})
        # Uncheck koren
        model = panel.data_source_combo._model
        for row in range(model.rowCount()):
            item = model.item(row)
            from PySide6.QtCore import Qt
            if item.data(Qt.ItemDataRole.UserRole) == "koren":
                item.setCheckState(Qt.CheckState.Unchecked)
        # Re-populate with additional model
        panel.set_available_models({"koren": "Koren", "dempwolf": "Dempwolf"})
        keys = panel.data_source_combo.checked_keys()
        assert "measurements" in keys
        assert "dempwolf" in keys  # New model → checked
        assert "koren" not in keys  # Was unchecked → stays unchecked
        panel.close()


# ═══════════════════════════════════════════════════════════════════
# 10. Circuit widget visibility
# ═══════════════════════════════════════════════════════════════════

class TestCircuitWidgetVisibility:
    """Test circuit-specific widget show/hide.

    Note: In offscreen mode, isVisible() requires parent to be shown.
    We use isVisibleTo(panel) or check the internal visibility flag
    via !isHidden() after the widget has been explicitly set visible.
    """

    def test_se_hides_all_circuit_widgets(self):
        panel = AmpControlPanel()
        panel.circuit_combo.setCurrentIndex(panel.circuit_combo.findData("se"))
        assert panel._xfmr_widget.isHidden()
        assert panel._cf_widget.isHidden()
        assert panel._pp_widget.isHidden()
        panel.close()

    def test_se_xfmr_shows_xfmr_widget(self):
        panel = AmpControlPanel()
        panel.circuit_combo.setCurrentIndex(panel.circuit_combo.findData("se_xfmr"))
        assert not panel._xfmr_widget.isHidden()
        assert panel._cf_widget.isHidden()
        assert panel._pp_widget.isHidden()
        panel.close()

    def test_cf_shows_cf_widget(self):
        panel = AmpControlPanel()
        panel.circuit_combo.setCurrentIndex(panel.circuit_combo.findData("cf"))
        assert panel._xfmr_widget.isHidden()
        assert not panel._cf_widget.isHidden()
        assert panel._pp_widget.isHidden()
        panel.close()

    def test_pp_shows_pp_widget(self):
        panel = AmpControlPanel()
        panel.circuit_combo.setCurrentIndex(panel.circuit_combo.findData("pp"))
        assert panel._xfmr_widget.isHidden()
        assert panel._cf_widget.isHidden()
        assert not panel._pp_widget.isHidden()
        panel.close()

    def test_pp_matched_hides_tube_b(self):
        panel = AmpControlPanel()
        panel.pp_matched_btn.setChecked(True)
        assert panel.pp_tube_b_combo.isHidden()
        panel.close()

    def test_pp_unmatched_shows_tube_b(self):
        panel = AmpControlPanel()
        panel.pp_matched_btn.setChecked(False)
        assert not panel.pp_tube_b_combo.isHidden()
        panel.close()

    def test_nfb_spin_disabled_when_unchecked(self):
        panel = AmpControlPanel()
        panel.nfb_check.setChecked(False)
        assert not panel.nfb_spin.isEnabled()
        panel.close()

    def test_nfb_spin_enabled_when_checked(self):
        panel = AmpControlPanel()
        panel.nfb_check.setChecked(True)
        assert panel.nfb_spin.isEnabled()
        panel.close()


class TestOptimizerRangesGateOnCheckedState:
    """ML-076: the Ub/Ug2 range gate must follow the collapsible's CHECKED
    state, not isVisible() — visibility is False for a hidden window or an
    offscreen run even when the user expanded the section and set ranges,
    silently dropping them."""

    @staticmethod
    def _panel():
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from app.amp_control_panel import AmpControlPanel
        return AmpControlPanel()

    def test_expanded_offscreen_panel_keeps_ranges(self):
        panel = self._panel()          # never .show()n — isVisible() False
        panel.opt_group.setChecked(True)
        panel.opt_ub_min_spin.setValue(200.0)
        panel.opt_ub_max_spin.setValue(350.0)
        c = panel.optimizer_constraints()
        assert c.ub_range == (200.0, 350.0),             "expanded section lost its Ub range on a hidden panel"
        assert c.ug2_range is not None

    def test_collapsed_section_drops_ranges(self):
        panel = self._panel()
        panel.opt_group.setChecked(False)
        c = panel.optimizer_constraints()
        assert c.ub_range is None
        assert c.ug2_range is None
