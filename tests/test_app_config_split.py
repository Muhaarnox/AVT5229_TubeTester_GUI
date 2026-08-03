"""Tests for split config loading (app/scan/srk/health)."""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19 import app_config


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _set_fake_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    fake_module_path = project_root / "lm19" / "app_config.py"
    fake_module_path.parent.mkdir(parents=True, exist_ok=True)
    fake_module_path.write_text("# fake file for path resolution\n", encoding="utf-8")
    monkeypatch.setattr(app_config, "__file__", str(fake_module_path))
    return project_root


def test_load_split_configs_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _set_fake_project_root(monkeypatch, tmp_path)
    cfg_dir = root / "config"

    _write_json(cfg_dir / "app.json", {"live_poll_ms": 1234, "log_level": "info"})
    _write_json(
        cfg_dir / "scan.json",
        {
            "ua_settle_per_volt_s": 0.003,
            "ua_settle_base_s": 0.2,
            "ua_tolerance": 1.2,
            "ua_retries": 4,
            "ug1_settle_per_volt_s": 0.11,
            "ug1_settle_base_s": 0.2,
            "ug1_tolerance": 0.12,
            "ug1_retries": 3,
            "ug2_settle_per_volt_s": 0.003,
            "ug2_settle_base_s": 0.2,
            "ug2_tolerance": 1.2,
            "ug2_retries": 3,
            "ia_samples": 4,
            "pig2_over_pct": 25,
        },
    )
    _write_json(
        cfg_dir / "srk.json",
        {
            "samples": 7,
            "settle_s": 1.2,
            "verify_retries": 4,
            "ua_tolerance": 1.5,
            "ug2_tolerance": 1.6,
            "settle_per_volt_s": 0.6,
            "settle_base_s": 1.1,
            "ug1_step": 0.08,
        },
    )
    _write_json(
        cfg_dir / "health.json",
        {
            "ua_tolerance_v": 1.4,
            "ug1_tolerance_v": 0.3,
            "ug2_tolerance_v": 1.5,
            "ua_retries": 3,
            "ug1_retries": 3,
            "ug2_retries": 3,
            "ua_settle_per_volt_s": 0.004,
            "ua_settle_base_s": 0.2,
            "ug1_settle_per_volt_s": 0.03,
            "ug1_settle_base_s": 0.2,
            "ug2_settle_per_volt_s": 0.004,
            "ug2_settle_base_s": 0.2,
            "ug1_delta_v": 1.2,
            "ia_samples": 6,
            "ig2_samples": 6,
            "use_median": False,
            "emission_enabled_default": False,
            "emission_uh_ratio": 0.75,
            "emission_stable_warmup_ratio": 0.3,
            "emission_stable_min_s": 25,
            "emission_stable_max_s": 140,
            "weight_s": 0.4,
            "verdict_good_min": 77.0,
            "verdict_weak_min": 57.0,
            "verdict_strong_min": 92.0,
            "weight_ia": 0.25,
            "weight_rh": 0.1,
            "weight_screen": 0.1,
            "weight_emission": 0.15,
        },
    )

    cfg = app_config.load_app_config()

    assert cfg.live_poll_ms == 1234
    assert cfg.log_level == "INFO"
    assert cfg.scan_ua_retries == 4
    assert cfg.srk_samples == 7
    assert cfg.health_ug1_delta_v == 1.2
    assert cfg.health_emission_uh_ratio == pytest.approx(0.75)
    assert cfg.health_use_median is False
    assert cfg.health_verdict_good_min == pytest.approx(77.0)


def test_missing_scan_file_warns_to_console_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    root = _set_fake_project_root(monkeypatch, tmp_path)
    cfg_dir = root / "config"
    _write_json(cfg_dir / "app.json", {"live_poll_ms": 900})
    _write_json(
        cfg_dir / "srk.json",
        {
            "samples": 5,
            "settle_s": 1.0,
            "verify_retries": 3,
            "ua_tolerance": 1.0,
            "ug2_tolerance": 1.0,
            "settle_per_volt_s": 0.5,
            "settle_base_s": 1.0,
            "ug1_step": 0.04,
        },
    )
    _write_json(
        cfg_dir / "health.json",
        {
            "ua_tolerance_v": 1.0,
            "ug1_tolerance_v": 0.2,
            "ug2_tolerance_v": 1.0,
            "ua_retries": 2,
            "ug1_retries": 2,
            "ug2_retries": 2,
            "ua_settle_per_volt_s": 0.002,
            "ua_settle_base_s": 0.15,
            "ug1_settle_per_volt_s": 0.02,
            "ug1_settle_base_s": 0.15,
            "ug2_settle_per_volt_s": 0.002,
            "ug2_settle_base_s": 0.15,
            "ug1_delta_v": 1.0,
            "ia_samples": 5,
            "ig2_samples": 5,
            "use_median": True,
            "emission_enabled_default": True,
            "emission_uh_ratio": 0.8,
            "emission_stable_warmup_ratio": 0.25,
            "emission_stable_min_s": 20,
            "emission_stable_max_s": 120,
            "weight_ia": 0.3,
            "weight_s": 0.35,
            "weight_rh": 0.1,
            "weight_screen": 0.1,
            "weight_emission": 0.15,
            "verdict_strong_min": 90.0,
            "verdict_good_min": 75.0,
            "verdict_weak_min": 55.0,
        },
    )

    caplog.set_level(logging.WARNING, logger=app_config.__name__)
    cfg = app_config.load_app_config()

    assert cfg.scan_ua_retries == 2  # default from AppConfig
    out = capsys.readouterr().out
    assert "Missing scan file" in out
    assert any("Missing scan file" in rec.message for rec in caplog.records)


def test_invalid_health_json_warns_and_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    root = _set_fake_project_root(monkeypatch, tmp_path)
    cfg_dir = root / "config"
    _write_json(cfg_dir / "app.json", {})
    _write_json(
        cfg_dir / "scan.json",
        {
            "ua_settle_per_volt_s": 0.002,
            "ua_settle_base_s": 0.15,
            "ua_tolerance": 1.0,
            "ua_retries": 2,
            "ug1_settle_per_volt_s": 0.02,
            "ug1_settle_base_s": 0.15,
            "ug1_tolerance": 0.1,
            "ug1_retries": 2,
            "ug2_settle_per_volt_s": 0.002,
            "ug2_settle_base_s": 0.15,
            "ug2_tolerance": 1.0,
            "ug2_retries": 1,
            "ia_samples": 2,
            "pig2_over_pct": 20.0,
        },
    )
    _write_json(
        cfg_dir / "srk.json",
        {
            "samples": 5,
            "settle_s": 1.0,
            "verify_retries": 3,
            "ua_tolerance": 1.0,
            "ug2_tolerance": 1.0,
            "settle_per_volt_s": 0.5,
            "settle_base_s": 1.0,
            "ug1_step": 0.04,
        },
    )
    health_path = cfg_dir / "health.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text("{bad json", encoding="utf-8")

    caplog.set_level(logging.WARNING, logger=app_config.__name__)
    cfg = app_config.load_app_config()

    assert cfg.health_emission_uh_ratio == pytest.approx(0.8)  # default
    out = capsys.readouterr().out
    assert "Failed to parse health file" in out
    assert any("Failed to parse health file" in rec.message for rec in caplog.records)


def test_missing_required_health_keys_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    root = _set_fake_project_root(monkeypatch, tmp_path)
    cfg_dir = root / "config"
    _write_json(cfg_dir / "app.json", {})
    _write_json(
        cfg_dir / "scan.json",
        {
            "ua_settle_per_volt_s": 0.002,
            "ua_settle_base_s": 0.15,
            "ua_tolerance": 1.0,
            "ua_retries": 2,
            "ug1_settle_per_volt_s": 0.02,
            "ug1_settle_base_s": 0.15,
            "ug1_tolerance": 0.1,
            "ug1_retries": 2,
            "ug2_settle_per_volt_s": 0.002,
            "ug2_settle_base_s": 0.15,
            "ug2_tolerance": 1.0,
            "ug2_retries": 1,
            "ia_samples": 2,
            "pig2_over_pct": 20.0,
        },
    )
    _write_json(
        cfg_dir / "srk.json",
        {
            "samples": 5,
            "settle_s": 1.0,
            "verify_retries": 3,
            "ua_tolerance": 1.0,
            "ug2_tolerance": 1.0,
            "settle_per_volt_s": 0.5,
            "settle_base_s": 1.0,
            "ug1_step": 0.04,
        },
    )
    _write_json(cfg_dir / "health.json", {"ua_tolerance_v": 1.2})

    caplog.set_level(logging.WARNING, logger=app_config.__name__)
    cfg = app_config.load_app_config()

    assert cfg.health_ua_tolerance_v == pytest.approx(1.2)
    assert cfg.health_ug1_tolerance_v == pytest.approx(0.2)  # default
    out = capsys.readouterr().out
    assert "health file missing keys" in out
    assert any("health file missing keys" in rec.message for rec in caplog.records)


def test_health_weight_defaults_match_dataclass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """load_app_config fallback defaults must match AppConfig dataclass defaults.

    Pin: the ``.get()`` fallback values in ``load_app_config`` and the
    dataclass field defaults must stay in sync; divergence (e.g.
    weight_ia 0.30 vs 0.35) silently changes scoring depending on
    whether the user has a config file or not.
    """
    root = _set_fake_project_root(monkeypatch, tmp_path)
    cfg_dir = root / "config"
    _write_json(cfg_dir / "app.json", {})
    _write_json(cfg_dir / "scan.json", {
        "ua_settle_per_volt_s": 0.002, "ua_settle_base_s": 0.15,
        "ua_tolerance": 1.0, "ua_retries": 2,
        "ug1_settle_per_volt_s": 0.02, "ug1_settle_base_s": 0.15,
        "ug1_tolerance": 0.1, "ug1_retries": 2,
        "ug2_settle_per_volt_s": 0.002, "ug2_settle_base_s": 0.15,
        "ug2_tolerance": 1.0, "ug2_retries": 1,
        "ia_samples": 2, "pig2_over_pct": 20.0,
    })
    _write_json(cfg_dir / "srk.json", {
        "samples": 5, "settle_s": 1.0, "verify_retries": 3,
        "ua_tolerance": 1.0, "ug2_tolerance": 1.0,
        "settle_per_volt_s": 0.5, "settle_base_s": 1.0, "ug1_step": 0.04,
    })
    # health.json intentionally absent — load_app_config uses fallback defaults
    cfg = app_config.load_app_config()
    dc = app_config.AppConfig()

    weight_fields = [
        "health_weight_ia", "health_weight_s", "health_weight_rh",
        "health_weight_screen", "health_weight_emission",
    ]
    for field in weight_fields:
        assert getattr(cfg, field) == pytest.approx(getattr(dc, field)), (
            f"{field}: load_app_config default {getattr(cfg, field)} "
            f"!= AppConfig default {getattr(dc, field)}"
        )



# ── ML-054: cal_measure_* must be read from app.json ────────────────

def test_cal_measure_keys_read_from_app_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The fields existed in the dataclass and in app.json, but the loader
    never read them - user edits were silently ignored (ML-054)."""
    root = _set_fake_project_root(monkeypatch, tmp_path)
    _write_json(root / "config" / "app.json",
                {"cal_measure_samples": 25, "cal_measure_interval_ms": 500})
    cfg = app_config.load_app_config()
    assert cfg.cal_measure_samples == 25
    assert cfg.cal_measure_interval_ms == 500
    # Mutation-audit: int() -> float() survived (25.0 == 25).
    # The value feeds make_int_spinbox(value=...) - PySide6 setValue
    # raises TypeError on float, so the type IS load-bearing.
    assert type(cfg.cal_measure_samples) is int
    assert type(cfg.cal_measure_interval_ms) is int


# -- ML-134: generic loader — snapshot equivalence --------------------
# The table was extracted by an AST script from the pre-refactor
# load_app_config (139 fields, zero default drift). It doubles as the
# greppable field <-> (file, key) map replacing the loader's literals.

# Auto-extracted from the pre-ML-134 load_app_config.
_LOADER_SNAPSHOT = {
    'amp_opt_ra_max_abs_kohm': ('app', 'amp_opt_ra_max_abs_kohm'),
    'amp_opt_ra_max_factor': ('app', 'amp_opt_ra_max_factor'),
    'amp_opt_ra_min_abs_kohm': ('app', 'amp_opt_ra_min_abs_kohm'),
    'amp_opt_ra_min_factor': ('app', 'amp_opt_ra_min_factor'),
    'amp_opt_ra_steps': ('app', 'amp_opt_ra_steps'),
    'amp_sweep_amplitude_steps': ('app', 'amp_sweep_amplitude_steps'),
    'amp_sweep_ra_max_abs_kohm': ('app', 'amp_sweep_ra_max_abs_kohm'),
    'amp_sweep_ra_max_factor': ('app', 'amp_sweep_ra_max_factor'),
    'amp_sweep_ra_min_abs_kohm': ('app', 'amp_sweep_ra_min_abs_kohm'),
    'amp_sweep_ra_min_factor': ('app', 'amp_sweep_ra_min_factor'),
    'amp_sweep_ra_steps': ('app', 'amp_sweep_ra_steps'),
    'cal_measure_interval_ms': ('app', 'cal_measure_interval_ms'),
    'cal_measure_samples': ('app', 'cal_measure_samples'),
    'compare_matching_algorithm': ('app', 'compare_matching_algorithm'),
    'debug_params': ('app', 'debug_params'),
    'debug_params_file': ('app', 'debug_params_file'),
    'default_com_port': ('app', 'default_com_port'),
    'health_delta_pct': ('health', 'delta_pct'),
    'health_delta_ua_max_v': ('health', 'delta_ua_max_v'),
    'health_delta_ua_min_v': ('health', 'delta_ua_min_v'),
    'health_delta_ug1_max_v': ('health', 'delta_ug1_max_v'),
    'health_delta_ug1_min_v': ('health', 'delta_ug1_min_v'),
    'health_delta_ug2_max_v': ('health', 'delta_ug2_max_v'),
    'health_delta_ug2_min_v': ('health', 'delta_ug2_min_v'),
    'health_delta_ug2_pct': ('health', 'delta_ug2_pct'),
    'health_bias_servo_enabled_default': ('health', 'bias_servo_enabled_default'),
    'health_bias_servo_max_iter': ('health', 'bias_servo_max_iter'),
    'health_bias_servo_pa_ceiling_pct': ('health', 'bias_servo_pa_ceiling_pct'),
    'health_bias_servo_shift_margin': ('health', 'bias_servo_shift_margin'),
    'health_bias_servo_step_v': ('health', 'bias_servo_step_v'),
    'health_bias_servo_ug1_floor_v': ('health', 'bias_servo_ug1_floor_v'),
    'health_bias_servo_max_shift_v': ('health', 'bias_servo_max_shift_v'),
    'health_bias_servo_tol_ma': ('health', 'bias_servo_tol_ma'),
    'health_emission_knee_drop_pct': ('health', 'emission_knee_drop_pct'),
    'health_emission_min_ik_ratio': ('health', 'emission_min_ik_ratio'),
    'health_emission_mode_default': ('health', 'emission_mode_default'),
    'health_emission_sweep_max_total_s': ('health', 'emission_sweep_max_total_s'),
    'health_emission_uh_sweep_abs_min_ratio':
        ('health', 'emission_uh_sweep_abs_min_ratio'),
    'health_emission_uh_sweep_min_ratio': ('health', 'emission_uh_sweep_min_ratio'),
    'health_emission_uh_sweep_steps': ('health', 'emission_uh_sweep_steps'),
    'health_emission_enabled_default': ('health', 'emission_enabled_default'),
    'health_emission_low_confidence_on_timeout': ('health', 'emission_low_confidence_on_timeout'),
    'health_emission_ratio_good_min': ('health', 'emission_ratio_good_min'),
    'health_emission_ratio_nominal': ('health', 'emission_ratio_nominal'),
    'health_emission_ratio_weak_min': ('health', 'emission_ratio_weak_min'),
    'health_emission_restore_uh_timeout_s': ('health', 'emission_restore_uh_timeout_s'),
    'health_emission_sample_period_s': ('health', 'emission_sample_period_s'),
    'health_emission_stable_max_s': ('health', 'emission_stable_max_s'),
    'health_emission_stable_min_s': ('health', 'emission_stable_min_s'),
    'health_emission_stable_slope_threshold_ma_per_s': ('health', 'emission_stable_slope_threshold_ma_per_s'),
    'health_emission_stable_warmup_ratio': ('health', 'emission_stable_warmup_ratio'),
    'health_emission_stable_window_points': ('health', 'emission_stable_window_points'),
    'health_emission_uh_ratio': ('health', 'emission_uh_ratio'),
    'health_ia_sample_delay_ms': ('health', 'ia_sample_delay_ms'),
    'health_ia_samples': ('health', 'ia_samples'),
    'health_ig2_samples': ('health', 'ig2_samples'),
    'health_matching_algorithm': ('health', 'matching_algorithm'),
    'health_matching_anode': ('health', 'matching_anode'),
    'health_matching_bias_adjust_range_pct':
        ('health', 'matching_bias_adjust_range_pct'),
    'health_matching_group_size': ('health', 'matching_group_size'),
    'health_matching_max_delta': ('health', 'matching_max_delta'),
    'health_matching_max_iq_imbalance_pct':
        ('health', 'matching_max_iq_imbalance_pct'),
    'health_matching_protocol': ('health', 'matching_protocol'),
    'health_matching_use': ('health', 'matching_use'),
    'health_matching_weight_ia': ('health', 'matching_weight_ia'),
    'health_matching_weight_r': ('health', 'matching_weight_r'),
    'health_matching_weight_s': ('health', 'matching_weight_s'),
    'health_op_ramp_enabled': ('health', 'op_ramp_enabled'),
    'health_op_ug1_ramp_step_v': ('health', 'op_ug1_ramp_step_v'),
    'health_outlier_trim_count': ('health', 'outlier_trim_count'),
    'health_pa_safety_pct': ('health', 'pa_safety_pct'),
    'health_pig2_safety_pct': ('health', 'pig2_safety_pct'),
    'health_preheat_required_ratio': ('health', 'preheat_required_ratio'),
    'health_renormalize_weights_if_metric_missing': ('health', 'renormalize_weights_if_metric_missing'),
    'health_ua_retries': ('health', 'ua_retries'),
    'health_ua_settle_base_s': ('health', 'ua_settle_base_s'),
    'health_ua_settle_per_volt_s': ('health', 'ua_settle_per_volt_s'),
    'health_ua_tolerance_v': ('health', 'ua_tolerance_v'),
    'health_ug1_delta_v': ('health', 'ug1_delta_v'),
    'health_ug1_retries': ('health', 'ug1_retries'),
    'health_ug1_settle_base_s': ('health', 'ug1_settle_base_s'),
    'health_ug1_settle_per_volt_s': ('health', 'ug1_settle_per_volt_s'),
    'health_ug1_tolerance_v': ('health', 'ug1_tolerance_v'),
    'health_ug2_retries': ('health', 'ug2_retries'),
    'health_ug2_settle_base_s': ('health', 'ug2_settle_base_s'),
    'health_ug2_settle_per_volt_s': ('health', 'ug2_settle_per_volt_s'),
    'health_ug2_tolerance_v': ('health', 'ug2_tolerance_v'),
    'health_use_median': ('health', 'use_median'),
    'health_verdict_good_min': ('health', 'verdict_good_min'),
    'health_verdict_strong_min': ('health', 'verdict_strong_min'),
    'health_verdict_weak_min': ('health', 'verdict_weak_min'),
    'health_weight_emission': ('health', 'weight_emission'),
    'health_weight_ia': ('health', 'weight_ia'),
    'health_weight_rh': ('health', 'weight_rh'),
    'health_weight_s': ('health', 'weight_s'),
    'health_weight_screen': ('health', 'weight_screen'),
    'heater_zero_warn_ih_a': ('app', 'heater_zero_warn_ih_a'),
    'heater_zero_warn_uh_v': ('app', 'heater_zero_warn_uh_v'),
    'ia_dead_threshold': ('app', 'ia_dead_threshold'),
    'ig2_hw_margin_pct': ('app', 'ig2_hw_margin_pct'),
    'live_poll_during_test': ('app', 'live_poll_during_test'),
    'manual_heater_tolerance_pct': ('app', 'manual_heater_tolerance_pct'),
    'live_poll_ms': ('app', 'live_poll_ms'),
    'locale': ('app', 'locale'),
    'log_file': ('app', 'log_file'),
    'log_file_level': ('app', 'log_file_level'),
    'log_level': ('app', 'log_level'),
    'marker_lock_px': ('app', 'marker_lock_px'),
    'plot_ia_max': ('app', 'plot_ia_max'),
    'ra_dialog_max_abs_kohm': ('app', 'ra_dialog_max_abs_kohm'),
    'ra_dialog_max_factor': ('app', 'ra_dialog_max_factor'),
    'ra_dialog_min_abs_kohm': ('app', 'ra_dialog_min_abs_kohm'),
    'ra_dialog_steps': ('app', 'ra_dialog_steps'),
    'read_lcd_timeout_s': ('app', 'read_lcd_timeout_s'),
    'ltspice_exe': ('app', 'ltspice_exe'),
    'ltspice_verify_dir': ('app', 'ltspice_verify_dir'),
    'read_param_timeout_s': ('app', 'read_param_timeout_s'),
    'report_ask': ('app', 'report_ask'),
    'report_language': ('app', 'report_language'),
    'report_sections': ('app', 'report_sections'),
    'scan_comm_retries': ('scan', 'comm_retries'),
    'scan_down_max_step_v': ('scan', 'down_max_step_v'),
    'scan_ia_outlier_ratio': ('scan', 'ia_outlier_ratio'),
    'scan_ia_outlier_reread_samples': ('scan', 'ia_outlier_reread_samples'),
    'scan_ia_samples': ('scan', 'ia_samples'),
    'scan_pa_over_pct': ('scan', 'pa_over_pct'),
    'scan_pig2_over_pct': ('scan', 'pig2_over_pct'),
    'scan_refine_curvature_thr': ('scan', 'refine_curvature_thr'),
    'scan_refine_delta_ia_thr': ('scan', 'refine_delta_ia_thr'),
    'scan_refine_enabled': ('scan', 'refine_enabled'),
    'scan_refine_gradient_ratio': ('scan', 'refine_gradient_ratio'),
    'scan_refine_ig2_delta_min': ('scan', 'refine_ig2_delta_min'),
    'scan_refine_max_depth': ('scan', 'refine_max_depth'),
    'scan_refine_min_step_ua': ('scan', 'refine_min_step_ua'),
    'scan_refine_onset_ma': ('scan', 'refine_onset_ma'),
    'scan_ua_retries': ('scan', 'ua_retries'),
    'scan_ua_settle_base_s': ('scan', 'ua_settle_base_s'),
    'scan_ua_settle_per_volt_s': ('scan', 'ua_settle_per_volt_s'),
    'scan_ua_tolerance': ('scan', 'ua_tolerance'),
    'scan_ug1_retries': ('scan', 'ug1_retries'),
    'scan_ug1_settle_base_s': ('scan', 'ug1_settle_base_s'),
    'scan_ug1_settle_per_volt_s': ('scan', 'ug1_settle_per_volt_s'),
    'scan_ug1_tolerance': ('scan', 'ug1_tolerance'),
    'scan_ug2_retries': ('scan', 'ug2_retries'),
    'scan_ug2_settle_base_s': ('scan', 'ug2_settle_base_s'),
    'scan_ug2_settle_per_volt_s': ('scan', 'ug2_settle_per_volt_s'),
    'scan_ug2_tolerance': ('scan', 'ug2_tolerance'),
    'serial_set_param_delay_s': ('app', 'serial_set_param_delay_s'),
    'serial_timeout_s': ('app', 'serial_timeout_s'),
    'serial_write_timeout_s': ('app', 'serial_write_timeout_s'),
    'settle_s': ('app', 'settle_s'),
    'srk_samples': ('srk', 'samples'),
    'srk_settle_base_s': ('srk', 'settle_base_s'),
    'srk_settle_per_volt_s': ('srk', 'settle_per_volt_s'),
    'srk_settle_s': ('srk', 'settle_s'),
    'srk_ua_tolerance': ('srk', 'ua_tolerance'),
    'srk_ug1_step': ('srk', 'ug1_step'),
    'srk_ug2_tolerance': ('srk', 'ug2_tolerance'),
    'srk_verify_retries': ('srk', 'verify_retries'),
    'ua_cluster_threshold': ('app', 'ua_cluster_threshold'),
    'ug1_after_stop': ('app', 'ug1_after_stop'),
    'ug1_cluster_threshold': ('app', 'ug1_cluster_threshold'),
    'ug1_settle_s': ('app', 'ug1_settle_s'),
    'ug1_verify_tolerance': ('app', 'ug1_verify_tolerance'),
    'ug2_cluster_threshold': ('app', 'ug2_cluster_threshold'),
}


def _non_default_raw(f):
    """(raw value for JSON, expected field value) — guaranteed to
    differ from the dataclass default."""
    d = f.default
    if f.name == "ig2_hw_margin_pct":
        return 55, 55                    # inside clamp [10..90], != 80
    if f.name in ("log_level", "log_file_level"):
        return "debug", "DEBUG"
    if f.type == "bool":
        return (not d), (not d)
    if f.type == "int":
        return d + 7, d + 7
    if f.type == "float":
        return float(d) + 1.25, float(d) + 1.25
    if f.type == "str":
        return str(d) + "_x", str(d) + "_x"
    raise AssertionError(f"unhandled field type {f.type!r} ({f.name})")


class TestGenericLoaderEquivalence:
    """ML-134: defaults live ONLY in the dataclass. The ML-054 ratchet
    survived the refactor in a stronger form: not 'the field is
    mentioned in kwargs' but 'each field really loads from its own
    (file, key) and converts' — across all 139 fields."""

    def test_snapshot_covers_every_field(self) -> None:
        import dataclasses
        names = {f.name for f in dataclasses.fields(app_config.AppConfig)}
        assert names == set(_LOADER_SNAPSHOT), (
            "AppConfig fields diverged from _LOADER_SNAPSHOT — new field? "
            "Extend the snapshot consciously (file, key).")

    def test_routing_matches_pre_refactor_snapshot(self) -> None:
        for name, (label, key) in _LOADER_SNAPSHOT.items():
            assert app_config.config_file_and_key(name) == (label, key), name

    def test_converter_covers_all_field_types(self) -> None:
        import dataclasses
        for f in dataclasses.fields(app_config.AppConfig):
            assert (f.name in app_config._FIELD_TRANSFORMS
                    or f.type in app_config._FIELD_CONVERTERS), (
                f"{f.name}: type {f.type!r} has no converter")

    def test_all_keys_non_default_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Every key of every file set to a non-default -> each field
        must (a) differ from the default and (b) equal the expected
        value. Catches wrong file, key drift and lost conversion."""
        import dataclasses
        root = _set_fake_project_root(monkeypatch, tmp_path)
        files: dict = {"app": {}, "scan": {}, "srk": {}, "health": {}}
        expected: dict = {}
        for f in dataclasses.fields(app_config.AppConfig):
            label, key = _LOADER_SNAPSHOT[f.name]
            raw, exp = _non_default_raw(f)
            files[label][key] = raw
            expected[f.name] = exp
        for label, payload in files.items():
            _write_json(root / "config" / f"{label}.json", payload)

        cfg = app_config.load_app_config()
        dc = app_config.AppConfig()
        bad = []
        for f in dataclasses.fields(app_config.AppConfig):
            got = getattr(cfg, f.name)
            if got != expected[f.name]:
                bad.append(f"{f.name}: got {got!r}, want "
                           f"{expected[f.name]!r}")
            elif got == getattr(dc, f.name):
                bad.append(f"{f.name}: stuck at dataclass default")
        assert bad == [], bad

    def test_empty_config_dir_yields_dataclass_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        import dataclasses
        root = _set_fake_project_root(monkeypatch, tmp_path)
        (root / "config").mkdir(parents=True, exist_ok=True)
        cfg = app_config.load_app_config()
        dc = app_config.AppConfig()
        for f in dataclasses.fields(app_config.AppConfig):
            assert getattr(cfg, f.name) == getattr(dc, f.name), f.name

    def test_no_inline_defaults_ratchet(self) -> None:
        """Dedup ratchet: .get(key, literal) must not return to
        load_app_config — a second source of defaults is forbidden."""
        import ast
        import inspect
        import textwrap
        src = textwrap.dedent(
            inspect.getsource(app_config.load_app_config))
        offenders = []
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and len(n.args) >= 2
                    and isinstance(n.args[1], ast.Constant)):
                offenders.append(ast.unparse(n))
        assert offenders == [], (
            f"inline defaults are back in load_app_config: {offenders}")

    def test_clamp_and_upper_transforms_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Both clamp ends of ig2_hw_margin_pct + .upper() log levels."""
        root = _set_fake_project_root(monkeypatch, tmp_path)
        _write_json(root / "config" / "app.json",
                    {"ig2_hw_margin_pct": 5, "log_level": "debug",
                     "log_file_level": "info"})
        cfg = app_config.load_app_config()
        assert cfg.ig2_hw_margin_pct == 10      # lower end
        assert cfg.log_level == "DEBUG"
        assert cfg.log_file_level == "INFO"
        _write_json(root / "config" / "app.json", {"ig2_hw_margin_pct": 95})
        assert app_config.load_app_config().ig2_hw_margin_pct == 90  # upper


# ── AppConfig → ScanSettings wiring (app/main_window_scan.py) ─────────

# Scan fields that intentionally come from the UI rather than from the
# config defaults; everything else must be handed over from app_config.
_UI_SOURCED_SCAN_FIELDS = {
    "ia_samples",       # Samples spinbox in the Actions bar
    "refine_enabled",   # Refine checkbox
    "pa_over_pct",      # Pa over-% spinbox
    "pig2_over_pct",    # Pg2 over-% spinbox
}


class TestScanSettingsWiring:
    """Every ``scan_*`` config field with a ScanSettings twin reaches the scan.

    The field list is derived from the two dataclasses, not hand-written:
    a new ``scan_*`` field nobody wires up fails here instead of silently
    falling back to the ScanSettings default at runtime.
    """

    def _scan_settings_kwargs(self) -> dict:
        import ast as _ast
        src = (Path(__file__).resolve().parents[1] / "app"
               / "main_window_scan.py").read_text(encoding="utf-8")
        for node in _ast.walk(_ast.parse(src)):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Name)
                    and node.func.id == "ScanSettings"):
                return {kw.arg: _ast.unparse(kw.value)
                        for kw in node.keywords if kw.arg}
        raise AssertionError("ScanSettings(...) call not found")

    def _paired_fields(self) -> set:
        from dataclasses import fields
        from lm19.scan.settings import ScanSettings
        scan_names = {f.name for f in fields(ScanSettings)}
        return {f.name[len("scan_"):] for f in fields(app_config.AppConfig)
                if f.name.startswith("scan_")
                and f.name[len("scan_"):] in scan_names}

    def test_config_fields_reach_scan_settings(self) -> None:
        kwargs = self._scan_settings_kwargs()
        missing, wrong_source = [], []
        for name in sorted(self._paired_fields() - _UI_SOURCED_SCAN_FIELDS):
            if name not in kwargs:
                missing.append(name)
            elif f"scan_{name}" not in kwargs[name]:
                wrong_source.append((name, kwargs[name]))
        assert missing == [], f"not passed to ScanSettings: {missing}"
        assert wrong_source == [], (
            f"passed from the wrong config field: {wrong_source}")

    def test_ui_sourced_whitelist_is_not_stale(self) -> None:
        """A whitelisted field that became config-driven must leave the list."""
        kwargs = self._scan_settings_kwargs()
        stale = [n for n in sorted(_UI_SOURCED_SCAN_FIELDS)
                 if f"scan_{n}" in kwargs.get(n, "")]
        assert stale == [], f"now wired from config, drop from whitelist: {stale}"
