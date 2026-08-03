
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

from .calibration import CalibrationData

LOGGER = logging.getLogger(__name__)

SCAN_REQUIRED_KEYS = (
    "ua_settle_per_volt_s", "ua_settle_base_s", "ua_tolerance", "ua_retries",
    "ug1_settle_per_volt_s", "ug1_settle_base_s", "ug1_tolerance", "ug1_retries",
    "ug2_settle_per_volt_s", "ug2_settle_base_s", "ug2_tolerance", "ug2_retries",
    "ia_samples", "pa_over_pct", "pig2_over_pct",
)

SRK_REQUIRED_KEYS = (
    "samples", "settle_s", "verify_retries", "ua_tolerance", "ug2_tolerance",
    "settle_per_volt_s", "settle_base_s", "ug1_step",
)

HEALTH_REQUIRED_KEYS = (
    "ua_tolerance_v", "ug1_tolerance_v", "ug2_tolerance_v",
    "ua_retries", "ug1_retries", "ug2_retries",
    "ua_settle_per_volt_s", "ua_settle_base_s",
    "ug1_settle_per_volt_s", "ug1_settle_base_s",
    "ug2_settle_per_volt_s", "ug2_settle_base_s",
    "ug1_delta_v", "ia_samples", "ig2_samples", "use_median",
    "emission_enabled_default", "emission_uh_ratio",
    "emission_stable_warmup_ratio", "emission_stable_min_s", "emission_stable_max_s",
    "preheat_required_ratio",
    "weight_ia", "weight_s", "weight_rh", "weight_screen", "weight_emission",
    "verdict_strong_min", "verdict_good_min", "verdict_weak_min",
)


@dataclass
class AppConfig:
    locale: str = "en"
    settle_s: float = 0.2
    default_com_port: str = ""
    live_poll_ms: int = 500
    live_poll_during_test: bool = True
    manual_heater_tolerance_pct: float = 10.0
    serial_timeout_s: float = 0.3
    serial_write_timeout_s: float = 0.3
    read_param_timeout_s: float = 1.0
    read_lcd_timeout_s: float = 2.0
    serial_set_param_delay_s: float = 0.05
    plot_ia_max: float = 300.0
    ug1_after_stop: float = -24.0
    ug1_settle_s: float = 0.3
    ug1_verify_tolerance: float = 0.2
    srk_samples: int = 5
    srk_settle_s: float = 1.0
    srk_verify_retries: int = 3
    srk_ua_tolerance: float = 2.0
    srk_ug2_tolerance: float = 2.0
    srk_settle_per_volt_s: float = 1.0
    srk_settle_base_s: float = 1.0
    srk_ug1_step: float = 0.0
    # srk_ug1_step: Ug1 step for SRK sweep mode (V). 0 = classic 2-point.
    # Min reliable: 0.04V (≥1 ADC LSB per step on ATmega16 10-bit ADC).
    # 0.08V = ironclad (≥2 LSB). < 0.04V may give identical ug1set values.
    scan_ua_settle_per_volt_s: float = 0.0025
    scan_ua_settle_base_s: float = 0.15
    scan_ua_tolerance: float = 1.0
    scan_ua_retries: int = 2
    scan_ug1_settle_per_volt_s: float = 0.02
    scan_ug1_settle_base_s: float = 0.15
    scan_ug1_tolerance: float = 0.1
    scan_ug1_retries: int = 2
    scan_ug2_settle_per_volt_s: float = 0.0025
    scan_ug2_settle_base_s: float = 0.15
    scan_ug2_tolerance: float = 1.0
    scan_ug2_retries: int = 1
    scan_ia_samples: int = 3
    scan_ia_outlier_ratio: float = 2.0
    scan_ia_outlier_reread_samples: int = 3
    scan_pa_over_pct: float = 20.0
    scan_pig2_over_pct: float = 20.0
    scan_comm_retries: int = 2
    # Adaptive refine (two-pass)
    scan_refine_enabled: bool = False
    scan_refine_max_depth: int = 2
    scan_refine_min_step_ua: float = 3.0
    scan_refine_onset_ma: float = 0.5
    scan_refine_curvature_thr: float = 0.15
    scan_refine_gradient_ratio: float = 3.0
    scan_refine_ig2_delta_min: float = 0.5
    scan_refine_delta_ia_thr: float = 0.25
    scan_down_max_step_v: float = 25.0
    ig2_hw_margin_pct: int = 80
    heater_zero_warn_uh_v: float = 0.1
    heater_zero_warn_ih_a: float = 0.1
    amp_sweep_amplitude_steps: int = 40
    amp_sweep_ra_min_factor: float = 0.2
    amp_sweep_ra_max_factor: float = 5.0
    amp_sweep_ra_min_abs_kohm: float = 0.5
    amp_sweep_ra_max_abs_kohm: float = 100.0
    amp_sweep_ra_steps: int = 60
    amp_opt_ra_min_factor: float = 0.1
    amp_opt_ra_max_factor: float = 10.0
    amp_opt_ra_min_abs_kohm: float = 0.5
    amp_opt_ra_max_abs_kohm: float = 200.0
    amp_opt_ra_steps: int = 100
    ra_dialog_max_factor: float = 4.0
    ra_dialog_max_abs_kohm: float = 20.0
    ra_dialog_min_abs_kohm: float = 0.5
    ra_dialog_steps: int = 80
    marker_lock_px: int = 15
    ua_cluster_threshold: float = 2.0
    ug1_cluster_threshold: float = 0.3
    ug2_cluster_threshold: float = 2.0
    cal_measure_samples: int = 10
    cal_measure_interval_ms: int = 200
    # Tube Health config (config/health.json)
    health_ua_tolerance_v: float = 1.0
    health_ug1_tolerance_v: float = 0.2
    health_ug2_tolerance_v: float = 1.0
    health_ua_retries: int = 2
    health_ug1_retries: int = 2
    health_ug2_retries: int = 2
    health_ua_settle_per_volt_s: float = 0.0025
    health_ua_settle_base_s: float = 0.15
    health_ug1_settle_per_volt_s: float = 0.2
    health_ug1_settle_base_s: float = 0.3
    health_ug2_settle_per_volt_s: float = 0.0025
    health_ug2_settle_base_s: float = 0.15
    health_ug1_delta_v: float = 1.0
    health_delta_pct: int = 10
    health_delta_ua_min_v: float = 10.0
    health_delta_ua_max_v: float = 50.0
    health_delta_ug1_min_v: float = 0.5
    health_delta_ug1_max_v: float = 2.0
    health_delta_ug2_pct: int = 5
    health_delta_ug2_min_v: float = 5.0
    health_delta_ug2_max_v: float = 50.0
    health_ia_samples: int = 5
    health_ia_sample_delay_ms: int = 50
    health_ig2_samples: int = 5
    health_use_median: bool = True
    health_outlier_trim_count: int = 1
    health_emission_enabled_default: bool = True
    health_preheat_required_ratio: float = 0.75
    health_emission_uh_ratio: float = 0.8
    health_emission_restore_uh_timeout_s: int = 20
    health_emission_low_confidence_on_timeout: bool = True
    health_emission_stable_warmup_ratio: float = 0.50
    health_emission_stable_min_s: int = 20
    health_emission_stable_max_s: int = 120
    health_emission_stable_slope_threshold_ma_per_s: float = 0.01
    health_emission_stable_window_points: int = 5
    health_emission_sample_period_s: float = 2.0
    health_weight_ia: float = 0.35
    health_weight_s: float = 0.40
    health_weight_rh: float = 0.10
    health_weight_screen: float = 0.0
    health_weight_emission: float = 0.15
    health_renormalize_weights_if_metric_missing: bool = True
    health_verdict_strong_min: float = 90.0
    health_verdict_good_min: float = 75.0
    health_verdict_weak_min: float = 55.0
    health_emission_ratio_nominal: float = 0.90
    health_emission_ratio_good_min: float = 0.70
    health_emission_ratio_weak_min: float = 0.50
    # Emission-probe sensitivity: below this fraction of Ik_max the tube
    # stays space-charge limited even with a depleted cathode, so the
    # ratio stops discriminating and the result is flagged.
    health_emission_min_ik_ratio: float = 0.30
    # Deep-emission (Ia vs Uh sweep) — locates the space-charge knee.
    health_emission_mode_default: str = "single"
    health_emission_uh_sweep_steps: int = 5
    health_emission_uh_sweep_min_ratio: float = 0.70
    # Adaptive Miram descent: past the configured grid the sweep keeps
    # stepping down until the falling branch holds two points — but never
    # below this absolute floor. Below ~50% the cathode sits in deep
    # emission limitation (the thinned space-charge cloud stops shielding
    # it from ion bombardment) and Ia is too small to measure anyway.
    health_emission_uh_sweep_abs_min_ratio: float = 0.50
    health_emission_knee_drop_pct: float = 10.0
    health_emission_sweep_max_total_s: float = 600.0
    # Bias servo — drive Ia to the reference current before measuring S.
    health_bias_servo_enabled_default: bool = False
    health_bias_servo_tol_ma: float = 0.5
    # Absolute excursion ceiling. The WORKING limit is per-tube:
    # margin * |deficit| / lamp.s, capped by this value — 6 V covers a
    # 6L6-class tetrode at 50% wear, while a 12AX7 auto-limits to ~0.5 V.
    health_bias_servo_max_shift_v: float = 6.0
    health_bias_servo_shift_margin: float = 2.0
    # Walk step: overshoot is bounded by S*step (~6 mA on an EL84).
    # The servo must approach the reference gradually — commanding the
    # excursion limit in one move drives a steep tube into the Pa trip.
    health_bias_servo_step_v: float = 0.5
    health_bias_servo_max_iter: int = 12
    # Stop walking up once Pa exceeds this % of the protection trip
    # limit — the trip stays a backstop, never the expected stop.
    health_bias_servo_pa_ceiling_pct: float = 90.0
    # Bisection stops when the bracket is below what the hardware can
    # actually set (device Ug1 resolution + settle tolerance).
    health_bias_servo_ug1_floor_v: float = 0.1
    # OP-approach soft start (Ug1 ramp from safe lock to target with
    # Pa/Pg2 protection after each step).
    health_op_ramp_enabled: bool = True
    health_op_ug1_ramp_step_v: float = 1.0
    health_pa_safety_pct: float = 135.0
    health_pig2_safety_pct: float = 120.0
    # Tube matching defaults
    health_matching_weight_ia: float = 0.5
    health_matching_weight_s: float = 0.5
    health_matching_weight_r: float = 0.0
    health_matching_max_delta: float = 0.0
    health_matching_group_size: int = 2
    health_matching_use: str = "latest"
    health_matching_anode: str = "each"
    health_matching_algorithm: str = "greedy"  # "greedy" | "optimal"
    # Matching protocol — what "matched" means for the buyer's amplifier
    # (lm19.tube_matching.MATCHING_PROTOCOLS): "strict" (servo and
    # fixed-bias runs never mix), "shared_bias" (one bias adjustment for
    # both tubes: plan-point Ia + δIq gate), "individual_bias" (per-tube
    # bias: S/R at reference current + adjustment-range gate).
    health_matching_protocol: str = "strict"
    # shared_bias gate: max predicted quiescent-current imbalance as a
    # percent of the pair's mean plan current (0 = off).
    health_matching_max_iq_imbalance_pct: float = 10.0
    # individual_bias gate: the amplifier's bias-adjustment authority as
    # a percent of the plan bias voltage (0 = off). Percent, not volts —
    # a fixed volt span would be huge for a −2 V high-mu triode and
    # negligible for a −60 V transmitting tube.
    health_matching_bias_adjust_range_pct: float = 30.0
    ia_dead_threshold: float = 0.30  # mA — dead-data detection threshold
    # Compare tab curve-matching algorithm (separate from Health).
    # ``optimal`` (Hungarian) preserves legacy behaviour; user can
    # switch to ``greedy`` via the Compare-tab dropdown.
    compare_matching_algorithm: str = "optimal"
    # PDF report export defaults (config/app.json). Dialog choices are
    # remembered per-session only — config files are read-only for the UI
    # (same pattern as health_matching_algorithm).
    report_language: str = ""  # "" = follow app locale; else a locales/*.json code
    report_sections: str = ""  # CSV of enabled section ids; "" = section defaults
    report_ask: bool = True    # show the report-options dialog before PDF export
    # LTspice verification working directory: "" = system temp; else a
    # per-run verify_<timestamp> subdir is created under it (relative
    # paths resolve against lm19_app/, same rule as measurements_dir).
    ltspice_verify_dir: str = ""
    # LTspice executable override: "" = standard install path
    # (lm19/ltspice_raw.py:LTSPICE_EXE).
    ltspice_exe: str = ""
    debug_params: bool = False
    debug_params_file: str = "logs/params_debug.log"
    log_file: str = "logs/lm19.log"
    log_level: str = "WARNING"
    log_file_level: str = ""


def _warn_config(message: str) -> None:
    text = f"[config] {message}"
    print(text)
    LOGGER.warning(text)


def _load_json_or_empty(path: Path, label: str, required_keys: tuple[str, ...] = ()) -> Dict[str, Any]:
    if not path.exists():
        _warn_config(f"Missing {label} file: {path}. Defaults will be used.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _warn_config(f"Failed to parse {label} file: {path}. Error: {exc}. Defaults will be used.")
        return {}
    missing = [k for k in required_keys if k not in data]
    if missing:
        _warn_config(
            f"{label} file missing keys ({len(missing)}): {', '.join(missing)}. "
            "Defaults will be used for missing values."
        )
    return data


# ── ML-134: generic loader — defaults live ONLY in the dataclass ────
# Routing rule: field-name prefix → config file, JSON key = field name
# minus the prefix; everything else reads app.json under the field name.
# Pre-refactor pre-flight check: all 139 fields map 1:1 and no
# loader default diverged from its dataclass default. The full mapping
# is pinned in tests/test_app_config_split.py::TestGenericLoaderEquivalence.
_CONFIG_FILE_PREFIXES: tuple = (
    ("scan_", "scan"),
    ("srk_", "srk"),
    ("health_", "health"),
)
_FIELD_CONVERTERS: Dict[str, Any] = {
    "str": str, "float": float, "int": int, "bool": bool,
}
# Historical per-field conversions (behaviour-preserving overrides).
_FIELD_TRANSFORMS: Dict[str, Any] = {
    "ig2_hw_margin_pct": lambda v: max(10, min(90, int(v))),
    "log_level": lambda v: str(v).upper(),
    "log_file_level": lambda v: str(v).upper(),
}


def config_file_and_key(field_name: str) -> tuple[str, str]:
    """Route an AppConfig field to its (config file label, JSON key)."""
    for prefix, label in _CONFIG_FILE_PREFIXES:
        if field_name.startswith(prefix):
            return label, field_name[len(prefix):]
    return "app", field_name


def load_app_config() -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "config"
    data: Dict[str, Dict[str, Any]] = {
        "app": _load_json_or_empty(config_dir / "app.json", "app"),
        "scan": _load_json_or_empty(
            config_dir / "scan.json", "scan", SCAN_REQUIRED_KEYS),
        "srk": _load_json_or_empty(
            config_dir / "srk.json", "srk", SRK_REQUIRED_KEYS),
        "health": _load_json_or_empty(
            config_dir / "health.json", "health", HEALTH_REQUIRED_KEYS),
    }
    kwargs: Dict[str, Any] = {}
    for f in fields(AppConfig):
        label, key = config_file_and_key(f.name)
        if key not in data[label]:
            continue  # dataclass default applies — the single source
        convert = (_FIELD_TRANSFORMS.get(f.name)
                   or _FIELD_CONVERTERS.get(f.type))
        if convert is None:
            # Programming error: a new field with an unsupported
            # annotation must extend _FIELD_CONVERTERS consciously
            # (CI pin: test_converter_covers_all_field_types).
            raise TypeError(
                f"AppConfig.{f.name}: unsupported field type {f.type!r} "
                "for the generic loader — extend _FIELD_CONVERTERS")
        kwargs[f.name] = convert(data[label][key])
    return AppConfig(**kwargs)

def calibration_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "config" / "calibration.json"


def load_calibration() -> CalibrationData:
    return CalibrationData.load(calibration_path())
