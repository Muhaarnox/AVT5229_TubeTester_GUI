"""Pure planning logic for health-test step preview.

Qt-free function that builds the list consumed by
``HealthTab._refresh_planned_info`` to populate the planned-steps
preview table — ``ua/ug1/ug2/uh/ih`` per step plus a localized
``step``/``details`` label. Lives outside the widget so it can be
unit-tested directly and reused for CLI/batch processing.
"""

from __future__ import annotations

from typing import Dict, List

from i18n_setup import t

from lm19.config import LampConfig
from lm19.constants import EPS_COARSE
from lm19.health import compute_shifted_r_center, compute_shifted_sg2_center
from lm19.label_formats import format_label


# ── module local constants ──
# Default deltas / point counts when the plan dict is missing values.
_DEFAULT_DELTA_UA_V = 25.0
_DEFAULT_DELTA_UG1_V = 0.84
_DEFAULT_DELTA_UG2_V = 13.0
_DEFAULT_POINTS = 5


def compute_planned_steps(
    *,
    plan: Dict,
    lamp: LampConfig,
    emission_enabled: bool,
    uh_ratio_default: float,
) -> List[Dict]:
    """Build the list of measurement steps planned for a health test.

    Pure function: no Qt dependencies, no I/O. The returned list is the
    UI's preview of what ``run_health_test`` will execute given the
    user-configured ``plan`` dict.

    Args:
        plan: Output of ``HealthTab._collect_measurement_plan()`` — a
            dict with ``op`` (Ua/Ug1/Ug2 targets), ``srk`` (deltas,
            points, repeats), ``emission`` (uh_ratio), ``ug2_track_ua``
            (bool), ``ug2_offset`` (float).
        lamp: ``LampConfig`` for the tube under test. Used for fallback
            target voltages and heater values (``uh``/``ih``).
        emission_enabled: True when emission test is requested. Adds two
            extra steps (Ia at 100% Uh, then at ``uh_ratio_default`` Uh).
        uh_ratio_default: Default ratio for emission Uh reduction
            (typically 0.8). Overridden by ``plan['emission']['uh_ratio']``
            when present.

    Returns:
        List of step dicts. Each step has keys:
          - ``step``: localized label ("OP", "S-", "S+", "R-", "R+",
            "Sg2-", "Sg2+", "Em100%", "Em80%")
          - ``ua``, ``ug1``, ``ug2``: target voltages (V)
          - ``uh``: heater voltage (V) or ``None`` (current-driven heater)
          - ``ih``: heater current (A) or ``None`` (voltage-driven heater)
          - ``details``: localized human-readable description
    """
    op = plan.get("op", {})
    srk = plan.get("srk", {})
    emission = plan.get("emission", {})
    ug2_track = bool(plan.get("ug2_track_ua", False))
    ug2_off = float(plan.get("ug2_offset", 0.0))

    if ug2_track and not lamp.is_triode:
        target_ug2 = max(0.0, float(op.get("ua", lamp.ua)) + ug2_off)
    else:
        target_ug2 = 0.0 if lamp.is_triode else float(op.get("ug2", lamp.ug2))

    ua0 = float(op.get("ua", lamp.ua))
    ug1_0 = float(op.get("ug1", lamp.ug1))
    d_ua = float(srk.get("delta_ua", _DEFAULT_DELTA_UA_V))
    d_ug1 = float(srk.get("delta_ug1", _DEFAULT_DELTA_UG1_V))
    d_ug2 = float(srk.get("delta_ug2", _DEFAULT_DELTA_UG2_V))
    pts = int(srk.get("points", _DEFAULT_POINTS))
    n_ug1 = pts - 2
    uh_ratio = float(emission.get("uh_ratio", uh_ratio_default))

    def _ug2_for_ua(ua: float) -> float:
        if lamp.is_triode:
            return 0.0
        if ug2_track:
            return max(0.0, ua + ug2_off)
        return target_ug2

    def _heater(ratio: float = 1.0) -> Dict:
        return {
            "uh": (lamp.uh * ratio) if lamp.uh > 0 else None,
            "ih": (lamp.ih * ratio) if lamp.uh <= 0 and lamp.ih > 0 else None,
        }

    steps: List[Dict] = [{
        "step": t("health.step_op"),
        "ua": ua0, "ug1": ug1_0, "ug2": _ug2_for_ua(ua0),
        **_heater(), "details": t("health.step_op_details"),
    }]

    # ── S sweep (Gm) ──
    # Two-point (S-/S+) when only 3 Ug1 points planned, otherwise an
    # evenly-spaced sweep S1..Sn skipping the centre point (which equals OP).
    if n_ug1 <= 3:
        steps.append({"step": "S-", "ua": ua0, "ug1": ug1_0 - d_ug1,
                      "ug2": _ug2_for_ua(ua0), **_heater(),
                      "details": t("health.step_gm_minus_details")})
        steps.append({"step": "S+", "ua": ua0, "ug1": ug1_0 + d_ug1,
                      "ug2": _ug2_for_ua(ua0), **_heater(),
                      "details": t("health.step_gm_plus_details")})
    else:
        step_v = 2.0 * d_ug1 / (n_ug1 - 1)
        for i in range(n_ug1):
            ug1_i = ug1_0 - d_ug1 + i * step_v
            if abs(ug1_i - ug1_0) < EPS_COARSE:
                continue
            steps.append({"step": f"S{i+1}", "ua": ua0, "ug1": round(ug1_i, 4),
                          "ug2": _ug2_for_ua(ua0), **_heater(),
                          "details": t("health.step_gm_sweep_details",
                                       ug1=format_label("ug1_value", ug1_i))})

    # ── R sweep (plate resistance) ──
    # ``compute_shifted_r_center`` may shift the center off ua0 if the
    # nominal ±d_ua range hits the device limit; ``shifted_op`` flag
    # surfaces that to the user via ``step_shifted_suffix``.
    r_center, r_method = compute_shifted_r_center(ua0, d_ua)
    r_detail_suffix = t("health.step_shifted_suffix") if r_method == "shifted_op" else ""
    steps.append({"step": "R-", "ua": r_center - d_ua, "ug1": ug1_0,
                  "ug2": _ug2_for_ua(r_center - d_ua), **_heater(),
                  "details": t("health.step_r_minus_details", suffix=r_detail_suffix)})
    steps.append({"step": "R+", "ua": r_center + d_ua, "ug1": ug1_0,
                  "ug2": _ug2_for_ua(r_center + d_ua), **_heater(),
                  "details": t("health.step_r_plus_details", suffix=r_detail_suffix)})

    # ── Sg2 sweep (pentode-only, independent Ug2) ──
    is_pentode_mode = not lamp.is_triode and not ug2_track
    if is_pentode_mode:
        sg2_c, sg2_m = compute_shifted_sg2_center(target_ug2, d_ug2)
        sg2_suffix = t("health.step_shifted_suffix") if sg2_m == "shifted_op" else ""
        steps.append({"step": "Sg2-", "ua": ua0, "ug1": ug1_0,
                      "ug2": sg2_c - d_ug2, **_heater(),
                      "details": t("health.step_sg2_minus_details", suffix=sg2_suffix)})
        steps.append({"step": "Sg2+", "ua": ua0, "ug1": ug1_0,
                      "ug2": sg2_c + d_ug2, **_heater(),
                      "details": t("health.step_sg2_plus_details", suffix=sg2_suffix)})

    # ── Emission test (optional) ──
    if emission_enabled:
        steps.append({"step": t("health.step_emission_100"),
                      "ua": ua0, "ug1": ug1_0, "ug2": _ug2_for_ua(ua0),
                      **_heater(),
                      "details": t("health.step_emission_100_details")})
        steps.append({"step": t("health.step_emission_80"),
                      "ua": ua0, "ug1": ug1_0, "ug2": _ug2_for_ua(ua0),
                      **_heater(uh_ratio),
                      "details": t("health.step_emission_80_details",
                                   ratio=f"{uh_ratio * 100:.0f}")})

    return steps
