"""HTML formatting for amplifier source results.

Pure formatter — no Qt dependencies. Takes a ``SourceResult`` plus
``pa_max`` / ``circuit`` / ``ug2_filter`` and emits the localized HTML
fragment shown in the Amplifier tab's results panel. Lives outside
``AmplifierTab`` so the format can be unit-tested in isolation and
reused outside the tab class.
"""

from __future__ import annotations

from typing import List, Optional

from i18n_setup import t
from lm19.amp_engine import SourceResult
from lm19.constants import MW_PER_W
from app.ui_theme import (
    AMP_WARNING_HTML_COLOR,
    COLOR_MUTED_TEXT,
    HEADROOM_GOOD_MIN,
    HEADROOM_WARN_MIN,
    PA_RATIO_GOOD_MAX,
    PA_RATIO_WARN_MAX,
    THD_GOOD_MAX,
    THD_WARN_MAX,
    level_color_high_good,
    level_color_low_good,
)
from lm19.amplifier.constants import (
    HD_METHOD_CHEBYSHEV,
)


# ── module local constants ──
HD45_DISPLAY_THRESHOLD_PCT = 0.1   # show HD4/HD5 in text if above this
POUT_W_DECIMALS = 3                # decimal places for Pout in watts
PERCENT = 100.0                    # multiplication factor for percentage


def _fmt_pout_w(pout_mw: float) -> str:
    """Format Pout from mW to W with consistent precision."""
    return f"{pout_mw / MW_PER_W:.{POUT_W_DECIMALS}f}"


def _format_method_tag(stage: dict) -> str:
    """Format method tag including SRK cross-check status."""
    srk_check = stage.get("srk_check")
    if srk_check == "ok":
        return t("amp.method_srk_ok")
    if srk_check == "divergence":
        pct = stage.get("srk_divergence_pct", 0)
        return t("amp.method_srk_divergence", pct=f"{pct:.0f}")
    return f"[{stage.get('method', 'numerical')}]"


def warning_text(w: dict) -> str:
    """Translate one SourceResult.warnings entry to localized text.

    Unknown codes fall back to the raw code string — a new engine code
    without an i18n key must stay visible, not vanish (failure-visibility).
    """
    code = w.get("code", "")
    params = {k: v for k, v in w.items() if k != "code"}
    key = f"amp.warn_{code}"
    translated = t(key, **params)
    return translated if translated != key else code


def format_warnings_html(warnings: List[dict]) -> List[str]:
    """Render SourceResult.warnings as ⚠ lines for the results panel."""
    return [
        f'<span style="color:{AMP_WARNING_HTML_COLOR}">⚠ '
        f"{warning_text(w)}</span>"
        for w in warnings
    ]


def _append_swing_lines(parts: List[str], dist: dict) -> None:
    """Operating-point swing block:

    - Ua_pp with min–max — the actual anode voltage swing (for se_xfmr it
      exceeds Ub; the numbers existed in every method's dict, unshown);
    - Ia min–max — cutoff margin: i_min→0 is WHY a point is class AB;
    - grid drive in Vpp — the driver-stage requirement;
    - P1 fundamental power (DFT) — what an external meter would read
      (peak-based Pout overstates under compression);
    - PP: per-tube quiescent current (bias is set, Iq is the result).
    """
    ua_min, ua_max = dist.get("ua_min"), dist.get("ua_max")
    if ua_min is not None and ua_max is not None:
        parts.append(t("amp.swing_ua_line",
                       upp=f"{abs(ua_max - ua_min):.0f}",
                       lo=f"{min(ua_min, ua_max):.0f}",
                       hi=f"{max(ua_min, ua_max):.0f}"))
    i_min, i_max = dist.get("i_min"), dist.get("i_max")
    if i_min is not None and i_max is not None:
        parts.append(t("amp.swing_ia_line",
                       lo=f"{min(i_min, i_max):.1f}",
                       hi=f"{max(i_min, i_max):.1f}"))
    hs = dist.get("half_swing")
    if hs:
        parts.append(t("amp.drive_line", half=f"{hs:.1f}",
                       vpp=f"{2 * hs:.1f}"))
    p1 = dist.get("pout_fund_mw")
    if p1 is not None:
        parts.append(t("amp.p1_line", p1=_fmt_pout_w(p1)))
    iq_pt = dist.get("iq_per_tube")
    if iq_pt is not None:
        parts.append(t("amp.iq_per_tube_line", iq=f"{iq_pt:.1f}"))


def dist_error_html(code: Optional[str],
                    params: Optional[dict] = None) -> str:
    """Translate a diagnose_distortion code to a localized HTML message.

    Returns the generic analysis_failed message if code is missing or
    unknown. Codes match DIST_ERR_* constants from lm19.amplifier.
    ``params`` (``SourceResult.dist_error_params``) selects the richer
    ``…_range`` key variant — e.g. the bias diagnostics then NAME the
    measured Ug1 span instead of just saying "adjust Ug1".
    """
    if not code:
        return t("amp.analysis_failed")
    if params:
        key_range = f"amp.dist_err_{code}_range"
        translated = t(key_range,
                       **{k: (v if isinstance(v, str) else f"{v:.1f}")
                          for k, v in params.items()})
        if translated != key_range:
            return translated
    key = f"amp.dist_err_{code}"
    translated = t(key)
    # t() returns the key itself when missing — fall back to generic msg.
    if translated == key:
        return t("amp.analysis_failed")
    return translated


def format_source_results(
    sr: SourceResult,
    pa_max: float,
    circuit: str,
    ug2_filter: Optional[float] = None,
) -> str:
    """Format a single SourceResult as HTML.

    Pure function: takes the SourceResult dataclass + amplifier params
    and returns the HTML fragment rendered in the Amplifier results
    panel. Falls back to ``dist_error_html(sr.dist_error)`` when no
    distortion data is available.
    """
    parts: List[str] = []
    parts.extend(format_warnings_html(sr.warnings))
    dist = sr.dist

    if not dist:
        parts.append(dist_error_html(
            sr.dist_error, getattr(sr, "dist_error_params", None)))
        return "<br>".join(parts)

    if dist.get("manual_swing_clamped"):
        parts.append(
            t(
                "amp.swing_clamped",
                requested=f"{dist.get('requested_half_swing', 0.0):.1f}",
                used=f"{dist.get('half_swing', 0.0):.1f}",
            )
        )
    if dist.get("insufficient_signal"):
        parts.append(t("amp.insufficient_signal"))

    pa_w = dist["ua_0"] * dist["ia_0"] / MW_PER_W
    pa_ratio = pa_w / pa_max if pa_max > 0 else 0.0
    pa_color = level_color_low_good(pa_ratio, PA_RATIO_GOOD_MAX, PA_RATIO_WARN_MAX)
    thd_color = level_color_low_good(dist["thd"], THD_GOOD_MAX, THD_WARN_MAX)

    q_line = t(
        "amp.q_point_line",
        ua=f"{dist['ua_0']:.0f}",
        ia=f"{dist['ia_0']:.1f}",
        ug1=f"{dist['ug1_0']:.1f}",
        cls=dist.get("amp_class", "?"),
    )
    if ug2_filter is not None and ug2_filter > 0:
        q_line += f"  Ug2={ug2_filter:.0f}V"
    parts.append(q_line)
    if not dist.get("insufficient_signal"):
        parts.append(
            t(
                "amp.thd_line",
                hd2=f"{dist.get('hd2', 0):.1f}",
                hd3=f"{dist.get('hd3', 0):.2f}",
                thd=f"{dist.get('thd', 0):.2f}",
                pout=_fmt_pout_w(dist.get('pout_mw', 0)),
                thd_color=thd_color,
            )
        )
        # HD4/HD5 always shown if above threshold
        hd45_parts = []
        hd4 = dist.get("hd4", 0.0)
        hd5 = dist.get("hd5", 0.0)
        if hd4 >= HD45_DISPLAY_THRESHOLD_PCT:
            hd45_parts.append(f"HD4={hd4:.2f}%")
        if hd5 >= HD45_DISPLAY_THRESHOLD_PCT:
            hd45_parts.append(f"HD5={hd5:.2f}%")
        if hd45_parts:
            parts.append(
                f"<span style='color:{COLOR_MUTED_TEXT}'>"
                f"{'  '.join(hd45_parts)}</span>"
            )

        # Chebyshev harmonic limit notice
        actual_max = dist.get("max_harmonic")
        if dist.get("method") == HD_METHOD_CHEBYSHEV and actual_max is not None and actual_max < 9:
            parts.append(
                f"<span style='color:{COLOR_MUTED_TEXT}'>"
                f"{t('amp.chebyshev_limited', n=actual_max)}</span>"
            )

        # HD method actually used (engine resolves auto/dft/etc).
        # Surfaces fallbacks (e.g., DFT requested but no model →
        # 5-point) so the user knows what generated this THD.
        method_used = dist.get("method") or sr.method_used
        if method_used:
            parts.append(
                f"<span style='color:{COLOR_MUTED_TEXT}'>"
                f"{t('amp.opt_method_used', method=method_used)}</span>"
            )

    if not dist.get("insufficient_signal"):
        _append_swing_lines(parts, dist)

    if sr.imd:
        parts.append(t("amp.imd_line", imd2=f"{sr.imd['imd2']:.1f}", imd3=f"{sr.imd['imd3']:.2f}"))

    if sr.headroom:
        hr_color = level_color_high_good(
            sr.headroom["max_swing"], HEADROOM_GOOD_MIN, HEADROOM_WARN_MIN
        )
        swing_value = f"{sr.headroom['max_swing']:.1f}"
        clip_pos = sr.headroom["clip_pos"]
        ig1 = sr.headroom.get("ig1_ma")
        if ig1 is not None and ig1 > 0.001:
            clip_pos = f"Ig1={ig1:.2f}mA"
        parts.append(
            f"<span style='color:{hr_color}'>"
            f"{t('amp.headroom_short', swing=swing_value)}</span> "
            f"{t('amp.clip_note', neg=sr.headroom['clip_neg'], pos=clip_pos)}"
        )

    if sr.stage:
        method_tag = _format_method_tag(sr.stage)
        df_str = f"{sr.stage['df']:.1f}" if sr.stage.get("df") is not None else "N/A"
        parts.append(
            t(
                "amp.gain_line",
                gain=f"{sr.stage['gain']:.1f}",
                gain_db=f"{sr.stage['gain_db']:.1f}",
                zout=f"{sr.stage['zout']:.1f}",
                df=df_str,
                method=method_tag,
                muted_color=COLOR_MUTED_TEXT,
            )
        )
    else:
        parts.append(
            f"<span style='color:{COLOR_MUTED_TEXT}'>{t('amp.gain_na')}</span>"
        )

    if sr.nfb and not dist.get("insufficient_signal"):
        df_nfb = "N/A"
        if sr.stage and sr.stage.get("df") is not None:
            df_nfb = f"{sr.stage['df'] * sr.nfb['desensitivity']:.1f}"
        parts.append(
            t(
                "amp.nfb_line",
                nfb_db=f"{sr.nfb['nfb_db']:.0f}",
                gain=f"{sr.nfb['gain_closed']:.1f}",
                gain_db=f"{sr.nfb['gain_closed_db']:.1f}",
                zout=f"{sr.nfb['zout_closed']:.2f}",
                thd=f"{sr.nfb['thd_closed']:.2f}",
                df=df_nfb,
                bw=f"{sr.nfb['bw_factor']:.2f}",
            )
        )

    rk = 0.0
    if dist["ia_0"] > 0.01:
        rk = abs(dist["ug1_0"]) / dist["ia_0"] * 1000.0
    pa_str = f"{pa_w:.1f}"
    # Show model-accurate Pa_avg when available
    if sr.pa_avg and sr.pa_avg.get("pa_avg_mw", 0) > 0:
        pa_avg_w = sr.pa_avg["pa_avg_mw"] / MW_PER_W
        pa_str += f" (avg {pa_avg_w:.1f})"
    pa_line = (
        t("amp.rk_auto_bias", rk=f"{rk:.0f}")
        + "  "
        + f"<span style='color:{pa_color}'>"
        + t("amp.pa_value", pa=pa_str)
        + f"</span> / {t('amp.pa_max_line', max_pa=f'{pa_max:.1f}')}"
    )
    if sr.pg2_mw is not None:
        pa_line += t("amp.pg2_value", pg2=f"{sr.pg2_mw / MW_PER_W:.2f}")
    parts.append(pa_line)

    # Pdc/η: use model-accurate values when pa_avg available
    if sr.pa_avg and sr.pa_avg.get("pdc_avg_mw", 0) > 0:
        pdc_display = sr.pa_avg["pdc_avg_mw"]
        if sr.pg2_mw is not None and sr.pg2_mw > 0:
            pdc_display += sr.pg2_mw
        eta_display = dist["pout_mw"] / pdc_display * PERCENT if pdc_display > 0 else 0
        parts.append(
            t("amp.eta_pdc_line",
              pdc=f"{pdc_display:.0f}",
              eta=f"{eta_display:.1f}")
        )
    elif dist.get("pdc_mw") is not None and dist.get("eta_pct") is not None:
        pdc_display = dist["pdc_mw"]
        eta_display = dist["eta_pct"]
        if sr.pg2_mw is not None and sr.pg2_mw > 0:
            pdc_display += sr.pg2_mw
            eta_display = dist["pout_mw"] / pdc_display * PERCENT
        parts.append(
            t("amp.eta_pdc_line",
              pdc=f"{pdc_display:.0f}",
              eta=f"{eta_display:.1f}")
        )

    return "<br>".join(parts)
