"""Three sweep modes: triode / Ug2-track / independent-Ug2.

All three share ``_SweepCtx`` (settle/read/limit callables + state).
Status codes emitted into ``ctx.curves_summary[*]["status"]``:
  ``completed``, ``user_stop``, ``aborted``, ``pa_first``, ``pa_partial``,
  ``pg2_break``/``pg2_first``, ``ig2_break``/``ig2_first``, ``ig2_predict``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

from lm19.scan.events import (
    CURVE_STATUS_ABORTED,
    CURVE_STATUS_COMPLETED,
    CURVE_STATUS_IG2_BREAK,
    CURVE_STATUS_IG2_FIRST,
    CURVE_STATUS_IG2_PREDICT,
    CURVE_STATUS_PA_FIRST,
    CURVE_STATUS_PA_PARTIAL,
    CURVE_STATUS_PG2_BREAK,
    CURVE_STATUS_PG2_FIRST,
    CURVE_STATUS_USER_STOP,
    ScanProgress,
)
from lm19.scan.exceptions import _BreakSweep, _SkipPoint
from lm19.scan.protection import _exceeds_ig2, _exceeds_pa, _exceeds_pg2
from lm19.scan.refine import (
    _build_down_sweep_ua,
    _closest_grid_idx,
    _predict_ig2,
    _refine_curve_inline,
)
from lm19.scan.settings import (
    _IG2_PREDICT_MARGIN,
    _MAX_CONSECUTIVE_PA_BREAKS,
    ScanSettings,
    _frange,
)


# ── module local constants ──
# Why a domain of its own rather than the CURVE_STATUS_* registry: this
# records WHY a sweep stopped, which the caller then maps to a curve
# status (up-sweep and down-sweep map the same reason differently). The
# values collide with status spellings by coincidence, not by contract.
_BREAK_IG2 = "ig2"
_BREAK_PG2 = "pg2"
_BREAK_IG2_PREDICT = "ig2_predict"


@dataclass
class _SweepCtx:
    """Shared context passed to the per-mode sweep functions."""
    settle_ua: Callable
    settle_ug1: Callable
    settle_ug2: Callable
    read_point: Callable
    stopped: Callable
    progress: Optional[Callable[[ScanProgress], None]]
    stop: Optional[Callable[[], bool]]
    settings: ScanSettings
    pa_limit: float
    pg2_limit: float
    ig2_limit: float
    ua_values: List[float]
    ug1_values: List[float]
    # Per-curve outcome log: {"ug1", "ug2", "points", "status"}
    # Status: "completed", "pa_first", "pa_partial", "pg2_break",
    # "pg2_first", "ig2_break", "ig2_first", "ig2_predict",
    # "aborted", "user_stop"
    curves_summary: List[Dict] = field(default_factory=list)


def _sweep_triode(ctx: _SweepCtx) -> List[Dict]:
    """True triode sweep: Ug2=0, iterate Ug1×Ua."""
    points: List[Dict] = []
    consecutive_pa_breaks = 0
    aborted = False
    for ug1 in ctx.ug1_values:
        if aborted:
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_ABORTED})
            continue
        if ctx.stopped():
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_USER_STOP})
            continue
        try:
            ctx.settle_ua(ctx.ua_values[0])
            ctx.settle_ug1(ug1)
        except _SkipPoint:
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_USER_STOP})
            continue
        pa_broke_first = False
        pa_broke = False
        break_point: Optional[Dict] = None
        try:
            curve_pts: List[Dict] = []
            for ua in ctx.ua_values:
                if ctx.stopped():
                    break
                try:
                    ctx.settle_ua(ua)
                except _SkipPoint:
                    continue
                point = ctx.read_point()
                if point is None:
                    continue
                points.append(point)
                curve_pts.append(point)
                if ctx.progress:
                    ctx.progress(point)
                if _exceeds_pa(point, ctx.pa_limit):
                    pa_broke = True
                    break_point = point
                    if len(curve_pts) <= 1:
                        pa_broke_first = True
                    break
            refine_pts = _refine_curve_inline(
                curve_pts, ctx.ua_values, ctx.settings,
                ctx.settle_ua, None, ctx.read_point, ctx.progress, ctx.stop,
                ctx.pa_limit, 0.0,
            )
            points.extend(refine_pts)
        except _BreakSweep:
            pass
        if ctx.progress:
            ctx.progress({"event": "curve_done", "ug2": 0, "ug1": ug1})
        # Record curve outcome — based on break flags, not point count.
        if ctx.stopped():
            status = CURVE_STATUS_USER_STOP
        elif pa_broke_first:
            status = CURVE_STATUS_PA_FIRST
        elif pa_broke:
            status = CURVE_STATUS_PA_PARTIAL
        else:
            status = CURVE_STATUS_COMPLETED
        ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                   "points": len(curve_pts),
                                   "status": status})
        if status == CURVE_STATUS_COMPLETED:
            log.info("Ug1=%.1f: %d points, completed", ug1, len(curve_pts))
        elif status == CURVE_STATUS_PA_PARTIAL and break_point:
            pa_val = break_point["ua"] * break_point["ia"] / 1000.0
            log.warning("Ug1=%.1f: Pa break at Ua=%.0f (Pa=%.2fW, "
                        "limit=%.2fW), %d points",
                        ug1, break_point["ua"], pa_val, ctx.pa_limit,
                        len(curve_pts))
        elif status == CURVE_STATUS_PA_FIRST and break_point:
            pa_val = break_point["ua"] * break_point["ia"] / 1000.0
            log.warning("Ug1=%.1f: Pa exceeded on first Ua=%.0f "
                        "(Pa=%.2fW, limit=%.2fW), no data",
                        ug1, break_point["ua"], pa_val, ctx.pa_limit)
        elif status == CURVE_STATUS_USER_STOP:
            log.info("Ug1=%.1f: interrupted (user/heater), %d points",
                     ug1, len(curve_pts))
        if pa_broke_first:
            consecutive_pa_breaks += 1
            if consecutive_pa_breaks >= _MAX_CONSECUTIVE_PA_BREAKS:
                log.warning(
                    "Pa exceeded on first Ua for %d consecutive Ug1 curves "
                    "— aborting sweep at Ug1=%.1f",
                    consecutive_pa_breaks, ug1,
                )
                if ctx.progress:
                    ctx.progress({"event": "pa_sweep_abort",
                                  "consecutive": consecutive_pa_breaks,
                                  "ug1": ug1})
                aborted = True
        else:
            consecutive_pa_breaks = 0
    return points


def _sweep_ug2_track(ctx: _SweepCtx) -> List[Dict]:
    """Triode-connected pentode: Ug2 tracks Ua with offset."""
    points: List[Dict] = []
    ug2 = 0.0
    consecutive_pa_breaks = 0
    aborted = False
    for ug1 in ctx.ug1_values:
        if aborted:
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_ABORTED})
            continue
        if ctx.stopped():
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_USER_STOP})
            continue
        # Drop Ug2 to the level matching the first Ua BEFORE changing Ug1,
        # otherwise the tube briefly sees low Ua with the previous curve's
        # high Ug2, causing a dangerous Ig2 spike.
        ug2_first = max(0, ctx.ua_values[0] + ctx.settings.ug2_offset)
        try:
            ctx.settle_ug2(ug2_first)
            ctx.settle_ua(ctx.ua_values[0])
            ctx.settle_ug1(ug1)
        except _SkipPoint:
            ctx.curves_summary.append({"ug1": ug1, "ug2": 0.0,
                                       "points": 0, "status": CURVE_STATUS_USER_STOP})
            continue
        pa_broke_first = False
        pa_broke = False
        g2_break_reason: Optional[str] = None   # _BREAK_IG2 | _BREAK_PG2
        g2_broke_first = False
        break_point: Optional[Dict] = None
        try:
            curve_pts: List[Dict] = []
            for ua in ctx.ua_values:
                if ctx.stopped():
                    break
                ug2 = max(0, ua + ctx.settings.ug2_offset)
                # Set Ua before Ug2: when sweeping upward, setting Ua first
                # means the tube briefly sees high Ua with old (lower) Ug2,
                # which lowers Ig2 -> safer for the screen grid.
                try:
                    ctx.settle_ua(ua)
                    ctx.settle_ug2(ug2)
                except _SkipPoint:
                    continue
                point = ctx.read_point()
                if point is None:
                    continue
                # Pa check first — break immediately to avoid prolonged overload
                if _exceeds_pa(point, ctx.pa_limit):
                    points.append(point)
                    curve_pts.append(point)
                    if ctx.progress:
                        ctx.progress(point)
                    pa_broke = True
                    break_point = point
                    if len(curve_pts) <= 1:
                        pa_broke_first = True
                    break
                # Ig2/Pg2: BREAK, not skip (ML-128). In triode connection Ug2
                # rises with Ua, so screen current and dissipation grow
                # monotonically along the sweep — after the first exceedance
                # every later point is further over the limit while yielding
                # no data (each would be discarded anyway), leaving the screen
                # in a growing overload for the rest of the curve. The
                # screen's thermal time constant is seconds, and the firmware
                # OVERIG trip fires at ADC saturation — it protects the
                # tester electronics, not the tube's Pg2 rating. The
                # over-limit point is NOT recorded (same data semantics as
                # the old skip). The independent sweeper keeps skip in its UP
                # sweep: there Ig2 FALLS with rising Ua, later points recover.
                if _exceeds_ig2(point, ctx.ig2_limit):
                    g2_break_reason = _BREAK_IG2
                    break_point = point
                    g2_broke_first = not curve_pts
                    break
                if _exceeds_pg2(point, ctx.pg2_limit, ug2):
                    g2_break_reason = _BREAK_PG2
                    break_point = point
                    g2_broke_first = not curve_pts
                    break
                points.append(point)
                curve_pts.append(point)
                if ctx.progress:
                    ctx.progress(point)
            refine_pts = _refine_curve_inline(
                curve_pts, ctx.ua_values, ctx.settings,
                ctx.settle_ua, ctx.settle_ug2, ctx.read_point,
                ctx.progress, ctx.stop,
                ctx.pa_limit, ctx.pg2_limit, ig2_limit=ctx.ig2_limit,
            )
            points.extend(refine_pts)
        except _BreakSweep:
            pass
        if ctx.progress:
            ctx.progress({"event": "curve_done", "ug2": ug2, "ug1": ug1})
        # Record curve outcome — based on break flags, not point count.
        if ctx.stopped():
            status = CURVE_STATUS_USER_STOP
        elif pa_broke_first:
            status = CURVE_STATUS_PA_FIRST
        elif g2_broke_first:
            status = (CURVE_STATUS_PG2_FIRST if g2_break_reason == _BREAK_PG2
                      else CURVE_STATUS_IG2_FIRST)
        elif pa_broke:
            status = CURVE_STATUS_PA_PARTIAL
        elif g2_break_reason == _BREAK_PG2:
            status = CURVE_STATUS_PG2_BREAK
        elif g2_break_reason == _BREAK_IG2:
            status = CURVE_STATUS_IG2_BREAK
        else:
            status = CURVE_STATUS_COMPLETED
        ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                   "points": len(curve_pts),
                                   "status": status})
        if status == CURVE_STATUS_COMPLETED:
            log.info("Ug1=%.1f Ug2~%.0f: %d points, completed",
                     ug1, ug2, len(curve_pts))
        elif status == CURVE_STATUS_PA_PARTIAL and break_point:
            pa_val = break_point["ua"] * break_point["ia"] / 1000.0
            log.warning("Ug1=%.1f Ug2~%.0f: Pa break at Ua=%.0f "
                        "(Pa=%.2fW, limit=%.2fW), %d points",
                        ug1, ug2, break_point["ua"], pa_val,
                        ctx.pa_limit, len(curve_pts))
        elif status == CURVE_STATUS_PA_FIRST and break_point:
            pa_val = break_point["ua"] * break_point["ia"] / 1000.0
            log.warning("Ug1=%.1f Ug2~%.0f: Pa exceeded at first Ua=%.0f "
                        "(Pa=%.2fW, limit=%.2fW)",
                        ug1, ug2, break_point["ua"], pa_val, ctx.pa_limit)
        elif status in (CURVE_STATUS_PG2_BREAK, CURVE_STATUS_PG2_FIRST) and break_point:
            pg2_val = ug2 * break_point["ig2"] / 1000.0
            log.warning("Ug1=%.1f Ug2~%.0f: Pg2 break at Ua=%.0f "
                        "(Pg2=%.2fW, limit=%.2fW), %d points — track sweep "
                        "stopped: Pg2 only grows with Ua",
                        ug1, ug2, break_point["ua"], pg2_val,
                        ctx.pg2_limit, len(curve_pts))
        elif status in (CURVE_STATUS_IG2_BREAK, CURVE_STATUS_IG2_FIRST) and break_point:
            log.warning("Ug1=%.1f Ug2~%.0f: Ig2 break at Ua=%.0f "
                        "(Ig2=%.2fmA, limit=%.2fmA), %d points — track sweep "
                        "stopped: Ig2 only grows with Ua",
                        ug1, ug2, break_point["ua"], break_point["ig2"],
                        ctx.ig2_limit, len(curve_pts))
        elif status == CURVE_STATUS_USER_STOP:
            log.info("Ug1=%.1f Ug2~%.0f: interrupted, %d points",
                     ug1, ug2, len(curve_pts))
        # First-point protection break (Pa OR Ig2/Pg2): in track mode every
        # quantity only grows as Ug1 opens toward 0, so the next curves can
        # only break earlier — abort the sweep (same event name kept:
        # KNOWN_SCAN_EVENTS pins "pa_sweep_abort").
        if pa_broke_first or g2_broke_first:
            consecutive_pa_breaks += 1
            if consecutive_pa_breaks >= _MAX_CONSECUTIVE_PA_BREAKS:
                log.warning(
                    "Protection break on first Ua for %d consecutive Ug1 "
                    "curves — aborting sweep at Ug1=%.1f",
                    consecutive_pa_breaks, ug1,
                )
                if ctx.progress:
                    ctx.progress({"event": "pa_sweep_abort",
                                  "consecutive": consecutive_pa_breaks,
                                  "ug1": ug1})
                aborted = True
        else:
            consecutive_pa_breaks = 0
    return points


def _sweep_ug2_independent(ctx: _SweepCtx, prev_ua: float) -> List[Dict]:
    """Independent Ug2 sweep — bidirectional from Ua ~ Ug2.

    First Ug1 at each Ug2: bidirectional sweep from start_idx.
    Subsequent Ug1: skip up-sweep (Pa only grows with Ug1 openness), start
    down-sweep from ``safe_entry_idx`` — the highest Ua that was Pa-safe in
    the previous curve.  Reset per Ug2 level.
    """
    points: List[Dict] = []
    ug2_values = _frange(ctx.settings.ug2.start, ctx.settings.ug2.stop,
                         ctx.settings.ug2.step)
    for ug2 in ug2_values:
        if ctx.stopped():
            for ug1 in ctx.ug1_values:
                ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                           "points": 0,
                                           "status": CURVE_STATUS_USER_STOP})
            continue
        # Safety: before raising Ug2, ensure Ua >= Ug2(new). Raise Ua to ug2
        # UNCONDITIONALLY — the old guard `if prev_ua < ug2` read a frozen
        # scan-start prev_ua that goes stale across Ug2 levels (the device Ua
        # changes every level), so it could skip the raise and let settle_ug2
        # push Ug2 above the actual (lower) Ua → Ig2 spike. ug2_values ascend,
        # so the still-set old Ug2 < ug2 and raising Ua to ug2 keeps Ua >= old
        # Ug2 throughout. (prev_ua param kept for call-site compat — no longer
        # read here.)
        try:
            ctx.settle_ug1(ctx.ug1_values[0])
            ctx.settle_ua(ug2)
            ctx.settle_ug2(ug2)
        except _SkipPoint:
            for ug1 in ctx.ug1_values:
                ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                           "points": 0,
                                           "status": CURVE_STATUS_USER_STOP})
            continue
        start_idx = _closest_grid_idx(ug2, ctx.ua_values)
        consecutive_pa_breaks = 0
        safe_entry_idx: Optional[int] = None
        aborted = False
        # Set when a hardware-protection trip (OVERIA) zeroes Ug2 mid-level, so
        # the next Ug1 curve re-establishes Ug2 instead of measuring at Ug2≈0.
        need_ug2_resettle = False
        for ug1 in ctx.ug1_values:
            if aborted:
                ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                           "points": 0,
                                           "status": CURVE_STATUS_ABORTED})
                continue
            if ctx.stopped():
                ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                           "points": 0,
                                           "status": CURVE_STATUS_USER_STOP})
                continue

            # Subsequent Ug1: start lower (safe_entry) and skip up-sweep.
            skip_up = safe_entry_idx is not None and safe_entry_idx < start_idx
            entry_idx = safe_entry_idx if skip_up else start_idx

            try:
                if need_ug2_resettle:
                    # A protection trip earlier this level zeroed Ug2. Restore it
                    # before measuring — otherwise every remaining Ug1 curve is
                    # silently measured at Ug2≈0 (a fictitious low-Ug2 cluster).
                    # Safe order: raise Ua to >= Ug2 first.
                    ctx.settle_ua(ug2)
                    ctx.settle_ug2(ug2)
                    need_ug2_resettle = False
                ctx.settle_ua(ctx.ua_values[entry_idx])
                ctx.settle_ug1(ug1)
            except _SkipPoint:
                ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                           "points": 0,
                                           "status": CURVE_STATUS_USER_STOP})
                continue
            up_pts: List[Dict] = []
            down_pts: List[Dict] = []
            pa_broke_up = False
            pa_broke_first = False
            # Track down-sweep termination reason: None or a _BREAK_* value
            down_break_reason: Optional[str] = None
            # The point that triggered a break (for diagnostics)
            break_point: Optional[Dict] = None
            try:
                if not skip_up:
                    # -- SWEEP UP: ua_values[entry_idx .. end]
                    for ua in ctx.ua_values[entry_idx:]:
                        if ctx.stopped():
                            break
                        try:
                            ctx.settle_ua(ua)
                        except _SkipPoint:
                            continue
                        point = ctx.read_point()
                        if point is None:
                            continue
                        # Pa check first — break immediately to avoid prolonged overload
                        if _exceeds_pa(point, ctx.pa_limit):
                            points.append(point)
                            up_pts.append(point)
                            if ctx.progress:
                                ctx.progress(point)
                            pa_broke_up = True
                            break_point = point
                            break
                        if _exceeds_ig2(point, ctx.ig2_limit):
                            continue
                        if _exceeds_pg2(point, ctx.pg2_limit, ug2):
                            continue
                        points.append(point)
                        up_pts.append(point)
                        if ctx.progress:
                            ctx.progress(point)

                # -- SWEEP DOWN
                # skip_up mode: down grid includes entry_idx (measure it here).
                # Normal mode: down grid is below entry_idx (measured by up).
                down_top_idx = entry_idx if skip_up else entry_idx - 1
                if down_top_idx >= 0 and not ctx.stopped():
                    if skip_up:
                        # Already settled to entry_idx; no soft-landing needed.
                        landing_ua = ctx.ua_values[entry_idx]
                    elif pa_broke_up and len(up_pts) <= 1:
                        # Pa exceeded at entry_idx — skip landing there, go directly below.
                        landing_ua = ctx.ua_values[down_top_idx]
                    else:
                        landing_ua = ctx.ua_values[entry_idx]
                    try:
                        ctx.settle_ua(landing_ua)
                    except _SkipPoint:
                        pass
                    else:
                        grid_desc = list(reversed(ctx.ua_values[:down_top_idx + 1]))
                        down_steps = _build_down_sweep_ua(
                            grid_desc, landing_ua,
                            ctx.settings.down_max_step_v,
                        )
                        try:
                            ctx.settle_ua(down_steps[0][0])
                        except _SkipPoint:
                            pass
                        else:
                            for ua, is_grid in down_steps:
                                if ctx.stopped():
                                    break
                                if ctx.ig2_limit > 0:
                                    ig2_est = _predict_ig2(
                                        down_pts, ua, ig2_limit=ctx.ig2_limit,
                                    )
                                    if ig2_est is not None and ig2_est > ctx.ig2_limit * _IG2_PREDICT_MARGIN:
                                        down_break_reason = _BREAK_IG2_PREDICT
                                        break_point = {"ua": ua,
                                                        "ig2_est": ig2_est}
                                        break
                                try:
                                    ctx.settle_ua(ua)
                                except _SkipPoint:
                                    continue
                                point = ctx.read_point()
                                if point is None:
                                    continue
                                if _exceeds_ig2(point, ctx.ig2_limit):
                                    down_break_reason = _BREAK_IG2
                                    break_point = point
                                    break
                                if _exceeds_pg2(point, ctx.pg2_limit, ug2):
                                    down_break_reason = _BREAK_PG2
                                    break_point = point
                                    break
                                if not is_grid:
                                    continue
                                if _exceeds_pa(point, ctx.pa_limit):
                                    continue  # down-sweep: Pa drops with Ua, skip point but keep going
                                points.append(point)
                                down_pts.append(point)
                                if ctx.progress:
                                    ctx.progress(point)
                # Merge into sorted curve for refine
                curve_pts = list(reversed(down_pts)) + up_pts
                min_safe_ua = down_pts[-1]["ua"] if down_pts else (
                    ctx.ua_values[entry_idx] if entry_idx > 0 else 0.0)
                refine_pts = _refine_curve_inline(
                    curve_pts, ctx.ua_values, ctx.settings,
                    ctx.settle_ua, None, ctx.read_point,
                    ctx.progress, ctx.stop,
                    ctx.pa_limit, ctx.pg2_limit, min_safe_ua, ctx.ig2_limit,
                )
                points.extend(refine_pts)
                # Detect fully over-limit curve.
                if skip_up:
                    # Skip-up mode: only down was attempted. Empty down = all over limit.
                    if len(down_pts) == 0:
                        pa_broke_first = True
                elif (pa_broke_up and len(up_pts) <= 1
                        and len(down_pts) == 0):
                    pa_broke_first = True

                # Update safe_entry_idx: highest Ua where Pa was OK this curve.
                highest_safe_ua: Optional[float] = None
                if up_pts:
                    if pa_broke_up and len(up_pts) > 1:
                        highest_safe_ua = up_pts[-2]["ua"]  # last valid before Pa break
                    elif not pa_broke_up:
                        highest_safe_ua = up_pts[-1]["ua"]  # up ran fully
                if highest_safe_ua is None and down_pts:
                    highest_safe_ua = down_pts[0]["ua"]     # first down = highest valid
                if highest_safe_ua is not None:
                    safe_entry_idx = _closest_grid_idx(
                        highest_safe_ua, ctx.ua_values)
            except _BreakSweep:
                # Protection (OVERIA) zeroed the Ua/Ug2 setpoints. Flag a Ug2
                # re-settle for the next Ug1 of this level (so it is not measured
                # at Ug2≈0) and surface the interruption.
                need_ug2_resettle = True
                log.warning(
                    "Ug2~%.0f Ug1=%.1f: hardware protection broke the sweep — "
                    "re-establishing Ug2 for the next curve", ug2, ug1)
            if ctx.progress:
                ctx.progress({"event": "curve_done", "ug2": ug2, "ug1": ug1})

            # Determine curve status from break reasons.
            # Note: valid_points count is not used — down-sweep's Pa
            # `continue` legitimately skips some points, so count
            # alone cannot distinguish "completed with Pa-skips" from
            # "interrupted".  Rely on break-reason flags instead.
            total_collected = len(up_pts) + len(down_pts)
            if ctx.stopped():
                status = CURVE_STATUS_USER_STOP
            elif pa_broke_first:
                # No usable data — classify by actual cause.
                if down_break_reason == _BREAK_PG2:
                    status = CURVE_STATUS_PG2_FIRST
                elif down_break_reason == _BREAK_IG2:
                    status = CURVE_STATUS_IG2_FIRST
                elif down_break_reason == _BREAK_IG2_PREDICT:
                    status = CURVE_STATUS_IG2_PREDICT
                else:
                    status = CURVE_STATUS_PA_FIRST
            elif down_break_reason == _BREAK_PG2:
                status = CURVE_STATUS_PG2_BREAK
            elif down_break_reason == _BREAK_IG2:
                status = CURVE_STATUS_IG2_BREAK
            elif down_break_reason == _BREAK_IG2_PREDICT:
                status = CURVE_STATUS_IG2_PREDICT
            elif pa_broke_up:
                status = CURVE_STATUS_PA_PARTIAL
            else:
                # Both up and down (if any) ran to completion without breaks.
                # Some Ua points may have been Pa-skipped in down-sweep,
                # but the curve covered its full intended range.
                status = CURVE_STATUS_COMPLETED
            ctx.curves_summary.append({"ug1": ug1, "ug2": ug2,
                                       "points": total_collected,
                                       "status": status})

            last_down_ua = down_pts[-1]["ua"] if down_pts else 0
            if status == CURVE_STATUS_COMPLETED:
                log.info("Ug1=%.1f Ug2=%.0f: %d points, completed%s",
                         ug1, ug2, total_collected,
                         " (skip_up)" if skip_up else "")
            elif status == CURVE_STATUS_PA_PARTIAL and break_point:
                pa_val = break_point["ua"] * break_point["ia"] / 1000.0
                log.warning("Ug1=%.1f Ug2=%.0f: Pa break at Ua=%.0f "
                            "(Pa=%.2fW, limit=%.2fW), %d points",
                            ug1, ug2, break_point["ua"], pa_val,
                            ctx.pa_limit, total_collected)
            elif status == CURVE_STATUS_PA_FIRST and break_point:
                pa_val = break_point["ua"] * break_point["ia"] / 1000.0
                log.warning("Ug1=%.1f Ug2=%.0f: Pa exceeded at first Ua=%.0f "
                            "(Pa=%.2fW, limit=%.2fW), no data",
                            ug1, ug2, break_point["ua"], pa_val, ctx.pa_limit)
            elif status in (CURVE_STATUS_PG2_BREAK, CURVE_STATUS_PG2_FIRST) and break_point:
                pg2_val = ug2 * break_point["ig2"] / 1000.0
                log.warning("Ug1=%.1f Ug2=%.0f: Pg2 break at Ua=%.0f "
                            "(Pg2=%.2fW, limit=%.2fW, last safe Ua=%.0f), "
                            "%d points", ug1, ug2, break_point["ua"], pg2_val,
                            ctx.pg2_limit, last_down_ua, total_collected)
            elif status in (CURVE_STATUS_IG2_BREAK, CURVE_STATUS_IG2_FIRST) and break_point:
                log.warning("Ug1=%.1f Ug2=%.0f: Ig2 break at Ua=%.0f "
                            "(Ig2=%.2fmA, limit=%.2fmA, last safe Ua=%.0f), "
                            "%d points", ug1, ug2, break_point["ua"],
                            break_point["ig2"], ctx.ig2_limit,
                            last_down_ua, total_collected)
            elif status == CURVE_STATUS_IG2_PREDICT and break_point:
                log.warning("Ug1=%.1f Ug2=%.0f: Ig2 predict break at Ua=%.0f "
                            "(predicted Ig2=%.2fmA, limit=%.2fmA, "
                            "last safe Ua=%.0f), %d points",
                            ug1, ug2, break_point["ua"],
                            break_point.get("ig2_est", 0), ctx.ig2_limit,
                            last_down_ua, total_collected)
            elif status == CURVE_STATUS_USER_STOP:
                log.info("Ug1=%.1f Ug2=%.0f: interrupted, %d points",
                         ug1, ug2, total_collected)

            if pa_broke_first:
                consecutive_pa_breaks += 1
                if consecutive_pa_breaks >= _MAX_CONSECUTIVE_PA_BREAKS:
                    log.warning(
                        "No safe Ua for %d consecutive Ug1 curves at "
                        "Ug2=%.0f — aborting at Ug1=%.1f",
                        consecutive_pa_breaks, ug2, ug1,
                    )
                    if ctx.progress:
                        ctx.progress({"event": "pa_sweep_abort",
                                      "consecutive": consecutive_pa_breaks,
                                      "ug1": ug1, "ug2": ug2})
                    aborted = True
            else:
                consecutive_pa_breaks = 0
    return points
