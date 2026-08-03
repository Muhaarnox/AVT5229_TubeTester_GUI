from typing import Dict, List, Optional

import pyqtgraph as pg
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from app.plotting import PlotRenderer
from lm19.plotting.grids import cluster_nominal, filter_ug2_multi, filter_ug2_slice
from lm19.analysis import select_analysis_points, get_available_series
import logging

from lm19.label_formats import format_label

log = logging.getLogger(__name__)
from app.ui_theme import COLOR_ACCENT_BLUE, COLOR_IA, COLOR_IG2, SERIES_PALETTE, STYLE_BOLD_LABEL
from i18n_setup import t
from lm19.constants import MW_PER_W, UG2_ROUND, UA_ROUND
from lm19.plot_style import DEFAULT_GRID_ALPHA
from lm19.amplifier.constants import (
    CIRCUIT_PP,
    CIRCUIT_SE_XFMR,
)

# ── plot_manager local constants ──────────────────────────────────────
RA_SWEEP_FACTOR_DEFAULT = 4.0
RA_SWEEP_MAX_ABS_DEFAULT = 20.0
RA_SWEEP_MIN_ABS_DEFAULT = 0.5
RA_SWEEP_STEPS_DEFAULT = 80
LEGEND_OFFSET = (30, 10)
PEN_WIDTH_PLOT = 2

# Transfer view presets (contract vocabulary): production and tests
# import the constants, not literals.
TRANSFER_VIEW_ALL = "all"
TRANSFER_VIEW_DATASHEET = "datasheet"
TRANSFER_VIEW_LOADLINE = "loadline"
TRANSFER_VIEW_CUSTOM = "custom"
TRANSFER_VIEW_MODES = frozenset({
    TRANSFER_VIEW_ALL, TRANSFER_VIEW_DATASHEET,
    TRANSFER_VIEW_LOADLINE, TRANSFER_VIEW_CUSTOM,
})
DATASHEET_UA_SLICES = 5     # datasheet-style Ia(Ug1) shows a few Ua only
LOADLINE_UA_NEIGHBORS = 1   # slices around Ua≈Ub in the load-line preset


# ── Pure filtering helpers (no Qt deps, testable without QApplication) ──

def _point_in_series(p: Dict, series_ids: list) -> bool:
    """Check if point belongs to one of the given series."""
    return p.get("series_id", 0) in series_ids


def filter_by_series(
    points: List[Dict], checked_ids: List[int], total_count: int,
) -> List[Dict]:
    """Filter points keeping only those whose series matches *checked_ids*.

    Returns all points unchanged when everything is checked.
    """
    if checked_ids is None or len(checked_ids) >= total_count:
        return points
    return [p for p in points if _point_in_series(p, checked_ids)]


def filter_by_ug2_display(
    points: List[Dict],
    checked_ug2: Optional[List[float]],
    is_triode: bool,
    ug2_cluster_thr: float,
) -> List[Dict]:
    """Filter points keeping only those matching checked Ug2 values."""
    if is_triode or not checked_ug2:
        return points
    groups = filter_ug2_multi(points, is_triode, checked_ug2, ug2_cluster_thr)
    return [p for sub in groups.values() for p in sub]


def filter_by_calc_series(
    points: List[Dict], series_id_or_none: Optional[int],
) -> List[Dict]:
    """Filter points for calc heatmaps by selected lamp_calc series.

    series_id_or_none: None → empty (no selection), int → specific series.
    """
    if series_id_or_none is None:
        return []
    return select_analysis_points(points, series_id=series_id_or_none)


def datasheet_ua_slices(nominals: List[float],
                        count: int = DATASHEET_UA_SLICES) -> List[float]:
    """Evenly spaced Ua subset including both endpoints (datasheet preset).

    Index-based spacing (not value-based) so a non-uniform Ua grid still
    yields *count* actually-existing slices.
    """
    s = sorted(nominals)
    if len(s) <= count:
        return s
    n = len(s)
    idxs = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
    return [s[i] for i in idxs]


def loadline_ua_slices(nominals: List[float], ub: float,
                       neighbors: int = LOADLINE_UA_NEIGHBORS) -> List[float]:
    """Nominal nearest to Ub ± *neighbors* slices (load-line preset)."""
    s = sorted(nominals)
    if not s:
        return []
    idx = min(range(len(s)), key=lambda i: abs(s[i] - ub))
    return s[max(0, idx - neighbors):idx + neighbors + 1]


class PlotManager:
    """Manages all measurement plot rendering and associated state.

    Owns the PlotRenderer, plot data (points, series), and delegates
    rendering calls.  Reads UI widget values through the ``w`` dict
    passed at construction time.
    """

    def __init__(self, plot_renderer: PlotRenderer, widgets: Dict) -> None:
        self.renderer = plot_renderer
        self.w = widgets  # references to UI widgets
        # The live-layer controller re-adds its items after a full
        # re-render (plot.clear() removes them).
        self.working_line_reattach = None

        # Plot data
        self.points: List[Dict] = []
        self.series_labels: Dict[int, str] = {}
        self.series_colors: Dict[int, str] = {}
        self.series_ug2_track: Dict[int, bool] = {}
        self.series_models: Dict = {}   # sid → TubeModelProtocol
        self.series_grids: Dict = {}    # sid → ScanGrid (for dense rendering)

        # Internal state
        self.legend_hidden = False
        self.is_triode = False
        self.current_curve_points: List[Dict] = []
        self.labeled_ug1: set = set()


    # ------------------------------------------------------------------
    # Triode / cache
    # ------------------------------------------------------------------

    def set_triode(self, is_triode: bool) -> None:
        """Update triode state for this manager and renderer."""
        self.is_triode = is_triode
        self.renderer.is_triode = is_triode
        self.renderer.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Clear renderer grouping caches (call when points change)."""
        self.renderer.invalidate_cache()

    # ------------------------------------------------------------------
    # Lamp & Ug2 selectors
    # ------------------------------------------------------------------

    def refresh_lamp_combos(self) -> None:
        """Populate lamp_display and lamp_calc from current data."""
        sources = get_available_series(self.points, self.series_labels)

        display = self.w.get("lamp_display_combo")
        if display is not None:
            vals = []
            for s in sources:
                vals.append((s["series_id"], s["label"]))
            display.blockSignals(True)
            display._updating = True
            display._model.clear()
            for sid, label in vals:
                item = QStandardItem(label)
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setData(Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
                item.setData(sid, Qt.ItemDataRole.UserRole)
                display._model.appendRow(item)
            display._updating = False
            display._update_text()
            display.blockSignals(False)
            # Keep selectors visible even for a single source to avoid UI jumps.
            display.setVisible(True)
            source_lbl = self.w.get("_source_label")
            if source_lbl is not None:
                source_lbl.setVisible(True)

        calc = self.w.get("lamp_calc_combo")
        if calc is not None:
            calc.blockSignals(True)
            prev = calc.currentData()
            calc.clear()
            for s in sources:
                calc.addItem(s["label"], userData=s["series_id"])
            # Restore previous selection if still available
            restored = False
            for i in range(calc.count()):
                if calc.itemData(i) == prev:
                    calc.setCurrentIndex(i)
                    restored = True
                    break
            if not restored and calc.count() > 0:
                calc.setCurrentIndex(0)
            calc.blockSignals(False)
            calc.setVisible(True)
            calc_lbl = self.w.get("_calc_lamp_label")
            if calc_lbl is not None:
                calc_lbl.setVisible(True)

    def _is_sid_ug2_track(self, sid: int) -> bool:
        """Check if a series uses Ug2-tracking (triode / triode-connected).

        For the current scan (sid 0) the recorded flag wins over the
        scan-setup widgets: it describes the data actually on the plot,
        while the widgets describe the run being armed next.  The widget
        fallback only applies before any scan has been recorded.
        """
        if sid == 0:
            stored = self.series_ug2_track.get(0)
            if stored is not None:
                return stored
            if self.is_triode:
                return True
            track = self.w.get("ug2_track_radio")
            return track is not None and track.isChecked()
        return self.series_ug2_track.get(sid, False)

    def set_scan_ug2_track(self, is_track: bool) -> None:
        """Record whether the current scan (sid 0) sweeps Ug2 along Ua.

        The flag belongs to the data, not to the scan-setup radio: once a
        scan is on the plot the user is free to arm the next run in the
        other Ug2 mode, and the drawn curves must keep the grouping they
        were measured with.  Without the record, re-arming "Ug2 sweep"
        after a triode-connected scan regroups it by (Ug1, Ug2) — where
        Ug2 = Ua + offset is unique per point — so every curve falls
        apart into isolated symbols on the next full re-render.
        """
        self.series_ug2_track[0] = bool(is_track)
        self.renderer.invalidate_cache()

    def replace_overlay_series(
        self,
        overlay_points: List[Dict],
        series_labels: Dict[int, str],
        series_colors: Dict[int, str],
        series_ug2_track: Optional[Dict[int, bool]] = None,
    ) -> None:
        """Swap every overlay series (sid != 0) for a new set.

        The current scan (sid 0) stays on the plot, so its own
        bookkeeping must survive the swap — rebuilding the dicts from the
        incoming overlay data alone would drop the scan's Ug2-track flag
        and re-derive it from the scan-setup widgets.

        Model entries are dropped for the sids the incoming series take
        over: a leftover ``series_models`` entry on a reused sid makes
        that series render as a dashed curve driven by someone else's
        model. Models on other sids stay — a fitted model is a result of
        its own, still offered as an analysis source on the amp tab, and
        dropping it would force a refit after every "show on main plot".
        """
        self.points = [p for p in self.points
                       if p.get("series_id", 0) == 0] + list(overlay_points)
        scan_label = self.series_labels.get(0)
        scan_color = self.series_colors.get(0)
        scan_track = self.series_ug2_track.get(0)
        self.series_labels = dict(series_labels)
        self.series_colors = dict(series_colors)
        self.series_ug2_track = dict(series_ug2_track) if series_ug2_track else {}
        if scan_label is not None:
            self.series_labels[0] = scan_label
        if scan_color is not None:
            self.series_colors[0] = scan_color
        if scan_track is not None:
            self.series_ug2_track[0] = scan_track
        taken = ({p.get("series_id", 0) for p in overlay_points}
                 | set(series_labels or {}))
        for sid in taken:
            self.series_models.pop(sid, None)
            self.series_grids.pop(sid, None)

    def _get_track_sids(self, points: List[Dict]) -> set:
        """Return set of series_id values that are ug2-tracking."""
        sids = {p.get("series_id", 0) for p in points}
        return {sid for sid in sids if self._is_sid_ug2_track(sid)}

    def _sync_track_sids(self) -> None:
        """Push current track_sids into renderer for methods that read it."""
        self.renderer.track_sids = self._get_track_sids(self.points)

    def refresh_ug2_combos(self, points: List[Dict]) -> None:
        """Populate ug2_display and ug2_calc with clustered Ug2 from points.

        Excludes Ug2 values from ug2-tracking series (triode / triode-connected)
        so the combos only show discrete Ug2 setpoints.
        Controls always stay visible; if no discrete Ug2 values exist the
        combos are simply cleared.
        """
        # Choke point: every data-change path already calls this refresh,
        # so the Transfer Ua-filter combo is refreshed here too rather
        # than duplicating the call at all 8 call sites.
        self.refresh_ua_combo(points)

        display = self.w.get("ug2_display_combo")
        calc = self.w.get("ug2_calc_combo")

        track_sids = self._get_track_sids(points)
        non_track = [p for p in points
                     if p.get("series_id", 0) not in track_sids]

        if not non_track:
            if display is not None:
                display.set_items([])
                display.setVisible(True)
                disp_lbl = self.w.get("_ug2_display_label")
                if disp_lbl is not None:
                    disp_lbl.setVisible(True)
            if calc is not None:
                calc.blockSignals(True)
                calc.clear()
                calc.blockSignals(False)
                calc.setVisible(True)
                calc_lbl = self.w.get("_ug2_calc_label")
                if calc_lbl is not None:
                    calc_lbl.setVisible(True)
            return

        raw = sorted({round(p.get("ug2", 0.0), UG2_ROUND) for p in non_track})
        thr = self.renderer.ug2_cluster_thr
        noms = cluster_nominal(raw, threshold=thr)
        values = []
        for nom in noms:
            members = [v for v in raw if abs(v - nom) <= thr]
            values.append(round(sum(members) / len(members), UA_ROUND))

        if display is not None:
            display.set_items(values)
            display.setVisible(True)
            disp_lbl = self.w.get("_ug2_display_label")
            if disp_lbl is not None:
                disp_lbl.setVisible(True)

        if calc is not None:
            calc.blockSignals(True)
            prev_text = calc.currentText()
            calc.clear()
            for val in values:
                calc.addItem(f"{val:g}")
            if prev_text:
                idx = calc.findText(prev_text)
                if idx >= 0:
                    calc.setCurrentIndex(idx)
            calc.blockSignals(False)
            calc.setVisible(True)
            calc_lbl = self.w.get("_ug2_calc_label")
            if calc_lbl is not None:
                calc_lbl.setVisible(True)

    def _get_display_points(self) -> List[Dict]:
        """Filter points by lamp_display for line plots.

        Ug2 filtering is handled via visibility toggle on existing plot
        items, not by re-filtering point data.
        """
        pts = self.points

        lamp = self.w.get("lamp_display_combo")
        if lamp is not None and lamp.isVisible():
            checked = lamp.checked_values()
            if checked is not None:
                total = lamp._model.rowCount()
                pts = filter_by_series(pts, checked, total)
        return pts

    def _get_calc_points(self) -> List[Dict]:
        """Filter points by lamp_calc for heatmaps."""
        combo = self.w.get("lamp_calc_combo")
        if combo is None:
            return self.points
        sid = combo.currentData()
        return filter_by_calc_series(self.points, sid)

    def select_ug2_calc(self, points: List[Dict]) -> float:
        """Return Ug2 value from ug2_calc_combo (callback for filter_ug2_slice)."""
        combo = self.w.get("ug2_calc_combo")
        if combo is not None and combo.count() > 0:
            try:
                return float(combo.currentText())
            except (ValueError, TypeError):
                pass
        return 0.0

    def on_display_filter_changed(self) -> None:
        """Called when lamp_display selection changes.

        Instant visibility toggle on existing plot items — no re-render.
        """
        lamp = self.w.get("lamp_display_combo")
        if lamp is None:
            return
        checked = lamp.checked_values()
        if checked is None:
            return
        visible_sids = set(checked)
        self.renderer.set_sid_visibility(visible_sids)

    def on_ug2_display_changed(self) -> None:
        """Called when ug2_display_combo checkboxes toggle.

        2D + Transfer + Curves: instant visibility toggle, no recomputation.
        """
        self._apply_ug2_visibility()

    def on_ug2_calc_changed(self) -> None:
        """Called when ug2_calc_combo changes.

        Re-renders 2D/load-line and all Ug2-calc dependent plots so load-line
        intersections and analysis are recalculated for the selected Ug2.
        """
        if not self.points:
            return
        self.render_all()

    def _apply_ug2_visibility(self) -> None:
        """Toggle 2D/Transfer/Curves items visibility based on ug2_display_combo."""
        ug2 = self.w.get("ug2_display_combo")
        if ug2 is None or self.is_triode:
            return
        if ug2._model.rowCount() == 0:
            return
        checked = ug2.checked_values()
        if checked is not None:
            self.renderer.set_ug2_visibility(
                checked, self.renderer.ug2_cluster_thr)

    # ------------------------------------------------------------------
    # Transfer Ua filter + view presets
    # ------------------------------------------------------------------

    def refresh_ua_combo(self, points: List[Dict]) -> None:
        """Populate the Transfer Ua-filter combo with clustered Ua nominals,
        then re-apply the active view preset.

        Data refresh REBUILDS the item list all-checked (``set_items``),
        so in Custom mode the manual selection resets to "all" — the same
        semantics as the Ug2 display filter (new data may carry a
        different Ua grid, stale checks would be meaningless). Presets
        other than Custom are re-applied on the new grid.
        """
        combo = self.w.get("transfer_ua_combo")
        if combo is None:
            return
        raw = sorted({round(p.get("ua", 0.0), UA_ROUND) for p in points})
        thr = self.renderer.ua_cluster_thr
        noms = cluster_nominal(raw, threshold=thr)
        values = []
        for nom in noms:
            members = [v for v in raw if abs(v - nom) <= thr]
            values.append(round(sum(members) / len(members), UA_ROUND))
        combo.set_items(values)
        self.apply_transfer_view_preset()

    def _transfer_view_mode(self) -> str:
        view = self.w.get("transfer_view_combo")
        if view is None:
            return TRANSFER_VIEW_ALL
        mode = view.currentData()
        return mode if mode in TRANSFER_VIEW_MODES else TRANSFER_VIEW_ALL

    def _current_ub(self) -> float:
        """Ub reference for the load-line preset and the ≈Ub accent."""
        panel = self.w.get("amp_control_panel")
        if panel is None:
            return 0.0
        return float(panel.ub_spin.value())

    def apply_transfer_view_preset(self) -> None:
        """Set the Ua-combo checked subset from the active view preset.

        Custom is user-managed — checks are left untouched.
        """
        combo = self.w.get("transfer_ua_combo")
        if combo is None:
            return
        values = combo.all_values()
        if not values:
            return
        mode = self._transfer_view_mode()
        if mode == TRANSFER_VIEW_CUSTOM:
            return
        if mode == TRANSFER_VIEW_DATASHEET:
            subset = datasheet_ua_slices(values)
        elif mode == TRANSFER_VIEW_LOADLINE:
            subset = loadline_ua_slices(values, self._current_ub())
        else:
            subset = values
        combo.set_checked_values(subset)

    def on_transfer_view_changed(self) -> None:
        """View preset combo changed → re-check Ua set + instant visibility."""
        self.apply_transfer_view_preset()
        self._apply_ua_visibility()

    def on_ua_display_changed(self) -> None:
        """User toggled a Ua checkbox → the view becomes Custom + visibility."""
        view = self.w.get("transfer_view_combo")
        if view is not None:
            idx = view.findData(TRANSFER_VIEW_CUSTOM)
            if idx >= 0 and view.currentIndex() != idx:
                view.blockSignals(True)
                view.setCurrentIndex(idx)
                view.blockSignals(False)
        self._apply_ua_visibility()

    def _apply_ua_visibility(self) -> None:
        """Push the checked Ua set into the renderer (instant toggle)."""
        combo = self.w.get("transfer_ua_combo")
        if combo is None or combo._model.rowCount() == 0:
            return
        checked = combo.checked_values()
        if checked is None:
            return
        self.renderer.set_transfer_ua_visibility(
            checked, self.renderer.ua_cluster_thr)

    @staticmethod
    def _point_in_series(p: Dict, series_ids: list) -> bool:
        return _point_in_series(p, series_ids)

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render_all(self) -> None:
        if not self.points:
            w = self.w
            if (w["pa_max_cb"].isChecked() or w["ua_max_cb"].isChecked()
                    or w["ia_max_limit_cb"].isChecked()):
                self._render_2d([])
            self._update_amplifier_tab()
            return
        self._sync_track_sids()
        display_pts = self._get_display_points()
        calc_pts = self._get_calc_points()
        self._render_2d(display_pts)
        self._render_transfer(display_pts)
        self._render_contour(calc_pts)
        self._render_gm_rp(calc_pts)
        self._render_pa_map(calc_pts)
        self._render_curves(display_pts)
        self.renderer.draw_qpoint_all()
        self._apply_ug2_visibility()
        self._apply_ua_visibility()
        self._update_amplifier_tab()

    # ------------------------------------------------------------------
    # Selective render helpers
    # ------------------------------------------------------------------

    def render_2d_only(self) -> None:
        """Re-render only the 2D Ia(Ua) plot."""
        if not self.points:
            w = self.w
            if (w["pa_max_cb"].isChecked() or w["ua_max_cb"].isChecked()
                    or w["ia_max_limit_cb"].isChecked()):
                self._render_2d([])
            return
        self._sync_track_sids()
        self._render_2d(self._get_display_points())
        self.renderer.draw_qpoint_all()
        self._apply_ug2_visibility()

    def render_2d_and_pa(self) -> None:
        """Re-render 2D plot and Pa map (Pa max affects both)."""
        if not self.points:
            self.render_2d_only()
            return
        self._sync_track_sids()
        self._render_2d(self._get_display_points())
        self._render_pa_map(self._get_calc_points())
        self._apply_ug2_visibility()

    def render_line_width_plots(self) -> None:
        """Re-render plots that use line width: 2D, transfer."""
        if not self.points:
            return
        self._sync_track_sids()
        display_pts = self._get_display_points()
        self._render_2d(display_pts)
        self._render_transfer(display_pts)
        self._apply_ug2_visibility()
        self._apply_ua_visibility()

    def render_slice_plots(self) -> None:
        """Re-render plots that depend on Ug2 calc: contour, Gm/Rp, Pa map, curves."""
        if not self.points:
            return
        self._sync_track_sids()
        calc_pts = self._get_calc_points()
        self._render_contour(calc_pts)
        self._render_gm_rp(calc_pts)
        self._render_pa_map(calc_pts)
        self._render_curves(self._get_display_points())
        self.renderer.draw_qpoint_all()
        self._apply_ug2_visibility()

    def render_curves_only(self) -> None:
        """Re-render only the Curves tab (called when Y/X combo changes)."""
        if not self.points:
            return
        self._sync_track_sids()
        self._render_curves(self._get_display_points())
        self._apply_ug2_visibility()

    # ------------------------------------------------------------------
    # 2D plot (Ia vs Ua)
    # ------------------------------------------------------------------

    def _render_2d(self, points: List[Dict]) -> None:
        w = self.w
        zone_rect = {
            "ua_min": w["zone_ua_min"].value(),
            "ua_max": w["zone_ua_max"].value(),
            "ug1_min": w["zone_ug1_min"].value(),
            "ug1_max": w["zone_ug1_max"].value(),
        }
        pa_max = w["pa_max_input"].value() if w["pa_max_cb"].isChecked() else None
        pg2_max = w["pg2_max_input"].value() if w["pg2_max_cb"].isChecked() else None
        pg2_points = None
        if pg2_max is not None and not self.is_triode:
            calc_pts = self._get_calc_points()
            pg2_points = filter_ug2_slice(
                calc_pts, self.is_triode, self.select_ug2_calc)
        ua_max_limit = w["ua_max_input"].value() if w["ua_max_cb"].isChecked() else None
        ia_max_limit = w["ia_max_limit_input"].value() if w["ia_max_limit_cb"].isChecked() else None
        self.renderer.render_plot_2d(
            points,
            ug2_mode_series=w["ug2_mode_series"].isChecked(),
            series_labels=self.series_labels,
            legend_hidden=self.legend_hidden,
            zone_rect=zone_rect,
            track_sids=self._get_track_sids(self.points),
            line_width=w["plot_line_width"].value(),
            series_colors=self.series_colors,
            pa_max=pa_max,
            ua_max_limit=ua_max_limit,
            ia_max_limit=ia_max_limit,
            overlay_pen_style=w["overlay_pen_style"].currentIndex(),
            series_models=self.series_models,
            series_grids=self.series_grids,
            pg2_max=pg2_max,
            pg2_points=pg2_points,
        )
        # The line is owned by WorkingLineController (incremental
        # items; the old full-render path cost 446 ms/tick).
        if self.working_line_reattach is not None:
            self.working_line_reattach()


    # ------------------------------------------------------------------
    # Transfer & contour
    # ------------------------------------------------------------------

    def _render_transfer(self, points: List[Dict]) -> None:
        track_sids = self._get_track_sids(points)
        self.renderer.render_curves(
            points,
            y_param="Ia",
            x_param="Ug1",
            line_width=self.w["plot_line_width"].value(),
            series_labels=self.series_labels,
            track_sids=track_sids,
            target_plot=self.renderer.transfer_plot,
            ub_ref=self._current_ub(),
        )

    def _render_contour(self, points: List[Dict]) -> None:
        self.renderer.render_contour(points, self.select_ug2_calc)

    def _render_gm_rp(self, points: List[Dict]) -> None:
        nominal_s = self.w.get("nominal_s")
        self.renderer.render_gm_rp(
            points,
            select_ug2_slice=self.select_ug2_calc,
            nominal_s=nominal_s,
        )

    def _render_pa_map(self, points: List[Dict]) -> None:
        pa_max = self.w["pa_max_input"].value() if self.w["pa_max_cb"].isChecked() else None
        self.renderer.render_pa_map(
            points,
            select_ug2_slice=self.select_ug2_calc,
            pa_max=pa_max,
        )

    def _render_curves(self, points: List[Dict]) -> None:
        y_combo = self.w.get("curves_y_combo")
        x_combo = self.w.get("curves_x_combo")
        y_param = y_combo.currentText() if y_combo else "Gm"
        x_param = x_combo.currentText() if x_combo else "Ua"
        nominal_s = self.w.get("nominal_s")
        track_sids = self._get_track_sids(points)
        self.renderer.render_curves(
            points,
            y_param=y_param,
            x_param=x_param,
            ug2_values=None,
            nominal_s=nominal_s,
            line_width=self.w["plot_line_width"].value(),
            series_labels=self.series_labels,
            track_sids=track_sids,
        )

    # ------------------------------------------------------------------
    # Amplifier tab
    # ------------------------------------------------------------------

    def _update_amplifier_tab(self) -> None:
        """Push data to ``AmplifierEngine`` and trigger analysis.

        Data flows through the engine (via ``MainWindow``); signal wiring
        happens in ``MainWindow._on_amp_update``.
        """
        from lm19.amp_engine import AmplifierEngine

        engine = self.w.get("amp_engine")
        if engine is None:
            return
        engine.set_data(
            self.points,
            series_labels=self.series_labels,
            srk=self.w.get("srk_data"),
            is_triode=self.is_triode,
            series_models=self.series_models,
        )

        # Populate control panel combos, then trigger analysis
        panel = self.w.get("amp_control_panel")
        if panel is not None:
            # Series (source) combo — which measurement to analyse
            if self.series_labels:
                current_sid = panel.selected_series_id()
                panel.set_series_items(self.series_labels, current_sid=current_sid)

            # Data-source combo — fitted models available for overlay
            model_labels = engine.available_models()
            panel.set_available_models(model_labels)

            # PP Tube B combo — other series for push-pull mismatch
            other = {
                sid: lbl
                for sid, lbl in self.series_labels.items()
                if sid != panel.selected_series_id()
            }
            panel.set_pp_tube_b_items(other)

            # Optimizer: show Ub/Ug2 ranges only when model available
            has_model = bool(engine.available_models())
            panel.set_optimizer_model_mode(has_model)

            panel.settings_changed.emit()

    # ------------------------------------------------------------------
    # Incremental rendering during scan
    # ------------------------------------------------------------------

    def render_curve_incremental(self, event: Dict) -> None:
        self._sync_track_sids()
        self.labeled_ug1 = self.renderer.render_curve_incremental(
            self.current_curve_points,
            self.points,
            event,
            ug2_mode_series=self.w["ug2_mode_series"].isChecked(),
            labeled_ug1=self.labeled_ug1,
            line_width=self.w["plot_line_width"].value(),
        )

    def update_scan_marker(self) -> None:
        """Rebuild 2D marker CurveData from completed scan points so
        tooltip works on already-measured curves during scanning."""
        curves = self.renderer._build_curves_2d(
            self.points, self._get_track_sids(self.points), self.series_labels)
        self.renderer._marker_2d.set_curves(curves)

    # ------------------------------------------------------------------
    # Legend toggle
    # ------------------------------------------------------------------

    def toggle_legend(self) -> None:
        self.legend_hidden = not self.legend_hidden
        self.w["legend_toggle_btn"].setText(
            t('plot.Show_legend') if self.legend_hidden else t('plot.Hide_legend'))
        legend = self.renderer.plot.getPlotItem().legend
        if legend is not None:
            legend.setVisible(not self.legend_hidden)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all plot data including imported series."""
        self.points = []
        self.series_labels = {}
        self.series_colors = {}
        self.series_ug2_track = {}
        self.series_models = {}
        self.series_grids = {}
        self.current_curve_points = []
        self.labeled_ug1 = set()
        self.renderer.invalidate_cache()
        self.renderer.clear()
        self.refresh_ug2_combos([])
        self.refresh_lamp_combos()

    def clear_series(self, sids: List[int]) -> None:
        """Remove specific series by their IDs.

        2D: removes items directly (instant).
        Transfer/Curves: lightweight re-render.
        Heatmaps: untouched (driven by calc combo).
        """
        if not sids:
            return
        sids_set = set(sids)
        self.points = [p for p in self.points
                       if p.get("series_id", 0) not in sids_set]
        for sid in sids:
            self.series_labels.pop(sid, None)
            self.series_colors.pop(sid, None)
            self.series_ug2_track.pop(sid, None)
            self.series_models.pop(sid, None)
            self.series_grids.pop(sid, None)
        # 2D: remove items directly (no full re-render)
        self.renderer.remove_series_items(sids_set)
        self.renderer.invalidate_cache()
        self.refresh_lamp_combos()
        self.refresh_ug2_combos(self.points)
        # Transfer + Curves: re-render with updated points
        if self.points:
            self._sync_track_sids()
            display_pts = self._get_display_points()
            self._render_transfer(display_pts)
            self._render_curves(display_pts)
            self._apply_ua_visibility()

    # ------------------------------------------------------------------
    # Series ID allocation
    # ------------------------------------------------------------------

    def allocate_series_id(self) -> int:
        """Return next available series_id (> 0).

        Scans existing points and model series to avoid collisions.
        """
        used = {p.get("series_id", 0) for p in self.points}
        used.update(self.series_models.keys())
        sid = 1
        while sid in used:
            sid += 1
        return sid

    def remove_series(self, sid: int) -> None:
        """Remove a specific series by series_id."""
        self.points = [p for p in self.points if p.get("series_id", 0) != sid]
        self.series_labels.pop(sid, None)
        self.series_colors.pop(sid, None)
        self.series_ug2_track.pop(sid, None)
        self.series_models.pop(sid, None)
        self.series_grids.pop(sid, None)
        self.renderer.invalidate_cache()
        self.refresh_lamp_combos()
        self.refresh_ug2_combos(self.points)

    # ------------------------------------------------------------------
    # Ra sweep dialog
    # ------------------------------------------------------------------

    def show_ra_sweep(self, parent: Optional["QWidget"] = None,
                      params=None) -> None:
        """The "HD/Pout vs Ra" dialog — parameters come from the AMP
        PANEL: circuit-aware — PP sweeps Ra_aa via the PP path
        (sweep_ra_pp), se_xfmr — the transformer sweep; the method
        label goes into the title (method-visibility rule)."""
        if not self.points:
            QMessageBox.warning(parent, t('plot.Ra_sweep'), t('msg.No_data_for_Ra_sweep'))
            return
        if params is None:
            log.warning("show_ra_sweep called without amp params — "
                        "dialog skipped (programming error at call site)")
            return
        w = self.w
        cfg = w.get("app_config")

        ra_factor = cfg.ra_dialog_max_factor if cfg else RA_SWEEP_FACTOR_DEFAULT
        ra_max_abs = cfg.ra_dialog_max_abs_kohm if cfg else RA_SWEEP_MAX_ABS_DEFAULT
        ra_min_abs = cfg.ra_dialog_min_abs_kohm if cfg else RA_SWEEP_MIN_ABS_DEFAULT
        ra_steps = cfg.ra_dialog_steps if cfg else RA_SWEEP_STEPS_DEFAULT
        ub = params.ub
        ug1_bias = params.ug1_bias
        hs = (params.half_swing
              if params.half_swing and params.half_swing > 0 else None)
        ug2_filter = params.ug2_filter
        is_pp = params.circuit == CIRCUIT_PP
        ra_current = params.pp_raa if is_pp else params.ra
        ra_max = max(ra_current * ra_factor, ra_max_abs)
        if is_pp:
            from lm19.amplifier import sweep_ra_pp
            results = sweep_ra_pp(
                self.points, ub, ra_min_abs, ra_max,
                ug1_bias=ug1_bias, half_swing=hs,
                ug2_filter=ug2_filter, steps=ra_steps,
            )
            method_label = f"{t('plot.Wl_method_5point')} · PP Ra_aa"
        else:
            results = self.renderer.compute_ra_sweep(
                self.points, ub, ra_min=ra_min_abs, ra_max=ra_max,
                steps=ra_steps, ug1_bias=ug1_bias, half_swing=hs,
                ug2_filter=ug2_filter,
                transformer=(params.circuit == CIRCUIT_SE_XFMR),
                ra_dc=params.ra_dc,
            )
            method_label = t('plot.Wl_method_5point')
        if not results:
            QMessageBox.warning(parent, t('plot.Ra_sweep'), t('msg.Not_enough_intersections'))
            return

        dlg = QDialog(parent)
        dlg.setWindowTitle(
            t('msg.Ra_sweep_title', ub=format_label("ua_value", ub))
            + f" — {method_label}")
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)

        thd_plot = pg.PlotWidget(title=t('plot.Distortion_vs_Ra'))
        thd_plot.setLabel("left", t('plot.HD_pct'))
        thd_plot.setLabel("bottom", t('plot.Ra_kOhm'))
        thd_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        thd_plot.addLegend()

        ra_vals = [r["ra"] for r in results]
        hd2_vals = [r["hd2"] for r in results]
        hd3_vals = [r["hd3"] for r in results]
        thd_vals = [np.sqrt(r["hd2"] ** 2 + r["hd3"] ** 2) for r in results]

        thd_plot.plot(ra_vals, hd2_vals, pen=pg.mkPen(COLOR_IG2, width=2), name=t("amp.legend_hd2"))
        thd_plot.plot(ra_vals, hd3_vals, pen=pg.mkPen(COLOR_IA, width=2), name=t("amp.legend_hd3"))
        thd_plot.plot(ra_vals, thd_vals, pen=pg.mkPen(SERIES_PALETTE[2], width=2,
                      style=pg.QtCore.Qt.PenStyle.DashLine), name=t("amp.legend_thd"))

        thd_plot.addItem(pg.InfiniteLine(
            pos=ra_current, angle=90,
            pen=pg.mkPen(COLOR_ACCENT_BLUE, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label=format_label("ra", ra_current), labelOpts={"color": COLOR_ACCENT_BLUE},
        ))

        pout_plot = pg.PlotWidget(title=t('plot.Output_power_vs_Ra'))
        pout_plot.setLabel("left", t('plot.Pout_mW'))
        pout_plot.setLabel("bottom", t('plot.Ra_kOhm'))
        pout_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)

        pout_vals = [r["pout_mw"] for r in results]
        pout_plot.plot(ra_vals, pout_vals, pen=pg.mkPen(SERIES_PALETTE[4], width=2))
        pout_plot.addItem(pg.InfiniteLine(
            pos=ra_current, angle=90,
            pen=pg.mkPen(COLOR_ACCENT_BLUE, width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label=format_label("ra", ra_current), labelOpts={"color": COLOR_ACCENT_BLUE},
        ))

        min_thd_idx = int(np.argmin(thd_vals))
        opt_ra = ra_vals[min_thd_idx]
        opt_thd = thd_vals[min_thd_idx]
        opt_pout = pout_vals[min_thd_idx]

        max_pout_idx = int(np.argmax(pout_vals))
        max_pout_ra = ra_vals[max_pout_idx]
        max_pout_val = pout_vals[max_pout_idx]

        info = QLabel(
            t('msg.Min_THD', ra=format_label("ra_value", opt_ra),
              thd=f"{opt_thd:.2f}", pout=format_label("ua_value", opt_pout))
            + "  |  "
            + t('msg.Max_Pout', ra=format_label("ra_value", max_pout_ra),
                pout=format_label("ua_value", max_pout_val))
        )
        info.setStyleSheet(STYLE_BOLD_LABEL)

        layout.addWidget(info)
        layout.addWidget(thd_plot)
        layout.addWidget(pout_plot)
        dlg.exec()
