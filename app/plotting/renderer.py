import pyqtgraph as pg
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from i18n_setup import t
from app.curve_marker import CurveMarker
from lm19.curve_data import (
    CurveData, FIELDS_2D, FIELDS_TRANSFER,
    build_curves_fields,
)
from lm19.label_formats import format_label
from app.heatmap_marker import HeatmapMarker
from lm19.plotting.grids import (
    cluster_nominal as _cluster_nominal_fn,
    nominal_key as _nominal_key_fn,
)
from lm19.constants import (
    UA_CLUSTER_THR as _DEFAULT_UA_CLUSTER_THR,
    UG1_CLUSTER_THR as _DEFAULT_UG1_CLUSTER_THR,
    UG2_CLUSTER_THR as _DEFAULT_UG2_CLUSTER_THR,
)
from lm19.plot_style import (
    DEFAULT_LINE_WIDTH as _DEFAULT_LINE_WIDTH,
    DEFAULT_GRID_ALPHA as _DEFAULT_GRID_ALPHA,
    SERIES_PALETTE as _SERIES_PALETTE,
    COLOR_ZONE as _COLOR_ZONE,
    MODEL_DENSE_SAMPLES as _MODEL_DENSE_SAMPLES,
    PLOT_PADDING,
)


from app.plotting._curves_plot_mixin import _CurvesPlotMixin
from app.plotting._heatmap_mixin import _HeatmapMixin
from app.plotting._plot_2d_mixin import _Plot2DMixin


@dataclass
class RendererContext:
    """Mutable per-render state shared across ``PlotRenderer`` mixins.

    Holds the fields that downstream code (``PlotManager``, tests) writes
    to directly; per-plot-type mixins read/write them via ``self.ctx``.
    Property descriptors on ``PlotRenderer`` forward
    ``r.is_triode = X`` / ``r.track_sids = …`` writes here so existing
    attribute-style call sites keep working.

    Cluster thresholds are constructor-time inputs (effectively immutable
    after init) but live here too so mixins can read them uniformly.
    """
    is_triode: bool = False
    track_sids: Set[int] = field(default_factory=set)
    ua_cluster_thr: float = _DEFAULT_UA_CLUSTER_THR
    ug1_cluster_thr: float = _DEFAULT_UG1_CLUSTER_THR
    ug2_cluster_thr: float = _DEFAULT_UG2_CLUSTER_THR


class PlotRenderer(_CurvesPlotMixin, _HeatmapMixin, _Plot2DMixin):
    DEFAULT_MARKER_LOCK_PX = 15
    DEFAULT_UA_CLUSTER_THR = _DEFAULT_UA_CLUSTER_THR
    DEFAULT_UG1_CLUSTER_THR = _DEFAULT_UG1_CLUSTER_THR
    DEFAULT_UG2_CLUSTER_THR = _DEFAULT_UG2_CLUSTER_THR
    DEFAULT_LINE_WIDTH = _DEFAULT_LINE_WIDTH
    SCAN_SYMBOL = "o"
    SCAN_SYMBOL_SIZE = 5
    OVERLAY_SYMBOL_SIZE = 4
    OVERLAY_LW_FACTOR = 0.6
    OVERLAY_LW_MIN = 1.0
    CURVES_SYMBOL_SIZE = 4
    MODEL_DENSE_SAMPLES = _MODEL_DENSE_SAMPLES
    DEFAULT_GRID_ALPHA = _DEFAULT_GRID_ALPHA

    def __init__(self, plot: pg.PlotWidget, contour_plot: pg.PlotWidget, contour_image: pg.ImageItem, transfer_plot: pg.PlotWidget = None,
                 gm_plot: pg.PlotWidget = None, gm_image: pg.ImageItem = None,
                 rp_plot: pg.PlotWidget = None, rp_image: pg.ImageItem = None,
                 mu_plot: pg.PlotWidget = None, mu_image: pg.ImageItem = None,
                 pa_map_plot: pg.PlotWidget = None, pa_map_image: pg.ImageItem = None,
                 curves_plot: pg.PlotWidget = None,
                 marker_lock_px: int = DEFAULT_MARKER_LOCK_PX,
                 ua_cluster_thr: float = DEFAULT_UA_CLUSTER_THR,
                 ug1_cluster_thr: float = DEFAULT_UG1_CLUSTER_THR,
                 ug2_cluster_thr: float = DEFAULT_UG2_CLUSTER_THR):
        self.plot = plot
        self.contour_plot = contour_plot
        self.contour_image = contour_image
        self.transfer_plot = transfer_plot
        self.gm_plot = gm_plot
        self.gm_image = gm_image
        self.rp_plot = rp_plot
        self.rp_image = rp_image
        self.mu_plot = mu_plot
        self.mu_image = mu_image
        self.pa_map_plot = pa_map_plot
        self.pa_map_image = pa_map_image
        self.curves_plot = curves_plot
        self._curves_labels: list = []
        # Shared mutable state — see RendererContext docstring.
        # Cluster thresholds live in ctx so mixins can read uniformly;
        # is_triode and track_sids are written directly via ``r.is_triode = …``
        # by PlotManager and tests, forwarded through property descriptors.
        self.ctx = RendererContext(
            ua_cluster_thr=ua_cluster_thr,
            ug1_cluster_thr=ug1_cluster_thr,
            ug2_cluster_thr=ug2_cluster_thr,
        )
        self.line_labels = []
        self._transfer_labels = []
        # Heatmap colorbars: one ColorBarItem per map, created lazily on
        # first render (_sync_heatmap_bar). Without them the maps had no
        # value scale at all — readings existed only in hover tooltips
        # and the color mapping silently changed with every autoLevels.
        self._heatmap_bars: Dict[str, object] = {}
        # "Lock scale" support: captured (lo, hi) per map while locked —
        # autoLevels would otherwise re-stretch the palette per render,
        # making two scans incomparable by color.
        self._heatmap_scale_locked: bool = False
        self._heatmap_locked_levels: Dict[str, tuple] = {}
        self.ug2_colorbar = None
        self._ug2_colorbar_image = None
        self._gm_overlay_items: list = []
        self._pa_overlay_items: list = []
        self._right_heatmap_mode = "rp"  # "rp" | "mu"
        self._qpoint_items: Dict[str, list] = {}  # plot_key → [pg items]
        self.palette = list(_SERIES_PALETTE)
        self.cmap = pg.colormap.get("viridis")
        # Snap-to-curve markers
        self._marker_2d = CurveMarker(self.plot, fields=FIELDS_2D,
                                      lock_px=marker_lock_px)
        self._marker_transfer = (
            CurveMarker(self.transfer_plot, fields=FIELDS_TRANSFER,
                        lock_px=marker_lock_px)
            if self.transfer_plot else None
        )
        self._marker_curves = (
            CurveMarker(self.curves_plot,
                        fields=build_curves_fields("Gm", "Ua"),
                        lock_px=marker_lock_px)
            if self.curves_plot else None
        )
        # Heatmap markers
        self._marker_contour = HeatmapMarker(
            self.contour_plot, "Ia", "mA", ".2f")
        self._marker_gm = (
            HeatmapMarker(self.gm_plot, "Gm", "mA/V", ".2f")
            if self.gm_plot else None
        )
        self._marker_rp = (
            HeatmapMarker(self.rp_plot, "Rp", "kΩ", ".2f")
            if self.rp_plot else None
        )
        self._marker_mu = (
            HeatmapMarker(self.mu_plot, "µ", "", ".1f")
            if self.mu_plot else None
        )
        self._marker_pa = (
            HeatmapMarker(self.pa_map_plot, "Pa", "W", ".3f")
            if self.pa_map_plot else None
        )
        self._curves_cd_accum: List[CurveData] = []
        self._ug2_plot_items: Dict[float, list] = {}
        self._ug2_transfer_items: Dict[float, list] = {}
        self._ug2_curves_items: Dict[float, list] = {}
        # Transfer Ua-filter: items per Ua nominal + last checked sets of
        # BOTH transfer filters (Ug2 × Ua) — visibility is their AND, so
        # neither setter may write setVisible directly (it would clobber
        # the other filter). See _apply_transfer_visibility().
        self._ua_transfer_items: Dict[float, list] = {}
        self._transfer_ug2_filter: Optional[Tuple[List[float], float]] = None
        self._transfer_ua_filter: Optional[Tuple[List[float], float]] = None
        self._sid_plot_items: Dict[int, list] = {}
        self._sid_transfer_items: Dict[int, list] = {}
        self._sid_curves_items: Dict[int, list] = {}
        # Full (unfiltered) marker curve lists — for Ug2 visibility filtering
        self._all_marker_curves_2d: List[CurveData] = []
        self._all_marker_curves_transfer: List[CurveData] = []
        self._all_marker_curves_curves: List[CurveData] = []
        self._curves_collect: Optional[list] = None
        # Grouping caches — avoid recomputing when only visual params change
        self._2d_cache = None
        self._ll_cache = {}
        # Feeds WorkingLineController (consumers: curves tab,
        # Q markers, report).
        self._load_line_intersections = []
        self._load_line_analysis = None
        # PP composite with working line (data path, mirrored around
        # _pp_bias); drawn on Ia(Ug1) folded into the positive quadrant.
        self._pp_composite: List[Dict] = []
        self._pp_bias: float = 0.0

    # ── Property delegation to RendererContext ──────────────────
    # PlotManager.set_triode and several tests assign directly to
    # ``r.is_triode`` / ``r.track_sids``. These properties forward those
    # writes to ``self.ctx`` so callers don't have to know about the ctx.
    # Some tests construct via ``object.__new__(PlotRenderer)`` to skip Qt
    # init; ``_get_ctx()`` lazy-creates the dataclass for those paths.

    def _get_ctx(self) -> RendererContext:
        ctx = self.__dict__.get("ctx")
        if ctx is None:
            ctx = RendererContext()
            self.__dict__["ctx"] = ctx
        return ctx

    @property
    def is_triode(self) -> bool:
        return self._get_ctx().is_triode

    @is_triode.setter
    def is_triode(self, value: bool) -> None:
        self._get_ctx().is_triode = value

    @property
    def track_sids(self) -> Set[int]:
        return self._get_ctx().track_sids

    @track_sids.setter
    def track_sids(self, value: Set[int]) -> None:
        self._get_ctx().track_sids = value

    @property
    def ua_cluster_thr(self) -> float:
        return self._get_ctx().ua_cluster_thr

    @ua_cluster_thr.setter
    def ua_cluster_thr(self, value: float) -> None:
        self._get_ctx().ua_cluster_thr = value

    @property
    def ug1_cluster_thr(self) -> float:
        return self._get_ctx().ug1_cluster_thr

    @ug1_cluster_thr.setter
    def ug1_cluster_thr(self, value: float) -> None:
        self._get_ctx().ug1_cluster_thr = value

    @property
    def ug2_cluster_thr(self) -> float:
        return self._get_ctx().ug2_cluster_thr

    @ug2_cluster_thr.setter
    def ug2_cluster_thr(self, value: float) -> None:
        self._get_ctx().ug2_cluster_thr = value

    def invalidate_cache(self) -> None:
        """Clear all grouping caches. Call when measurement data changes."""
        self._2d_cache = None
        self._ll_cache = {}

    def suspend_markers(self) -> None:
        """Disable all markers (call before live scanning)."""
        self._marker_2d.set_enabled(False)
        if self._marker_transfer:
            self._marker_transfer.set_enabled(False)
        if self._marker_curves:
            self._marker_curves.set_enabled(False)
        self._marker_contour.set_enabled(False)
        if self._marker_gm:
            self._marker_gm.set_enabled(False)
        if self._marker_rp:
            self._marker_rp.set_enabled(False)
        if self._marker_mu:
            self._marker_mu.set_enabled(False)
        if self._marker_pa:
            self._marker_pa.set_enabled(False)

    def resume_markers(self) -> None:
        """Re-enable all markers (call after scan finishes)."""
        self._marker_2d.set_enabled(True)
        if self._marker_transfer:
            self._marker_transfer.set_enabled(True)
        if self._marker_curves:
            self._marker_curves.set_enabled(True)
        self._marker_contour.set_enabled(True)
        if self._marker_gm:
            self._marker_gm.set_enabled(True)
        if self._marker_rp:
            self._marker_rp.set_enabled(True)
        if self._marker_mu:
            self._marker_mu.set_enabled(True)
        if self._marker_pa:
            self._marker_pa.set_enabled(True)

    def clear(self) -> None:
        self.plot.clear()
        self._clear_line_labels()
        self._ug2_plot_items.clear()
        self._sid_plot_items.clear()
        self._marker_2d.reattach()
        if self.transfer_plot is not None:
            self.transfer_plot.clear()
            self._ug2_transfer_items.clear()
            self._ua_transfer_items.clear()
            for item in self._transfer_labels:
                self.transfer_plot.removeItem(item)
            self._transfer_labels = []
            if self._marker_transfer:
                self._marker_transfer.reattach()
        self._marker_contour.clear()
        if self.gm_image is not None:
            self.gm_image.clear()
            self._remove_overlay_items(self.gm_plot, self.gm_image,
                                       self._gm_overlay_items)
            if self._marker_gm:
                self._marker_gm.clear()
        if self.rp_image is not None:
            self.rp_image.clear()
            if self._marker_rp:
                self._marker_rp.clear()
        if self.mu_image is not None:
            self.mu_image.clear()
            if self._marker_mu:
                self._marker_mu.clear()
        if self.pa_map_image is not None:
            self.pa_map_image.clear()
            self._remove_overlay_items(self.pa_map_plot, self.pa_map_image,
                                       self._pa_overlay_items)
            if self._marker_pa:
                self._marker_pa.clear()
        if self.curves_plot is not None:
            self.curves_plot.clear()
            self._ug2_curves_items.clear()
            for item in self._curves_labels:
                self.curves_plot.removeItem(item)
            self._curves_labels = []
            if self._marker_curves:
                self._marker_curves.reattach()
        self._clear_qpoint_items()
        # Bars next to blanked maps must not keep claiming the previous
        # scan's value range; keys with surviving data keep their bar.
        for key, image in self._heatmap_images().items():
            if image is None or image.image is None:
                self._hide_heatmap_bars(key)

    # ------------------------------------------------------------------
    # Clustering helpers (used by 2D + curves + heatmap mixins)
    # ------------------------------------------------------------------

    _cluster_nominal = staticmethod(_cluster_nominal_fn)
    _nominal_key = staticmethod(_nominal_key_fn)

    @staticmethod
    def _find_load_line_intersections(
        points: List[Dict], ub: float, ra: float,
    ) -> List[Dict]:
        """Find intersections of load line Ia=(Ub-Ua)/Ra with Ia(Ua) curves.

        Thin wrapper around ``amplifier.find_intersections`` exposed on
        the renderer so plot code can call it without importing the
        amplifier package directly.
        """
        from lm19 import amplifier
        if not points or ra <= 0 or ub <= 0:
            return []
        ll = amplifier.ResistiveLoadLine(ub, ra)
        return amplifier.find_intersections(points, ll)

    @staticmethod
    def _interp_intersection(pts: List[Dict], target_ug1: float) -> Optional[Dict]:
        """Interpolate/extrapolate Ia and Ua at target_ug1 from intersection list.

        Thin wrapper around amplifier.interp_intersection.
        """
        from lm19 import amplifier
        return amplifier.interp_intersection(pts, target_ug1)

    @staticmethod
    def _compute_5point_distortion(
        intersections: List[Dict], ug1_bias: float = None,
        half_swing: float = None,
    ) -> Optional[Dict]:
        """Compute HD2, HD3, Pout from load line intersections using 5-point method.

        Thin wrapper around amplifier.compute_distortion.
        """
        from lm19 import amplifier
        return amplifier.compute_distortion(intersections, ug1_bias, half_swing)

    @staticmethod
    def compute_imd(intersections: List[Dict], ug1_bias: float = None,
                    half_swing: float = None) -> Optional[Dict]:
        """Compute intermodulation distortion from transfer curve nonlinearity.

        Thin wrapper around amplifier.compute_imd.
        """
        from lm19 import amplifier
        return amplifier.compute_imd(intersections, ug1_bias, half_swing)

    def compute_ra_sweep(
        self, points: List[Dict], ub: float,
        ra_min: float = 0.5, ra_max: float = 50.0, steps: int = 60,
        ug1_bias: float = None, half_swing: float = None,
        ug2_filter: Optional[float] = None,
        transformer: bool = False, ra_dc: float = 0.05,
    ) -> List[Dict]:
        """Sweep Ra and compute HD2, HD3, Pout for each value.

        Thin wrapper around amplifier.sweep_ra. ``ug2_filter`` restricts the
        analysis to one screen-grid level (matching the 2D load line); without
        it HD/Pout would be computed over all Ug2 families mixed together.
        """
        from lm19 import amplifier
        return amplifier.sweep_ra(
            points, ub, ra_min, ra_max, ug1_bias, half_swing,
            ug2_filter=ug2_filter, steps=steps,
            transformer=transformer, ra_dc=ra_dc,
        )

    def apply_ranges(self, ua_min: float, ua_max: float, ia_max: float, ug1_min: float, ug1_max: float) -> None:
        self.plot.setXRange(ua_min, ua_max, padding=PLOT_PADDING)
        self.plot.setYRange(0, ia_max, padding=PLOT_PADDING)
        if self.transfer_plot is not None:
            self.transfer_plot.setXRange(ug1_min, ug1_max, padding=PLOT_PADDING)
            self.transfer_plot.setYRange(0, ia_max, padding=PLOT_PADDING)
        self.contour_plot.setXRange(ua_min, ua_max, padding=PLOT_PADDING)
        self.contour_plot.setYRange(ug1_min, ug1_max, padding=PLOT_PADDING)

    def apply_ia_axis(self, ia_max: float) -> None:
        self.plot.setYRange(0, ia_max, padding=PLOT_PADDING)
        if self.transfer_plot is not None:
            self.transfer_plot.setYRange(0, ia_max, padding=PLOT_PADDING)

    def apply_ua_axis(self, ua_min: float, ua_max: float) -> None:
        self.plot.setXRange(ua_min, ua_max, padding=PLOT_PADDING)
        self.contour_plot.setXRange(ua_min, ua_max, padding=PLOT_PADDING)

    def configure_base(self, grid_alpha: float = DEFAULT_GRID_ALPHA) -> None:
        self.plot.setLabel("left", t('plot.Ia_mA'))
        self.plot.setLabel("bottom", t('plot.Ua_V'))
        self.plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.transfer_plot is not None:
            self.transfer_plot.setLabel("left", t('plot.Ia_mA'))
            self.transfer_plot.setLabel("bottom", t('plot.Ug1_V'))
            self.transfer_plot.showGrid(x=True, y=True, alpha=grid_alpha)
        self.contour_plot.setLabel("left", t('plot.Ug1_V'))
        self.contour_plot.setLabel("bottom", t('plot.Ua_V'))
        self.contour_plot.showGrid(x=True, y=True, alpha=grid_alpha)

    def _is_triode_eff(self, points) -> bool:
        """True if every point belongs to a ug2-tracking series.

        Deliberately independent of ``is_triode``: that flag mirrors the
        LAMP SELECTOR, so mixing it in made picking a triode lamp after a
        pentode scan bypass the Ug2 slice filter — every screen level
        then averaged into one Ia(Ua, Ug1) grid and the maps reported
        plausible but wrong Gm/Rp. Which series track Ug2 is resolved
        from the data by ``PlotManager._is_sid_ug2_track`` and arrives
        here as ``track_sids`` (a true-triode scan is always in it).
        """
        if not self.track_sids:
            return False
        return all(p.get("series_id", 0) in self.track_sids for p in points)

    # ------------------------------------------------------------------
    # Matching delta plot (standalone, for dialog)
    # ------------------------------------------------------------------

    @staticmethod
    def render_matching_dialog(plot_widget: pg.PlotWidget, curves: List[Dict],
                                mode: str = "absolute", line_width: float = 2.0) -> None:
        """Render matching delta curves on a given plot widget.

        Args:
            plot_widget: target PlotWidget
            curves: output from quality.compute_matching_curves()
            mode: 'absolute' for mA, 'percent' for %
            line_width: curve width
        """
        palette = list(_SERIES_PALETTE)
        plot_widget.clear()

        for idx, curve in enumerate(curves):
            ug1 = curve["ug1"]
            xs = curve["ua_values"]
            ys = curve["delta_ia"] if mode == "absolute" else curve["delta_pct"]
            color = palette[idx % len(palette)]
            plot_widget.plot(
                xs, ys,
                pen=pg.mkPen(color, width=line_width),
                symbol="o", symbolSize=3, symbolBrush=color,
            )
            if xs and ys:
                label = pg.TextItem(format_label("ug1_short", ug1),
                                       color=color, anchor=(0, 1))
                label.setPos(xs[-1], ys[-1])
                plot_widget.addItem(label)

        # Zero reference line
        plot_widget.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(_COLOR_ZONE, width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
        ))

        if mode == "absolute":
            plot_widget.setLabel("left", t('plot.Delta_Ia_mA'))
        else:
            plot_widget.setLabel("left", t('plot.Delta_Ia_pct'))
        plot_widget.setLabel("bottom", t('plot.Ua_V'))

    # ------------------------------------------------------------------
    # Configure base (extended for new plots)
    # ------------------------------------------------------------------

    def configure_base_extended(self, grid_alpha: float = 0.3) -> None:
        """Configure axes and grid for the new plot widgets."""
        if self.gm_plot is not None:
            self.gm_plot.setLabel("left", t('plot.Ug1_V'))
            self.gm_plot.setLabel("bottom", t('plot.Ua_V'))
            self.gm_plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.rp_plot is not None:
            self.rp_plot.setLabel("left", t('plot.Ug1_V'))
            self.rp_plot.setLabel("bottom", t('plot.Ua_V'))
            self.rp_plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.mu_plot is not None:
            self.mu_plot.setLabel("left", t('plot.Ug1_V'))
            self.mu_plot.setLabel("bottom", t('plot.Ua_V'))
            self.mu_plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.pa_map_plot is not None:
            self.pa_map_plot.setLabel("left", t('plot.Ug1_V'))
            self.pa_map_plot.setLabel("bottom", t('plot.Ua_V'))
            self.pa_map_plot.showGrid(x=True, y=True, alpha=grid_alpha)
        if self.curves_plot is not None:
            self.curves_plot.showGrid(x=True, y=True, alpha=grid_alpha)

