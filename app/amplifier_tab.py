"""Amplifier design analysis tab — plots only.

Contains the two analysis plots (THD vs Amplitude, HD vs Ra). Controls
live in ``AmpControlPanel``, computation lives in ``AmplifierEngine``.
This tab receives an ``AnalysisResult`` and renders it.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lm19.amplifier import (
    PushPullLoadLine,
    compute_nfb_effect,
)
from lm19.amp_engine import (
    SOURCE_MEASUREMENTS, AnalysisResult, MIN_HALF_SWING_V, SourceResult,
)
from app.ui_theme import (
    BALANCE_GOOD_MAX,
    BALANCE_WARN_MAX,
    COLOR_LEVEL_BAD,
    COLOR_LEVEL_GOOD,
    COLOR_LEVEL_WARN,
    COLOR_MUTED_TEXT,
    COLOR_POUT,
    COLOR_WHITE,
    HEADROOM_GOOD_MIN,
    HEADROOM_WARN_MIN,
    PA_RATIO_GOOD_MAX,
    PA_RATIO_WARN_MAX,
    SOURCE_COLORS,
    SOURCE_COLOR_DEFAULT,
    THD_GOOD_MAX,
    THD_WARN_MAX,
    level_color_high_good,
    level_color_low_good,
)
from i18n_setup import t
from lm19.constants import MW_PER_W
from lm19.plot_style import (
    DEFAULT_GRID_ALPHA,
    COLOR_HD4, COLOR_HD5, COLOR_GAIN, COLOR_ZOUT, COLOR_PA,
    COLOR_OPT_PARETO, COLOR_TOOLTIP_BG, COLOR_TOOLTIP_BORDER,
)
from lm19.amplifier.constants import (
    CIRCUIT_PP,
)

log = logging.getLogger(__name__)

# ── amplifier_tab local constants ─────────────────────────────────────
LEGEND_OFFSET = (30, 10)       # (x, y) pixel offset for plot legends
SPLITTER_INITIAL = [500, 500]  # initial splitter sizes
THICK_WIDTH_FACTOR = 1.33      # THD line width = base * this factor
DEFAULT_BASE_WIDTH = 2.0       # fallback when no spinbox available
POUT_YRANGE_MARGIN = 1.1      # multiply max Pout by this for Y-range
THD_REF_LINE_PCT = 1.0         # horizontal reference line at 1% THD
PERCENT = 100.0                # multiplication factor for percentage
POUT_W_DECIMALS = 3            # decimal places for Pout in watts
HD45_DISPLAY_THRESHOLD_PCT = 0.1  # show HD4/HD5 in text if above this


def _fmt_pout_w(pout_mw: float) -> str:
    """Format Pout from mW to W with consistent precision."""
    return f"{pout_mw / MW_PER_W:.{POUT_W_DECIMALS}f}"


def _format_method_tag(stage: Dict) -> str:
    """Format method tag including SRK cross-check status."""
    srk_check = stage.get("srk_check")
    if srk_check == "ok":
        return t("amp.method_srk_ok")
    if srk_check == "divergence":
        pct = stage.get("srk_divergence_pct", 0)
        return t("amp.method_srk_divergence", pct=f"{pct:.0f}")
    return f"[{stage.get('method', 'numerical')}]"


class AmplifierTab(QWidget):
    """Amplifier design analysis sub-tab — plots only.

    Placed as a tab in plot_tabs. Receives AnalysisResult from the
    engine and renders two plots + returns results HTML.
    """

    ra_clicked = Signal(float)  # emitted when user clicks on Ra plot
    # (ub, ug2, ug1, ra, swing, ul_tap) — ul_tap included so applying a
    # point reproduces the optimizer's numbers (ML-029: a 43%-tap optimum
    # applied with a stale tap spin silently shows different THD/Pout).
    pareto_clicked = Signal(float, float, float, float, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._amp_sweep_data: List[Dict] = []  # cached for left plot crosshair
        self._ra_sweep_data: List[Dict] = []   # cached for right plot crosshair
        # Whose sweep the crosshair interpolates when several sources are
        # overlaid (None = single source, no tag needed). Rule:
        # every displayed number carries its source.
        self._amp_sweep_source: Optional[str] = None
        self._ra_sweep_source: Optional[str] = None
        self._pareto_data: List = []           # cached OptPoint list for Pareto click
        self._pareto_mode = False              # True = left plot shows Pareto
        self._line_width_spin = None  # set via set_line_width_spin()
        self._build_ui()

    def set_line_width_spin(self, spin) -> None:
        """Connect to the main UI line width spinbox."""
        self._line_width_spin = spin

    @property
    def _pen_thin(self) -> float:
        """Base pen width from UI spinbox or default."""
        if self._line_width_spin is not None:
            return self._line_width_spin.value()
        return DEFAULT_BASE_WIDTH

    @property
    def _pen_thick(self) -> float:
        """Thick pen width (for THD main curve)."""
        return self._pen_thin * THICK_WIDTH_FACTOR

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Plots
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: stacked widget (THD vs Amplitude / Pareto)
        self._left_stack = QStackedWidget()

        # Page 0: THD vs Amplitude
        self.thd_pout_plot = pg.PlotWidget(title=t("amp.thd_vs_amplitude"))
        self.thd_pout_plot.setLabel("left", t("amp.axis_hd_thd_pct"))
        self.thd_pout_plot.setLabel("bottom", t("amp.half_swing_v"))
        self.thd_pout_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        self.thd_pout_plot.addLegend(offset=LEGEND_OFFSET)

        self.thd_pout_vb2 = pg.ViewBox()
        self.thd_pout_plot.scene().addItem(self.thd_pout_vb2)
        self.thd_pout_plot.getAxis("right").linkToView(self.thd_pout_vb2)
        self.thd_pout_plot.getAxis("right").setLabel(t("amp.axis_pout_w"))
        self.thd_pout_plot.showAxis("right")
        self._sync_thd_views()
        self.thd_pout_plot.getViewBox().sigResized.connect(self._sync_thd_views)
        self._restore_amp_crosshair()
        self.thd_pout_plot.setMouseTracking(True)
        self.thd_pout_plot.viewport().setMouseTracking(True)
        self._amp_mouse_proxy = pg.SignalProxy(
            self.thd_pout_plot.scene().sigMouseMoved,
            rateLimit=60, slot=lambda args: self._on_amp_mouse_moved(args[0]),
        )
        self._left_stack.addWidget(self.thd_pout_plot)  # index 0

        # Page 1: Pareto (THD vs Pout)
        self.pareto_plot = pg.PlotWidget(title=t("amp.pareto_title"))
        self.pareto_plot.setLabel("left", t("amp.axis_thd_pct"))
        self.pareto_plot.setLabel("bottom", t("amp.axis_pout_pareto_w"))
        self.pareto_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        self.pareto_plot.addLegend(offset=LEGEND_OFFSET)
        self.pareto_plot.setMouseTracking(True)
        self.pareto_plot.viewport().setMouseTracking(True)
        self._restore_pareto_crosshair()
        self._pareto_mouse_proxy = pg.SignalProxy(
            self.pareto_plot.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_pareto_mouse_proxy,
        )
        self.pareto_plot.scene().sigMouseClicked.connect(self._on_pareto_clicked)
        self._left_stack.addWidget(self.pareto_plot)  # index 1
        splitter.addWidget(self._left_stack)

        self.hd_ra_plot = pg.PlotWidget(title=t("amp.hd_vs_ra"))
        self.hd_ra_plot.setLabel("left", t("amp.axis_hd_thd_pct"))
        self.hd_ra_plot.setLabel("bottom", t("amp.axis_ra_kohm"))
        self.hd_ra_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        self.hd_ra_plot.addLegend(offset=LEGEND_OFFSET)

        self.hd_ra_vb2 = pg.ViewBox()
        self.hd_ra_plot.scene().addItem(self.hd_ra_vb2)
        self.hd_ra_plot.getAxis("right").linkToView(self.hd_ra_vb2)
        self.hd_ra_plot.getAxis("right").setLabel(t("amp.axis_pout_w"))
        self.hd_ra_plot.showAxis("right")
        self._sync_ra_views()
        self.hd_ra_plot.getViewBox().sigResized.connect(self._sync_ra_views)

        # Crosshair on Ra plot (will be re-created after each clear)
        self._restore_ra_crosshair()
        self.hd_ra_plot.setMouseTracking(True)
        self.hd_ra_plot.viewport().setMouseTracking(True)
        self._ra_mouse_proxy = pg.SignalProxy(
            self.hd_ra_plot.scene().sigMouseMoved,
            rateLimit=60, slot=lambda args: self._on_ra_mouse_moved(args[0]),
        )
        self.hd_ra_plot.scene().sigMouseClicked.connect(self._on_ra_mouse_clicked)

        splitter.addWidget(self.hd_ra_plot)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(SPLITTER_INITIAL)
        layout.addWidget(splitter, stretch=1)

        # Tooltips as QLabel on viewport (renders on top of scene, like CurveMarker)
        self._amp_tooltip = self._make_tooltip(self.thd_pout_plot)
        self._ra_tooltip = self._make_tooltip(self.hd_ra_plot)
        self._pareto_tooltip = self._make_tooltip(self.pareto_plot)

    def _sync_thd_views(self) -> None:
        self.thd_pout_vb2.setGeometry(self.thd_pout_plot.getViewBox().sceneBoundingRect())

    def _sync_ra_views(self) -> None:
        self.hd_ra_vb2.setGeometry(self.hd_ra_plot.getViewBox().sceneBoundingRect())

    # ── Public API ────────────────────────────────────────────────

    def render(self, result: AnalysisResult) -> str:
        """Render analysis result on plots and return results HTML.

        Supports multi-source overlay: each source gets a distinct color
        on the plots. Results show a comparison table for 2+ sources.

        Args:
            result: AnalysisResult from AmplifierEngine.analyze().

        Returns:
            HTML string for the results label.
        """
        if result.error:
            self._clear_plots()
            return self._error_html(result.error)

        if not result.per_source:
            self._clear_plots()
            return t("amp.no_data")

        # Render plots with overlay
        self._render_thd_pout_multi(result)
        self._render_hd_ra_multi(result)

        return self.format_results_html(result)

    def format_results_html(self, result: AnalysisResult) -> str:
        """Results HTML WITHOUT touching the plots.

        The PDF export rebuilds this text under the report language
        (``i18n_setup.locale_override``) — re-rendering the plots there
        would flip the on-screen legend language as a side effect.
        """
        if result.error:
            return self._error_html(result.error)
        if not result.per_source:
            return t("amp.no_data")
        if result.circuit == CIRCUIT_PP:
            return self._format_pp_results(result)
        if len(result.per_source) > 1:
            return self._format_multi_source_results(result)
        return self._format_se_results(result)

    def collect_warnings(self, result: AnalysisResult) -> List[str]:
        """Localized warning strings from all sources of an analysis run —
        feeds the main-window status-bar indicator (the results panel only
        renders the formatted source's block)."""
        from app.amplifier_report import warning_text
        out: List[str] = []
        for source_name, sr in result.per_source.items():
            for w in sr.warnings:
                text = warning_text(w)
                if len(result.per_source) > 1:
                    text = f"[{source_name}] {text}"
                out.append(text)
        return out

    def clear(self) -> None:
        """Clear plots and return empty state."""
        self._clear_plots()

    # ── Plot rendering ────────────────────────────────────────────

    def _clear_plots(self) -> None:
        self.thd_pout_plot.clear()
        self.thd_pout_vb2.clear()
        self.hd_ra_plot.clear()
        self.hd_ra_vb2.clear()
        self._amp_sweep_data = []
        self._ra_sweep_data = []
        self._amp_sweep_source = None
        self._ra_sweep_source = None

    # NOTE: single-source `_render_thd_pout` / `_render_hd_ra` variants
    # were dead code (render() always goes through the *_multi path) and
    # carried a units bug (raw mW on the W axis) — removed in a graph audit.

    # ── Multi-source overlay ─────────────────────────────────────

    @staticmethod
    def _source_color(name: str) -> str:
        return SOURCE_COLORS.get(name, SOURCE_COLOR_DEFAULT)

    def _render_thd_pout_multi(self, result: AnalysisResult) -> None:
        """THD vs Amplitude with multi-source overlay."""
        self.thd_pout_plot.clear()
        self.thd_pout_vb2.clear()
        self._restore_amp_crosshair()
        sources = list(result.per_source.items())
        if not sources:
            return
        # Cache first source sweep data for crosshair interpolation;
        # remember whose it is so the tooltip can say so in multi-source.
        first_name, first_sr = sources[0]
        self._amp_sweep_data = list(first_sr.sweep_amp) if first_sr.sweep_amp else []
        self._amp_sweep_source = first_name if len(sources) > 1 else None

        show_hd45 = result.params.show_hd45 if result.params else False
        single = len(sources) == 1
        all_pout_max = 0.0
        for i, (name, sr) in enumerate(sources):
            data = sr.sweep_amp
            if not data:
                continue
            x = [d["half_swing"] for d in data]
            hd2 = [d["hd2"] for d in data]
            hd3 = [d["hd3"] for d in data]
            thd = [d["thd"] for d in data]
            pout_w = [d["pout_mw"] / MW_PER_W for d in data]

            color = self._source_color(name)
            suffix = f" [{name}]" if not single else ""

            # HD2/HD3: per-source color in multi, fixed colors in single
            if single:
                self.thd_pout_plot.plot(x, hd2, pen=pg.mkPen(COLOR_LEVEL_BAD, width=self._pen_thin), name=t("amp.legend_hd2"))
                self.thd_pout_plot.plot(x, hd3, pen=pg.mkPen(COLOR_LEVEL_GOOD, width=self._pen_thin), name=t("amp.legend_hd3"))
            else:
                self.thd_pout_plot.plot(x, hd2, pen=pg.mkPen(color, width=self._pen_thin, style=Qt.PenStyle.DotLine), name=f"HD2%{suffix}")
                self.thd_pout_plot.plot(x, hd3, pen=pg.mkPen(color, width=self._pen_thin, style=Qt.PenStyle.DashDotLine), name=f"HD3%{suffix}")

            self.thd_pout_plot.plot(
                x, thd,
                pen=pg.mkPen(color if not single else COLOR_LEVEL_WARN, width=self._pen_thick),
                name=t("amp.legend_thd") + suffix,
            )

            if show_hd45:
                hd4 = [d.get("hd4", 0.0) for d in data]
                hd5 = [d.get("hd5", 0.0) for d in data]
                if any(v > 0 for v in hd4):
                    self.thd_pout_plot.plot(x, hd4, pen=pg.mkPen(COLOR_HD4, width=self._pen_thin), name=t("amp.legend_hd4") + suffix)
                if any(v > 0 for v in hd5):
                    self.thd_pout_plot.plot(x, hd5, pen=pg.mkPen(COLOR_HD5, width=self._pen_thin), name=t("amp.legend_hd5") + suffix)

            pout_pen = pg.mkPen(color if not single else COLOR_POUT, width=self._pen_thin, style=Qt.PenStyle.DashLine)
            pout_curve = pg.PlotCurveItem(x, pout_w, pen=pout_pen)
            self.thd_pout_vb2.addItem(pout_curve)
            if pout_w:
                all_pout_max = max(all_pout_max, max(pout_w))
            # Legend entry for Pout (secondary axis items don't appear in legend)
            self.thd_pout_plot.plot([], [], pen=pout_pen, name=t("amp.legend_pout") + suffix)

        if all_pout_max > 0:
            self.thd_pout_vb2.setRange(yRange=(0, all_pout_max * POUT_YRANGE_MARGIN))

        swing = result.params.half_swing
        if swing and swing > MIN_HALF_SWING_V:
            vline = pg.InfiniteLine(
                pos=swing, angle=90,
                pen=pg.mkPen(COLOR_WHITE, width=1, style=Qt.PenStyle.DotLine),
            )
            self.thd_pout_plot.addItem(vline)

        hline = pg.InfiniteLine(
            pos=THD_REF_LINE_PCT, angle=0,
            pen=pg.mkPen(COLOR_MUTED_TEXT, width=1, style=Qt.PenStyle.DashLine),
        )
        self.thd_pout_plot.addItem(hline)

    @staticmethod
    def _make_tooltip(plot_widget: pg.PlotWidget) -> QLabel:
        """Create a tooltip QLabel on the plot viewport (renders on top)."""
        tip = QLabel(plot_widget.viewport())
        tip.setStyleSheet(
            f"background-color: {COLOR_TOOLTIP_BG};"
            f" border: 1px solid {COLOR_TOOLTIP_BORDER};"
            " padding: 4px; font-size: 11px; color: #000;"
        )
        tip.hide()
        return tip

    @staticmethod
    def _position_tooltip(tip: QLabel, plot: pg.PlotWidget,
                          x: float, y: float) -> None:
        """Position tooltip near data point, keeping it inside viewport."""
        vb = plot.getPlotItem().vb
        scene_pos = vb.mapViewToScene(QPointF(x, y))
        widget_pos = plot.mapFromScene(scene_pos)
        tip_x = widget_pos.x() + 18
        tip_y = widget_pos.y() - 10
        vp = plot.viewport().rect()
        tw, th = tip.width(), tip.height()
        if tip_x + tw > vp.width():
            tip_x = widget_pos.x() - tw - 10
        if tip_x < 5:
            tip_x = 5
        if tip_y + th > vp.height():
            tip_y = vp.height() - th - 5
        if tip_y < 0:
            tip_y = 5
        tip.move(int(tip_x), int(tip_y))
        tip.raise_()
        tip.show()

    def _restore_amp_crosshair(self) -> None:
        """Re-add crosshair items to THD vs Amplitude plot after clear."""
        self._amp_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._amp_hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._amp_vline.setVisible(False)
        self._amp_hline.setVisible(False)
        self._amp_vline.setZValue(1000)
        self._amp_hline.setZValue(1000)
        self.thd_pout_plot.addItem(self._amp_vline, ignoreBounds=True)
        self.thd_pout_plot.addItem(self._amp_hline, ignoreBounds=True)

    def _restore_ra_crosshair(self) -> None:
        """Re-add crosshair items after plot clear."""
        self._ra_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._ra_hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._ra_vline.setVisible(False)
        self._ra_hline.setVisible(False)
        self._ra_vline.setZValue(1000)
        self._ra_hline.setZValue(1000)
        self.hd_ra_plot.addItem(self._ra_vline, ignoreBounds=True)
        self.hd_ra_plot.addItem(self._ra_hline, ignoreBounds=True)

    def _restore_pareto_crosshair(self) -> None:
        """Re-add crosshair items to Pareto plot after clear."""
        self._pareto_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._pareto_hline = pg.InfiniteLine(angle=0, pen=pg.mkPen(COLOR_MUTED_TEXT, width=1))
        self._pareto_vline.setVisible(False)
        self._pareto_hline.setVisible(False)
        self._pareto_vline.setZValue(1000)
        self._pareto_hline.setZValue(1000)
        self.pareto_plot.addItem(self._pareto_vline, ignoreBounds=True)
        self.pareto_plot.addItem(self._pareto_hline, ignoreBounds=True)

    def _render_hd_ra_multi(self, result: AnalysisResult) -> None:
        """HD vs Ra with multi-source overlay."""
        self.hd_ra_plot.clear()
        self.hd_ra_vb2.clear()
        self._restore_ra_crosshair()
        sources = list(result.per_source.items())
        if not sources:
            return

        show_hd45 = result.params.show_hd45 if result.params else False
        show_gzp = result.params.show_gzp if result.params else False
        single = len(sources) == 1
        all_pout_max = 0.0
        for i, (name, sr) in enumerate(sources):
            data = sr.sweep_ra
            if not data:
                continue
            x = [d["ra"] for d in data]
            hd2 = [d["hd2"] for d in data]
            hd3 = [d["hd3"] for d in data]
            thd = [d["thd"] for d in data]
            pout_w = [d["pout_mw"] / MW_PER_W for d in data]

            color = self._source_color(name)
            suffix = f" [{name}]" if not single else ""

            # HD2/HD3: per-source color in multi, fixed colors in single
            if single:
                self.hd_ra_plot.plot(x, hd2, pen=pg.mkPen(COLOR_LEVEL_BAD, width=self._pen_thin), name=t("amp.legend_hd2"))
                self.hd_ra_plot.plot(x, hd3, pen=pg.mkPen(COLOR_LEVEL_GOOD, width=self._pen_thin), name=t("amp.legend_hd3"))
            else:
                self.hd_ra_plot.plot(x, hd2, pen=pg.mkPen(color, width=self._pen_thin, style=Qt.PenStyle.DotLine), name=f"HD2%{suffix}")
                self.hd_ra_plot.plot(x, hd3, pen=pg.mkPen(color, width=self._pen_thin, style=Qt.PenStyle.DashDotLine), name=f"HD3%{suffix}")

            self.hd_ra_plot.plot(
                x, thd,
                pen=pg.mkPen(color if not single else COLOR_LEVEL_WARN, width=self._pen_thick),
                name=t("amp.legend_thd") + suffix,
            )

            if show_hd45:
                hd4 = [d.get("hd4", 0.0) for d in data]
                hd5 = [d.get("hd5", 0.0) for d in data]
                if any(v > 0 for v in hd4):
                    self.hd_ra_plot.plot(x, hd4, pen=pg.mkPen(COLOR_HD4, width=self._pen_thin), name=t("amp.legend_hd4") + suffix)
                if any(v > 0 for v in hd5):
                    self.hd_ra_plot.plot(x, hd5, pen=pg.mkPen(COLOR_HD5, width=self._pen_thin), name=t("amp.legend_hd5") + suffix)

            pout_pen = pg.mkPen(color if not single else COLOR_POUT, width=self._pen_thin, style=Qt.PenStyle.DashLine)
            pout_curve = pg.PlotCurveItem(x, pout_w, pen=pout_pen)
            self.hd_ra_vb2.addItem(pout_curve)
            if pout_w:
                all_pout_max = max(all_pout_max, max(pout_w))
            # Legend entry for Pout (secondary axis items don't appear in legend)
            self.hd_ra_plot.plot([], [], pen=pout_pen, name=t("amp.legend_pout") + suffix)

            # Gain/Zout/Pa curves (on primary axis — left)
            if show_gzp:
                gain = [d.get("gain", 0.0) for d in data]
                zout = [d.get("zout", 0.0) for d in data]
                pa_w = [d.get("pa_mw", 0.0) / 1000.0 for d in data]
                if any(v > 0 for v in gain):
                    self.hd_ra_plot.plot(x, gain, pen=pg.mkPen(COLOR_GAIN, width=self._pen_thin, style=Qt.PenStyle.DashLine),
                                         name=t("amp.legend_gain") + suffix)
                if any(v > 0 for v in zout):
                    self.hd_ra_plot.plot(x, zout, pen=pg.mkPen(COLOR_ZOUT, width=self._pen_thin, style=Qt.PenStyle.DashLine),
                                         name=t("amp.legend_zout") + suffix)
                if any(v > 0 for v in pa_w):
                    self.hd_ra_plot.plot(x, pa_w, pen=pg.mkPen(COLOR_PA, width=self._pen_thin, style=Qt.PenStyle.DashLine),
                                         name=t("amp.legend_pa") + suffix)
                    # Pa_max horizontal line
                    if result.params and result.params.pa_max > 0:
                        pa_max_line = pg.InfiniteLine(
                            pos=result.params.pa_max, angle=0,
                            pen=pg.mkPen(COLOR_PA, width=1, style=Qt.PenStyle.DotLine),
                        )
                        self.hd_ra_plot.addItem(pa_max_line)

        if all_pout_max > 0:
            self.hd_ra_vb2.setRange(yRange=(0, all_pout_max * POUT_YRANGE_MARGIN))

        vline = pg.InfiniteLine(
            pos=result.params.ra, angle=90,
            pen=pg.mkPen(COLOR_WHITE, width=1, style=Qt.PenStyle.DotLine),
        )
        self.hd_ra_plot.addItem(vline)

        # Markers: min THD and max Pout from first source
        first_name, first_sr = next(iter(result.per_source.items()),
                                    (None, None))
        if first_sr and first_sr.sweep_ra:
            self._ra_sweep_data = list(first_sr.sweep_ra)
            self._ra_sweep_source = (first_name
                                     if len(result.per_source) > 1 else None)
            self._add_ra_markers(first_sr.sweep_ra,
                                 source=self._ra_sweep_source)

    def _add_ra_markers(self, sweep_data: List[Dict],
                        source: Optional[str] = None) -> None:
        """Add min-THD and max-Pout markers to the Ra plot.

        ``source`` tags the labels when several sources are overlaid —
        the extrema come from ONE source's sweep, not the ensemble.
        """
        if not sweep_data:
            return
        tag = f"\n[{source}]" if source else ""

        # Min THD marker
        min_thd_pt = min(sweep_data, key=lambda d: d["thd"])
        self.hd_ra_plot.plot(
            [min_thd_pt["ra"]], [min_thd_pt["thd"]],
            pen=None, symbol="o", symbolSize=10,
            symbolBrush=pg.mkBrush(COLOR_LEVEL_GOOD),
            symbolPen=pg.mkPen(COLOR_WHITE, width=1),
        )
        min_label = pg.TextItem(
            f"min THD {min_thd_pt['thd']:.2f}%\nRa={min_thd_pt['ra']:.1f}"
            f"{tag}",
            color=COLOR_LEVEL_GOOD, anchor=(0, 1),
        )
        min_label.setPos(min_thd_pt["ra"], min_thd_pt["thd"])
        self.hd_ra_plot.addItem(min_label)

        # Max Pout marker (on secondary axis, but show on primary for visibility)
        max_pout_pt = max(sweep_data, key=lambda d: d["pout_mw"])
        max_pout_ra = max_pout_pt["ra"]
        # Find THD at max-Pout Ra to place marker on primary Y axis
        max_pout_thd = max_pout_pt["thd"]
        self.hd_ra_plot.plot(
            [max_pout_ra], [max_pout_thd],
            pen=None, symbol="s", symbolSize=10,
            symbolBrush=pg.mkBrush(COLOR_POUT),
            symbolPen=pg.mkPen(COLOR_WHITE, width=1),
        )
        pout_label = pg.TextItem(
            f"max Pout {_fmt_pout_w(max_pout_pt['pout_mw'])}W\nRa={max_pout_ra:.1f}"
            f"{tag}",
            color=COLOR_POUT, anchor=(0, 0),
        )
        pout_label.setPos(max_pout_ra, max_pout_thd)
        self.hd_ra_plot.addItem(pout_label)

    # ── Crosshair & click ────────────────────────────────────────

    def _interp_amp_sweep(self, swing_val: float) -> Optional[Dict]:
        """Interpolate sweep_amp data at a given half-swing value."""
        data = self._amp_sweep_data
        if len(data) < 2:
            return None
        x_arr = [d["half_swing"] for d in data]
        if swing_val < x_arr[0] or swing_val > x_arr[-1]:
            return None
        for j in range(len(x_arr) - 1):
            if x_arr[j] <= swing_val <= x_arr[j + 1]:
                frac = (swing_val - x_arr[j]) / max(x_arr[j + 1] - x_arr[j], 1e-9)
                result: Dict = {}
                for key in data[j]:
                    v0 = data[j][key]
                    v1 = data[j + 1][key]
                    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                        result[key] = v0 + (v1 - v0) * frac
                return result
        return None

    def _on_amp_mouse_moved(self, pos) -> None:
        """Update crosshair and readout on THD vs Amplitude plot."""
        vb = self.thd_pout_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._amp_vline.setVisible(False)
            self._amp_hline.setVisible(False)
            self._amp_tooltip.hide()
            return

        mouse_point = vb.mapSceneToView(pos)
        x_val = mouse_point.x()
        y_val = mouse_point.y()

        self._amp_vline.setPos(x_val)
        self._amp_hline.setPos(y_val)
        self._amp_vline.setVisible(True)
        self._amp_hline.setVisible(True)

        interp = self._interp_amp_sweep(x_val)
        if interp:
            self._amp_tooltip.setText(self._amp_tooltip_text(x_val, interp))
            self._amp_tooltip.adjustSize()
            self._position_tooltip(self._amp_tooltip, self.thd_pout_plot, x_val, y_val)
        else:
            self._amp_tooltip.hide()

    def _amp_tooltip_text(self, x_val: float, interp: Dict) -> str:
        """Crosshair readout for THD-vs-amplitude; tagged with the source
        when several are overlaid (the crosshair follows ONE sweep)."""
        head = (f"<b>[{self._amp_sweep_source}]</b><br>"
                if self._amp_sweep_source else "")
        return (
            f"{head}Swing={x_val:.1f}V<br>"
            f"THD={interp.get('thd', 0):.2f}%<br>"
            f"HD2={interp.get('hd2', 0):.2f}%  HD3={interp.get('hd3', 0):.2f}%<br>"
            f"Pout={_fmt_pout_w(interp.get('pout_mw', 0))}W"
        )

    def _interp_sweep(self, ra_val: float) -> Optional[Dict]:
        """Interpolate sweep_ra data at a given Ra value."""
        data = self._ra_sweep_data
        if len(data) < 2:
            return None
        ra_arr = [d["ra"] for d in data]
        if ra_val < ra_arr[0] or ra_val > ra_arr[-1]:
            return None
        # Find bracketing indices
        for j in range(len(ra_arr) - 1):
            if ra_arr[j] <= ra_val <= ra_arr[j + 1]:
                frac = (ra_val - ra_arr[j]) / max(ra_arr[j + 1] - ra_arr[j], 1e-9)
                result: Dict = {}
                for key in data[j]:
                    v0 = data[j][key]
                    v1 = data[j + 1][key]
                    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                        result[key] = v0 + (v1 - v0) * frac
                return result
        return None

    def _on_ra_mouse_moved(self, pos) -> None:
        """Update crosshair and readout on Ra plot mouse move."""
        vb = self.hd_ra_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._ra_vline.setVisible(False)
            self._ra_hline.setVisible(False)
            self._ra_tooltip.hide()
            return

        mouse_point = vb.mapSceneToView(pos)
        ra_val = mouse_point.x()
        thd_val = mouse_point.y()

        self._ra_vline.setPos(ra_val)
        self._ra_hline.setPos(thd_val)
        self._ra_vline.setVisible(True)
        self._ra_hline.setVisible(True)

        interp = self._interp_sweep(ra_val)
        if interp:
            self._ra_tooltip.setText(self._ra_tooltip_text(ra_val, interp))
            self._ra_tooltip.adjustSize()
            self._position_tooltip(self._ra_tooltip, self.hd_ra_plot, ra_val, thd_val)
        else:
            self._ra_tooltip.hide()

    def _ra_tooltip_text(self, ra_val: float, interp: Dict) -> str:
        """Crosshair readout for HD-vs-Ra; source-tagged in multi-source."""
        parts = []
        if self._ra_sweep_source:
            parts.append(f"<b>[{self._ra_sweep_source}]</b><br>")
        parts.append(
            f"Ra={ra_val:.1f} kΩ<br>"
            f"THD={interp.get('thd', 0):.2f}%<br>"
            f"HD2={interp.get('hd2', 0):.2f}%  HD3={interp.get('hd3', 0):.2f}%<br>"
            f"Pout={_fmt_pout_w(interp.get('pout_mw', 0))}W"
        )
        if "gain" in interp:
            parts.append(f"<br>Gain={interp['gain']:.1f}  Zout={interp.get('zout', 0):.1f}")
        if "pa_mw" in interp:
            parts.append(f"<br>Pa={interp['pa_mw'] / 1000:.2f}W")
        return "".join(parts)

    def _on_pareto_mouse_proxy(self, args) -> None:
        """SignalProxy slot for Pareto mouse move."""
        self._on_pareto_mouse_moved(args[0])

    def _on_pareto_mouse_moved(self, pos) -> None:
        """Update crosshair and readout on Pareto plot."""
        vb = self.pareto_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._pareto_vline.setVisible(False)
            self._pareto_hline.setVisible(False)
            self._pareto_tooltip.hide()
            return

        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        self._pareto_vline.setPos(mx)
        self._pareto_hline.setPos(my)
        self._pareto_vline.setVisible(True)
        self._pareto_hline.setVisible(True)

        nearest = self._find_nearest_pareto(mx, my)
        if nearest:
            parts = [
                f"<b>THD={nearest.thd:.2f}%  Pout={nearest.pout_mw / MW_PER_W:.3f}W</b><br>"
                f"Ub={nearest.ub:.0f}V  Ra={nearest.ra:.1f}kΩ  Ug1={nearest.ug1:.1f}V"
            ]
            if nearest.half_swing > 0:
                parts.append(f"<br>Swing={nearest.half_swing:.1f}V")
            if nearest.ug2 > 0:
                parts.append(f"<br>Ug2={nearest.ug2:.0f}V")
            parts.append(f"<br>Class {nearest.amp_class}  Pa={nearest.pa_mw / MW_PER_W:.2f}W")
            if getattr(nearest, "ul_tap", 0.0) > 0:
                parts.append(f"<br>UL tap {nearest.ul_tap * 100:.0f}%")
            self._pareto_tooltip.setText("".join(parts))
            self._pareto_tooltip.adjustSize()
            self._position_tooltip(self._pareto_tooltip, self.pareto_plot, mx, my)
        else:
            self._pareto_tooltip.hide()

    def _find_nearest_pareto(self, mx: float, my: float):
        """Find nearest Pareto point to mouse position (normalized distance)."""
        if not self._pareto_data:
            return None
        pout_range = max(p.pout_mw for p in self._pareto_data) - min(p.pout_mw for p in self._pareto_data)
        thd_range = max(p.thd for p in self._pareto_data) - min(p.thd for p in self._pareto_data)
        pout_scale = max(pout_range / MW_PER_W, 0.01)
        thd_scale = max(thd_range, 0.01)

        best_dist = float("inf")
        best_pt = None
        for pt in self._pareto_data:
            dx = (pt.pout_mw / MW_PER_W - mx) / pout_scale
            dy = (pt.thd - my) / thd_scale
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist = d
                best_pt = pt
        return best_pt

    def _on_ra_mouse_clicked(self, event) -> None:
        """Click on Ra plot → emit ra_clicked signal."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.hd_ra_plot.getViewBox()
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        ra_val = mouse_point.x()
        if ra_val > 0:
            self.ra_clicked.emit(ra_val)

    # ── Pareto plot ────────────────────────────────────────────────

    def render_pareto(self, opt_result) -> None:
        """Render Pareto front on the dedicated Pareto plot and switch to it.

        Args:
            opt_result: OptimizerResult from optimizer.
        """
        self._pareto_mode = True
        self.pareto_plot.clear()
        self._restore_pareto_crosshair()

        grid = opt_result.grid_points
        pareto = opt_result.pareto_front
        best = opt_result.best
        refined = opt_result.refined

        # All valid grid points (gray)
        valid = [p for p in grid if p.valid]
        if valid:
            x = [p.pout_mw / MW_PER_W for p in valid]
            y = [p.thd for p in valid]
            self.pareto_plot.plot(
                x, y, pen=None,
                symbol="o", symbolSize=4,
                symbolBrush=pg.mkBrush(COLOR_MUTED_TEXT),
                symbolPen=None,
            )

        # Invalid grid points (red, small)
        invalid = [p for p in grid if not p.valid]
        if invalid:
            x = [p.pout_mw / MW_PER_W for p in invalid]
            y = [p.thd for p in invalid]
            self.pareto_plot.plot(
                x, y, pen=None,
                symbol="x", symbolSize=4,
                symbolBrush=pg.mkBrush(COLOR_LEVEL_BAD),
                symbolPen=pg.mkPen(COLOR_LEVEL_BAD, width=0.5),
            )

        # Pareto front (green line + dots)
        if pareto:
            x = [p.pout_mw / MW_PER_W for p in pareto]
            y = [p.thd for p in pareto]
            self.pareto_plot.plot(
                x, y,
                pen=pg.mkPen(COLOR_LEVEL_GOOD, width=self._pen_thin, style=Qt.PenStyle.DashLine),
                symbol="o", symbolSize=6,
                symbolBrush=pg.mkBrush(COLOR_LEVEL_GOOD),
                symbolPen=pg.mkPen(COLOR_WHITE, width=0.5),
                name=t("amp.legend_pareto"),
            )

        # Refined Pareto front (cyan line + diamonds) — clickable
        refined_pareto = getattr(opt_result, "refined_pareto", [])
        if refined_pareto:
            self._pareto_data = list(refined_pareto)
            x = [p.pout_mw / MW_PER_W for p in refined_pareto]
            y = [p.thd for p in refined_pareto]
            self.pareto_plot.plot(
                x, y,
                pen=pg.mkPen(COLOR_OPT_PARETO, width=self._pen_thick),
                symbol="d", symbolSize=8,
                symbolBrush=pg.mkBrush(COLOR_OPT_PARETO),
                symbolPen=pg.mkPen(COLOR_WHITE, width=1),
                name=t("amp.legend_refined_pareto"),
            )
        else:
            self._pareto_data = list(pareto)

        # Best point (large yellow star)
        if best:
            self.pareto_plot.plot(
                [best.pout_mw / MW_PER_W], [best.thd],
                pen=None, symbol="star", symbolSize=14,
                symbolBrush=pg.mkBrush(COLOR_LEVEL_WARN),
                symbolPen=pg.mkPen(COLOR_WHITE, width=1),
                name=t("amp.legend_best"),
            )

        # Refined best point (cyan diamond, larger)
        if refined:
            self.pareto_plot.plot(
                [refined.pout_mw / MW_PER_W], [refined.thd],
                pen=None, symbol="star", symbolSize=14,
                symbolBrush=pg.mkBrush(COLOR_OPT_PARETO),
                symbolPen=pg.mkPen(COLOR_WHITE, width=1),
                name=t("amp.legend_refined"),
            )

        self._switch_stack_page(1)

    def _switch_stack_page(self, index: int) -> None:
        """Switch QStackedWidget page and fix geometry.

        QStackedWidget doesn't resize the new page when the parent
        widget is not visible (e.g. tab not active). Force resize.
        """
        self._left_stack.setCurrentIndex(index)
        page = self._left_stack.currentWidget()
        if page is not None:
            page.resize(self._left_stack.size())

    def clear_pareto(self) -> None:
        """Switch back to THD vs Amplitude plot."""
        self._pareto_mode = False
        self._switch_stack_page(0)

    def show_pareto(self) -> None:
        """Switch to Pareto plot (if data exists)."""
        if self._pareto_data:
            self._pareto_mode = True
            self._switch_stack_page(1)

    def _on_pareto_clicked(self, event) -> None:
        """Click on Pareto plot → find nearest point, emit signal."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._pareto_data:
            return
        vb = self.pareto_plot.getViewBox()
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse = vb.mapSceneToView(pos)
        best_pt = self._find_nearest_pareto(mouse.x(), mouse.y())

        if best_pt is not None:
            self.pareto_clicked.emit(
                best_pt.ub, best_pt.ug2, best_pt.ug1, best_pt.ra,
                best_pt.half_swing, getattr(best_pt, "ul_tap", 0.0) or 0.0,
            )

    # ── Results formatting ────────────────────────────────────────

    def _error_html(self, error: str) -> str:
        """Map error code to human-readable HTML."""
        mapping = {
            "no_data": t("amp.no_data"),
            "enable_ll": t("amp.enable_ll"),
            "not_enough_isects": t("amp.not_enough_isects"),
            "pp_no_tube_b": t("amp.pp_no_tube_b"),
        }
        return mapping.get(error, t("amp.analysis_failed"))

    def _dist_error_html(self, code: Optional[str]) -> str:
        """Thin delegate — see ``app.amplifier_report.dist_error_html``."""
        from app.amplifier_report import dist_error_html
        return dist_error_html(code)

    def _format_se_results(self, result: AnalysisResult) -> str:
        """Format SE/CF/SE_xfmr results as HTML (single source)."""
        source_names = list(result.per_source.keys())
        primary = result.per_source[source_names[0]]
        pa_max = result.params.pa_max
        ug2 = result.params.ug2_filter

        return self._format_source_results(primary, pa_max, result.circuit, ug2_filter=ug2)

    def _format_multi_source_results(self, result: AnalysisResult) -> str:
        """Format multi-source comparison as HTML delta table."""
        sources = list(result.per_source.items())
        ref_name, ref_sr = sources[0]
        pa_max = result.params.pa_max
        ug2 = result.params.ug2_filter

        # Header: primary source results
        parts = [self._format_source_results(ref_sr, pa_max, result.circuit, ug2_filter=ug2)]

        # Delta table
        rows = []
        metrics = [
            ("HD2, %", "hd2"), ("HD3, %", "hd3"), ("THD, %", "thd"),
            ("Pout, W", "pout_mw"),
        ]

        # Table header
        hdr = "<tr><th></th>"
        for name, _ in sources:
            color = self._source_color(name)
            hdr += f"<th style='color:{color}'>{name}</th>"
        hdr += "<th>Δ%</th></tr>"
        rows.append(hdr)

        ref_dist = ref_sr.dist or {}
        for label, key in metrics:
            ref_val = ref_dist.get(key)
            row = f"<tr><td><b>{label}</b></td>"
            vals = []
            for name, sr in sources:
                d = sr.dist or {}
                v = d.get(key)
                vals.append(v)
                if v is not None:
                    cell = _fmt_pout_w(v) if key == "pout_mw" else f"{v:.2f}"
                    row += f"<td>{cell}</td>"
                else:
                    row += "<td>—</td>"

            # Delta: compare last vs first
            if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None and vals[0] != 0:
                delta = (vals[-1] - vals[0]) / abs(vals[0]) * PERCENT
                sign = "+" if delta > 0 else ""
                row += f"<td>{sign}{delta:.1f}%</td>"
            else:
                row += "<td>—</td>"
            row += "</tr>"
            rows.append(row)

        # Gain comparison if available
        gains = [(name, sr.stage.get("gain", 0) if sr.stage else 0) for name, sr in sources]
        if all(g > 0 for _, g in gains):
            row = "<tr><td><b>Gain</b></td>"
            for _, g in gains:
                row += f"<td>{g:.1f}</td>"
            if len(gains) >= 2 and gains[0][1] != 0:
                delta = (gains[-1][1] - gains[0][1]) / abs(gains[0][1]) * PERCENT
                sign = "+" if delta > 0 else ""
                row += f"<td>{sign}{delta:.1f}%</td>"
            else:
                row += "<td>—</td>"
            row += "</tr>"
            rows.append(row)

        table = (
            "<br><table border='1' cellpadding='3' cellspacing='0' "
            "style='border-collapse:collapse; font-size:11px;'>"
            + "".join(rows) + "</table>"
        )
        parts.append(table)
        return "".join(parts)

    def _format_pp_results(self, result: AnalysisResult) -> str:
        """Format Push-Pull results as HTML."""
        from app.amplifier_report import format_warnings_html
        parts: List[str] = []
        sr_meas = result.per_source.get(SOURCE_MEASUREMENTS)
        if sr_meas is not None:
            parts.extend(format_warnings_html(sr_meas.warnings))
        pp_dist = result.pp_dist
        ll = result.load_line

        if pp_dist and isinstance(ll, PushPullLoadLine):
            if pp_dist.get("manual_swing_clamped"):
                # ML-052: same notice the SE panel shows.
                parts.append(t(
                    "amp.swing_clamped",
                    requested=f"{pp_dist.get('requested_half_swing', 0.0):.1f}",
                    used=f"{pp_dist.get('half_swing', 0.0):.1f}",
                ))
            if pp_dist.get("b_extrapolation_span_v", 0.0) > 0:
                # ML-139 notice tier: part of the ANALYZED window is
                # derived from the cutoff extrapolation, not measured.
                parts.append(t(
                    "amp.b_extrap_notice",
                    span=f"{pp_dist.get('b_extrapolation_span_v', 0.0):.1f}",
                ))
            thd_color = level_color_low_good(pp_dist["thd"], THD_GOOD_MAX, THD_WARN_MAX)
            balance_color = level_color_low_good(
                pp_dist["balance_error"], BALANCE_GOOD_MAX, BALANCE_WARN_MAX
            )
            parts.append(
                t("amp.pp_summary", ra_aa=f"{ll.ra_aa:.1f}", ra_per_tube=f"{ll.ra_per_tube:.2f}")
            )
            parts.append(
                t(
                    "amp.thd_line",
                    hd2=f"{pp_dist['hd2']:.2f}",
                    hd3=f"{pp_dist['hd3']:.2f}",
                    thd=f"{pp_dist['thd']:.2f}",
                    pout=_fmt_pout_w(pp_dist['pout_mw']),
                    thd_color=thd_color,
                )
            )
            hd45_parts = []
            hd4 = pp_dist.get("hd4", 0.0)
            hd5 = pp_dist.get("hd5", 0.0)
            if hd4 >= HD45_DISPLAY_THRESHOLD_PCT:
                hd45_parts.append(f"HD4={hd4:.2f}%")
            if hd5 >= HD45_DISPLAY_THRESHOLD_PCT:
                hd45_parts.append(f"HD5={hd5:.2f}%")
            if hd45_parts:
                parts.append(
                    f"<span style='color:{COLOR_MUTED_TEXT}'>"
                    f"{'  '.join(hd45_parts)}</span>"
                )
            # Swing block: Ia range / drive / P1 / Iq per
            # tube from pp_dist; Ua_pp for PP — via Ra_per_tube (the
            # composite carries no ua_min/max).
            from app.amplifier_report import _append_swing_lines
            i_min, i_max = pp_dist.get("i_min"), pp_dist.get("i_max")
            if i_min is not None and i_max is not None:
                ua_pp = abs(i_max - i_min) * ll.ra_per_tube
                parts.append(t("amp.swing_ua_pp_only",
                               upp=f"{ua_pp:.0f}"))
            _append_swing_lines(parts, pp_dist)
            balance_value = f"{pp_dist['balance_error']:.2f}"
            parts.append(
                f"<span style='color:{balance_color}'>"
                f"{t('amp.balance_error', value=balance_value)}</span> "
                f"{t('amp.balance_note')}"
            )

            # HD method actually used for PP composite distortion
            # (5point / chebyshev_pp / dft_pp set by pp_distortion variants).
            pp_method = pp_dist.get("method")
            if pp_method:
                parts.append(
                    f"<span style='color:{COLOR_MUTED_TEXT}'>"
                    f"{t('amp.opt_method_used', method=pp_method)}</span>"
                )

            # Headroom from first source
            source_names = list(result.per_source.keys())
            if source_names:
                headroom = result.per_source[source_names[0]].headroom
                if headroom:
                    parts.append(
                        t(
                            "amp.headroom_line",
                            swing=f"{headroom['max_swing']:.1f}",
                            neg=headroom["clip_neg"],
                            pos=headroom["clip_pos"],
                        )
                    )
        else:
            parts.append(self._dist_error_html(result.pp_dist_error))

        return "<br>".join(parts)

    def _format_source_results(
        self, sr: SourceResult, pa_max: float, circuit: str,
        ug2_filter: Optional[float] = None,
    ) -> str:
        """Thin delegate — see ``app.amplifier_report.format_source_results``."""
        from app.amplifier_report import format_source_results
        return format_source_results(sr, pa_max, circuit, ug2_filter=ug2_filter)
