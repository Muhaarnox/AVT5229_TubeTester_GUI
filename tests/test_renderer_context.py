"""``RendererContext`` property-delegation tests.

``PlotRenderer`` keeps its shared mutable state (``is_triode``,
``track_sids``, cluster thresholds, caches, …) in a ``RendererContext``
dataclass so per-plot-type mixins can read/write it via ``self.ctx``.

``PlotRenderer.is_triode = X`` and ``r.track_sids = {1, 2}`` must keep
working because ``PlotManager.set_triode`` and several tests write
directly to those attributes. The renderer implements this with
property descriptors that forward to ``self.ctx.<field>``; these tests
pin that contract.
"""

from __future__ import annotations

import os
import sys
import unittest

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication

_qapp = QApplication.instance() or QApplication([])


def _make_renderer():
    """Build a minimal PlotRenderer for state-delegation tests."""
    import pyqtgraph as pg
    from app.plotting.renderer import PlotRenderer
    plot = pg.PlotWidget()
    contour_plot = pg.PlotWidget()
    contour_image = pg.ImageItem()
    return PlotRenderer(plot, contour_plot, contour_image)


class TestRendererContextDelegation(unittest.TestCase):
    """``r.is_triode``/``r.track_sids`` etc. should read/write ``r.ctx``.

    Direct attribute access on the renderer must continue to work so
    existing ``PlotManager.set_triode`` and tests don't break.
    """

    def test_is_triode_default_false(self) -> None:
        r = _make_renderer()
        self.assertFalse(r.is_triode)
        self.assertFalse(r.ctx.is_triode)

    def test_is_triode_write_via_property(self) -> None:
        r = _make_renderer()
        r.is_triode = True
        self.assertTrue(r.is_triode)
        self.assertTrue(r.ctx.is_triode)

    def test_is_triode_write_via_ctx(self) -> None:
        r = _make_renderer()
        r.ctx.is_triode = True
        self.assertTrue(r.is_triode)
        self.assertTrue(r.ctx.is_triode)

    def test_track_sids_default_empty(self) -> None:
        r = _make_renderer()
        self.assertEqual(r.track_sids, set())
        self.assertEqual(r.ctx.track_sids, set())

    def test_track_sids_write_via_property(self) -> None:
        r = _make_renderer()
        r.track_sids = {1, 2, 3}
        self.assertEqual(r.track_sids, {1, 2, 3})
        self.assertEqual(r.ctx.track_sids, {1, 2, 3})

    def test_track_sids_mutate_in_place(self) -> None:
        r = _make_renderer()
        r.track_sids.add(7)
        self.assertIn(7, r.ctx.track_sids)

    def test_ug2_cluster_thr_read(self) -> None:
        r = _make_renderer()
        # Default value comes from constants — just ensure attr exists,
        # is numeric, and reads through both r.* and r.ctx.*
        self.assertEqual(r.ug2_cluster_thr, r.ctx.ug2_cluster_thr)
        self.assertIsInstance(r.ug2_cluster_thr, (int, float))

    def test_ctx_is_dataclass_instance(self) -> None:
        from dataclasses import is_dataclass
        from app.plotting.renderer import RendererContext  # noqa: F401
        r = _make_renderer()
        self.assertTrue(is_dataclass(r.ctx))

    def test_ctx_default_construction(self) -> None:
        from app.plotting.renderer import RendererContext
        ctx = RendererContext()
        self.assertFalse(ctx.is_triode)
        self.assertEqual(ctx.track_sids, set())


class TestCurvesPlotMixinExtraction(unittest.TestCase):
    """The ``render_curves`` family lives in ``_CurvesPlotMixin``.

    Regression guards: the mixin must be in ``PlotRenderer.__mro__``
    and every method it owns must remain accessible via ``r.<name>``
    on a ``PlotRenderer`` instance, so external callers (e.g.
    ``PlotManager``) keep working.
    """

    def test_curves_mixin_in_mro(self) -> None:
        from app.plotting._curves_plot_mixin import _CurvesPlotMixin
        from app.plotting.renderer import PlotRenderer
        self.assertIn(_CurvesPlotMixin, PlotRenderer.__mro__)

    def test_curves_methods_inherited(self) -> None:
        """Every mixin method is reachable on the renderer instance."""
        r = _make_renderer()
        for name in (
            "render_curves",
            "_render_curves_grid",
            "_render_curves_load_line",
            "_render_curves_vs_ua",
            "_render_curves_vs_ug1",
            "_render_curves_raw",
            "_render_curves_raw_multi_ug2",
            "_finalize_curves_marker",
            "_grid_extra",
            "_grid_extra_ua",
            "_raw_extra",
            "_raw_y_values",
            "_plot_curve_line",
        ):
            self.assertTrue(callable(getattr(r, name, None)),
                            f"PlotRenderer missing method {name!r}")

    def test_curves_y_labels_class_attr(self) -> None:
        """``_CURVES_Y_LABELS`` (label key map) lives on the mixin."""
        from app.plotting._curves_plot_mixin import _CurvesPlotMixin
        self.assertIn("Gm", _CurvesPlotMixin._CURVES_Y_LABELS)
        self.assertIn("Rp", _CurvesPlotMixin._CURVES_Y_LABELS)
        self.assertIn("mu", _CurvesPlotMixin._CURVES_Y_LABELS)

    def test_render_curves_smoke_no_curves_plot(self) -> None:
        """``render_curves`` with no target plot is a safe no-op."""
        import pyqtgraph as pg
        from app.plotting.renderer import PlotRenderer
        # PlotRenderer with curves_plot=None — render_curves should early-return.
        r = PlotRenderer(pg.PlotWidget(), pg.PlotWidget(), pg.ImageItem(),
                         curves_plot=None)
        # Should not raise even with an empty point list
        r.render_curves([], y_param="Gm", x_param="Ua")


class TestHeatmapMixinExtraction(unittest.TestCase):
    """The heatmap rendering family lives in ``_HeatmapMixin``.

    Same regression strategy as the curves mixin — every method in the
    mixin must remain reachable on a ``PlotRenderer`` instance, and
    ``_HeatmapMixin`` must be in the MRO.
    """

    def test_heatmap_mixin_in_mro(self) -> None:
        from app.plotting._heatmap_mixin import _HeatmapMixin
        from app.plotting.renderer import PlotRenderer
        self.assertIn(_HeatmapMixin, PlotRenderer.__mro__)

    def test_heatmap_methods_inherited(self) -> None:
        """Every heatmap method is reachable on the renderer instance."""
        r = _make_renderer()
        for name in (
            "render_contour",
            "render_gm_rp",
            "render_pa_map",
            "set_right_heatmap_mode",
            "_grid_from_points",
        ):
            self.assertTrue(callable(getattr(r, name, None)),
                            f"PlotRenderer missing heatmap method {name!r}")

    def test_render_contour_empty_points_clears(self) -> None:
        """``render_contour`` with no points clears the contour image."""
        r = _make_renderer()
        # Pass empty points — should call clear() without raising
        r.render_contour([], select_ug2_slice=lambda pts: 0.0)

    def test_set_right_heatmap_mode_writes_attr(self) -> None:
        r = _make_renderer()
        r.set_right_heatmap_mode("mu")
        self.assertEqual(r._right_heatmap_mode, "mu")
        r.set_right_heatmap_mode("rp")
        self.assertEqual(r._right_heatmap_mode, "rp")

    def test_render_gm_rp_no_gm_image_returns(self) -> None:
        """``render_gm_rp`` with no gm_image attribute returns silently."""
        r = _make_renderer()
        # Default _make_renderer doesn't pass gm_image, so it's None
        self.assertIsNone(r.gm_image)
        # Should not raise
        r.render_gm_rp([])

    def test_render_pa_map_no_pa_image_returns(self) -> None:
        """``render_pa_map`` with no pa_map_image returns silently."""
        r = _make_renderer()
        self.assertIsNone(r.pa_map_image)
        r.render_pa_map([])


class TestPlot2DMixinExtraction(unittest.TestCase):
    """The ``render_plot_2d`` family lives in ``_Plot2DMixin``.

    Regression guards mirror the other mixin tests: the mixin must be
    in ``PlotRenderer.__mro__``, every method it owns must be
    reachable via ``r.<name>``, and the cross-mixin contract (Q-point
    family reads ``self._load_line_analysis`` set by ``render_plot_2d``)
    is exercised by the ``draw_qpoint_all`` smoke test.
    """

    def test_plot_2d_mixin_in_mro(self) -> None:
        from app.plotting._plot_2d_mixin import _Plot2DMixin
        from app.plotting.renderer import PlotRenderer
        self.assertIn(_Plot2DMixin, PlotRenderer.__mro__)

    def test_plot_2d_methods_inherited(self) -> None:
        """Every method on the 2D mixin is reachable on the renderer."""
        r = _make_renderer()
        for name in (
            "render_plot_2d",
            "_ensure_2d_cache",
            "_render_compare_overlay",
            "_build_curves_2d",
            "_render_current_scan",
            "_render_current_track_mode",
            "_render_current_color_mode",
            "_render_current_series_mode",
            "_render_zone_rect",
            "_render_pa_hyperbola",
            "_render_pg2_zone",
            "_render_ua_limit",
            "_render_ia_limit",
            "render_curve_incremental",
            "set_ug2_visibility",
            "_apply_ug2_marker_filter",
            "remove_series_items",
            "set_sid_visibility",
            "_curve_ug2_visible",
            "_clear_line_labels",
            "_add_line_label",
            "_make_line_label",
            "_show_ug2_colorbar",
            "_hide_ug2_colorbar",
            "draw_qpoint_all",
            "_clear_qpoint_items",
        ):
            self.assertTrue(callable(getattr(r, name, None)),
                            f"PlotRenderer missing 2D method {name!r}")

    def test_overlay_pen_styles_class_attr(self) -> None:
        """``_overlay_pen_styles`` lives on the mixin (4 entries)."""
        from app.plotting._plot_2d_mixin import _Plot2DMixin
        self.assertEqual(len(_Plot2DMixin._overlay_pen_styles), 4)

    def test_render_plot_2d_smoke_empty(self) -> None:
        """``render_plot_2d`` with no points completes without raising."""
        r = _make_renderer()
        r.render_plot_2d([], ug2_mode_series=False, series_labels={},
                         legend_hidden=True)
        # Cache populated; analysis cleared
        self.assertIsNotNone(r._2d_cache)
        self.assertIsNone(r._load_line_analysis)

    def test_draw_qpoint_all_no_analysis_clears(self) -> None:
        """``draw_qpoint_all`` with no analysis is a safe no-op."""
        r = _make_renderer()
        # No load_line_analysis set yet; must clear without raising
        r._load_line_analysis = None
        r.draw_qpoint_all()
        self.assertEqual(r._qpoint_items, {})


if __name__ == "__main__":
    unittest.main()
