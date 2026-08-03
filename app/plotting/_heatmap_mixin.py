"""Heatmap rendering mixin for ``PlotRenderer``.

Contains the contour / gm_rp / pa_map heatmap rendering family
(5 methods).

The mixin assumes its host class provides (set by ``PlotRenderer.__init__``):
  - ``self.contour_plot`` / ``self.contour_image`` / ``self._marker_contour``
  - ``self.gm_plot`` / ``self.gm_image`` / ``self._marker_gm``
  - ``self.rp_plot`` / ``self.rp_image`` / ``self._marker_rp``
  - ``self.mu_plot`` / ``self.mu_image`` / ``self._marker_mu``
  - ``self.pa_map_plot`` / ``self.pa_map_image`` / ``self._marker_pa``
  - ``self._gm_overlay_items`` / ``self._pa_overlay_items``
  - ``self._is_triode_eff()`` method
  - ``self.ua_cluster_thr`` / ``self.ug1_cluster_thr`` / ``self.ug2_cluster_thr``
    via ``RendererContext`` properties
  - ``self._right_heatmap_mode`` (writable str attr)

Does not define ``__init__`` — relies on the host class for setup.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from i18n_setup import t
from lm19.label_formats import format_label
from lm19.plot_style import COLOR_LIMIT as _COLOR_LIMIT
from lm19.plotting.grids import (
    build_ia_grid,
    build_ia_grid_averaged,
    build_pa_grid,
    compute_gm_rp_grids,
    compute_mu_grid,
    fill_nan_nearest,
    filter_ug2_slice,
    suppress_sparse,
)


# ── module local constants ──
# Iso-lines sit above the heatmap image but below Q-point markers
# (overlays.py uses 899-902 for the Q family).
_ISO_CURVE_Z = 850


class _HeatmapMixin:
    """Render Ia/Gm/Rp/µ/Pa as heatmaps."""

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_overlay_items(plot, image, items: list) -> None:
        """Detach heatmap overlay items and empty *items* in place.

        Iso-curves are CHILDREN of the ImageItem (they need its
        index→axes transform), so ``plot.removeItem`` alone would
        silently no-op on them and stale lines would pile up under the
        image across re-renders.
        """
        for item in items:
            if image is not None and item.parentItem() is image:
                item.setParentItem(None)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            else:
                plot.removeItem(item)
        items.clear()

    def _heatmap_images(self) -> Dict[str, Optional[pg.ImageItem]]:
        """key → ImageItem map for lock-scale and colorbar plumbing."""
        return {
            "contour": self.contour_image,
            "gm": self.gm_image,
            "rp": self.rp_image,
            "mu": self.mu_image,
            "pa_map": self.pa_map_image,
        }

    def _apply_heatmap_levels(self, key: str,
                              image: Optional[pg.ImageItem]) -> None:
        """Re-apply captured levels after an autoLevels render (lock)."""
        if not self._heatmap_scale_locked or image is None:
            return
        levels = self._heatmap_locked_levels.get(key)
        if levels is None:
            # Map had no data when the lock was enabled — capture now.
            current = image.getLevels()
            if current is not None:
                self._heatmap_locked_levels[key] = tuple(current)
        else:
            image.setLevels(levels)

    def _sync_heatmap_bar(self, key: str, plot: Optional[pg.PlotWidget],
                          image: Optional[pg.ImageItem],
                          label: str) -> None:
        """Create the map's ColorBarItem lazily and sync it to the
        image's current levels (display-only: interactive=False — the
        levels are owned by autoLevels / the lock, not by dragging)."""
        if plot is None or image is None:
            return
        bar = self._heatmap_bars.get(key)
        if bar is None:
            bar = pg.ColorBarItem(colorMap=self.cmap, label=label,
                                  interactive=False)
            bar.setImageItem(image, insert_in=plot.getPlotItem())
            self._heatmap_bars[key] = bar
        levels = image.getLevels()
        if levels is not None:
            bar.setLevels(tuple(levels))
        bar.show()          # may have been hidden by a blank render

    def _hide_heatmap_bars(self, *keys: str) -> None:
        """Hide bars whose maps just went blank — a value scale next to
        an empty image would keep claiming the previous scan's range."""
        for key in keys:
            bar = self._heatmap_bars.get(key)
            if bar is not None:
                bar.hide()

    def set_heatmap_scale_locked(self, locked: bool) -> None:
        """Freeze/unfreeze the color scale of every heatmap.

        Locking captures each map's CURRENT levels; subsequent renders
        keep them (two scans become color-comparable). Unlocking clears
        the capture — autoLevels resumes on the next render.
        """
        self._heatmap_scale_locked = bool(locked)
        self._heatmap_locked_levels.clear()
        if locked:
            for key, image in self._heatmap_images().items():
                if image is not None and image.image is not None:
                    levels = image.getLevels()
                    if levels is not None:
                        self._heatmap_locked_levels[key] = tuple(levels)

    def update_heatmap_colorbars_cmap(self, cmap: pg.ColorMap) -> None:
        """Follow the colormap combo: bars re-lut their linked images."""
        self.cmap = cmap
        for bar in self._heatmap_bars.values():
            bar.setColorMap(cmap)
        # The 2D Ug2 colorbar is created once with the then-current
        # cmap; without this update its gradient would keep the old
        # scheme while the curves are recolored by the new one.
        bar_2d = getattr(self, "ug2_colorbar", None)
        if bar_2d is not None:
            bar_2d.setColorMap(cmap)

    def _add_iso_line(self, image, img_data, level: float,
                      overlay_items: list) -> None:
        """Draw an iso-line of *img_data* (the SAME array the image
        displays, already transposed) at *level*, parented to *image*."""
        iso = pg.IsocurveItem(
            data=img_data, level=level,
            pen=pg.mkPen(_COLOR_LIMIT, width=2,
                         style=pg.QtCore.Qt.PenStyle.DashLine))
        iso.setParentItem(image)
        iso.setZValue(_ISO_CURVE_Z)
        overlay_items.append(iso)

    def render_contour(self, points, select_ug2_slice) -> None:
        grid = self._grid_from_points(points, select_ug2_slice)
        if not grid:
            self.contour_image.clear()
            self._marker_contour.clear()
            self._hide_heatmap_bars("contour")
            return
        z = grid["z"]
        self.contour_image.setImage(suppress_sparse(z).T, autoLevels=True)
        ua_vals = grid["ua"]
        ug1_vals = grid["ug1"]
        rect = pg.QtCore.QRectF(
            min(ua_vals),
            min(ug1_vals),
            max(ua_vals) - min(ua_vals),
            max(ug1_vals) - min(ug1_vals),
        )
        self.contour_image.setRect(rect)
        self._apply_heatmap_levels("contour", self.contour_image)
        self._sync_heatmap_bar("contour", self.contour_plot,
                               self.contour_image, t('plot.Ia_mA'))
        if self._is_triode_eff(points):
            self.contour_plot.setLabel("left", t('plot.Ug1_V'))
        else:
            self.contour_plot.setLabel("left",
                f"Ug1 (V)  {format_label('ug2', grid['ug2'])}")
        self.contour_plot.setLabel("bottom", t('plot.Ua_V'))
        self._marker_contour.set_grid(z, ua_vals, ug1_vals)

    def _grid_from_points(self, points, select_ug2_slice):
        """Build Ia grid for contour: Ug2 filter → averaged grid → dict."""
        if not points:
            return None
        triode_eff = self._is_triode_eff(points)
        subset = filter_ug2_slice(
            points, triode_eff, select_ug2_slice, self.ug2_cluster_thr)
        if not subset:
            return None
        g = build_ia_grid_averaged(subset, self.ug1_cluster_thr, self.ua_cluster_thr)
        if g is None:
            return None
        ug2_target = 0.0 if triode_eff else select_ug2_slice(points)
        return {"ua": g["ua_vals"], "ug1": g["ug1_vals"],
                "z": g["ia_grid"], "ug2": ug2_target}

    def render_gm_rp(self, points: List[Dict], select_ug2_slice=None,
                     nominal_s: float = None) -> None:
        """Render Gm, Rp, and µ heatmaps.

        ``nominal_s`` (lamp-card S, mA/V) draws an iso-line on the Gm
        map — the locus of operating points where the tube meets its
        nominal transconductance.
        """
        if self.gm_image is None or self.rp_image is None:
            return
        self.gm_image.clear()
        self.rp_image.clear()
        if self.mu_image is not None:
            self.mu_image.clear()
        self._remove_overlay_items(self.gm_plot, self.gm_image,
                                   self._gm_overlay_items)
        if self._marker_gm:
            self._marker_gm.clear()
        if self._marker_rp:
            self._marker_rp.clear()
        if self._marker_mu:
            self._marker_mu.clear()
        # Images are blank from here; each map's bar re-shows in its
        # own _sync_heatmap_bar when (and only when) it renders data —
        # this also covers the per-map all-NaN skips below.
        self._hide_heatmap_bars("gm", "rp", "mu")

        if not points:
            return

        triode_eff = self._is_triode_eff(points)
        subset = filter_ug2_slice(
            points, triode_eff, select_ug2_slice, self.ug2_cluster_thr)
        g = build_ia_grid(subset, self.ug1_cluster_thr, self.ua_cluster_thr)
        if g is None:
            return
        ua_vals, ug1_vals, ia_grid = g["ua_vals"], g["ug1_vals"], g["ia_grid"]
        gm_grid, rp_grid = compute_gm_rp_grids(ua_vals, ug1_vals, ia_grid)

        # Pre-compute midpoints shared by Gm, Rp, µ
        ug1_mid = [(ug1_vals[i] + ug1_vals[i + 1]) / 2.0
                   for i in range(len(ug1_vals) - 1)]
        ua_mid = [(ua_vals[j] + ua_vals[j + 1]) / 2.0
                  for j in range(len(ua_vals) - 1)]

        # ── Gm heatmap ──
        if not np.all(np.isnan(gm_grid)):
            gm_display = np.abs(gm_grid)
            gm_img_data = suppress_sparse(gm_display).T
            self.gm_image.setImage(gm_img_data, autoLevels=True)
            rect = pg.QtCore.QRectF(
                min(ua_vals), min(ug1_mid),
                max(ua_vals) - min(ua_vals),
                max(ug1_mid) - min(ug1_mid),
            )
            self.gm_image.setRect(rect)
            self._apply_heatmap_levels("gm", self.gm_image)
            self._sync_heatmap_bar("gm", self.gm_plot, self.gm_image,
                                   t('plot.Gm_mA_V'))
            if self.gm_plot is not None:
                self.gm_plot.autoRange()
            if self._marker_gm:
                self._marker_gm.set_grid(
                    fill_nan_nearest(gm_display), ua_vals, ug1_mid)
                self._marker_gm.set_aux_grids([
                    ("Ia", "mA", ".2f",
                     fill_nan_nearest(ia_grid), ua_vals, ug1_vals),
                ])
            # S-nominal iso-line (the parameter used to arrive here and
            # get dropped — the Curves tab drew the reference, the map
            # silently didn't).
            if nominal_s is not None and nominal_s > 0:
                self._add_iso_line(self.gm_image, gm_img_data, nominal_s,
                                   self._gm_overlay_items)
                s_label = pg.TextItem(f"S nom={nominal_s:.1f}",
                                      color=_COLOR_LIMIT, anchor=(1, 0))
                s_label.setPos(max(ua_vals), min(ug1_mid))
                self.gm_plot.addItem(s_label)
                self._gm_overlay_items.append(s_label)

        if self.gm_plot is not None:
            self.gm_plot.setLabel("left", t('plot.Ug1_V'))
            self.gm_plot.setLabel("bottom", t('plot.Ua_V'))

        # ── Rp heatmap ──
        if not np.all(np.isnan(rp_grid)):
            rp_abs_raw = np.abs(rp_grid)
            p95 = np.nanpercentile(rp_abs_raw, 95)
            rp_display = np.clip(rp_abs_raw, 0,
                                 p95 * 1.5 if p95 > 0 else 100)
            self.rp_image.setImage(suppress_sparse(rp_display).T, autoLevels=True)
            rect = pg.QtCore.QRectF(
                min(ua_mid), min(ug1_vals),
                max(ua_mid) - min(ua_mid),
                max(ug1_vals) - min(ug1_vals),
            )
            self.rp_image.setRect(rect)
            self._apply_heatmap_levels("rp", self.rp_image)
            self._sync_heatmap_bar("rp", self.rp_plot, self.rp_image,
                                   t('plot.Rp_kOhm'))
            if self._marker_rp:
                # The clip is display contrast only — the hover tooltip
                # must report the COMPUTED value, not the clipped one
                # (the µ marker's aux Rp grid was already unclipped).
                self._marker_rp.set_grid(
                    fill_nan_nearest(rp_abs_raw), ua_mid, ug1_vals)
                self._marker_rp.set_aux_grids([
                    ("Ia", "mA", ".2f",
                     fill_nan_nearest(ia_grid), ua_vals, ug1_vals),
                ])

        if self.rp_plot is not None:
            self.rp_plot.setLabel("left", t('plot.Ug1_V'))
            self.rp_plot.setLabel("bottom", t('plot.Ua_V'))

        # ── µ heatmap ──
        gm_abs = fill_nan_nearest(np.abs(gm_grid))
        rp_abs = fill_nan_nearest(np.abs(rp_grid))
        ia_filled = fill_nan_nearest(ia_grid)
        if self.mu_image is not None:
            mu_grid = compute_mu_grid(gm_grid, rp_grid)
            if not np.all(np.isnan(mu_grid)):
                mu_display = np.abs(mu_grid)
                p97 = np.nanpercentile(mu_display[~np.isnan(mu_display)], 97) if not np.all(np.isnan(mu_display)) else 200
                mu_clip_limit = p97 * 1.3 if p97 > 0 else 200
                mu_display_vis = np.clip(mu_display, 0, mu_clip_limit)
                self.mu_image.setImage(suppress_sparse(mu_display_vis).T, autoLevels=True)
                rect = pg.QtCore.QRectF(
                    min(ua_mid), min(ug1_mid),
                    max(ua_mid) - min(ua_mid),
                    max(ug1_mid) - min(ug1_mid),
                )
                self.mu_image.setRect(rect)
                self._apply_heatmap_levels("mu", self.mu_image)
                self._sync_heatmap_bar("mu", self.mu_plot, self.mu_image,
                                       t('plot.mu'))
                if self._marker_mu:
                    # Unclipped: the clip is display contrast only (see
                    # the Rp marker above).
                    self._marker_mu.set_grid(
                        fill_nan_nearest(mu_display), ua_mid, ug1_mid)
                    self._marker_mu.set_aux_grids([
                        ("Gm", "mA/V", ".2f", gm_abs, ua_vals, ug1_mid),
                        ("Rp", "kΩ", ".2f", rp_abs, ua_mid, ug1_vals),
                        ("Ia", "mA", ".2f", ia_filled, ua_vals, ug1_vals),
                    ])

            if self.mu_plot is not None:
                self.mu_plot.setLabel("left", t('plot.Ug1_V'))
                self.mu_plot.setLabel("bottom", t('plot.Ua_V'))

    def set_right_heatmap_mode(self, mode: str) -> None:
        """Store current right heatmap mode ('rp' or 'mu').

        The actual page switching is done by QStackedWidget in main_window.
        This is stored for any renderer logic that might need it.
        """
        self._right_heatmap_mode = mode

    def render_pa_map(self, points: List[Dict], select_ug2_slice=None,
                      pa_max: float = None) -> None:
        """Render Pa dissipation heatmap: color = Ua * Ia.

        ``pa_max`` (W) draws the dissipation-limit iso-line — everything
        on the hotter side of it exceeds the tube's rating.
        """
        if self.pa_map_image is None:
            return
        self.pa_map_image.clear()
        self._remove_overlay_items(self.pa_map_plot, self.pa_map_image,
                                   self._pa_overlay_items)
        if self._marker_pa:
            self._marker_pa.clear()
        self._hide_heatmap_bars("pa_map")   # re-shown on data render

        if not points:
            return

        triode_eff = self._is_triode_eff(points)
        subset = filter_ug2_slice(
            points, triode_eff, select_ug2_slice, self.ug2_cluster_thr)
        g = build_pa_grid(subset, self.ug1_cluster_thr, self.ua_cluster_thr)
        if g is None:
            return
        ua_vals, ug1_vals = g["ua_vals"], g["ug1_vals"]
        pa_grid = g["pa_grid"]

        pa_img_data = suppress_sparse(pa_grid).T
        self.pa_map_image.setImage(pa_img_data, autoLevels=True)
        rect = pg.QtCore.QRectF(
            min(ua_vals), min(ug1_vals),
            max(ua_vals) - min(ua_vals),
            max(ug1_vals) - min(ug1_vals),
        )
        self.pa_map_image.setRect(rect)
        self._apply_heatmap_levels("pa_map", self.pa_map_image)
        self._sync_heatmap_bar("pa_map", self.pa_map_plot,
                               self.pa_map_image, t('plot.Pa_W'))
        if self._marker_pa:
            self._marker_pa.set_grid(
                fill_nan_nearest(pa_grid), ua_vals, ug1_vals)
            # Build Ia grid for tooltip (same axes as Pa)
            ia_g = build_ia_grid(subset, self.ug1_cluster_thr, self.ua_cluster_thr)
            if ia_g is not None:
                self._marker_pa.set_aux_grids([
                    ("Ia", "mA", ".2f",
                     ia_g["ia_grid"], ia_g["ua_vals"], ia_g["ug1_vals"]),
                ])

        if self.pa_map_plot is not None:
            self.pa_map_plot.setLabel("left", t('plot.Ug1_V'))
            self.pa_map_plot.setLabel("bottom", t('plot.Ua_V'))

            # Pa_max iso-line (pa_grid is in watts, level compares 1:1).
            if pa_max is not None and pa_max > 0:
                self._add_iso_line(self.pa_map_image, pa_img_data, pa_max,
                                   self._pa_overlay_items)
                label = pg.TextItem(
                    f"Pa_max={pa_max:.1f}W", color=_COLOR_LIMIT, anchor=(1, 0),
                )
                label.setPos(max(ua_vals), min(ug1_vals))
                self.pa_map_plot.addItem(label)
                self._pa_overlay_items.append(label)
