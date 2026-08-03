# -*- coding: utf-8 -*-
"""Working-line bench on the 2D plot (perf gate).

Measures wall-clock on a real scan (offscreen Qt):
  1) full render_plot_2d WITH load-line (what every spin tick used to
     trigger via _rerender_2d) — 'full tick of the OLD path';
  2) isolated overlays.draw_load_line (line-layer compute+draw);
  3) lm19 'light' math of the live layer: find_intersections,
     5-point, Chebyshev, PP composite (5pt/cheb), model DFT.

Run (manually, wall-clock — NOT part of the suite):
    py tools/working_line_bench.py
Results are recorded in docs/WORKING_LINE_PLAN.md (measure-first).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPEAT_RENDER = 20
REPEAT_LAYER = 50
REPEAT_MATH = 100
DATASET = ROOT / "tests" / "spice_test_data" / "converted" / "pentode_6P1P_real.json"


def _t(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), max(ts)


def main() -> None:
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    points = data["points"]
    print(f"dataset: {DATASET.name}, points={len(points)}")

    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    import i18n_setup
    i18n_setup.setup("en")
    _app = QApplication.instance() or QApplication([])

    from app.plotting.renderer import PlotRenderer

    plot = pg.PlotWidget()
    line_labels: list = []
    renderer = PlotRenderer(plot, pg.PlotWidget(), pg.ImageItem())

    ll_dict = {"ub": 250.0, "ra": 5.0, "ug1_0": -7.0, "half_swing": 3.0,
               "ug2_filter": 250.0}

    # 1) full render (informational; the line was removed from the
    #    renderer — the 446 ms baseline with line is recorded in the plan)
    def full_render():
        renderer.render_plot_2d(points, ug2_mode_series=False,
                                series_labels={}, legend_hidden=True)
    med, mx = _t(full_render, REPEAT_RENDER)
    print(f"1) FULL render_plot_2d (no line now) median {med:7.1f} ms   max {mx:7.1f} ms")

    # 2) NEW path: full controller tick (lm19 recompute + incremental
    #    item update), cache busted by changing Ra every tick.
    from lm19.amp_engine import AmplifierEngine, AmpParams
    from app.working_line import WorkingLineController
    engine = AmplifierEngine()
    engine.set_data(points, series_labels={}, srk=None, is_triode=False)
    state = {"p": AmpParams(ub=250.0, ra=5.0, ug1_bias=-10.0,
                            half_swing=1.0, ug2_filter=250.0,
                            hd_method="5point")}
    ctrl = WorkingLineController(plot, engine, lambda: state["p"])
    ctrl.set_visible(True)

    def tick(circuit_params):
        state["p"] = circuit_params
        state["p"].ra = state["p"].ra + 0.001   # cache-bust
        ctrl._recompute()

    for label, mk in (
        ("SE 5-point", lambda: AmpParams(ub=250.0, ra=5.0, ug1_bias=-10.0,
                                         half_swing=1.0, ug2_filter=250.0,
                                         hd_method="5point")),
        ("PP Chebyshev", lambda: AmpParams(ub=250.0, circuit="pp",
                                           pp_raa=8.0, ug1_bias=-10.0,
                                           half_swing=1.5,
                                           ug2_filter=250.0,
                                           hd_method="chebyshev")),
    ):
        params = mk()
        med, mx = _t(lambda: tick(params), REPEAT_LAYER)
        print(f"2) CONTROLLER tick [{label:13s}]    median {med:7.1f} ms   max {mx:7.1f} ms")

    # 2b) heaviest live path: PP + UL 35% (model-Chebyshev + family)
    try:
        from lm19.tube_sim import quick_pentode
        model_ul, _ = quick_pentode("EL84")
        engine._series_models = {0: model_ul}
        p_ul = AmpParams(ub=300.0, circuit="pp", pp_raa=8.0,
                         ug1_bias=-7.0, half_swing=3.0, ug2_filter=250.0,
                         hd_method="chebyshev", ul_tap=0.35, series_id=0)
        med, mx = _t(lambda: tick(p_ul), REPEAT_LAYER)
        print(f"2) CONTROLLER tick [PP UL35 cheb ]    median {med:7.1f} ms   max {mx:7.1f} ms")
    except Exception as exc:  # bench-only
        print(f"2b) UL tick failed: {exc}")

    # 3) lm19 light math
    from lm19.amplifier import (
        PushPullLoadLine, ResistiveLoadLine,
        compute_distortion, compute_distortion_chebyshev,
        compute_distortion_dft, find_intersections,
        pp_distortion, compute_distortion_chebyshev_pp,
    )
    ll_se = ResistiveLoadLine(250.0, 5.0)
    isects = find_intersections(points, ll_se, ug2_filter=250.0)

    med, mx = _t(lambda: find_intersections(points, ll_se,
                                            ug2_filter=250.0), REPEAT_MATH)
    print(f"3a) find_intersections (SE)          median {med:7.1f} ms   max {mx:7.1f} ms")
    med, mx = _t(lambda: compute_distortion(isects, -7.0,
                                            half_swing=3.0), REPEAT_MATH)
    print(f"3b) 5-point                          median {med:7.2f} ms   max {mx:7.2f} ms")
    med, mx = _t(lambda: compute_distortion_chebyshev(
        isects, -7.0, half_swing=3.0), REPEAT_MATH)
    print(f"3c) Chebyshev (data)                 median {med:7.1f} ms   max {mx:7.1f} ms")

    ll_pp = PushPullLoadLine(250.0, ra_aa=8.0)
    med, mx = _t(lambda: pp_distortion(points, ll_pp, -7.0,
                                       half_swing=3.0,
                                       ug2_filter=250.0), REPEAT_MATH)
    print(f"3d) PP composite 5-point             median {med:7.1f} ms   max {mx:7.1f} ms")
    med, mx = _t(lambda: compute_distortion_chebyshev_pp(
        points, ll_pp, -7.0, half_swing=3.0,
        ug2_filter=250.0), REPEAT_MATH)
    print(f"3e) PP composite Chebyshev           median {med:7.1f} ms   max {mx:7.1f} ms")

    # DFT needs a model (fit on this data, once)
    try:
        from lm19.tube_sim import fit_koren
        t0 = time.perf_counter()
        fit = fit_koren(points, data.get("topology", "pentode"))
        print(f"   (koren fit for DFT: {(time.perf_counter()-t0):.1f} s)")
        model = fit.model
        med, mx = _t(lambda: compute_distortion_dft(
            model, ll_se, -7.0, half_swing=3.0, ug2=250.0,
            ub=250.0), max(REPEAT_MATH // 5, 10))
        print(f"3f) DFT SE (model)                   median {med:7.1f} ms   max {mx:7.1f} ms")
    except Exception as exc:  # bench-only: report and move on
        print(f"3f) DFT: fit failed ({exc}) — skipped")


if __name__ == "__main__":
    main()
