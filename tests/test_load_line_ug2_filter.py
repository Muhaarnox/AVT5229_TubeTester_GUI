"""Working-line Ug2 filter in pentode mode.

The old tool (overlays.draw_load_line + plot-side spins) is removed —
the Ug2 slice is now guaranteed by the single engine path: both full
Analyze and the live layer (compute_working_line) filter
intersections by params.ug2_filter. The pins below discriminate
screen-level merging.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.amp_engine import AmplifierEngine, AmpParams


def _pt(ua, ug1, ia, ug2, series_id=0):
    return {"ua": ua, "ug1": ug1, "ia": ia, "ug2": ug2,
            "ig2": 0.1, "uh": 6.3, "ih": 0.3, "series_id": series_id}


def _multi_ug2_points():
    """Two screen levels with DIFFERENT currents — merging the levels
    changes the intersections (non-symmetric data)."""
    pts = []
    for ua in range(0, 301, 10):
        for ug1 in (-1.0, -2.0, -3.0):
            base = 30.0 + 8.0 * (ug1 + 2.0)
            pts.append(_pt(float(ua), ug1, base + ua * 0.01, 200.0))
            pts.append(_pt(float(ua), ug1, base * 0.5 + ua * 0.01, 250.0))
    return pts


def _engine():
    eng = AmplifierEngine()
    eng.set_data(_multi_ug2_points(), series_labels={}, srk=None,
                 is_triode=False)
    return eng


class TestWorkingLineUg2Filter:

    def _params(self, ug2_filter):
        return AmpParams(ub=300.0, ra=5.0, ug1_bias=-2.0,
                         half_swing=0.8, ug2_filter=ug2_filter)

    def test_filter_selects_slice(self):
        """Intersections under filter 250 differ from 200 (currents x2)
        — the filter really slices, not ignored."""
        eng = _engine()
        v250 = eng.compute_working_line(self._params(250.0))
        v200 = eng.compute_working_line(self._params(200.0))
        assert v250.intersections and v200.intersections
        ua250 = {round(p["ug1"], 1): p["ua"] for p in v250.intersections}
        ua200 = {round(p["ug1"], 1): p["ua"] for p in v200.intersections}
        # Same bias curves, different currents -> different crossings.
        assert ua250[-2.0] != ua200[-2.0]

    def test_analyze_and_live_agree_on_filtered_slice(self):
        """Single path: full Analyze and the live layer give the same
        intersections under the same filter (routing cannot diverge)."""
        eng = _engine()
        p = self._params(250.0)
        live = eng.compute_working_line(p)
        full = eng.analyze(p).working_line
        assert [i["ua"] for i in live.intersections] ==             [i["ua"] for i in full.intersections]
