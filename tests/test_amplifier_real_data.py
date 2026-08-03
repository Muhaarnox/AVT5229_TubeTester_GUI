"""Real-data sanity checks for amplifier analysis.

Real measurement fixtures live in `tests/spice_test_data/converted/`.
The original sources in `lm19_app/measurements/` are user data and must
not be referenced from tests.
"""

import json

import pytest

from lm19.amplifier import (
    ResistiveLoadLine,
    PushPullLoadLine,
    UltralinearModelWrapper,
    compute_distortion,
    compute_distortion_chebyshev,
    compute_distortion_dft,
    compute_imd,
    compute_pa_avg,
    compute_headroom,
    compute_stage_params,
    estimate_ig2_at_q,
    compute_pg2,
    find_intersections,
    find_intersections_model,
    model_gm_ra,
    _numerical_gm_ra,
    interp_intersection,
    sweep_amplitude,
)
from lm19.tube_sim import fit_koren, quick_triode, quick_pentode

from tests._real_data import (
    EL84_ER_L1_PENT,
    EL84_PENTODE_FILES,
    converted_path,
    load_converted,
)


# ── module local constants ──
# Physical sanity bounds: THD above this means the analyzer collapsed
# numerically; HD above this is unphysical for any real tube.
_THD_MAX_PCT = 70.0
_HD_MAX_PCT = 100.0


def _load_real_files():
    """Existing pentode EL84 fixtures, in deterministic order."""
    return [converted_path(n) for n in EL84_PENTODE_FILES if converted_path(n).exists()]


def _load_spice_file(name: str):
    return load_converted(name)


# Pa_avg / gm / Ig2 thresholds in this module are calibrated to one
# specific specimen. Pin it explicitly so reordering EL84_PENTODE_FILES
# never shifts the numerical baseline.
_FIT_FIXTURE = EL84_ER_L1_PENT


# Points within this half-width of a slice's nominal Ug2 belong to the
# slice (the fixture levels are 25 V apart, settle noise is < 1 V).
_UG2_SLICE_HALF_WIDTH_V = 5.0


def _top_ug2_slices(points, top_n=2):
    """Most-populated DISTINCT Ug2 levels of a scan.

    Candidates closer than two slice half-widths to an already selected
    level are the same physical level (settle noise splits one level
    into several rounded values, e.g. 174.0/174.3/174.7/175.0) — they
    would yield overlapping point sets, not a second slice.
    """
    vals = sorted({round(p.get("ug2", 0.0), 1) for p in points})
    counts = []
    for v in vals:
        n = sum(1 for p in points if abs(p.get("ug2", 0.0) - v) <= 0.6)
        counts.append((n, v))
    picked = []
    for _, v in sorted(counts, reverse=True):
        if all(abs(v - w) >= 2 * _UG2_SLICE_HALF_WIDTH_V for w in picked):
            picked.append(v)
        if len(picked) == top_n:
            break
    return picked


def test_auto_swing_real_data_stays_physical():
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    checked = 0
    for path in _load_real_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        points = data.get("points", [])
        if len(points) < 50:
            continue
        for ug2 in _top_ug2_slices(points, top_n=2):
            ps = [p for p in points if abs(p.get("ug2", 0.0) - ug2) <= _UG2_SLICE_HALF_WIDTH_V]
            isects = find_intersections(ps, ll)
            if len(isects) < 3:
                continue
            dist = compute_distortion(isects, ug1_bias=-7.3, half_swing=None)
            if dist is None:
                continue
            checked += 1
            assert dist["ia_0"] >= -1e-9
            assert dist["i_min"] >= -1e-9
            assert dist["i_max"] >= -1e-9
            assert dist["ua_0"] <= ll.ub + 1e-6
            assert 0 <= dist["thd"] < _THD_MAX_PCT
            assert 0 <= dist["hd2"] < _HD_MAX_PCT
            assert 0 <= dist["hd3"] < _HD_MAX_PCT
    # Every fixture x every slice computes today; fewer means a fixture
    # went missing or a reject path silently swallowed one.
    assert checked == 2 * len(EL84_PENTODE_FILES)


def test_manual_swing_real_data_has_no_nonphysical_points():
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    checked = 0
    nonphysical = 0
    for path in _load_real_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        points = data.get("points", [])
        if len(points) < 50:
            continue
        for ug2 in _top_ug2_slices(points, top_n=2):
            ps = [p for p in points if abs(p.get("ug2", 0.0) - ug2) <= _UG2_SLICE_HALF_WIDTH_V]
            isects = find_intersections(ps, ll)
            if len(isects) < 3:
                continue
            dist = compute_distortion(isects, ug1_bias=-7.3, half_swing=3.0)
            if dist is None:
                continue
            checked += 1
            # Bias -7.3 +/- 3.0 V lies fully inside the measured Ug1
            # range of these fixtures — a clamped manual swing here
            # means the range shrank or interpolation regressed.
            assert not dist.get("manual_swing_clamped"), (
                f"unexpected swing clamp in {path.name} @ ug2={ug2}"
            )
            if (
                dist["ia_0"] < -1e-9
                or dist["i_min"] < -1e-9
                or dist["i_max"] < -1e-9
                or dist["ua_0"] > ll.ub + 1e-6
            ):
                nonphysical += 1
            assert 0 <= dist["thd"] < _THD_MAX_PCT
            assert 0 <= dist["hd2"] < _HD_MAX_PCT
            assert 0 <= dist["hd3"] < _HD_MAX_PCT
    assert checked == 2 * len(EL84_PENTODE_FILES)
    assert nonphysical == 0


def test_stage_gain_with_ug2_slice_has_no_outlier_spikes():
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    checked = 0
    for path in _load_real_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        points = data.get("points", [])
        if len(points) < 50:
            continue
        for ug2 in _top_ug2_slices(points, top_n=2):
            ps = [p for p in points if abs(p.get("ug2", 0.0) - ug2) <= _UG2_SLICE_HALF_WIDTH_V]
            isects = find_intersections(ps, ll)
            if len(isects) < 3:
                continue
            stage = compute_stage_params(isects, ll, ug1_bias=-7.3, srk=None, points=ps)
            if stage is None:
                continue
            checked += 1
            # Anti-regression threshold for EL84 real data: no 100+ gain
            # spikes; a zero/negative gain is equally a regression (at
            # Ug2=99 V near cutoff the honest gain is ~0.6, still > 0).
            assert 0 < stage["gain"] < 100.0
            assert 0 < stage["zout"] < 20.0
    assert checked == 2 * len(EL84_PENTODE_FILES)


def test_non_el84_spice_smoke_physicality():
    ll = ResistiveLoadLine(ub=250.0, ra=10.0)
    datasets = [
        "triode_ecc82_datasheet.json",
        "triode_300B_curvetracedata.json",
    ]
    checked = 0
    for name in datasets:
        data = _load_spice_file(name)
        points = data.get("points", [])
        isects = find_intersections(points, ll)
        if len(isects) < 3:
            continue
        dist = compute_distortion(isects, ug1_bias=None, half_swing=None)
        if dist is None:
            continue
        checked += 1
        assert dist["ia_0"] >= -1e-9
        assert dist["i_min"] >= -1e-9
        assert dist["i_max"] >= -1e-9
        assert dist["ua_0"] <= ll.ub + 1e-6
        assert 0 <= dist["thd"] < _THD_MAX_PCT
        assert 0 <= dist["hd2"] < _HD_MAX_PCT
        assert 0 <= dist["hd3"] < _HD_MAX_PCT
    # Both datasets compute today; a lower count means one of them was
    # silently swallowed by a reject path.
    assert checked == len(datasets)


# ---------------------------------------------------------------
# Pa_avg on real EL84 data via Koren fit
# ---------------------------------------------------------------

def _fit_el84_pentode():
    """Load the calibration EL84 pentode fixture and fit Koren model."""
    path = converted_path(_FIT_FIXTURE)
    if not path.exists():
        return None, []
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = data.get("points", [])
    if len(pts) < 50:
        return None, pts
    # No try/except: a fit failure on the shipped fixture is a
    # regression and must surface with its real traceback.
    result = fit_koren(pts, "pentode")
    return result.model, pts


def test_pa_avg_real_el84_energy_conservation():
    """Pa_avg from real EL84 data: Pa + P_load = Pdc (energy conservation)."""
    model, pts = _fit_el84_pentode()
    assert model is not None, "EL84 fit must succeed on shipped real data"
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
    ug2 = max(ug2_vals) if ug2_vals else 200.0

    r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=ug2, ub=250.0)
    assert r is not None
    assert r["pa_avg_mw"] > 0
    assert "pdc_avg_mw" in r
    p_load = r["pdc_avg_mw"] - r["pa_avg_mw"]
    assert p_load > 0, "Load power must be positive"


def test_pa_avg_real_el84_decreases_with_swing():
    """Pa_avg decreases with larger swing (more power to load)."""
    model, pts = _fit_el84_pentode()
    assert model is not None, "model unexpectedly None on real data"
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
    ug2 = max(ug2_vals) if ug2_vals else 200.0

    pa_values = []
    for swing in [1.0, 3.0, 5.0]:
        r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=swing, ug2=ug2, ub=250.0)
        if r:
            pa_values.append(r["pa_avg_mw"])

    assert len(pa_values) >= 2
    # Pa_avg should decrease as more signal goes to load
    for i in range(1, len(pa_values)):
        assert pa_values[i] < pa_values[i - 1], (
            f"Pa_avg should decrease: swing step {i}, {pa_values[i]:.0f} >= {pa_values[i-1]:.0f}"
        )


def test_pa_avg_real_el84_under_pa_max():
    """Pa_avg should be well under Pa_max=12W for EL84."""
    model, pts = _fit_el84_pentode()
    assert model is not None, "model unexpectedly None on real data"
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
    ug2 = max(ug2_vals) if ug2_vals else 200.0

    r = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=ug2, ub=250.0)
    assert r is not None
    assert r["pa_avg_mw"] < 12000, f"Pa_avg={r['pa_avg_mw']:.0f}mW exceeds Pa_max=12W"


def test_headroom_real_el84_no_crash():
    """Headroom on real EL84 data: no crash, reasonable values."""
    model, pts = _fit_el84_pentode()
    assert model is not None, "model unexpectedly None on real data"
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
    ug2 = max(ug2_vals) if ug2_vals else 200.0

    ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
    isects = find_intersections_model(model, ll, ug1_vals, ug2=ug2)
    hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=12.0, load_line=ll)
    assert hr is not None
    assert hr["max_swing"] > 0
    assert hr["max_swing"] < 20.0


# ---------------------------------------------------------------
# Cross-validation: distortion methods on real data
# ---------------------------------------------------------------

def _real_el84_context():
    """Load real EL84 pentode: model, points, intersections, ug2."""
    model, pts = _fit_el84_pentode()
    if model is None:
        return None
    ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
    ug2 = max(ug2_vals) if ug2_vals else 200.0
    filtered = [p for p in pts if abs(p.get("ug2", 0.0) - ug2) < 10]
    ll = ResistiveLoadLine(ub=250.0, ra=5.0)
    isects = find_intersections(filtered, ll)
    ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
    isects_model = find_intersections_model(model, ll, ug1_vals, ug2=ug2)
    return {
        "model": model, "pts": pts, "filtered": filtered,
        "ll": ll, "isects": isects, "isects_model": isects_model,
        "ug2": ug2,
    }


def test_chebyshev_real_el84_physicality():
    """Chebyshev distortion on real EL84: THD > 0, reasonable range."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    # Bias -7.0 sits at the EDGE of the measured ug1 range
    # (-11.1..-7.0) where the data path honestly rejects; use the
    # mid-range bias -9.0 so the computation actually runs.
    dc = compute_distortion_chebyshev(ctx["isects"], ug1_bias=-9.0, half_swing=2.0)
    assert dc is not None, "dc unexpectedly None on real data"
    assert dc["thd"] > 0
    assert dc["thd"] < 50.0  # reasonable for EL84 pentode
    assert dc["pout_mw"] > 0


def test_dft_real_el84_physicality():
    """DFT distortion on real EL84 model: THD > 0, reasonable range."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    dd = compute_distortion_dft(
        ctx["model"], ctx["ll"],
        ug1_bias=-7.0, half_swing=3.0, ug2=ctx["ug2"], ub=250.0,
    )
    assert dd is not None, "dd unexpectedly None on real data"
    assert dd["thd"] > 0
    assert dd["thd"] < 50.0
    assert dd["pout_mw"] > 0
    assert "pa_avg_mw" in dd
    assert dd["pa_avg_mw"] > 0


def test_5point_vs_chebyshev_real_el84():
    """5-point and Chebyshev THD should be in same range on real data."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    # -9.0 = middle of the measured ug1 range (the -7.0 edge yields None)
    d5 = compute_distortion(ctx["isects"], ug1_bias=-9.0, half_swing=2.0)
    dc = compute_distortion_chebyshev(ctx["isects"], ug1_bias=-9.0, half_swing=2.0)
    assert d5 is not None and dc is not None, "both methods must compute"
    # Both should give non-zero THD
    assert d5["thd"] > 0 and dc["thd"] > 0
    # Within 50% of each other (real data has more noise than synthetic)
    assert d5["thd"] == pytest.approx(dc["thd"], rel=0.5), (
        f"5pt THD={d5['thd']:.2f}% vs Cheby THD={dc['thd']:.2f}%"
    )


def test_5point_vs_dft_real_el84():
    """5-point and DFT THD should be in same range on real data."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    d5 = compute_distortion(ctx["isects"], ug1_bias=-9.0, half_swing=2.0)
    dd = compute_distortion_dft(
        ctx["model"], ctx["ll"],
        ug1_bias=-9.0, half_swing=2.0, ug2=ctx["ug2"],
    )
    assert d5 is not None and dd is not None, "both methods must compute"
    assert d5["thd"] > 0 and dd["thd"] > 0
    # Within factor 2 (model vs measurement interpolation differ)
    ratio = max(d5["thd"], dd["thd"]) / max(min(d5["thd"], dd["thd"]), 0.01)
    assert ratio < 3.0, f"5pt THD={d5['thd']:.2f}% vs DFT THD={dd['thd']:.2f}%"


# ---------------------------------------------------------------
# model_gm_ra vs numerical on real data
# ---------------------------------------------------------------

def test_model_gm_ra_vs_numerical_real_el84():
    """Model gm vs numerical gm on real EL84 pentode data."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    q = interp_intersection(ctx["isects_model"], -7.0)
    assert q is not None, "q unexpectedly None on real data"
    r_model = model_gm_ra(ctx["model"], ua_q=q["ua"], ug1_q=-7.0, ug2=ctx["ug2"])
    r_num = _numerical_gm_ra(ctx["filtered"], ctx["isects"], ug1_bias=-7.0)
    assert r_model is not None and r_num is not None, "gm/ra must compute"
    # Model and numerical should agree within 30%
    assert r_model["gm"] == pytest.approx(r_num["gm"], rel=0.3), (
        f"model gm={r_model['gm']:.2f} vs numerical gm={r_num['gm']:.2f}"
    )


# ---------------------------------------------------------------
# Sweep amplitude on real data
# ---------------------------------------------------------------

def test_sweep_amplitude_real_el84_monotonic_pout():
    """Amplitude sweep on real EL84: Pout should increase with swing."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    sw = sweep_amplitude(
        ctx["filtered"], ctx["ll"],
        ug1_bias=-9.0, ug2_filter=ctx["ug2"], steps=15,
    )
    assert len(sw) >= 3, "amplitude sweep must yield >= 3 rows"
    pouts = [s["pout_mw"] for s in sw if s.get("pout_mw", 0) > 0]
    assert len(pouts) >= 3
    # Pout generally increases with swing
    assert pouts[-1] > pouts[0], "Pout should increase with swing"


# ---------------------------------------------------------------
# IMD on real data
# ---------------------------------------------------------------

def test_imd_real_el84_reasonable():
    """IMD on real EL84 data: should produce non-zero, bounded values."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    # -9.0 = middle of the measured ug1 range: at the -7.0 edge the
    # polynomial fit window is one-sided (half the swing has no data)
    # and the IMD coefficients are extrapolation, not physics.
    imd = compute_imd(ctx["isects"], ug1_bias=-9.0, half_swing=3.0)
    assert imd is not None, "imd unexpectedly None on real data"
    assert imd["imd_total"] > 0
    # A mid-range Q on a healthy EL84 must stay well under 20% IMD.
    assert imd["imd_total"] < 20.0


# ---------------------------------------------------------------
# Ig2 estimation and Pg2 on real data
# ---------------------------------------------------------------

def test_ig2_estimation_real_el84():
    """estimate_ig2_at_q on real EL84 pentode data with Ig2 measurements."""
    ctx = _real_el84_context()
    assert ctx is not None, "ctx unexpectedly None on real data"
    # Real pentode data should have ig2 field
    has_ig2 = any("ig2" in p for p in ctx["filtered"])
    assert has_ig2, "real EL84 pentode scan must carry ig2 data"
    ig2 = estimate_ig2_at_q(ctx["filtered"], ug1_q=-7.0, ua_q=200.0, ug2_filter=ctx["ug2"])
    # EL84 screen current: typically 1-5 mA. Zero would mean the Q-point
    # lookup silently failed — the fixture carries ig2 at this Q.
    assert ig2 > 0
    assert ig2 < 15.0, f"Ig2={ig2:.1f}mA seems too high for EL84"
    pg2 = compute_pg2(ctx["ug2"], ig2)
    assert pg2 > 0
    # Pg2 should be < 2W for EL84
    assert pg2 < 2000, f"Pg2={pg2:.0f}mW seems too high"


# ---------------------------------------------------------------
# End-to-end physical sanity: synthetic data
# ---------------------------------------------------------------

class TestE2EPhysicsSynthetic:
    """Full pipeline physical sanity on synthetic Koren models.

    Each test runs the complete chain: model → intersections → distortion
    → stage params → pa_avg → headroom. Checks physical bounds.
    """

    def test_12ax7_se_full_chain(self):
        """12AX7 SE triode: all metrics within physical bounds."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)
        q = interp_intersection(isects, -1.0)
        assert q is not None

        # Q-point
        assert 0.3 < q["ia"] < 5.0, f"Ia_q={q['ia']:.2f} out of range"
        assert 80 < q["ua"] < 220, f"Ua_q={q['ua']:.0f} out of range"

        # Distortion
        dist = compute_distortion(isects, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert dist is not None
        assert 0 < dist["thd"] < 20
        assert dist["pout_mw"] > 0

        # Stage (model)
        stage = compute_stage_params(isects, ll, ug1_bias=-1.0, model=model)
        assert stage is not None
        assert stage["method"] == "model"
        assert 30 < stage["gain"] < 100
        assert 20 < stage["zout"] < 70
        assert 60 < stage["mu"] < 150

        # Pa avg
        pa = compute_pa_avg(model, ll, ug1_bias=-1.0, half_swing=0.5, ub=250.0)
        assert pa is not None
        assert pa["pa_avg_mw"] > 0
        assert pa["pa_avg_mw"] < pa.get("pdc_avg_mw", 99999)  # energy conservation

    def test_el84_pp_full_chain(self):
        """EL84 PP pentode at safe operating point: all metrics physical."""
        model, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)  # lower Ub for safe Pa
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(
            model, ll, ug1_vals, ug2=250.0, ug1_bias=-9.0,
        )

        # Distortion
        dist = compute_distortion(isects, ug1_bias=-9.0, half_swing=3.0, ub=250.0)
        assert dist is not None
        assert 0 < dist["thd"] < 30
        assert dist["pout_mw"] > 0
        assert dist["amp_class"] in ("A", "AB", "B")

        # Stage (model)
        stage = compute_stage_params(
            isects, ll, ug1_bias=-9.0, model=model, model_ug2=250.0,
        )
        assert stage is not None
        assert stage["method"] == "model"
        assert stage["gain"] > 1
        assert stage["zout"] > 0

        # Pa avg
        pa = compute_pa_avg(
            model, ll, ug1_bias=-9.0, half_swing=3.0, ug2=250.0, ub=250.0,
        )
        assert pa is not None
        assert pa["pa_avg_mw"] > 0
        assert pa["ia_avg"] > 0
        # Pa_avg < Pdc (energy conservation)
        if "pdc_avg_mw" in pa:
            assert pa["pa_avg_mw"] < pa["pdc_avg_mw"]

    def test_el84_ul_vs_pentode_physics(self):
        """EL84 UL 20%: Zout and gain lower than pentode, ra lower."""
        model, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})

        # Pentode
        isects_p = find_intersections_model(
            model, ll, ug1_vals, ug2=250.0, ug1_bias=-9.0,
        )
        stage_p = compute_stage_params(
            isects_p, ll, ug1_bias=-9.0, model=model, model_ug2=250.0,
        )

        # UL 20%
        wrapped = UltralinearModelWrapper(model, ug2_nom=250.0, tap=0.20)
        isects_ul = find_intersections_model(
            wrapped, ll, ug1_vals, ug2=250.0, ug1_bias=-9.0,
        )
        stage_ul = compute_stage_params(
            isects_ul, ll, ug1_bias=-9.0, model=wrapped, model_ug2=250.0,
        )

        assert stage_p is not None and stage_ul is not None
        # UL should reduce Zout (toward triode)
        assert stage_ul["zout"] < stage_p["zout"]
        # UL should reduce gain
        assert stage_ul["gain"] < stage_p["gain"]
        # UL ra < pentode ra (from model_gm_ra), evaluated at the
        # actual Q point of each configuration
        q_p = interp_intersection(isects_p, -9.0)
        q_ul = interp_intersection(isects_ul, -9.0)
        assert q_p is not None and q_ul is not None
        gm_p = model_gm_ra(model, q_p["ua"], -9.0, 250.0)
        gm_ul = model_gm_ra(wrapped, q_ul["ua"], -9.0, 250.0)
        assert gm_p is not None and gm_ul is not None
        assert gm_ul["ra"] < gm_p["ra"]

    def test_12au7_cf_full_chain(self):
        """12AU7 cathode follower: gain < 1, low Zout."""
        from lm19.amplifier import CathodeFollowerLoadLine
        model, pts = quick_triode("12AU7")
        ll = CathodeFollowerLoadLine(ub=250, rk=10.0, rl=10.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)

        stage = compute_stage_params(isects, ll, ug1_bias=-10.0, model=model)
        assert stage is not None
        assert stage["method"] == "model"
        assert 0 < stage["gain"] < 1.0  # CF gain always < 1
        assert stage["zout"] < 2.0  # CF Zout is low

    def test_energy_conservation_all_tubes(self):
        """For all synthetic tubes: Pa_avg < Pdc_avg (energy conservation)."""
        configs = [
            ("12AX7", "triode", ResistiveLoadLine(ub=250, ra=100.0), -1.0, 0.5, 0.0),
            ("12AU7", "triode", ResistiveLoadLine(ub=250, ra=47.0), -8.0, 3.0, 0.0),
            ("EL84", "pentode", ResistiveLoadLine(ub=250, ra=5.0), -9.0, 3.0, 250.0),
        ]
        for name, topo, ll, ug1, swing, ug2 in configs:
            if topo == "triode":
                model, _ = quick_triode(name)
            else:
                model, _ = quick_pentode(name)
            pa = compute_pa_avg(model, ll, ug1_bias=ug1, half_swing=swing, ug2=ug2, ub=ll.ub)
            assert pa is not None, f"{name}: pa_avg returned None"
            assert pa["pa_avg_mw"] > 0, f"{name}: Pa_avg <= 0"
            if "pdc_avg_mw" in pa:
                assert pa["pa_avg_mw"] < pa["pdc_avg_mw"], (
                    f"{name}: Pa_avg={pa['pa_avg_mw']:.0f} >= Pdc={pa['pdc_avg_mw']:.0f}"
                )


# ---------------------------------------------------------------
# End-to-end physical sanity: real EL84 data
# ---------------------------------------------------------------

class TestE2EPhysicsRealData:
    """Full pipeline physical sanity on real EL84 measurement data."""

    def _load(self):
        model, pts = _fit_el84_pentode()
        if model is None:
            return None
        ug2_vals = sorted({round(p.get("ug2", 0.0), 0) for p in pts})
        ug2 = max(ug2_vals) if ug2_vals else 200.0
        return {"model": model, "pts": pts, "ug2": ug2}

    def test_real_el84_full_chain(self):
        """Real EL84: model fit → stage → pa_avg → all physical."""
        ctx = self._load()
        assert ctx is not None, "ctx unexpectedly None on real data"
        model, pts, ug2 = ctx["model"], ctx["pts"], ctx["ug2"]
        ll = ResistiveLoadLine(ub=250.0, ra=5.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=ug2)

        # Stage params (model)
        stage = compute_stage_params(
            isects, ll, ug1_bias=-7.0, model=model, model_ug2=ug2,
        )
        assert stage is not None
        assert stage["method"] == "model"
        assert stage["gain"] > 1
        assert stage["zout"] > 0

        # Pa avg
        pa = compute_pa_avg(model, ll, ug1_bias=-7.0, half_swing=3.0, ug2=ug2, ub=250.0)
        assert pa is not None
        assert pa["pa_avg_mw"] > 0
        assert pa["pa_avg_mw"] < 12000  # Pa_max = 12W for EL84
        if "pdc_avg_mw" in pa:
            assert pa["pa_avg_mw"] < pa["pdc_avg_mw"]  # energy

        # Headroom
        hr = compute_headroom(isects, ug1_bias=-7.0, pa_max=12.0, load_line=ll)
        assert hr is not None
        assert hr["max_swing"] > 0

    def test_real_el84_methods_agree(self):
        """Real EL84: 5pt, DFT, model stage — all in same ballpark."""
        ctx = self._load()
        assert ctx is not None, "ctx unexpectedly None on real data"
        model, pts, ug2 = ctx["model"], ctx["pts"], ctx["ug2"]
        ll = ResistiveLoadLine(ub=250.0, ra=5.0)
        filtered = [p for p in pts if abs(p.get("ug2", 0) - ug2) < 10]
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects_pts = find_intersections(filtered, ll)
        isects_mdl = find_intersections_model(model, ll, ug1_vals, ug2=ug2)

        # Distortion: 5pt from points vs DFT from model
        # -9.0: middle of the real ug1 range; at the -7.0 edge both
        # methods honestly reject and the checks below would not run.
        d5 = compute_distortion(isects_pts, ug1_bias=-9.0, half_swing=2.0, ub=250.0)
        dd = compute_distortion_dft(model, ll, ug1_bias=-9.0, half_swing=2.0, ug2=ug2, ub=250.0)

        assert d5 is not None and dd is not None, (
            "distortion must compute on real EL84 at the standard op")
        # Both should give non-zero THD, within factor 3
        assert d5["thd"] > 0 and dd["thd"] > 0
        ratio = max(d5["thd"], dd["thd"]) / max(min(d5["thd"], dd["thd"]), 0.01)
        assert ratio < 3.0

        # Stage: model vs numerical — same order of magnitude
        stage_mdl = compute_stage_params(
            isects_mdl, ll, ug1_bias=-7.0, model=model, model_ug2=ug2,
        )
        stage_num = compute_stage_params(
            isects_pts, ll, ug1_bias=-7.0, points=filtered,
        )
        assert stage_mdl is not None and stage_num is not None
        assert stage_mdl["gain"] == pytest.approx(stage_num["gain"], rel=0.5)


# ---------------------------------------------------------------
# Multiple operating points: sweep through bias/swing/Ra
# ---------------------------------------------------------------

class TestMultipleOperatingPoints:
    """Verify physics across a range of operating points.

    Same tube, different Ub/Ra/Ug1/swing — all must stay physical.
    """

    # ── 12AX7 triode: sweep bias ────────────────────────────────

    @pytest.mark.parametrize("ug1", [-0.5, -1.0, -1.5, -2.0, -2.5])
    def test_12ax7_bias_sweep_physical(self, ug1):
        """12AX7 at different bias points: all metrics positive and bounded."""
        model, pts = quick_triode("12AX7")
        ll = ResistiveLoadLine(ub=250, ra=100.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)
        # half_swing must be >= the 0.5 V grid step of quick_triode:
        # a smaller swing window holds < MIN_CURVES_IN_SWING curves and
        # the sparse-data guard rejects EVERY bias point.
        dist = compute_distortion(isects, ug1_bias=ug1, half_swing=0.5, ub=250.0)
        assert dist is not None, f"distortion must compute at ug1={ug1}"
        assert dist["thd"] >= 0
        assert dist["pout_mw"] >= 0
        assert dist["ia_0"] >= 0
        stage = compute_stage_params(isects, ll, ug1_bias=ug1, model=model)
        assert stage is not None, f"stage must compute at ug1={ug1}"
        assert stage["gain"] > 0
        assert stage["zout"] > 0

    # ── EL84 pentode: sweep Ra ───────────────────────────────────

    @pytest.mark.parametrize("ra", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_el84_ra_sweep_physical(self, ra):
        """EL84 pentode at different Ra: gain and Zout change, all positive."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=ra)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals, ug2=250.0)
        stage = compute_stage_params(
            isects, ll, ug1_bias=-9.0, model=model, model_ug2=250.0,
        )
        assert stage is not None, f"stage must compute on synthetic EL84 at ra={ra}"
        assert stage["gain"] > 0
        assert stage["zout"] > 0
        assert stage["mu"] > 0

    # ── EL84: sweep swing amplitude ──────────────────────────────

    @pytest.mark.parametrize("swing", [0.5, 1.0, 2.0, 4.0, 6.0])
    def test_el84_swing_sweep_pa_avg(self, swing):
        """EL84: Pa_avg physical for different signal amplitudes."""
        model, pts = quick_pentode("EL84")
        ll = ResistiveLoadLine(ub=250, ra=5.0)
        pa = compute_pa_avg(
            model, ll, ug1_bias=-9.0, half_swing=swing, ug2=250.0, ub=250.0,
        )
        assert pa is not None, f"pa_avg must compute on synthetic EL84 at swing={swing}"
        assert pa["pa_avg_mw"] > 0
        assert pa["ia_avg"] > 0
        assert pa["pa_peak_mw"] >= pa["pa_avg_mw"]
        if "pdc_avg_mw" in pa:
            assert pa["pa_avg_mw"] < pa["pdc_avg_mw"]

    # ── 12AU7: sweep Ub ──────────────────────────────────────────

    @pytest.mark.parametrize("ub", [100, 150, 200, 250, 300])
    def test_12au7_ub_sweep_physical(self, ub):
        """12AU7 at different supply voltages: all metrics bounded."""
        model, pts = quick_triode("12AU7")
        ll = ResistiveLoadLine(ub=float(ub), ra=47.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(model, ll, ug1_vals)
        dist = compute_distortion(isects, ug1_bias=-8.0, half_swing=2.0, ub=float(ub))
        assert dist is not None, f"dist must compute on synthetic 12AU7 at ub={ub}"
        assert dist["thd"] >= 0
        assert dist["pout_mw"] >= 0
        stage = compute_stage_params(isects, ll, ug1_bias=-8.0, model=model)
        assert stage is not None, f"stage must compute on synthetic 12AU7 at ub={ub}"
        assert stage["gain"] > 0

    # ── EL84 PP: sweep Raa ───────────────────────────────────────

    @pytest.mark.parametrize("raa", [4.0, 6.0, 8.0, 10.0, 16.0])
    def test_el84_pp_raa_sweep_physical(self, raa):
        """EL84 PP at different Raa: stage params positive."""
        model, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=raa)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})
        isects = find_intersections_model(
            model, ll, ug1_vals, ug2=250.0, ug1_bias=-9.0,
        )
        stage = compute_stage_params(
            isects, ll, ug1_bias=-9.0, model=model, model_ug2=250.0,
        )
        # quick_pentode guarantees 11 Ug1 curves intersecting at every
        # Raa in the sweep — None here means a stage-analysis regression.
        assert stage is not None
        assert stage["gain"] > 0
        assert stage["zout"] > 0

    # ── UL tap sweep ─────────────────────────────────────────────

    @pytest.mark.parametrize("tap", [0.0, 0.10, 0.20, 0.43, 0.60, 1.0])
    def test_el84_ul_tap_sweep_physical(self, tap):
        """EL84 UL at different taps: stage params positive, Pa physical."""
        model, pts = quick_pentode("EL84")
        ll = PushPullLoadLine(ub=250, ra_aa=8.0)
        ug1_vals = sorted({round(p["ug1"], 1) for p in pts})

        if tap == 0.0:
            m = model
        else:
            m = UltralinearModelWrapper(model, ug2_nom=250.0, tap=tap)
        isects = find_intersections_model(
            m, ll, ug1_vals, ug2=250.0, ug1_bias=-9.0,
        )
        stage = compute_stage_params(
            isects, ll, ug1_bias=-9.0, model=m, model_ug2=250.0,
        )
        # The UL wrapper is transparent to stage analysis: for any tap
        # in 0..1 the model yields valid intersections — None means a
        # regression.
        assert stage is not None
        assert stage["gain"] > 0
        assert stage["zout"] > 0
        pa = compute_pa_avg(
            m, ll, ug1_bias=-9.0, half_swing=2.0, ug2=250.0, ub=250.0,
        )
        assert pa is not None
        assert pa["pa_avg_mw"] > 0

