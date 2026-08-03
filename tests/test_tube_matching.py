"""Tests for lm19.tube_matching module."""

import math
import unittest

from lm19.tube_matching import (
    TubeRecord,
    MatchGroup,
    MatchResult,
    CurveDistanceInfo,
    CurveMatchResult,
    MIN_OVERLAP_POINTS,
    WARN_OVERLAP_POINTS,
    compute_distance,
    compute_sample_means,
    find_similar,
    group_pairs,
    group_n,
    group_by_distance_matrix,
    build_curve_distance_matrix,
    match_curves,
    match_tubes,
    select_measurements,
    predicted_iq_imbalance_ma,
    _extract_record,
    _conditions_key,
    _greedy_pair_assign,
    default_weights_for_mode,
    delta_quality,
    DEFAULT_WEIGHTS_PENTODE,
    DEFAULT_WEIGHTS_TRIODE,
    DELTA_EXCELLENT,
    DELTA_GOOD,
    DELTA_FAIR,
    CURVE_DELTA_THRESHOLDS,
    MATCHING_PROTOCOLS,
    MATCHING_PROTOCOL_STRICT,
    MATCHING_PROTOCOL_SHARED,
    MATCHING_PROTOCOL_INDIVIDUAL,
    DEFAULT_MATCHING_PROTOCOL,
    MATCH_ANCHOR_ERRORS,
    ANCHOR_ERR_NOT_FOUND,
    ANCHOR_ERR_INCOMPATIBLE,
    _weights_for_protocol,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE_CONNECTED,
)


def _rec(lid: str, ia: float, s: float, r: float,
         an: int = 1, ts: str = "2026-01-01T00:00:00",
         index: float = 90.0) -> TubeRecord:
    """Helper to create TubeRecord for tests."""
    return TubeRecord(
        lamp_id=lid, timestamp=ts, an=an,
        ia=ia, s=s, r=r, index=index,
    )


def _entry(lid: str, ia: float, s: float, r: float,
           an: int = 1, ts: str = "2026-01-01T00:00:00",
           index: float = 90.0,
           ua: float = 250.0, ug1: float = -2.5, ug2: float = 150.0,
           ug2_mode: str = TOPOLOGY_PENTODE) -> dict:
    """Helper to create a minimal health measurement dict."""
    return {
        "lamp_id": lid,
        "timestamp": ts,
        "conditions": {"ua": ua, "ug1": ug1, "ug2": ug2, "an": an,
                        "ug2_mode": ug2_mode},
        "srk": {"s": s, "r": r, "ia_op": ia},
        "health": {"raw": {"ia_op": ia}, "index": index},
    }


def _servo_entry(lid: str, *, ia_plan, bias_shift, s: float, r: float,
                 ia_op: float = 48.0, an: int = 1,
                 ts: str = "2026-01-01T00:00:00", index: float = 90.0,
                 ua: float = 250.0, ug1: float = -7.3, ug2: float = 250.0,
                 ug2_mode: str = TOPOLOGY_PENTODE) -> dict:
    """A bias-servo health entry: op Ia sits at the reference by
    construction; the wear-honest current lives in raw.ia_plan_ma.
    ``ia_plan``/``bias_shift`` = None model a legacy servo entry saved
    before those fields existed."""
    e = _entry(lid, ia_op, s, r, an=an, ts=ts, index=index,
               ua=ua, ug1=ug1, ug2=ug2, ug2_mode=ug2_mode)
    e["conditions"]["bias_servo"] = True
    if ia_plan is not None:
        e["health"]["raw"]["ia_plan_ma"] = ia_plan
    if bias_shift is not None:
        e["health"]["metrics"] = {"bias_shift_v": bias_shift}
    return e


def _fixed_entry(lid: str, ia: float, s: float, r: float, **kw) -> dict:
    """A fixed-bias entry at the EL84-like plan point shared with
    ``_servo_entry`` (same conditions key modulo the servo flag)."""
    kw.setdefault("ug1", -7.3)
    kw.setdefault("ug2", 250.0)
    return _entry(lid, ia, s, r, **kw)


class TestMatchingProtocolRegistry(unittest.TestCase):

    def test_registry_contents(self):
        self.assertEqual(
            MATCHING_PROTOCOLS,
            (MATCHING_PROTOCOL_STRICT, MATCHING_PROTOCOL_SHARED,
             MATCHING_PROTOCOL_INDIVIDUAL))
        self.assertEqual(DEFAULT_MATCHING_PROTOCOL, MATCHING_PROTOCOL_STRICT)

    def test_unknown_protocol_raises(self):
        with self.assertRaises(ValueError):
            match_tubes([_fixed_entry("A", 46.0, 11.0, 30.0)],
                        protocol="autobias")


class TestExtractRecordProtocolFields(unittest.TestCase):

    def test_servo_entry_carries_plan_point(self):
        rec = _extract_record(_servo_entry(
            "A", ia_plan=35.2, bias_shift=-0.9, s=10.8, r=31.0))
        self.assertTrue(rec.servo)
        self.assertAlmostEqual(rec.ia_plan, 35.2)
        self.assertAlmostEqual(rec.bias_shift, -0.9)
        # op current stays the reference-point one
        self.assertAlmostEqual(rec.ia, 48.0)

    def test_fixed_entry_is_its_own_plan_point(self):
        rec = _extract_record(_fixed_entry("B", 46.5, 11.1, 30.0))
        self.assertFalse(rec.servo)
        self.assertAlmostEqual(rec.ia_plan, 46.5)
        self.assertEqual(rec.bias_shift, 0.0)

    def test_legacy_servo_entry_has_unknown_plan(self):
        rec = _extract_record(_servo_entry(
            "C", ia_plan=None, bias_shift=None, s=10.8, r=31.0))
        self.assertTrue(rec.servo)
        self.assertIsNone(rec.ia_plan)
        self.assertIsNone(rec.bias_shift)


class TestProtocolPools(unittest.TestCase):
    """Pool membership per protocol — the conditions key of the pool is
    a servo one, entries are a servo/fixed mix at the same OP."""

    def _entries(self):
        return [
            _servo_entry("SA", ia_plan=40.0, bias_shift=-0.5, s=11.0, r=30.0),
            _servo_entry("SB", ia_plan=42.0, bias_shift=-0.2, s=11.2, r=30.5),
            _fixed_entry("FA", 45.0, 10.9, 29.5),
        ]

    def test_strict_pool_keeps_servo_only(self):
        entries = self._entries()
        recs = select_measurements(
            entries, conditions=_conditions_key(entries[0]),
            protocol=MATCHING_PROTOCOL_STRICT)
        self.assertEqual(sorted(r.lamp_id for r in recs), ["SA", "SB"])

    def test_shared_pool_mixes_both_kinds(self):
        entries = self._entries()
        recs = select_measurements(
            entries, conditions=_conditions_key(entries[0]),
            protocol=MATCHING_PROTOCOL_SHARED)
        self.assertEqual(sorted(r.lamp_id for r in recs), ["FA", "SA", "SB"])

    def test_individual_pool_keeps_servo_only_from_fixed_key(self):
        # Even when the CONDITIONS key comes from a fixed entry, the
        # individual protocol admits servo runs only.
        entries = self._entries()
        recs = select_measurements(
            entries, conditions=_conditions_key(entries[2]),
            protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertEqual(sorted(r.lamp_id for r in recs), ["SA", "SB"])

    def test_different_op_never_mixes(self):
        entries = self._entries() + [
            _fixed_entry("XX", 45.0, 10.9, 29.5, ua=200.0)]
        recs = select_measurements(
            entries, conditions=_conditions_key(entries[0]),
            protocol=MATCHING_PROTOCOL_SHARED)
        self.assertNotIn("XX", [r.lamp_id for r in recs])


class TestSharedIaTerm(unittest.TestCase):
    """shared_bias must match on the PLAN-point current.

    Discriminator: two servo tubes both sit at op Ia = 48 (reference, by
    construction) but their plan currents diverge (35 worn vs 47 strong).
    An implementation matching on op Ia sees them as identical and pairs
    them; the correct one pairs the strong servo tube with the fixed
    46.5 mA tube."""

    def _result(self):
        entries = [
            _servo_entry("WORN", ia_plan=35.0, bias_shift=-1.1,
                         s=11.0, r=30.0),
            _servo_entry("STRONG", ia_plan=47.0, bias_shift=-0.1,
                         s=11.0, r=30.0),
            _fixed_entry("FIX", 46.5, 11.0, 30.0),
        ]
        return match_tubes(
            entries, group_size=2,
            protocol=MATCHING_PROTOCOL_SHARED,
            max_iq_imbalance_pct=0.0,  # gate off: isolate the Ia term
        )

    def test_pairs_by_plan_current(self):
        result = self._result()
        self.assertEqual(len(result.groups), 1)
        ids = sorted(r.lamp_id for r in result.groups[0].records)
        self.assertEqual(ids, ["FIX", "STRONG"])

    def test_records_expose_the_matched_current(self):
        # Method visibility: the record the UI prints carries the value
        # the metric actually used (plan current), not the servo op Ia.
        result = self._result()
        by_id = {r.lamp_id: r for g in result.groups for r in g.records}
        by_id.update({r.lamp_id: r for r in result.unmatched})
        self.assertAlmostEqual(by_id["STRONG"].ia, 47.0)
        self.assertAlmostEqual(by_id["WORN"].ia, 35.0)


class TestSharedIqGate(unittest.TestCase):
    """The δIq gate poisons the matrix BEFORE selection.

    Discriminator vs a post-filter: the tightest-by-S pair violates the
    gate; a gate-before-selection walks on and still forms ONE allowed
    pair, while a post-filter would pick the tight pair first, discard
    it, and deliver zero groups."""

    def _entries(self):
        # S makes A-B the tightest pair by far; their plan currents are
        # 10 mA apart (δIq ≈ 24% of the 41 mA mean — over a 10% gate).
        # C is the S-outsider but δIq-compatible with A.
        return [
            _servo_entry("A", ia_plan=46.0, bias_shift=-0.2,
                         s=11.50, r=30.0),
            _servo_entry("B", ia_plan=36.0, bias_shift=-1.2,
                         s=11.48, r=30.0),
            _fixed_entry("C", 44.0, 10.90, 30.0),
        ]

    def _match(self, pct):
        return match_tubes(
            self._entries(), group_size=2,
            protocol=MATCHING_PROTOCOL_SHARED,
            max_iq_imbalance_pct=pct,
            weights={"ia": 0.0, "s": 1.0, "r": 0.0},
        )

    def test_gate_off_picks_the_tight_pair(self):
        result = self._match(0.0)
        ids = sorted(r.lamp_id for r in result.groups[0].records)
        self.assertEqual(ids, ["A", "B"])

    def test_gated_pair_is_replaced_not_dropped(self):
        result = self._match(10.0)
        self.assertEqual(len(result.groups), 1,
                         "gate must reroute selection, not empty it")
        ids = sorted(r.lamp_id for r in result.groups[0].records)
        self.assertEqual(ids, ["A", "C"])

    def test_group_carries_predicted_imbalance(self):
        result = self._match(10.0)
        self.assertAlmostEqual(result.groups[0].iq_imbalance_ma, 2.0)

    def test_gate_boundary_equal_passes(self):
        # δIq exactly at the limit passes (strict > comparison): A vs C
        # differ by 2 mA on a 45 mA mean → 4.4444%. At exactly that pct
        # the pair must survive; just below it must not.
        pct_exact = 2.0 / 45.0 * 100.0
        self.assertEqual(len(self._match(pct_exact).groups), 1)
        just_below = self._match(pct_exact - 0.01)
        self.assertEqual(len(just_below.groups), 0)

    def test_legacy_servo_never_matched_silently(self):
        entries = self._entries() + [
            _servo_entry("LEG", ia_plan=None, bias_shift=None,
                         s=11.50, r=30.0)]
        result = match_tubes(
            entries, group_size=2,
            protocol=MATCHING_PROTOCOL_SHARED,
            max_iq_imbalance_pct=0.0,
            weights={"ia": 0.0, "s": 1.0, "r": 0.0},
        )
        # LEG has the best S twin (A at 11.50) but no plan current — it
        # must end up unmatched, not matched on the fake op Ia.
        for g in result.groups:
            self.assertNotIn("LEG", [r.lamp_id for r in g.records])
        self.assertIn("LEG", [r.lamp_id for r in result.unmatched])


class TestSharedLegacyVisibility(unittest.TestCase):
    """Legacy servo records are gated out of shared pools — the WARNING
    must name them (a silently shrinking pool looks like 'no match')."""

    def test_legacy_exclusion_is_logged_with_lamp_ids(self):
        entries = [
            _servo_entry("LEG", ia_plan=None, bias_shift=None,
                         s=11.0, r=30.0),
            _fixed_entry("A", 44.0, 11.0, 30.0),
            _fixed_entry("B", 45.0, 11.1, 30.0),
        ]
        with self.assertLogs("lm19.tube_matching", level="WARNING") as cm:
            match_tubes(entries, group_size=2,
                        protocol=MATCHING_PROTOCOL_SHARED)
        self.assertTrue(any("LEG" in line for line in cm.output))

    def test_clean_pool_stays_silent(self):
        entries = [
            _fixed_entry("A", 44.0, 11.0, 30.0),
            _fixed_entry("B", 45.0, 11.1, 30.0),
        ]
        with self.assertNoLogs("lm19.tube_matching", level="WARNING"):
            match_tubes(entries, group_size=2,
                        protocol=MATCHING_PROTOCOL_SHARED)


class TestIndividualProtocol(unittest.TestCase):

    def test_ia_carries_no_weight(self):
        # Plan currents differ by ~30%, S/R equal → distance ≈ 0.
        entries = [
            _servo_entry("A", ia_plan=35.0, bias_shift=-1.1,
                         s=11.0, r=30.0),
            _servo_entry("B", ia_plan=46.0, bias_shift=-0.2,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(entries, group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertEqual(len(result.groups), 1)
        self.assertLess(result.groups[0].delta, 0.5)

    def test_weights_renormalised_to_percent_scale(self):
        # With pentode weights {ia .5, s .5, r 0} the s-weight must
        # renormalise to 1.0: a 10% S difference reads ≈ 10 (not ≈ 7 from
        # sqrt(0.5)·10 that unnormalised weights would give).
        entries = [
            _servo_entry("A", ia_plan=40.0, bias_shift=-0.5,
                         s=10.0, r=30.0),
            _servo_entry("B", ia_plan=40.0, bias_shift=-0.5,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(entries, group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL,
                             weights=dict(DEFAULT_WEIGHTS_PENTODE))
        self.assertAlmostEqual(result.groups[0].delta, 100.0 * 1.0 / 10.5,
                               places=1)

    def test_shift_outside_adjust_range_is_unmatched(self):
        # Plan bias −7.3 V, range 30% → limit 2.19 V. B's shift 2.5 V is
        # beyond any bias pot of such an amp — it cannot be dialed in.
        entries = [
            _servo_entry("A", ia_plan=44.0, bias_shift=-0.5,
                         s=11.0, r=30.0),
            _servo_entry("B", ia_plan=45.0, bias_shift=2.5,
                         s=11.0, r=30.0),
            _servo_entry("C", ia_plan=43.0, bias_shift=-1.0,
                         s=10.8, r=30.5),
        ]
        result = match_tubes(entries, group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL,
                             bias_adjust_range_pct=30.0)
        self.assertEqual(len(result.groups), 1)
        ids = sorted(r.lamp_id for r in result.groups[0].records)
        self.assertEqual(ids, ["A", "C"])
        self.assertIn("B", [r.lamp_id for r in result.unmatched])

    def test_shift_at_exact_range_boundary_passes(self):
        # limit = 30% · 7.3 = 2.19 V; a shift of exactly −2.19 V passes.
        entries = [
            _servo_entry("A", ia_plan=44.0, bias_shift=-2.19,
                         s=11.0, r=30.0),
            _servo_entry("B", ia_plan=45.0, bias_shift=0.3,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(entries, group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL,
                             bias_adjust_range_pct=30.0)
        self.assertEqual(len(result.groups), 1)

    def test_range_gate_off_by_default(self):
        entries = [
            _servo_entry("A", ia_plan=44.0, bias_shift=-5.5,
                         s=11.0, r=30.0),
            _servo_entry("B", ia_plan=45.0, bias_shift=5.0,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(entries, group_size=2,
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertEqual(len(result.groups), 1)


class TestSimilarModeProtocol(unittest.TestCase):

    def test_shared_similar_gates_and_annotates(self):
        entries = [
            _fixed_entry("ANCH", 44.0, 11.0, 30.0),
            _servo_entry("NEAR", ia_plan=43.0, bias_shift=-0.3,
                         s=11.0, r=30.0),
            _servo_entry("FAR", ia_plan=33.0, bias_shift=-1.4,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(
            entries, mode="similar", anchor_lamp_id="ANCH",
            protocol=MATCHING_PROTOCOL_SHARED,
            max_iq_imbalance_pct=10.0,
        )
        ranked_ids = [g.records[0].lamp_id for g in result.groups]
        self.assertIn("NEAR", ranked_ids)
        self.assertNotIn("FAR", ranked_ids,
                         "δIq 11 mA on a 38.5 mA mean is past the gate")
        near = result.groups[ranked_ids.index("NEAR")]
        self.assertAlmostEqual(near.iq_imbalance_ma, 1.0)

    def test_specific_anchor_rebuilds_gate_from_its_conditions(self):
        # Bulk conditions auto-detect from entries[0] (plan −7.3 V →
        # range limit 2.19 V), but the SPECIFIC anchor sits at −20 V
        # (limit 6 V). Candidates shifted ~3 V are legal for the
        # anchor's amp; a stale gate built from the bulk conditions
        # would block them.
        entries = [
            _fixed_entry("BULK", 44.0, 11.0, 30.0),      # −7.3 V pool
            _servo_entry("ANCH", ia_plan=90.0, bias_shift=-3.0,
                         s=8.0, r=25.0, ug1=-20.0,
                         ts="2026-01-02T00:00:00"),
            _servo_entry("CAND", ia_plan=92.0, bias_shift=3.2,
                         s=8.1, r=25.5, ug1=-20.0),
        ]
        result = match_tubes(
            entries, mode="similar",
            anchor_lamp_id="ANCH", anchor_timestamp="2026-01-02T00:00:00",
            protocol=MATCHING_PROTOCOL_INDIVIDUAL,
            bias_adjust_range_pct=30.0,
        )
        self.assertIn("CAND", [g.records[0].lamp_id for g in result.groups])

    def test_strict_similar_unannotated(self):
        entries = [
            _fixed_entry("ANCH", 44.0, 11.0, 30.0),
            _fixed_entry("OTHER", 43.0, 11.0, 30.0),
        ]
        result = match_tubes(entries, mode="similar",
                             anchor_lamp_id="ANCH")
        self.assertIsNone(result.groups[0].iq_imbalance_ma)


class TestSimilarAnchorPool(unittest.TestCase):
    """Default "Find similar" must build the pool from the ANCHOR lamp's
    own conditions — with the bias-servo flag in the key, the newest
    entry of another lamp deciding the pool silently anchored the
    ranking on a different lamp."""

    def test_default_anchor_uses_its_own_lamp_conditions(self):
        # Newest history entry is a SERVO run of another lamp at the same
        # OP: the old bulk auto-detect made the pool servo-only, dropped
        # the clicked fixed-bias lamp and fell back to ranking around SRV.
        entries = [
            _servo_entry("SRV", ia_plan=40.0, bias_shift=-0.5,
                         s=11.0, r=30.0, ts="2026-03-01T00:00:00"),
            _fixed_entry("ANCH", 44.0, 11.0, 30.0,
                         ts="2026-02-01T00:00:00"),
            _fixed_entry("CAND", 43.5, 11.1, 30.2,
                         ts="2026-01-01T00:00:00"),
        ]
        result = match_tubes(entries, mode="similar",
                             anchor_lamp_id="ANCH")
        self.assertIsNone(result.anchor_error)
        self.assertEqual(result.anchor.lamp_id, "ANCH")
        ranked = [g.records[0].lamp_id for g in result.groups]
        self.assertIn("CAND", ranked)
        self.assertNotIn("SRV", ranked)
        # The pool the match ran on is the anchor's fixed-bias one.
        self.assertIs(result.conditions_used[4], False)

    def test_individual_default_anchor_prefers_its_servo_run(self):
        # individual_bias admits only servo runs: the lamp's newest entry
        # is fixed-bias, so the anchor must be its older SERVO run, not a
        # missing-anchor error and not the fixed run.
        entries = [
            _fixed_entry("A", 44.0, 11.0, 30.0, ts="2026-03-01T00:00:00"),
            _servo_entry("A", ia_plan=40.0, bias_shift=-0.5, s=11.0,
                         r=30.0, ts="2026-02-01T00:00:00"),
            _servo_entry("B", ia_plan=41.0, bias_shift=-0.4, s=11.1,
                         r=30.1, ts="2026-01-01T00:00:00"),
        ]
        result = match_tubes(entries, mode="similar", anchor_lamp_id="A",
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertIsNone(result.anchor_error)
        self.assertEqual(result.anchor.lamp_id, "A")
        self.assertEqual(result.anchor.timestamp, "2026-02-01T00:00:00")
        self.assertIn("B", [g.records[0].lamp_id for g in result.groups])

    def test_individual_specific_fixed_anchor_is_incompatible(self):
        # The pool rule exists BECAUSE a fixed run's S is not measured at
        # the reference current; an explicitly anchored fixed measurement
        # must refuse to rank, not silently violate the premise.
        entries = [
            _fixed_entry("A", 44.0, 11.0, 30.0, ts="2026-03-01T00:00:00"),
            _servo_entry("B", ia_plan=41.0, bias_shift=-0.4, s=11.1,
                         r=30.1),
        ]
        result = match_tubes(
            entries, mode="similar", anchor_lamp_id="A",
            anchor_timestamp="2026-03-01T00:00:00",
            protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertEqual(result.anchor_error, ANCHOR_ERR_INCOMPATIBLE)
        self.assertIsNone(result.anchor)
        self.assertEqual(result.groups, [])

    def test_anchor_an_selects_the_clicked_system(self):
        # Twin-triode: clicking the An2 row must anchor on the An2 record
        # (entry order puts An1 first — an unfiltered lookup grabs it).
        # An2 was measured at a different OP, so the pool conditions must
        # come from the An2 entry too, not the lamp's An1 one.
        entries = [
            _entry("L1", 50.0, 11.0, 30.0, an=1),               # ua=250
            _entry("L1", 40.0, 9.0, 32.0, an=2, ua=300.0),
            _entry("L2", 41.0, 9.1, 31.8, an=2, ua=300.0),
        ]
        result = match_tubes(entries, mode="similar",
                             anchor_lamp_id="L1", anchor_an=2)
        self.assertIsNone(result.anchor_error)
        self.assertEqual(result.anchor.an, 2)
        self.assertAlmostEqual(result.anchor.s, 9.0)
        self.assertEqual(result.conditions_used[0], 300.0)
        self.assertEqual([g.records[0].lamp_id for g in result.groups],
                         ["L2"])

    def test_conditions_used_reports_the_actual_pool(self):
        # Similar mode with a specific anchor switches the pool to the
        # anchor's key; groups mode keeps the caller's tuple. The field
        # is what the UI conditions label shows — it must be the truth.
        bulk = _fixed_entry("BULK", 44.0, 11.0, 30.0)
        anch = _entry("ANCH", 50.0, 10.0, 28.0, ua=300.0, ug1=-10.0,
                      ug2=300.0, ts="2026-01-02T00:00:00")
        cand = _entry("CAND", 51.0, 10.1, 28.2, ua=300.0, ug1=-10.0,
                      ug2=300.0)
        res = match_tubes([bulk, anch, cand], mode="similar",
                          anchor_lamp_id="ANCH",
                          anchor_timestamp="2026-01-02T00:00:00")
        self.assertEqual(res.conditions_used, _conditions_key(anch))

        res_groups = match_tubes([bulk, anch, cand], mode="groups",
                                 conditions=_conditions_key(bulk))
        self.assertEqual(res_groups.conditions_used, _conditions_key(bulk))


class TestSimilarUnmatchedVisibility(unittest.TestCase):
    """Candidates cut by the protocol gate or max_delta must surface in
    ``unmatched`` — groups mode shows them, similar mode used to drop
    them silently."""

    def test_gate_excluded_candidate_lands_in_unmatched(self):
        entries = [
            _fixed_entry("ANCH", 44.0, 11.0, 30.0),
            _servo_entry("NEAR", ia_plan=43.0, bias_shift=-0.3,
                         s=11.0, r=30.0),
            _servo_entry("FAR", ia_plan=33.0, bias_shift=-1.4,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(
            entries, mode="similar", anchor_lamp_id="ANCH",
            protocol=MATCHING_PROTOCOL_SHARED, max_iq_imbalance_pct=10.0)
        self.assertEqual([r.lamp_id for r in result.unmatched], ["FAR"])

    def test_over_max_delta_candidate_lands_in_unmatched(self):
        entries = [
            _entry("ANCH", 50.0, 2.0, 10.0),
            _entry("NEAR", 50.5, 2.01, 10.05),
            _entry("FAR", 90.0, 4.0, 22.0),
        ]
        result = match_tubes(entries, mode="similar",
                             anchor_lamp_id="ANCH", max_delta=5.0)
        self.assertIn("FAR", [r.lamp_id for r in result.unmatched])
        self.assertNotIn("FAR",
                         [g.records[0].lamp_id for g in result.groups])

    def test_legacy_servo_anchor_yields_all_unmatched(self):
        # A legacy servo anchor (no plan-point current) cannot pass the
        # shared gate against anyone: the whole pool must surface as
        # unmatched instead of an inexplicably empty ranking.
        entries = [
            _servo_entry("A", ia_plan=None, bias_shift=None, s=11.0,
                         r=30.0, ts="2026-01-02T00:00:00"),
            _servo_entry("B", ia_plan=40.0, bias_shift=-0.5, s=11.0,
                         r=30.0),
        ]
        result = match_tubes(
            entries, mode="similar", anchor_lamp_id="A",
            anchor_timestamp="2026-01-02T00:00:00",
            protocol=MATCHING_PROTOCOL_SHARED)
        self.assertIsNotNone(result.anchor)
        self.assertEqual(result.groups, [])
        self.assertEqual([r.lamp_id for r in result.unmatched], ["B"])


class TestSimilarCombinedAnode(unittest.TestCase):
    """anode="combined" pools carry an == 0 aggregates: the anchor lamp's
    own aggregate must never rank as a candidate (self-match), yet the
    aggregate IS the anchor when the lamp itself is the reference."""

    def _entries(self):
        return [
            _entry("L1", 50.0, 2.0, 10.0, an=1, ts="2026-01-05T00:00:00"),
            _entry("L1", 52.0, 2.1, 10.2, an=2, ts="2026-01-04T00:00:00"),
            _entry("L2", 51.0, 2.05, 10.1, an=1, ts="2026-01-03T00:00:00"),
            _entry("L2", 51.5, 2.02, 10.15, an=2, ts="2026-01-02T00:00:00"),
        ]

    def test_specific_anchor_excludes_own_combined_twin(self):
        result = match_tubes(
            self._entries(), mode="similar", anchor_lamp_id="L1",
            anchor_timestamp="2026-01-05T00:00:00", anode="combined")
        ranked = [g.records[0].lamp_id for g in result.groups]
        self.assertNotIn("L1", ranked, "a tube must not match itself")
        self.assertIn("L2", ranked)

    def test_default_combined_anchor_is_the_aggregate(self):
        # The clicked row carries an=1, the combined pool holds an=0
        # aggregates — the aggregate must still be accepted as anchor.
        result = match_tubes(
            self._entries(), mode="similar", anchor_lamp_id="L1",
            anchor_an=1, anode="combined")
        self.assertIsNone(result.anchor_error)
        self.assertEqual(result.anchor.lamp_id, "L1")
        self.assertEqual(result.anchor.an, 0)
        ranked = [g.records[0].lamp_id for g in result.groups]
        self.assertEqual(ranked, ["L2"])


class TestCombinedProtocolFields(unittest.TestCase):
    """anode="combined" aggregation of the protocol-aware fields."""

    def _pool(self, e1, e2, protocol):
        return select_measurements([e1, e2], anode="combined",
                                   conditions=_conditions_key(e1),
                                   protocol=protocol)

    def test_combined_averages_plan_current(self):
        e1 = _servo_entry("L", ia_plan=30.0, bias_shift=0.5, an=1,
                          s=11.0, r=30.0)
        e2 = _servo_entry("L", ia_plan=40.0, bias_shift=-0.9, an=2,
                          s=11.0, r=30.0)
        rec = self._pool(e1, e2, MATCHING_PROTOCOL_SHARED)[0]
        self.assertAlmostEqual(rec.ia_plan, 35.0)

    def test_combined_keeps_worst_abs_shift(self):
        # Signs differ: max() without key=abs would pick +0.5 — the
        # adjustment-range gate must hold for BOTH anode systems.
        e1 = _servo_entry("L", ia_plan=30.0, bias_shift=0.5, an=1,
                          s=11.0, r=30.0)
        e2 = _servo_entry("L", ia_plan=40.0, bias_shift=-0.9, an=2,
                          s=11.0, r=30.0)
        rec = self._pool(e1, e2, MATCHING_PROTOCOL_SHARED)[0]
        self.assertAlmostEqual(rec.bias_shift, -0.9)

    def test_combined_servo_flag_requires_all(self):
        e1 = _servo_entry("L", ia_plan=30.0, bias_shift=0.5, an=1,
                          s=11.0, r=30.0)
        e2 = _fixed_entry("L", 44.0, 11.0, 30.0, an=2)
        # shared protocol mixes servo and fixed in one pool
        rec = self._pool(e1, e2, MATCHING_PROTOCOL_SHARED)[0]
        self.assertFalse(rec.servo)
        e2s = _servo_entry("L", ia_plan=40.0, bias_shift=-0.2, an=2,
                           s=11.0, r=30.0)
        rec_all = self._pool(e1, e2s, MATCHING_PROTOCOL_SHARED)[0]
        self.assertTrue(rec_all.servo)

    def test_combined_legacy_plan_propagates_none(self):
        e1 = _servo_entry("L", ia_plan=30.0, bias_shift=0.5, an=1,
                          s=11.0, r=30.0)
        e2 = _servo_entry("L", ia_plan=None, bias_shift=None, an=2,
                          s=11.0, r=30.0)
        rec = self._pool(e1, e2, MATCHING_PROTOCOL_SHARED)[0]
        self.assertIsNone(rec.ia_plan)
        self.assertIsNone(rec.bias_shift)


class TestIndividualSimilar(unittest.TestCase):
    """individual_bias in similar mode: Ia carries no weight and the
    adjustment-range gate feeds unmatched."""

    def test_ranking_ignores_ia(self):
        entries = [
            _servo_entry("A", ia_plan=40.0, bias_shift=-0.5, s=11.0,
                         r=30.0, ia_op=48.0, ts="2026-01-02T00:00:00"),
            # B: identical S, wildly different current draw
            _servo_entry("B", ia_plan=20.0, bias_shift=-1.0, s=11.0,
                         r=30.0, ia_op=30.0),
            # C: same current, sagged transconductance
            _servo_entry("C", ia_plan=40.0, bias_shift=-0.4, s=9.0,
                         r=30.0, ia_op=48.0),
        ]
        result = match_tubes(entries, mode="similar", anchor_lamp_id="A",
                             protocol=MATCHING_PROTOCOL_INDIVIDUAL)
        ranked = [g.records[0].lamp_id for g in result.groups]
        self.assertEqual(ranked[0], "B",
                         "Ia must carry no weight under individual bias")

    def test_shift_gate_feeds_unmatched(self):
        entries = [
            _servo_entry("A", ia_plan=40.0, bias_shift=-0.5, s=11.0,
                         r=30.0, ts="2026-01-02T00:00:00"),
            _servo_entry("NEAR", ia_plan=41.0, bias_shift=-0.6, s=11.0,
                         r=30.0),
            # 3.0 V shift on a −7.3 V plan is past the 30% authority
            _servo_entry("FARSHIFT", ia_plan=42.0, bias_shift=-3.0,
                         s=11.0, r=30.0),
        ]
        result = match_tubes(
            entries, mode="similar", anchor_lamp_id="A",
            protocol=MATCHING_PROTOCOL_INDIVIDUAL,
            bias_adjust_range_pct=30.0)
        self.assertIn("NEAR",
                      [g.records[0].lamp_id for g in result.groups])
        self.assertEqual([r.lamp_id for r in result.unmatched],
                         ["FARSHIFT"])


class TestGroupNProtocolGate(unittest.TestCase):
    """Quad grouping honours the pairwise protocol gate and carries the
    worst pairwise δIq (both were pinned only for pairs)."""

    def _servo4(self, plans):
        return [
            _servo_entry(f"L{i}", ia_plan=p, bias_shift=-0.5,
                         s=11.0 + i * 0.01, r=30.0)
            for i, p in enumerate(plans)
        ]

    def test_quad_blocked_by_gate_goes_unmatched(self):
        # One outlier's δIq poisons every chunk containing it → inf in
        # the matrix → the chunk must not group (ML-069 semantics).
        entries = self._servo4([40.0, 40.5, 41.0, 80.0])
        result = match_tubes(entries, mode="groups", group_size=4,
                             protocol=MATCHING_PROTOCOL_SHARED,
                             max_iq_imbalance_pct=10.0)
        self.assertEqual(result.groups, [])
        self.assertEqual(len(result.unmatched), 4)

    def test_quad_carries_worst_pairwise_iq(self):
        entries = self._servo4([40.0, 41.0, 42.0, 44.0])
        result = match_tubes(entries, mode="groups", group_size=4,
                             protocol=MATCHING_PROTOCOL_SHARED,
                             max_iq_imbalance_pct=0.0)
        self.assertEqual(len(result.groups), 1)
        self.assertAlmostEqual(result.groups[0].iq_imbalance_ma, 4.0)


class TestWeightsForProtocolEdge(unittest.TestCase):

    def test_zero_sr_weights_fall_back_to_s(self):
        w = _weights_for_protocol({"ia": 1.0, "s": 0.0, "r": 0.0},
                                  MATCHING_PROTOCOL_INDIVIDUAL)
        self.assertEqual(w["ia"], 0.0)
        self.assertEqual(w["s"], 1.0)


class TestPredictedIqImbalance(unittest.TestCase):

    def test_direct_difference(self):
        a = _rec("A", 40.0, 11.0, 30.0)
        b = _rec("B", 44.5, 11.0, 30.0)
        a.ia_plan, b.ia_plan = 40.0, 44.5
        self.assertAlmostEqual(predicted_iq_imbalance_ma(a, b), 4.5)

    def test_unknown_plan_is_none(self):
        a = _rec("A", 40.0, 11.0, 30.0)
        b = _rec("B", 44.5, 11.0, 30.0)
        a.ia_plan = None
        b.ia_plan = 44.5
        self.assertIsNone(predicted_iq_imbalance_ma(a, b))


class TestComputeDistance(unittest.TestCase):

    def test_identical_is_zero(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 50.0, 2.0, 10.0)
        means = compute_sample_means([a, b])
        self.assertAlmostEqual(compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means), 0.0)

    def test_symmetry(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 55.0, 2.5, 12.0)
        means = compute_sample_means([a, b])
        d_ab = compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means)
        d_ba = compute_distance(b, a, DEFAULT_WEIGHTS_PENTODE, means)
        self.assertAlmostEqual(d_ab, d_ba)

    def test_positive_for_different(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 60.0, 3.0, 15.0)
        means = compute_sample_means([a, b])
        d = compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means)
        self.assertGreater(d, 0.0)

    def test_weights_affect_result(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 55.0, 2.0, 10.0)  # only Ia differs
        means = compute_sample_means([a, b])

        w_ia_high = {"ia": 1.0, "s": 0.0, "r": 0.0}
        w_ia_low = {"ia": 0.1, "s": 0.0, "r": 0.0}

        d_high = compute_distance(a, b, w_ia_high, means)
        d_low = compute_distance(a, b, w_ia_low, means)
        self.assertGreater(d_high, d_low)

    def test_zero_weight_ignored(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 50.0, 2.0, 999.0)  # R differs a lot
        means = compute_sample_means([a, b])

        w_no_r = {"ia": 0.5, "s": 0.5, "r": 0.0}
        d = compute_distance(a, b, w_no_r, means)
        self.assertAlmostEqual(d, 0.0)

    def test_all_weights_zero(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 99.0, 9.0, 99.0)
        means = compute_sample_means([a, b])
        d = compute_distance(a, b, {"ia": 0, "s": 0, "r": 0}, means)
        self.assertAlmostEqual(d, 0.0)

    def test_pentode_weights_ignore_r(self):
        """With pentode defaults (R=0), R difference has no effect."""
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 50.0, 2.0, 999.0)  # huge R diff
        means = compute_sample_means([a, b])
        d = compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means)
        self.assertAlmostEqual(d, 0.0)

    def test_triode_weights_sensitive_to_r(self):
        """With triode defaults (R=0.3), R difference matters."""
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 50.0, 2.0, 20.0)  # R differs
        means = compute_sample_means([a, b])
        d_triode = compute_distance(a, b, DEFAULT_WEIGHTS_TRIODE, means)
        d_pentode = compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means)
        self.assertGreater(d_triode, 0.0)
        self.assertAlmostEqual(d_pentode, 0.0)

    def test_identical_tubes_zero_delta(self):
        """All identical tubes produce groups with delta=0."""
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.0, 2.0, 10.0),
            _rec("L3", 50.0, 2.0, 10.0),
            _rec("L4", 50.0, 2.0, 10.0),
        ]
        result = group_pairs(recs)
        for g in result.groups:
            self.assertAlmostEqual(g.delta, 0.0)

    def test_triangle_inequality(self):
        a = _rec("L1", 50.0, 2.0, 10.0)
        b = _rec("L2", 60.0, 2.5, 12.0)
        c = _rec("L3", 70.0, 3.0, 14.0)
        means = compute_sample_means([a, b, c])
        d_ab = compute_distance(a, b, DEFAULT_WEIGHTS_PENTODE, means)
        d_bc = compute_distance(b, c, DEFAULT_WEIGHTS_PENTODE, means)
        d_ac = compute_distance(a, c, DEFAULT_WEIGHTS_PENTODE, means)
        self.assertLessEqual(d_ac, d_ab + d_bc + 1e-9)


class TestComputeSampleMeans(unittest.TestCase):

    def test_single_record(self):
        recs = [_rec("L1", 50.0, 2.0, 10.0)]
        means = compute_sample_means(recs)
        self.assertAlmostEqual(means["ia"], 50.0)
        self.assertAlmostEqual(means["s"], 2.0)
        self.assertAlmostEqual(means["r"], 10.0)

    def test_multiple_records(self):
        recs = [_rec("L1", 40.0, 2.0, 8.0), _rec("L2", 60.0, 4.0, 12.0)]
        means = compute_sample_means(recs)
        self.assertAlmostEqual(means["ia"], 50.0)
        self.assertAlmostEqual(means["s"], 3.0)
        self.assertAlmostEqual(means["r"], 10.0)

    def test_empty_returns_ones(self):
        means = compute_sample_means([])
        for k in ("ia", "s", "r"):
            self.assertAlmostEqual(means[k], 1.0)


class TestFindSimilar(unittest.TestCase):

    def test_ordering_closest_first(self):
        anchor = _rec("L1", 50.0, 2.0, 10.0)
        close = _rec("L2", 51.0, 2.1, 10.1)
        far = _rec("L3", 60.0, 3.0, 15.0)

        result = find_similar(anchor, [anchor, close, far])
        self.assertEqual(result.mode, "similar")
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(result.groups[0].records[0].lamp_id, "L2")
        self.assertEqual(result.groups[1].records[0].lamp_id, "L3")
        self.assertLessEqual(result.groups[0].delta, result.groups[1].delta)

    def test_max_delta_filter(self):
        anchor = _rec("L1", 50.0, 2.0, 10.0)
        close = _rec("L2", 50.1, 2.01, 10.01)
        far = _rec("L3", 100.0, 5.0, 30.0)

        result = find_similar(anchor, [close, far], max_delta=5.0)
        # Far tube should be excluded
        lamp_ids = [g.records[0].lamp_id for g in result.groups]
        self.assertIn("L2", lamp_ids)
        self.assertNotIn("L3", lamp_ids)

    def test_anchor_excluded_from_results(self):
        anchor = _rec("L1", 50.0, 2.0, 10.0)
        other = _rec("L2", 55.0, 2.5, 12.0)

        result = find_similar(anchor, [anchor, other])
        lamp_ids = [g.records[0].lamp_id for g in result.groups]
        self.assertNotIn("L1", lamp_ids)

    def test_empty_records(self):
        anchor = _rec("L1", 50.0, 2.0, 10.0)
        result = find_similar(anchor, [])
        self.assertEqual(len(result.groups), 0)

    def test_anchor_set_in_result(self):
        anchor = _rec("L1", 50.0, 2.0, 10.0)
        other = _rec("L2", 55.0, 2.5, 12.0)
        result = find_similar(anchor, [other])
        self.assertIs(result.anchor, anchor)


class TestGroupPairs(unittest.TestCase):

    def test_four_tubes_two_pairs(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.5, 2.05, 10.1),
            _rec("L3", 70.0, 3.0, 15.0),
            _rec("L4", 70.5, 3.05, 15.1),
        ]
        result = group_pairs(recs)
        self.assertEqual(result.mode, "groups")
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(len(result.unmatched), 0)

        # Each group has 2 records
        for g in result.groups:
            self.assertEqual(len(g.records), 2)

        # Similar tubes should be paired together
        g1_ids = {r.lamp_id for r in result.groups[0].records}
        g2_ids = {r.lamp_id for r in result.groups[1].records}
        # L1+L2 and L3+L4 (or vice versa)
        self.assertTrue(
            (g1_ids == {"L1", "L2"} and g2_ids == {"L3", "L4"}) or
            (g1_ids == {"L3", "L4"} and g2_ids == {"L1", "L2"})
        )

    def test_odd_number_one_unmatched(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.5, 2.05, 10.1),
            _rec("L3", 70.0, 3.0, 15.0),
        ]
        result = group_pairs(recs)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(len(result.unmatched), 1)

    def test_single_tube_unmatched(self):
        recs = [_rec("L1", 50.0, 2.0, 10.0)]
        result = group_pairs(recs)
        self.assertEqual(len(result.groups), 0)
        self.assertEqual(len(result.unmatched), 1)

    def test_empty(self):
        result = group_pairs([])
        self.assertEqual(len(result.groups), 0)
        self.assertEqual(len(result.unmatched), 0)

    def test_max_delta_moves_to_unmatched(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.1, 2.01, 10.01),  # very close pair
            _rec("L3", 50.0, 2.0, 10.0),
            _rec("L4", 90.0, 5.0, 30.0),    # very far pair
        ]
        result = group_pairs(recs, max_delta=5.0)
        # Close pair should be in groups, far pair may be unmatched
        self.assertGreaterEqual(len(result.unmatched), 1)

    def test_sorted_by_delta(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 70.0, 3.0, 15.0),
            _rec("L3", 50.5, 2.05, 10.1),
            _rec("L4", 70.5, 3.05, 15.1),
        ]
        result = group_pairs(recs)
        deltas = [g.delta for g in result.groups]
        self.assertEqual(deltas, sorted(deltas))

    def test_two_tubes_one_pair(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 51.0, 2.1, 10.2),
        ]
        result = group_pairs(recs)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(len(result.unmatched), 0)
        ids = {r.lamp_id for r in result.groups[0].records}
        self.assertEqual(ids, {"L1", "L2"})

    def test_groups_renumbered_after_max_delta(self):
        """Groups exceeding max_delta move to unmatched, remaining renumbered."""
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.1, 2.01, 10.01),
            _rec("L3", 50.0, 2.0, 10.0),
            _rec("L4", 99.0, 9.0, 99.0),   # very far
        ]
        result = group_pairs(recs, max_delta=5.0)
        # At least one group should survive
        if result.groups:
            self.assertEqual(result.groups[0].number, 1)


class TestGroupN(unittest.TestCase):

    def test_eight_tubes_two_quads(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.5, 2.05, 10.1),
            _rec("L3", 51.0, 2.1, 10.2),
            _rec("L4", 51.5, 2.15, 10.3),
            _rec("L5", 70.0, 3.0, 15.0),
            _rec("L6", 70.5, 3.05, 15.1),
            _rec("L7", 71.0, 3.1, 15.2),
            _rec("L8", 71.5, 3.15, 15.3),
        ]
        result = group_n(recs, group_size=4)
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(len(result.unmatched), 0)
        for g in result.groups:
            self.assertEqual(len(g.records), 4)

    def test_remainder_unmatched(self):
        recs = [_rec(f"L{i}", 50.0 + i, 2.0, 10.0) for i in range(6)]
        result = group_n(recs, group_size=4)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(len(result.unmatched), 2)

    def test_too_few_for_group(self):
        recs = [_rec("L1", 50.0, 2.0, 10.0), _rec("L2", 55.0, 2.5, 12.0)]
        result = group_n(recs, group_size=4)
        self.assertEqual(len(result.groups), 0)
        self.assertEqual(len(result.unmatched), 2)

    def test_max_delta_filters_groups(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.1, 2.01, 10.01),
            _rec("L3", 50.2, 2.02, 10.02),
            _rec("L4", 50.3, 2.03, 10.03),  # tight quad
            _rec("L5", 10.0, 0.5, 2.0),
            _rec("L6", 90.0, 5.0, 30.0),
            _rec("L7", 11.0, 0.6, 2.5),
            _rec("L8", 91.0, 5.1, 31.0),    # loose quad
        ]
        result_no_limit = group_n(recs, group_size=4)
        self.assertEqual(len(result_no_limit.groups), 2)

        result_strict = group_n(recs, group_size=4, max_delta=2.0)
        # Tight quad should survive, loose should go to unmatched
        self.assertGreaterEqual(len(result_strict.unmatched), 4)


class TestSelectMeasurements(unittest.TestCase):

    def test_latest_per_lamp(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ts="2026-01-01T10:00:00", index=90.0),
            _entry("L1", 48.0, 1.9, 9.5, ts="2026-01-02T10:00:00", index=88.0),
            _entry("L2", 55.0, 2.5, 12.0, ts="2026-01-01T10:00:00"),
        ]
        recs = select_measurements(entries, use="latest")
        self.assertEqual(len(recs), 2)
        l1 = [r for r in recs if r.lamp_id == "L1"][0]
        self.assertEqual(l1.timestamp, "2026-01-02T10:00:00")
        self.assertAlmostEqual(l1.ia, 48.0)

    def test_best_per_lamp(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ts="2026-01-01T10:00:00", index=90.0),
            _entry("L1", 48.0, 1.9, 9.5, ts="2026-01-02T10:00:00", index=95.0),
        ]
        recs = select_measurements(entries, use="best")
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].index, 95.0)

    def test_conditions_filter(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ua=250.0, ug1=-2.5, ug2=150.0),
            _entry("L2", 55.0, 2.5, 12.0, ua=200.0, ug1=-3.0, ug2=100.0),
        ]
        recs = select_measurements(entries, conditions=(250.0, -2.5, 150.0, "pentode", False))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].lamp_id, "L1")

    def test_combined_anode(self):
        entries = [
            _entry("L1", 1.0, 2.0, 60.0, an=1),
            _entry("L1", 1.2, 2.4, 50.0, an=2),
        ]
        recs = select_measurements(entries, anode="combined")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].an, 0)  # combined marker
        self.assertAlmostEqual(recs[0].ia, 1.1)
        self.assertAlmostEqual(recs[0].s, 2.2)
        self.assertAlmostEqual(recs[0].r, 55.0)

    def test_empty_entries(self):
        recs = select_measurements([])
        self.assertEqual(len(recs), 0)

    def test_missing_fields_skipped(self):
        entries = [
            {"lamp_id": "L1", "conditions": {}, "srk": {}},  # no ia/s/r
            _entry("L2", 55.0, 2.5, 12.0),
        ]
        recs = select_measurements(entries)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].lamp_id, "L2")

    def test_combined_anode_single_anode_passthrough(self):
        """Combined mode with only one anode per lamp returns it as-is."""
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, an=1),
            _entry("L2", 55.0, 2.5, 12.0, an=1),
        ]
        recs = select_measurements(entries, anode="combined")
        self.assertEqual(len(recs), 2)
        # No averaging — single anode kept
        l1 = [r for r in recs if r.lamp_id == "L1"][0]
        self.assertAlmostEqual(l1.ia, 50.0)

    def test_multiple_measurements_per_lamp_selects_one(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ts="2026-01-01T10:00:00"),
            _entry("L1", 48.0, 1.9, 9.5, ts="2026-01-02T10:00:00"),
            _entry("L1", 52.0, 2.1, 10.5, ts="2026-01-03T10:00:00"),
        ]
        recs = select_measurements(entries, use="latest")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].timestamp, "2026-01-03T10:00:00")

    def test_different_ug2_mode_not_mixed(self):
        """Pentode and triode-connected measurements must not be compared."""
        entries = [
            _entry("L1", 50.0, 11.0, 40.0, ua=250, ug2=250, ug2_mode=TOPOLOGY_PENTODE),
            _entry("L2", 52.0, 11.5, 38.0, ua=250, ug2=250, ug2_mode=TOPOLOGY_PENTODE),
            _entry("L3", 30.0, 5.0, 8.0, ua=250, ug2=250, ug2_mode=TOPOLOGY_TRIODE_CONNECTED),
        ]
        # Auto-conditions from entries[0] = pentode → L3 excluded
        recs = select_measurements(
            entries, conditions=(250.0, -2.5, 250.0, "pentode", False))
        lamp_ids = {r.lamp_id for r in recs}
        self.assertIn("L1", lamp_ids)
        self.assertIn("L2", lamp_ids)
        self.assertNotIn("L3", lamp_ids)


class TestExtractRecord(unittest.TestCase):

    def test_valid_entry(self):
        e = _entry("L1", 50.0, 2.0, 10.0, index=95.0)
        rec = _extract_record(e)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.lamp_id, "L1")
        self.assertAlmostEqual(rec.ia, 50.0)
        self.assertAlmostEqual(rec.s, 2.0)
        self.assertAlmostEqual(rec.r, 10.0)
        self.assertAlmostEqual(rec.index, 95.0)

    def test_no_lamp_id(self):
        e = _entry("L1", 50.0, 2.0, 10.0)
        e["lamp_id"] = ""
        self.assertIsNone(_extract_record(e))

    def test_missing_srk(self):
        e = {"lamp_id": "L1", "timestamp": "x", "conditions": {"an": 1},
             "srk": {}, "health": {}}
        self.assertIsNone(_extract_record(e))

    def test_fallback_srk_ia_op(self):
        """When health.raw.ia_op is missing, falls back to srk.ia_op."""
        e = {
            "lamp_id": "L1", "timestamp": "x",
            "conditions": {"an": 1},
            "srk": {"s": 2.0, "r": 10.0, "ia_op": 42.0},
            "health": {},  # no raw.ia_op
        }
        rec = _extract_record(e)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.ia, 42.0)

    def test_negative_values_rejected(self):
        e = _entry("L1", -1.0, 2.0, 10.0)
        self.assertIsNone(_extract_record(e))

    def test_zero_ia_accepted(self):
        """Zero Ia is valid (tube at cutoff)."""
        e = _entry("L1", 0.0, 2.0, 10.0)
        rec = _extract_record(e)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.ia, 0.0)

    def test_no_conditions_defaults_an1(self):
        e = _entry("L1", 50.0, 2.0, 10.0)
        del e["conditions"]
        e["conditions"] = {}
        rec = _extract_record(e)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.an, 1)


class TestConditionsKey(unittest.TestCase):

    def test_rounding(self):
        e = {"conditions": {"ua": 250.04, "ug1": -2.549, "ug2": 150.05, "ug2_mode": TOPOLOGY_PENTODE}}
        self.assertEqual(_conditions_key(e), (250.0, -2.5, 150.1, "pentode", False))

    def test_mode_included(self):
        """Same voltages but different ug2_mode produce different keys."""
        e_pent = {"conditions": {"ua": 250, "ug1": -7, "ug2": 250, "ug2_mode": TOPOLOGY_PENTODE}}
        e_tric = {"conditions": {"ua": 250, "ug1": -7, "ug2": 250, "ug2_mode": TOPOLOGY_TRIODE_CONNECTED}}
        self.assertNotEqual(_conditions_key(e_pent), _conditions_key(e_tric))

    def test_missing_mode_defaults_pentode(self):
        e = {"conditions": {"ua": 250, "ug1": -7, "ug2": 250}}
        key = _conditions_key(e)
        self.assertEqual(key[3], "pentode")


class TestGreedyPairAssign(unittest.TestCase):

    def test_simple_2x2(self):
        cost = [[0, 1], [1, 0]]
        pairs = _greedy_pair_assign(cost)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (0, 1))

    def test_4x4_closest(self):
        # 0-1 close, 2-3 close
        cost = [
            [0,   1, 100, 100],
            [1,   0, 100, 100],
            [100, 100, 0,   2],
            [100, 100, 2,   0],
        ]
        pairs = _greedy_pair_assign(cost)
        self.assertEqual(len(pairs), 2)
        pair_sets = [frozenset(p) for p in pairs]
        self.assertIn(frozenset({0, 1}), pair_sets)
        self.assertIn(frozenset({2, 3}), pair_sets)


class TestMatchTubes(unittest.TestCase):

    def test_groups_mode(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0),
            _entry("L2", 50.5, 2.05, 10.1),
            _entry("L3", 70.0, 3.0, 15.0),
            _entry("L4", 70.5, 3.05, 15.1),
        ]
        result = match_tubes(entries, mode="groups", group_size=2)
        self.assertEqual(result.mode, "groups")
        self.assertEqual(len(result.groups), 2)

    def test_similar_mode(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0),
            _entry("L2", 55.0, 2.5, 12.0),
            _entry("L3", 51.0, 2.1, 10.2),
        ]
        result = match_tubes(entries, mode="similar", anchor_lamp_id="L1")
        self.assertEqual(result.mode, "similar")
        self.assertIsNotNone(result.anchor)
        self.assertEqual(result.anchor.lamp_id, "L1")
        self.assertGreater(len(result.groups), 0)

    def test_similar_anchor_timestamp_picks_specific_measurement(self):
        """'Find similar (this measurement)' must anchor on the chosen
        timestamp, not the lamp's latest. L1 has 3 measurements; anchoring on
        the earliest must use its Ia, while the default uses the latest."""
        entries = [
            _entry("L1", 40.0, 1.8, 9.0, ts="2026-01-01T10:00:00"),
            _entry("L1", 50.0, 2.0, 10.0, ts="2026-02-01T10:00:00"),
            _entry("L1", 60.0, 2.2, 11.0, ts="2026-03-01T10:00:00"),  # latest
            _entry("L2", 55.0, 2.5, 12.0),
        ]
        # Anchor on the EARLIEST specific measurement.
        res = match_tubes(entries, mode="similar", anchor_lamp_id="L1",
                          anchor_timestamp="2026-01-01T10:00:00")
        self.assertEqual(res.anchor.timestamp, "2026-01-01T10:00:00")
        self.assertEqual(res.anchor.ia, 40.0)
        # Default (no timestamp) anchors on the latest.
        res_latest = match_tubes(entries, mode="similar", anchor_lamp_id="L1")
        self.assertEqual(res_latest.anchor.timestamp, "2026-03-01T10:00:00")
        self.assertEqual(res_latest.anchor.ia, 60.0)
        # Absent timestamp → graceful fallback to latest/best.
        res_missing = match_tubes(entries, mode="similar", anchor_lamp_id="L1",
                                  anchor_timestamp="2099-not-present")
        self.assertEqual(res_missing.anchor.ia, 60.0)

    def test_similar_this_measurement_filters_to_anchor_conditions(self):
        """Anchoring on a specific measurement must rank candidates at THAT
        measurement's operating point, not the (possibly different) bulk
        conditions — otherwise distances are skewed."""
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ua=250.0, ts="2026-01-01T00:00:00"),
            _entry("L2", 52.0, 2.1, 10.2, ua=250.0),   # same op-point as anchor
            _entry("L3", 90.0, 3.0, 5.0, ua=300.0),     # different op-point
        ]
        # Pass L3's conditions as the bulk filter; the fix must override with
        # the anchor's (ua=250), so L3 is excluded and L2 is the candidate.
        res = match_tubes(
            entries, mode="similar", anchor_lamp_id="L1",
            anchor_timestamp="2026-01-01T00:00:00",
            conditions=(300.0, -2.5, 150.0, "pentode", False))  # L3's conditions
        ranked = {g.records[0].lamp_id for g in res.groups}
        self.assertIn("L2", ranked)
        self.assertNotIn("L3", ranked)

    def test_auto_conditions(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ua=250.0, ug1=-2.5, ug2=150.0,
                   ts="2026-01-02T00:00:00"),
            _entry("L2", 55.0, 2.5, 12.0, ua=200.0, ug1=-3.0, ug2=100.0,
                   ts="2026-01-01T00:00:00"),
        ]
        # entries[0] is latest, conditions auto-detected from it
        result = match_tubes(entries, mode="groups")
        # L2 has different conditions, should be excluded
        total_tubes = sum(len(g.records) for g in result.groups) + len(result.unmatched)
        self.assertEqual(total_tubes, 1)  # only L1 matches conditions

    def test_empty(self):
        result = match_tubes([], mode="groups")
        self.assertEqual(len(result.groups), 0)

    def test_auto_conditions_includes_ug2_mode(self):
        """Auto-detect must pick ug2_mode from latest entry."""
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ts="2026-01-02T00:00:00",
                   ug2_mode=TOPOLOGY_PENTODE),
            _entry("L2", 52.0, 2.1, 10.2, ts="2026-01-01T00:00:00",
                   ug2_mode=TOPOLOGY_PENTODE),
            _entry("L3", 30.0, 5.0, 8.0, ts="2026-01-01T00:00:00",
                   ug2_mode=TOPOLOGY_TRIODE_CONNECTED),
        ]
        result = match_tubes(entries, mode="groups")
        # L3 is triode_connected, should be excluded (auto = pentode from L1)
        all_ids = set()
        for g in result.groups:
            for r in g.records:
                all_ids.add(r.lamp_id)
        for r in result.unmatched:
            all_ids.add(r.lamp_id)
        self.assertNotIn("L3", all_ids)

    def test_mixed_modes_end_to_end(self):
        """Only entries matching the dominant mode are grouped."""
        entries = [
            _entry("L1", 50.0, 11.0, 40.0, ug2_mode=TOPOLOGY_PENTODE),
            _entry("L2", 51.0, 11.2, 39.0, ug2_mode=TOPOLOGY_PENTODE),
            _entry("L3", 52.0, 11.5, 38.0, ug2_mode=TOPOLOGY_PENTODE),
            _entry("L4", 30.0, 5.0, 8.0, ug2_mode=TOPOLOGY_TRIODE_CONNECTED),
            _entry("L5", 31.0, 5.2, 8.5, ug2_mode=TOPOLOGY_TRIODE_CONNECTED),
        ]
        result = match_tubes(entries, mode="groups", group_size=2)
        all_ids = set()
        for g in result.groups:
            for r in g.records:
                all_ids.add(r.lamp_id)
        for r in result.unmatched:
            all_ids.add(r.lamp_id)
        # Only pentode entries (auto from entries[0])
        self.assertTrue(all_ids <= {"L1", "L2", "L3"})
        # Triode entries excluded
        self.assertNotIn("L4", all_ids)
        self.assertNotIn("L5", all_ids)

    def test_explicit_conditions(self):
        entries = [
            _entry("L1", 50.0, 2.0, 10.0, ua=250.0, ug1=-2.5, ug2=150.0),
            _entry("L2", 55.0, 2.5, 12.0, ua=250.0, ug1=-2.5, ug2=150.0),
            _entry("L3", 60.0, 3.0, 14.0, ua=200.0, ug1=-3.0, ug2=100.0),
        ]
        result = match_tubes(entries, mode="groups",
                             conditions=(250.0, -2.5, 150.0, "pentode", False))
        total = sum(len(g.records) for g in result.groups) + len(result.unmatched)
        self.assertEqual(total, 2)  # L1 + L2 only

    def test_similar_anchor_not_found_is_a_visible_error(self):
        # The old behaviour silently ranked around records[0] — the user
        # clicked lamp X and got "similar to Y" with no warning. A missing
        # anchor must be an explicit error, never a substituted lamp.
        entries = [
            _entry("L1", 50.0, 2.0, 10.0),
            _entry("L2", 55.0, 2.5, 12.0),
        ]
        result = match_tubes(entries, mode="similar",
                             anchor_lamp_id="NONEXISTENT")
        self.assertIsNone(result.anchor)
        self.assertEqual(result.groups, [])
        self.assertEqual(result.anchor_error, ANCHOR_ERR_NOT_FOUND)

    def test_similar_without_anchor_request_keeps_first_record(self):
        # Programmatic similar with NO anchor request still ranks around
        # the pool's first record — nothing was promised to the user.
        entries = [
            _entry("L1", 50.0, 2.0, 10.0),
            _entry("L2", 55.0, 2.5, 12.0),
        ]
        result = match_tubes(entries, mode="similar")
        self.assertIsNotNone(result.anchor)
        self.assertIsNone(result.anchor_error)

    def test_quads_via_match_tubes(self):
        entries = [_entry(f"L{i}", 50.0 + i * 0.5, 2.0 + i * 0.05, 10.0 + i * 0.1)
                   for i in range(8)]
        result = match_tubes(entries, mode="groups", group_size=4)
        self.assertEqual(len(result.groups), 2)
        for g in result.groups:
            self.assertEqual(len(g.records), 4)


class TestHungarianGreedyConsistency(unittest.TestCase):
    """Hungarian and greedy should agree on simple cases."""

    def test_obvious_pairs_same_result(self):
        recs = [
            _rec("L1", 50.0, 2.0, 10.0),
            _rec("L2", 50.1, 2.01, 10.01),
            _rec("L3", 80.0, 4.0, 20.0),
            _rec("L4", 80.1, 4.01, 20.01),
        ]
        means = compute_sample_means(recs)
        from lm19.tube_matching import _build_distance_matrix, _greedy_pair_assign
        dist = _build_distance_matrix(recs, DEFAULT_WEIGHTS_PENTODE, means)
        greedy_pairs = _greedy_pair_assign(dist)
        greedy_sets = {frozenset(p) for p in greedy_pairs}
        self.assertIn(frozenset({0, 1}), greedy_sets)
        self.assertIn(frozenset({2, 3}), greedy_sets)


class TestPairAlgorithmChoice(unittest.TestCase):
    """Pair-matching algorithm dispatch — greedy (default) vs optimal."""

    def test_default_is_greedy(self):
        from lm19.tube_matching import DEFAULT_PAIR_ALGORITHM, PAIR_ALGORITHM_GREEDY
        self.assertEqual(DEFAULT_PAIR_ALGORITHM, PAIR_ALGORITHM_GREEDY)

    def test_known_constants(self):
        from lm19.tube_matching import (
            PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL, PAIR_ALGORITHMS,
        )
        self.assertEqual(PAIR_ALGORITHM_GREEDY, "greedy")
        self.assertEqual(PAIR_ALGORITHM_OPTIMAL, "optimal")
        self.assertIn(PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHMS)
        self.assertIn(PAIR_ALGORITHM_OPTIMAL, PAIR_ALGORITHMS)

    def test_invalid_algorithm_raises(self):
        from lm19.tube_matching import _group_pairs_from_matrix
        dist = [[0, 1], [1, 0]]
        with self.assertRaises(ValueError):
            _group_pairs_from_matrix(dist, 2, max_delta=0, algorithm="nonsense")

    def test_algorithms_diverge_on_constructed_case(self):
        """Constructed 6-node case where greedy and optimal pick different pairs.

        Distances chosen so greedy locks (L3,L4)=2 early, then L5/L6 must
        accept their bad pair (50); optimal sees the split (L3,L5)+(L4,L6)
        with sum 6, much better than greedy's 1+2+50=53.
        """
        from lm19.tube_matching import (
            _group_pairs_from_matrix,
            PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL,
        )
        dist = [
            [0,  1, 50, 50, 50, 50],
            [1,  0, 50, 50, 50, 50],
            [50, 50, 0,  2,  3, 50],
            [50, 50, 2,  0, 50,  3],
            [50, 50, 3, 50,  0, 50],
            [50, 50, 50, 3, 50,  0],
        ]
        greedy, _ = _group_pairs_from_matrix(dist, 6, max_delta=0,
                                              algorithm=PAIR_ALGORITHM_GREEDY)
        optimal, _ = _group_pairs_from_matrix(dist, 6, max_delta=0,
                                               algorithm=PAIR_ALGORITHM_OPTIMAL)
        greedy_sets = {frozenset(p) for p, _d in greedy}
        optimal_sets = {frozenset(p) for p, _d in optimal}
        # Greedy: (L1,L2), (L3,L4) — tight pair forces bad final pair (L5,L6)
        self.assertIn(frozenset({2, 3}), greedy_sets)
        self.assertIn(frozenset({4, 5}), greedy_sets)
        # Optimal: splits (L3,L5) and (L4,L6) for better total sum
        self.assertIn(frozenset({2, 4}), optimal_sets)
        self.assertIn(frozenset({3, 5}), optimal_sets)

    def test_greedy_max_delta_drops_bad_pairs(self):
        """With max_delta, greedy stops early — leaves bad pairs unmatched.

        Key property for the "box of tubes" workflow: only tight pairs go
        into groups, mediocre lamps return to unmatched.
        """
        from lm19.tube_matching import (
            _group_pairs_from_matrix, PAIR_ALGORITHM_GREEDY,
        )
        dist = [
            [0,  1, 50, 50, 50, 50],
            [1,  0, 50, 50, 50, 50],
            [50, 50, 0,  2,  3, 50],
            [50, 50, 2,  0, 50,  3],
            [50, 50, 3, 50,  0, 50],
            [50, 50, 50, 3, 50,  0],
        ]
        groups, unmatched = _group_pairs_from_matrix(
            dist, 6, max_delta=5, algorithm=PAIR_ALGORITHM_GREEDY)
        # 2 tight pairs picked, L5/L6 (indices 4, 5) returned to supplier
        self.assertEqual(len(groups), 2)
        self.assertSetEqual(set(unmatched), {4, 5})

    def test_match_tubes_passes_algorithm(self):
        """match_tubes accepts algorithm and forwards to group_pairs."""
        from lm19.tube_matching import match_tubes
        entries = [
            _entry("L1", ia=50.0, s=2.0, r=10.0),
            _entry("L2", ia=50.1, s=2.01, r=10.01),
            _entry("L3", ia=80.0, s=4.0, r=20.0),
            _entry("L4", ia=80.1, s=4.01, r=20.01),
        ]
        for algo in ("greedy", "optimal"):
            result = match_tubes(entries, mode="groups", group_size=2,
                                  algorithm=algo)
            self.assertEqual(result.mode, "groups")
            self.assertEqual(len(result.groups), 2)

    def test_odd_count_leaves_one_unmatched(self):
        """Both algorithms handle odd N: take best pair, leave one out."""
        from lm19.tube_matching import (
            _group_pairs_from_matrix,
            PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL,
        )
        dist = [
            [0,  1, 10],
            [1,  0, 10],
            [10, 10, 0],
        ]
        for algo in (PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL):
            groups, unmatched = _group_pairs_from_matrix(
                dist, 3, max_delta=0, algorithm=algo)
            self.assertEqual(len(groups), 1, f"{algo}: expected 1 pair")
            self.assertEqual(unmatched, [2], f"{algo}: index 2 should be unmatched")

    def test_inf_distance_excluded(self):
        """Incomparable lamps (inf distance) never get paired together,
        even when no other choice exists."""
        from lm19.tube_matching import (
            _group_pairs_from_matrix,
            PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL,
        )
        INF = math.inf
        # Two clusters of comparable lamps, no cross-cluster comparability
        dist = [
            [0, 1, INF, INF],
            [1, 0, INF, INF],
            [INF, INF, 0, 2],
            [INF, INF, 2, 0],
        ]
        for algo in (PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL):
            groups, unmatched = _group_pairs_from_matrix(
                dist, 4, max_delta=0, algorithm=algo)
            pair_sets = {frozenset(p) for p, _ in groups}
            self.assertIn(frozenset({0, 1}), pair_sets, f"{algo}: 0,1 must pair")
            self.assertIn(frozenset({2, 3}), pair_sets, f"{algo}: 2,3 must pair")
            self.assertEqual(unmatched, [], f"{algo}: all 4 paired")

    def test_all_inf_returns_empty(self):
        """All-inf matrix → no pairs, all lamps unmatched."""
        from lm19.tube_matching import (
            _group_pairs_from_matrix,
            PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL,
        )
        INF = math.inf
        dist = [[0, INF], [INF, 0]]
        for algo in (PAIR_ALGORITHM_GREEDY, PAIR_ALGORITHM_OPTIMAL):
            groups, unmatched = _group_pairs_from_matrix(
                dist, 2, max_delta=0, algorithm=algo)
            self.assertEqual(groups, [], f"{algo}: no pairs from inf")
            self.assertEqual(sorted(unmatched), [0, 1], f"{algo}: both unmatched")

    def test_max_delta_zero_means_no_limit(self):
        """max_delta=0 is the "no limit" sentinel — even huge distances pair."""
        from lm19.tube_matching import (
            _group_pairs_from_matrix, PAIR_ALGORITHM_GREEDY,
        )
        dist = [[0, 1000], [1000, 0]]
        groups, unmatched = _group_pairs_from_matrix(
            dist, 2, max_delta=0, algorithm=PAIR_ALGORITHM_GREEDY)
        self.assertEqual(len(groups), 1)
        self.assertEqual(unmatched, [])

    def test_greedy_early_stop_when_max_delta_exceeded(self):
        """Greedy stops iterating once a pair exceeds max_delta —
        subsequent edges (sorted) are all worse so checking them is wasted
        work. We verify behavior, not perf: anything below threshold is
        included, anything above is excluded."""
        from lm19.tube_matching import (
            _group_pairs_from_matrix, PAIR_ALGORITHM_GREEDY,
        )
        # Three perfect-fit pairs (δ=1) + one mediocre pair (δ=8) above threshold
        dist = [
            [0,  1, 50, 50, 50, 50, 50, 50],
            [1,  0, 50, 50, 50, 50, 50, 50],
            [50, 50, 0,  1, 50, 50, 50, 50],
            [50, 50, 1,  0, 50, 50, 50, 50],
            [50, 50, 50, 50, 0,  1, 50, 50],
            [50, 50, 50, 50, 1,  0, 50, 50],
            [50, 50, 50, 50, 50, 50, 0,  8],
            [50, 50, 50, 50, 50, 50, 8,  0],
        ]
        groups, unmatched = _group_pairs_from_matrix(
            dist, 8, max_delta=5, algorithm=PAIR_ALGORITHM_GREEDY)
        # 3 perfect pairs, the 8-pair excluded, last 2 lamps unmatched
        self.assertEqual(len(groups), 3)
        self.assertSetEqual(set(unmatched), {6, 7})


class TestCurveMatchingAlgorithm(unittest.TestCase):
    """Compare-tab curve matching must default to ``optimal`` so existing
    behavior is preserved when no algorithm is specified."""

    def test_match_curves_defaults_to_optimal(self):
        from lm19.tube_matching import match_curves
        # Inspect default parameter via signature inspection
        import inspect
        sig = inspect.signature(match_curves)
        default = sig.parameters["algorithm"].default
        self.assertEqual(default, "optimal")


class TestDefaultWeightsForMode(unittest.TestCase):

    def test_pentode(self):
        w = default_weights_for_mode("pentode")
        self.assertEqual(w, DEFAULT_WEIGHTS_PENTODE)

    def test_triode(self):
        w = default_weights_for_mode("triode")
        self.assertEqual(w, DEFAULT_WEIGHTS_TRIODE)
        self.assertGreater(w["r"], DEFAULT_WEIGHTS_PENTODE["r"])

    def test_triode_connected(self):
        w = default_weights_for_mode("triode_connected")
        self.assertEqual(w, DEFAULT_WEIGHTS_TRIODE)

    def test_unknown_defaults_pentode(self):
        w = default_weights_for_mode("something_else")
        self.assertEqual(w, DEFAULT_WEIGHTS_PENTODE)

    def test_returns_copy(self):
        """Mutating returned dict must not change the module default."""
        w = default_weights_for_mode("pentode")
        w["ia"] = 999.0
        self.assertAlmostEqual(DEFAULT_WEIGHTS_PENTODE["ia"], 0.5)

    def test_pentode_r_zero(self):
        """Pentode default has R weight = 0 (Rp irrelevant)."""
        w = default_weights_for_mode("pentode")
        self.assertAlmostEqual(w["r"], 0.0)

    def test_triode_r_significant(self):
        """Triode default has significant R weight (Rp determines µ)."""
        w = default_weights_for_mode("triode")
        self.assertGreater(w["r"], 0.2)


class TestDeltaQuality(unittest.TestCase):

    def test_excellent(self):
        self.assertEqual(delta_quality(0.0), "excellent")
        self.assertEqual(delta_quality(1.5), "excellent")
        self.assertEqual(delta_quality(DELTA_EXCELLENT), "excellent")

    def test_good(self):
        self.assertEqual(delta_quality(3.0), "good")
        self.assertEqual(delta_quality(DELTA_GOOD), "good")

    def test_fair(self):
        self.assertEqual(delta_quality(7.0), "fair")
        self.assertEqual(delta_quality(DELTA_FAIR), "fair")

    def test_poor(self):
        self.assertEqual(delta_quality(15.0), "poor")
        self.assertEqual(delta_quality(10.1), "poor")

    def test_class_ab_thresholds(self):
        exc, good, fair = CURVE_DELTA_THRESHOLDS["class_ab"]
        self.assertEqual(delta_quality(exc, "class_ab"), "excellent")
        self.assertEqual(delta_quality(exc + 0.1, "class_ab"), "good")
        self.assertEqual(delta_quality(good, "class_ab"), "good")
        self.assertEqual(delta_quality(good + 0.1, "class_ab"), "fair")
        self.assertEqual(delta_quality(fair, "class_ab"), "fair")
        self.assertEqual(delta_quality(fair + 0.1, "class_ab"), "poor")

    def test_class_a_stricter(self):
        """Class A has tighter thresholds than AB."""
        # 6% is "good" in AB but "poor" in A
        self.assertEqual(delta_quality(6.0, "class_ab"), "good")
        self.assertEqual(delta_quality(6.0, "class_a"), "fair")

    def test_class_b_more_relaxed(self):
        """Class B has relaxed thresholds."""
        # 12% is "fair" in AB but "good" in B
        self.assertEqual(delta_quality(12.0, "class_ab"), "fair")
        self.assertEqual(delta_quality(12.0, "class_b"), "good")

    def test_no_amp_class_uses_health_defaults(self):
        """Without amp_class, uses Health tab thresholds."""
        self.assertEqual(delta_quality(3.0, None), "good")
        self.assertEqual(delta_quality(3.0), "good")


# ── Curve matching: amp_class weighting ─────────────────────────────

class TestAmpClassWeighting(unittest.TestCase):
    """Tests for amp_class-dependent weighting in compute_matching."""

    def _make_curves(self, offset=0.0, ua_start=0, ua_stop=300, step=5, ug1=-5.0):
        """Two sets of points with a known Ia offset."""
        pts = []
        ua = ua_start
        while ua <= ua_stop:
            # Ia has a knee-like shape: low at small Ua, rising, then flattening
            ia = max(0.0, 20.0 * (1 - math.exp(-ua / 50.0))) + offset
            pts.append({"ua": float(ua), "ug1": ug1, "ia": ia,
                        "ug2": 0.0, "ig2": 0.0})
            ua += step
        return pts

    def test_class_a_weights_high_ia_more(self):
        """Class A should give better match than B for same data
        when difference is mainly at low Ia (knee)."""
        from lm19.quality import compute_matching, AMP_CLASS_A, AMP_CLASS_B
        # Two tubes identical except a small offset in knee region
        pts_a = self._make_curves(offset=0.0)
        pts_b = []
        for p in self._make_curves(offset=0.0):
            # Add difference only at low Ia (knee)
            ia = p["ia"]
            if ia < 5.0:
                ia += 2.0  # big relative diff at knee
            pts_b.append({**p, "ia": ia})

        r_a = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_A)
        r_b = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_B)
        self.assertIsNotNone(r_a)
        self.assertIsNotNone(r_b)
        # Class A should show better match (knee weighted less)
        self.assertGreater(r_a.match_pct, r_b.match_pct)

    def test_all_classes_agree_on_identical(self):
        """Identical curves should give ~100% match regardless of class."""
        from lm19.quality import compute_matching, AMP_CLASS_A, AMP_CLASS_AB, AMP_CLASS_B
        pts = self._make_curves()
        for cls in [AMP_CLASS_A, AMP_CLASS_AB, AMP_CLASS_B]:
            r = compute_matching(pts, pts, amp_class=cls)
            self.assertIsNotNone(r)
            self.assertAlmostEqual(r.match_pct, 100.0, places=1)

    def test_class_b_uniform_weight_matches_unweighted_rms(self):
        """class_b uses weights=ones, so the weighted RMS reduces to the
        plain RMS of deltas. Verifies the formula directly."""
        from lm19.quality import compute_matching, AMP_CLASS_B
        pts_a = self._make_curves(offset=0.0)
        pts_b = [
            {**p, "ia": p["ia"] + 1.0}  # uniform 1.0 mA shift everywhere
            for p in self._make_curves(offset=0.0)
        ]
        r = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_B)
        self.assertIsNotNone(r)
        # Uniform shift → every delta = 1.0 → RMS = 1.0
        self.assertAlmostEqual(r.rms_delta, 1.0, places=2)

    def test_default_amp_class_is_class_ab(self):
        """compute_matching() with no amp_class arg must use class_ab
        (sqrt weighting). Lock in the documented default."""
        from lm19.quality import compute_matching, AMP_CLASS_AB
        pts_a = self._make_curves(offset=0.0)
        pts_b = [{**p, "ia": p["ia"] + 0.5} for p in self._make_curves(offset=0.0)]
        r_default = compute_matching(pts_a, pts_b)
        r_ab = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_AB)
        self.assertIsNotNone(r_default)
        self.assertIsNotNone(r_ab)
        self.assertAlmostEqual(r_default.match_pct, r_ab.match_pct, places=4)
        self.assertAlmostEqual(r_default.rms_delta, r_ab.rms_delta, places=6)


# ── Curve matching: multi-Ug2 grouping ──────────────────────────────

class TestMultiUg2Matching(unittest.TestCase):
    """Tests for (Ug1, Ug2) grouping in compute_matching."""

    def test_multi_ug2_not_mixed(self):
        """Points at different Ug2 should not be mixed in one curve."""
        from lm19.quality import compute_matching
        # Two entries, each with 2 Ug2 levels
        pts_a = [
            {"ua": 100, "ug1": -5.0, "ug2": 100.0, "ia": 10.0},
            {"ua": 200, "ug1": -5.0, "ug2": 100.0, "ia": 20.0},
            {"ua": 100, "ug1": -5.0, "ug2": 200.0, "ia": 30.0},
            {"ua": 200, "ug1": -5.0, "ug2": 200.0, "ia": 50.0},
        ]
        pts_b = [
            {"ua": 100, "ug1": -5.0, "ug2": 100.0, "ia": 10.5},
            {"ua": 200, "ug1": -5.0, "ug2": 100.0, "ia": 20.5},
            {"ua": 100, "ug1": -5.0, "ug2": 200.0, "ia": 30.5},
            {"ua": 200, "ug1": -5.0, "ug2": 200.0, "ia": 50.5},
        ]
        r = compute_matching(pts_a, pts_b)
        self.assertIsNotNone(r)
        # Offset is only 0.5 mA — should be high match
        self.assertGreater(r.match_pct, 95.0)
        # Without Ug2 grouping, mixing Ug2=100 (Ia=10) with Ug2=200 (Ia=30)
        # would create huge deltas. This verifies they're separated.
        self.assertLess(r.max_delta, 1.0)


# ── Same lamp_id exclusion in curve distance matrix ─────────────────

class TestSameLampIdExclusion(unittest.TestCase):

    def test_same_lamp_id_inf_distance(self):
        """Entries with same lamp_id should have infinite distance."""
        entries = [
            {"lamp_id": "L1", "points": [
                {"ua": 100, "ug1": -5, "ug2": 0, "ia": 10},
                {"ua": 200, "ug1": -5, "ug2": 0, "ia": 20},
            ]},
            {"lamp_id": "L1", "points": [
                {"ua": 100, "ug1": -5, "ug2": 0, "ia": 10},
                {"ua": 200, "ug1": -5, "ug2": 0, "ia": 20},
            ]},
        ]
        dist, info = build_curve_distance_matrix(entries, min_overlap=2)
        self.assertTrue(math.isinf(dist[0][1]))

    def test_different_lamp_id_finite(self):
        """Entries with different lamp_id should have finite distance."""
        entries = [
            {"lamp_id": "L1", "points": [
                {"ua": 100, "ug1": -5, "ug2": 0, "ia": 10},
                {"ua": 200, "ug1": -5, "ug2": 0, "ia": 20},
            ]},
            {"lamp_id": "L2", "points": [
                {"ua": 100, "ug1": -5, "ug2": 0, "ia": 10.5},
                {"ua": 200, "ug1": -5, "ug2": 0, "ia": 20.5},
            ]},
        ]
        dist, info = build_curve_distance_matrix(entries, min_overlap=2)
        self.assertFalse(math.isinf(dist[0][1]))


# ── group_by_distance_matrix tests ──────────────────────────────────

class TestGroupByDistanceMatrix(unittest.TestCase):
    """Tests for the shared group_by_distance_matrix function."""

    def test_simple_pairs(self):
        """4 items, 2 obvious pairs."""
        # Items: 0≈1 (d=1), 2≈3 (d=2), 0-2 far (d=50)
        dist = [
            [0, 1, 50, 51],
            [1, 0, 51, 50],
            [50, 51, 0, 2],
            [51, 50, 2, 0],
        ]
        groups, unmatched = group_by_distance_matrix(
            dist, ["A", "B", "C", "D"], group_size=2)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(unmatched), 0)
        # Best pair first
        self.assertAlmostEqual(groups[0][1], 1.0)
        self.assertAlmostEqual(groups[1][1], 2.0)

    def test_inf_distances(self):
        """Pairs with inf distance → unmatched."""
        INF = float("inf")
        dist = [
            [0, 1, INF],
            [1, 0, INF],
            [INF, INF, 0],
        ]
        groups, unmatched = group_by_distance_matrix(
            dist, ["A", "B", "C"], group_size=2)
        self.assertEqual(len(groups), 1)
        self.assertIn(2, unmatched)  # C is unmatched

    def test_all_inf(self):
        """All distances inf → all unmatched."""
        INF = float("inf")
        dist = [[0, INF], [INF, 0]]
        groups, unmatched = group_by_distance_matrix(
            dist, ["A", "B"], group_size=2)
        self.assertEqual(len(groups), 0)
        self.assertEqual(len(unmatched), 2)

    def test_max_delta_filters(self):
        """Groups exceeding max_delta → unmatched."""
        dist = [
            [0, 1, 50, 51],
            [1, 0, 51, 50],
            [50, 51, 0, 20],
            [51, 50, 20, 0],
        ]
        groups, unmatched = group_by_distance_matrix(
            dist, ["A", "B", "C", "D"], group_size=2, max_delta=10.0)
        self.assertEqual(len(groups), 1)  # Only A-B pair (d=1)
        self.assertAlmostEqual(groups[0][1], 1.0)
        self.assertIn(2, unmatched)
        self.assertIn(3, unmatched)

    def test_quads(self):
        """Group size 4."""
        # 8 items, 2 groups of 4
        dist = [[0.0] * 8 for _ in range(8)]
        for i in range(8):
            for j in range(8):
                if i != j:
                    # First 4 close to each other, last 4 close
                    if (i < 4) == (j < 4):
                        dist[i][j] = 1.0
                    else:
                        dist[i][j] = 50.0
        labels = [f"T{i}" for i in range(8)]
        groups, unmatched = group_by_distance_matrix(
            dist, labels, group_size=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(unmatched), 0)

    def test_single_item(self):
        """One item → unmatched."""
        groups, unmatched = group_by_distance_matrix(
            [[0]], ["A"], group_size=2)
        self.assertEqual(len(groups), 0)
        self.assertEqual(unmatched, [0])


# ── Curve-based matching tests ──────────────────────────────────────

def _make_curve_entry(lamp_id, ia_offset=0.0, ua_start=100, ua_stop=300,
                      step=10, ug1=-5.0):
    """Create a synthetic compare entry with linear Ia(Ua) curve."""
    points = []
    ua = ua_start
    while ua <= ua_stop:
        ia = 0.1 * ua + ia_offset  # linear: Ia = 0.1*Ua + offset
        points.append({"ua": float(ua), "ug1": ug1, "ia": ia,
                       "ug2": 0.0, "ig2": 0.0})
        ua += step
    return {"lamp_id": lamp_id, "points": points}


class TestBuildCurveDistanceMatrix(unittest.TestCase):

    def test_identical_curves(self):
        """Identical curves → distance ≈ 0."""
        entries = [
            _make_curve_entry("A", ia_offset=0, ua_start=100, ua_stop=500, step=5),
            _make_curve_entry("B", ia_offset=0, ua_start=100, ua_stop=500, step=5),
        ]
        dist, info = build_curve_distance_matrix(entries)
        self.assertAlmostEqual(dist[0][1], 0.0, places=1)
        self.assertGreater(info[(0, 1)].n_points, 0)
        self.assertFalse(info[(0, 1)].low_overlap)

    def test_different_curves(self):
        """Different curves → distance > 0."""
        entries = [
            _make_curve_entry("A", ia_offset=0),
            _make_curve_entry("B", ia_offset=5.0),  # 5mA offset
        ]
        dist, info = build_curve_distance_matrix(entries)
        self.assertGreater(dist[0][1], 0.0)
        self.assertLess(dist[0][1], 100.0)

    def test_no_overlap(self):
        """Disjoint Ua ranges → inf distance."""
        entries = [
            _make_curve_entry("A", ua_start=100, ua_stop=200),
            _make_curve_entry("B", ua_start=300, ua_stop=400),
        ]
        dist, info = build_curve_distance_matrix(entries)
        self.assertTrue(math.isinf(dist[0][1]))

    def test_partial_overlap_below_threshold(self):
        """Partial overlap with too few points → inf."""
        # Only 3 common points
        entries = [
            _make_curve_entry("A", ua_start=100, ua_stop=300, step=10),
            _make_curve_entry("B", ua_start=280, ua_stop=400, step=10),
        ]
        dist, info = build_curve_distance_matrix(entries, min_overlap=MIN_OVERLAP_POINTS)
        # Overlap is only ~3 points (280, 290, 300)
        self.assertTrue(math.isinf(dist[0][1]))
        self.assertTrue(info[(0, 1)].low_overlap)

    def test_empty_points(self):
        """Entry with no points → inf."""
        entries = [
            _make_curve_entry("A"),
            {"lamp_id": "B", "points": []},
        ]
        dist, info = build_curve_distance_matrix(entries)
        self.assertTrue(math.isinf(dist[0][1]))

    def test_warn_overlap(self):
        """Overlap above min but below warn → low_overlap=True, finite dist."""
        # Create entries with ~15 overlap points (above 10, below 30)
        entries = [
            _make_curve_entry("A", ua_start=100, ua_stop=300, step=10),
            _make_curve_entry("B", ua_start=200, ua_stop=400, step=10),
        ]
        dist, info = build_curve_distance_matrix(entries, min_overlap=5)
        self.assertFalse(math.isinf(dist[0][1]))
        # n_points should be moderate (interpolated grid from 200-300)
        self.assertGreater(info[(0, 1)].n_points, 0)


class TestMatchCurves(unittest.TestCase):

    def test_groups_mode(self):
        """4 entries → 2 pairs."""
        entries = [
            _make_curve_entry("A", ia_offset=0),
            _make_curve_entry("B", ia_offset=0.1),  # very close to A
            _make_curve_entry("C", ia_offset=10),
            _make_curve_entry("D", ia_offset=10.1),  # very close to C
        ]
        labels = ["A", "B", "C", "D"]
        result = match_curves(entries, labels, mode="groups", group_size=2)
        self.assertEqual(result.mode, "groups")
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(len(result.unmatched), 0)
        # Best pair (A+B) should be first
        self.assertLess(result.groups[0].delta, result.groups[1].delta)

    def test_similar_mode(self):
        """Rank by distance from anchor."""
        entries = [
            _make_curve_entry("A", ia_offset=0),
            _make_curve_entry("B", ia_offset=1),    # close
            _make_curve_entry("C", ia_offset=10),   # far
        ]
        labels = ["A", "B", "C"]
        result = match_curves(entries, labels, mode="similar", anchor_idx=0)
        self.assertEqual(result.mode, "similar")
        self.assertEqual(result.anchor_idx, 0)
        self.assertEqual(len(result.groups), 2)  # B and C ranked
        # B should be closer
        self.assertLess(result.groups[0].delta, result.groups[1].delta)

    def test_max_delta_filters(self):
        """Groups exceeding max_delta → unmatched."""
        entries = [
            _make_curve_entry("A", ia_offset=0),
            _make_curve_entry("B", ia_offset=0.1),
            _make_curve_entry("C", ia_offset=50),   # very far
            _make_curve_entry("D", ia_offset=50.1),
        ]
        labels = ["A", "B", "C", "D"]
        result = match_curves(entries, labels, mode="groups",
                              group_size=2, max_delta=5.0)
        # Only A+B pair should survive (C+D too far? depends on match_pct)
        # At minimum, A+B should be a valid group
        self.assertGreaterEqual(len(result.groups), 1)

    def test_single_entry(self):
        """One entry → unmatched."""
        entries = [_make_curve_entry("A")]
        result = match_curves(entries, ["A"], mode="groups")
        self.assertEqual(len(result.groups), 0)
        self.assertEqual(result.unmatched, [0])

    def test_algorithm_choice_accepted(self):
        """match_curves accepts and uses both algorithms without error."""
        entries = [
            _make_curve_entry("A", ia_offset=0),
            _make_curve_entry("B", ia_offset=0.1),
            _make_curve_entry("C", ia_offset=0.2),
            _make_curve_entry("D", ia_offset=0.3),
        ]
        labels = ["A", "B", "C", "D"]
        for algo in ("greedy", "optimal"):
            result = match_curves(entries, labels, mode="groups",
                                   group_size=2, algorithm=algo)
            self.assertEqual(len(result.groups), 2, f"{algo}: 2 pairs expected")


class TestSimilarLinearCost(unittest.TestCase):
    """ML-145: Find similar must be O(N), not O(N²).

    The similar branch only needs the anchor's row — the old code built
    the whole matrix and threw the rest away.
    """

    def _entries(self, n):
        return [_make_curve_entry(f"L{i}", ia_offset=i * 0.3,
                                  ua_start=100, ua_stop=400, step=10)
                for i in range(n)]

    def _count_compute_matching(self, fn):
        import lm19.quality as Q
        calls = {"n": 0}
        orig = Q.compute_matching

        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        Q.compute_matching = spy
        try:
            fn()
        finally:
            Q.compute_matching = orig
        return calls["n"]

    def test_similar_is_linear_groups_is_quadratic(self):
        from lm19.tube_matching import match_curves
        n = 10
        entries = self._entries(n)
        labels = [e["lamp_id"] for e in entries]
        sim = self._count_compute_matching(
            lambda: match_curves(entries, labels, mode="similar",
                                 anchor_idx=0))
        grp = self._count_compute_matching(
            lambda: match_curves(entries, labels, mode="groups",
                                 group_size=2))
        assert sim <= n - 1, f"similar did {sim} comparisons (O(N)={n-1})"
        assert grp == n * (n - 1) // 2, f"groups {grp} != O(N²)"
        # discriminator: the old code made similar == groups
        assert sim < grp

    def test_row_builder_matches_matrix_row(self):
        """build_curve_distance_row must reproduce the matrix's anchor
        row exactly (same per-pair math)."""
        from lm19.tube_matching import (
            build_curve_distance_matrix, build_curve_distance_row,
        )
        entries = self._entries(6)
        dist, _ = build_curve_distance_matrix(entries)
        row, _ = build_curve_distance_row(entries, anchor_idx=2)
        # Normalise inf → a finite sentinel so the whole row is compared
        # unconditionally (avoids the assert-under-if vacuity pattern).
        SENT = -1.0
        norm = lambda v: SENT if math.isinf(v) else round(v, 9)
        got = [norm(v) for v in row]
        want = [norm(v) for v in dist[2]]
        self.assertEqual(got, want)
        # discriminator: at least one real (finite, non-self) comparison
        assert sum(1 for v in want if v != SENT) >= 2

    def test_similar_honors_max_delta(self):
        """ML-142: max_delta must filter the ranked list."""
        from lm19.tube_matching import match_curves
        entries = self._entries(8)
        labels = [e["lamp_id"] for e in entries]
        wide = match_curves(entries, labels, mode="similar", anchor_idx=0)
        deltas = sorted(g.delta for g in wide.groups)
        assert len(deltas) >= 3, "precondition: several comparables"
        cap = deltas[len(deltas) // 2]
        capped = match_curves(entries, labels, mode="similar",
                              anchor_idx=0, max_delta=cap)
        assert all(g.delta <= cap for g in capped.groups)
        assert len(capped.groups) < len(wide.groups)

    def test_similar_honors_min_overlap(self):
        """ML-142: min_overlap above the available overlap makes every
        pair incomparable (empty ranked list)."""
        from lm19.tube_matching import match_curves
        entries = self._entries(5)
        labels = [e["lamp_id"] for e in entries]
        r = match_curves(entries, labels, mode="similar", anchor_idx=0,
                         min_overlap=10_000)
        assert r.groups == []

    def test_row_builder_same_lamp_is_incomparable(self):
        """Mutation-audit (twin checklist): the matrix's
        same-lamp_id → inf rule must hold in the ROW builder too — the
        original pins fed it all-distinct lamp_ids."""
        from lm19.tube_matching import build_curve_distance_row
        entries = self._entries(4)
        entries[2]["lamp_id"] = entries[0]["lamp_id"]   # duplicate of anchor
        row, pair_info = build_curve_distance_row(entries, anchor_idx=0)
        assert math.isinf(row[2]), "same lamp must not match itself"
        info = pair_info[(0, 2)]
        assert info.n_points == 0 and info.low_overlap is False

    def test_row_builder_pair_info_has_min_max_key(self):
        """Mutation-audit: the Compare summary looks pairs up
        by (min, max) — for j < anchor that is the REVERSE of the
        insertion order, so both directions must be present."""
        from lm19.tube_matching import build_curve_distance_row
        entries = self._entries(5)
        _row, pair_info = build_curve_distance_row(entries, anchor_idx=3)
        for j in (0, 1, 2):
            assert (min(3, j), max(3, j)) in pair_info, j
            assert (max(3, j), min(3, j)) in pair_info, j

    def test_row_builder_progress_and_cancel(self):
        from lm19.tube_matching import (
            build_curve_distance_row, MatchCancelled,
        )
        entries = self._entries(5)
        calls = []
        build_curve_distance_row(
            entries, 0, progress=lambda d, t: calls.append((d, t)) or True)
        assert calls and calls[-1][1] == 5
        with self.assertRaises(MatchCancelled):
            build_curve_distance_row(entries, 0, progress=lambda d, t: False)


class TestMatchCurvesProgress(unittest.TestCase):
    """Progress-callback contract for the O(N²) curve-matching loop.

    The Compare-tab UI uses this to drive a QProgressDialog with Cancel
    instead of spinning a worker thread (see ``_run_curve_matching``).
    """

    def test_progress_called_with_done_and_total(self):
        from lm19.tube_matching import build_curve_distance_matrix
        entries = [
            _make_curve_entry("A"),
            _make_curve_entry("B"),
            _make_curve_entry("C"),
        ]
        calls: list = []

        def progress(done, total):
            calls.append((done, total))
            return True

        build_curve_distance_matrix(entries, progress=progress)
        # Called once per outer iteration (3 entries → 3 calls)
        self.assertEqual(len(calls), 3)
        # Totals consistent
        for done, total in calls:
            self.assertEqual(total, 3)
            self.assertIn(done, (0, 1, 2))

    def test_match_curves_forwards_progress(self):
        """match_curves passes its progress callback through to
        build_curve_distance_matrix."""
        from lm19.tube_matching import match_curves
        entries = [
            _make_curve_entry("A"),
            _make_curve_entry("B"),
        ]
        seen: list = []

        def progress(done, total):
            seen.append((done, total))
            return True

        match_curves(entries, ["A", "B"], mode="groups",
                     group_size=2, progress=progress)
        self.assertTrue(seen, "progress callback never invoked")

    def test_cancel_raises_match_cancelled(self):
        from lm19.tube_matching import (
            build_curve_distance_matrix, MatchCancelled,
        )
        entries = [_make_curve_entry(f"L{i}") for i in range(5)]

        def progress(done, total):
            return done < 2  # cancel after the second outer iteration

        with self.assertRaises(MatchCancelled):
            build_curve_distance_matrix(entries, progress=progress)

    def test_cancel_propagates_through_match_curves(self):
        from lm19.tube_matching import match_curves, MatchCancelled
        entries = [_make_curve_entry(f"L{i}") for i in range(5)]
        with self.assertRaises(MatchCancelled):
            match_curves(entries, [f"L{i}" for i in range(5)],
                         mode="groups", group_size=2,
                         progress=lambda d, t: False)  # cancel immediately

    def test_no_progress_no_op(self):
        """``progress=None`` (default) — no callback, no raise."""
        from lm19.tube_matching import build_curve_distance_matrix
        entries = [_make_curve_entry("A"), _make_curve_entry("B")]
        # Should not raise — None is a valid default
        dist, info = build_curve_distance_matrix(entries, progress=None)
        self.assertEqual(len(dist), 2)

    def test_pair_info_populated(self):
        """pair_info should contain CurveDistanceInfo for each pair."""
        entries = [
            _make_curve_entry("A"),
            _make_curve_entry("B"),
            _make_curve_entry("C"),
        ]
        result = match_curves(entries, ["A", "B", "C"], mode="groups")
        # 3 entries → 3 pairs in pair_info
        self.assertIn((0, 1), result.pair_info)
        self.assertIn((0, 2), result.pair_info)
        self.assertIn((1, 2), result.pair_info)
        self.assertIsInstance(result.pair_info[(0, 1)], CurveDistanceInfo)


# ── Physical sanity tests on real measurement data ──────────────────

class TestGroupInfNeverGroups(unittest.TestCase):
    """ML-069: inf distance means INCOMPARABLE (different conditions /
    same lamp) — such chunks must never form a group, even when the
    max_delta threshold filter is disabled (== 0). The old condition
    `max_delta > 0 and (isinf or ...)` short-circuited the inf check."""

    def test_inf_chunk_unmatched_with_zero_max_delta(self):
        from lm19.tube_matching import _group_n_from_matrix
        inf = float("inf")
        dist = [[0.0, 1.0, inf],
                [1.0, 0.0, 1.0],
                [inf, 1.0, 0.0]]
        groups, unmatched = _group_n_from_matrix(dist, 3, 3, max_delta=0.0)
        self.assertEqual(groups, [])
        self.assertEqual(sorted(unmatched), [0, 1, 2])

    def test_finite_chunk_groups_with_zero_max_delta(self):
        """Negative control: max_delta=0 still disables the THRESHOLD."""
        from lm19.tube_matching import _group_n_from_matrix
        dist = [[0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0]]
        groups, unmatched = _group_n_from_matrix(dist, 3, 3, max_delta=0.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(unmatched, [])


from tests._real_data import (
    CONVERTED_DIR,
    EL84_SOVTEK_L1_PENT as _SOVTEK_L1_PENT,
    EL84_SOVTEK_L2_PENT as _SOVTEK_L2_PENT,
    EL84_ER_L1_PENT as _ER_L1_PENT,
    EL84_ER_L2_PENT as _ER_L2_PENT,
    EL84_SOVTEK_L1_TRI as _SOVTEK_L1_TRIOD,
    load_points,
)


def _load_real_pts(filename: str):
    """Load points from a converted real-measurement fixture, or None."""
    pts = load_points(filename)
    return pts if pts else None


# Known pairs:
#   SOVTEK L1+L2 (pentode) — well-matched pair, expect delta < 5%
#   ER L1+L2 (pentode) — decent pair, expect delta < 25%
#   SOVTEK L1 pentode vs triode — different modes, expect very different curves


@unittest.skipUnless(CONVERTED_DIR.exists(), "no real measurement data")
class TestRealDataSanity(unittest.TestCase):
    """Physical sanity checks on real EL84 measurements."""

    def test_sovtek_well_matched_pair(self):
        """SOVTEK L1+L2 are a known matched pair — delta should be low."""
        from lm19.quality import compute_matching, AMP_CLASS_AB
        pts_a = _load_real_pts(_SOVTEK_L1_PENT)
        pts_b = _load_real_pts(_SOVTEK_L2_PENT)
        if pts_a is None or pts_b is None:
            self.skipTest("measurement files not found")
        r = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_AB)
        self.assertIsNotNone(r, "compute_matching returned None")
        self.assertGreater(r.n_points, 100, "too few comparison points")
        delta = 100.0 - r.match_pct
        self.assertLess(delta, 5.0,
                        f"matched SOVTEK pair delta={delta:.1f}% > 5% — too high")
        self.assertGreater(r.match_pct, 0, "match_pct must be positive")
        self.assertGreater(r.rms_delta, 0, "rms should be >0 for real tubes")

    def test_er_pair_reasonable(self):
        """ER L1+L2 — decent pair, delta should be < 25%."""
        from lm19.quality import compute_matching, AMP_CLASS_AB
        pts_a = _load_real_pts(_ER_L1_PENT)
        pts_b = _load_real_pts(_ER_L2_PENT)
        if pts_a is None or pts_b is None:
            self.skipTest("measurement files not found")
        r = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_AB)
        self.assertIsNotNone(r)
        delta = 100.0 - r.match_pct
        self.assertLess(delta, 25.0,
                        f"ER pair delta={delta:.1f}% > 25% — unreasonably high")
        self.assertGreater(r.n_points, 500,
                           "multi-Ug2 pentode scan should produce many points")

    def test_same_tube_self_match(self):
        """A tube compared with itself should give ~0% delta."""
        from lm19.quality import compute_matching
        pts = _load_real_pts(_SOVTEK_L1_PENT)
        if pts is None:
            self.skipTest("measurement file not found")
        r = compute_matching(pts, pts)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.match_pct, 100.0, places=0)
        self.assertAlmostEqual(r.rms_delta, 0.0, places=2)

    def test_pentode_vs_triode_incomparable(self):
        """Pentode and triode scans are incomparable by design.

        compute_matching groups points by (Ug1, Ug2); a pentode scan
        (Ug2 in 99..249 V here) and a triode scan (Ug2 = 0) share no
        groups, so the documented contract is None. This also guards
        the grouping itself: if Ug2 ever dropped out of the key,
        cross-mode scans would suddenly appear comparable.
        """
        from lm19.quality import compute_matching
        pts_pent = _load_real_pts(_SOVTEK_L1_PENT)
        pts_tri = _load_real_pts(_SOVTEK_L1_TRIOD)
        assert pts_pent is not None and pts_tri is not None, (
            "SOVTEK L1 fixture files must load")
        r = compute_matching(pts_pent, pts_tri)
        assert r is None, (
            "pentode-vs-triode must be incomparable (documented design)")

    def test_class_a_gives_better_match_for_sovtek(self):
        """Class A should give lower delta than B for matched pair
        (operating point weighted more, knee differences less)."""
        from lm19.quality import compute_matching, AMP_CLASS_A, AMP_CLASS_B
        pts_a = _load_real_pts(_SOVTEK_L1_PENT)
        pts_b = _load_real_pts(_SOVTEK_L2_PENT)
        if pts_a is None or pts_b is None:
            self.skipTest("measurement files not found")
        r_a = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_A)
        r_b = compute_matching(pts_a, pts_b, amp_class=AMP_CLASS_B)
        self.assertIsNotNone(r_a)
        self.assertIsNotNone(r_b)
        self.assertGreaterEqual(r_a.match_pct, r_b.match_pct,
                                "Class A should show better or equal match than B")

    def test_curve_distance_matrix_real(self):
        """Distance matrix on real data: matched pair closer than cross-pair."""
        pts_s1 = _load_real_pts(_SOVTEK_L1_PENT)
        pts_s2 = _load_real_pts(_SOVTEK_L2_PENT)
        pts_er1 = _load_real_pts(_ER_L1_PENT)
        if any(p is None for p in [pts_s1, pts_s2, pts_er1]):
            self.skipTest("measurement files not found")
        entries = [
            {"lamp_id": "L1", "points": pts_s1},
            {"lamp_id": "L2", "points": pts_s2},
            {"lamp_id": "ER_L1", "points": pts_er1},
        ]
        dist, info = build_curve_distance_matrix(entries, min_overlap=10)
        # L1+L2 (matched) should be closer than L1+ER_L1 (different brand)
        self.assertLess(dist[0][1], dist[0][2],
                        "matched SOVTEK pair should be closer than cross-brand")
        # Same lamp_id check not triggered here (all different IDs)
        self.assertFalse(math.isinf(dist[0][1]))


if __name__ == "__main__":
    unittest.main()
