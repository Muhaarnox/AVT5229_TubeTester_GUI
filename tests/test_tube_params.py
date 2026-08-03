"""Unit tests for tube_params reference database.

Run:  py -m unittest tests.test_tube_params -v

Covers:
  - Database loading and indexing
  - Name extraction from lamps.json type codes
  - Alias resolution (Western ↔ European ↔ Soviet)
  - Combo tube section mapping (PCL86/ECL86/ECL82, ECC832)
  - Public API: lookup_tube, get_koren_initial, get_caps, get_topology
"""

import os
import sys
import json
import unittest

# Ensure the app root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm19.tube_params import (
    lookup_tube, get_koren_initial, get_caps, get_topology,
    list_tubes, _extract_base_type,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


# ======================================================================
# Database loading
# ======================================================================

class TestTubeParamsLoading(unittest.TestCase):
    """Test tube_params.json loading and indexing."""

    def test_database_loads(self):
        """Database should load without errors."""
        tubes = list_tubes()
        self.assertGreater(len(tubes), 10, "Should have many tube entries")

    def test_12ax7_exists(self):
        """12AX7 should be in the database."""
        ref = lookup_tube("12AX7")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "12AX7")
        self.assertEqual(ref.topology, "triode")

    def test_12ax7_koren_params(self):
        """12AX7 should have published Koren parameters."""
        ref = lookup_tube("12AX7")
        self.assertIsNotNone(ref.koren)
        self.assertAlmostEqual(ref.koren.mu, 100.0)
        self.assertAlmostEqual(ref.koren.ex, 1.4)
        self.assertAlmostEqual(ref.koren.kg1, 1060.0)
        self.assertAlmostEqual(ref.koren.kp, 600.0)
        self.assertAlmostEqual(ref.koren.kvb, 300.0)

    def test_12ax7_capacitances(self):
        """12AX7 should have capacitance values."""
        ref = lookup_tube("12AX7")
        self.assertIsNotNone(ref.caps)
        self.assertGreater(ref.caps.ccg, 0)
        self.assertGreater(ref.caps.cgp, 0)
        self.assertGreater(ref.caps.ccp, 0)

    def test_kt88_pentode(self):
        """KT88 should be a pentode with kg2."""
        ref = lookup_tube("KT88")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.topology, "pentode")
        self.assertIsNotNone(ref.koren)
        self.assertIsNotNone(ref.koren.kg2)
        self.assertGreater(ref.koren.kg2, 0)


# ======================================================================
# Name resolution
# ======================================================================

class TestTubeParamsNameResolution(unittest.TestCase):
    """Test mapping from lamps.json type codes to database entries."""

    def test_extract_base_ecc81(self):
        """ECC81 should remain ECC81."""
        base = _extract_base_type("ECC81")
        self.assertEqual(base, "ECC81")

    def test_extract_base_6n1p(self):
        """6N1P should remain 6N1P."""
        base = _extract_base_type("6N1P")
        self.assertEqual(base, "6N1P")

    def test_extract_base_6sl7(self):
        """6SL7 should remain 6SL7."""
        base = _extract_base_type("6SL7")
        self.assertEqual(base, "6SL7")

    def test_extract_combo_triode(self):
        """PCL86T should extract 'PCL86_triode'."""
        base = _extract_base_type("PCL86T")
        self.assertEqual(base, "PCL86_triode")

    def test_extract_combo_pentode(self):
        """ECL86P should extract 'ECL86_pentode'."""
        base = _extract_base_type("ECL86P")
        self.assertEqual(base, "ECL86_pentode")

    def test_extract_e180cc(self):
        """E180CC should extract base 'E180CC'."""
        base = _extract_base_type("E180CC")
        self.assertEqual(base, "E180CC")

    def test_lookup_via_alias_ecc83(self):
        """ECC83 (alias of 12AX7) should resolve to 12AX7."""
        ref = lookup_tube("ECC83")
        self.assertIsNotNone(ref, "ECC83 should resolve via 12AX7 alias")
        self.assertEqual(ref.name, "12AX7")

    def test_lookup_via_alias_ecc82(self):
        """ECC82 (alias of 12AU7) should resolve."""
        ref = lookup_tube("ECC82")
        self.assertIsNotNone(ref)

    def test_lookup_via_alias_ecc88(self):
        """ECC88 (alias of 6DJ8) should resolve."""
        ref = lookup_tube("ECC88")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "6DJ8")

    def test_lookup_via_alias_pcc88(self):
        """PCC88 (alias of 6DJ8/ECC88) should resolve."""
        ref = lookup_tube("PCC88")
        self.assertIsNotNone(ref)

    def test_lookup_6p18p(self):
        """6P18P (alias of EL84) should resolve."""
        ref = lookup_tube("6P18P")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "EL84")

    def test_lookup_el84(self):
        """EL84 should resolve to EL84."""
        ref = lookup_tube("EL84")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.topology, "pentode")

    def test_lookup_ecc35_alias_6sl7(self):
        """ECC35 (alias of 6SL7) should resolve."""
        ref = lookup_tube("ECC35")
        self.assertIsNotNone(ref)

    def test_lookup_6n5s_alias_6as7(self):
        """6N5S (alias of 6AS7) should resolve."""
        ref = lookup_tube("6N5S")
        self.assertIsNotNone(ref)

    def test_lookup_6v6s(self):
        """6V6S (with trailing S) should resolve to 6V6."""
        ref = lookup_tube("6V6S")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "6V6")

    def test_lookup_el34(self):
        """EL34 should resolve to EL34."""
        ref = lookup_tube("EL34")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "EL34")

    def test_lookup_combo_pcl86_triode(self):
        """PCL86T should resolve to PCL86_triode."""
        ref = lookup_tube("PCL86T")
        self.assertIsNotNone(ref, "PCL86 triode section should resolve")
        self.assertEqual(ref.topology, "triode")

    def test_lookup_combo_pcl86_pentode(self):
        """PCL86P should resolve to PCL86_pentode."""
        ref = lookup_tube("PCL86P")
        self.assertIsNotNone(ref, "PCL86 pentode section should resolve")
        self.assertEqual(ref.topology, "pentode")

    def test_lookup_ecc832_section1(self):
        """ECC832_1 should resolve to 12AT7."""
        ref = lookup_tube("ECC832_1")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "12AT7")

    def test_lookup_ecc832_section2(self):
        """ECC832_2 should resolve to 12AU7."""
        ref = lookup_tube("ECC832_2")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.name, "12AU7")

    def test_lookup_unknown_tube(self):
        """Unknown tube type should return None."""
        ref = lookup_tube("XXXX")
        self.assertIsNone(ref)

    def test_get_koren_initial(self):
        """get_koren_initial should return KorenParams for known tubes."""
        koren = get_koren_initial("ECC83")
        self.assertIsNotNone(koren)
        self.assertAlmostEqual(koren.mu, 100.0)

    def test_get_caps(self):
        """get_caps should return TubeCaps for known tubes."""
        caps = get_caps("ECC83")
        self.assertIsNotNone(caps)
        self.assertAlmostEqual(caps.ccg, 1.6)  # TubeLib.inc (tubedata.org)

    def test_caps_triode_fields(self):
        """Triode caps: ccg, cgp, ccp — no cpk, no pentode fields."""
        caps = get_caps("12AX7")
        self.assertIsNotNone(caps)
        # TubeLib values for 12AX7
        self.assertAlmostEqual(caps.ccg, 1.6)
        self.assertAlmostEqual(caps.cgp, 1.6)
        self.assertAlmostEqual(caps.ccp, 0.33)
        # ``TubeCaps`` exposes ``ccp`` only — no ``cpk`` synonym
        self.assertFalse(hasattr(caps, "cpk"), "TubeCaps must not define cpk; use ccp")
        # Pentode-specific fields should be 0
        self.assertAlmostEqual(caps.ccg1, 0.0)

    def test_caps_pentode_fields(self):
        """Pentode caps: ccg1, ccg2, cpg1, cg1g2, ccp from TubeLib."""
        caps = get_caps("EL84")
        self.assertIsNotNone(caps)
        # TubeLib pentode values for EL84
        self.assertAlmostEqual(caps.ccg1, 10.8)
        self.assertAlmostEqual(caps.ccg2, 0.0)
        self.assertAlmostEqual(caps.cpg1, 0.5)
        self.assertAlmostEqual(caps.cg1g2, 0.0)
        self.assertAlmostEqual(caps.ccp, 6.5)
        # Triode-only fields must be 0 on a pentode entry
        self.assertAlmostEqual(caps.ccg, 0.0, msg="triode ccg not used for pentodes")
        self.assertAlmostEqual(caps.cgp, 0.0, msg="triode cgp not used for pentodes")

    def test_caps_pentode_el34(self):
        """EL34 pentode caps from TubeLib."""
        caps = get_caps("EL34")
        self.assertIsNotNone(caps)
        self.assertAlmostEqual(caps.ccg1, 15.4)
        self.assertAlmostEqual(caps.cpg1, 1.1)
        self.assertAlmostEqual(caps.ccp, 8.4)

    def test_caps_no_cpk_field(self):
        """``TubeCaps`` exposes ``ccp`` only — no ``cpk`` field."""
        from lm19.tube_params import TubeCaps
        self.assertFalse(hasattr(TubeCaps, "cpk"),
                         "TubeCaps must not define cpk; use ccp")

    def test_get_topology(self):
        """get_topology should return correct topology."""
        self.assertEqual(get_topology("ECC83"), "triode")
        self.assertEqual(get_topology("EL34"), "pentode")
        self.assertEqual(get_topology("UNKNOWN"), "triode")  # default


# ======================================================================
# Concurrency: lazy-init must not race
# ======================================================================
#
# Background: ``_ensure_loaded()`` lazily populates module-level ``_db``
# and ``_lookup`` on first call. Without proper locking + atomic
# publication, concurrent first-time callers race:
#   1. Thread A enters ``_ensure_loaded``, sees ``_db is None``,
#      calls ``_load_db``. A naive implementation would assign
#      ``_db = {}`` and then start populating it.
#   2. Thread B enters ``_ensure_loaded``, sees ``_db is not None``
#      ({} is not None), skips loading, returns to ``lookup_tube``.
#   3. Thread B reads from the still-empty ``_db`` → returns ``None``
#      for an existing tube.
#
# Without the safe pattern below this is empirically reproducible 5/5
# (7 of 8 threads got ``None``). The safe pattern is two-part:
#   - a lock around the slow path (double-checked locking)
#   - build ``db`` / ``lookup`` in **local** dicts and atomically publish
#     them to module level only after fully populated.
#
# This test guards both parts. If either regresses, ``nones > 0``.


class TestTubeParamsConcurrency(unittest.TestCase):
    """Lazy load must not race when called from multiple threads."""

    def test_concurrent_first_lookup_returns_complete_data(self):
        """8 threads + Barrier: all must see fully-populated DB on first call.

        Resets ``_db`` to ``None`` to force the lazy-init path, then has
        8 threads call ``lookup_tube('EL84')`` simultaneously. All must
        return the populated entry — never ``None`` (which would mean a
        thread saw a half-built dict).
        """
        import threading
        import lm19.tube_params as tp

        # Force lazy-init path
        tp._db = None
        tp._lookup = None

        results = []
        errors = []
        barrier = threading.Barrier(8)

        def worker():
            try:
                barrier.wait()
                results.append(tp.lookup_tube('EL84'))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Threads raised: {errors}")
        self.assertEqual(len(results), 8, "All 8 threads must produce a result")
        nones = [i for i, r in enumerate(results) if r is None]
        self.assertEqual(
            nones, [],
            f"{len(nones)} of 8 threads got None (race in lazy init); "
            f"_db now has {len(tp._db) if tp._db else 0} entries"
        )
        # All must be the same EL84 entry
        for r in results:
            self.assertEqual(r.name, 'EL84')


# ======================================================================
# Run standalone
# ======================================================================

class TestCorruptDbDegradation(unittest.TestCase):
    """ML-fix (tube_params.py:218): a hand-edited typo must degrade loudly,
    not crash lookup_tube with a raw exception (and retry the parse on
    every subsequent lookup because _db stayed None)."""

    def _reload_with(self, content: str):
        """Point the loader at a temp file, force a fresh load, and restore
        module state afterwards."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import lm19.tube_params as tp
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        old_db, old_lookup = tp._db, tp._lookup
        try:
            tp._db = None
            tp._lookup = None
            with patch.object(tp, "_resolve_config_path",
                              return_value=path):
                result = tp.lookup_tube("EL84")
                loaded_flag = tp._db is not None
            return result, loaded_flag, tp
        finally:
            tp._db, tp._lookup = old_db, old_lookup
            path.unlink()

    def test_corrupt_json_degrades_without_raise(self):
        result, loaded, _ = self._reload_with("{broken json")
        self.assertIsNone(result)
        # _db must be published (empty) — no re-parse retry per lookup
        self.assertTrue(loaded)

    def test_broken_entry_skipped_others_load(self):
        content = json.dumps({"tubes": {
            "BADTUBE": {"koren": {"mu": 100}},          # missing ex/kg1/...
            "EL84": {"topology": TOPOLOGY_PENTODE,
                     "koren": {"mu": 19, "ex": 1.35, "kg1": 570,
                               "kp": 105, "kvb": 12, "kg2": 4500}},
        }})
        result, loaded, _ = self._reload_with(content)
        self.assertTrue(loaded)
        self.assertIsNotNone(result)          # healthy entry survives
        self.assertAlmostEqual(result.koren.mu, 19)


if __name__ == "__main__":
    unittest.main(verbosity=2)
