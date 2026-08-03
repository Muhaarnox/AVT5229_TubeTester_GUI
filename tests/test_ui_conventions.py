"""UI conventions: small but checkable.

Pins:
- ML-037: every ``tip.Spice_*`` key used in code exists in BOTH
  locales (a tooltip showing a raw key = a broken tooltip); the pin
  is generic — it also catches future broken tip keys in app/;
- ML-046/047: the curves-mixin Ua clustering uses
  ``self.ua_cluster_thr``, not a hardcoded threshold;
- ML-048: ``make_double_spinbox`` applies decimals BEFORE setRange —
  bounds are not rounded by the default 2 decimals.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from i18n_setup import available_locales

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── ML-037: no raw-key tooltips ──────────────────────────────────────

class TestTooltipKeysExist:

    @pytest.mark.parametrize("locale", available_locales())
    def test_all_used_tip_keys_translated(self, locale):
        """Every ``tip.<key>`` in app/*.py must exist in the locale —
        t() returns the raw key and the user sees 'tip.Spice_flow'
        instead of a hint."""
        used = set()
        for f in (PROJECT_ROOT / "app").rglob("*.py"):
            for m in re.finditer(r"[\"']tip\.([A-Za-z0-9_]+)[\"']",
                                 f.read_text(encoding="utf-8")):
                used.add(m.group(1))
        assert used, "no tip.* usages found — pattern drift?"
        data = json.loads((PROJECT_ROOT / "locales" / f"{locale}.json")
                          .read_text(encoding="utf-8"))
        tips = data.get("tip", {})
        missing = sorted(k for k in used if k not in tips)
        assert not missing, f"{locale}: missing tip keys: {missing}"


# ── ML-046/047: cluster threshold is the named attribute ─────────────

class TestCurvesClusterThreshold:

    def test_no_hardcoded_ua_threshold(self):
        src = (PROJECT_ROOT / "app" / "plotting"
               / "_curves_plot_mixin.py").read_text(encoding="utf-8")
        assert '"ua", 1.0' not in src, \
            "Ua cluster threshold must be self.ua_cluster_thr (no magic 1.0)"
        assert src.count('"ua", self.ua_cluster_thr') == 2


# ── ML-048: decimals before range ────────────────────────────────────

class TestSpinboxDecimalsOrder:

    def test_bounds_not_rounded_by_default_decimals(self, qapp):
        """Factory contract: bounds are exact at decimals>2. NB: in Qt
        6.10 both call orders are equivalent (setDecimals restores full
        bound precision) — the pin holds the CONTRACT, not the order;
        the factory order was fixed defensively (revert-verify of this
        mutation is impossible by construction, documented)."""
        from app.widget_factory import make_double_spinbox
        sb = make_double_spinbox(min_val=0.0, max_val=0.902, value=0.5,
                                 decimals=3)
        assert sb.maximum() == pytest.approx(0.902)

    def test_min_bound_precision(self, qapp):
        from app.widget_factory import make_double_spinbox
        sb = make_double_spinbox(min_val=0.001, max_val=1.0, value=0.5,
                                 decimals=3)
        assert sb.minimum() == pytest.approx(0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
