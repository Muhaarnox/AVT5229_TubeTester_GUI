"""ML-130/131 — per-lamp limit chain: lamps.json → load_lamps → UI clamps.

Before ML-130/131 the sync/extract tools copied ``uh_max``/``ih_max`` onto
the lamp cards but ``load_lamps`` never read them (dead per-lamp heater
protection), and TDSL's ``Vg2Max`` was parsed and dropped before ever
reaching a card. These tests pin the whole consumption chain.

Units contract (docs/CONFIG_REFERENCE.md): ``uh_max`` — V stored WITH the
sync tool's +10% headroom; ``ih_max`` — mA holding the TDSL NOMINAL heater
current, converted to A with the same +10% headroom by the loader;
``ug2_max`` — V. The device limit is always the hard ceiling.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lm19.config import (
    DEFAULT_LIMITS, IH_MAX_HEADROOM, find_lamp, load_lamps,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
)


class TestLoaderReadsPerLampLimits:
    """Against the real shipped config — 6P18P is the user-maintained
    manual card (sync explicitly skips it), so its values are stable."""

    @pytest.fixture(scope="class")
    def lamps(self):
        return load_lamps()

    def test_6p18p_heater_caps_loaded(self, lamps):
        lamp = find_lamp(lamps, "6P18P")
        assert lamp is not None
        assert lamp.limits["uh_max"] == pytest.approx(6.9)
        # card stores 820 mA nominal -> 0.82 A * 1.1 headroom
        assert lamp.limits["ih_max"] == pytest.approx(0.82 * IH_MAX_HEADROOM)

    def test_lamp_without_card_cap_keeps_device_limit(self, lamps):
        raw = json.loads(
            (PROJECT_ROOT / "config" / "lamps.json").read_text("utf-8"))
        no_ih = [x["type"] for x in raw["lamps"] if "ih_max" not in x]
        assert no_ih, "census drift: every lamp now carries ih_max — " \
                      "pick the invariant test another way"
        lamp = find_lamp(lamps, no_ih[0])
        assert lamp.limits["ih_max"] == pytest.approx(
            load_device_limit("ih_max"))

    def test_caps_never_below_nominal(self, lamps):
        """Broken cards (seen: E182CC/ECC99 — 12.6 V heaters carrying
        uh_max=6.9) must be floored at nominal+headroom by the loader:
        a cap below the nominal would silently clamp the heater to
        under-voltage on lamp selection."""
        checked = 0
        for lamp in lamps:
            if lamp.uh > 0:
                assert lamp.limits["uh_max"] >= lamp.uh, lamp.tube_type
                checked += 1
            if lamp.ih > 0:
                assert lamp.limits["ih_max"] >= lamp.ih, lamp.tube_type
                checked += 1
        assert checked > 0, "no lamp with a nominal heater - vacuous config"


    def test_device_limit_is_hard_ceiling(self, lamps):
        dev = _device_limits()
        for lamp in lamps:
            assert lamp.limits["uh_max"] <= dev["uh_max"] + 1e-9, lamp.tube_type
            assert lamp.limits["ih_max"] <= dev["ih_max"] + 1e-9, lamp.tube_type
            assert lamp.limits["ug2_max"] <= dev["ug2_max"] + 1e-9, lamp.tube_type


class TestShippedCardsNeedNoFlooring:
    """The runtime floor is a safety net, not a licence to ship bad cards.

    ``test_caps_never_below_nominal`` asserts the POST-flooring result, so
    it passes on dirty data — the loader repairs it and logs a warning per
    lamp on every startup (seen in shipped data: E182CC/ECC99 carrying the
    6.3 V parallel-heater cap against a 12.6 V series nominal). These pins
    keep the shipped data clean at the source instead.
    """

    @staticmethod
    def _cards():
        raw = json.loads(
            (PROJECT_ROOT / "config" / "lamps.json").read_text("utf-8"))
        return raw["lamps"]

    def test_load_emits_no_flooring_warning(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="lm19.config")
        load_lamps()
        floored = [r.message for r in caplog.records if "flooring" in r.message]
        assert not floored, (
            "shipped cards need runtime repair: " + "; ".join(floored))

    def test_card_caps_declared_at_or_above_nominal(self):
        """Enumerated from the file itself — a hand-written list of lamps
        would reproduce exactly the blind spot that shipped the bad pair."""
        checked = 0
        for card in self._cards():
            nominal_uh = float(card.get("uh") or 0)
            nominal_ih = float(card.get("ih") or 0)
            if card.get("uh_max") is not None and nominal_uh > 0:
                assert float(card["uh_max"]) >= nominal_uh, (
                    f"{card['type']}: uh_max={card['uh_max']} < uh={nominal_uh}")
                checked += 1
            if card.get("ih_max") is not None and nominal_ih > 0:
                # Card unit is mA nominal; the loader adds the headroom.
                cap_a = float(card["ih_max"]) / 1000.0 * IH_MAX_HEADROOM
                assert cap_a >= nominal_ih, (
                    f"{card['type']}: ih_max={card['ih_max']}mA < ih={nominal_ih}A")
                checked += 1
        assert checked > 0, "no card declares a heater cap — vacuous pin"

    def test_limits_table_agrees_with_card_nominals(self):
        """config/lamp_limits.json feeds the sync tools — a stale cap there
        would be copied back onto the cards on the next sync run."""
        table = json.loads(
            (PROJECT_ROOT / "config" / "lamp_limits.json").read_text("utf-8")
        )["limits"]
        nominals = {c["type"]: float(c.get("uh") or 0) for c in self._cards()}
        checked = 0
        for name, entry in table.items():
            cap = entry.get("uh_max")
            nominal = nominals.get(name, 0.0)
            if cap is None or nominal <= 0:
                continue
            assert float(cap) >= nominal, (
                f"lamp_limits.json {name}: uh_max={cap} < card uh={nominal}")
            checked += 1
        assert checked > 0, "no limits entry matches a card — vacuous pin"


def _device_limits():
    from lm19.config import load_device_limits
    return load_device_limits()


def load_device_limit(key: str) -> float:
    return _device_limits()[key]


class TestLoaderUg2Cap:
    """ug2_max consumption on a crafted config (no shipped lamp carries it
    yet — the sync tool writes it only on fresh syncs)."""

    @pytest.fixture()
    def crafted_root(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config"
        cfg.mkdir()
        lamps = {"lamps": [{
            "type": "TL1", "topology": TOPOLOGY_PENTODE,
            "uh": 6.3, "ih": 0.0, "ug1": -8.0, "ua": 250.0, "ia": 48.0,
            "ug2": 250.0, "ig2": 5.0, "s": 11.3, "r": 38.0, "k": 19.5,
            "ug2_max": 250.0,
            "ranges": {
                "ua":  {"min": 50, "max": 300, "step": 10},
                "ug1": {"min": 0, "max": 12, "step": 1},
                "ug2": {"min": 100, "max": 300, "step": 10},
            },
        }, {
            "type": "TL2", "topology": TOPOLOGY_PENTODE,
            "uh": 6.3, "ih": 0.0, "ug1": -8.0, "ua": 250.0, "ia": 48.0,
            "ug2": 250.0, "ig2": 5.0, "s": 11.3, "r": 38.0, "k": 19.5,
            "uh_max": 99.0,          # over the device ceiling -> capped
            "ih_max": 820.0,         # mA nominal
        }]}
        (cfg / "lamps.json").write_text(
            json.dumps(lamps), encoding="utf-8")
        import lm19.config as config_mod
        monkeypatch.setattr(config_mod, "_resolve_paths", lambda: tmp_path)
        return tmp_path

    def test_ug2_cap_lands_in_limits_and_clamps_explicit_range(
            self, crafted_root):
        lamps = load_lamps()
        tl1 = find_lamp(lamps, "TL1")
        assert tl1.limits["ug2_max"] == pytest.approx(250.0)
        # the card's explicit ug2 range (max=300) must be clamped too
        assert tl1.ranges["ug2"].max == pytest.approx(250.0)

    def test_device_ceiling_and_ih_units(self, crafted_root):
        lamps = load_lamps()
        tl2 = find_lamp(lamps, "TL2")
        assert tl2.limits["uh_max"] == pytest.approx(
            DEFAULT_LIMITS["uh_max"])  # 99 V capped by the device ceiling
        assert tl2.limits["ih_max"] == pytest.approx(
            0.82 * IH_MAX_HEADROOM)   # mA -> A + headroom


class TestExtractLampsAllowsUg2Max:
    """extract_lamps must copy ug2_max from lamp_limits.json onto the card
    (the allowed-list silently dropped it before, ML-131)."""

    def test_apply_limits_passes_ug2_max(self):
        script = PROJECT_ROOT / "tools" / "extract_lamps.py"
        spec = importlib.util.spec_from_file_location(
            "extract_lamps_under_test", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lamps = [{"type": "EL84"}]
        mod._apply_limits(lamps, {"EL84": {"ug2_max": 300.0,
                                           "source": "TDSL"}})
        assert lamps[0].get("ug2_max") == 300.0
        assert "source" not in lamps[0]   # non-allowed keys stay filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
