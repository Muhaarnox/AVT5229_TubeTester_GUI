"""Reference tube parameter database for SPICE model generation.

Loads tube parameters from config/tube_params.json and provides lookup
by tube type string (as used in lamps.json).

The database contains:
  - Koren model parameters (mu, ex, kg1, kp, kvb, [kg2]) for fitting
  - Interelectrode capacitances (ccg, cgp, ccp) in pF
  - Grid current model resistor (rgi) in ohms
  - Contact potential (vct) in volts
  - Topology (triode/pentode)

Name resolution from lamps.json type codes:
  - Compact names from lamps.json: "ECC81", "PCL86T", "ECC832_1"
  - Firmware-emitted suffix form (from UART): "ECC81_G11" → "ECC81"
  - Handle combo tube sections: "PCL86T" or "PCL86TJ12" → "PCL86_triode"
  - Match by primary key or alias: "ECC81" → found via 12AT7 aliases
  - Special section map: "ECC832_1" → "12AT7"

Sources: Norman Koren tube.lib, published datasheets, community models.
"""


from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from lm19.constants import TOPOLOGY_PENTODE, TOPOLOGY_TRIODE

log = logging.getLogger(__name__)

# Lazy-init guard for ``_db`` / ``_lookup``. Without this lock, concurrent
# first-time callers of ``lookup_tube`` would race on the ``if _db is None``
# check in ``_ensure_loaded``: the second thread sees ``_db = {}`` (set on
# entry to ``_load_db``) before the first thread has populated it, and
# returns ``None`` for an existing tube. Empirically reproducible 5/5 with
# 8 threads + Barrier on a fresh module state.
_load_lock = threading.Lock()


@dataclass
class KorenParams:
    """Koren model parameters for fitting initial guess / reference."""
    mu: float
    ex: float
    kg1: float
    kp: float
    kvb: float
    kg2: Optional[float] = None  # pentode only


@dataclass
class DempwolfParams:
    """Dempwolf Extended v2 model parameters.

    See docs/DEMPWOLF_EXTENDED_MODEL.md §14.5 for equations.

    Core (all topologies):
        mu, G, gamma, C     — cathode emission
        Gg, xi, Cg          — grid current

    Triode only:
        Kvb_t               — Region A correction (V²)

    Pentode / beam tetrode:
        Kvb, Kvb1, Kn       — knee shape
        fg2                  — geometric screen interception (v2)
        A                    — anode Durchgriff (v2)

    Beam tetrode:
        sigma, Ks            — secondary emission yield / shape
        lam, nu, w           — crossover voltage Vco(Vg, Vg2) (v2)

    Variable-mu:
        mu_b, gamma_b, svar  — second cathode section
    """
    # Core (7 params)
    mu: float
    G: float
    gamma: float
    C: float
    Gg: float
    xi: float
    Cg: float
    # Triode Region A
    Kvb_t: float = 300.0
    # Pentode knee
    Kvb: float = 24.0
    Kvb1: float = 0.0
    Kn: float = 1.0
    # v2: screen interception + Durchgriff
    fg2: float = 0.0
    A: float = 0.0
    # Beam tetrode secondary emission
    sigma: float = 0.0
    Ks: float = 4.0
    # v2: crossover voltage
    lam: float = 1.0
    nu: float = 2.0
    w: float = 0.0
    # Variable-mu
    mu_b: Optional[float] = None
    gamma_b: Optional[float] = None
    svar: float = 0.0


@dataclass
class ReefmanParams:
    """Reefman (Derk/DerkE) model parameters.

    Based on Reefman Theory.pdf (2016) and TubeLib.inc subcircuits.

    Core (all variants):
        mu, ex, kg1, kg2, kp, kvb  — cathode current (Koren core)
        als, be, A                  — current splitting (αs, β, Durchgriff)

    Secondary emission (BTetrodeD/BTetrodeDE only):
        Sc, ap, w, nu, lam         — Psec = Sc/Kg2·Va·(1+tanh(...))

    Variable-mu (PenthodeVD/PenthodeVDE only):
        mu_b, ex_b, svar           — second cathode section blend
    """
    type: str       # "BTetrodeD", "BTetrodeDE", "PenthodeD", etc.
    # Core
    mu: float
    ex: float
    kg1: float
    kg2: float
    kp: float
    kvb: float
    # Splitting
    als: float      # αs
    be: float       # β
    A: float        # Durchgriff
    # Secondary emission (BTetrode only)
    Sc: float = 0.0
    ap: float = 0.0
    w: float = 0.0
    nu: float = 0.0
    lam: float = 1.0
    # Variable-mu (V-variants)
    mu_b: Optional[float] = None
    ex_b: Optional[float] = None
    svar: float = 0.0


@dataclass
class TubeCaps:
    """Interelectrode capacitances in pF.

    Triodes use:  ccg, cgp, ccp  (3 values)
    Pentodes use: ccg1, ccg2, cpg1, cg1g2, ccp  (5 values)

    Source: TubeLib.inc (tubedata.org).
    """
    # Triode (cathode-grid, grid-plate, cathode-plate)
    ccg: float = 0.0
    cgp: float = 0.0
    # Pentode (cathode-g1, cathode-g2, plate-g1, g1-g2)
    ccg1: float = 0.0
    ccg2: float = 0.0
    cpg1: float = 0.0
    cg1g2: float = 0.0
    # Shared: cathode-plate
    ccp: float = 0.0


@dataclass
class TubeRefParams:
    """Complete reference parameters for a tube type."""
    name: str
    aliases: List[str]
    topology: str          # "triode" or "pentode"
    koren: Optional[KorenParams] = None
    dempwolf: Optional[DempwolfParams] = None
    reefman: Optional[ReefmanParams] = None
    caps: Optional[TubeCaps] = None
    rgi: int = 2000        # grid current resistor, ohms
    vct: float = 0.0       # contact potential, V
    source: str = ""


# Module-level cache
_db: Optional[Dict[str, TubeRefParams]] = None
_lookup: Optional[Dict[str, str]] = None  # normalized name → primary key


def _resolve_config_path() -> Path:
    """Return path to tube_params.json."""
    return Path(__file__).resolve().parents[1] / "config" / "tube_params.json"


def _load_tube_entry(key: str, item: Dict, db: Dict[str, TubeRefParams],
                     lookup: Dict[str, str]) -> None:
    """Build one TubeRefParams from its JSON dict and index it.

    Raises KeyError/TypeError/ValueError on an incomplete or typo'd
    entry — the caller skips that tube with a WARNING (ML-fix: one bad
    hand-edited record must not take down the whole reference DB).
    """
    koren_data = item.get("koren")
    koren = None
    if koren_data:
        koren = KorenParams(
            mu=koren_data["mu"],
            ex=koren_data["ex"],
            kg1=koren_data["kg1"],
            kp=koren_data["kp"],
            kvb=koren_data["kvb"],
            kg2=koren_data.get("kg2"),
        )

    demp_data = item.get("dempwolf")
    dempwolf = None
    if demp_data:
        dempwolf = DempwolfParams(
            mu=demp_data["mu"],
            G=demp_data["G"],
            gamma=demp_data["gamma"],
            C=demp_data["C"],
            Gg=demp_data["Gg"],
            xi=demp_data["xi"],
            Cg=demp_data["Cg"],
            Kvb_t=demp_data.get("Kvb_t", 300.0),
            Kvb=demp_data.get("Kvb", 24.0),
            Kvb1=demp_data.get("Kvb1", 0.0),
            Kn=demp_data.get("Kn", 1.0),
            fg2=demp_data.get("fg2", 0.0),
            A=demp_data.get("A", 0.0),
            sigma=demp_data.get("sigma", 0.0),
            Ks=demp_data.get("Ks", 4.0),
            lam=demp_data.get("lam", 1.0),
            nu=demp_data.get("nu", 2.0),
            w=demp_data.get("w", 0.0),
            mu_b=demp_data.get("mu_b"),
            gamma_b=demp_data.get("gamma_b"),
            svar=demp_data.get("svar", 0.0),
        )

    reef_data = item.get("reefman")
    reefman = None
    if reef_data:
        reefman = ReefmanParams(
            type=reef_data["type"],
            mu=reef_data["mu"],
            ex=reef_data["ex"],
            kg1=reef_data["kg1"],
            kg2=reef_data["kg2"],
            kp=reef_data["kp"],
            kvb=reef_data["kvb"],
            als=reef_data["als"],
            be=reef_data["be"],
            A=reef_data["A"],
            Sc=reef_data.get("Sc", 0.0),
            ap=reef_data.get("ap", 0.0),
            w=reef_data.get("w", 0.0),
            nu=reef_data.get("nu", 0.0),
            lam=reef_data.get("lam", 1.0),
            mu_b=reef_data.get("mu_b"),
            ex_b=reef_data.get("ex_b"),
            svar=reef_data.get("svar", 0.0),
        )

    caps_data = item.get("caps_pF")
    caps = None
    if caps_data:
        ccp_val = caps_data.get("ccp", 0.0)
        caps = TubeCaps(
            ccg=caps_data.get("ccg", 0.0),
            cgp=caps_data.get("cgp", 0.0),
            ccg1=caps_data.get("ccg1", 0.0),
            ccg2=caps_data.get("ccg2", 0.0),
            cpg1=caps_data.get("cpg1", 0.0),
            cg1g2=caps_data.get("cg1g2", 0.0),
            ccp=ccp_val,
        )

    entry = TubeRefParams(
        name=key,
        aliases=item.get("aliases", []),
        topology=item.get("topology", TOPOLOGY_TRIODE),
        koren=koren,
        dempwolf=dempwolf,
        reefman=reefman,
        caps=caps,
        rgi=item.get("rgi", 2000),
        vct=item.get("vct", 0.0),
        source=item.get("source", ""),
    )
    db[key] = entry

    # Index: primary key and all aliases → primary key (uppercase)
    lookup[key.upper()] = key
    for alias in entry.aliases:
        lookup[alias.upper()] = key



def _load_db() -> None:
    """Load and index tube_params.json (called once, cached).

    Builds the database in **local** dicts and only assigns to the
    module-level ``_db`` / ``_lookup`` once fully populated. This makes
    publication atomic from the perspective of other threads: they
    either see ``_db is None`` (still loading) or a fully-populated
    dict — never a half-built one.
    """
    global _db, _lookup
    db: Dict[str, TubeRefParams] = {}
    lookup: Dict[str, str] = {}

    config_path = _resolve_config_path()
    if not config_path.exists():
        log.warning("tube_params.json not found at %s", config_path)
        # Publish empty dicts so subsequent calls don't re-attempt the load.
        _db = db
        _lookup = lookup
        return

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        # tube_params.json is hand-edited (curated datasheet data) — a typo used to crash lookup_tube with a raw
        # exception AND retry the load on every subsequent lookup because
        # _db stayed None (ML: tube_params.py:218). Degrade loudly to an
        # empty reference DB instead.
        log.warning("Failed to load %s: %s — reference params disabled",
                    config_path, exc)
        _db = db
        _lookup = lookup
        return
    tubes = data.get("tubes", {})
    section_map = data.get("section_map", {})

    for key, item in tubes.items():
        if key.startswith("_"):
            continue
        try:
            _load_tube_entry(key, item, db, lookup)
        except (KeyError, TypeError, ValueError) as exc:
            # Incomplete/typo'd hand-edited entry: skip THIS tube loudly,
            # keep the rest of the reference DB usable.
            log.warning("Skipping tube entry %r in tube_params.json: %s",
                        key, exc)

    # Index section_map entries
    for section_key, target in section_map.items():
        if not section_key.startswith("_"):
            lookup[section_key.upper()] = target

    # Atomic publication: assign the fully-built dicts to module-level
    # bindings only now. Other threads that pass the ``_db is None``
    # check after this point see complete data.
    _db = db
    _lookup = lookup
    log.info("Loaded %d tube reference entries from %s", len(_db), config_path)


def _ensure_loaded() -> None:
    # Double-checked locking: fast path for the common case (already loaded)
    # avoids lock acquisition; slow path serializes the one-time init.
    if _db is None:
        with _load_lock:
            if _db is None:
                _load_db()


# ---------- Name extraction from lamps.json type codes --------------------

# Patterns for suffix stripping (firmware-emitted names from UART):
#   "ECC81_G11"  → base="ECC81"  suffix="G11"
#   "6N1P__G21"  → base="6N1P"   suffix="G21"
#   "6SL7__I12"  → base="6SL7"   suffix="I12"
#   "EL84__B02"  → base="EL84"   suffix="B02"
#   "E180CCG12"  → base="E180CC" suffix="G12"
#   "ECC832G11"  → base="ECC832" suffix="G11"
#   "PCL86TJ12"  → base="PCL86"  section="T" suffix="J12"

_SUFFIX_RE = re.compile(
    r'^(.{3,}?)_*([TPUD]?)([A-J]\d{1,2})$'
    #  base(≥3)  section  panel+num
)
_COMBO_SHORT_RE = re.compile(r'^(PCL86|ECL86|ECL82|PCL82|PCL85)([TP])$')

# Known combo tube prefixes with separate triode/pentode sections
_COMBO_PREFIXES = {"PCL86", "ECL86", "ECL82", "PCL82", "PCL85"}


def _extract_base_type(lamp_type: str) -> str:
    """Extract base tube type from lamps.json type code.

    Examples:
        "ECC81"      → "ECC81"
        "ECC832_1"   → "ECC832_1"
        "PCL86T"     → "PCL86_triode"
        "ECC81_G11"  → "ECC81"
        "6N1P__G21"  → "6N1P"
        "PCL86TJ12"  → "PCL86_triode"
        "PCL86PJ22"  → "PCL86_pentode"
        "ECC832G11"  → "ECC832_1"
        "ECC832G21"  → "ECC832_2"
    """
    combo_short = _COMBO_SHORT_RE.match(lamp_type)
    if combo_short:
        suffix = (TOPOLOGY_TRIODE if combo_short.group(2) == "T"
                  else TOPOLOGY_PENTODE)
        return f"{combo_short.group(1)}_{suffix}"

    if re.match(r"^.+_[12]$", lamp_type):
        return lamp_type

    m = _SUFFIX_RE.match(lamp_type)
    if not m:
        return lamp_type

    base = m.group(1)
    section_letter = m.group(2)  # T/P/U/D or empty
    panel_code = m.group(3)      # e.g. G11, I12, J22

    # Combo tubes: triode/pentode section
    if base in _COMBO_PREFIXES and section_letter:
        if section_letter == "T":
            return f"{base}_triode"
        elif section_letter == "P":
            return f"{base}_pentode"

    # Dual-section tubes like ECC832: section from panel code digit
    # G11 → section 1, G21 → section 2
    if len(panel_code) >= 2 and panel_code[1] in "12":
        section_num = panel_code[1]
        section_key = f"{base}_{section_num}"
        # Only use section key if it's in the section_map
        _ensure_loaded()
        if section_key.upper() in _lookup:
            return section_key

    return base


# ---------- Public API ---------------------------------------------------

def lookup_tube(lamp_type: str) -> Optional[TubeRefParams]:
    """Look up reference parameters for a tube type.

    Args:
        lamp_type: tube type string from lamps.json (e.g. "ECC81_G11")

    Returns:
        TubeRefParams or None if not found.
    """
    _ensure_loaded()

    # ML-070: try the FULL name first — the suffix regex in
    # _extract_base_type false-positives on legitimate names ending in
    # [A-J]+digit(s) ("RCA 2A3" → "RCA 2", "12BH7" → "12B"), which made
    # such aliases unfindable. A name that resolves as-is is never mangled.
    key = _lookup.get(lamp_type.upper())
    if key and key in _db:
        return _db[key]

    base = _extract_base_type(lamp_type)

    # Try direct lookup
    key = _lookup.get(base.upper())
    if key and key in _db:
        return _db[key]

    # Try without trailing digits/letters (e.g. "6V6S" → "6V6")
    stripped = re.sub(r'[A-Z]$', '', base)
    if stripped != base:
        key = _lookup.get(stripped.upper())
        if key and key in _db:
            return _db[key]

    log.debug("No reference params found for '%s' (base='%s')", lamp_type, base)
    return None


def get_koren_initial(lamp_type: str) -> Optional[KorenParams]:
    """Get Koren parameter initial guess for fitting.

    Returns KorenParams or None if not available.
    """
    ref = lookup_tube(lamp_type)
    return ref.koren if ref else None


def get_caps(lamp_type: str) -> Optional[TubeCaps]:
    """Get interelectrode capacitances.

    Returns TubeCaps or None if not available.
    """
    ref = lookup_tube(lamp_type)
    return ref.caps if ref else None


def get_topology(lamp_type: str) -> str:
    """Get tube topology ('triode' or 'pentode').

    Returns 'triode' if not found (safe default).
    """
    ref = lookup_tube(lamp_type)
    return ref.topology if ref else TOPOLOGY_TRIODE


def list_tubes() -> List[str]:
    """Return list of all primary tube type keys in the database."""
    _ensure_loaded()
    return [k for k in _db.keys() if not k.startswith("_")]
