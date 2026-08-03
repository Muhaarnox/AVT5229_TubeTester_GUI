"""Typed schema for ``run_scan`` progress events.

Background
----------
``ScanWorker.progress`` is declared as ``Signal(object)`` because Qt cannot
introspect Python ``TypedDict`` at the signal level — declaring
``Signal(SomeTypedDict)`` works syntactically but provides **no** runtime
validation (``emit('not a dict')`` is silently accepted).

The discipline therefore lives one layer down: the ``progress`` callback
in ``run_scan`` (and helpers in ``sweepers.py`` / ``refine.py`` /
``protection.py``) is annotated with the ``ScanProgress`` Union below.
Static type-checkers see the shape; receiver code dispatches on
``payload.get("event")`` against the literal names in this module.

The CI pin test in ``tests/test_code_quality.py`` greps every
``"event": "<name>"`` literal in ``lm19/scan/`` and verifies the set
matches ``KNOWN_SCAN_EVENTS`` exactly — catches typos
(``"hw_protect"`` vs ``"hw_protection"``) and silent additions of new
events that no receiver knows how to handle.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TypedDict, Union


# ── Bare measurement point (no "event" key) ──────────────────────────
# Most-common payload shape: a single Ia(Ua, Ug1[, Ug2]) reading.

class ScanPoint(TypedDict, total=False):
    """One IV measurement point. Optional fields depend on topology."""
    ua: float
    ug1: float
    ia: float
    ug2: float       # pentode only
    ig2: float       # pentode only
    uh: float
    ih: float
    series_id: int   # 0 = main scan, >0 = overlay/import


# ── Tagged event variants (each carries "event": <Literal>) ──────────

class _RefineCount(TypedDict):
    event: str        # Literal["refine_count"]  — see KNOWN_SCAN_EVENTS
    count: int


class _CurveDone(TypedDict):
    event: str        # Literal["curve_done"]
    ug1: float
    ug2: float


class _PaSweepAbort(TypedDict, total=False):
    event: str        # Literal["pa_sweep_abort"]
    consecutive: int
    ug1: float
    ug2: float        # absent in triode / ug2-track sweep


class _HwProtection(TypedDict):
    event: str        # Literal["hw_protection"]
    er: int
    errors: List[Tuple[str, str]]   # [(abbreviation, key), ...]


class _HwProtectionCleared(TypedDict):
    event: str        # Literal["hw_protection_cleared"]


class _Protection(TypedDict):
    """Emitted when a settle wrapper detects protection mid-sweep."""
    event: str        # Literal["protection"]
    param: str
    ua: float
    ug1: float
    ug2: float
    ia: float
    ig2: float
    uh: float
    ih: float


class _HeaterRestoring(TypedDict):
    event: str        # Literal["heater_restoring"]
    uh: float
    ih: float


class _HeaterLost(TypedDict):
    event: str        # Literal["heater_lost"]
    message: str


class _ScanSummary(TypedDict):
    event: str        # Literal["scan_summary"]
    duration_s: float
    total_points: int
    curves: List[dict]
    heater_lost: Optional[str]
    # ML-108/109 degradation counters (0 = clean scan)
    settle_out_of_tolerance: int
    ia_outlier_rereads: int
    ia_unstable_points: int


# ── Public unions and whitelist ──────────────────────────────────────

ScanEvent = Union[
    _RefineCount, _CurveDone, _PaSweepAbort,
    _HwProtection, _HwProtectionCleared, _Protection,
    _HeaterRestoring, _HeaterLost, _ScanSummary,
]
"""Discriminated by ``["event"]`` literal name."""

ScanProgress = Union[ScanPoint, ScanEvent]
"""Either a bare measurement point (no "event" key) or a tagged event."""


# -- Scan curve statuses (contract vocabulary) -----------------------
# Emitted by the sweepers (curve_done/scan_summary); UI maps to
# msg.Scan_status_<code>; registry <-> locales tied by a bijection
# pin in
# tests/test_conventions_guards.py.
CURVE_STATUS_COMPLETED = "completed"
CURVE_STATUS_PA_FIRST = "pa_first"
CURVE_STATUS_PA_PARTIAL = "pa_partial"
CURVE_STATUS_PG2_BREAK = "pg2_break"
CURVE_STATUS_PG2_FIRST = "pg2_first"
CURVE_STATUS_IG2_BREAK = "ig2_break"
CURVE_STATUS_IG2_FIRST = "ig2_first"
CURVE_STATUS_IG2_PREDICT = "ig2_predict"
CURVE_STATUS_ABORTED = "aborted"
CURVE_STATUS_USER_STOP = "user_stop"
SCAN_CURVE_STATUSES = frozenset({
    CURVE_STATUS_COMPLETED, CURVE_STATUS_PA_FIRST, CURVE_STATUS_PA_PARTIAL,
    CURVE_STATUS_PG2_BREAK, CURVE_STATUS_PG2_FIRST, CURVE_STATUS_IG2_BREAK,
    CURVE_STATUS_IG2_FIRST, CURVE_STATUS_IG2_PREDICT, CURVE_STATUS_ABORTED,
    CURVE_STATUS_USER_STOP,
})

KNOWN_SCAN_EVENTS: frozenset = frozenset({
    "refine_count",
    "curve_done",
    "pa_sweep_abort",
    "hw_protection",
    "hw_protection_cleared",
    "protection",
    "heater_restoring",
    "heater_lost",
    "scan_summary",
})
"""Whitelist of every ``"event": "<name>"`` emitted from ``lm19/scan/``.

Mirrored by ``tests/test_code_quality.py::TestProgressEventSchema``.
Add to BOTH this set and the matching ``TypedDict`` variant when
introducing a new event.
"""
