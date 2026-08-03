"""Typed schema for ``run_health_test`` progress events.

Companion to ``lm19/scan/events.py`` — same rationale, same CI pin test.
``HealthWorker.progress`` is also ``Signal(object)``; the discipline lives
in the ``progress`` callback type annotation in ``lm19/health.py``.
"""

from __future__ import annotations

from typing import Optional, TypedDict, Union

from lm19.scan.events import ScanPoint


class _Step(TypedDict):
    """Phase boundary marker: which test phase is starting."""
    event: str        # Literal["step"]
    step: str         # one of {"op", "uh80", "srk"}


class _LivePoint(TypedDict):
    """Single measurement point during health test (OP / SRK probe)."""
    event: str        # Literal["live_point"]
    point: dict       # ScanPoint shape (kept as plain dict for runtime ease)


class _SrkProgress(TypedDict):
    event: str        # Literal["srk_progress"]
    done: int
    total: int


class _AnodeSync(TypedDict, total=False):
    """Anode-selector readback result."""
    event: str        # Literal["anode_sync"]
    confirmed: bool
    requested_an: int
    actual_an: int    # absent on readback failure
    error: str        # only when confirmed=False


class _Uh80Stabilizing(TypedDict, total=False):
    """Heater-warmup stabilization heartbeat."""
    event: str        # Literal["uh80_stabilizing"]
    ua: float  # measured anode voltage (ML-041: UI computes Pa from it)
    elapsed_s: int
    t_max_s: int
    eta_s: int
    ia_ma: float
    ig2_ma: float
    uh: float
    ih: float
    slope_ma_per_s: Optional[float]
    stable: bool


class _OpRamp(TypedDict, total=False):
    """OP-approach Ug1 ramp step (one per intermediate Ug1 setpoint)."""
    event: str            # Literal["op_ramp"]
    step_idx: int         # 1-based
    total_steps: int
    ug1: float            # actual Ug1 after settle
    target_ug1: float
    start_ug1: float      # safe-lock starting Ug1
    ua: float
    ug2: float
    uh: float             # measured heater voltage (calibrated)
    ih: float             # measured heater current (calibrated)
    ia_ma: float
    ig2_ma: Optional[float]
    pa_w: float
    pg2_w: Optional[float]


class _BiasServo(TypedDict, total=False):
    """One bias-servo probe while driving Ia to the reference current."""
    event: str            # Literal["bias_servo"]
    iteration: int        # 1-based
    max_iterations: int
    ug1: float            # actual Ug1 after settle
    ia_ma: float
    ref_ia_ma: float      # current the servo is aiming for
    target_ug1: float     # plan bias the servo started from


class _BiasServoAccept(TypedDict, total=False):
    """The servo accepted its operating point.

    Emitted AFTER the accepted probe's live_point (which went out with a
    plain probe tag before acceptance was known) — the live table uses
    it to retag its last servo row and show the applied shift without
    waiting for the final re-render from the saved measurement.
    """
    event: str            # Literal["bias_servo_accept"]
    ug1: float            # accepted Ug1
    ia_ma: float          # current at the accepted point
    bias_shift_v: float   # accepted Ug1 minus the plan bias


class _EmissionSweep(TypedDict, total=False):
    """One point of the Ia(Uh) sweep in deep-emission mode."""
    event: str            # Literal["emission_sweep"]
    step_idx: int         # 1-based
    total_steps: int
    uh: float             # actual heater voltage (or current for Ih lamps)
    ratio: float          # uh / uh_nominal
    ia_ma: float
    ia100_ma: float       # plateau reference measured at nominal heater


HealthEvent = Union[
    _Step, _LivePoint, _SrkProgress, _AnodeSync, _Uh80Stabilizing, _OpRamp,
    _BiasServo, _BiasServoAccept, _EmissionSweep,
]
"""Discriminated by ``["event"]`` literal name."""

HealthProgress = Union[ScanPoint, HealthEvent]
"""Either a bare measurement point (rare) or a tagged health event."""


KNOWN_HEALTH_EVENTS: frozenset = frozenset({
    "step",
    "live_point",
    "srk_progress",
    "anode_sync",
    "uh80_stabilizing",
    "op_ramp",
    "bias_servo",
    "bias_servo_accept",
    "emission_sweep",
})
"""Whitelist of every ``"event": "<name>"`` emitted from ``lm19/health.py``.

Mirrored by ``tests/test_code_quality.py::TestProgressEventSchema``.
"""
