"""Device-protection / scan flow-control exception types.

Lives under ``lm19/scan/`` for historical reasons (scan was the first
consumer), but is shared with ``lm19/health.py``:

- ``_SkipPoint`` / ``_BreakSweep`` are private flow-control signals raised
  inside settle wrappers / read paths.  They are caught by the sweep
  functions and never propagate to ``run_scan`` callers.
- ``ProtectionError`` and ``HeaterLostError`` are public; UI presents them
  to the user on scan abort.
- ``HealthProtectionError`` is raised from ``run_health_test`` during the
  OP-approach Ug1 ramp when Pa/Pg2 exceed the safety limit. Carries a
  structured ``HealthProtectionPayload`` for the UI dialog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class _SkipPoint(Exception):
    """Raised by settle wrappers when user chooses 'skip' on comm error.

    Internal flow-control: caught by the sweep loop, never propagates
    outside ``run_scan``. Effect: skip one measurement point, continue.
    """


class _BreakSweep(Exception):
    """Hardware protection triggered — break current Ua sweep direction.

    Internal flow-control: caught by the sweep loop, never propagates
    outside ``run_scan``. Effect: end the current Ua direction; Ug1
    iteration continues with the next value.
    """


class ProtectionError(RuntimeError):
    """Device protection triggered (overcurrent/overheat zeroed setpoints).

    Public — propagates to UI on scan abort.
    """


class HeaterLostError(RuntimeError):
    """Heater voltage/current dropped to near-zero during scan.

    Public — propagates to UI on scan abort.
    """


@dataclass
class HealthProtectionPayload:
    """Structured diagnostics for ``HealthProtectionError``.

    Carries everything the UI dialog needs to explain to the user what
    tripped and why. Held as a separate dataclass so the exception itself
    stays a regular ``RuntimeError`` for clean ``str()`` and pickling.
    """
    kind: str  # "pa" | "pg2"
    ua: float
    ug1: float
    ug2: float
    ia_ma: float
    ig2_ma: float
    measured_w: float
    limit_w: float
    datasheet_max_w: Optional[float]
    safety_pct: float
    step_idx: int
    total_steps: int
    start_ug1: float
    target_ug1: float
    tube_type: str
    lamp_id: str
    topology: str
    ug2_mode: str
    # True when the post-trip Ug1 safe-lock restore FAILED — the tube may
    # not be cut off; the UI dialog must show this prominently.
    ug1_restore_failed: bool = False


class HealthProtectionError(RuntimeError):
    """Pa/Pg2 safety limit exceeded during OP-approach ramp.

    Raised from ``run_health_test`` (specifically from ``_ramp_ug1_to_op``)
    when the measured anode or screen dissipation crosses the configured
    safety threshold. The UI catches this via a dedicated worker signal
    and opens ``HealthProtectionDialog`` with the full diagnostics in
    ``payload``.
    """

    def __init__(self, message: str, payload: HealthProtectionPayload) -> None:
        super().__init__(message)
        self.payload = payload
