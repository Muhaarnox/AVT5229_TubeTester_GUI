"""Shared application context — typed dependency container for tab widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lm19.app_config import AppConfig
    from lm19.calibration import CalibrationData
    from lm19.config import LampConfig
    from lm19.protocol import LM19Serial


@dataclass
class AppContext:
    """Single typed object replacing dozens of callback kwargs.

    Each field is a zero-arg callable (getter) or action callable.
    All fields are required except the preheat/stop group which
    defaults to ``None`` for tabs that don't need scan control.
    """

    # --- getters ---
    get_client: Callable[[], Optional[LM19Serial]]
    get_write_locked: Callable[[], bool]
    get_app_config: Callable[[], AppConfig]
    get_calibration: Callable[[], CalibrationData]
    get_lamps: Callable[[], List[LampConfig]]
    get_current_tube_type: Callable[[], str]
    get_current_lamp_id: Callable[[], str]

    # --- actions ---
    set_poller_active: Callable[[bool], None]

    # --- optional hardware-ownership arbiter ---
    # Returns a short token of the subsystem currently driving the device
    # (e.g. "scan"/"preheat"/"SRK"/"health"), or None if the hardware is
    # free. Used to refuse a second subsystem from commanding the device.
    get_hw_busy: Optional[Callable[[], Optional[str]]] = None

    # --- optional preheat / stop actions ---
    get_preheat_enabled: Optional[Callable[[], bool]] = None
    get_preheat_done: Optional[Callable[[], bool]] = None
    request_start_preheat: Optional[Callable[[], None]] = None
    request_stop_all: Optional[Callable[[], None]] = None
    request_stop_keep_heater: Optional[Callable[[], None]] = None
