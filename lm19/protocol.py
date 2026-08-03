from __future__ import annotations

import logging
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

log = logging.getLogger(__name__)

import serial
from serial import SerialException
from lm19.constants import IA_HW_SCALE


PARAM_RE = re.compile(r"([A-Za-z0-9]+)=([0-9]+);")

DEFAULT_BAUDRATE = 9600
DEFAULT_SERIAL_TIMEOUT_S = 0.3
DEFAULT_SERIAL_WRITE_TIMEOUT_S = 0.3
DEFAULT_READ_PARAM_TIMEOUT_S = 1.0
DEFAULT_READ_LCD_TIMEOUT_S = 2.0
DEFAULT_SET_PARAM_DELAY_S = 0.05
# 9600 baud ≈ 960 bytes/s; longest response ~12 bytes ≈ 12.5 ms.
# 50 ms gives ~4× margin for the device to finish any in-flight reply.
RETRY_DRAIN_DELAY_S = 0.05

# ── Hardware precision constants (from PROTOCOL_COM.md) ──────────────
# Ua / Ug2: 8-bit PWM, 300 V / 310 steps ≈ 0.97 V/step → round to 1 V
UA_RESOLUTION_V = 1.0
UG2_RESOLUTION_V = 1.0
# Ug1: charge-pump + 10-bit ADC, Vref ≈ 3.6 V → 0.036 V/LSB
UG1_RESOLUTION_V = 0.04     # min reliable step (≥1 ADC LSB)
UG1_NOISE_V = 0.005          # effective after 64-sample HW averaging
# Ia: 10-bit ADC, two ranges
IA_NOISE_20MA = 0.005        # mA, noise after 64×N averaging (20 mA range)
IA_NOISE_200MA = 0.05        # mA, noise after 64×N averaging (200 mA range)
# Ia hardware scaling: firmware raw integer → mA
# IA_HW_SCALE imported from lm19.constants
# Default delta for SRK measurement (% of OP voltage)
# from "µTracer User Manual", see SOURCES_INDEX.md
DEFAULT_HEALTH_DELTA_PCT = 10

# Absolute min/max bounds for δ (clamp after percentage calculation)
DELTA_UA_MIN_V = 10.0       # min δUa — below this SNR is poor for Rp
DELTA_UA_MAX_V = 50.0       # max δUa — above this non-linearity grows
DELTA_UG1_MIN_V = 0.5       # min δUg1 — RCA standard ≈ 1V; 0.5V is safe floor
DELTA_UG1_MAX_V = 2.0       # max δUg1 — keeps S measurement local
DELTA_UG2_MIN_V = 5.0       # min δUg2
DELTA_UG2_MAX_V = 50.0      # max δUg2


# Global debug logger for params
_debug_params_enabled = False
_debug_params_file: Optional[Path] = None
_debug_params_callback: Optional[Callable[[str], None]] = None


def setup_param_debug(enabled: bool, file_path: Optional[str] = None) -> None:
    """Enable/disable parameter debug logging."""
    global _debug_params_enabled, _debug_params_file
    _debug_params_enabled = enabled
    if file_path:
        _debug_params_file = Path(file_path)
        _debug_params_file.parent.mkdir(parents=True, exist_ok=True)


def set_param_debug_callback(callback: Optional[Callable[[str], None]]) -> None:
    """Set callback for debug messages (e.g., for UI display)."""
    global _debug_params_callback
    _debug_params_callback = callback


def _debug_param_log(message: str) -> None:
    """Log parameter debug message to console and file."""
    if not _debug_params_enabled:
        return
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {message}"
    # Console output
    print(line)
    # File output
    if _debug_params_file:
        try:
            with _debug_params_file.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d')} {line}\n")
        except OSError:
            log.debug("Failed to write debug params file", exc_info=True)
    # Callback
    if _debug_params_callback:
        try:
            _debug_params_callback(line)
        except (TypeError, ValueError, OSError):
            log.debug("Debug params callback failed", exc_info=True)


def _decode_param_value(name: str, raw: int) -> str:
    """Decode raw parameter value to human-readable format."""
    decoders = {
        "Ua": lambda v: f"{v} V",
        "Ug1": lambda v: f"{-(v / 10.0):.2f} V",
        "Ug2": lambda v: f"{v} V",
        "Uh": lambda v: f"{v / 10.0:.2f} V",
        "Ih": lambda v: f"{v / 100.0:.3f} A",
        "Ia": lambda v: f"{v} raw",
        "Ig2": lambda v: f"{v / 100.0:.2f} mA",
        "An": lambda v: f"{v}",
    }
    decoder = decoders.get(name)
    if decoder:
        return f"{raw} -> {decoder(raw)}"
    return str(raw)


@dataclass
class ParamValue:
    name: str
    value: int


class LM19Serial:
    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_SERIAL_TIMEOUT_S,
        write_timeout: float = DEFAULT_SERIAL_WRITE_TIMEOUT_S,
        read_param_timeout: float = DEFAULT_READ_PARAM_TIMEOUT_S,
        read_lcd_timeout: float = DEFAULT_READ_LCD_TIMEOUT_S,
        set_param_delay_s: float = DEFAULT_SET_PARAM_DELAY_S,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.read_param_timeout = read_param_timeout
        self.read_lcd_timeout = read_lcd_timeout
        self.set_param_delay_s = set_param_delay_s
        self.ser: Optional[serial.Serial] = None
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.last_rx_time: Optional[float] = None
        self.lock = threading.Lock()
        self.trace_enabled = False
        self.trace_path: Optional[Path] = None

    def open(self) -> None:
        if self.ser and self.ser.is_open:
            return
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def reopen(self) -> None:
        """Close and reopen the serial port (e.g. after USB-Serial reconnect).

        The old ``serial.Serial`` object is dead after adapter disconnect;
        we must create a fresh one on the same port.
        """
        with self.lock:
            try:
                if self.ser:
                    self.ser.close()
            except (SerialException, OSError):
                # Port already gone or OS-level close failure — we're
                # about to reopen anyway. Programming errors propagate.
                pass
            self.ser = None
            self.open()

    def is_open(self) -> bool:
        return bool(self.ser and self.ser.is_open)

    def flush_input(self) -> None:
        """Discard all pending bytes in the serial input buffer."""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()

    def _write(self, text: str) -> None:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port is not open")
        payload = text.encode("ascii")
        self.ser.write(payload)
        self.tx_bytes += len(payload)
        self._trace_log(f"TX {text}")

    def _read_until_semicolon(self, timeout: float = DEFAULT_READ_PARAM_TIMEOUT_S) -> str:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port is not open")
        buf = bytearray()
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            ch = self.ser.read(1)
            if not ch:
                continue
            buf += ch
            self.rx_bytes += 1
            self.last_rx_time = time.time()
            if ch == b";":
                break
        return buf.decode("ascii", errors="ignore")

    def _read_param_response(self) -> ParamValue:
        raw = self._read_until_semicolon(timeout=self.read_param_timeout)
        self._trace_log(f"RX {raw}")
        raw = raw.strip()
        raw = raw.lstrip("\r\n")
        match = PARAM_RE.search(raw)
        if not match:
            raise ValueError(f"Invalid response: {raw!r}")
        return ParamValue(name=match.group(1), value=int(match.group(2)))

    def set_param(self, name: str, value: int, delay: Optional[float] = None) -> None:
        cmd = f"!{name}={value};"
        _debug_param_log(f"SET {name} = {_decode_param_value(name, value)}")
        set_delay = self.set_param_delay_s if delay is None else delay
        with self.lock:
            self._write(cmd)
            time.sleep(set_delay)
            self._write(cmd)

    def get_param(self, name: str, real: bool = False) -> int:
        cmd = f"?{':' if real else ''}{name};"
        with self.lock:
            self._write(cmd)
            pv = self._read_param_response()
            if pv.name != name:
                log.warning("Param name mismatch: got %r, expected %r — flushing and retrying",
                            pv.name, name)
                time.sleep(RETRY_DRAIN_DELAY_S)
                self.flush_input()
                self._write(cmd)
                pv = self._read_param_response()
        if pv.name != name:
            raise ValueError(f"Unexpected param {pv.name}, expected {name}")
        _debug_param_log(f"GET {name}{'(real)' if real else ''} = {_decode_param_value(name, pv.value)}")
        return pv.value

    def request_lcd_copy(self) -> None:
        with self.lock:
            self._write("\x1b")

    def read_lcd_copy(self, timeout: Optional[float] = None) -> str:
        timeout = timeout if timeout is not None else self.read_lcd_timeout
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port is not open")
        with self.lock:
            start = time.monotonic()
            buf = bytearray()
            while time.monotonic() - start < timeout:
                ch = self.ser.read(1)
                if not ch:
                    continue
                buf += ch
                self.rx_bytes += 1
                self.last_rx_time = time.time()
                if b"\r\n" in buf:
                    break
            # read remaining line (62 chars)
            while len(buf) < 2 + 62 and time.monotonic() - start < timeout:
                ch = self.ser.read(1)
                if not ch:
                    continue
                buf += ch
                self.rx_bytes += 1
                self.last_rx_time = time.time()
            text = buf.decode("ascii", errors="ignore")
            text = text.replace("\r\n", "", 1)
            lcd = text[:62]
            self._trace_log(f"RX LCD {lcd}")
            return lcd

    def stats(self) -> Tuple[int, int, Optional[float]]:
        return self.tx_bytes, self.rx_bytes, self.last_rx_time

    def set_trace(self, enabled: bool, path: Optional[Path] = None) -> None:
        self.trace_enabled = enabled
        if enabled and path:
            self.trace_path = path

    def _trace_log(self, message: str) -> None:
        if not self.trace_enabled:
            return
        if not self.trace_path:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {message}\n")


def encode_uh(volts: float) -> int:
    return int(round(volts * 10.0))


def decode_uh(raw: int) -> float:
    return raw / 10.0


def encode_ih(amps: float) -> int:
    return int(round(amps * 100.0))


def decode_ih(raw: int) -> float:
    return raw / 100.0


def encode_ug1(volts: float) -> int:
    """Encode Ug1 voltage to raw protocol value (hundredths of V).

    Example: -2.06 V → 206.  Range 0-2400 (0-24.00 V).
    """
    return int(round(abs(volts) * 100.0))


def decode_ug1(raw: int) -> float:
    """Decode raw Ug1 protocol value (hundredths of V) to physical voltage.

    Example: 206 → -2.06 V.
    """
    return -(raw / 100.0)


def decode_ig2(raw: int) -> float:
    return raw / 100.0


def decode_ia(raw: int) -> float:
    """Decode raw Ia protocol value to physical mA."""
    return float(raw) * IA_HW_SCALE


# --- Hardware error flags (Er bitmask from firmware) ---
ERR_OVERIH = 0x01   # Ih overcurrent
ERR_OVERIA = 0x02   # Ia overcurrent
ERR_OVERIG = 0x04   # Ig2 overcurrent
ERR_OVERTE = 0x08   # Overheat

# Abbreviation → i18n tooltip key
ERR_FLAGS = [
    (ERR_OVERIH, "OC:Ih", "live.err_overih"),
    (ERR_OVERIA, "OC:Ia", "live.err_overia"),
    (ERR_OVERIG, "OC:Ig2", "live.err_overig"),
    (ERR_OVERTE, "OT",     "live.err_overte"),
]


def decode_err(raw: int) -> list[tuple[str, str]]:
    """Decode Er bitmask into list of (abbreviation, i18n_key) tuples."""
    return [(abbr, key) for mask, abbr, key in ERR_FLAGS if raw & mask]
