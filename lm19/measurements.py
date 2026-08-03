
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from lm19.data_paths import measurements_root
from lm19.io_utils import make_unique_path, write_json
from lm19.schema import (
    MEASUREMENT_SCHEMA_VERSION,
    _check_schema_version,
    stamp_schema_version,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)

log = logging.getLogger(__name__)

# ── module local constants ──
_UG2_TRACK_TOL = 5.0  # V — max |Ug2 - Ua - offset| for auto-detection
_MEASUREMENT_LABEL = "measurement"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "unknown"


def _lamp_dir(lamp_type: str) -> Path:
    base = measurements_root(_root()) / _sanitize(lamp_type)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _measurement_filename(lamp_id: str, timestamp: str, name: str) -> str:
    base = "__".join(
        part for part in [_sanitize(lamp_id), _sanitize(timestamp), _sanitize(name)] if part
    )
    return f"{base or 'measurement'}.json"


def _import_measurement_filename(lamp_id: str, source: str, source_stem: str) -> str:
    base = "__".join(
        part
        for part in [
            _sanitize(lamp_id),
            f"import_{_sanitize(source).lower()}",
            _sanitize(source_stem),
        ]
        if part
    )
    return f"{base or 'imported'}.json"


def measurement_path(lamp_type: str, lamp_id: str, timestamp: str, name: str) -> Path:
    return _lamp_dir(lamp_type) / _measurement_filename(lamp_id, timestamp, name)


def measurement_filename(measurement: Dict) -> str:
    lamp_id = str(measurement.get("lamp_id", ""))
    timestamp = str(measurement.get("timestamp", ""))
    name = str(measurement.get("name", ""))
    return _measurement_filename(lamp_id, timestamp, name)


def save_measurement(lamp_type: str, lamp_id: str, measurement: Dict) -> Path:
    stamp_schema_version(measurement, MEASUREMENT_SCHEMA_VERSION)
    timestamp = str(measurement.get("timestamp", ""))
    name = str(measurement.get("name", ""))
    path = make_unique_path(
        measurement_path(lamp_type, lamp_id, timestamp, name)
    )
    write_json(path, measurement)
    return path


def save_imported_measurement(
    lamp_type: str,
    lamp_id: str,
    measurement: Dict,
    *,
    source: str,
    source_stem: str,
) -> Path:
    """Save imported measurement using filename without timestamp.

    Pattern: <lamp_id>__import_<source>__<source_stem>.json
    """
    stamp_schema_version(measurement, MEASUREMENT_SCHEMA_VERSION)
    path = make_unique_path(
        _lamp_dir(lamp_type) / _import_measurement_filename(lamp_id, source, source_stem)
    )
    write_json(path, measurement)
    return path


def load_measurements(lamp_type: str, lamp_id: str) -> List[Dict]:
    dir_path = _lamp_dir(lamp_type)
    measurements: List[Dict] = []
    for path in dir_path.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # One locked / non-UTF8 / corrupt file must not take down the
            # whole listing (ML-117; mirrors health_measurements.py).
            log.warning("Unreadable measurement %s, skipping: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        _check_schema_version(data, MEASUREMENT_SCHEMA_VERSION,
                              _MEASUREMENT_LABEL, str(path))
        if data.get("lamp_id") == lamp_id:
            measurements.append(data)
    return measurements


def list_lamp_ids(lamp_type: str) -> List[str]:
    dir_path = _lamp_dir(lamp_type)
    ids = set()
    for path in dir_path.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # One locked / non-UTF8 / corrupt file must not take down the
            # whole listing (ML-117; mirrors health_measurements.py).
            log.warning("Unreadable measurement %s, skipping: %s", path, exc)
            continue
        if isinstance(data, dict) and data.get("lamp_id"):
            # Skip _check_schema_version here — load_measurements covers
            # the full read path and will warn once per file. This pass
            # only collects IDs and would double-warn otherwise.
            ids.add(str(data.get("lamp_id")))
    return sorted(ids)


def load_all_measurements(lamp_type: str) -> Dict[str, List[Dict]]:
    data: Dict[str, List[Dict]] = {}
    for lamp_id in list_lamp_ids(lamp_type):
        data[lamp_id] = load_measurements(lamp_type, lamp_id)
    return data


def list_measurement_entries() -> List[Dict]:
    root = measurements_root(_root())
    entries: List[Dict] = []
    if not root.exists():
        return entries
    for lamp_dir in root.iterdir():
        if not lamp_dir.is_dir():
            continue
        lamp_type = lamp_dir.name
        for file_path in lamp_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                log.warning("Unreadable measurement %s, skipping: %s",
                            file_path, exc)
                continue
            if not isinstance(data, dict):
                continue
            _check_schema_version(data, MEASUREMENT_SCHEMA_VERSION,
                                  _MEASUREMENT_LABEL, str(file_path))
            entries.append(
                {
                    "lamp_type": data.get("tube_type", lamp_type),
                    "lamp_id": data.get("lamp_id", ""),
                    "timestamp": data.get("timestamp", ""),
                    "name": data.get("name", ""),
                    "mfg_date": data.get("mfg_date", ""),
                    "points": data.get("points", []),
                    "data": data,
                    "path": str(file_path),
                }
            )
    return entries


# ---------------------------------------------------------------------------
# Topology detection
# ---------------------------------------------------------------------------

def get_ug2_mode(entry: Dict) -> str:
    """Determine ug2_mode from measurement entry data.

    Returns 'triode', 'triode_connected', or 'pentode'.

    Falls back to auto-detection from point data when scan metadata
    is missing (older measurement files).
    """
    data = entry.get("data", {})
    scan = data.get("scan", {})
    if "ug2_mode" in scan:
        return scan["ug2_mode"]
    topology = data.get("topology", "")
    if topology == TOPOLOGY_TRIODE:
        return TOPOLOGY_TRIODE
    if scan.get("ug2_track_ua", False):
        return TOPOLOGY_TRIODE_CONNECTED
    # Auto-detect from point data for files without explicit metadata
    points = entry.get("points", [])
    if points and is_ug2_track_mode(entry, points):
        return TOPOLOGY_TRIODE_CONNECTED
    return TOPOLOGY_PENTODE


def is_entry_triode(entry: Dict) -> bool:
    """Check if measurement entry is from a true triode."""
    return get_ug2_mode(entry) == TOPOLOGY_TRIODE


def is_ug2_track_mode(entry: Dict, points: List[Dict]) -> bool:
    """Check if measurement was done in Ug2-tracks-Ua mode.

    Uses explicit scan flag if present; otherwise auto-detects from
    point data by checking if Ug2 ≈ Ua + constant offset for every point.
    Tolerates ADC noise up to ``_UG2_TRACK_TOL`` V per point.
    """
    data = entry.get("data", {})
    scan = data.get("scan", {})
    if "ug2_track_ua" in scan:
        return scan["ug2_track_ua"]

    if not points or len(points) < 2:
        return False

    # Compute median offset across all points
    offsets = [p.get("ug2", 0) - p.get("ua", 0) for p in points]
    offsets.sort()
    median_offset = offsets[len(offsets) // 2]

    # Every point must be within tolerance of the median offset
    for off in offsets:
        if abs(off - median_offset) > _UG2_TRACK_TOL:
            return False
    return True
