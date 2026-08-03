
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from lm19.data_paths import health_measurements_root
from lm19.io_utils import make_unique_path, write_json
from lm19.schema import (
    HEALTH_MEASUREMENT_SCHEMA_VERSION,
    _check_schema_version,
    stamp_schema_version,
)

log = logging.getLogger(__name__)

_HEALTH_LABEL = "health_measurement"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return safe or "unknown"


def _tube_dir(tube_type: str) -> Path:
    path = health_measurements_root(_root()) / _sanitize(tube_type)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _filename(lamp_id: str, timestamp: str, name: str) -> str:
    base = "__".join(
        p for p in [_sanitize(lamp_id), _sanitize(timestamp), _sanitize(name)] if p
    )
    return f"{base or 'health'}.json"


def save_health_measurement(tube_type: str, lamp_id: str, measurement: Dict) -> Path:
    stamp_schema_version(measurement, HEALTH_MEASUREMENT_SCHEMA_VERSION)
    timestamp = str(measurement.get("timestamp", ""))
    name = str(measurement.get("name", ""))
    path = make_unique_path(
        _tube_dir(tube_type) / _filename(lamp_id, timestamp, name)
    )
    write_json(path, measurement)
    return path


def load_health_measurements(tube_type: str, lamp_id: Optional[str] = None) -> List[Dict]:
    path = _tube_dir(tube_type)
    out: List[Dict] = []
    for file_path in path.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse health measurement %s: %s", file_path, exc)
            continue
        if not isinstance(data, dict):
            continue
        _check_schema_version(data, HEALTH_MEASUREMENT_SCHEMA_VERSION,
                              _HEALTH_LABEL, str(file_path))
        if lamp_id and str(data.get("lamp_id", "")) != str(lamp_id):
            continue
        out.append(data)
    out.sort(key=lambda d: str(d.get("timestamp", "")), reverse=True)
    return out


def list_health_lamp_ids(tube_type: str) -> List[str]:
    ids = {str(m.get("lamp_id", "")) for m in load_health_measurements(tube_type) if m.get("lamp_id")}
    return sorted(ids)


def list_health_entries(tube_type: Optional[str] = None) -> List[Dict]:
    root = health_measurements_root(_root())
    out: List[Dict] = []
    if not root.exists():
        return out

    dirs = [_sanitize(tube_type)] if tube_type else [p.name for p in root.iterdir() if p.is_dir()]
    for d in dirs:
        t_dir = root / d
        if not t_dir.exists():
            continue
        for file_path in t_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to parse health measurement %s: %s", file_path, exc)
                continue
            if not isinstance(data, dict):
                continue
            _check_schema_version(data, HEALTH_MEASUREMENT_SCHEMA_VERSION,
                                  _HEALTH_LABEL, str(file_path))
            data["_file_path"] = str(file_path)
            out.append(data)
    out.sort(key=lambda d: str(d.get("timestamp", "")), reverse=True)
    return out


def delete_health_measurement(file_path: str) -> bool:
    """Delete a health measurement file. Returns True if deleted."""
    p = Path(file_path)
    if p.exists() and p.suffix == ".json":
        try:
            p.unlink()
            log.info("Deleted health measurement: %s", p)
            return True
        except OSError as exc:
            log.warning("Failed to delete %s: %s", p, exc)
    return False
