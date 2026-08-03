
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import LampConfig
from .data_paths import health_refs_root
from .io_utils import write_json

log = logging.getLogger(__name__)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return safe or "unknown"


def _type_dir(tube_type: str) -> Path:
    path = health_refs_root(_root()) / "type" / _sanitize(tube_type)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _personal_dir(tube_type: str) -> Path:
    path = health_refs_root(_root()) / "personal" / _sanitize(tube_type)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_type_refs(tube_type: str) -> List[Dict]:
    out: List[Dict] = []
    for p in _type_dir(tube_type).glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # ML-102: invisible at the default INFO level, and a broken
            # ref silently shifts the health-scoring base — warn.
            log.warning("Skipping unreadable type ref %s: %s", p, exc)
            continue
        if not isinstance(data, dict):
            continue
        data["_path"] = str(p)
        out.append(data)
    out.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
    return out


def load_type_ref(tube_type: str, ref_id: str) -> Optional[Dict]:
    path = _type_dir(tube_type) / f"{_sanitize(ref_id)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load type ref %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def get_active_type_ref(tube_type: str) -> Optional[Dict]:
    for ref in list_type_refs(tube_type):
        if bool(ref.get("active")):
            return ref
    return None


def save_type_ref(tube_type: str, ref_id: str, payload: Dict) -> Path:
    path = _type_dir(tube_type) / f"{_sanitize(ref_id)}.json"
    write_json(path, payload)
    return path


def set_active_type_ref(tube_type: str, ref_id: str) -> None:
    ref_id_s = _sanitize(ref_id)
    for p in _type_dir(tube_type).glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # ML-102: invisible at the default INFO level, and a broken
            # ref silently shifts the health-scoring base — warn.
            log.warning("Skipping unreadable type ref %s: %s", p, exc)
            continue
        if not isinstance(data, dict):
            continue
        data["active"] = (p.stem == ref_id_s)
        write_json(p, data)


def load_personal_baseline(tube_type: str, lamp_id: str) -> Optional[Dict]:
    path = _personal_dir(tube_type) / f"{_sanitize(lamp_id)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load personal baseline %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def save_personal_baseline(tube_type: str, lamp_id: str, payload: Dict) -> Path:
    path = _personal_dir(tube_type) / f"{_sanitize(lamp_id)}.json"
    write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------

def resolve_reference(
    mode: str,
    tube_type: str,
    lamp_id: str,
    ref_id: Optional[str],
    lamp: LampConfig,
    *,
    datasheet_label: str = "Datasheet",
    type_median_label: str = "Type median",
) -> Dict:
    """Resolve health reference using fallback chain.

    Lookup order depends on *mode*:
      - "personal" → personal baseline; fallback to "type"
      - "type"     → specific ref_id → active type ref → median of all type refs
      - "datasheet" (or final fallback) → values from LampConfig
    """

    def datasheet_ref() -> Dict:
        rh = lamp.uh / lamp.ih if lamp.ih > 0 else None
        screen_ratio = (lamp.ig2 / lamp.ia) if lamp.ia > 0 else None
        return {
            "id": "datasheet",
            "label": datasheet_label,
            "source": "datasheet",
            "reference": {
                "ia": lamp.ia,
                "s": lamp.s,
                "r": lamp.r,
                "k": lamp.k,
                "rh": rh,
                "screen_ratio": screen_ratio,
            },
        }

    if mode == "personal":
        b = load_personal_baseline(tube_type, lamp_id)
        if b:
            return b
        mode = "type"

    if mode == "type":
        if ref_id:
            ref = load_type_ref(tube_type, ref_id)
            if ref:
                return ref
        active = get_active_type_ref(tube_type)
        if active:
            return active
        refs = list_type_refs(tube_type)
        if refs:
            keys = ("ia", "s", "r", "k", "rh", "screen_ratio", "emission_ratio")
            med: Dict[str, float] = {}
            for k in keys:
                values = []
                for r in refs:
                    v = ((r.get("reference") or {}).get(k))
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                if values:
                    values.sort()
                    med[k] = values[len(values) // 2]
            return {
                "id": "type_median",
                "label": type_median_label,
                "source": "type_median",
                "reference": med,
            }

    return datasheet_ref()


def build_reference_from_measurement(measurement: Dict) -> Dict:
    """Extract reference values from a completed health measurement."""
    return {
        "ia": ((measurement.get("health") or {}).get("raw") or {}).get("ia_op"),
        "s": (measurement.get("srk") or {}).get("s"),
        "r": (measurement.get("srk") or {}).get("r"),
        "k": (measurement.get("srk") or {}).get("k"),
        "emission_ratio": ((measurement.get("health") or {}).get("metrics") or {}).get("emission_ratio"),
    }
