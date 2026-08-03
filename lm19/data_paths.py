"""Resolve configurable data directories (measurements / health refs).

Anchored on the project root (``lm19_app/``) so test fixtures that monkey-patch
``_root`` in the consuming module continue to work — the anchor is passed in
by the caller. Relative paths in ``config/app.json`` resolve against this
anchor; absolute paths are used as-is. Missing directories are auto-created
with a one-shot WARNING per unique path per process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Tracks paths we've already warned about creating (one-shot per process).
_warned_created: set[str] = set()


def resolve_data_dir(anchor: Path, key: str, default_subdir: str) -> Path:
    """Resolve a data directory from ``<anchor>/config/app.json``.

    Args:
        anchor: project root. Used to locate ``config/app.json`` and as the
            base for relative paths in the config.
        key: key name to read from ``app.json`` (e.g. ``"measurements_dir"``).
        default_subdir: subdirectory under ``anchor`` used when the key is
            missing or empty. May be nested (e.g. ``"config/health_refs"``).

    Returns:
        Absolute, normalized ``Path``. Directory is auto-created if missing
        (with a one-shot WARNING log).
    """
    app_json = anchor / "config" / "app.json"
    raw = ""
    if app_json.exists():
        try:
            data = json.loads(app_json.read_text(encoding="utf-8"))
            raw = str(data.get(key, "") or "").strip()
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read %s for key %r: %s", app_json, key, exc)

    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = anchor / candidate
    else:
        candidate = anchor / default_subdir

    resolved = candidate.resolve()
    _ensure_dir(resolved)
    return resolved


def measurements_root(anchor: Path) -> Path:
    """Resolve the I-V measurements root directory for the given project anchor."""
    return resolve_data_dir(anchor, "measurements_dir", "measurements")


def health_measurements_root(anchor: Path) -> Path:
    """Resolve the Tube Health measurements root directory for the given anchor."""
    return resolve_data_dir(anchor, "health_measurements_dir", "health_measurements")


def health_refs_root(anchor: Path) -> Path:
    """Resolve the Tube Health reference root (holds ``type/`` and ``personal/``).

    Defaults under ``config/`` rather than next to the measurement archives:
    type refs are curated instrument settings, not captured data. Both halves
    share one root so the whole reference set relocates as a unit.
    """
    return resolve_data_dir(anchor, "health_refs_dir", "config/health_refs")


def _ensure_dir(p: Path) -> None:
    if p.exists():
        return
    key = str(p)
    p.mkdir(parents=True, exist_ok=True)
    if key not in _warned_created:
        log.warning("Data directory did not exist; auto-created: %s", p)
        _warned_created.add(key)


def _reset_warning_cache() -> None:
    """Test helper — clear the one-shot warning cache."""
    _warned_created.clear()
