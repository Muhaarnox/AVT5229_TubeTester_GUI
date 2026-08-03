"""Schema-version helpers for user-data JSON files.

Three persisted JSON formats produced by the app are user data — losing
or silently mis-loading them is a real risk:

- ``measurements/<lamp_type>/*.json``   — saved IV scans
- ``health_measurements/<lamp_type>/*.json`` — saved Health tests
- arbitrary settings files (Save / Load Settings dialog)

Each save now embeds ``_schema_version: <int>``. Each load reads it and
warns when the file is newer than the current app (forward-compat hint).
Files written before this change have no version field; they are treated
as ``v0`` and load unchanged — no migration is needed yet.

When a real format change lands, the migration ladder grows here:

    _MEASUREMENT_MIGRATIONS = {
        0: lambda d: ...,   # v0 -> v1
    }

and ``_check_schema_version`` will route through it. Today the dict is
empty by design — no migrations to invent in advance.
"""

from __future__ import annotations

import logging
from typing import Dict

log = logging.getLogger(__name__)


# ── Current schema versions ──────────────────────────────────────────
# Bump when the on-disk shape changes in a way that requires a load-
# time transformation. Pure additive changes (new optional field) do
# NOT require a bump — the loader will read the new field on new files
# and ignore its absence on old files via .get(default).

MEASUREMENT_SCHEMA_VERSION = 1
HEALTH_MEASUREMENT_SCHEMA_VERSION = 1
SETTINGS_SCHEMA_VERSION = 1


# ── Sentinel for files written before versioning was introduced ──────

_PRE_VERSIONING = 0


def _check_schema_version(
    data: Dict,
    current: int,
    label: str,
    file_path: str = "",
) -> int:
    """Read ``_schema_version`` from ``data``, warn if newer than ``current``.

    Returns the effective version (``0`` if the field is absent — i.e.
    the file pre-dates this scheme).

    A future-version file is loaded anyway: callers should still try
    to read it (most likely additive changes will work). The WARNING
    surfaces the situation in the log so a user reporting a bug after
    upgrading the app and downgrading later sees why fields are missing.
    """
    raw = data.get("_schema_version", _PRE_VERSIONING)
    try:
        version = int(raw)
    except (TypeError, ValueError):
        log.warning(
            "%s%s: malformed _schema_version=%r, treating as %d (pre-versioning)",
            label, f" ({file_path})" if file_path else "",
            raw, _PRE_VERSIONING,
        )
        return _PRE_VERSIONING
    if version > current:
        log.warning(
            "%s%s: file _schema_version=%d is newer than this app (%d). "
            "Some fields may be ignored.",
            label, f" ({file_path})" if file_path else "",
            version, current,
        )
    return version


def stamp_schema_version(data: Dict, version: int) -> Dict:
    """Inject ``_schema_version: <version>`` into ``data`` (mutates in place).

    Returns the same dict for convenient chaining. Idempotent: if the
    field is already present, it is overwritten with the supplied value.
    """
    data["_schema_version"] = int(version)
    return data
