"""Filesystem helpers shared across save/export call sites.

Pure utility module — no Qt, no logger, no config. Functions here
must be safe to call from any thread and from any layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_unique_path(path: Path) -> Path:
    """Return *path* unchanged if it doesn't exist, otherwise append
    ``_1``, ``_2``, … to the stem until a non-existing name is found.

    Preserves the original suffix (``.json``, ``.csv``, etc.) so the
    helper isn't silently json-only — callers that pass a non-json
    target still get a sensibly-named unique file.

    Used by save_measurement / save_health_measurement / save_imported_
    measurement / compare_tab export — same-second scans of the same
    lamp produce identical timestamp-based filenames; this is the
    sequential-collision counter that lets them coexist.

    Note: this is a pre-write check, not an atomic create.
    Concurrent writers from a second process could still collide;
    the desktop app does not have such writers, so the simpler
    pattern is sufficient. If concurrency is later introduced,
    swap the loop for ``os.open(..., O_CREAT|O_EXCL)``.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    candidate = path.with_name(f"{stem}_{counter}{suffix}")
    while candidate.exists():
        counter += 1
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
    return candidate


def write_json(path: Path, data: Any, *, indent: int = 2,
               ensure_ascii: bool = False) -> None:
    """Write *data* as UTF-8 JSON with LF endings and a trailing newline.

    Single writer for every JSON artifact of the project (configs, saved
    measurements, health refs, calibration). Two invariants it exists to
    hold, neither of which a bare ``write_text`` gives:

    - LF regardless of platform. ``write_text``/``open(..., "w")`` run in
      text mode, so on Windows they translate ``\\n`` to ``\\r\\n`` and flip
      the line endings of an otherwise unchanged file — a whole-file diff
      on every save, which is how unrelated work gets lost during cleanup
      reverts.
    - Trailing newline, so appending a line yields a one-line diff and
      POSIX line-oriented tools see a complete last line.

    Bytes are written directly: ``json.dumps`` never emits CR, so the
    encode is the only translation step and it is explicit.
    """
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent) + "\n"
    path.write_bytes(text.encode("utf-8"))
