"""Pytest session hooks.

xdist + Qt on Windows: worker processes hang on shutdown because Qt
C-code holds the GIL while waiting on thread/event-loop cleanup, so a
daemon thread inside the worker can't force-exit it. This is a known
issue (pytest-xdist#620, pytest-qt#225/#382, execnet gateway shutdown).

The community-accepted fallback is to kill zombie workers externally
from the master process after pytest_sessionfinish — by then all results
are already collected, so it's safe.

Guards:
- Only on Windows (other platforms don't have this hang).
- Only on successful exit (non-zero means something real crashed; we
  don't want the reaper to mask that).
- Only in master process (workers skip).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

# Headless Qt for the whole session. conftest is imported by pytest before any
# test module, so this runs before Qt is first imported — covering every UI
# test centrally (offscreen Qt is the project test convention), so a
# new UI test can't fail on headless CI just because it forgot the per-file
# setdefault. Individual setdefaults in test files remain as documentation.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_WORKER_REAP_DELAY_SEC = 2.0


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return
    if sys.platform != "win32":
        return
    if int(exitstatus) != 0:
        return

    master_pid = os.getpid()

    def _reap_children():
        time.sleep(_WORKER_REAP_DELAY_SEC)
        subprocess.run(
            [
                "wmic", "process", "where",
                f"ParentProcessId={master_pid} and Name='python.exe'",
                "call", "terminate",
            ],
            check=False,
            capture_output=True,
            timeout=10,
        )

    threading.Thread(target=_reap_children, daemon=True).start()
