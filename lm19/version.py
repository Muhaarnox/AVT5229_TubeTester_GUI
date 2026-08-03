"""Application version — the single source of truth.

``APP_VERSION`` identifies the PROGRAM. It is unrelated to the data-format
versions in :mod:`lm19.schema` (``MEASUREMENT_SCHEMA_VERSION`` and friends):
those gate file compatibility and are bumped only when a stored layout
changes, while this one is bumped on every release the user should be able
to name.

Displayed in the main window title and in the footer of every generated
PDF (scan report, amplifier report, matched-tubes certificate) — a report
whose producing build cannot be identified is not reproducible later.

Format is ``MAJOR.MINOR.PATCH``, bumped by hand at release time.
"""

from __future__ import annotations

APP_VERSION = "0.1.0"
