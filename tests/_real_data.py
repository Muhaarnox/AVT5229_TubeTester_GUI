"""Shared helpers for real-data tests.

Real measurement test fixtures live exclusively in
`tests/spice_test_data/converted/`. **Never** read from
`lm19_app/measurements/` (those are user data, not test fixtures).

Use relative paths derived from `__file__` so the suite runs identically
from any working directory and on any machine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


CONVERTED_DIR = Path(__file__).resolve().parent / "spice_test_data" / "converted"


# ── EL84 / 6P14P real measurements (AVT5229) ────────────────────
# Each file = one specimen. Filenames embed the specimen tag (L1/L2)
# so multi-specimen matching tests can pick the exact pair they need.
EL84_PENTODE_FILES: List[str] = [
    "pentode_EL84_SOVTEK_L1_real.json",
    "pentode_EL84_SOVTEK_L2_real.json",
    "pentode_EL84_ER_L1_real.json",
    "pentode_EL84_ER_L2_real.json",
]

EL84_TRIODE_FILES: List[str] = [
    "triode_EL84_SOVTEK_L1_real.json",
    "triode_EL84_ER_L1_real.json",
]

EL84_REAL_FILES: List[str] = EL84_PENTODE_FILES + EL84_TRIODE_FILES


# Specific pairs / specimens used by matching tests
EL84_SOVTEK_L1_PENT = "pentode_EL84_SOVTEK_L1_real.json"
EL84_SOVTEK_L2_PENT = "pentode_EL84_SOVTEK_L2_real.json"
EL84_ER_L1_PENT = "pentode_EL84_ER_L1_real.json"
EL84_ER_L2_PENT = "pentode_EL84_ER_L2_real.json"
EL84_SOVTEK_L1_TRI = "triode_EL84_SOVTEK_L1_real.json"


def converted_path(filename: str) -> Path:
    """Absolute path to a fixture file inside `converted/`."""
    return CONVERTED_DIR / filename


def load_converted(filename: str) -> dict:
    """Load a JSON fixture. Caller is responsible for skipping if missing."""
    return json.loads(converted_path(filename).read_text(encoding="utf-8"))


def load_points(filename: str) -> list:
    """Convenience: return the `points` list, or [] if file or key missing."""
    path = converted_path(filename)
    if not path.exists():
        return []
    return load_converted(filename).get("points", [])


def el84_pentode_paths() -> List[Path]:
    """Existing pentode fixtures, in the documented order."""
    return [converted_path(n) for n in EL84_PENTODE_FILES if converted_path(n).exists()]
