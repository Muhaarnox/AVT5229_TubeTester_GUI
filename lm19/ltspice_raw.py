"""Parse LTspice .raw binary output files.

LTspice .raw format:
  - Header: UTF-16-LE text until 'Binary:\\n'
  - Binary data: sweep variable as float64, rest as float32 per row

Usage:
    result = parse_raw("output.raw")
    # result["variables"] = ["V1", "I(R1)", ...]
    # result["data"] = numpy array (n_points × n_variables)
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np

LTSPICE_EXE = r"C:\Program Files\ADI\LTspice\LTspice.exe"


def parse_raw(path: str) -> Dict:
    """Parse LTspice .raw file.

    Returns:
        Dict with keys:
            variables: List[str] — variable names
            types: List[str] — variable types ("voltage", "device_current")
            data: np.ndarray — shape (n_points, n_variables), float64
            n_points: int
    """
    raw = Path(path).read_bytes()

    # Header is UTF-16-LE
    text = raw.decode("utf-16-le", errors="replace")
    marker = "Binary:\n"
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise ValueError("No 'Binary:' marker found in .raw file")

    header = text[:marker_idx]
    data_offset = (marker_idx + len(marker)) * 2  # UTF-16 = 2 bytes/char

    # Parse header
    n_vars = int(re.search(r"No\. Variables:\s+(\d+)", header).group(1))
    n_pts = int(re.search(r"No\. Points:\s+(\d+)", header).group(1))
    var_lines = re.findall(r"\t(\d+)\t(.+?)\t(.+)", header)

    variables = [name.strip() for _, name, _ in var_lines]
    types = [vtype.strip() for _, _, vtype in var_lines]

    # Binary: sweep var = float64 (8 bytes), others = float32 (4 bytes)
    row_size = 8 + (n_vars - 1) * 4
    binary = raw[data_offset:]

    if len(binary) < row_size * n_pts:
        raise ValueError(
            f"Insufficient binary data: {len(binary)} bytes, "
            f"expected {row_size * n_pts}"
        )

    data = np.empty((n_pts, n_vars), dtype=np.float64)
    for i in range(n_pts):
        off = i * row_size
        data[i, 0] = struct.unpack_from("<d", binary, off)[0]
        others = struct.unpack_from(f"<{n_vars - 1}f", binary, off + 8)
        data[i, 1:] = others

    return {
        "variables": variables,
        "types": types,
        "data": data,
        "n_points": n_pts,
    }


def get_variable(result: Dict, name: str) -> np.ndarray:
    """Extract a named variable from parse_raw result.

    Case-insensitive partial match: 'I(V2)' matches 'I(V2)'.
    """
    name_lower = name.lower()
    for i, var in enumerate(result["variables"]):
        if var.lower() == name_lower:
            return result["data"][:, i]
    raise KeyError(f"Variable '{name}' not found in {result['variables']}")
