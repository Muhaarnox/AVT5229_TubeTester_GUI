
from __future__ import annotations

from typing import Dict


def _field(text: str) -> str:
    return text.strip()


def parse_lcd_line(line: str) -> Dict[str, str]:
    if len(line) < 62:
        line = line.ljust(62)
    return {
        "type": _field(line[0:2]),
        "name": _field(line[3:12]),
        "uh": _field(line[13:17]),
        "ih": _field(line[18:21]),
        "ug1": _field(line[23:27]),
        "ua": _field(line[28:31]),
        "ia": _field(line[32:37]),
        "ug2": _field(line[38:41]),
        "ig2": _field(line[42:47]),
        "s": _field(line[48:52]),
        "r": _field(line[53:57]),
        "k": _field(line[58:62]),
    }
