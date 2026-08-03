"""Centralized UI colors, styles and style thresholds."""

from typing import Dict

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLayout


# Layout density constants
MARGIN = 4
SPACING_TIGHT = 3
SPACING_NORMAL = 4


def apply_tight(layout: QLayout) -> QLayout:
    """Apply standard tight margins/spacing (groupbox internals, form rows)."""
    layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
    layout.setSpacing(SPACING_TIGHT)
    return layout


def apply_no_margin(layout: QLayout) -> QLayout:
    """Apply zero margins with tight spacing (panel columns inside splitter)."""
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(SPACING_TIGHT)
    return layout

# Generic colors
COLOR_RED = "#d32f2f"
COLOR_GREEN = "#2e7d32"
COLOR_ORANGE = "#ef6c00"
# Amp-tab warnings block / status-bar warning indicator text color
AMP_WARNING_HTML_COLOR = COLOR_ORANGE
COLOR_BLUE = "#1976d2"
COLOR_ACCENT_BLUE = "#0066cc"
COLOR_LIGHT_BLUE = "#2255aa"
COLOR_WHITE = "#ffffff"
COLOR_LIGHT_GRAY = "#eeeeee"
COLOR_MID_GRAY = "#666666"
COLOR_DARK_BG = "#1a1a2e"
COLOR_MUTED_TEXT = "#888888"
COLOR_SECONDARY_TEXT = "#555555"

STYLE_SECONDARY_SMALL = f"color: {COLOR_SECONDARY_TEXT}; font-size: 9pt;"

# ── Validation colors (input fields, status labels) ──────────────
COLOR_ERROR_BG = "#ffe6e6"          # light red background
COLOR_WARN_BG = "#fff9c4"           # light yellow background
COLOR_ERROR_TEXT = "#c62828"         # red text
COLOR_WARN_TEXT = "#e65100"          # orange text
COLOR_PREVIEW_TEXT = "#333366"       # muted blue-gray for previews

# ── Reusable style strings ───────────────────────────────────────
STYLE_BOLD = "font-weight: bold;"
STYLE_BOLD_LARGE = "font-weight: bold; font-size: 12pt;"
STYLE_BOLD_LABEL = "font-weight: bold; padding: 4px;"
STYLE_BOLD_LABEL_SM = "font-weight: bold; padding: 2px 4px;"
STYLE_BOLD_RESULT = "font-weight: bold; padding: 8px;"
STYLE_MUTED_SMALL = f"color: {COLOR_MUTED_TEXT}; font-size: 11px;"
STYLE_MUTED_ITALIC = f"color: {COLOR_MUTED_TEXT}; font-style: italic;"
STYLE_ITALIC_PREVIEW = f"color: {COLOR_PREVIEW_TEXT}; font-style: italic;"

# ── Validation state styles ──────────────────────────────────────
STYLE_INPUT_ERROR = f"background-color: {COLOR_ERROR_BG};"
STYLE_INPUT_WARN = f"background-color: {COLOR_WARN_BG};"
STYLE_INPUT_OK = ""
STYLE_STATUS_ERROR = f"color: {COLOR_ERROR_TEXT};"
STYLE_STATUS_WARN = f"color: {COLOR_WARN_TEXT};"
STYLE_STATUS_OK = f"color: {COLOR_GREEN};"

# Plot/series palette (canonical definition in lm19/plot_style.py)
from lm19.constants import (
    MODEL_TYPE_DEMPWOLF,
    MODEL_TYPE_KOREN,
    MODEL_TYPE_REEFMAN,
    SOURCE_MEASUREMENTS,
)
from lm19.plot_style import SERIES_PALETTE as SERIES_PALETTE  # re-export
# ML-024: canonical values live in lm19/plot_style.py — re-export only.
from lm19.plot_style import COLOR_IA as COLOR_IA  # re-export
from lm19.plot_style import COLOR_IG2 as COLOR_IG2  # re-export

# Quality colors
QUALITY_COLORS: Dict[str, str] = {
    "Strong": "#2196F3",
    "Good": "#4CAF50",
    "Weak": "#FF9800",
    "Replace": "#F44336",
}

# Amplifier/analysis colors — used on DARK plot backgrounds (pyqtgraph)
COLOR_LEVEL_GOOD = "#4ecdc4"
COLOR_LEVEL_WARN = "#ffe66d"
COLOR_LEVEL_BAD = "#ff6b6b"
COLOR_POUT = "#a29bfe"

# Same status semantics for HTML text on LIGHT QLabel backgrounds.
# Pastel plot colors lack contrast against white; these are darkened for
# readability (≈4.5:1 contrast ratio on white per WCAG AA).
COLOR_LEVEL_GOOD_TEXT = "#0d8b80"   # dark teal
COLOR_LEVEL_WARN_TEXT = "#996600"   # dark amber
COLOR_LEVEL_BAD_TEXT = "#c0392b"    # dark red

# Data source colors for multi-source amplifier overlay
SOURCE_COLORS = {
    SOURCE_MEASUREMENTS: "#55aaff",  # Blue
    MODEL_TYPE_KOREN: "#ffaa00",     # Orange
    MODEL_TYPE_DEMPWOLF: "#55ff55",  # Green
    MODEL_TYPE_REEFMAN: "#ff55ff",   # Magenta
}
SOURCE_COLOR_DEFAULT = "#aaaaaa"

# Amplifier style thresholds
THD_GOOD_MAX = 1.0
THD_WARN_MAX = 5.0
BALANCE_GOOD_MAX = 0.5
BALANCE_WARN_MAX = 2.0
PA_RATIO_GOOD_MAX = 0.8
PA_RATIO_WARN_MAX = 1.0
HEADROOM_GOOD_MIN = 2.0
HEADROOM_WARN_MIN = 0.5


# ── Health tab: step row background colors ──────────────────────────
HEALTH_STEP_OP = QColor(230, 255, 230)              # light green
HEALTH_STEP_SRK = QColor(230, 240, 255)             # light blue
HEALTH_STEP_EMISSION_100 = QColor(255, 245, 220)    # light yellow-orange
HEALTH_STEP_EMISSION_80 = QColor(255, 235, 205)     # deeper orange

# ── Health tab: verdict row background colors ───────────────────────
HEALTH_VERDICT_STRONG_BG = QColor(220, 245, 220)    # light green
HEALTH_VERDICT_GOOD_BG = QColor(210, 230, 255)      # light blue
HEALTH_VERDICT_WEAK_BG = QColor(255, 240, 210)      # light orange
HEALTH_VERDICT_REPLACE_BG = QColor(255, 220, 220)   # light red

# Cell background for "headline" result columns (Index, Ia) in the
# history table — a near-white cream that pops slightly against the
# verdict row tints without fighting them. Bold font applies on top.
HEALTH_HISTORY_HIGHLIGHT_BG = QColor(255, 252, 235)
# History-table column indices that get the headline highlight
# (bold + HEALTH_HISTORY_HIGHLIGHT_BG, and skip the verdict row tint
# so the cells stay distinguishable on every row).
HEALTH_HISTORY_HIGHLIGHT_COLS = (10, 15)  # Index, Ia (shifted by the col-9 dbias insert)
# History-table OP-condition columns hidden by default; toggled via the
# filter row "Show conditions" checkbox.
HEALTH_HISTORY_CONDITION_COLS = (6, 7, 8)  # Ua, Ug1, Ug2

# ── Health tab: layout ──────────────────────────────────────────────
HEALTH_SPLITTER_SIZES = [300, 600]                   # left : right initial ratio
HEALTH_MIN_SECTION_SIZE = 28                         # min column width, px

# ── Health tab: progress bar step mapping (%) ──────────────────────
HEALTH_PROGRESS_OP = 10
HEALTH_PROGRESS_SRK = 45
HEALTH_PROGRESS_UH80 = 80
HEALTH_PROGRESS_SRK_SPAN = 0.55                      # SRK fills 10%–65% of bar

# ── Health tab: preheat poll interval ──────────────────────────────
HEALTH_PREHEAT_POLL_MS = 500
# ML-083: hard ceiling on waiting for preheat before a health test.
# Expected wait = lamp.warmup_s; the factor + margin absorb a slow ramp.
# Past the deadline the wait is aborted with a visible error instead of
# polling forever (dead preheat worker / heater fault = silent no-op).
HEALTH_PREHEAT_TIMEOUT_FACTOR = 3.0
HEALTH_PREHEAT_TIMEOUT_MARGIN_S = 30.0

# ── Health tab: matching group row colors (alternating) ────────────
HEALTH_MATCH_GROUP_COLORS = [
    QColor(220, 240, 255),   # light blue
    QColor(255, 240, 220),   # light peach
    QColor(220, 255, 230),   # light green
    QColor(245, 225, 255),   # light lavender
    QColor(255, 255, 210),   # light yellow
    QColor(220, 245, 245),   # light cyan
    QColor(255, 225, 225),   # light rose
    QColor(235, 245, 220),   # light lime
]
HEALTH_MATCH_INACTIVE_FG = QColor(160, 160, 160)   # dimmed text
HEALTH_MATCH_UNMATCHED_BG = QColor(240, 240, 240)  # light gray
HEALTH_MATCH_GROUP_SIZE_MIN = 2
HEALTH_MATCH_GROUP_SIZE_MAX = 8
HEALTH_MATCH_MAX_DELTA_LIMIT = 100.0               # upper bound for UI spinbox

# Delta quality colors (text foreground for Δ display)
HEALTH_MATCH_DELTA_EXCELLENT = QColor(0, 140, 0)   # green  ≤2%
HEALTH_MATCH_DELTA_GOOD = QColor(0, 100, 180)      # blue   ≤5%
HEALTH_MATCH_DELTA_FAIR = QColor(200, 140, 0)      # orange ≤10%
HEALTH_MATCH_DELTA_POOR = QColor(200, 0, 0)        # red    >10%

# Maps for setForeground (QColor) and HTML rendering (.name() hex strings)
DELTA_QUALITY_COLOR_MAP = {
    "excellent": HEALTH_MATCH_DELTA_EXCELLENT,
    "good": HEALTH_MATCH_DELTA_GOOD,
    "fair": HEALTH_MATCH_DELTA_FAIR,
    "poor": HEALTH_MATCH_DELTA_POOR,
}
DELTA_QUALITY_HEX_MAP = {
    k: v.name() for k, v in DELTA_QUALITY_COLOR_MAP.items()
}

# ── Model dialog: compare table ───────────────────────────────────
MODEL_COMPARE_BEST_BG = QColor(200, 255, 200)      # light green for best value

# ── Manual tab: setpoint-row marker column ──────────────────────────
# Every Set-values row reserves this width right of its spinbox, so the
# heater warning marker appearing (or clearing) never shifts the controls
# beside it and the Set buttons of all rows stay in one column.
MANUAL_WARN_COL_WIDTH = 16

# ── Health tab: column widths ───────────────────────────────────────
# Health History table — only free-text columns need fixed widths.
# All numeric / short-enum / fixed-format columns use QHeaderView.ResizeToContents.
HEALTH_HISTORY_LAMP_ID_WIDTH = 50   # free-text "tube_001" etc.
HEALTH_HISTORY_NAME_WIDTH = 70      # user-entered measurement name
HEALTH_HISTORY_REF_WIDTH = 60       # reference label (free-text)
HEALTH_STEPS_COL_WIDTHS = [
    80,   # Step
    50, 50, 50,  # Ua Ug1 Ug2
    40,   # Uh
    48,   # Ih
    50,   # Ia
    42,   # Ig2
    48, 48,  # Pa Pg2
    80,   # Details
]


def connection_led_stylesheet(color: str) -> str:
    return f"background-color: {color}; border-radius: 5px; border: 1px solid {COLOR_MID_GRAY};"


def io_activity_stylesheet(active: bool) -> str:
    if active:
        return (
            f"QLabel {{ color: {COLOR_WHITE}; background-color: {COLOR_BLUE}; "
            "border-radius: 3px; padding: 1px 4px; font-weight: 600; }"
        )
    return (
        f"QLabel {{ color: {COLOR_MID_GRAY}; background-color: {COLOR_LIGHT_GRAY}; "
        "border-radius: 3px; padding: 1px 4px; }"
    )


def level_color_low_good(value: float, good_max: float, warn_max: float) -> str:
    """Return a TEXT-mode level color (darker, for HTML on light bg)."""
    if value < good_max:
        return COLOR_LEVEL_GOOD_TEXT
    if value < warn_max:
        return COLOR_LEVEL_WARN_TEXT
    return COLOR_LEVEL_BAD_TEXT


def level_color_high_good(value: float, good_min: float, warn_min: float) -> str:
    """Return a TEXT-mode level color (darker, for HTML on light bg)."""
    if value > good_min:
        return COLOR_LEVEL_GOOD_TEXT
    if value > warn_min:
        return COLOR_LEVEL_WARN_TEXT
    return COLOR_LEVEL_BAD_TEXT
