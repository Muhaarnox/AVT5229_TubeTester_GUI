"""Plot visual constants: colours, sizes, palettes, rendering defaults.

Pure data — no Qt/pyqtgraph imports.  Safe to import from both ``lm19/``
(logic layer) and ``app/`` (UI layer).

``app/ui_theme.py`` re-exports these so UI modules can pull them from
either path (``from lm19.plot_style import …`` for logic-level code,
``from app.ui_theme import …`` for UI code).
"""


from __future__ import annotations

# ── Line rendering ──────────────────────────────────────────────────
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_GRID_ALPHA = 0.3
PLOT_PADDING = 0.05
MODEL_DENSE_SAMPLES = 500       # points per curve for smooth model lines

# ── Series palette (ColorBrewer "Set1") ────────────────────────────
SERIES_PALETTE = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
]

# Canonical Ia/Ig2 series colours (ML-024): were duplicated as raw hex in
# app/ui_theme.py — ui_theme re-exports these.
COLOR_IA = SERIES_PALETTE[1]    # blue — anode current
COLOR_IG2 = SERIES_PALETTE[0]   # red — screen current

# ── Overlay / annotation colours ────────────────────────────────────
COLOR_LIMIT = "#cc0000"           # red — Pa_max, Ua_max, Ia_max limits
COLOR_PG2 = "#ff8c00"             # orange — Pg2_max zone, over-limit markers
COLOR_LOAD_LINE = "#0066cc"       # blue — load line, intersections
COLOR_QPOINT = "#cc0000"          # red — Q-point marker + label
COLOR_SWING = "#ff8c00"           # orange — swing endpoints, range shading
COLOR_ZONE = "#888888"            # gray — zone rect, crosshair, zero-ref lines
# Alpha tints (0–255). ML-019/020: the Q-point marker/crosshair duplicated
# COLOR_LOAD_LINE as raw RGB with inline alphas; ML-022/023: the danger-zone
# fill was a raw (255,0,0,25) in three overlays — now DERIVED from
# COLOR_LIMIT (tint shift 255→204 red).
LOAD_LINE_MARKER_ALPHA = 220      # Q-point cross marker fill
LOAD_LINE_HALO_ALPHA = 120
# Dashed model UL family of the working line.
WORKING_LINE_FAMILY_ALPHA = 110        # Q-point crosshair lines
DANGER_FILL_ALPHA = 25            # Pa/Ua/Ia over-limit zone fill

# ── Heatmap tooltip ─────────────────────────────────────────────────
COLOR_TOOLTIP_BG = "#ffffcc"      # light yellow background
COLOR_TOOLTIP_BORDER = "#999999"  # medium gray border

# ── Amplifier-tab curves (THD, gain, Zout, Pa) ──────────────────────
COLOR_HD4 = "#cc77ff"             # purple — HD4 curve
COLOR_HD5 = "#77ccff"             # light blue — HD5 curve
COLOR_GAIN = "#ffcc44"            # amber — Gain curve
COLOR_ZOUT = "#44cccc"            # teal — Zout curve
COLOR_PA = "#ff6644"              # red-orange — Pa curve
COLOR_OPT_PARETO = "#00ccff"      # cyan — Pareto-front markers in optimizer

# -- Transfer tab (Ua filter) ----------------------------------------
# Ua slices outside the working-line swing are dimmed so the eye
# catches the working region at once; the dynamic intersection curve
# is bold with a white underlay, else it merges with the static
# slice family.
TRANSFER_DIM_ALPHA = 70           # out-of-swing slice pen/label alpha
LOAD_LINE_CURVE_MIN_WIDTH = 3.0   # dynamic transfer curve minimum pen width
LOAD_LINE_HALO_EXTRA_W = 2.0      # halo underlay extra width vs main pen
COLOR_CURVE_HALO = "#ffffff"      # halo/underlay stroke behind accent curves
# PP composite transfer curve (magenta — not from SERIES_PALETTE, to
# stand off the slice family; blue is taken by the dynamic SE curve).
COLOR_PP_COMPOSITE = "#cc00aa"

# ── Marker sizes ────────────────────────────────────────────────────
QPOINT_SIZE = 14                  # Q-point symbol size (all plots)
