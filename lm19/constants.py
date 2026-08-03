"""Shared constants used across multiple lm19 modules.

Collect magic numbers here to avoid duplication and make tuning easier.
"""


from __future__ import annotations

# ── Floating-point tolerances ────────────────────────────────────────
EPS = 1e-9
EPS_COARSE = 1e-6

# ── Hardware scale ───────────────────────────────────────────────────
IA_HW_SCALE = 0.01  # firmware raw Ia integer → mA

# ── Cluster thresholds for nominal grouping ──────────────────────────
UG1_CLUSTER_THR = 0.3   # V
UG2_CLUSTER_THR = 2.0   # V
UA_CLUSTER_THR = 2.0    # V

# ── Zone tolerances & expand margins ───────────────────────────────
UG2_ZONE_TOLERANCE = 5.0    # V — max |Ug2 - zone.ug2| for point to be in zone
UG2_ZONE_DEFAULT = 0.0      # V — default Ug2 in Zone (no screen grid)
EXPAND_MARGIN_UG1 = 0.2     # V — margin added around neighboring Ug1 level
EXPAND_MARGIN_UA = 3.0      # V — margin added around neighboring Ua level
EXPAND_GAP = 0.05           # V — min gap to consider a level "outside" zone
MIN_UG1_SPREAD = 0.1        # V — min Ug1 range to compute S
MIN_UA_SPREAD = 2.0         # V — min Ua range to compute R

# ── Rounding precision (ndigits for round()) ─────────────────────────
UA_ROUND = 1
UG1_ROUND = 2
UG2_ROUND = 2

# ── Domain defaults ──────────────────────────────────────────────────
DEFAULT_UB_V = 250.0    # typical plate / supply voltage
DEFAULT_UG2_V = 250.0   # typical screen grid voltage for pentodes

# -- Topologies / ug2_mode (contract vocabulary) ---------------------
# topology: "triode" | "pentode" (models, measurements, fitters);
# ug2_mode: "triode" | "triode_connected" | "pentode" (scan mode) —
# shares values with topology, TRIODE_CONNECTED is legal only there.
TOPOLOGY_TRIODE = "triode"
TOPOLOGY_PENTODE = "pentode"
TOPOLOGY_TRIODE_CONNECTED = "triode_connected"
TOPOLOGIES = frozenset({TOPOLOGY_TRIODE, TOPOLOGY_PENTODE})
UG2_MODES = frozenset({TOPOLOGY_TRIODE, TOPOLOGY_TRIODE_CONNECTED,
                       TOPOLOGY_PENTODE})
MAX_RA_KOHM = 999.0     # upper limit for Ra spinboxes and fallback

# -- Model / source contract vocabularies ----------------------------
# Defined in this leaf module, not next to the fitters: a plain contract
# string must be readable without pulling in numpy. tube_model_base and
# amp_engine re-export them, so their canonical import path is unchanged.
MODEL_TYPE_KOREN = "koren"
MODEL_TYPE_DEMPWOLF = "dempwolf"
MODEL_TYPE_REEFMAN = "reefman"
MODEL_TYPES = frozenset({MODEL_TYPE_KOREN, MODEL_TYPE_DEMPWOLF,
                         MODEL_TYPE_REEFMAN})
SOURCE_MEASUREMENTS = "measurements"  # canonical source key for raw data



# ── Model solver (bisection) ────────────────────────────────────────
BISECT_MAX_ITER = 50            # max bisection iterations
BISECT_TOLERANCE_V = 0.001      # voltage convergence threshold (V)
MODEL_SEARCH_POINTS = 200       # initial grid for sign-change detection
MODEL_UA_MIN_V = 0.1            # lower bound for Ua search range
MODEL_UA_MAX_DEFAULT_V = 500.0  # fallback upper Ua when data unavailable

# ── Heater near-zero thresholds ─────────────────────────────────────
HEATER_NEAR_ZERO_V = 0.5   # voltage-heated: below this → heater is "off"
HEATER_NEAR_ZERO_A = 0.05  # current-heated: below this → heater is "off"

# ── Power conversion ──────────────────────────────────────────────
MW_PER_W = 1000.0           # mA·V → W  (Pa = Ua * Ia / MW_PER_W)

# ── Distortion sanity limits ──────────────────────────────────────
# Hard clipping of a sine → ~48% THD (square wave). Normal operation < 20%.
MAX_SANE_THD_PCT = 50.0
# Individual harmonic > fundamental = mathematical garbage.
MAX_SANE_HD_PCT = 100.0

# ── Near-zero guards ─────────────────────────────────────────────
IG2_NEAR_ZERO_MA = 0.01    # Ig2 threshold for Ia/Ig2 ratio (avoid div-by-zero)
