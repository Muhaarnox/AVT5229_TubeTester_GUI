"""Tube simulator based on Koren model equations.

Model equations from "Norman Koren — Improved Vacuum Tube Models for
SPICE Simulations", see SOURCES_INDEX.md.

Generates realistic measurement point sets from theoretical models
using reference parameters from tube_params.json and Koren equations
from spice_export.py.

No Qt dependencies. Used by tests and (optionally) by the app for demo mode.
"""


from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from lm19.spice_export import _koren_ia, _koren_ia_pentode, _koren_ig2_pentode
from lm19.tube_params import KorenParams, lookup_tube, list_tubes
from lm19.tube_model_base import (
    TubeModelProtocol, ModelFitResult, register_model,
    extract_arrays, build_fit_result,
)
from lm19.constants import (
    TOPOLOGY_PENTODE,
    TOPOLOGY_TRIODE,
    TOPOLOGY_TRIODE_CONNECTED,
)
from lm19.tube_model_base import (
    MODEL_TYPE_KOREN,
)
from lm19.tube_model_base import (
    MODEL_WARN_KOREN_KG2_UNFITTED,
)

log = logging.getLogger(__name__)


@dataclass
class ScanGrid:
    """Parameter grid for simulated scan.

    Mirrors the concept of ScanSettings/ScanRange from scan.py,
    but simplified for simulation (no settle times, no hardware).

    Attributes:
        ua: (start, stop, step) for anode voltage, V.
        ug1: (start, stop, step) for grid voltage, V (negative values).
        ug2: (start, stop, step) for screen grid voltage, V. None for triodes.
        ug2_track_ua: if True, Ug2 follows Ua (triode-connected pentode).
        ug2_offset: Ug2 = Ua + offset when ug2_track_ua is True.
        uh: heater voltage, V.
        ih: heater current, A.
    """
    ua: Tuple[float, float, float]
    ug1: Tuple[float, float, float]
    ug2: Optional[Tuple[float, float, float]] = None
    ug2_track_ua: bool = False
    ug2_offset: float = 0.0
    uh: float = 6.3
    ih: float = 0.3


@dataclass
class TubeModel:
    """Simulated tube with Koren model parameters.

    Can generate measurement points in the standard LM19 format.
    Satisfies TubeModelProtocol.
    """
    name: str
    topology: str
    koren: KorenParams
    uh: float = 6.3
    ih: float = 0.3
    pa_max: float = 12.5
    model_type: str = MODEL_TYPE_KOREN

    def ia(self, ua: float, ug1: float, ug2: float = 0.0) -> float:
        """Compute Ia (mA) at a single operating point."""
        k = self.koren
        if self.topology == TOPOLOGY_TRIODE:
            ia_amps = _koren_ia(
                np.float64(ua), np.float64(ug1),
                k.mu, k.ex, k.kg1, k.kp, k.kvb,
            )
        else:
            ia_amps = _koren_ia_pentode(
                np.float64(ua), np.float64(ug1), np.float64(ug2),
                k.mu, k.ex, k.kg1, k.kp, k.kvb,
            )
        return float(ia_amps) * 1000.0

    def ia_array(self, ua, ug1, ug2=0.0) -> np.ndarray:
        """Vectorized Ia (mA) over broadcastable arrays — same Koren math
        as scalar ``ia()``, evaluated element-wise (optimizer hot path)."""
        k = self.koren
        ua_a = np.asarray(ua, dtype=float)
        ug1_a = np.asarray(ug1, dtype=float)
        if self.topology == TOPOLOGY_TRIODE:
            ia_amps = _koren_ia(ua_a, ug1_a, k.mu, k.ex, k.kg1, k.kp, k.kvb)
        else:
            ia_amps = _koren_ia_pentode(
                ua_a, ug1_a, np.asarray(ug2, dtype=float),
                k.mu, k.ex, k.kg1, k.kp, k.kvb,
            )
        return np.asarray(ia_amps, dtype=float) * 1000.0

    def ig2(self, ua: float, ug1: float, ug2: float) -> float:
        """Compute Ig2 (mA) for pentodes. Returns 0 for triodes.

        ua is accepted for TubeModelProtocol compatibility (Dempwolf needs it)
        but ignored by the Koren model.
        """
        if self.topology == TOPOLOGY_TRIODE or self.koren.kg2 is None:
            return 0.0
        k = self.koren
        ig2_amps = _koren_ig2_pentode(
            np.float64(ug1), np.float64(ug2),
            k.mu, k.ex, k.kg2,
        )
        return float(ig2_amps) * 1000.0

    def params_dict(self) -> Dict:
        """Return Koren parameters as a plain dict."""
        d = {
            "mu": self.koren.mu,
            "ex": self.koren.ex,
            "kg1": self.koren.kg1,
            "kp": self.koren.kp,
            "kvb": self.koren.kvb,
        }
        if self.koren.kg2 is not None:
            d["kg2"] = self.koren.kg2
        return d

    def generate_scan(self, grid: ScanGrid) -> List[Dict]:
        """Generate measurement points matching LM19 scan format.

        Iterates over the parameter grid and computes Ia (and Ig2 for
        pentodes) at each point. Returns list of standard measurement dicts.

        Points are generated in scan order: Ug2 -> Ug1 -> Ua (outer to inner).
        """
        points: List[Dict] = []
        ua_values = np.arange(
            grid.ua[0], grid.ua[1] + grid.ua[2] * 0.5, grid.ua[2],
        )
        ug1_values = np.arange(
            grid.ug1[0], grid.ug1[1] + grid.ug1[2] * 0.5, grid.ug1[2],
        )

        if self.topology == TOPOLOGY_TRIODE or grid.ug2_track_ua:
            ug2_values = [0.0]
        elif grid.ug2 is not None:
            ug2_values = np.arange(
                grid.ug2[0], grid.ug2[1] + grid.ug2[2] * 0.5, grid.ug2[2],
            )
        else:
            ug2_values = [250.0]

        for ug2_nom in ug2_values:
            for ug1 in ug1_values:
                for ua in ua_values:
                    if self.topology == TOPOLOGY_TRIODE:
                        ug2_actual = 0.0
                    elif grid.ug2_track_ua:
                        ug2_actual = float(ua) + grid.ug2_offset
                    else:
                        ug2_actual = float(ug2_nom)

                    ia_val = self.ia(float(ua), float(ug1), ug2_actual)
                    ig2_val = self.ig2(float(ua), float(ug1), ug2_actual)

                    points.append({
                        "ua": float(ua),
                        "ug1": float(ug1),
                        "ug2": ug2_actual,
                        "ia": ia_val,
                        "ig2": ig2_val,
                        "uh": grid.uh,
                        "ih": grid.ih,
                    })
        return points

    def add_noise(
        self,
        points: List[Dict],
        sigma_pct: float = 0.5,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Add Gaussian measurement noise to simulated points.

        Adds noise to Ia and Ig2, proportional to the signal level.

        Args:
            points: clean simulated points.
            sigma_pct: noise standard deviation as % of Ia/Ig2 value.
            seed: random seed for reproducibility. None for random.

        Returns:
            New list of dicts with noisy Ia/Ig2 (originals unchanged).
        """
        rng = np.random.default_rng(seed)
        noisy: List[Dict] = []
        for p in points:
            p2 = dict(p)
            if p2["ia"] > 0:
                noise = rng.normal(0, p2["ia"] * sigma_pct / 100.0)
                p2["ia"] = max(0.0, p2["ia"] + noise)
            if p2.get("ig2", 0) > 0:
                noise = rng.normal(0, p2["ig2"] * sigma_pct / 100.0)
                p2["ig2"] = max(0.0, p2["ig2"] + noise)
            noisy.append(p2)
        return noisy


def load_model(tube_name: str) -> Optional[TubeModel]:
    """Load a tube model from tube_params.json by name or alias.

    Returns TubeModel or None if tube not found in database.
    """
    ref = lookup_tube(tube_name)
    if ref is None or ref.koren is None:
        return None
    return TubeModel(
        name=ref.name,
        topology=ref.topology,
        koren=ref.koren,
    )


TRIODE_PRESETS: Dict[str, ScanGrid] = {
    "12AX7": ScanGrid(
        ua=(0, 300, 10), ug1=(-4, 0, 0.5), uh=12.6, ih=0.15,
    ),
    "12AU7": ScanGrid(
        ua=(0, 300, 10), ug1=(-20, 0, 2), uh=12.6, ih=0.15,
    ),
    "6SN7": ScanGrid(
        ua=(0, 300, 10), ug1=(-12, 0, 1), uh=6.3, ih=0.6,
    ),
}

PENTODE_PRESETS: Dict[str, ScanGrid] = {
    "EL34": ScanGrid(
        ua=(0, 400, 10), ug1=(-30, 0, 3),
        ug2=(250, 250, 1), uh=6.3, ih=1.5,
    ),
    "EL84": ScanGrid(
        ua=(0, 300, 10), ug1=(-15, 0, 1.5),
        ug2=(250, 250, 1), uh=6.3, ih=0.76,
    ),
    "KT88": ScanGrid(
        ua=(0, 500, 10), ug1=(-40, 0, 4),
        ug2=(250, 250, 1), uh=6.3, ih=1.6,
    ),
    "6L6": ScanGrid(
        ua=(0, 400, 10), ug1=(-25, 0, 2.5),
        ug2=(250, 250, 1), uh=6.3, ih=0.9,
    ),
    "EL34_triode_connected": ScanGrid(
        ua=(0, 400, 10), ug1=(-30, 0, 3),
        ug2_track_ua=True, ug2_offset=0, uh=6.3, ih=1.5,
    ),
    "EL84_multi_ug2": ScanGrid(
        ua=(0, 300, 10), ug1=(-15, 0, 3),
        ug2=(150, 300, 50), uh=6.3, ih=0.76,
    ),
}


def quick_triode(name: str = "12AX7") -> Tuple[TubeModel, List[Dict]]:
    """Load model + generate scan in one call. For tests."""
    model = load_model(name)
    if model is None:
        raise ValueError(f"Unknown triode: {name}")
    grid = TRIODE_PRESETS.get(name, TRIODE_PRESETS["12AX7"])
    return model, model.generate_scan(grid)


def quick_pentode(name: str = "EL84") -> Tuple[TubeModel, List[Dict]]:
    """Load model + generate scan in one call. For tests."""
    model = load_model(name)
    if model is None:
        raise ValueError(f"Unknown pentode: {name}")
    grid = PENTODE_PRESETS.get(name, PENTODE_PRESETS["EL84"])
    return model, model.generate_scan(grid)


# ==========================================================================
# fit_koren — fit-only (no file write), returns ModelFitResult
# ==========================================================================

def fit_koren(points: List[Dict], topology: str) -> ModelFitResult:
    """Fit Koren model to measured data (no file I/O).

    Delegates to the existing fitters in spice_export.py but wraps
    the result as a ModelFitResult with a usable TubeModel.

    Args:
        points: measurement dicts with ua, ug1, ug2, ia, ig2 keys.
        topology: "triode", "pentode", or "triode_connected".

    Returns:
        ModelFitResult with fitted TubeModel ready for evaluation.

    Raises:
        RuntimeError: if not enough valid data points.
    """
    from lm19.spice_export import (
        _fit_koren_scipy, _fit_koren_numpy,
        _fit_pentode_scipy, _fit_pentode_numpy,
        _HAS_SCIPY,
    )

    effective_topology = (TOPOLOGY_TRIODE if topology == TOPOLOGY_TRIODE_CONNECTED
                          else topology)

    if effective_topology == TOPOLOGY_PENTODE:
        return _fit_koren_pentode(points, _HAS_SCIPY)
    else:
        return _fit_koren_triode(points, _HAS_SCIPY)


def _fit_koren_triode(points: List[Dict], has_scipy: bool) -> ModelFitResult:
    """Fit Koren triode model."""
    from lm19.spice_export import _fit_koren_scipy, _fit_koren_numpy

    data = extract_arrays(points, topology=TOPOLOGY_TRIODE)

    if has_scipy:
        params, _, converged = _fit_koren_scipy(data.ua, data.ug1, data.ia)
    else:
        params, _, converged = _fit_koren_numpy(data.ua, data.ug1, data.ia)

    mu, ex, kg1, kp, kvb = params
    ia_pred = _koren_ia(data.ua, data.ug1, mu, ex, kg1, kp, kvb)

    koren = KorenParams(mu=float(mu), ex=float(ex), kg1=float(kg1),
                        kp=float(kp), kvb=float(kvb))
    model = TubeModel(name="fit", topology=TOPOLOGY_TRIODE, koren=koren)

    return build_fit_result(
        model_type=MODEL_TYPE_KOREN,
        topology=TOPOLOGY_TRIODE,
        model=model,
        ia_pred_A=ia_pred,
        ia_meas_A=data.ia,
        n_points=data.n_points,
        converged=converged,
    )


def _fit_koren_pentode(points: List[Dict], has_scipy: bool) -> ModelFitResult:
    """Fit Koren pentode model."""
    from lm19.spice_export import _fit_pentode_scipy, _fit_pentode_numpy

    data = extract_arrays(points, topology=TOPOLOGY_PENTODE)

    if has_scipy:
        params, _, converged = _fit_pentode_scipy(
            data.ua, data.ug1, data.ug2, data.ia, data.ig2)
    else:
        params, _, converged = _fit_pentode_numpy(
            data.ua, data.ug1, data.ug2, data.ia, data.ig2)

    fit_warnings: list = []
    if not data.has_ig2:
        # ML-116: without Ig2 data the residual carries no kg2 gradient —
        # kg2 stays at the initial guess, yet model.ig2() will happily
        # fabricate a screen current from it (Pg2 checks, SPICE export).
        log.warning("Koren pentode fit without Ig2 data — kg2 is NOT "
                    "fitted (initial guess kept); model.ig2() output is "
                    "an extrapolation, not a fit")
        fit_warnings.append({"code": MODEL_WARN_KOREN_KG2_UNFITTED})

    mu, ex, kg1, kp, kvb, kg2 = params

    ia_pred = _koren_ia_pentode(
        data.ua, data.ug1, data.ug2, mu, ex, kg1, kp, kvb)
    ig2_pred = (
        _koren_ig2_pentode(data.ug1, data.ug2, mu, ex, kg2)
        if data.has_ig2 else None
    )

    koren = KorenParams(mu=float(mu), ex=float(ex), kg1=float(kg1),
                        kp=float(kp), kvb=float(kvb), kg2=float(kg2))
    model = TubeModel(name="fit", topology=TOPOLOGY_PENTODE, koren=koren)

    return build_fit_result(
        model_type=MODEL_TYPE_KOREN,
        topology=TOPOLOGY_PENTODE,
        model=model,
        ia_pred_A=ia_pred,
        ia_meas_A=data.ia,
        n_points=data.n_points,
        ig2_pred_A=ig2_pred,
        ig2_meas_A=data.ig2,
        converged=converged,
        warnings=fit_warnings,
    )


# ==========================================================================
# Register Koren in MODEL_REGISTRY
# ==========================================================================

register_model(
    model_type=MODEL_TYPE_KOREN,
    label="Koren",
    loader=load_model,
    fitter=fit_koren,
    list_tubes=list_tubes,
)
