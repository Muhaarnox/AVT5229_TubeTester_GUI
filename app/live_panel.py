import math

from PySide6.QtWidgets import QGroupBox, QGridLayout, QLabel

from lm19.calibration import CalibrationData
from lm19.label_formats import format_label
from lm19.protocol import decode_ia, decode_ig2, decode_ih, decode_ug1, decode_uh, decode_err
from app.ui_theme import COLOR_GREEN, COLOR_ORANGE, COLOR_RED
from i18n_setup import t

# Pa label colours
_PA_NORMAL = ""
_PA_WARN = f"color: {COLOR_ORANGE}"       # orange — approaching limit
_PA_OVER = f"color: {COLOR_RED}"          # red — over limit

# Error label styles
_ERR_OK_STYLE = f"color: {COLOR_GREEN}; font-weight: bold;"
_ERR_FAULT_STYLE = f"color: {COLOR_RED}; font-weight: bold; font-size: 12pt;"

# ── module local constants ──
# The heater readout flags as "off-nominal" when |actual - nominal| / nominal
# exceeds this. Loose enough to ignore verify-tolerance jitter (~1-2 %), tight
# enough to catch a stuck reduced level (e.g. 80 % left after an interrupted
# emission Uh80 phase — where the auto-preheat 75 % gate would silently pass
# and the next test would measure a cold cathode).
HEATER_NOMINAL_TOLERANCE_PCT = 5.0


class LivePanel(QGroupBox):
    """Reusable live-parameters display panel.

    Parameters
    ----------
    title : str
        Group box title.
    sep : str
        Separator between parameter name and value, default ``": "``.
    bold_ia : bool
        If True, Ia label is bold and larger (used on Manual tab).
    layout_mode : str
        ``"compact"`` — 2-row layout (Measure tab).
        ``"grouped"`` — 4-row layout grouped by domain (Manual tab).
    """

    def __init__(
        self,
        title: str = "",
        sep: str = ": ",
        bold_ia: bool = False,
        layout_mode: str = "compact",
        parent=None,
    ):
        super().__init__(title or t("live.Live_parameters"), parent)
        self._sep = sep
        self._pa_max: float = 0.0
        self._pa_over_pct: float = 0.0
        self._pg2_max: float = 0.0
        self._pg2_over_pct: float = 0.0
        # Lamp nominal heater (0 = no reference → no off-nominal badge).
        self._nominal_uh: float = 0.0
        self._nominal_ih: float = 0.0

        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)

        # --- labels ---
        self.lbl_ua = QLabel(f"{t('common.Ua')}{sep}— {t('common.V')}")
        self.lbl_ug1 = QLabel(f"{t('common.Ug1')}{sep}— {t('common.V')}")
        self.lbl_ug2 = QLabel(f"{t('common.Ug2')}{sep}— {t('common.V')}")
        self.lbl_uh = QLabel(f"{t('common.Uh')}{sep}— {t('common.V')}")
        self.lbl_ia = QLabel(f"{t('common.Ia')}{sep}— {t('common.mA')}")
        self.lbl_ig2 = QLabel(f"{t('common.Ig2')}{sep}— {t('common.mA')}")
        self.lbl_ih = QLabel(f"{t('common.Ih')}{sep}— {t('common.A')}")
        self.lbl_an = QLabel(f"{t('common.An')}{sep}—")
        self.lbl_pa = QLabel(f"{t('common.Pa')}{sep}— {t('common.W')}")
        self.lbl_pg2 = QLabel(f"{t('common.Pg2')}{sep}— {t('common.W')}")

        # Error status label — green "OK" or red abbreviation
        self.lbl_err = QLabel(t("live.err_ok"))
        self.lbl_err.setStyleSheet(_ERR_OK_STYLE)
        self.lbl_err.setToolTip(t("live.err_ok_tooltip"))
        self._has_hw_error = False
        self._has_protection = False

        # Uh is always bold / larger
        font_uh = self.lbl_uh.font()
        font_uh.setBold(True)
        font_uh.setPointSize(font_uh.pointSize() + 2)
        self.lbl_uh.setFont(font_uh)

        if bold_ia:
            font_ia = self.lbl_ia.font()
            font_ia.setBold(True)
            font_ia.setPointSize(font_ia.pointSize() + 2)
            self.lbl_ia.setFont(font_ia)

        # --- layout ---
        if layout_mode == "compact":
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(2)
            grid.addWidget(self.lbl_ua, 0, 0)
            grid.addWidget(self.lbl_ug1, 0, 1)
            grid.addWidget(self.lbl_ug2, 0, 2)
            grid.addWidget(self.lbl_uh, 0, 3)
            grid.addWidget(self.lbl_pa, 0, 4)
            grid.addWidget(self.lbl_ia, 1, 0)
            grid.addWidget(self.lbl_an, 1, 1)
            grid.addWidget(self.lbl_ig2, 1, 2)
            grid.addWidget(self.lbl_ih, 1, 3)
            grid.addWidget(self.lbl_pg2, 1, 4)
            grid.addWidget(self.lbl_err, 1, 5)
        else:  # "grouped"
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(4)
            grid.addWidget(self.lbl_ua, 0, 0)
            grid.addWidget(self.lbl_ia, 0, 1)
            grid.addWidget(self.lbl_pa, 0, 2)
            grid.addWidget(self.lbl_err, 0, 3)
            grid.addWidget(self.lbl_ug1, 1, 0)
            grid.addWidget(self.lbl_an, 1, 1)
            grid.addWidget(self.lbl_ug2, 2, 0)
            grid.addWidget(self.lbl_ig2, 2, 1)
            grid.addWidget(self.lbl_pg2, 2, 2)
            grid.addWidget(self.lbl_uh, 3, 0)
            grid.addWidget(self.lbl_ih, 3, 1)

    # ------------------------------------------------------------------
    # Pa limits
    # ------------------------------------------------------------------

    def set_pa_limits(self, pa_max: float, pa_over_pct: float) -> None:
        """Store Pa limits so every subsequent update uses them."""
        self._pa_max = pa_max
        self._pa_over_pct = pa_over_pct

    def set_pg2_limits(self, pg2_max: float, pg2_over_pct: float) -> None:
        """Store Pg2 limits so every subsequent update uses them."""
        self._pg2_max = pg2_max
        self._pg2_over_pct = pg2_over_pct

    # ------------------------------------------------------------------
    # Heater nominal (off-nominal indicator)
    # ------------------------------------------------------------------

    def set_nominal_heater(self, uh: float, ih: float) -> None:
        """Store the lamp's nominal heater so the live readout flags when the
        actual Uh/Ih differs from standard. A stuck reduced heater (e.g. 80 %
        after an interrupted emission test) otherwise makes every subsequent
        measurement silently low; the badge makes that visible."""
        self._nominal_uh = float(uh)
        self._nominal_ih = float(ih)

    def _heater_off_pct(self, actual: float, nominal: float):
        """Integer percent of nominal when *actual* is off-nominal — i.e. a
        finite reading, with a reference (``nominal > 0``), deviating beyond the
        tolerance. ``None`` otherwise.

        Non-finite (NaN / ±inf) readings yield ``None`` so they never reach the
        ``round()`` below — which would raise (ValueError / OverflowError)
        inside the Qt live-update slot. The percent is clamped to ``>= 0`` so a
        (glitch) negative reading reads as "0 %", not a confusing "-8 %"."""
        if nominal <= 0.0 or not math.isfinite(actual):
            return None
        if abs(actual - nominal) / nominal * 100.0 <= HEATER_NOMINAL_TOLERANCE_PCT:
            return None
        return max(0, round(actual / nominal * 100.0))

    def _update_uh_label(self, uh_cal: float) -> None:
        """Set the Uh label (with an off-nominal badge + tooltip when it
        differs from the lamp nominal)."""
        pct = self._heater_off_pct(uh_cal, self._nominal_uh)
        badge = t("live.Heater_off_badge", pct=pct) if pct is not None else ""
        self.lbl_uh.setText(
            f"{t('common.Uh')}{self._sep}{format_label('uh_unit', uh_cal)}{badge}")
        self.lbl_uh.setToolTip(
            t("live.Uh_off_nominal_tooltip",
              actual=f"{uh_cal:.2f}", nominal=f"{self._nominal_uh:.2f}", pct=pct)
            if pct is not None else "")

    def _update_ih_label(self, ih_cal: float) -> None:
        """Set the Ih label (with an off-nominal badge + tooltip when it
        differs from the lamp nominal)."""
        pct = self._heater_off_pct(ih_cal, self._nominal_ih)
        badge = t("live.Heater_off_badge", pct=pct) if pct is not None else ""
        self.lbl_ih.setText(
            f"{t('common.Ih')}{self._sep}{format_label('ih_unit', ih_cal)}{badge}")
        self.lbl_ih.setToolTip(
            t("live.Ih_off_nominal_tooltip",
              actual=f"{ih_cal:.3f}", nominal=f"{self._nominal_ih:.3f}", pct=pct)
            if pct is not None else "")

    def _update_pa_label(self, pa_w: float) -> None:
        """Format Pa text and highlight colour based on stored limits."""
        s = self._sep
        pa_max = self._pa_max
        pa_over = self._pa_over_pct

        # Gate on the limit existing, NOT on over_pct>0 — pa_over_pct=0 means an
        # exact limit (no tolerance band), not "protection off" (matches the
        # scan layer, which gates only on pa_max_w>0). pa_lim==pa_max at over=0.
        if pa_max > 0:
            pa_lim = pa_max * (1.0 + pa_over / 100.0)
            self.lbl_pa.setText(f"{t('common.Pa')}{s}{format_label('pa_limit', pa_w, limit=pa_lim)}")
            if pa_w > pa_lim:
                self.lbl_pa.setStyleSheet(_PA_OVER)
            elif pa_w > pa_max:
                self.lbl_pa.setStyleSheet(_PA_WARN)
            else:
                self.lbl_pa.setStyleSheet(_PA_NORMAL)
        else:
            self.lbl_pa.setText(f"{t('common.Pa')}{s}{format_label('pa_unit', pa_w)}")
            self.lbl_pa.setStyleSheet(_PA_NORMAL)

    def _update_pg2_label(self, pg2_w: float) -> None:
        """Format Pg2 text and highlight colour based on stored limits."""
        s = self._sep
        pg2_max = self._pg2_max
        pg2_over = self._pg2_over_pct

        if pg2_max > 0:  # see _update_pa_label — pg2_over_pct=0 is an exact limit
            pg2_lim = pg2_max * (1.0 + pg2_over / 100.0)
            self.lbl_pg2.setText(f"{t('common.Pg2')}{s}{format_label('pg2_limit', pg2_w, limit=pg2_lim)}")
            if pg2_w > pg2_lim:
                self.lbl_pg2.setStyleSheet(_PA_OVER)
            elif pg2_w > pg2_max:
                self.lbl_pg2.setStyleSheet(_PA_WARN)
            else:
                self.lbl_pg2.setStyleSheet(_PA_NORMAL)
        else:
            self.lbl_pg2.setText(f"{t('common.Pg2')}{s}{format_label('pg2_unit', pg2_w)}")
            self.lbl_pg2.setStyleSheet(_PA_NORMAL)

    # ------------------------------------------------------------------
    # Hardware error indicator
    # ------------------------------------------------------------------

    def _update_err_label(self, er_raw: int) -> None:
        """Update error label + panel border from raw Er bitmask."""
        errors = decode_err(er_raw)
        if not errors:
            if self._has_hw_error:
                self.lbl_err.setText(t("live.err_ok"))
                self.lbl_err.setStyleSheet(_ERR_OK_STYLE)
                self.lbl_err.setToolTip(t("live.err_ok_tooltip"))
                self._has_hw_error = False
                self._clear_err_border()
            return
        abbrs = ", ".join(abbr for abbr, _ in errors)
        tooltip = "\n".join(t(key) for _, key in errors)
        self.lbl_err.setText(abbrs)
        self.lbl_err.setStyleSheet(_ERR_FAULT_STYLE)
        self.lbl_err.setToolTip(tooltip)
        if not self._has_hw_error:
            self._has_hw_error = True
            self._apply_err_border()

    def _apply_err_border(self) -> None:
        self.setStyleSheet(self._PROTECTION_STYLE)

    def _clear_err_border(self) -> None:
        if not self._has_protection:
            self.setStyleSheet("")

    # ------------------------------------------------------------------
    # Public update helpers
    # ------------------------------------------------------------------

    def update_values(self, data: dict, calibration: CalibrationData) -> float:
        """Update labels from raw poller *data*.  Returns Pa (watts).

        Poller data contains raw protocol integers — decode + calibrate here.
        """
        s = self._sep
        ua_cal = calibration.apply_read("ua", float(data["ua"]))
        ug1_cal = calibration.apply_read("ug1", decode_ug1(data["ug1"]))
        ug2_cal = calibration.apply_read("ug2", float(data["ug2"]))
        uh_cal = calibration.apply_read("uh", decode_uh(data["uh"]))
        ih_cal = calibration.apply_read("ih", decode_ih(data["ih"]))
        ia_cal = calibration.apply_read("ia", decode_ia(data["ia"]))
        ig2_cal = calibration.apply_read("ig2", decode_ig2(data["ig2"]))

        self.lbl_ua.setText(f"{t('common.Ua')}{s}{format_label('ua_unit', ua_cal)}")
        self.lbl_ug1.setText(f"{t('common.Ug1')}{s}{format_label('ug1_unit', ug1_cal)}")
        self.lbl_ug2.setText(f"{t('common.Ug2')}{s}{format_label('ug2_unit', ug2_cal)}")
        self._update_uh_label(uh_cal)
        self.lbl_ia.setText(f"{t('common.Ia')}{s}{format_label('ia_unit', ia_cal)}")
        self.lbl_ig2.setText(f"{t('common.Ig2')}{s}{format_label('ig2_unit', ig2_cal)}")
        self._update_ih_label(ih_cal)
        self.lbl_an.setText(f"{t('common.An')}{s}{data['an']}")

        self._update_err_label(data.get("er", 0))

        pa_w = ua_cal * ia_cal / 1000.0
        self._update_pa_label(pa_w)

        pg2_w = ug2_cal * ig2_cal / 1000.0
        self._update_pg2_label(pg2_w)
        return pa_w

    def update_from_point(self, point: dict) -> float:
        """Update labels from already-decoded and calibrated *point* dict.  Returns Pa.

        Heater keys are optional. A partial point carries no uh/ih, and
        defaulting them to 0.0 would render a false "heater off" reading
        indistinguishable from a real one — an absent key leaves the
        heater labels at their last known value instead.
        """
        s = self._sep
        ia_val = point.get("ia", 0.0)

        self.lbl_ua.setText(f"{t('common.Ua')}{s}{format_label('ua_unit', point.get('ua', 0))}")
        self.lbl_ug1.setText(f"{t('common.Ug1')}{s}{format_label('ug1_unit', point.get('ug1', 0.0))}")
        self.lbl_ug2.setText(f"{t('common.Ug2')}{s}{format_label('ug2_unit', point.get('ug2', 0))}")
        uh_val = point.get("uh")
        if isinstance(uh_val, (int, float)):
            self._update_uh_label(float(uh_val))
        self.lbl_ia.setText(f"{t('common.Ia')}{s}{format_label('ia_unit', ia_val)}")
        self.lbl_ig2.setText(f"{t('common.Ig2')}{s}{format_label('ig2_unit', point.get('ig2', 0.0))}")
        ih_val = point.get("ih")
        if isinstance(ih_val, (int, float)):
            self._update_ih_label(float(ih_val))

        pa_w = point.get("ua", 0.0) * ia_val / 1000.0
        self._update_pa_label(pa_w)

        pg2_w = point.get("ug2", 0.0) * point.get("ig2", 0.0) / 1000.0
        self._update_pg2_label(pg2_w)
        return pa_w

    def set_an(self, an_value) -> None:
        """Update only the An label."""
        self.lbl_an.setText(f"{t('common.An')}{self._sep}{an_value}")

    # ------------------------------------------------------------------
    # Protection indicator
    # ------------------------------------------------------------------

    _PROTECTION_STYLE = (
        f"QGroupBox {{ border: 2px solid {COLOR_RED}; }}"
        f"QGroupBox::title {{ color: {COLOR_RED}; }}"
    )

    def show_protection(self, param: str = "") -> None:
        """Activate red protection indicator on the panel."""
        title = self.title()
        tag = f" {t('live.Protection_tag')}"
        if tag not in title:
            self.setTitle(title + tag)
        self._has_protection = True
        self.setStyleSheet(self._PROTECTION_STYLE)

    def clear_protection(self) -> None:
        """Remove protection indicator."""
        title = self.title()
        tag = f" {t('live.Protection_tag')}"
        if tag in title:
            self.setTitle(title.replace(tag, ""))
        self._has_protection = False
        if not self._has_hw_error:
            self.setStyleSheet("")
