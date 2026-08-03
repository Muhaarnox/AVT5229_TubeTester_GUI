import logging
import time
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from serial import SerialException

from lm19.config import LampConfig
from lm19.protocol import (
    LM19Serial,
    decode_ia,
    decode_ig2,
    decode_ih,
    decode_ug1,
    decode_uh,
    encode_ih,
    encode_uh,
    encode_ug1,
)
from app.lamp_panel import read_heater_level
from app.live_panel import HEATER_NOMINAL_TOLERANCE_PCT, LivePanel
from app.ui_theme import (
    COLOR_IA, COLOR_IG2, COLOR_ORANGE, MANUAL_WARN_COL_WIDTH,
)
from app.widget_factory import make_double_spinbox, make_int_spinbox
from i18n_setup import t
from lm19.constants import HEATER_NEAR_ZERO_A, HEATER_NEAR_ZERO_V
from lm19.plot_style import DEFAULT_GRID_ALPHA


# ── module local constants ──
# Inline off-nominal marker next to a heater setpoint field.
_HEATER_WARN_GLYPH = "⚠"


class ManualTab(QWidget):
    """Manual control tab for setting parameters and taking single point measurements."""

    def __init__(
        self,
        get_client: Callable[[], Optional[LM19Serial]],
        get_app_config: Callable,
        get_calibration: Callable,
        get_write_locked: Callable[[], bool],
        on_add_to_main_plot: Callable[[List[Dict]], None],
        get_hw_busy: Callable[[], Optional[str]] = lambda: None,
        on_save: Callable[[List[Dict], bool], None] = lambda pts, inc: None,
        get_current_lamp: Callable[[], Optional[LampConfig]] = lambda: None,
        parent=None,
    ):
        super().__init__(parent)
        self.get_client = get_client
        self.get_calibration = get_calibration
        self.get_write_locked = get_write_locked
        self.get_hw_busy = get_hw_busy
        self.get_app_config = get_app_config
        self.on_add_to_main_plot = on_add_to_main_plot
        self.on_save = on_save
        self.get_current_lamp = get_current_lamp
        self.manual_points: List[Dict] = []
        self.ia_start_time: Optional[float] = None
        self.ia_data: List[float] = []
        self.ig2_data: List[float] = []
        self.ia_time: List[float] = []
        self._auto_debounce: Dict[str, QTimer] = {}
        self._build_ui()
        self._setup_auto_debounce()
        self.uh_spin.valueChanged.connect(
            lambda _v: self.refresh_heater_setpoint_warnings())
        self.ih_spin.valueChanged.connect(
            lambda _v: self.refresh_heater_setpoint_warnings())
        self.refresh_heater_setpoint_warnings()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        left.addWidget(self._build_set_values_group())
        self.live_panel = LivePanel(
            title=t('manual.Live_Readings'), sep=": ", bold_ia=True, layout_mode="grouped"
        )
        left.addWidget(self.live_panel)
        left.addWidget(self._build_point_collection_group())
        left.addWidget(self._build_chart_controls_group())
        left.addStretch(1)
        right.addWidget(self._build_realtime_chart())

        layout.addLayout(left, 0)
        layout.addLayout(right, 1)

    def _build_set_values_group(self) -> QGroupBox:
        set_box = QGroupBox(t('manual.Set_Values'))
        set_layout = QFormLayout(set_box)

        self.ua_spin = make_double_spinbox(
            min_val=0, max_val=300, value=100,
            decimals=1, tooltip_key="tip.Manual_ua_target",
        )
        self.ua_auto_cb = QCheckBox(t('manual.Auto'))
        self.ua_auto_cb.setToolTip(t("tip.Manual_auto_apply"))
        ua_row = QHBoxLayout()
        ua_row.addWidget(self.ua_spin)
        self.ua_warn_slot = self._make_warn_slot()
        ua_row.addWidget(self.ua_warn_slot)
        ua_row.addWidget(self.ua_auto_cb)
        self.ua_btn = QPushButton(t('manual.Set'))
        self.ua_btn.setToolTip(t("tip.Manual_set_single"))
        self.ua_btn.clicked.connect(self._set_ua)
        ua_row.addWidget(self.ua_btn)
        set_layout.addRow(t('common.Label_unit_colon', label=t('common.Ua'), unit=t('common.V')), ua_row)

        self.ug1_spin = make_double_spinbox(
            min_val=-50, max_val=0, value=-5,
            step=0.1, decimals=2,
            tooltip_key="tip.Manual_ug1_target",
        )
        self.ug1_auto_cb = QCheckBox(t('manual.Auto'))
        self.ug1_auto_cb.setToolTip(t("tip.Manual_auto_apply"))
        ug1_row = QHBoxLayout()
        ug1_row.addWidget(self.ug1_spin)
        self.ug1_warn_slot = self._make_warn_slot()
        ug1_row.addWidget(self.ug1_warn_slot)
        ug1_row.addWidget(self.ug1_auto_cb)
        self.ug1_btn = QPushButton(t('manual.Set'))
        self.ug1_btn.setToolTip(t("tip.Manual_set_single"))
        self.ug1_btn.clicked.connect(self._set_ug1)
        ug1_row.addWidget(self.ug1_btn)
        set_layout.addRow(t('common.Label_unit_colon', label=t('common.Ug1'), unit=t('common.V')), ug1_row)

        self.ug2_spin = make_double_spinbox(
            min_val=0, max_val=300, value=100,
            decimals=1, tooltip_key="tip.Manual_ug2_target",
        )
        self.ug2_auto_cb = QCheckBox(t('manual.Auto'))
        self.ug2_auto_cb.setToolTip(t("tip.Manual_auto_apply"))
        ug2_row = QHBoxLayout()
        ug2_row.addWidget(self.ug2_spin)
        self.ug2_warn_slot = self._make_warn_slot()
        ug2_row.addWidget(self.ug2_warn_slot)
        ug2_row.addWidget(self.ug2_auto_cb)
        self.ug2_btn = QPushButton(t('manual.Set'))
        self.ug2_btn.setToolTip(t("tip.Manual_set_single"))
        self.ug2_btn.clicked.connect(self._set_ug2)
        ug2_row.addWidget(self.ug2_btn)
        set_layout.addRow(t('common.Label_unit_colon', label=t('common.Ug2'), unit=t('common.V')), ug2_row)

        # Heater mode radio: Uh (voltage) vs Ih (current) — mutually exclusive in firmware
        self.heater_mode_group = QButtonGroup(self)
        self.heater_uh_radio = QRadioButton(
            t('common.Label_unit_colon', label=t('common.Uh'), unit=t('common.V'))
        )
        self.heater_ih_radio = QRadioButton(
            t('common.Label_unit_colon', label=t('common.Ih'), unit=t('common.A'))
        )
        self.heater_uh_radio.setToolTip(t("tip.Heater_mode_uh"))
        self.heater_ih_radio.setToolTip(t("tip.Heater_mode_ih"))
        self.heater_uh_radio.setChecked(True)
        self.heater_mode_group.addButton(self.heater_uh_radio, 0)
        self.heater_mode_group.addButton(self.heater_ih_radio, 1)
        self.heater_mode_group.idToggled.connect(self._on_heater_mode_changed)

        # Manual tab is DELIBERATELY not clamped to per-lamp uh_max/ih_max
        # (ML-130): the scan path enforces the lamp-card caps, the manual
        # tab is the operator's escape hatch for rejuvenation/experiments
        # — device limits only. Do not "fix" this by wiring lamp.limits
        # here.
        self.uh_spin = make_double_spinbox(
            min_val=0, max_val=20, value=6.3,
            step=0.1, decimals=2,
            tooltip_key="tip.Manual_uh_target",
        )
        self.uh_auto_cb = QCheckBox(t('manual.Auto'))
        self.uh_auto_cb.setToolTip(t("tip.Manual_auto_apply"))
        uh_row = QHBoxLayout()
        uh_row.addWidget(self.uh_spin)
        self.uh_warn_lbl = self._make_heater_warn_label()
        uh_row.addWidget(self.uh_warn_lbl)
        uh_row.addWidget(self.uh_auto_cb)
        self.uh_btn = QPushButton(t('manual.Set'))
        self.uh_btn.setToolTip(t("tip.Manual_set_single"))
        self.uh_btn.clicked.connect(self._set_uh)
        uh_row.addWidget(self.uh_btn)
        set_layout.addRow(self.heater_uh_radio, uh_row)

        self.ih_spin = make_double_spinbox(
            min_val=0, max_val=5, value=0,
            decimals=3, tooltip_key="tip.Manual_ih_target",
        )
        self.ih_auto_cb = QCheckBox(t('manual.Auto'))
        self.ih_auto_cb.setToolTip(t("tip.Manual_auto_apply"))
        ih_row = QHBoxLayout()
        ih_row.addWidget(self.ih_spin)
        self.ih_warn_lbl = self._make_heater_warn_label()
        ih_row.addWidget(self.ih_warn_lbl)
        ih_row.addWidget(self.ih_auto_cb)
        self.ih_btn = QPushButton(t('manual.Set'))
        self.ih_btn.setToolTip(t("tip.Manual_set_single"))
        self.ih_btn.clicked.connect(self._set_ih)
        ih_row.addWidget(self.ih_btn)
        set_layout.addRow(self.heater_ih_radio, ih_row)

        # Disable Ih controls initially (Uh mode is default)
        self._set_heater_controls_enabled(ih_mode=False)

        self.anode_group = QButtonGroup(self)
        self.anode_1 = QRadioButton("1")
        self.anode_2 = QRadioButton("2")
        self.anode_1.setChecked(True)
        self.anode_group.addButton(self.anode_1, 1)
        self.anode_group.addButton(self.anode_2, 2)
        self.anode_1.setToolTip(t("tip.Anode_select"))
        self.anode_2.setToolTip(t("tip.Anode_select"))
        an_row = QHBoxLayout()
        an_row.addWidget(self.anode_1)
        an_row.addWidget(self.anode_2)
        self.an_set_btn = QPushButton(t('manual.Set'))
        self.an_set_btn.setToolTip(t("tip.Manual_set_an"))
        self.an_set_btn.clicked.connect(self._set_an)
        an_row.addWidget(self.an_set_btn)
        an_row.addStretch(1)
        set_layout.addRow(t("lamp.Anode"), an_row)

        btn_row = QHBoxLayout()
        self.apply_all_btn = QPushButton(t('manual.Apply_All'))
        self.apply_all_btn.setToolTip(t("tip.Manual_apply_all"))
        self.apply_all_btn.clicked.connect(self._apply_all)
        self.read_all_btn = QPushButton(t('manual.Read_All'))
        self.read_all_btn.setToolTip(t('tip.Read_all'))
        self.read_all_btn.clicked.connect(self._read_all)
        self.load_lamp_btn = QPushButton(t('manual.From_Lamp'))
        self.load_lamp_btn.setToolTip(t('tip.From_lamp'))
        self.load_lamp_btn.clicked.connect(self._load_from_lamp)
        self.reset_hv_btn = QPushButton(t('manual.Reset_HV'))
        self.reset_hv_btn.setToolTip(t('tip.Reset_HV'))
        self.reset_hv_btn.clicked.connect(self._reset_hv)
        self.reset_all_btn = QPushButton(t('manual.Reset_All'))
        self.reset_all_btn.setToolTip(t("tip.Manual_reset_all"))
        self.reset_all_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(self.apply_all_btn)
        btn_row.addWidget(self.read_all_btn)
        btn_row.addWidget(self.load_lamp_btn)
        btn_row.addWidget(self.reset_hv_btn)
        btn_row.addWidget(self.reset_all_btn)
        set_layout.addRow(btn_row)

        return set_box

    def _build_point_collection_group(self) -> QGroupBox:
        point_box = QGroupBox(t('manual.Single_Point'))
        point_layout = QVBoxLayout(point_box)

        point_btn_row = QHBoxLayout()
        self.take_point_btn = QPushButton(t('manual.Take_Point'))
        self.take_point_btn.setToolTip(t("tip.Manual_take_point"))
        self.take_point_btn.clicked.connect(self._take_point)
        self.add_to_plot_btn = QPushButton(t('manual.Add_to_Main_Plot'))
        self.add_to_plot_btn.setToolTip(t("tip.Manual_add_to_plot"))
        self.add_to_plot_btn.clicked.connect(self._add_to_main_plot)
        self.clear_points_btn = QPushButton(t('manual.Clear_Points'))
        self.clear_points_btn.setToolTip(t("tip.Manual_clear_points"))
        self.clear_points_btn.clicked.connect(self._clear_points)
        point_btn_row.addWidget(self.take_point_btn)
        point_btn_row.addWidget(self.add_to_plot_btn)
        point_btn_row.addWidget(self.clear_points_btn)
        point_layout.addLayout(point_btn_row)

        save_row = QHBoxLayout()
        self.copy_points_btn = QPushButton(t('manual.Copy'))
        self.copy_points_btn.setToolTip(t('tip.Copy_points'))
        self.copy_points_btn.clicked.connect(self._copy_points_to_clipboard)
        self.save_btn = QPushButton(t('manual.Save'))
        self.save_btn.setToolTip(t('tip.Save_manual'))
        self.save_btn.clicked.connect(self._save)
        self.include_points_cb = QCheckBox(t('manual.Include_points'))
        self.include_points_cb.setChecked(True)
        self.include_points_cb.setToolTip(t('tip.Include_points'))
        save_row.addWidget(self.copy_points_btn)
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.include_points_cb)
        save_row.addStretch(1)
        point_layout.addLayout(save_row)

        self.points_label = QLabel(t('manual.Points_taken', count=0))
        point_layout.addWidget(self.points_label)

        # Table for taken points
        self.points_table = QTableWidget(0, 5)
        self.points_table.setHorizontalHeaderLabels([
            t('common.Ua'),
            t('common.Ug1'),
            t('common.Ug2'),
            t('common.Ig2'),
            t('common.Ia'),
        ])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.setMaximumHeight(150)
        point_layout.addWidget(self.points_table)

        return point_box

    def _build_chart_controls_group(self) -> QGroupBox:
        chart_ctrl_box = QGroupBox(t('manual.Chart_Controls'))
        chart_ctrl_grid = QGridLayout(chart_ctrl_box)
        chart_ctrl_grid.setContentsMargins(6, 6, 6, 6)
        chart_ctrl_grid.setHorizontalSpacing(6)

        self._ia_color = QColor(COLOR_IA)
        self._ig2_color = QColor(COLOR_IG2)

        # Ia line settings
        self.ia_visible_cb = QCheckBox(t('common.Label_colon', label=t('common.Ia')))
        self.ia_visible_cb.setChecked(True)
        self.ia_visible_cb.toggled.connect(lambda on: self._toggle_curve("ia", on))
        chart_ctrl_grid.addWidget(self.ia_visible_cb, 0, 0)
        self.ia_color_btn = QPushButton()
        self.ia_color_btn.setFixedSize(24, 24)
        self.ia_color_btn.setStyleSheet(f"background-color: {self._ia_color.name()};")
        self.ia_color_btn.setToolTip(t('tip.Ia_line_color'))
        self.ia_color_btn.clicked.connect(lambda: self._pick_color("ia"))
        chart_ctrl_grid.addWidget(self.ia_color_btn, 0, 1)
        self.ia_width_spin = make_int_spinbox(
            min_val=1, max_val=6, value=2,
            tooltip_key="tip.Ia_line_width",
            on_change=lambda _v: self._apply_pen("ia"),
        )
        chart_ctrl_grid.addWidget(self.ia_width_spin, 0, 2)

        # Ig2 line settings
        self.ig2_visible_cb = QCheckBox(t('common.Label_colon', label=t('common.Ig2')))
        self.ig2_visible_cb.setChecked(True)
        self.ig2_visible_cb.toggled.connect(lambda on: self._toggle_curve("ig2", on))
        chart_ctrl_grid.addWidget(self.ig2_visible_cb, 0, 3)
        self.ig2_color_btn = QPushButton()
        self.ig2_color_btn.setFixedSize(24, 24)
        self.ig2_color_btn.setStyleSheet(f"background-color: {self._ig2_color.name()};")
        self.ig2_color_btn.setToolTip(t('tip.Ig2_line_color'))
        self.ig2_color_btn.clicked.connect(lambda: self._pick_color("ig2"))
        chart_ctrl_grid.addWidget(self.ig2_color_btn, 0, 4)
        self.ig2_width_spin = make_int_spinbox(
            min_val=1, max_val=6, value=2,
            tooltip_key="tip.Ig2_line_width",
            on_change=lambda _v: self._apply_pen("ig2"),
        )
        chart_ctrl_grid.addWidget(self.ig2_width_spin, 0, 5)

        # Y from 0 checkbox
        self.y_from_zero_cb = QCheckBox(t('manual.Y_from_0'))
        self.y_from_zero_cb.setChecked(True)
        self.y_from_zero_cb.setToolTip(t('tip.Y_from_0'))
        self.y_from_zero_cb.toggled.connect(self._apply_y_from_zero)
        chart_ctrl_grid.addWidget(self.y_from_zero_cb, 0, 6)

        # Reset button
        self.reset_chart_btn = QPushButton(t('manual.Reset_Chart'))
        self.reset_chart_btn.setToolTip(t("tip.Manual_reset_chart"))
        self.reset_chart_btn.clicked.connect(self._reset_ia_chart)
        chart_ctrl_grid.addWidget(self.reset_chart_btn, 0, 7)

        chart_ctrl_grid.setColumnStretch(8, 1)
        return chart_ctrl_box

    def _build_realtime_chart(self) -> pg.PlotWidget:
        self.ia_plot = pg.PlotWidget(title=t('plot.Ia_Ig2_realtime'))
        self.ia_plot.setLabel("left", t('plot.Ia_mA'), color=COLOR_IA)
        self.ia_plot.setLabel("bottom", t('plot.Time_s'))
        self.ia_plot.showGrid(x=True, y=True, alpha=DEFAULT_GRID_ALPHA)
        self.ia_curve = self.ia_plot.plot([], [], pen=pg.mkPen(COLOR_IA, width=2), name=t("common.Ia"))

        # Second Y axis for Ig2
        self.ig2_vb = pg.ViewBox()
        self.ia_plot.scene().addItem(self.ig2_vb)
        self.ia_plot.getAxis("right").linkToView(self.ig2_vb)
        self.ig2_vb.setXLink(self.ia_plot)
        self.ia_plot.showAxis("right")
        self.ia_plot.getAxis("right").setLabel(t('plot.Ig2_mA'), color=COLOR_IG2)
        self.ig2_curve = pg.PlotCurveItem(pen=pg.mkPen(COLOR_IG2, width=2))
        self.ig2_vb.addItem(self.ig2_curve)

        # Keep Ig2 view synced on resize
        self.ia_plot.getViewBox().sigResized.connect(self._sync_ig2_view)

        # Disable Y auto-range initially (Y from 0 is on by default)
        self.ia_plot.enableAutoRange(axis="y", enable=False)
        self.ig2_vb.enableAutoRange(axis="y", enable=False)

        # Legend
        legend = self.ia_plot.addLegend(offset=(60, 10))
        legend.addItem(self.ia_curve, t('common.Ia'))
        legend.addItem(self.ig2_curve, t('common.Ig2'))

        return self.ia_plot

    def _get_client_or_warn(self) -> Optional[LM19Serial]:
        client = self.get_client()
        if not client or not client.is_open():
            QMessageBox.warning(self, t('msg.COM'), t('msg.Connect_first'))
            return None
        return client

    def _can_write_or_warn(self) -> bool:
        if self.get_write_locked():
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Emergency_write_block'))
            return False
        busy = self.get_hw_busy()
        if busy:
            log.warning("Manual write blocked — hardware busy: %s", busy)
            QMessageBox.warning(self, t('msg.Reset_all_title'), t('msg.Hw_busy'))
            return False
        return True

    def _set_an(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        client.set_param("An", int(self.anode_group.checkedId()))

    def _set_ua(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        # Working-point command → SET feedforward (plan B).
        value = int(round(self.get_calibration().apply_set(
            "ua", self.ua_spin.value())))
        client.set_param("Ua", value)

    def _set_ug1(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        # Spinbox holds negative volts — same canonical domain as
        # apply_set("ug1", …), so the value passes through unchanged.
        value = self.get_calibration().apply_set("ug1", self.ug1_spin.value())
        client.set_param("Ug1", encode_ug1(value))

    def _set_ug2(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        value = int(round(self.get_calibration().apply_set(
            "ug2", self.ug2_spin.value())))
        client.set_param("Ug2", value)

    def _set_uh(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        client.set_param("Uh", encode_uh(self.get_calibration().apply_set(
            "uh", self.uh_spin.value())))
        # Zeroing the inactive heater channel stays raw:
        # apply_set(0) == offset could command a non-zero value.
        client.set_param("Ih", 0)

    def _set_ih(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        client.set_param("Ih", encode_ih(self.get_calibration().apply_set(
            "ih", self.ih_spin.value())))
        client.set_param("Uh", 0)  # raw zero — see _set_uh

    @staticmethod
    def _make_warn_slot(glyph: str = "") -> QLabel:
        """Fixed-width slot right of a setpoint spinbox.

        Every Set-values row gets one, and the heater ones keep their size
        while hidden: a marker that appeared or cleared would otherwise
        re-flow its row and break the column alignment of the controls
        next to it.
        """
        lbl = QLabel(glyph)
        lbl.setFixedWidth(MANUAL_WARN_COL_WIDTH)
        policy = lbl.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        lbl.setSizePolicy(policy)
        if glyph:
            lbl.setStyleSheet(f"color: {COLOR_ORANGE};")
            lbl.setVisible(False)
        return lbl

    def _make_heater_warn_label(self) -> QLabel:
        """Inline off-nominal marker shown next to a heater spinbox."""
        return self._make_warn_slot(_HEATER_WARN_GLYPH)

    def refresh_heater_setpoint_warnings(self) -> None:
        """Flag heater setpoints that are off the selected lamp's rating.

        Advisory only, and deliberately tighter than the Apply All gate
        (``HEATER_NOMINAL_TOLERANCE_PCT`` — the same band that drives the
        live-panel badge — against the configurable
        ``manual_heater_tolerance_pct`` that blocks): the marker appears
        while the value is still merely questionable, the dialog only when
        it is far enough out to be worth interrupting for.
        """
        lamp = self.get_current_lamp() if callable(self.get_current_lamp) else None
        for spin, label, nominal, other_nominal, unit in (
            (self.uh_spin, self.uh_warn_lbl,
             lamp.uh if lamp else 0.0, lamp.ih if lamp else 0.0,
             t("common.V")),
            (self.ih_spin, self.ih_warn_lbl,
             lamp.ih if lamp else 0.0, lamp.uh if lamp else 0.0,
             t("common.A")),
        ):
            value = spin.value()
            if lamp is None:
                label.setVisible(False)
                continue
            if nominal <= 0:
                # No rating on this channel: only flag a value the user
                # actually dialled in, and only when the lamp is heated
                # through the other channel.
                off = value > 0 and other_nominal > 0
                tip = t("manual.Heater_warn_channel_tip", tube=lamp.tube_type)
            else:
                off = self._off_by_more_than_tolerance(
                    value, nominal, HEATER_NOMINAL_TOLERANCE_PCT)
                tip = t("manual.Heater_warn_tip",
                        nominal=f"{nominal:g}", unit=unit,
                        tube=lamp.tube_type,
                        pct=str(int(HEATER_NOMINAL_TOLERANCE_PCT)))
            label.setToolTip(tip if off else "")
            label.setVisible(off)

    def _on_heater_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        self._set_heater_controls_enabled(ih_mode=(id_ == 1))

    def _set_heater_controls_enabled(self, ih_mode: bool) -> None:
        for w in (self.uh_spin, self.uh_auto_cb, self.uh_btn):
            w.setEnabled(not ih_mode)
        for w in (self.ih_spin, self.ih_auto_cb, self.ih_btn):
            w.setEnabled(ih_mode)

    def _read_all(self) -> None:
        """Read current parameter values from device and populate spin boxes."""
        client = self._get_client_or_warn()
        if not client:
            return
        try:
            # Physical (calibrated) values — the spinboxes feed apply_set
            # later, so both sides must live in the same domain. Ug1 stays
            # negative end-to-end (decode_ug1 → apply_read → spinbox).
            cal = self.get_calibration()
            self.ua_spin.setValue(cal.apply_read(
                "ua", float(client.get_param("Ua", real=True))))
            self.ug1_spin.setValue(cal.apply_read(
                "ug1", decode_ug1(client.get_param("Ug1", real=True))))
            self.ug2_spin.setValue(cal.apply_read(
                "ug2", float(client.get_param("Ug2", real=True))))
            self.uh_spin.setValue(cal.apply_read(
                "uh", decode_uh(client.get_param("Uh", real=True))))
            self.ih_spin.setValue(cal.apply_read(
                "ih", decode_ih(client.get_param("Ih", real=True))))
        except Exception as exc:
            log.exception("Failed to read params from device")
            QMessageBox.warning(self, t('msg.Error'), t('msg.Failed_to_read_params', error=exc))

    def _load_from_lamp(self) -> None:
        """Load nominal parameters from the currently selected lamp into spin boxes."""
        lamp = self.get_current_lamp()
        if not lamp:
            QMessageBox.warning(self, t('msg.Lamp'), t('msg.No_lamp_selected'))
            return
        self.ua_spin.setValue(lamp.ua)
        self.ug1_spin.setValue(lamp.ug1)
        self.ug2_spin.setValue(lamp.ug2)
        self.uh_spin.setValue(lamp.uh)
        self.ih_spin.setValue(lamp.ih)
        # Auto-select heater mode based on lamp config
        if lamp.ih > 0:
            self.heater_ih_radio.setChecked(True)
        else:
            self.heater_uh_radio.setChecked(True)

    def load_from_conditions(self, conditions: Dict) -> None:
        """Load spinbox values from a measurement conditions dict."""
        if "ua" in conditions:
            self.ua_spin.setValue(float(conditions["ua"]))
        if "ug1" in conditions:
            self.ug1_spin.setValue(float(conditions["ug1"]))
        if "ug2" in conditions:
            self.ug2_spin.setValue(float(conditions["ug2"]))
        if "uh" in conditions:
            self.uh_spin.setValue(float(conditions["uh"]))
        an = conditions.get("an")
        if an is not None:
            btn = self.anode_group.button(int(an))
            if btn:
                btn.setChecked(True)

    def _setup_auto_debounce(self) -> None:
        """Create debounce timers and connect spinbox valueChanged to auto-apply."""
        params = {
            "ua":  (self.ua_spin,  self.ua_auto_cb,  self._set_ua),
            "ug1": (self.ug1_spin, self.ug1_auto_cb, self._set_ug1),
            "ug2": (self.ug2_spin, self.ug2_auto_cb, self._set_ug2),
            "uh":  (self.uh_spin,  self.uh_auto_cb,  self._set_uh),
            "ih":  (self.ih_spin,  self.ih_auto_cb,  self._set_ih),
        }
        for key, (spin, cb, setter) in params.items():
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(300)
            timer.timeout.connect(setter)
            self._auto_debounce[key] = timer
            spin.valueChanged.connect(lambda _v, k=key, c=cb: self._on_auto_value_changed(k, c))

    def _on_auto_value_changed(self, key: str, cb: QCheckBox) -> None:
        """Restart debounce timer if the corresponding Auto checkbox is checked."""
        if cb.isChecked():
            self._auto_debounce[key].start()

    @staticmethod
    def _off_by_more_than_tolerance(value: float, reference: float,
                                    tolerance_pct: float) -> bool:
        """True when *value* deviates from *reference* beyond the band.

        Both directions count: an over-driven heater is no more "ready"
        than a cold one.
        """
        if reference <= 0:
            return True
        return abs(value - reference) / reference * 100.0 > tolerance_pct

    def _heater_setpoint_reasons(self, use_uh: bool, target: float,
                                 unit: str, fmt: str,
                                 tolerance_pct: float) -> List[str]:
        """Reasons why the heater SETPOINT itself is not a normal operating
        point for the selected lamp.

        Checking the live reading against the setpoint alone cannot catch
        this: a 1 V setpoint reached exactly is "on target" and still
        starves a 6.3 V cathode. Without a selected lamp there is no
        nominal to compare against, so this contributes nothing.
        """
        lamp = self.get_current_lamp() if callable(self.get_current_lamp) else None
        if lamp is None:
            return []
        nominal = lamp.uh if use_uh else lamp.ih
        other_nominal = lamp.ih if use_uh else lamp.uh
        if nominal <= 0:
            if other_nominal > 0:
                # Voltage-driving a current-heated lamp (or vice versa):
                # the card gives no nominal for the selected channel.
                return [t("manual.Heater_reason_channel_mismatch",
                          tube=lamp.tube_type,
                          channel=t("common.Uh") if use_uh else t("common.Ih"),
                          nominal=t("common.Ih") if use_uh else t("common.Uh"))]
            return []
        if target > 0 and not self._off_by_more_than_tolerance(
                target, nominal, tolerance_pct):
            return []
        return [t("manual.Heater_reason_setpoint_off_lamp",
                  target=fmt % target, nominal=fmt % nominal, unit=unit,
                  tube=lamp.tube_type,
                  pct=str(max(0, round(target / nominal * 100.0))))]

    def _heater_ready_or_confirmed(self, client) -> bool:
        """Gate Apply All on the heater being at a sane operating point.

        Apply All commands the heater and the HV in one shot, so on a cold
        tube the anode voltage lands on an unemissive cathode: readings are
        meaningless and the cathode takes ion bombardment. Two independent
        things must hold, and each is reported on its own line:

        * the SETPOINT is a normal point for the selected lamp — a 1 V
          setpoint held perfectly still starves a 6.3 V cathode;
        * the tube is actually AT that setpoint — a deliberate reduced-
          heater experiment is legitimate, but only once it has settled.

        Returns True when Apply All may proceed.
        """
        use_uh = not self.heater_ih_radio.isChecked()
        target = self.uh_spin.value() if use_uh else self.ih_spin.value()
        unit = t("common.V") if use_uh else t("common.A")
        fmt = "%.2f" if use_uh else "%.3f"
        try:
            now = read_heater_level(client, use_uh, self.get_calibration())
        except (OSError, ValueError, RuntimeError, TimeoutError,
                SerialException) as exc:
            # Cannot confirm the heater state → apply nothing. Silently
            # commanding HV on an unknown heater is the failure this gate
            # exists to prevent.
            log.warning("Apply All blocked — heater read failed: %s", exc)
            QMessageBox.warning(self, t("msg.Heater_warning"),
                                t("manual.Heater_read_failed", error=str(exc)))
            return False

        tolerance_pct = float(
            self.get_app_config().manual_heater_tolerance_pct)
        reasons = self._heater_setpoint_reasons(
            use_uh, target, unit, fmt, tolerance_pct)
        near_zero = HEATER_NEAR_ZERO_V if use_uh else HEATER_NEAR_ZERO_A
        if target <= 0:
            reasons.append(t("manual.Heater_reason_setpoint_zero",
                             now=fmt % now, unit=unit))
        elif now < near_zero:
            reasons.append(t("manual.Heater_reason_off",
                             now=fmt % now, target=fmt % target, unit=unit))
        elif self._off_by_more_than_tolerance(now, target, tolerance_pct):
            reasons.append(t("manual.Heater_reason_off_setpoint",
                             now=fmt % now, target=fmt % target, unit=unit,
                             pct=str(max(0, round(now / target * 100.0)))))
        if not reasons:
            return True

        log.warning("Apply All on off-nominal heater: now=%.3f %s, "
                    "setpoint=%.3f %s — asking for confirmation (%d reason(s))",
                    now, unit, target, unit, len(reasons))
        question = "\n\n".join(reasons + [t("manual.Heater_apply_anyway")])
        reply = QMessageBox.question(
            self, t("msg.Heater_warning"), question,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _apply_all(self) -> None:
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        if not self._heater_ready_or_confirmed(client):
            return
        # Safe order: An, Heater (active first then zero inactive to avoid
        # PWM dropout), drop HV before changing Ug1 to prevent Ia surge
        # if bias becomes less negative, then reapply Ua/Ug2.
        # Working points get SET feedforward (plan B); zeros and the An
        # selector stay raw — apply_set(0) == offset could command a
        # non-zero voltage during the safety drop.
        cal = self.get_calibration()
        client.set_param("An", int(self.anode_group.checkedId()))
        if self.heater_ih_radio.isChecked():
            client.set_param("Ih", encode_ih(cal.apply_set("ih", self.ih_spin.value())))
            client.set_param("Uh", 0)
        else:
            client.set_param("Uh", encode_uh(cal.apply_set("uh", self.uh_spin.value())))
            client.set_param("Ih", 0)
        client.set_param("Ug2", 0)
        client.set_param("Ua", 0)
        client.set_param("Ug1", encode_ug1(cal.apply_set("ug1", self.ug1_spin.value())))
        client.set_param("Ua", int(round(cal.apply_set("ua", self.ua_spin.value()))))
        client.set_param("Ug2", int(round(cal.apply_set("ug2", self.ug2_spin.value()))))

    def _reset_hv(self) -> None:
        """Reset only Ua, Ug1, Ug2 (keep heater running)."""
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        # Reset in safe order: Ug2, Ug1, Ua
        ug1_after_stop = self.get_app_config().ug1_after_stop
        client.set_param("Ug2", 0)
        client.set_param("Ug1", encode_ug1(ug1_after_stop))
        client.set_param("Ua", 0)

    def _reset_all(self) -> None:
        """Reset all parameters including heater."""
        if not self._can_write_or_warn():
            return
        client = self._get_client_or_warn()
        if not client:
            return
        # Reset in safe order: Ug2, Ug1, Ua, heater (both Uh and Ih — the
        # firmware drives the heater via either channel, so zero both).
        ug1_after_stop = self.get_app_config().ug1_after_stop
        client.set_param("Ug2", 0)
        client.set_param("Ug1", encode_ug1(ug1_after_stop))
        client.set_param("Ua", 0)
        client.set_param("Uh", 0)
        client.set_param("Ih", 0)

    def _take_point(self) -> None:
        client = self._get_client_or_warn()
        if not client:
            return
        try:
            ia_raw = client.get_param("Ia", real=True)
            ig2_raw = client.get_param("Ig2", real=True)
            ua_raw = client.get_param("Ua", real=True)
            ug1_raw = client.get_param("Ug1", real=True)
            ug2_raw = client.get_param("Ug2", real=True)
            uh_raw = client.get_param("Uh", real=True)
            ih_raw = client.get_param("Ih", real=True)

            cal = self.get_calibration()
            point = {
                "ua": cal.apply_read("ua", float(ua_raw)),
                "ug1": cal.apply_read("ug1", decode_ug1(ug1_raw)),
                "ug2": cal.apply_read("ug2", float(ug2_raw)),
                "ia": cal.apply_read("ia", decode_ia(ia_raw)),
                "ig2": cal.apply_read("ig2", decode_ig2(ig2_raw)),
                "uh": cal.apply_read("uh", decode_uh(uh_raw)),
                "ih": cal.apply_read("ih", decode_ih(ih_raw)),
            }
            self.manual_points.append(point)
            self._update_points_table()
            self.points_label.setText(t('manual.Points_taken', count=len(self.manual_points)))
        except Exception as exc:
            log.exception("Failed to take manual measurement point")
            QMessageBox.warning(self, t('msg.Error'), t('msg.Failed_to_take_point', error=exc))

    def _update_points_table(self) -> None:
        self.points_table.setRowCount(len(self.manual_points))
        for row, p in enumerate(self.manual_points):
            self.points_table.setItem(row, 0, QTableWidgetItem(f"{p['ua']:.1f}"))
            self.points_table.setItem(row, 1, QTableWidgetItem(f"{p['ug1']:.2f}"))
            self.points_table.setItem(row, 2, QTableWidgetItem(f"{p['ug2']:.1f}"))
            self.points_table.setItem(row, 3, QTableWidgetItem(f"{p['ig2']:.2f}"))
            self.points_table.setItem(row, 4, QTableWidgetItem(f"{p['ia']:.2f}"))

    def _add_to_main_plot(self) -> None:
        if not self.manual_points:
            QMessageBox.warning(self, t('msg.Plot'), t('msg.No_points_to_add'))
            return
        self.on_add_to_main_plot(list(self.manual_points))
        QMessageBox.information(self, t('msg.Plot'), t('msg.Added_points', count=len(self.manual_points)))

    def _copy_points_to_clipboard(self) -> None:
        if not self.manual_points:
            return
        headers = [t('common.Ua'), t('common.Ug1'), t('common.Ug2'), t('common.Ig2'), t('common.Ia')]
        lines = ["\t".join(headers)]
        for p in self.manual_points:
            lines.append(f"{p['ua']:.1f}\t{p['ug1']:.2f}\t{p['ug2']:.1f}\t{p['ig2']:.2f}\t{p['ia']:.2f}")
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _save(self) -> None:
        """Trigger save: measure SRK and save measurement via callback."""
        include = self.include_points_cb.isChecked()
        points = list(self.manual_points) if include else []
        self.on_save(points, include)

    def set_save_enabled(self, enabled: bool) -> None:
        """Enable/disable save button (called during SRK measurement)."""
        self.save_btn.setEnabled(enabled)

    def _clear_points(self) -> None:
        self.manual_points = []
        self.points_table.setRowCount(0)
        self.points_label.setText(t('manual.Points_taken', count=0))

    def _toggle_curve(self, curve_id: str, visible: bool) -> None:
        """Show or hide a curve and its Y axis."""
        if curve_id == "ia":
            self.ia_curve.setVisible(visible)
            self.ia_plot.getAxis("left").setVisible(visible)
        else:
            self.ig2_curve.setVisible(visible)
            self.ia_plot.getAxis("right").setVisible(visible)

    def _pick_color(self, curve_id: str) -> None:
        """Open color dialog for the given curve and apply."""
        if curve_id == "ia":
            old = self._ia_color
        else:
            old = self._ig2_color
        color = QColorDialog.getColor(old, self, t('msg.Choose_color', name=curve_id.upper()))
        if not color.isValid():
            return
        if curve_id == "ia":
            self._ia_color = color
            self.ia_color_btn.setStyleSheet(f"background-color: {color.name()};")
        else:
            self._ig2_color = color
            self.ig2_color_btn.setStyleSheet(f"background-color: {color.name()};")
        self._apply_pen(curve_id)

    def _apply_pen(self, curve_id: str) -> None:
        """Update pen for the given curve from current color/width settings."""
        if curve_id == "ia":
            pen = pg.mkPen(self._ia_color.name(), width=self.ia_width_spin.value())
            self.ia_curve.setPen(pen)
            self.ia_plot.setLabel("left", t('plot.Ia_mA'), color=self._ia_color.name())
        else:
            pen = pg.mkPen(self._ig2_color.name(), width=self.ig2_width_spin.value())
            self.ig2_curve.setPen(pen)
            self.ia_plot.getAxis("right").setLabel(t('plot.Ig2_mA'), color=self._ig2_color.name())

    def _apply_y_from_zero(self, checked: bool) -> None:
        """Toggle Y-from-0 mode: when on, Y axes auto-scale but start from 0."""
        if checked:
            self.ia_plot.enableAutoRange(axis="y", enable=False)
            self.ig2_vb.enableAutoRange(axis="y", enable=False)
            self._update_y_ranges()
        else:
            self.ia_plot.enableAutoRange(axis="y", enable=True)
            self.ig2_vb.enableAutoRange(axis="y", enable=True)

    def _update_y_ranges(self) -> None:
        """Set Y ranges from 0 to max of visible data (with padding)."""
        if not self.ia_data:
            return
        ia_max = max(self.ia_data)
        ig2_max = max(self.ig2_data) if self.ig2_data else 0
        pad = 1.1
        self.ia_plot.setYRange(0, max(ia_max * pad, 0.1), padding=0)
        self.ig2_vb.setYRange(0, max(ig2_max * pad, 0.1), padding=0)

    def _sync_ig2_view(self) -> None:
        """Keep Ig2 ViewBox geometry in sync with main plot."""
        self.ig2_vb.setGeometry(self.ia_plot.getViewBox().sceneBoundingRect())

    def _reset_ia_chart(self) -> None:
        """Reset the real-time Ia/Ig2 chart."""
        self.ia_start_time = None
        self.ia_data = []
        self.ig2_data = []
        self.ia_time = []
        self.ia_curve.setData([], [])
        self.ig2_curve.setData([], [])

    def update_live_params(self, data: Dict) -> None:
        """Update live readings and real-time Ia chart."""
        calibration = self.get_calibration()
        self.live_panel.update_values(data, calibration)

        ia_val = calibration.apply_read("ia", decode_ia(data["ia"]))

        ig2_val = calibration.apply_read("ig2", decode_ig2(data["ig2"]))

        if self.ia_start_time is None:
            self.ia_start_time = time.time()
            self.ia_data = []
            self.ig2_data = []
            self.ia_time = []

        elapsed = time.time() - self.ia_start_time
        self.ia_time.append(elapsed)
        self.ia_data.append(ia_val)
        self.ig2_data.append(ig2_val)

        # Keep only last 60 seconds of data
        max_history = 60.0
        while self.ia_time and (elapsed - self.ia_time[0]) > max_history:
            self.ia_time.pop(0)
            self.ia_data.pop(0)
            self.ig2_data.pop(0)

        self.ia_curve.setData(self.ia_time, self.ia_data)
        self.ig2_curve.setData(self.ia_time, self.ig2_data)

        if self.y_from_zero_cb.isChecked():
            self._update_y_ranges()
