"""Convention guards: colors, i18n, constants, type hints, Protocol,
tooltips, contract vocabularies.

Pins (discriminating-pin checklist):
- ML-019/020/022/023/024: canonical colors — static pins on raw RGB
  in renderers, value identity, danger-fill derived from COLOR_LIMIT;
- ML-017: LampPanel mode label via t() (per maintained locale + static);
- ML-014: ra_range without magic numbers (static pin);
- ML-021/026: annotation completeness (inspect ratchet);
- ML-025: HealthConfig Protocol — AST typo pin in both directions;
- ML-018: generic ratchet: every interactive ModelDialog widget
  carries a tooltip (catches future widgets added without one).

Run:  py -m pytest tests/test_conventions_guards.py -v
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from i18n_setup import available_locales  # noqa: E402


def _src(rel: str) -> str:
    return io.open(PROJECT_ROOT / rel, encoding="utf-8").read()


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ═══════════════════════════════════════════════════════════════════
#  Colors
# ═══════════════════════════════════════════════════════════════════

class TestColorCanon:

    def test_color_ia_ig2_canonical_in_plot_style(self):
        from lm19 import plot_style as ps
        assert ps.COLOR_IA == ps.SERIES_PALETTE[1]
        assert ps.COLOR_IG2 == ps.SERIES_PALETTE[0]

    def test_ui_theme_reexports_not_redefines(self):
        from app import ui_theme
        from lm19 import plot_style as ps
        assert ui_theme.COLOR_IA == ps.COLOR_IA
        assert ui_theme.COLOR_IG2 == ps.COLOR_IG2
        # static: ui_theme must not carry its own hex literal
        src = _src("app/ui_theme.py")
        assert 'COLOR_IA = "#' not in src
        assert 'COLOR_IG2 = "#' not in src

    def test_no_raw_load_line_rgb_in_renderers(self):
        """ML-019/020: raw COLOR_LOAD_LINE RGB in renderers is banned."""
        for rel in ("app/plotting/_curves_plot_mixin.py",
                    "app/plotting/_plot_2d_mixin.py",
                    "app/plotting/_heatmap_mixin.py",
                    "app/plotting/overlays.py",
                    "app/plotting/renderer.py"):
            assert "0, 102, 204" not in _src(rel), rel
            assert "0,102,204" not in _src(rel), rel

    def test_no_raw_danger_rgb_in_overlays(self):
        """ML-022/023: all three danger zones go through _danger_brush."""
        src = _src("app/plotting/overlays.py")
        assert "255, 0, 0" not in src
        assert src.count("_danger_brush()") >= 3

    def test_danger_brush_derived_from_color_limit(self, qapp):
        import pyqtgraph as pg
        from app.plotting.overlays import _danger_brush
        from lm19.plot_style import COLOR_LIMIT, DANGER_FILL_ALPHA
        c = _danger_brush().color()
        ref = pg.mkColor(COLOR_LIMIT)
        assert (c.red(), c.green(), c.blue()) == (
            ref.red(), ref.green(), ref.blue())
        assert c.alpha() == DANGER_FILL_ALPHA

    def test_alpha_roles_not_swapped(self):
        """Call-site vs function: the helper pin
        proves the helper, not WHICH alpha each call site uses."""
        lines = _src("app/plotting/_curves_plot_mixin.py").splitlines()
        marker = [l for l in lines
                  if "symbolBrush=pg.mkBrush(_load_line_tint(" in l]
        halo = [l for l in lines
                if "cross_pen = pg.mkPen(_load_line_tint(" in l]
        assert marker, "marker call site not found"
        assert halo, "halo call site not found"
        assert all("_LL_MARKER_ALPHA" in l for l in marker)
        assert all("_LL_HALO_ALPHA" in l for l in halo)

    def test_load_line_tint_helper(self, qapp):
        import pyqtgraph as pg
        from app.plotting._curves_plot_mixin import _load_line_tint
        from lm19.plot_style import COLOR_LOAD_LINE, LOAD_LINE_HALO_ALPHA
        c = _load_line_tint(LOAD_LINE_HALO_ALPHA)
        ref = pg.mkColor(COLOR_LOAD_LINE)
        assert (c.red(), c.green(), c.blue()) == (
            ref.red(), ref.green(), ref.blue())
        assert c.alpha() == LOAD_LINE_HALO_ALPHA


# ═══════════════════════════════════════════════════════════════════
#  ML-017: mode label via t()
# ═══════════════════════════════════════════════════════════════════

class TestLampModeLabelI18n:

    def _lamp(self, triode: bool):
        from lm19.config import LampConfig
        return LampConfig(
            tube_type="X", socket="B", anodes=1, warmup_s=120,
            topology=("triode" if triode else "pentode"),
            uh=6.3, ih=0.76, ug1=-7.0, ua=250.0, ia=48.0,
            ug2=(0.0 if triode else 250.0),
            ig2=(0.0 if triode else 5.0),
            s=5.0, r=10.0, k=50.0, ranges={}, limits={})

    def test_no_hardcoded_mode_string(self):
        src = _src("app/lamp_panel.py")
        assert 'f"Mode: Uh' not in src

    @pytest.mark.parametrize("locale", available_locales())
    def test_keys_exist(self, locale):
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        assert "Mode_line_triode" in data["lamp"]
        assert "Mode_line_pentode" in data["lamp"]

    def test_label_rendered_via_t(self, qapp):
        import i18n_setup
        i18n_setup.setup("en")
        from i18n_setup import t
        from app.lamp_panel import LampPanel
        panel = LampPanel()
        lamp = self._lamp(triode=False)
        panel.apply_lamp(lamp)
        pa = (lamp.ua * lamp.ia) / 1000.0
        pg2 = (lamp.ug2 * lamp.ig2) / 1000.0
        expected = t("lamp.Mode_line_pentode",
                     uh=f"{lamp.uh:.1f}", ih=f"{lamp.ih:.1f}",
                     ua=f"{lamp.ua:.0f}", ia=f"{lamp.ia:.0f}",
                     ug1=f"{lamp.ug1:.1f}", ug2=f"{lamp.ug2:.0f}",
                     pa=f"{pa:.2f}", pg2=f"{pg2:.2f}")
        assert panel.mode_label.text() == expected
        assert "%{" not in panel.mode_label.text(), "unfilled placeholder"

    def test_triode_label_rendered_via_t(self, qapp):
        """Twin pin: the pentode pin says nothing about the triode
        branch — a key swap there used to survive."""
        import i18n_setup
        i18n_setup.setup("en")
        from i18n_setup import t
        from app.lamp_panel import LampPanel
        panel = LampPanel()
        lamp = self._lamp(triode=True)
        panel.apply_lamp(lamp)
        pa = (lamp.ua * lamp.ia) / 1000.0
        expected = t("lamp.Mode_line_triode",
                     uh=f"{lamp.uh:.1f}", ih=f"{lamp.ih:.1f}",
                     ua=f"{lamp.ua:.0f}", ia=f"{lamp.ia:.0f}",
                     ug1=f"{lamp.ug1:.1f}", pa=f"{pa:.2f}")
        assert panel.mode_label.text() == expected
        assert "%{" not in panel.mode_label.text(), "unfilled placeholder"

    @pytest.mark.parametrize("triode", [True, False])
    @pytest.mark.parametrize("locale", available_locales())
    def test_line_is_one_line_of_values_without_a_caption(self, qapp,
                                                          locale, triode):
        """One line, values only, starting with the heater.

        Both branches, every locale — the quantity symbols (Uh/Ia/Pa) are
        physical notation and stay untranslated, so the same check holds
        everywhere. A caption in front would push the widest string in the
        panel further right for no information.
        """
        import i18n_setup
        from app.lamp_panel import LampPanel
        i18n_setup.setup(locale)
        try:
            panel = LampPanel()
            lamp = self._lamp(triode=triode)
            panel.apply_lamp(lamp)
            text = panel.mode_label.text()
        finally:
            i18n_setup.setup("en")
        assert text.startswith("Uh "), text
        assert "\n" not in text, text
        assert "%{" not in text, text
        for token in ("Ih ", "Ua ", "Ia ", "Ug1 ", "Pa "):
            assert token in text, (token, text)
        assert ("Ug2 " in text) is (not triode), text

    @pytest.mark.parametrize("locale", available_locales())
    def test_every_locale_shows_the_heater(self, locale):
        """Twin of the render pin at template level: a translation that
        dropped %{uh} would quietly hide the heater in that locale only."""
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        for key in ("Mode_line_triode", "Mode_line_pentode"):
            tmpl = data["lamp"][key]
            assert "%{uh}" in tmpl and "%{ih}" in tmpl, (locale, key)
            assert tmpl.startswith("Uh "), (locale, key)

    def test_line_starts_under_the_socket_letter(self, qapp):
        """Layout pin: the line is a full-width row of the panel, not a
        form row — indented into the form column it loses the socket
        column's width and wraps sooner."""
        from app.lamp_panel import LampPanel
        panel = LampPanel()
        panel.apply_lamp(self._lamp(triode=False))
        panel.resize(panel.sizeHint())
        panel.show()
        try:
            assert panel.mode_label.x() <= panel.socket_label.x()
            assert panel.mode_label.y() > panel.socket_label.y()
        finally:
            panel.close()


class TestLocalePlaceholderParity:
    """Keys existed in several locale files but the placeholder SETS
    were never compared — a %{...} dropped from a translation used to
    survive. Global ratchet: a %{...} set mismatch between the default
    locale and ANY other locale file fails the test. Keys missing from
    a translation are skipped — the default locale covers them at
    runtime by design."""

    def test_all_keys_placeholder_parity(self):
        import re
        locales_dir = PROJECT_ROOT / "locales"
        en = json.loads((locales_dir / "en.json")
                        .read_text(encoding="utf-8"))
        others = sorted(p for p in locales_dir.glob("*.json")
                        if p.name != "en.json")
        assert others, "no translation files — pin would pass vacuously"
        ph = lambda s: frozenset(re.findall(r"%\{(\w+)\}", s))
        mismatched = []
        for path in others:
            loc = json.loads(path.read_text(encoding="utf-8"))
            for sec, keys in en.items():
                if not isinstance(keys, dict):
                    continue
                for k, v in keys.items():
                    lv = loc.get(sec, {}).get(k)
                    if isinstance(v, str) and isinstance(lv, str) \
                            and ph(v) != ph(lv):
                        mismatched.append(f"{path.stem}: {sec}.{k}")
        assert mismatched == [], (
            "placeholder sets differ from the default locale: "
            f"{mismatched}")


class TestLocaleKeysResolvable:
    """The t() resolver splits the lookup key on dots and walks NESTED
    dicts — a literal key name containing a dot (e.g. a top-level
    "msg.Settings_loaded" instead of "Settings_loaded" inside the "msg"
    section) is unreachable by construction: t() shows the raw key in
    the UI while every set-based locale check (key parity, flatten
    comparisons) sees the flattened path as present. The placeholder
    pin above skips non-dict top-level values, so nothing else catches
    this shape."""

    def test_no_dotted_key_names_in_any_locale(self):
        offenders = []
        for path in sorted((PROJECT_ROOT / "locales").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))

            def walk(node, crumb):
                for k, v in node.items():
                    if "." in k:
                        offenders.append(f"{path.stem}: {crumb}{k}")
                    if isinstance(v, dict):
                        walk(v, f"{crumb}{k}/")

            walk(data, "")
        assert offenders == [], (
            "locale key NAMES must not contain dots — t() resolves "
            "dots as nesting, these keys are unreachable: "
            f"{offenders}")


# ═══════════════════════════════════════════════════════════════════
#  ML-014: ra_range without magic (static pin)
# ═══════════════════════════════════════════════════════════════════

class TestRaRangeNoMagic:

    def test_constants_used_in_constraints(self):
        src = _src("app/amp_control_panel.py")
        assert "_OPT_RA_MIN_KOHM" in src
        assert "_OPT_RA_SPAN_FACTOR" in src
        assert "_OPT_RA_MAX_FLOOR_KOHM" in src
        assert "ra_range=(0.5" not in src, "raw magic numbers returned"

    def test_behavior_matches_documented_formula(self, qapp):
        """Numbers here are independent of the constants (catch drift)."""
        from app.amp_control_panel import AmpControlPanel
        panel = AmpControlPanel()
        panel.ra_spin.setValue(20.0)
        assert panel.optimizer_constraints().ra_range == (0.5, 100.0)
        panel.ra_spin.setValue(2.0)
        assert panel.optimizer_constraints().ra_range == (0.5, 50.0)
        panel.close()


# ═══════════════════════════════════════════════════════════════════
#  ML-021 / ML-026: annotation completeness (inspect ratchet)
# ═══════════════════════════════════════════════════════════════════

class TestTypeHintRatchets:

    def _assert_fully_annotated(self, fn, *, skip=("self",)):
        sig = inspect.signature(fn)
        missing = [n for n, p in sig.parameters.items()
                   if n not in skip and p.annotation is inspect.Parameter.empty]
        assert missing == [], f"{fn.__qualname__}: unannotated {missing}"
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"{fn.__qualname__}: missing return annotation")

    def test_plot_2d_public_render_annotated(self, qapp):
        from app.plotting._plot_2d_mixin import _Plot2DMixin
        self._assert_fully_annotated(_Plot2DMixin.render_plot_2d)
        self._assert_fully_annotated(_Plot2DMixin._render_compare_overlay)
        self._assert_fully_annotated(_Plot2DMixin._render_zone_rect)
        # _render_load_line was removed: the line is drawn by
        # WorkingLineController.

    def test_koren_internals_annotated(self):
        from lm19.spice_export import koren as K
        for name in ("_koren_ia", "_make_initial_guess", "_fit_koren_scipy",
                     "_residuals", "_fit_koren_numpy", "_koren_ia_pentode",
                     "_koren_ig2_pentode", "_make_pentode_initial_guess",
                     "_pentode_residuals", "_fit_pentode_scipy",
                     "_fit_pentode_numpy"):
            self._assert_fully_annotated(getattr(K, name))


# ═══════════════════════════════════════════════════════════════════
#  ML-025: HealthConfig Protocol — AST typo pin
# ═══════════════════════════════════════════════════════════════════

class TestHealthConfigProtocol:

    def _cfg_uses(self):
        tree = ast.parse(_src("lm19/health.py"))
        used = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "cfg"):
                used.add(node.attr)
        return used, tree

    def test_every_cfg_attr_declared_in_protocol(self):
        """A typo in ``cfg.<attr>`` (or a new attribute without a
        declaration) fails CI — what the bare parameter used to mask."""
        from lm19.health import HealthConfig
        used, _ = self._cfg_uses()
        declared = set(HealthConfig.__annotations__)
        missing = sorted(used - declared)
        assert missing == [], f"cfg attrs not in HealthConfig: {missing}"

    def test_protocol_subset_of_app_config(self):
        from lm19.health import HealthConfig
        from lm19.app_config import AppConfig
        fields = {f.name for f in dataclasses.fields(AppConfig)}
        extra = sorted(set(HealthConfig.__annotations__) - fields)
        assert extra == [], f"HealthConfig attrs not in AppConfig: {extra}"

    def test_protocol_types_match_app_config(self):
        """Names were compared but TYPES were not
        (health_ua_retries: str used to survive)."""
        import typing
        from lm19.health import HealthConfig
        from lm19.app_config import AppConfig
        proto = typing.get_type_hints(HealthConfig)
        app = typing.get_type_hints(AppConfig)
        mismatched = sorted(
            n for n, t_ in proto.items() if app.get(n) is not t_)
        assert mismatched == [], f"type drift vs AppConfig: {mismatched}"

    def test_all_cfg_params_annotated(self):
        _, tree = self._cfg_uses()
        bare = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for a in node.args.args:
                    if a.arg == "cfg" and a.annotation is None:
                        bare.append(node.name)
        assert bare == [], f"defs with unannotated cfg: {bare}"


# ═══════════════════════════════════════════════════════════════════
#  ML-018: generic ModelDialog tooltip ratchet
# ═══════════════════════════════════════════════════════════════════

class TestModelDialogTooltips:

    def _interactive_children(self, dlg):
        from PySide6.QtWidgets import (
            QAbstractSpinBox, QCheckBox, QComboBox, QDialogButtonBox,
            QLineEdit, QPushButton, QRadioButton, QWidget,
        )
        kinds = (QPushButton, QComboBox, QAbstractSpinBox, QRadioButton,
                 QCheckBox, QLineEdit)
        out = []
        for w in dlg.findChildren(QWidget):
            if not isinstance(w, kinds):
                continue
            # Qt-internal children (e.g. the QLineEdit inside every
            # spinbox, objectName "qt_spinbox_lineedit") — not user-facing
            # controls of ours.
            if w.objectName().startswith("qt_"):
                continue
            # standard OK/Cancel of the dialog button box — exempt
            p = w.parent()
            exempt = False
            while p is not None:
                if isinstance(p, QDialogButtonBox):
                    exempt = True
                    break
                p = p.parent()
            if not exempt:
                out.append(w)
        return out

    def _points(self, pentode: bool):
        pts = []
        for ug1 in (-8.0, -6.0, -4.0):
            for ua in (100.0, 200.0, 300.0):
                p = {"ua": ua, "ug1": ug1, "ia": 10.0 + ua / 50.0}
                if pentode:
                    p["ug2"] = 250.0
                pts.append(p)
        return pts

    @pytest.mark.parametrize("case", ["pentode", "track", "triode"])
    def test_every_interactive_widget_has_tooltip(self, qapp, case):
        """Ratchet: any future widget without a tooltip fails here."""
        from app.model_dialog import ModelDialog
        kw = {}
        if case == "triode":
            kw = dict(points=self._points(False), is_triode=True)
        elif case == "track":
            kw = dict(points=self._points(True), is_triode=False,
                      scan_settings={"ug2_track_ua": True})
        else:
            kw = dict(points=self._points(True), is_triode=False)
        dlg = ModelDialog(**kw)
        children = self._interactive_children(dlg)
        naked = [w.__class__.__name__ + ":" + (w.objectName() or w.text()
                 if hasattr(w, "text") else "?")
                 for w in children if not w.toolTip().strip()]
        # A raw key instead of a translation (lost t()) is a non-empty
        # tooltip — but garbage for the user.
        rawkey = [w.toolTip() for w in children
                  if w.toolTip().strip().startswith(
                      ("model.", "tip.", "amp.", "scan.", "health."))
                  or "%{" in w.toolTip()]
        dlg.close()
        assert naked == [], f"widgets without tooltip: {naked}"
        assert rawkey == [], f"untranslated tooltip keys: {rawkey}"

    @pytest.mark.parametrize("locale", available_locales())
    def test_tip_keys_exist(self, locale):
        data = json.loads(
            (PROJECT_ROOT / "locales" / f"{locale}.json")
            .read_text(encoding="utf-8"))
        model = data["model"]
        for k in ("Model_type_tip", "Reference_tip", "Fit_tip",
                  "Ref_tube_tip", "Grid_source_tip", "Ua_grid_tip",
                  "Ug1_grid_tip", "Ug2_grid_tip", "Ug2_offset_tip",
                  "Compare_all_tip", "Compare_cancel_tip",
                  "Add_selected_tip", "Compare_check_tip"):
            assert k in model, (locale, k)


# ═══════════════════════════════════════════════════════════════════
#  Contract vocabularies (no-magic-strings rule)
# ═══════════════════════════════════════════════════════════════════

class TestContractVocabularies:
    """Vocabulary migration ratchet: hd_method + circuit + statuses.

    Source of truth: lm19/amplifier/constants.py (HD_METHOD_*/
    HD_METHODS, CIRCUIT_*/CIRCUITS) and peer registries. Raw
    vocabulary literals are banned in semantic contexts across the
    whole tree (lm19/app/tests): hd_method=/circuit= kwargs (ANY
    string — catches input typos like "chebyshv"), parameter/field
    defaults, comparisons whose text mentions method/circuit,
    combo addItem data, .get("method", ...) defaults, {"method": ...}
    literals, and return literals in resolve paths. Comments and
    docstrings are not flagged (they are outside the contexts).
    """

    _SCAN_DIRS = ("lm19", "app", "tests")
    _OWNER = "lm19/amplifier/constants.py"
    _SELF = "tests/test_conventions_guards.py"
    # kwarg/parameter/field names whose values must come from the
    # registries (any string literal is a violation).
    _VOCAB_ARG_NAMES = frozenset({"hd_method", "circuit", "topology",
                                  "ug2_mode", "model_type"})
    # Context markers for comparisons: without them "auto"/"pp" in
    # unrelated domains would be flagged.
    _CMP_MARKERS = ("method", "circuit", "status", "warning", "error",
                    "code", "topology", "ug2_mode", "model_type")

    @staticmethod
    def _vocab() -> frozenset:
        from lm19.amp_engine import ENGINE_WARNING_CODES
        from lm19.amplifier.constants import (
            CIRCUITS, HD_METHOD_CHEBYSHEV_PP, HD_METHOD_DFT_PP, HD_METHODS)
        from lm19.constants import TOPOLOGIES, UG2_MODES
        from lm19.optimizer import OPT_ERROR_CODES, OPT_WARNING_CODES
        from lm19.scan.events import SCAN_CURVE_STATUSES
        from lm19.tube_model_base import MODEL_TYPES, MODEL_WARNING_CODES
    # "resistive" is a historical circuit typo (lived in 3 tests via
    # the default branch) — reintroduction is banned.
        return frozenset(HD_METHODS | CIRCUITS | SCAN_CURVE_STATUSES
                         | ENGINE_WARNING_CODES | OPT_WARNING_CODES
                         | OPT_ERROR_CODES | TOPOLOGIES | UG2_MODES
                         | MODEL_TYPES | MODEL_WARNING_CODES
                         | {HD_METHOD_CHEBYSHEV_PP, HD_METHOD_DFT_PP,
                            "resistive"})

    def _iter_files(self):
        for d in self._SCAN_DIRS:
            for p in sorted((PROJECT_ROOT / d).rglob("*.py")):
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                if rel in (self._OWNER, self._SELF):
                    continue
                if "spice_test_data" in rel:
                    continue
                yield rel, p

    def _violations_in(self, rel: str, tree: ast.AST, vocab: frozenset):
        out = []
        arg_names = self._VOCAB_ARG_NAMES
        markers = self._CMP_MARKERS
        # Collection nodes that ARE a registry definition (right-hand side
        # of an ALL_CAPS assignment) — exempt from the shadow-registry rule.
        definition_sites: set = set()

        def flag(node, ctx):
            out.append(f"{rel}:{node.lineno}: {ctx}")

        class V(ast.NodeVisitor):
            def visit_Call(self, node):
                # Preset circuits are canonical CIRCUITS values now —
                # no exemption needed.
                is_preset = False
                # i18n formatters: kwargs are template placeholders
                # (t("report.Amp_params", circuit=...)), not vocabulary.
                is_i18n = (isinstance(node.func, ast.Name)
                           and node.func.id in ("t", "t_global", "tr"))
                for kw in node.keywords:
                    if (not is_preset and not is_i18n
                            and kw.arg in arg_names
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)):
                        flag(kw.value, f"kwarg {kw.arg}=<literal>")
                    # method= is a generic name (scipy method="Nelder-
                    # Mead" is legit): flag vocabulary values only.
                    elif (kw.arg == "method"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value in vocab):
                        flag(kw.value, "kwarg method=<vocab literal>")
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "addItem"):
                    for a in node.args:
                        if isinstance(a, ast.Constant) and a.value in vocab:
                            flag(a, "addItem data literal")
                # .append(<vocab literal>) — optimizer warn codes
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "append" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value in vocab):
                    flag(node.args[0], "append(<vocab literal>)")
                # .get(<any key>, <vocab literal>) — the fallback decides
                # the branch just as the key does.
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get" and len(node.args) > 1
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in vocab):
                    flag(node.args[1], ".get(..., <vocab literal>)")
                # Positional vocabulary argument — the kwarg check above
                # only sees keywords, so ``fit(points, "pentode")`` slips.
                # Skipped: the KEY slot of dict lookups. ``item.get("koren")``
                # names a config key that merely spells like a vocabulary
                # member; config keys have their own guards and are exempt
                # by the no-magic rule. The default slot is still checked
                # by the .get rule above.
                lookup = (isinstance(node.func, ast.Attribute)
                          and node.func.attr in ("get", "pop", "setdefault"))
                for i, a in enumerate(node.args):
                    if lookup and i == 0:
                        continue
                    if (not is_i18n and isinstance(a, ast.Constant)
                            and a.value in vocab):
                        flag(a, f"positional arg <literal {a.value!r}>")
                self.generic_visit(node)

            def visit_Compare(self, node):
                flat = [node.left]
                for c in node.comparators:
                    flat.extend(c.elts if isinstance(c, ast.Tuple) else [c])
                if any(isinstance(c, ast.Constant) and c.value in vocab
                       for c in flat):
                    txt = ast.unparse(node)
                    if any(m in txt for m in markers):
                        flag(node, f"compare: {txt[:60]}")
                self.generic_visit(node)

            def visit_AnnAssign(self, node):
                if (isinstance(node.target, ast.Name)
                        and node.target.id in arg_names
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    flag(node, f"field {node.target.id}=<literal>")
                self.generic_visit(node)

            def _check_defaults(self, node):
                a = node.args
                pos = a.posonlyargs + a.args
                for arg, default in zip(pos[len(pos) - len(a.defaults):],
                                        a.defaults):
                    if (arg.arg in arg_names or arg.arg == "method") \
                            and isinstance(default, ast.Constant) \
                            and default.value in vocab:
                        flag(default, f"default {arg.arg}=<literal>")
                for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                    if default is not None \
                            and (arg.arg in arg_names or arg.arg == "method") \
                            and isinstance(default, ast.Constant) \
                            and default.value in vocab:
                        flag(default, f"default {arg.arg}=<literal>")

            def visit_FunctionDef(self, node):
                self._check_defaults(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self._check_defaults(node)
                self.generic_visit(node)

            def visit_Dict(self, node):
                for k, val in zip(node.keys, node.values):
                    if (isinstance(k, ast.Constant)
                            and k.value in ("method", "hd_method",
                                            "circuit", "status", "code",
                                            "topology", "ug2_mode",
                                            "model_type")
                            and isinstance(val, ast.Constant)
                            and val.value in vocab):
                        flag(val, f"dict {{{k.value!r}: <literal>}}")
                    # Vocabulary literal as a KEY — a label/handler map
                    # keyed by hand. The registry is the only list that
                    # stays complete when a member is added.
                    if isinstance(k, ast.Constant) and k.value in vocab:
                        flag(k, f"dict key <literal {k.value!r}>")
                self.generic_visit(node)

            def visit_IfExp(self, node):
                # ``"triode" if x else "pentode"`` — the branch value is a
                # Constant, but it sits where none of the value-position
                # checks look.
                for branch in (node.body, node.orelse):
                    if isinstance(branch, ast.Constant) and branch.value in vocab:
                        flag(branch, f"ternary branch <literal {branch.value!r}>")
                self.generic_visit(node)

            def visit_Return(self, node):
                if isinstance(node.value, ast.Constant) \
                        and node.value.value in vocab:
                    flag(node, f"return <literal {node.value.value!r}>")
                self.generic_visit(node)

            def _shadow_registry(self, node):
                # A tuple/set/list of >=2 vocabulary literals is a shadow
                # registry — checked wherever it appears, not only as an
                # assignment value (``x in {"triode", "pentode"}`` hid one).
                hits = [e for e in node.elts
                        if isinstance(e, ast.Constant) and e.value in vocab]
                if len(hits) >= 2 and id(node) not in definition_sites:
                    flag(node, "shadow registry (collection of literals)")
                self.generic_visit(node)

            visit_Tuple = _shadow_registry
            visit_List = _shadow_registry
            visit_Set = _shadow_registry

            def visit_Assign(self, node):
                # ALL_CAPS target = the registry definition site itself:
                # the literal must be spelled out exactly once, and this
                # is that once — including the frozenset that lists them.
                # Exempt the VALUE ONLY, never the subtree: a map keyed by
                # raw literals (``_MODE_LABELS = {"pentode": ...}``) is
                # also an ALL_CAPS assignment, and skipping the subtree
                # hid exactly that (mutation P3 survived until this split).
                if any(isinstance(t, ast.Name) and t.id.isupper()
                       for t in node.targets):
                    value = node.value
                    definition_sites.add(id(value))
                    if isinstance(value, ast.Call):
                        for a in value.args:      # frozenset({...})
                            definition_sites.add(id(a))
                    if isinstance(value, ast.Constant):
                        return
                    self.generic_visit(node)
                    return
                # Any name bound to a vocabulary literal: the old check
                # only saw status/warning/error, so ``topo = "pentode"``
                # and ``mode = "triode"`` stayed invisible.
                if (isinstance(node.value, ast.Constant)
                        and node.value.value in vocab
                        and any(isinstance(tgt, ast.Name)
                                for tgt in node.targets)):
                    name = next(t.id for t in node.targets
                                if isinstance(t, ast.Name))
                    flag(node, f"assign {name}=<literal {node.value.value!r}>")
                self.generic_visit(node)

        V().visit(tree)
        return out

    # Pre-existing test-side debt, frozen per file. Production
    # (lm19/ + app/) carries NO debt and is asserted at zero, so any
    # file absent from this table -- including every new test file -- is
    # held to zero. Counts may only go DOWN: a lower actual count fails
    # too, forcing the number here to be tightened rather than left as
    # headroom for the next regression.
    _TEST_BASELINE = {
        "tests/test_amp_control_panel.py": 12,
        "tests/test_amp_engine.py": 6,
        "tests/test_amp_math_guards.py": 6,
        "tests/test_amplifier.py": 10,
        "tests/test_amplifier_real_data.py": 1,
        "tests/test_amplifier_tab_regression.py": 3,
        "tests/test_analysis_full.py": 3,
        "tests/test_dempwolf.py": 20,
        "tests/test_dialog_logic.py": 3,
        "tests/test_distortion_cross.py": 1,
        "tests/test_distortion_dft.py": 2,
        "tests/test_etracer_import.py": 12,
        "tests/test_export_visibility.py": 6,
        "tests/test_fit_koren.py": 11,
        "tests/test_health_accuracy_phases.py": 1,
        "tests/test_health_emission_verdict.py": 1,
        "tests/test_health_history.py": 3,
        "tests/test_health_history_ui.py": 5,
        "tests/test_health_logic.py": 25,
        "tests/test_health_refs_resolve.py": 4,
        "tests/test_health_tab_logic.py": 7,
        "tests/test_import_controller.py": 2,
        "tests/test_import_helpers.py": 3,
        "tests/test_koren.py": 8,
        "tests/test_lm19.py": 1,
        "tests/test_ltspice_asc.py": 28,
        "tests/test_ltspice_roundtrip.py": 59,
        "tests/test_ltspice_verify.py": 11,
        "tests/test_main_window_smoke.py": 1,
        "tests/test_model_base.py": 1,
        "tests/test_model_compare.py": 33,
        "tests/test_opt_apply_fidelity.py": 1,
        "tests/test_opt_output_completeness.py": 1,
        "tests/test_optimizer.py": 7,
        "tests/test_optimizer_physical_validation.py": 2,
        "tests/test_optimizer_se_xfmr.py": 3,
        "tests/test_optimizer_vectorized.py": 4,
        "tests/test_reefman.py": 9,
        "tests/test_scan_limits.py": 7,
        "tests/test_spice_export.py": 7,
        "tests/test_spice_export_models.py": 8,
        "tests/test_spice_from_model.py": 8,
        "tests/test_tools_fit_benchmark.py": 6,
        "tests/test_tools_lamp_sources.py": 4,
        "tests/test_tube_matching.py": 7,
        "tests/test_tube_params.py": 10,
        "tests/test_tube_sim.py": 11,
        "tests/test_ui_warnings.py": 1,
        "tests/test_worker_cleanup.py": 1,
    }

    def _counts(self) -> dict:
        vocab = self._vocab()
        found: dict = {}
        for rel, p in self._iter_files():
            tree = ast.parse(io.open(p, encoding="utf-8").read())
            v = self._violations_in(rel, tree, vocab)
            if v:
                found[rel] = v
        return found

    def test_no_raw_vocabulary_literals_in_production(self) -> None:
        """lm19/ and app/ must be free of raw vocabulary literals."""
        violations = [line for rel, lines in self._counts().items()
                      if not rel.startswith("tests/") for line in lines]
        assert not violations, (
            f"{len(violations)} raw vocabulary literals in production "
            f"- import the constants from the owning registry:\n"
            + "\n".join(violations))

    def test_untracked_test_files_are_clean(self) -> None:
        """A test file outside the baseline table must have zero.

        Literal INPUTS in tests are the dangerous half: a typo routes the
        test into another branch and its expectations still pass.
        """
        found = self._counts()
        offenders = [line for rel, lines in found.items()
                     if rel.startswith("tests/")
                     and rel not in self._TEST_BASELINE
                     for line in lines]
        assert not offenders, (
            f"{len(offenders)} raw vocabulary literals in test files that "
            f"carry no baseline - use the registry constants:\n"
            + "\n".join(offenders))

    def test_test_baseline_only_shrinks(self) -> None:
        """Ratchet: per-file counts may only decrease."""
        found = self._counts()
        grew, shrank = [], []
        for rel, allowed in self._TEST_BASELINE.items():
            actual = len(found.get(rel, []))
            if actual > allowed:
                grew.append(f"{rel}: {actual} > {allowed} allowed")
            elif actual < allowed:
                shrank.append(f"{rel}: {actual} (baseline says {allowed})")
        assert not grew, ("vocabulary literals grew:\n" + "\n".join(grew))
        assert not shrank, (
            "baseline is stale - lower these numbers (drop the entry at "
            "0):\n" + "\n".join(shrank))

    # ---- self-pins: the guard's own detection forms -------------------
    # A ratchet nobody tests is a ratchet that silently stops ratcheting.
    # Each form is exercised on a synthetic snippet, and each documented
    # exemption is pinned as NOT flagged -- otherwise a broadened exemption
    # (as happened with ALL_CAPS swallowing a whole subtree) goes unseen.

    def _flags(self, code: str):
        return self._violations_in("x.py", ast.parse(textwrap.dedent(code)),
                                   self._vocab())

    @pytest.mark.parametrize("code,form", [
        ('x = "triode" if flag else "pentode"', "ternary"),
        ('def f():\n    return "pentode"', "return"),
        ('M = {"pentode": "Pent", "triode": "Tri"}', "dict key"),
        ('fit(points, "pentode")', "positional arg"),
        ('topo = "pentode"', "assign"),
        ('mode = d.get("k", "pentode")', "get default"),
        ('if m in {"triode", "pentode"}:\n    pass', "shadow registry"),
        ('fit(points, topology="pentode")', "kwarg"),
    ])
    def test_form_is_detected(self, code: str, form: str) -> None:
        assert self._flags(code), f"{form}: raw literal not detected"

    @pytest.mark.parametrize("code,why", [
        ('TOPOLOGY_PENTODE = "pentode"', "registry definition site"),
        ('TOPOLOGIES = frozenset({"triode", "pentode"})', "registry listing"),
        ('v = item.get("koren")', "config key that spells like a member"),
        ('label = t("pentode")', "i18n lookup key"),
        ('x = "some_other_string"', "unrelated literal"),
    ])
    def test_exemption_is_not_flagged(self, code: str, why: str) -> None:
        assert not self._flags(code), f"false positive: {why}"

    def test_all_caps_exemption_covers_value_only(self) -> None:
        """The definition-site exemption must not swallow the subtree.

        ``_MODE_LABELS = {"pentode": ...}`` is an ALL_CAPS assignment whose
        VALUE is legitimate to write out, but whose KEYS are raw vocabulary.
        """
        assert self._flags('M = {"pentode": "Pent"}')
        assert self._flags('M = [f(x, "pentode")]')
        assert not self._flags('M = "pentode"')

    def test_ui_combos_expose_full_registries(self) -> None:
        """Completeness from the source of truth: the amp-panel combos
        must offer the FULL registry, via constants (a raw literal in
        addItem loses the name and fails this pin)."""
        import lm19.amplifier.constants as C
        tree = ast.parse(_src("app/amp_control_panel.py"))
        hd_used, circ_used = set(), set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "addItem"):
                for a in node.args:
                    if isinstance(a, ast.Name):
                        if a.id.startswith("HD_METHOD_"):
                            hd_used.add(getattr(C, a.id))
                        elif a.id.startswith("CIRCUIT_"):
                            circ_used.add(getattr(C, a.id))
        assert hd_used == set(C.HD_METHODS), hd_used
        assert circ_used == set(C.CIRCUITS), circ_used

    @pytest.mark.parametrize("rel,combo", [
        ("app/compare_tab.py", "match_mode_combo"),
        ("app/health_tab.py", "filter_mode_combo"),
        ("app/import_dialog.py", "topology_combo"),
    ])
    def test_topology_combos_expose_full_ug2_modes(self, rel,
                                                   combo) -> None:
        """Symmetric completeness pin for the topology combos: the
        Compare/Health/Import mode combos must offer the FULL
        UG2_MODES set, via constants."""
        import lm19.constants as C
        tree = ast.parse(_src(rel))
        used = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "addItem"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == combo):
                for a in node.args:
                    if isinstance(a, ast.Name) and a.id.startswith(
                            "TOPOLOGY_"):
                        used.add(getattr(C, a.id))
        assert used == set(C.UG2_MODES), (rel, combo, used)

    @pytest.mark.parametrize("locale", available_locales())
    def test_code_registries_biject_with_locales(self, locale) -> None:
        """Every registered code has an i18n key AND every key matches
        a registered code (both directions: a code without a key is a
        mute UI failure; a key without a code is a dead translation or
        post-rename drift)."""
        from lm19.amp_engine import ENGINE_WARNING_CODES
        from lm19.optimizer import OPT_ERROR_CODES, OPT_WARNING_CODES
        from lm19.scan.events import SCAN_CURVE_STATUSES
        data = json.loads(_src(f"locales/{locale}.json"))
        amp, msg = data["amp"], data["msg"]

        def keyset(section, prefix):
            return {k[len(prefix):] for k in section
                    if k.startswith(prefix)}

        assert keyset(amp, "warn_") == set(ENGINE_WARNING_CODES)
        assert keyset(amp, "opt_warn_") == set(OPT_WARNING_CODES)
        assert keyset(amp, "opt_err_") == set(OPT_ERROR_CODES)
        assert keyset(msg, "Scan_status_") == set(SCAN_CURVE_STATUSES)
        from lm19.tube_model_base import MODEL_WARNING_CODES
        assert keyset(data["model"], "warn_") == set(MODEL_WARNING_CODES)

    def test_resolvers_cover_every_registered_method(self) -> None:
        """Registry <-> BOTH resolvers (engine.resolve_hd_method and
        optimizer._resolve_methods), both directions: a new method in
        HD_METHODS without a branch fails (the auto fallback returns
        something else); a removed one fails on the constant import."""
        from lm19.amp_engine import SOURCE_MEASUREMENTS, resolve_hd_method
        from lm19.amplifier.constants import (
            HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV, HD_METHOD_DFT,
            HD_METHODS)
        from lm19.optimizer import _resolve_methods
        concrete = {HD_METHOD_5POINT, HD_METHOD_CHEBYSHEV, HD_METHOD_DFT}
        for m in sorted(HD_METHODS):
            for src in (SOURCE_MEASUREMENTS, "model"):
                r = resolve_hd_method(m, src)
                assert r in concrete, (m, src, r)
                if m in concrete:
                    assert r == m, (m, src, r)
            for has_model in (False, True):
                grid, refine, _warn = _resolve_methods(m, has_model)
                assert grid in concrete and refine in concrete, (m, grid)
                if m in concrete and (has_model or m != HD_METHOD_DFT):
                    assert grid == m, (m, grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
