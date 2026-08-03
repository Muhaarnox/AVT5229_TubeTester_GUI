"""Smoke tests for config and i18n bootstrap."""

import json
import re
import sys
from pathlib import Path

import pytest

from i18n_setup import available_locales

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i18n_setup import setup, t
from lm19.app_config import load_app_config
from lm19.config import load_lamps
from lm19.version import APP_VERSION

pytestmark = [pytest.mark.smoke_config]

_LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"
_BARE_VAR_RE = re.compile(r"(?<!%)(\{[a-z_]+\})")


@pytest.mark.smoke
def test_load_app_config_lamps_and_locales_without_exceptions():
    cfg = load_app_config()
    lamps = load_lamps()

    titles = {}
    for loc in available_locales():
        setup(loc)
        titles[loc] = t("app.Window_title", version=APP_VERSION)
    setup("en")

    assert cfg.live_poll_ms > 0
    assert len(lamps) > 0
    assert titles, "no locale files discovered"
    for loc, title in titles.items():
        assert title != "app.Window_title", loc
        assert "%{" not in title, (loc, title)


def _flatten(obj: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested locale dict to dot-separated keys."""
    result: dict[str, str] = {}
    for k, v in obj.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, full))
        elif isinstance(v, str):
            result[full] = v
    return result


def _locale_files():
    return sorted(_LOCALES_DIR.glob("*.json"))


@pytest.mark.smoke
@pytest.mark.parametrize("locale_path", _locale_files(), ids=lambda p: p.stem)
def test_no_bare_interpolation_vars(locale_path: Path):
    """All interpolation variables must use %{var} format, not bare {var}."""
    data = json.loads(locale_path.read_text(encoding="utf-8"))
    flat = _flatten(data)
    bad = []
    for key, value in flat.items():
        matches = _BARE_VAR_RE.findall(value)
        if matches:
            bad.append(f"  {key}: {matches} in {value!r}")
    assert not bad, (
        f"Bare {{var}} found in {locale_path.name} (should be %{{var}}):\n"
        + "\n".join(bad)
    )


@pytest.mark.smoke
@pytest.mark.parametrize("locale_path", _locale_files(), ids=lambda p: p.stem)
def test_locales_have_same_keys(locale_path: Path):
    """Every locale file must have the same set of keys as en.json."""
    en_path = _LOCALES_DIR / "en.json"
    if locale_path == en_path:
        pytest.skip("reference locale")
    en_keys = set(_flatten(json.loads(en_path.read_text(encoding="utf-8"))).keys())
    other_keys = set(_flatten(json.loads(locale_path.read_text(encoding="utf-8"))).keys())
    missing = en_keys - other_keys
    extra = other_keys - en_keys
    msgs = []
    if missing:
        msgs.append(f"Missing in {locale_path.name}: {sorted(missing)}")
    if extra:
        msgs.append(f"Extra in {locale_path.name}: {sorted(extra)}")
    assert not msgs, "\n".join(msgs)


@pytest.mark.smoke
@pytest.mark.parametrize("locale_path", _locale_files(), ids=lambda p: p.stem)
def test_interpolation_vars_consistent(locale_path: Path):
    """Each key must have the same %{var} placeholders across all locales."""
    en_path = _LOCALES_DIR / "en.json"
    if locale_path == en_path:
        pytest.skip("reference locale")
    var_re = re.compile(r"%\{([a-z_]+)\}")
    en_flat = _flatten(json.loads(en_path.read_text(encoding="utf-8")))
    other_flat = _flatten(json.loads(locale_path.read_text(encoding="utf-8")))
    bad = []
    for key in sorted(set(en_flat) & set(other_flat)):
        en_vars = set(var_re.findall(en_flat[key]))
        other_vars = set(var_re.findall(other_flat[key]))
        if en_vars != other_vars:
            bad.append(f"  {key}: en={en_vars} {locale_path.stem}={other_vars}")
    assert not bad, (
        f"Mismatched interpolation vars in {locale_path.name}:\n" + "\n".join(bad)
    )
