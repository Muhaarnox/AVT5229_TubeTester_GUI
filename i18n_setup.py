"""Internationalisation helpers.

Usage::

    from i18n_setup import t

    label.setText(t('conn.Refresh'))              # text of the active locale
    msg = t('msg.Saved_points', count=5)          # "Measurement saved (5 points)."

Keys are designed to be maximally readable so that even without
a translation file, the fallback (key path) is still meaningful.
"""

import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, List

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

_translations: dict = {}
_fallback: dict = {}


def _load_json(locale: str) -> dict:
    path = _LOCALES_DIR / f"{locale}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve(data: dict, key: str):
    """Resolve a dotted key in a nested dict."""
    parts = key.split(".")
    value = data
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
        if value is None:
            return None
    return value


def setup(locale: str = "en") -> None:
    """Initialise i18n with the given *locale*."""
    global _translations, _fallback
    _translations = _load_json(locale)
    _fallback = _load_json("en") if locale != "en" else {}


def set_locale(locale: str) -> None:
    """Switch locale at runtime (requires UI refresh)."""
    setup(locale)


def t(key: str, **kwargs) -> str:
    """Translate *key*, forwarding named arguments for interpolation."""
    value = _resolve(_translations, key)
    if value is None:
        value = _resolve(_fallback, key)
    if value is None:
        return key
    if kwargs:
        for k, v in kwargs.items():
            value = value.replace(f"%{{{k}}}", str(v))
    return value


def available_locales() -> List[str]:
    """Locale codes derived from ``locales/*.json`` (source of truth) —
    a newly dropped-in translation file appears without code changes."""
    return sorted(p.stem for p in _LOCALES_DIR.glob("*.json"))


@lru_cache(maxsize=None)
def _locale_data(locale: str) -> tuple:
    return (_load_json(locale), _load_json("en") if locale != "en" else {})


@contextmanager
def locale_override(locale: str) -> Iterator[None]:
    """Temporarily switch the GLOBAL translation state.

    Used by the amplifier PDF export: its HTML formatters are bound to
    the global ``t()``, and the document language may differ from the
    UI one. Always restored — including on exceptions. Any UI string
    built INSIDE the context comes out in the document language, so
    keep the scope minimal (no dialogs inside).
    """
    global _translations, _fallback
    saved = (_translations, _fallback)
    _translations, _fallback = _locale_data(locale)
    try:
        yield
    finally:
        _translations, _fallback = saved


def translator_for(locale: str) -> Callable[..., str]:
    """Return a ``t()``-like callable bound to *locale* WITHOUT touching
    the global translation state — used by document generation (PDF
    reports), where the document language may differ from the UI one."""
    data, fallback = _locale_data(locale)

    def _t(key: str, **kwargs) -> str:
        value = _resolve(data, key)
        if value is None:
            value = _resolve(fallback, key)
        if value is None:
            return key
        for k, v in kwargs.items():
            value = value.replace(f"%{{{k}}}", str(v))
        return value

    return _t


# Auto-setup with the default locale; call setup(<locale code>) before
# QApplication to switch.
setup("en")
