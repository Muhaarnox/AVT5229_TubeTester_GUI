"""Tests for app/save_recovery.py — the Retry / Save As / Close loop.

A failed measurement save used to be a dead end (critical dialog, result
in memory until the next test). The recovery loop must offer a real path
to persistence; these tests drive every branch with _ask_recovery patched
(the dialog itself is separated from the loop for exactly this reason).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

import app.save_recovery as sr

QApplication.instance() or QApplication([])

PAYLOAD = {"tube_type": "EL84", "_schema_version": 1, "points": []}


class TestSaveWithRecovery:

    def test_success_needs_no_dialog(self, monkeypatch, tmp_path):
        ask = MagicMock()
        monkeypatch.setattr(sr, "_ask_recovery", ask)
        target = tmp_path / "m.json"
        result = sr.save_with_recovery(None, lambda: target, PAYLOAD, "m.json")
        assert result == target
        ask.assert_not_called()

    def test_retry_calls_save_again_until_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sr, "_ask_recovery",
                            MagicMock(return_value=sr.RETRY))
        target = tmp_path / "m.json"
        save_fn = MagicMock(side_effect=[OSError("disk full"),
                                         OSError("disk full"), target])
        result = sr.save_with_recovery(None, save_fn, PAYLOAD, "m.json")
        assert result == target
        assert save_fn.call_count == 3

    def test_close_keeps_in_memory(self, monkeypatch):
        monkeypatch.setattr(sr, "_ask_recovery",
                            MagicMock(return_value=sr.CLOSE))
        save_fn = MagicMock(side_effect=OSError("nope"))
        result = sr.save_with_recovery(None, save_fn, PAYLOAD, "m.json")
        assert result is None
        save_fn.assert_called_once()   # no silent retry after Close

    def test_save_as_writes_payload_with_schema(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sr, "_ask_recovery",
                            MagicMock(return_value=sr.SAVE_AS))
        alt = tmp_path / "rescued.json"
        monkeypatch.setattr(sr.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(alt), "")))
        save_fn = MagicMock(side_effect=OSError("dir gone"))
        result = sr.save_with_recovery(None, save_fn, PAYLOAD, "m.json")
        assert result == alt
        data = json.loads(alt.read_text(encoding="utf-8"))
        assert data["_schema_version"] == 1     # schema survives Save As
        assert data["tube_type"] == "EL84"

    def test_save_as_cancel_returns_to_dialog(self, monkeypatch):
        ask = MagicMock(side_effect=[sr.SAVE_AS, sr.CLOSE])
        monkeypatch.setattr(sr, "_ask_recovery", ask)
        monkeypatch.setattr(sr.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        result = sr.save_with_recovery(
            None, MagicMock(side_effect=OSError("x")), PAYLOAD, "m.json")
        assert result is None
        assert ask.call_count == 2   # cancel -> dialog shown again

    def test_save_as_failure_loops_with_new_error(self, monkeypatch, tmp_path):
        ask = MagicMock(side_effect=[sr.SAVE_AS, sr.CLOSE])
        monkeypatch.setattr(sr, "_ask_recovery", ask)
        locked = tmp_path / "no_dir" / "x.json"   # parent missing -> OSError
        monkeypatch.setattr(sr.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(locked), "")))
        result = sr.save_with_recovery(
            None, MagicMock(side_effect=OSError("primary")), PAYLOAD, "m.json")
        assert result is None
        # second dialog carries the Save-As error, not the primary one
        assert "primary" not in ask.call_args_list[1][0][1]

    def test_type_error_propagates(self, monkeypatch):
        """Serialization bugs are programming errors — no recovery loop."""
        ask = MagicMock()
        monkeypatch.setattr(sr, "_ask_recovery", ask)
        with pytest.raises(TypeError):
            sr.save_with_recovery(
                None, MagicMock(side_effect=TypeError("not serializable")),
                PAYLOAD, "m.json")
        ask.assert_not_called()


class TestSuggestedFilename:

    def test_windows_unsafe_chars_replaced(self):
        name = sr.suggested_filename("EL84", "L1", "2026-07-02T12:30:00")
        assert ":" not in name
        assert name.endswith(".json")
        assert name.startswith("EL84_L1_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
