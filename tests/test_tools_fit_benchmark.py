"""#23 pins for tools/fit_benchmark.py.

The old benchmark passed tube_type="t" (lookup_tube("t") -> None), so Koren
reference seeding NEVER engaged — with or without --no-ref — and the --no-ref
monkeypatch targeted a module attribute the from-import consumers never read.
Both documented behaviors were fiction; these pins keep the fixed semantics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    """Load fit_benchmark.py as a module (it's a script, not a package)."""
    script_path = PROJECT_ROOT / "tools" / "fit_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "fit_benchmark_under_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fb():
    return _load_module()


class TestRefNameSemantics:
    """Reference participation is controlled by the tube name passed to
    fit_and_export_spice — real name resolves refs, mangled name disables."""

    def _lookup_name_passed(self, fb, use_ref: bool) -> str:
        stub = MagicMock()
        stub.rms_error, stub.max_error, stub.params = 1.0, 2.0, {}
        with patch.object(fb, "fit_and_export_spice",
                          return_value=stub) as spice:
            out = fb._fit_safe([{"ua": 1.0}], "triode", "koren", "EL84",
                               use_ref=use_ref)
        assert out is not None
        return spice.call_args[0][1]

    def test_use_ref_passes_real_tube_name(self, fb):
        """The old code passed "t" — refs never resolved even with refs on."""
        assert self._lookup_name_passed(fb, use_ref=True) == "EL84"

    def test_no_ref_passes_unresolvable_name(self, fb):
        from lm19.tube_params import lookup_tube
        name = self._lookup_name_passed(fb, use_ref=False)
        assert name != "EL84"
        assert lookup_tube(name) is None, \
            f"--no-ref name {name!r} still resolves a reference"

    def test_expected_error_prints_and_returns_none(self, fb, capsys):
        """Failure visibility: a fit error must be visible, not a silent
        'FAIL' cell indistinguishable from a legit non-convergence."""
        with patch.object(fb, "fit_and_export_spice",
                          side_effect=RuntimeError("boom")):
            out = fb._fit_safe([{"ua": 1.0}], "triode", "koren", "X",
                               use_ref=True)
        assert out is None
        assert "boom" in capsys.readouterr().out

    def test_programming_error_propagates(self, fb):
        """API drift (AttributeError/TypeError) must fail loudly, not read
        as 'all models lost' (failure-visibility principle 1)."""
        with patch.object(fb, "fit_and_export_spice",
                          side_effect=AttributeError("api drift")):
            with pytest.raises(AttributeError):
                fb._fit_safe([{"ua": 1.0}], "triode", "koren", "X",
                             use_ref=True)


class TestSmoke:
    def test_benchmark_runs_on_small_dataset(self, fb, capsys):
        """End-to-end on the smallest real dataset (E88CC, 48 pts): table
        row + summary printed, no fitter FAILs."""
        fb.run_benchmark(filters=["e88cc_datasheet"])
        out = capsys.readouterr().out
        assert "Summary:" in out
        assert "E88CC" in out
        assert "FAIL" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
