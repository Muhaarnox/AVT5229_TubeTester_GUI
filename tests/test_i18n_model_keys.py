"""Model i18n keys must exist in every locale file."""

import json
from pathlib import Path

import pytest

from i18n_setup import available_locales


LOCALES_DIR = Path(__file__).parent.parent / "locales"

REQUIRED_MODEL_KEYS = [
    "Title", "Model_type", "Reference", "Fit",
    "Grid", "Grid_from_data", "Grid_from_scan",
    "Add", "RMS", "No_data", "Fitting", "Fit_done",
    "Not_found", "Label_ref", "Label_fit", "Fit_source",
]

REQUIRED_DEAD_DATA_KEYS = [
    "Title", "Dead_levels", "Partial_levels", "Summary",
    "Clean_before_fit", "Clean_btn", "Clean_confirm",
    "Clean_done", "No_dead_data", "Warning_fit",
]


@pytest.fixture(params=[f"{loc}.json" for loc in available_locales()])
def locale_data(request):
    path = LOCALES_DIR / request.param
    with open(path, encoding="utf-8") as f:
        return json.load(f), request.param


class TestModelI18nKeys:
    """All model.* keys must be present in every locale file."""

    def test_model_section_exists(self, locale_data):
        data, name = locale_data
        assert "model" in data, f"Missing 'model' section in {name}"

    @pytest.mark.parametrize("key", REQUIRED_MODEL_KEYS)
    def test_model_key_present(self, locale_data, key):
        data, name = locale_data
        model = data.get("model", {})
        assert key in model, f"Missing model.{key} in {name}"
        assert model[key].strip(), f"Empty model.{key} in {name}"

    def test_plot_model_key(self, locale_data):
        data, name = locale_data
        plot = data.get("plot", {})
        assert "Model" in plot, f"Missing plot.Model in {name}"

    def test_tip_model_dialog_key(self, locale_data):
        data, name = locale_data
        tip = data.get("tip", {})
        assert "Model_dialog" in tip, f"Missing tip.Model_dialog in {name}"


class TestDeadDataI18nKeys:
    """All dead_data.* keys must be present in every locale file."""

    def test_dead_data_section_exists(self, locale_data):
        data, name = locale_data
        assert "dead_data" in data, f"Missing 'dead_data' section in {name}"

    @pytest.mark.parametrize("key", REQUIRED_DEAD_DATA_KEYS)
    def test_dead_data_key_present(self, locale_data, key):
        data, name = locale_data
        section = data.get("dead_data", {})
        assert key in section, f"Missing dead_data.{key} in {name}"
        assert section[key].strip(), f"Empty dead_data.{key} in {name}"
