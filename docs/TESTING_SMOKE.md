# Smoke Testing Guide

This project uses `pytest` markers to keep critical-path checks fast and focused.

## Main Smoke Run

Run all smoke tests:

`python -m pytest -m smoke -q`

## Smoke Categories

Each smoke test also has one category marker:

- `smoke_ui` - app bootstrap and UI-adjacent flows
- `smoke_data` - import/export and measurement persistence
- `smoke_protocol` - protocol parse/encode/decode checks
- `smoke_analysis` - amplifier and analysis pipeline checks
- `smoke_workers` - worker thread logic checks (Scan, Reset, Poller, Preheat, Health)
- `smoke_spice` - SPICE fit/export checks
- `smoke_scan` - scan counting and scan-related logic
- `smoke_config` - config, i18n bootstrap, and locale validation

## Useful Commands

Run one category:

- `python -m pytest -m "smoke and smoke_ui" -q`
- `python -m pytest -m "smoke and smoke_data" -q`
- `python -m pytest -m "smoke and smoke_protocol" -q`
- `python -m pytest -m "smoke and smoke_analysis" -q`
- `python -m pytest -m "smoke and smoke_workers" -q`
- `python -m pytest -m "smoke and smoke_spice" -q`
- `python -m pytest -m "smoke and smoke_scan" -q`
- `python -m pytest -m "smoke and smoke_config" -q`

Run all smoke tests except one category:

- `python -m pytest -m "smoke and not smoke_spice" -q`

List collected smoke tests:

- `python -m pytest -m smoke --collect-only -q`

## Notable Tests

### Health Workers (`test_workers_smoke.py`)

- `test_health_worker_happy_path` — verifies `HealthWorker._execute()` emits `finished` with valid measurement structure.
- `test_health_worker_error_emits_failed` — verifies client errors are caught and `failed` signal is emitted.

### i18n Validation (`test_config_i18n_smoke.py`)

- `test_no_bare_interpolation_vars` — catches `{var}` format (must be `%{var}`).
- `test_locales_have_same_keys` — parametrized over every `locales/*.json`: each
  locale must have exactly the key set of `en.json` (the reference), no missing
  and no extra keys. A new translation file is picked up automatically.
- `test_interpolation_vars_consistent` — ensures `%{var}` placeholders match between locales.

### Amplifier Control Panel (`test_amp_control_panel.py`)

- Panel creation, circuit combos, parameter spins, params_snapshot per circuit type.
- Bidirectional spin sync (MainWindow._sync_spin_pair) — no infinite loop.
- settings_changed signal: emits on value change, suppressed during set_series_items.
- Circuit widget visibility: SE/SE_XFMR/CF/PP show/hide correct widgets.
- set_series_items, set_pp_tube_b_items, set_available_models (preserves unchecked state).

### Amplifier Tab Regression (`test_amplifier_tab_regression.py`)

- AmplifierTab.render() with various AnalysisResult shapes.
- Format edge cases: swing_clamped, insufficient_signal, multi_source, zero_ref div-by-0.
- PP-specific: no pp_dist graceful handling.
- Source color mapping, clear resets both plots.

### CheckableComboBox (`test_checkable_combo.py`)

- set_items (float keys), set_string_items (string keys), checked_keys.
- set_all_checked, selectionChanged signal suppression during bulk ops.
- _summary_text edge cases: empty, all selected, partial, none selected.

### AmplifierEngine (`test_amp_engine.py`)

- Multi-source end-to-end: each source gets dist/sweep/headroom.
- Edge cases: empty engine, extreme bias, very small/large Ra, swing near zero.
- NFB without gain, DFT without model, set_data replaces previous data.
- Sweep edge cases: produces lists, CF disables Ra sweep.

---

## Developer Tools

### Fit Benchmark (`tools/fit_benchmark.py`)

Compare Koren / Dempwolf / Reefman accuracy across all test datasets.

```bash
python tools/fit_benchmark.py                    # all datasets (89 as of 2026-08-01)
python tools/fit_benchmark.py EL84               # filter by tube name
python tools/fit_benchmark.py pentode            # filter by topology
python tools/fit_benchmark.py 6S19P 6C33C        # multiple filters (OR)
python tools/fit_benchmark.py --real             # only real measurements (*_real.json)
python tools/fit_benchmark.py --no-ref           # disable reference params (fair comparison)
python tools/fit_benchmark.py --real --no-ref    # combine flags
```

Output: table with RMS error per model, winner, Dempwolf/Koren ratio.
Test data: `tests/spice_test_data/converted/`. Real measurements have `_real.json` suffix.

Key findings (run of 2026-08-01, 89 datasets — re-run the tool rather than
trusting these numbers, they move with every fitter change):

| Topology | Datasets | Winners |
|---|---:|---|
| Triode | 80 | Koren 66, Dempwolf 14, Reefman 0 |
| Pentode | 9 | Reefman 5, Dempwolf 4, Koren 0 |

So: Koren is the triode workhorse; for pentodes Reefman and Dempwolf are at
parity and Koren never wins. One dataset is an out-of-class outlier where every
model fails (THF51 — emission saturation at multi-ampere currents); it is kept
in the corpus on purpose, see `docs/DEMPWOLF_EXTENDED_MODEL.md` §10.8.

### Model Compare (`python -m lm19.model_compare`)

Compare overlay model fitters on measurement files in `measurements/`.
Outputs formatted table with rms_ia, max_ia, rms_ig2, rms_gm per model.

### LTspice Round-Trip (`tests/test_ltspice_roundtrip.py`)

Exports model → runs LTspice batch → parses .raw → compares Ia with Python.
Requires LTspice at `C:\Program Files\ADI\LTspice\LTspice.exe`.

```bash
python -m pytest tests/test_ltspice_roundtrip.py -v    # all 54 round-trip tests
```
