# Configuration Reference

Unified reference for LM19 project configuration files.

---

## Config Files Map

| File | Scope |
|---|---|
| `config/app.json` | General application settings (serial, timeouts, UI, logging, amplifier UI) |
| `config/scan.json` | I-V scan parameters (`scan_*`) |
| `config/srk.json` | SRK measurement parameters (`srk_*`) |
| `config/health.json` | Tube Health parameters |
| `config/calibration.json` | Calibration coefficients (gain/offset per channel, meter accuracy) |
| `config/device.json` | Device hardware limits (ua_max, ig2_max, etc.) |
| `config/lamps.json` | Tube database (nominals, topology, ranges, limits) |
| `config/lamp_limits.json` | Source/set of tube limits (including sync tools) |
| `config/tube_params.json` | Model parameters (Koren, caps, topology aliases) |

---

## `config/device.json` (Device Hardware Limits)

Hardware limits of the AVT5229 tester. Used for range clamping, safety checks, and calibration wizard.

| Key | Type | Default | Description |
|---|---|---:|---|
| `ua_max` | float | `300.0` | Maximum anode voltage, V |
| `ug2_max` | float | `300.0` | Maximum screen grid voltage, V |
| `ug1_max` | float | `24.0` | Maximum control grid voltage (absolute), V |
| `uh_max` | float | `15.0` | Maximum heater voltage, V |
| `ih_max` | float | `2.5` | Maximum heater current, A |
| `ia_max` | float | `200.0` | Maximum anode current, mA |
| `ig2_max` | float | `25.0` | Maximum screen current (hardware trip), mA |

The `Default` column shows the value shipped in `config/device.json`.
Fallback: if the file is missing, built-in `DEFAULT_LIMITS` (`lm19/config.py`)
are used — identical except `ig2_max = 20.0`.

---

## `config/app.json` (General)

| Key | Type | Default | Description |
|---|---|---:|---|
| `locale` | str | `"en"` | UI language — any code that has a `locales/<code>.json` file. Locales are auto-discovered, so adding a file is enough; `en` is the fallback for missing keys |
| `settle_s` | float | `0.3` | Base stabilization delay (global) |
| `default_com_port` | str | `"COM5"` | Preselected COM port |
| `live_poll_ms` | int | `500` | Live parameter polling interval, ms |
| `live_poll_during_test` | bool | `true` | Keep the live panel polling during scans / health tests. The port lock keeps every read atomic and `?Er;` does not clear the firmware flag, so this is safe — but a poll cycle costs ~140 ms of bus time at 9600 baud, so a measurement worker waits longer for the lock (roughly +25-30 % run time at a 500 ms period). Set to `false` to restore the pause-while-measuring behaviour |
| `manual_heater_tolerance_pct` | float | `10.0` | Heater deviation (%, both directions) tolerated by the Manual tab `Apply All` gate. Checked twice: the heater setpoint against the selected lamp's rated heater, and the live reading against that setpoint. Beyond it the action asks for confirmation and applies nothing if declined |
| `serial_timeout_s` | float | `0.3` | Serial read timeout |
| `serial_write_timeout_s` | float | `0.3` | Serial write timeout |
| `read_param_timeout_s` | float | `1.0` | Device parameter read timeout |
| `read_lcd_timeout_s` | float | `2.0` | LCD data read timeout |
| `serial_set_param_delay_s` | float | `0.05` | Delay after `set_param` at protocol level |
| `plot_ia_max` | float | `100.0` | Upper Ia limit on plots (mA) |
| `ug1_after_stop` | float | `-24.0` | Ug1 after stop/reset. Also used as the **safe lock** starting point for the Tube Health OP-approach ramp (`health.json:op_ramp_enabled`) — tube is biased fully closed at this Ug1 before Ua/Ug2 are applied |
| `ug1_settle_s` | float | `0.5` | Settle after setting the blocking Ug1 during output reset (post-scan / emergency, `ResetWorker`). SRK settle is governed by `srk.json:settle_per_volt_s/settle_base_s` instead (ML-136: the dead pass-through into SRK was removed) |
| `ug1_verify_tolerance` | float | `0.1` | Ug1 verify tolerance |
| `heater_zero_warn_uh_v` | float | `0.1` | Uh warning threshold at zero |
| `heater_zero_warn_ih_a` | float | `0.1` | Ih warning threshold at zero |
| `amp_sweep_amplitude_steps` | int | `40` | Number of amplitude steps (Amplifier) |
| `amp_sweep_ra_min_factor` | float | `0.2` | Lower Ra sweep factor |
| `amp_sweep_ra_max_factor` | float | `5.0` | Upper Ra sweep factor |
| `amp_sweep_ra_min_abs_kohm` | float | `0.5` | Lower absolute Ra sweep bound, kOhm |
| `amp_sweep_ra_max_abs_kohm` | float | `100.0` | Upper absolute Ra sweep bound, kOhm |
| `amp_sweep_ra_steps` | int | `60` | Number of Ra sweep steps |
| `amp_opt_ra_min_factor` | float | `0.1` | Lower Ra optimization factor |
| `amp_opt_ra_max_factor` | float | `10.0` | Upper Ra optimization factor |
| `amp_opt_ra_min_abs_kohm` | float | `0.5` | Lower absolute Ra optimization bound, kOhm |
| `amp_opt_ra_max_abs_kohm` | float | `200.0` | Upper absolute Ra optimization bound, kOhm |
| `amp_opt_ra_steps` | int | `100` | Number of Ra optimization steps |
| `ra_dialog_max_factor` | float | `4.0` | Upper factor in Ra dialog |
| `ra_dialog_max_abs_kohm` | float | `20.0` | Upper absolute bound in Ra dialog |
| `ra_dialog_min_abs_kohm` | float | `0.5` | Lower absolute bound in Ra dialog |
| `ra_dialog_steps` | int | `80` | Number of steps in Ra dialog |
| `marker_lock_px` | int | `10` | Marker snap-to-curve radius, px |
| `ua_cluster_threshold` | float | `2.0` | Max Ua spread (V) merged into one grid column; increase if heatmap has gaps |
| `ug1_cluster_threshold` | float | `0.3` | Max Ug1 spread (V) merged into one curve; increase if close setpoints split |
| `ug2_cluster_threshold` | float | `2.0` | Max Ug2 spread (V) merged into one curve; increase if close setpoints split |
| `ig2_hw_margin_pct` | int | `80` | Ig2 hardware safety margin (% of device.json ig2_max). Clamped [10..90]. Pentode/track only |
| `ia_dead_threshold` | float | `0.30` | Minimum Ia (mA) for a measured point to enter model fitting (`ModelDialog` → `extract_arrays(ia_thr_mA=...)`). Points below are treated as dead/noise and dropped |
| `cal_measure_samples` | int | `10` | Default N for calibration wizard averaging |
| `cal_measure_interval_ms` | int | `200` | Interval between wizard samples, ms |
| `debug_params` | bool | `false` | Enable parameter debug logging |
| `debug_params_file` | str | `"logs/params_debug.log"` | Parameter debug log file |
| `log_file` | str | `"logs/lm19.log"` | Main log file |
| `log_level` | str | `"INFO"` | Console/main log level |
| `log_file_level` | str | `"DEBUG"` | File log level |
| `measurements_dir` | str | `"measurements"` | Where I-V scan measurements are stored. Relative paths resolve against `lm19_app/`. Absolute paths used as-is. Auto-created with WARNING in log if missing |
| `health_measurements_dir` | str | `"health_measurements"` | Where Tube Health measurements are stored. Same resolution rules as `measurements_dir` |
| `health_refs_dir` | str | `"config/health_refs"` | Root of the Tube Health reference set — holds both `type/<TubeType>/<ref_id>.json` (curated type references) and `personal/<TubeType>/<LampID>.json` (per-tube baselines). Same resolution rules as `measurements_dir`. Defaults under `config/` because type refs are curated instrument settings rather than captured data; point it elsewhere to keep the reference set on a shared drive or outside the install |
| `compare_matching_algorithm` | str | `"optimal"` | Pair-matching strategy for Compare-tab curve matching (group_size=2 only). `"optimal"` = Hungarian (legacy default, preserves backward compat). `"greedy"` = tightest-first (same trade-off as Health `matching_algorithm`). Independent from Health setting. |
| `report_language` | str | `""` | PDF-report/certificate text language (a `locales/*.json` code) — the ONLY place it is set, there is no UI selector. `""` = English; an unknown code falls back to English with a log WARNING. Independent from the UI `locale` |
| `report_sections` | str | `""` | Default enabled PDF-report sections, CSV of ids from `REPORT_SECTIONS` (`app/report.py`), e.g. `"nominal,srk,plot_curves"`. `""` = all sections. Unknown ids are dropped with a WARNING |
| `report_ask` | bool | `true` | Show the PDF-report options dialog before each export. `false` = export silently with the defaults above (availability-clipped) |
| `ltspice_verify_dir` | str | `""` | Working directory for LTspice verification runs. `""` = system temp (`%TEMP%\lm19_verify_*`); else a per-run `verify_<timestamp>` subdir is created under it. Relative paths resolve against `lm19_app/`. Netlist/`.log`/`.sub` are kept for manual re-runs; bulky `.raw` waveforms are deleted after parsing |
| `ltspice_exe` | str | `""` | LTspice executable override. `""` = standard install path (`C:\Program Files\ADI\LTspice\LTspice.exe`) |

---

## `config/scan.json`

| Key | Type | Default | Description |
|---|---|---:|---|
| `ua_settle_per_volt_s` | float | `0.0025` | Dynamic Ua settle, s/V |
| `ua_settle_base_s` | float | `0.15` | Base Ua settle, s |
| `ua_tolerance` | float | `1.0` | Ua verify tolerance, V |
| `ua_retries` | int | `2` | Ua verify retries |
| `ug1_settle_per_volt_s` | float | `0.1` | Dynamic Ug1 settle, s/V |
| `ug1_settle_base_s` | float | `0.15` | Base Ug1 settle, s |
| `ug1_tolerance` | float | `0.1` | Ug1 verify tolerance, V |
| `ug1_retries` | int | `2` | Ug1 verify retries |
| `ug2_settle_per_volt_s` | float | `0.0025` | Dynamic Ug2 settle, s/V |
| `ug2_settle_base_s` | float | `0.15` | Base Ug2 settle, s |
| `ug2_tolerance` | float | `1.0` | Ug2 verify tolerance, V |
| `ug2_retries` | int | `2` | Ug2 verify retries |
| `ia_samples` | int | `3` | Number of Ia/Ig2 reads per point (>=3 enables robust averaging) |
| `ia_outlier_ratio` | float | `2.0` | Ia outlier threshold: if max/min > ratio, re-read extra samples. 0 = disabled. Check is armed only when max(Ia) > 0.5 mA (`_IA_OUTLIER_FLOOR`) |
| `ia_outlier_reread_samples` | int | `3` | Extra Ia samples read after an outlier. `0` = warn and count only, no re-read. Pool size picks the averaging rule: 3–4 → median, ≥5 → trimmed mean |
| `pa_over_pct` | float | `20` | Pa over-limit tolerance, %. 0 = disabled |
| `pig2_over_pct` | float | `20` | Pig2 over-limit tolerance, % |
| `refine_enabled` | bool | `false` | Enable adaptive refine pass |
| `refine_max_depth` | int | `2` | Maximum bisection depth |
| `refine_min_step_ua` | float | `3` | Minimum Ua step after refine, V |
| `refine_onset_ma` | float | `0.5` | Ia onset threshold, mA |
| `refine_curvature_thr` | float | `0.15` | Refine curvature threshold |
| `refine_gradient_ratio` | float | `3.0` | Gradient ratio threshold |
| `refine_ig2_delta_min` | float | `0.5` | Minimum Ig2 jump, mA |
| `refine_delta_ia_thr` | float | `0.25` | Ia jump threshold (relative) |
| `comm_retries` | int | `2` | Silent auto-retries on UART comm error before asking user |
| `down_max_step_v` | float | `25` | Max Ua step in pentode down-sweep, V. Larger steps are bisected. 0 = disabled |

---

## `config/srk.json`

| Key | Type | Default | Description |
|---|---|---:|---|
| `samples` | int | `5` | Number of reads per SRK point |
| `settle_s` | float | `1.0` | Base settle for SRK step, s |
| `verify_retries` | int | `3` | Verify retries in SRK |
| `ua_tolerance` | float | `1.0` | Ua verify tolerance in SRK, V |
| `ug2_tolerance` | float | `1.0` | Ug2 verify tolerance in SRK, V |
| `settle_per_volt_s` | float | `0.5` | Dynamic settle for Ug1 (SRK), s/V |
| `settle_base_s` | float | `1.0` | Base settle for Ug1 (SRK), s |
| `ug1_step` | float | `0.04` | Ug1 step in SRK sweep mode, V |

---

## `config/health.json`

| Key | Type | Default | Description |
|---|---|---:|---|
| `ua_tolerance_v` | float | `1.0` | Ua verify tolerance in Health, V |
| `ug1_tolerance_v` | float | `0.2` | Ug1 verify tolerance in Health, V |
| `ug2_tolerance_v` | float | `1.0` | Ug2 verify tolerance in Health, V |
| `ua_retries` | int | `2` | Ua verify retries |
| `ug1_retries` | int | `2` | Ug1 verify retries |
| `ug2_retries` | int | `2` | Ug2 verify retries |
| `ua_settle_per_volt_s` | float | `0.0025` | Dynamic Ua settle, s/V |
| `ua_settle_base_s` | float | `0.5` | Base Ua settle, s |
| `ug1_settle_per_volt_s` | float | `0.2` | Dynamic Ug1 settle, s/V |
| `ug1_settle_base_s` | float | `0.3` | Base Ug1 settle, s |
| `ug2_settle_per_volt_s` | float | `0.0025` | Dynamic Ug2 settle, s/V |
| `ug2_settle_base_s` | float | `0.15` | Base Ug2 settle, s |
| `ug1_delta_v` | float | `1.0` | Offset for Ug1+-Delta points |
| `delta_pct` | int | `5` | Default δV as % of OP voltage for S/R |
| `delta_ua_min_v` | float | `10.0` | Minimum δUa (V) — floor for Rp measurement SNR |
| `delta_ua_max_v` | float | `50.0` | Maximum δUa (V) — ceiling to limit non-linearity |
| `delta_ug1_min_v` | float | `0.5` | Minimum δUg1 (V) — safe floor per RCA standard |
| `delta_ug1_max_v` | float | `2.0` | Maximum δUg1 (V) — keeps gm measurement local |
| `delta_ug2_pct` | int | `5` | Default δVg2 as % of Ug2 for Sg2 (pentode mode) |
| `delta_ug2_min_v` | float | `5.0` | Minimum δUg2 (V) |
| `delta_ug2_max_v` | float | `50.0` | Maximum δUg2 (V) |
| `ia_samples` | int | `5` | Number of Ia reads per point |
| `ia_sample_delay_ms` | int | `50` | Delay between consecutive Ia reads within one point, ms |
| `ig2_samples` | int | `5` | Number of Ig2 reads per point |
| `use_median` | bool | `true` | Robust averaging (median) |
| `outlier_trim_count` | int | `1` | Number of extreme outliers to trim |
| `emission_enabled_default` | bool | `true` | Emission step enabled by default |
| `preheat_required_ratio` | float | `0.75` | Required heater level ratio for Tube Health start when preheat is not marked done |
| `emission_uh_ratio` | float | `0.8` | Uh ratio for Ia80 |
| `emission_restore_uh_timeout_s` | int | `20` | Timeout to restore Uh to nominal |
| `emission_low_confidence_on_timeout` | bool | `true` | Mark result as low confidence on timeout |
| `emission_stable_warmup_ratio` | float | `0.5` | Fraction of warmup_s used for stabilization t_max |
| `emission_stable_min_s` | int | `20` | Minimum stabilization t_max |
| `emission_stable_max_s` | int | `120` | Maximum stabilization t_max |
| `emission_stable_slope_threshold_ma_per_s` | float | `0.01` | Stabilization threshold by dIa/dt |
| `emission_stable_window_points` | int | `5` | Point window for stabilization estimation |
| `emission_sample_period_s` | float | `2.0` | Ia read period during stabilization |
| `weight_ia` | float | `0.35` | Weight of Ia% metric |
| `weight_s` | float | `0.40` | Weight of S% metric |
| `weight_rh` | float | `0.10` | Weight of Rh metric |
| `weight_screen` | float | `0.0` | Weight of screen ratio (pentodes). Shipped as `0` — screen ratio is reported but does not move the index until you give it weight |
| `weight_emission` | float | `0.15` | Weight of emission ratio |
| `renormalize_weights_if_metric_missing` | bool | `true` | Renormalize weights when metrics are missing |
| `verdict_strong_min` | float | `90.0` | Lower bound for Strong |
| `verdict_good_min` | float | `75.0` | Lower bound for Good |
| `verdict_weak_min` | float | `55.0` | Lower bound for Weak |
| `emission_ratio_nominal` | float | `0.90` | Expected emission ratio for a healthy tube — scoring baseline for `emission_score`, used only as **fallback** when the active reference carries no `emission_ratio` |
| `emission_ratio_good_min` | float | `0.70` | Lower bound of the absolute `normal` emission verdict (`metrics.emission_verdict`) |
| `emission_ratio_weak_min` | float | `0.50` | Lower bound of `weakened`; below it the verdict is `exhausted` |
| `emission_min_ik_ratio` | float | `0.30` | Below this fraction of `lamp.ia_max` the reduced-heater probe cannot see a depleted cathode (tube stays space-charge limited) — the result is flagged `emission_low_sensitivity` |
| `emission_mode_default` | str | `"single"` | Default emission mode: `single` (one reduced point) or `sweep` (walk Ia(Uh), locate the knee). Codes from `EMISSION_MODES` in `lm19/health.py` |
| `emission_uh_sweep_steps` | int | `5` | Number of heater points in sweep mode (the configured `emission_uh_ratio` is always added on top) |
| `emission_uh_sweep_min_ratio` | float | `0.70` | Lowest heater ratio of the CONFIGURED grid; the adaptive Miram descent may continue below it (see `emission_uh_sweep_abs_min_ratio`) |
| `emission_uh_sweep_abs_min_ratio` | float | `0.50` | Absolute floor of the adaptive descent: past the grid the sweep keeps stepping down (same pace) until the falling branch holds two points, but never below this ratio — an underheated cathode loses its space-charge shield against ion bombardment, and Ia becomes too small to measure. The last step clamps TO the floor |
| `emission_knee_drop_pct` | float | `10.0` | Plateau-membership criterion of the two-line Miram fit: a swept point stays on the (possibly tilted) plateau line while it sits within half this percentage of the line's extrapolation; the first point that falls away starts the emission branch. The knee itself is the intersection of the two fitted lines, not a threshold crossing |
| `emission_sweep_max_total_s` | float | `600.0` | Wall-clock budget for the sweep; on overrun the curve is truncated, flagged and confidence drops |
| `bias_servo_enabled_default` | bool | `false` | Default state of the "Bias to reference Ia" plan checkbox |
| `bias_servo_tol_ma` | float | `0.5` | Ia tolerance for servo convergence |
| `bias_servo_max_shift_v` | float | `6.0` | **Absolute ceiling** on the Ug1 excursion. The WORKING limit is per-tube: `bias_servo_shift_margin × |ref_ia − Ia| / lamp.s`, capped by this key (6 V covers a 6L6-class tetrode at ~50% wear; a 12AX7 auto-limits to ~0.5 V). Lamps without a usable `s` fall back to this ceiling with a log WARNING |
| `bias_servo_shift_margin` | float | `2.0` | Multiplier on the estimated shift `deficit/S` — covers a tube whose real S sagged below the datasheet value (×2 = honest down to ~50% of datasheet S). The applied limit is stored as `health.bias_servo.shift_limit_v` |
| `bias_servo_step_v` | float | `0.5` | Walk step toward the reference current. Overshoot is bounded by S·step (~6 mA on an EL84); the 2026-08-01 field incident (Pa trip at ~2× nominal Ia) was the old edge-first probe jumping the whole excursion |
| `bias_servo_max_iter` | int | `12` | Total probe budget: walk (≤ max_shift/step) + bisection inside the last step |
| `bias_servo_pa_ceiling_pct` | float | `90.0` | Walking up stops once Pa exceeds this % of the protection trip limit (`pa_max × pa_safety_pct`) — the trip stays a backstop, never the expected stop. Not applied walking down (Pa falls) |
| `bias_servo_ug1_floor_v` | float | `0.1` | Bisection stops when the bracket is below this width — the hardware cannot honor sub-resolution Ug1 setpoints. On a floor stop the servo **accepts the closest probe seen** (residual bounded by S×floor, ~1.1 mA on an EL84) instead of reporting a false unreachable: with `tol_ma` finer than the S×floor quantum the tolerance alone can never be met (2026-08-02 field report) |
| `op_ramp_enabled` | bool | `true` | OP-approach Ug1 ramp + Pa/Pg2 safety protection. Set to `false` only for emergency debugging — disables tube protection during `_setup_op` |
| `op_ug1_ramp_step_v` | float | `1.0` | Ug1 ramp step (V) from safe lock (`app.json:ug1_after_stop`) to target. Smaller = safer + slower; ~17 steps at default for typical −7 V bias |
| `pa_safety_pct` | float | `135.0` | Pa safety threshold as % of `lamp.pa_max`. Trips at `Pa > pa_max × pa_safety_pct/100` during OP ramp. Skipped when `lamp.pa_max=None` |
| `pig2_safety_pct` | float | `120.0` | Pg2 safety threshold as % of `lamp.pig2_max`. Skipped for triodes (Ug2=0) and when `lamp.pig2_max=None` |

**Tube Matching defaults:**

| Key | Type | Default | Description |
|---|---|---|---|
| `matching_weight_ia` | float | `0.5` | Weight of Ia in matching distance (pentode default) |
| `matching_weight_s` | float | `0.5` | Weight of S in matching distance (pentode default) |
| `matching_weight_r` | float | `0.0` | Weight of R in matching distance (pentode default; Rp irrelevant for pentodes) |
| `matching_max_delta` | float | `0.0` | Max allowed Δ (0 = no limit) |
| `matching_group_size` | int | `2` | Default group size (2=pairs, 4=quads) |
| `matching_use` | string | `"latest"` | Which measurement per lamp: `latest` or `best` |
| `matching_anode` | string | `"each"` | Anode mode: `each` or `combined` |
| `matching_algorithm` | string | `"greedy"` | Pair-matching strategy (group_size=2 only). `"greedy"` = tightest-first (pick globally closest pair, lock, repeat — best for "select N best pairs from a box, return the rest"). `"optimal"` = Hungarian min-sum (best when every tube must end up in some pair within Max Δ) |
| `matching_protocol` | string | `"strict"` | What "matched" means for the target amplifier: `strict` — servo and fixed-bias runs never mix (exact conditions match); `shared_bias` — one bias adjustment for both tubes: Ia term uses the plan-point current, servo/fixed runs mix, δIq gate applies; `individual_bias` — per-tube bias adjustment: Ia weight zeroed, S/R decide, only servo runs qualify, adjustment-range gate applies. Pre-selects the Match-panel dropdown |
| `matching_max_iq_imbalance_pct` | float | `10.0` | `shared_bias` gate: max predicted quiescent-current imbalance δIq = \|ia_plan₁ − ia_plan₂\| as a percent of the pair's mean plan current. Violating pairs turn incomparable BEFORE selection. 0 = off |
| `matching_bias_adjust_range_pct` | float | `30.0` | `individual_bias` gate: the amplifier's bias-adjustment authority as a percent of the plan bias voltage — each tube's \|Δbias\| must fit it. Percent, not volts (scales from a −2 V triode to a −60 V transmitting tube). 0 = off |

These `matching_*` keys apply to the **Health** tab only. Compare-tab matching
settings (Mode, Class, Size, Max Δ, Min pts) are UI-only controls with no config
keys, except `app.json:compare_matching_algorithm`.

Triode defaults (applied automatically when tube mode is triode/triode_connected):
Ia=0.4, S=0.3, R=0.3. R weight is higher for triodes because Rp determines µ.

**Health tab** delta quality thresholds (single-point Ia/S/R matching):
≤2% excellent, ≤5% good, ≤10% fair, >10% poor.

**Compare tab** delta quality thresholds depend on amp class (curve matching):
- Class A: ≤2% / ≤5% / ≤10%  (tight — knee not reached)
- Class AB: ≤5% / ≤10% / ≤20%  (balanced — knee contributes)
- Class B: ≤8% / ≤15% / ≤30%  (relaxed — full curve matters)

Emission stabilization limit formula:
`t_max_stable = clamp(emission_stable_warmup_ratio * warmup_s, emission_stable_min_s, emission_stable_max_s)`

---

## `config/calibration.json`

Version 2 schema. Auto-created with defaults if missing.

**Top-level fields:**

| Key | Type | Description |
|---|---|---|
| `version` | int | Schema version (currently `2`) |
| `meter_accuracy_pct` | dict | External meter accuracy ±% per channel |
| `channels` | dict | Per-channel calibration coefficients |

**`meter_accuracy_pct` defaults:**

| Channel | Default ±% |
|---|---|
| `ua`, `ug1`, `ug2`, `uh` | 0.5 |
| `ih`, `ia_low`, `ia_high`, `ig2` | 1.0 |

**`channels` entries** (13 total: 8 READ + 5 SET):

| Key | Type | Default | Description |
|---|---|---|---|
| `gain` | float | `1.0` | Linear correction gain |
| `offset` | float | `0.0` | Linear correction offset |
| `calibrated_at` | str\|null | `null` | ISO 8601 timestamp of last calibration |
| `quality` | dict\|null | `null` | Quality metadata from wizard |

READ channels: `ua_read`, `ug1_read`, `ug2_read`, `uh_read`, `ih_read`, `ia_low_read`, `ia_high_read`, `ig2_read`.
SET channels: `ua_set`, `ug1_set`, `ug2_set`, `uh_set`, `ih_set`.

SET coefficients are auto-derived from READ by the calibration wizard
(`derive_set_two_point`) and applied as feedforward to every
working-point command (plan B, `docs/CALIBRATION_PLAN.md`); they are
read-only in the Calibration tab manual editor. Ug1 SET works in the
negative physical domain (clamp range −24…0 V).

Migration: v1 `ia_read` → `ia_low_read` + `ia_high_read` (automatic on load).

Load validation: any stored fit outside sanity bounds
(`fit_within_bounds`: gain 0.8–1.2, offset per `OFFSET_BOUNDS`) is
reset to default with a WARNING — with feedforward a bad coefficient
(corrupted or hand-edited file) would drive real commands.

---

## `config/lamps.json`

Contains an array of tubes with fields:
  - identification: `type`, `socket`, `topology`, `anodes`, `anode_default`;
  - nominal values: `uh`, `ih`, `ug1`, `ua`, `ia`, `ug2`, `ig2`, `s`, `r`, `k`;
  - warmup/UI: `warmup_s`;
  - scan ranges: `ranges.ua|ug1|ug2 (min/max/step)`;
  - limits: `Pa_max`, `Pig2_max`, `ua_max`, `ia_max`, `uh_max`, `ih_max`,
    `ug2_max`, `Ra`. Units/semantics (consumed by `load_lamps` since
    2026-07-02, ML-130/131): `uh_max` — V, stored WITH the sync tool's +10%
    headroom; `ih_max` — **mA**, holds the TDSL *nominal* heater current
    (the loader converts to A and applies the same +10% headroom);
    `ug2_max` — V. Each is capped by the device limit and lands in
    `LampConfig.limits`, clamping the heater inputs and the Ug2 scan spins
    when the lamp is selected.

---

## `config/lamp_limits.json`

Contains normalized limits by tube type and value sources (used by the tools-sync pipeline and `lamps.json` revision).

---

## `config/tube_params.json`

Contains parametric model data:
- `koren` (`mu`, `ex`, `kg1`, `kp`, `kvb`, `kg2`, etc.);
- parasitic capacitances `caps_pF`;
- additional parameters (`rgi`, `vct`, `topology`, `aliases`, `source`).

---

## `config/templates/`

LTSpice template files used by `lm19/ltspice_asc.py` for test schematic generation:

| File | Description |
|------|-------------|
| `test_triode.asc` | Triode test schematic (V1 Ua sweep, V2 Ug1 step, `.dc` directive) |
| `test_pentode.asc` | Pentode test schematic (+V3 Ug2) |
| `triode.asy` | Triode symbol (3 pins: A, G, K with SpiceOrder 1, 2, 3) |
| `pentode.asy` | Pentode symbol (4 pins: A, G, K, G2 with SpiceOrder 1, 2, 3, 4) |

Templates use `{placeholder}` substitution: `{tube_name}`, `{sub_file}`,
`{ua_max}`, `{ua_step}`, `{ug1_start}`, `{ug1_stop}`, `{ug1_step}`, `{ug2}`.

---

## Notes

- Tube Health architecture plan: `docs/TUBE_HEALTH_PLAN.md`.
- Historical decisions/details for health config: `docs/TUBE_HEALTH_CONFIG.md`.
