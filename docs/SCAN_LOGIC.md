# Scan Logic — Full IV Scan Algorithm

## Overview

The scan measures tube IV characteristics by stepping through voltage ranges
and reading anode current (Ia) at each point. Three sweep modes exist,
selected by tube topology and UI settings.

Entry point: `run_scan()` in `lm19/scan/runner.py`. The `lm19/scan/`
package is split into: `settings` (ScanRange/ScanSettings/defaults),
`exceptions`, `io` (param set/read), `protection` (heater & limit checks),
`refine` (adaptive refinement), `sweepers` (3 sweep modes) and `runner`
(orchestrator). SRK measurement lives in standalone `lm19/srk.py`.

## Common Setup (all modes)

1. Set amplifier channel (`An`), heater (`Uh`/`Ih`)
2. Compute protection limits from settings:
   - `pa_limit = pa_max_w × (1 + pa_over_pct / 100)` — anode power limit (W)
   - `pg2_limit = pig2_max_w × (1 + pig2_over_pct / 100)` — screen power limit (W)
   - `ig2_limit = ig2_max_ma` — screen current hard limit (mA)
   - If `pa_max_w = 0` → `pa_limit = 0` (Pa protection disabled)
   - If `pa_over_pct = 0` and `pa_max_w > 0` → `pa_limit = pa_max_w` (exact limit, no margin)
3. Generate voltage grids: `ua_values` (ascending), `ug1_values` (ascending: closed → open)
4. Read current real values for initial settle-time calculation

## Voltage Ranges and Directions

UI fields are labelled **Min / Max** — user cannot set Min > Max.

| Parameter | Direction | Physical meaning |
|-----------|-----------|------------------|
| `ua_values` | Ascending (low → high) | Anode voltage grows |
| `ug1_values` | Ascending (e.g. −20 → −2) | Grid opens, Ia grows |
| `ug2_values` | Ascending (low → high) | Screen voltage grows |

## Per-Point Operations

### `_set_param_calibrated()` → `_set_param_with_settle()`

Scan, SRK and health set working points through the calibration adapter
`_set_param_calibrated` (feedforward, plan B — `docs/CALIBRATION_PLAN.md`):

1. `cmd = apply_set(channel, target_phys)` — pre-corrected command, once
2. `expected = read_inverse(channel, target_phys)` — verify target
3. Call the raw pipe `_set_param_with_settle(..., verify_target=expected)`
4. Return `apply_read(channel, decoded actual)` — physical actual

The raw pipe `_set_param_with_settle` (no calibration inside):

1. Encode and send target value
2. Wait: `settle = |target − prev| × settle_per_volt_s + settle_base_s`,
   slept in ~0.05 s chunks via `_interruptible_sleep` when a `stop` callback
   is supplied — Cancel becomes responsive even on a multi-second settle
   (SRK Ug1 can reach ~28 s). On cancel the function returns the device-domain
   `expected` value (so the calibrated wrapper's `apply_read` round-trips to
   the physical target and no spurious verify error fires) and the worker
   exits at its next boundary.
3. Read back actual value, compare against `verify_target`
   (or target if None), retry (re-send same raw) if outside tolerance
4. Detect firmware protection: if setpoint zeroed → `ProtectionError`
   (setpoint check stays in the raw command domain)

Caller domain is physical: `prev_*` tracking, returned actuals and
caller-side verifies are all in physical units. With default calibration
(gain=1, offset=0) the adapter is a bit-exact no-op.

`stop` is threaded from `run_scan`/`measure_srk` through the settle
wrappers. **Safety-writes are intentionally NOT interruptible**: the Ug1
safe-lock restore after a protection trip and the heater
`_set_uh_with_verify` omit `stop` so they always complete.

### `_read_measurement_point()`

1. Read Ia, Ig2, Ua, Ug1, Ug2, Uh, Ih — N samples (default 3)
2. Outlier detection: if `max/min > ia_outlier_ratio` (armed only when
   `max(Ia) > _IA_OUTLIER_FLOOR`) → read `ia_outlier_reread_samples` more
3. Robust averaging (trimmed mean / median for ≥ 3 samples)
4. Apply calibration corrections
5. Read `Er` (hardware error register)

### `_read_point()` wrapper (inside `run_scan`)

After reading a point:

1. **Comm error** → auto-retry (2×), then user dialog (retry/skip/abort)
2. **Hardware error** (`Er ≠ 0`) → user dialog → wait for device reset → restore heater → `_BreakSweep`
3. **Heater lost** (`Uh < 0.5V` or `Ih < 0.02A`) → mark lost, return None, scan will stop

## Mode 1: Triode (`_sweep_triode`)

Topology: true triode. Ug2 = 0 (fixed).

```
Ug2 = 0
for ug1 in ug1_values:                    # closed → open
    settle Ua to ua_values[0]
    settle Ug1
    for ua in ua_values:                   # low → high
        settle Ua
        read point
        SAVE point
        if Pa exceeded → SAVE + BREAK inner
    [optional: adaptive refine for this curve]
    if Pa broke on first Ua → ABORT Ug1 loop
```

### Protection behavior

| Check | Action | Rationale |
|-------|--------|-----------|
| Pa exceeded | `break` Ua loop | Ua ascending → Pa only grows |
| Pa broke on 1st Ua | abort entire Ug1 loop | Ug1 opens further → Ia only grows → Pa worse |

No Ig2/Pg2 checks — Ug2 is zero.

## Mode 2: Triode-Connected Pentode (`_sweep_ug2_track`)

Topology: pentode with Ug2 = Ua + offset (triode connection).

```
for ug1 in ug1_values:                    # closed → open
    settle Ug2 to match first Ua (prevent Ig2 spike)
    settle Ua to ua_values[0]
    settle Ug1
    for ua in ua_values:                   # low → high
        ug2 = max(0, ua + offset)
        settle Ua first, then Ug2          # Ua first → lower Ig2 transient
        read point
        if Pa exceeded → SAVE + BREAK inner
        if Ig2 exceeded → BREAK inner (point NOT saved)
        if Pg2 exceeded → BREAK inner (point NOT saved)
        SAVE point
    [optional: adaptive refine]
    if Pa/Ig2/Pg2 broke on 1st Ua → ABORT Ug1 loop
```

### Protection behavior

| Check | Action | Rationale |
|-------|--------|-----------|
| Pa exceeded | `break` Ua loop | Ua ascending, Ug2 tracks → Pa only grows |
| Ig2 exceeded | `break` Ua loop | Ug2 = Ua + offset → Ig2/Pg2 only grow along the sweep; the old skip kept the screen in a growing overload for the rest of the curve while every point was discarded anyway (ML-128, 2026-07-02). Over-limit point is NOT saved — same data semantics as the old skip |
| Pg2 exceeded | `break` Ua loop | Same as Ig2 |
| Pa/Ig2/Pg2 broke on 1st Ua | abort Ug1 loop | Same as triode — next Ug1 is more open, currents only grow |

Note: the firmware OVERIG trip fires at ADC saturation — it protects the
tester electronics, not the tube's Pg2 rating, so the soft limits above are
the only screen-dissipation protection. The **independent** sweeper keeps
the skip in its UP sweep (Ig2 falls as Ua rises past the knee — later
points recover) and breaks only in the down sweep; the statuses
`pg2_break`/`pg2_first`/`ig2_break`/`ig2_first` are shared by both sweepers.

### Order of operations for Ug2 safety

- **Start of curve**: Ug2 dropped to match first Ua **before** Ug1 changes —
  prevents briefly seeing low Ua with high Ug2 from previous curve.
- **Within curve**: Ua set **before** Ug2 — tube briefly sees higher Ua with
  old (lower) Ug2, which reduces Ig2. Safe for upward sweep.

## Mode 3: Independent Pentode (`_sweep_ug2_independent`)

Topology: pentode with independent Ug2. **Bidirectional** Ua sweep per curve.
Uses `safe_entry_idx` memory to skip up-sweep for subsequent Ug1 after the
first one is Pa-limited.

```
for ug2 in ug2_values:                    # low → high
    settle Ug1 to cutoff (ug1_values[0])
    if Ua < Ug2: raise Ua to Ug2 first   # prevent Ig2 at low Ua
    settle Ug2
    start_idx = closest Ua grid point to Ug2
    safe_entry_idx = None                 # reset per Ug2

    for ug1 in ug1_values:                # closed → open
        skip_up = safe_entry_idx < start_idx
        entry_idx = safe_entry_idx if skip_up else start_idx
        settle Ua to ua_values[entry_idx]
        settle Ug1

        ── SWEEP UP (only if not skip_up): ua_values[entry_idx → end] ──
        for ua ascending:
            settle Ua
            read point
            if Pa exceeded → SAVE + BREAK (pa_broke_up = True)
            if Ig2 exceeded → SKIP (continue)
            if Pg2 exceeded → SKIP (continue)
            SAVE point

        ── SWEEP DOWN ──
        down_top_idx = entry_idx if skip_up else entry_idx - 1
        (covers ua_values[:down_top_idx+1] descending)
        build down-steps with bisection (max_step limit)
        for ua descending:
            predictive Ig2 check → break if predicted over limit
            settle Ua
            read point
            if Ig2 exceeded → BREAK (Ig2 grows as Ua drops below Ug2)
            if Pg2 exceeded → BREAK (same)
            if intermediate (non-grid) point → skip saving (continue)
            if Pa exceeded → CONTINUE (Pa drops with Ua, skip but keep going)
            SAVE point

        [optional: adaptive refine]
        update safe_entry_idx = highest Ua where Pa was OK this curve
        if Pa broke on 1st up AND down empty → mark pa_broke_first
```

### safe_entry_idx mechanism

After each Ug1 curve, `safe_entry_idx` records the highest Ua where Pa
was acceptable.  The next (more open) Ug1 starts from this point going
down instead of re-attempting the up-sweep at `start_idx`, because a
more open grid will have higher Pa at the same Ua.

- First Ug1 at each Ug2: full bidirectional from `start_idx`
- Subsequent Ug1: down-only from `safe_entry_idx`, includes it in grid
- Reset at each Ug2 change (Pa-safe zone shifts with Ug2)

This reduces Pa-overload pulses: each subsequent Ug1 avoids re-probing
the known-bad Ua > safe_entry zone.

### Protection behavior — Up-sweep

| Check | Action | Rationale |
|-------|--------|-----------|
| Pa exceeded | `break` | Ua ascending → Pa only grows |
| Ig2 exceeded | `continue` | Ua > Ug2 zone, Ig2 drops with rising Ua |
| Pg2 exceeded | `continue` | Same |

### Protection behavior — Down-sweep

| Check | Action | Rationale |
|-------|--------|-----------|
| Ig2 exceeded | `break` | Ua dropping below Ug2 → Ig2 grows → abort |
| Pg2 exceeded | `break` | Same |
| Pa exceeded | `continue` | Ua dropping → Pa drops → keep going for good points |
| Predictive Ig2 | `break` | Extrapolation says Ig2 will exceed limit |

### Pa abort for independent mode

`pa_broke_first = True` when no useful data collected for a curve:
- Up-sweep stopped at first point (Pa, 1-point up_pts) AND down-sweep
  collected nothing (empty), OR
- skip_up mode with empty down-sweep (all down points were Pa/Pg2/Ig2
  skipped/broken)

On `_MAX_CONSECUTIVE_PA_BREAKS` consecutive `pa_broke_first`, remaining
Ug1 curves at this Ug2 are skipped (marked `aborted` in summary).
Reset per Ug2: aborting Ug1 loop for one Ug2 does not skip subsequent
Ug2 levels.

### Bidirectional sweep rationale

Starting from `Ua ≈ Ug2` (where Ig2 is moderate) and sweeping in both
directions avoids the dangerous zone of low Ua + high Ug2 at the start.
Down-sweep uses step bisection (`down_max_step_v`) for gradual descent
and predictive Ig2 extrapolation to abort early.

## Adaptive Refinement (all modes)

After each Ug1 curve, `_refine_curve_inline()` analyses the coarse data:

- **C1**: Onset — Ia crosses threshold (0 → conducting)
- **C2**: Curvature — second derivative spike
- **C3**: Gradient ratio — slope change
- **C4**: Ig2 kink — non-monotonicity (pentode only)
- **C5**: Large Ia jump

Triggered intervals are bisected (up to `refine_max_depth` levels).
New points are measured while Ug1 (and Ug2) are still settled.
Refine points respect the same Pa/Pg2/Ig2 limits.

## Hardware Protection (`_BreakSweep`)

Firmware can trigger OVERIA (Ia overcurrent), OVERIG (Ig2 overcurrent),
OVERIH (heater overcurrent), or OVERTE (overheat). When triggered:

1. Firmware zeroes `uaset` and `ug2set`, sets `err` flags
2. Python detects via `Er` register in `_read_point()`
3. User dialog — retry waits for physical device reset
4. After `err` cleared: heater restored (single channel — Uh or Ih, not both), settle values reset to 0
5. `_BreakSweep` raised — current Ua curve abandoned, next Ug1 continues

**Note**: `err` is only cleared by physical device restart (`start == 1`
in firmware). There is no UART command to clear it. If `err` persists,
each subsequent curve will trigger the dialog again.

## Scan Summary Event

At the end of `run_scan()`, a `scan_summary` event is emitted via the
progress callback (before return).  Structure:

```python
{
    "event": "scan_summary",
    "duration_s": 123.4,        # wall-clock scan time
    "total_points": 47,          # all collected points (incl. boundary)
    "curves": [
        {"ug1": -4, "ug2": 250, "points": 16, "status": "completed"},
        {"ug1": -3, "ug2": 250, "points": 3,  "status": "pg2_break"},
        {"ug1": -2, "ug2": 250, "points": 0,  "status": "pg2_first"},
        {"ug1": -1, "ug2": 250, "points": 0,  "status": "aborted"},
    ],
    "heater_lost": None,         # or str message if heater died
    # ML-108/109 degradation counters (0 = clean scan). Filled by the io
    # helpers through the optional ``stats`` dict; the summary dialog
    # shows ⚠ lines for non-zero values even on an otherwise clean scan.
    "settle_out_of_tolerance": 0,  # setpoints that failed to settle
    "ia_outlier_rereads": 0,       # points with unstable Ia (re-read)
}
```

### Curve status codes

| Status | Meaning |
|--------|---------|
| `completed` | All requested Ua points collected |
| `pa_partial` | Pa exceeded mid-curve, some points below available |
| `pa_first` | Pa exceeded at very first point, no usable data |
| `pg2_break` | Pg2 stopped down-sweep, partial data collected |
| `pg2_first` | Pg2 exceeded at first point of down-sweep, no data |
| `ig2_break` | Ig2 stopped down-sweep, partial data |
| `ig2_first` | Ig2 exceeded at first point, no data |
| `ig2_predict` | Predictive Ig2 stopped down-sweep |
| `aborted` | Curve skipped after consecutive Pa-first failures |
| `user_stop` | Scan stopped by user or `_SkipPoint` on settle |

### Status determination

Status is derived from break-reason flags, not from point count.
Pa `continue` in down-sweep legitimately skips some Ua points, so
`valid_points < expected` doesn't indicate interruption.  Priority:

1. `ctx.stopped()` → `user_stop`
2. `pa_broke_first` (no usable data) → `pa_first` / `pg2_first` / `ig2_first` / `ig2_predict`
3. `down_break_reason` set → `pg2_break` / `ig2_break` / `ig2_predict`
4. `pa_broke_up` → `pa_partial`
5. Otherwise → `completed`

### Logging

Every curve emits one log line on completion:
- **INFO** for `completed` and `user_stop`
- **WARNING** for break-cases with exact values:
  ```
  Ug1=-4.0 Ug2=250: Pa break at Ua=259 (Pa=17.28W, limit=16.00W), 17 points
  Ug1=-3.0 Ug2=250: Pg2 break at Ua=180 (Pg2=2.45W, limit=2.40W, last safe Ua=190), 13 points
  ```

`run_scan()` also logs start configuration and end summary:
```
INFO [lm19.scan] Scan start: mode=pentode (independent Ug2), Ua=100..300/10, ...
INFO [lm19.scan] Scan end: 46 points in 2m34s, 4 curves (pa_partial=1, pg2_break=3)
```

UI logs lamp identity at start:
```
INFO [app.main_window] Starting scan: lamp=EL84 id=L1
```

Default log level is `INFO` (set in `config/app.json`).

### UI dialog

`main_window._save_and_display_measurement` shows a `QMessageBox.information`
after saving the scan:
- **All completed**: short "Collected N points in M:SS"
- **Any incomplete**: list of problematic curves with reasons from i18n
  keys (`msg.Scan_status_*`)

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_MAX_CONSECUTIVE_PA_BREAKS` | 1 | Pa abort: immediate on first-point Pa break |
| `_IG2_PREDICT_MARGIN` | 1.0 | Predictive Ig2 fires at exactly the limit |
| `_DOWN_SWEEP_GAP_FACTOR` | 1.01 | Tolerance for step bisection |
| `_IA_OUTLIER_FLOOR` | 0.5 mA | Outlier check needs max(Ia) above this |
| Default Pa over % | 20% | From `config/scan.json` |
| Default Pig2 over % | 20% | From `config/scan.json` |
