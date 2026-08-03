# Calibration — User Guide

Practical guide to calibrating the LM19 with the **Calibration** tab.
For the full UI reference see `FEATURES.md` → *Calibration*; for the file
format see `CONFIG_REFERENCE.md` → `config/calibration.json`.

Software calibration corrects systematic errors of the device's
measurement (READ) and command (SET) paths. Each channel has independent
gain/offset coefficients (`corrected = raw × gain + offset`) stored in
`config/calibration.json`. You only ever calibrate READ — the SET
correction is derived from it automatically by the wizard.

---

## What you need

- **Voltage channels** (Ua, Ug1, Ug2, Uh, Ih): a multimeter connected to
  the channel output, as instructed by the wizard's prep page. Enter your
  meter's accuracy as **Meter ±%** — it is saved with the quality data.
- **Current channels** (Ia low, Ia high, Ig2): a load resistor and an
  ammeter; the prep page shows load recommendations per channel.
- A connected, powered device (the wizard refuses to start without an
  open COM port, and while another subsystem — scan, health, preheat —
  is using the hardware).

## Running the wizard

1. In the **Coefficients** table select the channel row and click
   **Wizard**.
2. **Low point** — the device sets a value near the low end of the
   channel range; enter the meter reading. The measurement is an
   N-point average (N configurable, 1–100) with a live device reading
   shown at ~1 Hz.
3. **High point** — same near the high end of the range.
4. **Result page** — the READ correction is computed from your meter
   readings; the SET correction is **auto-derived** from READ plus the
   observed device transfer, marked "auto" in the UI. Quality metrics
   (point spread, residuals, meter accuracy) are shown and saved for
   both.
5. Back in the tab, click **Save** — the wizard only updates the
   in-memory coefficients; nothing is written to
   `config/calibration.json` until you save (the app warns about unsaved
   changes, and **Discard** reverts them).

Notes:

- The wizard commands its test points **uncalibrated on purpose** — it
  characterizes the raw device transfer. Re-running it after a previous
  calibration is fine.
- Ia has two hardware ranges (20 mA / 200 mA) calibrated independently;
  the application switches ranges automatically.
- When calibrating one heater channel (Uh or Ih) the other heater
  setpoint is zeroed for the duration of the measurement, and both are
  safely reset when the wizard closes.
- On Finish or Cancel the wizard resets every channel it touched to a
  safe state; if that reset fails you get an explicit error dialog — do
  not walk away from the device until you see the wizard close cleanly.

## Verifying the result

- **Test buttons** (Manual Edit section): enter a test value and compare
  **Set** (calibrated) vs **Set raw** (uncalibrated) against your meter.
- **Manual tab**: command e.g. Ua = 250 V — the meter should read
  250 ± 1 V.
- **Scan check**: scan the same tube before and after calibration with
  identical settings — the curves should differ by less than the channel
  tolerances.

## How the corrections are applied

- **READ** corrections apply to every measurement everywhere (scan, SRK,
  health, manual readings, live panels).
- **SET** corrections pre-correct every working-point command (scan,
  SRK, health, manual tab, preheat). When non-zero SET coefficients are
  loaded, the log records one INFO line at startup saying so.
- Shutdown and zeroing commands are **always raw** by design — a
  calibrated "zero" could command a non-zero voltage.

## Limitations

- Two-point calibration is linear: residual device nonlinearity near the
  extremes of a range is not corrected.
- `*_set` rows in the Coefficients table are read-only — SET is always
  derived from READ by the wizard, never edited directly.
- **Reset** / **Reset All** return channels to identity
  (gain = 1, offset = 0); remember to **Save**.

Design rationale (domain model, feedforward, verify semantics) lives in
the internal document `CALIBRATION_PLAN.md`.
