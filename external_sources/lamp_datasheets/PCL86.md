# PCL86 — Datasheet Extract (triode + pentode)

Primary: `PCL86`
Aliases: `14GW8` (US equivalent)
Topology: **combo tube** — 1× medium-µ triode (driver) + 1× power pentode (output)
Base/Construction: noval 9-pin, all-glass, **9 V series-string heater**
Compiled: 2026-05-12

In `config/tube_params.json` this tube is split into two entries: `PCL86_triode` and `PCL86_pentode` (each fitted separately as required by the model framework).

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [PCL86_philips.pdf](PCL86_philips.pdf) | <https://frank.pocnet.net/sheets/010/p/PCL86.pdf> | Philips | scan, ~281 KB |
| [PCL86_mullard.pdf](PCL86_mullard.pdf) | <https://frank.pocnet.net/sheets/030/p/PCL86.pdf> | Mullard | scan, ~523 KB |

## Triode section — typical operating point

Reference (Philips PCL86): Ua = 250 V, Rk = 1.5 kΩ self-bias (≈ Ug1 = −2 V)

| Parameter | Philips PCL86 (triode) | Unit |
|---|---|---|
| Ia | 1.3 | mA |
| S (gm) | 1.7 | mA/V |
| μ | 75 | — |
| Ri (rp) | 44 | kΩ |
| C_in (g-k) | 2.3 | pF |
| C_gp | 1.4 | pF |
| C_out (a-k) | 1.5 | pF |

## Pentode section — typical operating point (SE class A)

Reference (Philips PCL86): Ua = 200 V, Ug2 = 200 V, Ug1 = −9 V

| Parameter | Philips PCL86 (pentode) | Unit |
|---|---|---|
| Ia | 36 | mA |
| Ig2 | 5.5 | mA |
| S (gm) | 10 | mA/V |
| Ri (rp) | 30 | kΩ |
| μ_g1-g2 | 55 | — |
| C_in (ccg1) | 10 | pF |
| C_gp | 0.4 | pF |
| C_out (ccp) | 5.5 | pF |
| Pout (SE class A) | ≈ 4.0 | W |

## Maximum ratings

| Parameter | Triode | Pentode | Unit |
|---|---|---|---|
| Ua_max | 550 | 300 | V |
| Ug2_max | — | 300 | V |
| Pa_max | 1 | 9 | W |
| Pg2_max | — | 1.8 | W |
| **Uh** | **9 V** (series-string) | (shared) | V |
| Ih (combined) | 300 | (shared) | mA |

## Pinout (noval 9-pin, bottom view)

Per Philips datasheet:

| Pin | Function |
|---|---|
| 1 | Triode anode (Pa_T) |
| 2 | Triode grid (Gg_T) |
| 3 | Pentode anode (Pa_P) |
| 4 | Heater H |
| 5 | Heater H |
| 6 | Pentode g2 (screen) |
| 7 | Pentode g1 |
| 8 | Common cathode (K_T + K_P + g3_P) |
| 9 | Internal shield |
| top cap | none |

## Equivalents

- US: `14GW8` (direct US designation)
- 6.3 V heater equivalent: `ECL86` / `6GW8` — same topology, different heater voltage
- Used heavily in 1960s European table radios for "single-tube amplifier" sections

## Search log

- ✓ frank.pocnet.net/sheets/030/p/PCL86.pdf — Mullard, saved
- ✓ frank.pocnet.net/sheets/010/p/PCL86.pdf — Philips, saved

## Notes

- **Combo tube** — driver + output in one envelope. Single-stage tube amplifier on one bottle, used as "all-in-one" for cheap 1960s console radios / record players
- **9 V series-string heater** — must use `uh: 9.0` in `lamps.json`, not 6.3 V
- In AMC analysis, triode and pentode sections are fitted as **separate tubes** (`PCL86_triode`, `PCL86_pentode`) because each has its own Koren parameters
- PDFs are image-only scans
