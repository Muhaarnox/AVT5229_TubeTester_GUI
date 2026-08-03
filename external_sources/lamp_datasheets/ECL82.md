# ECL82 — Datasheet Extract (triode + pentode)

Primary: `ECL82`
Aliases: `6BM8` (US — direct equivalent, same heater)
Topology: **combo tube** — 1× medium-µ triode (driver) + 1× power pentode (output)
Base/Construction: noval 9-pin, all-glass, 6.3 V heater
Compiled: 2026-05-12

In `config/tube_params.json` this tube is split into two entries: `ECL82_triode` and `ECL82_pentode`.

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECL82_philips.pdf](ECL82_philips.pdf) | <https://frank.pocnet.net/sheets/010/e/ECL82.pdf> | Philips | scan, ~106 KB |
| [ECL82_mullard.pdf](ECL82_mullard.pdf) | <https://frank.pocnet.net/sheets/030/e/ECL82.pdf> | Mullard | scan, ~1.4 MB |
| [ECL82_rca.pdf](ECL82_rca.pdf) | not recorded | RCA (per file name) | scan, 126 KB, 3 p; filed under the US designation 6BM8 |

## Triode section — typical operating point

Reference (Philips ECL82): Ua = 170 V, Rk = 4.7 kΩ (≈ Ug1 = −5 V)

| Parameter | Philips ECL82 (triode) | Unit |
|---|---|---|
| Ia | 1 | mA |
| S (gm) | 2.5 | mA/V |
| μ | 70 | — |
| Ri (rp) | 28 | kΩ |
| C_in (g-k) | 2.0 | pF |
| C_gp | 1.0 | pF |
| C_out (a-k) | 1.5 | pF |

## Pentode section — typical operating point (SE class A)

Reference (Philips ECL82): Ua = 200 V, Ug2 = 200 V, Ug1 = −12 V

| Parameter | Philips ECL82 (pentode) | Unit |
|---|---|---|
| Ia | 35 | mA |
| Ig2 | 5 | mA |
| S (gm) | 6 | mA/V |
| Ri (rp) | 25 | kΩ |
| μ_g1-g2 | 50 | — |
| C_in (ccg1) | 9.3 | pF |
| C_gp | 0.3 | pF |
| C_out (ccp) | 8 | pF |
| Pout (SE class A) | ≈ 3.5 | W |

## Maximum ratings

| Parameter | Triode | Pentode | Unit |
|---|---|---|---|
| Ua_max | 550 | 250 | V |
| Ug2_max | — | 250 | V |
| Pa_max | 1 | 7 | W |
| Pg2_max | — | 1.8 | W |
| Uh | 6.3 ± 10% | (shared) | V |
| Ih (combined) | 780 | (shared) | mA |

## Pinout (noval 9-pin, bottom view)

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

- US: `6BM8` (direct, same heater voltage)
- 9 V heater variant: not common (different family of P-prefix tubes)
- Soviet: no direct equivalent
- Same family as ECL86 (slightly higher Pa) and PCL86 (9 V heater)

## Search log

- ✓ frank.pocnet.net/sheets/010/e/ECL82.pdf — Philips, saved
- ✓ frank.pocnet.net/sheets/030/e/ECL82.pdf — Mullard, saved

## Notes

- **Combo tube** — predecessor to ECL86 (lower power pentode section, smaller envelope option)
- Popular in 1960s table radios, record players, intercoms
- 6BM8 is the US designation — used in Heathkit "EA-2" and similar small amps
- PDFs are image-only scans
