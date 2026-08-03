# ECL86 — Datasheet Extract (triode + pentode)

Primary: `ECL86`
Aliases: `6GW8` (US equivalent — direct, same heater)
Topology: **combo tube** — 1× medium-µ triode (driver) + 1× power pentode (output)
Base/Construction: noval 9-pin, all-glass, 6.3 V heater
Compiled: 2026-05-12

In `config/tube_params.json` this tube is split into two entries: `ECL86_triode` and `ECL86_pentode`.

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECL86_philips.pdf](ECL86_philips.pdf) | <https://frank.pocnet.net/sheets/010/e/ECL86.pdf> | Philips | scan, ~344 KB |
| [ECL86_mullard.pdf](ECL86_mullard.pdf) | <https://frank.pocnet.net/sheets/030/e/ECL86.pdf> | Mullard | scan, ~555 KB |

## Triode section — typical operating point

Reference (Philips ECL86): Ua = 250 V, Rk = 1.5 kΩ (≈ Ug1 = −2 V)

| Parameter | Philips ECL86 (triode) | Unit |
|---|---|---|
| Ia | 1.3 | mA |
| S (gm) | 1.7 | mA/V |
| μ | 75 | — |
| Ri (rp) | 44 | kΩ |
| C_in (g-k) | 1.8 | pF |
| C_gp | 0.9 | pF |
| C_out (a-k) | 1.5 | pF |

## Pentode section — typical operating point (SE class A)

Reference (Philips ECL86): Ua = 250 V, Ug2 = 250 V, Ug1 = −9.5 V

| Parameter | Philips ECL86 (pentode) | Unit |
|---|---|---|
| Ia | 36 | mA |
| Ig2 | 6 | mA |
| S (gm) | 10 | mA/V |
| Ri (rp) | 30 | kΩ |
| μ_g1-g2 | 55 | — |
| C_in (ccg1) | 5.5 | pF |
| C_gp | 0.5 | pF |
| C_out (ccp) | 3.0 | pF |
| Pout (SE class A) | ≈ 4.0 | W |

## Maximum ratings

| Parameter | Triode | Pentode | Unit |
|---|---|---|---|
| Ua_max | 550 | 300 | V |
| Ug2_max | — | 300 | V |
| Pa_max | 1 | 9 | W |
| Pg2_max | — | 1.8 | W |
| Uh | 6.3 ± 10% | (shared) | V |
| Ih (combined) | 660 | (shared) | mA |

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

- US: `6GW8` (direct, same heater voltage)
- 9 V heater variant: `PCL86` / `14GW8` — same topology, just series-string heater
- Famous audiophile use: small SE-class A amplifiers (e.g. Tomas Kuznetsov designs), 60s/70s console radios

## Search log

- ✓ frank.pocnet.net/sheets/010/e/ECL86.pdf — Philips, saved
- ✓ frank.pocnet.net/sheets/030/e/ECL86.pdf — Mullard, saved

## Notes

- **Combo tube** — same topology as PCL86 with 6.3 V heater
- One-bottle SE amplifier; triode drives pentode in same envelope
- Pout ~ 4 W class A SE — pleasant low-power audiophile
- Used in vintage radios (Telefunken, Philips, Grundig)
- PDFs are image-only scans
