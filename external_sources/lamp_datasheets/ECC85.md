# ECC85 — Datasheet Extract

Primary: `ECC85`
Aliases: `6AQ8`, `B719` (Philips industrial designation)
Topology: dual triode (medium-µ, FM-radio front-end)
Base/Construction: noval 9-pin, all-glass
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECC85_philips.pdf](ECC85_philips.pdf) | <https://frank.pocnet.net/sheets/010/e/ECC85.pdf> | Philips | scan, ~230 KB |
| [ECC85_mullard.pdf](ECC85_mullard.pdf) | <https://frank.pocnet.net/sheets/030/e/ECC85.pdf> | Mullard | scan, ~176 KB |

## Typical operating point

Reference: Ua = 170 V, Rk = 180 Ω self-bias (≈ Ug1 = −2.3 V)

| Parameter | Philips/Mullard ECC85 | Unit |
|---|---|---|
| Ia per section | 10 | mA |
| S (gm) | 5.5 | mA/V |
| μ | 57 | — |
| Ri (rp) | 10.4 | kΩ |
| C_in (g-k) | 3.1 | pF |
| C_gp | 1.5 | pF |
| C_out (a-k) | 0.18 | pF |

## Maximum ratings

| Parameter | Philips ECC85 | Unit |
|---|---|---|
| Ua_max (DC) | 300 | V |
| Pa_max per section | 2.5 | W |
| Uh | 6.3 ± 10% | V |
| Ih | 435 | mA |
| Rg1_max (cath. bias) | 1 | MΩ |

## Pinout (noval 9-pin, bottom view)

Same as 12AX7 / 12AT7 / 12AU7 / ECC8x family.

| Pin | Function |
|---|---|
| 1 | Anode A1 |
| 2 | Grid G1 |
| 3 | Cathode K1 |
| 4 | Heater H |
| 5 | Heater H |
| 6 | Anode A2 |
| 7 | Grid G2 |
| 8 | Cathode K2 |
| 9 | Internal shield (no centre-tap) |
| top cap | none |

## Equivalents

- Western: `6AQ8` (US designation), `B719`
- Soviet: there is **no direct equivalent**. `6Н1П` (6N1P) is sometimes listed but has different pinout and slightly different parameters (μ=35 vs 57). Not a drop-in replacement
- 12AT7 / ECC81 is parametrically similar (μ=60, gm=5.5, Ri=10.9k) but pin compatibility differs in shield handling — practically compatible for many audio uses, NOT in FM-front-end designs where shielding matters

## Search log

- ✓ frank.pocnet.net/sheets/010/e/ECC85.pdf — Philips, saved
- ✓ frank.pocnet.net/sheets/030/e/ECC85.pdf — Mullard, saved

## Notes

- ECC85 was specifically designed for VHF FM radio front-ends — combines cascode-style low noise with sufficient μ for two-stage gain blocks
- Often confused with 12AT7 (ECC81) due to similar headline numbers — but the internal shielding and pin 9 usage differ; FM tuner schematics that specify ECC85 should not be substituted blindly
- `config/tube_params.json` lists `6N1P` as alias of ECC85 — this is **incorrect parametrically** (μ=35 vs 57, gm=4.5 vs 5.5) and should probably be removed (see also notes in 12AU7.md). Soviet `6Н3П` (6N3P, μ=39, gm=5.0) is closer but not perfect
- PDFs are image-only scans
