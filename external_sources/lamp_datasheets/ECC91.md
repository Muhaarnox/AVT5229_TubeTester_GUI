# ECC91 — Datasheet Extract

Primary: `ECC91`
Aliases: `6J6` (US — direct pin-compatible equivalent), `6J6A`, `CV858`
Topology: dual triode (high-µ, common-cathode pair, VHF / RF)
Base/Construction: 7-pin miniature (B7G), all-glass
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECC91_mullard.pdf](ECC91_mullard.pdf) | <https://frank.pocnet.net/sheets/030/e/ECC91.pdf> | Mullard | scan, ~137 KB |

## Typical operating point

Reference: Ua = 100 V, Rk = 270 Ω self-bias (≈ Ug1 = −1 V)

| Parameter | Mullard ECC91 / 6J6 | Unit |
|---|---|---|
| Ia per section | 8.5 | mA |
| S (gm) | 5.3 | mA/V |
| μ | 38 | — |
| Ri (rp) | 7.1 | kΩ |
| C_in (g-k) | 2.6 | pF |
| C_gp | 1.5 | pF |
| C_out (a-k) | 1.6 | pF |

## Maximum ratings

| Parameter | Mullard ECC91 | Unit |
|---|---|---|
| Ua_max | 300 | V |
| Pa_max per section | 1.5 | W |
| Uh | 6.3 ± 10% | V |
| Ih | 450 | mA |
| Ik_max per section | 17 | mA |

## Pinout (B7G 7-pin miniature, bottom view)

**Common cathode** (pin 7) shared between both triodes — like 6SC7.

| Pin | Function |
|---|---|
| 1 | Plate A2 |
| 2 | Grid G2 |
| 3 | Heater H |
| 4 | Heater H |
| 5 | Plate A1 |
| 6 | Grid G1 |
| 7 | Common cathode K |
| top cap | none |

## Equivalents

- Western: `6J6` (RCA — direct), `6J6A` (improved version), `5964` (premium), `CV858` (UK mil-spec)
- No Soviet equivalent

## Search log

- ✓ frank.pocnet.net/sheets/030/e/ECC91.pdf — Mullard, saved

## Notes

- **Common cathode topology** — both triodes share pin 7. Same architecture as 6SC7
- Designed for VHF amplifier / FM front-end use; also popular as voltage-controlled oscillator
- Compact 7-pin B7G base — same socket as 6AQ5/EL90, EL95
- Mullard PDF is image-only scan
