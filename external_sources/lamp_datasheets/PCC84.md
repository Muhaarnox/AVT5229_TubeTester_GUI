# PCC84 — Datasheet Extract

Primary: `PCC84`
Aliases: `7AN7` (US industrial)
Topology: dual triode (medium-µ, VHF cascode / FM tuner)
Base/Construction: noval 9-pin, all-glass. **9 V series-string heater**
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [PCC84_philips.pdf](PCC84_philips.pdf) | <https://frank.pocnet.net/sheets/050/p/PCC84.pdf> | Philips (GE listing) | scan, ~67 KB |
| [PCC84_mullard.pdf](PCC84_mullard.pdf) | not recorded | Mullard (per file name) | scan, 641 KB, 16 p |

## Typical operating point

Reference: Ua = 100 V, Rk = 80 Ω

| Parameter | Philips PCC84 | Unit |
|---|---|---|
| Ia per section | 15 | mA |
| S (gm) | 6.5 | mA/V |
| μ | 17 | — |
| Ri (rp) | 2.6 | kΩ |
| C_in (g-k) | 2.0 | pF |
| C_gp | 1.0 | pF |
| C_out (a-k) | 0.7 | pF |

## Maximum ratings

| Parameter | Philips PCC84 | Unit |
|---|---|---|
| Ua_max | 250 | V |
| Pa_max per section | 1.5 | W |
| **Uh** | **9 V** (series-string!) | V |
| Ih | 300 | mA |

## Pinout (noval 9-pin, bottom view)

Standard noval — same as ECC8x family.

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
| 9 | Internal shield |
| top cap | none |

## Equivalents

- Heater-equivalent at 9V: `7AN7` (US industrial)
- 6.3V heater analog: **`6CC84`** (rare) or parametric near-cousin `6JD8` / `6JC8` — different μ
- NOT pin-compatible with PCC85 (which has internal shield differently)

## Search log

- ✓ frank.pocnet.net/sheets/050/p/PCC84.pdf — Philips, saved

## Notes

- **9 V heater** — designed for series-string TV/FM tuner heater chains where 6.3 V wouldn't divide the line voltage evenly. NOT a drop-in for 6.3 V circuits without heater rewiring
- Originally for FM-front-end cascode in low-cost European TV/radio sets
- Low μ + high gm suits cascode neutralisation
- AVT5229 must be configured with `uh: 9.0` for this tube (and `uh_max` adjusted accordingly)
- PDF is image-only scan
