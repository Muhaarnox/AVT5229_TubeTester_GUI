# PCC85 — Datasheet Extract

Primary: `PCC85`
Aliases: `9AQ8` (US industrial — 9V heater version of 6AQ8/ECC85)
Topology: dual triode (medium-µ, FM tuner front-end)
Base/Construction: noval 9-pin, all-glass. **9 V series-string heater**
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [PCC85_philips.pdf](PCC85_philips.pdf) | <https://frank.pocnet.net/sheets/010/p/PCC85.pdf> | Philips | scan, ~195 KB |
| [PCC85_mullard.pdf](PCC85_mullard.pdf) | not recorded | Mullard (per file name) | scan, 868 KB, 23 p |

## Typical operating point

Reference: Ua = 170 V, Rk = 180 Ω (≈ Ug1 = −2.3 V)

| Parameter | Philips PCC85 | Unit |
|---|---|---|
| Ia per section | 10 | mA |
| S (gm) | 5.5 | mA/V |
| μ | 57 | — |
| Ri (rp) | 10.4 | kΩ |
| C_in (g-k) | 3.5 | pF |
| C_gp | 1.4 | pF |
| C_out (a-k) | 1.5 | pF |

## Maximum ratings

| Parameter | Philips PCC85 | Unit |
|---|---|---|
| Ua_max | 300 | V |
| Pa_max per section | 2.5 | W |
| **Uh** | **9 V** (series-string!) | V |
| Ih | 300 | mA |

## Pinout (noval 9-pin, bottom view)

Same as ECC85 — internal shield on pin 9.

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

- 6.3 V heater equivalent: **`ECC85` / `6AQ8`** — same triode design, just 6.3 V parallel heater instead of 9 V series
- 9 V heater alternative: `PCC189` (different μ, same series-string chain)

## Search log

- ✓ frank.pocnet.net/sheets/010/p/PCC85.pdf — Philips, saved

## Notes

- **9 V heater** — series-string TV/FM tuner version of ECC85
- Same triode geometry and parameters as ECC85, only the heater is different
- AVT5229 must be configured with `uh: 9.0` for this tube
- For audio applications, ECC85 (6.3 V) is more practical
- PDF is image-only scan
