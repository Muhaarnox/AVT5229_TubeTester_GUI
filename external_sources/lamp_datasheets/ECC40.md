# ECC40 — Datasheet Extract

Primary: `ECC40`
Aliases: `B65` (similar Mullard naming generation)
Topology: dual triode (medium-µ, audio)
Base/Construction: rimlock (B8A 8-pin), all-glass, early-1950s European
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECC40_philips.pdf](ECC40_philips.pdf) | <https://frank.pocnet.net/sheets/046/e/ECC40.pdf> | Philips | scan, ~885 KB |
| [ECC40_mullard.pdf](ECC40_mullard.pdf) | not recorded | Mullard (per file name) | scan, 171 KB, 8 p |

## Typical operating point

Reference: Ua = 250 V, Rk = 1.3 kΩ self-bias (≈ Ug1 = −2 V)

| Parameter | Philips ECC40 | Unit |
|---|---|---|
| Ia per section | 6 | mA |
| S (gm) | 2.5 | mA/V |
| μ | 32 | — |
| Ri (rp) | 12.5 | kΩ |
| C_in (g-k) | 2.8 | pF |
| C_gp | 2.7 | pF |
| C_out (a-k) | 1.1 | pF |

## Maximum ratings

| Parameter | Philips ECC40 | Unit |
|---|---|---|
| Ua_max | 300 | V |
| Pa_max per section | 1.0 | W |
| Uh | 6.3 ± 10% | V |
| Ih | 600 | mA |

## Pinout (rimlock B8A, 8-pin bottom view)

Rimlock socket — uncommon today, requires specific adapter. Pin numbering reference:

| Pin | Function |
|---|---|
| 1 | Heater H |
| 2 | Cathode K1 |
| 3 | Grid G1 |
| 4 | Plate A1 |
| 5 | Plate A2 |
| 6 | Grid G2 |
| 7 | Cathode K2 |
| 8 | Heater H |
| top cap | none |

## Equivalents

- No direct pin-compatible noval substitute — rimlock base is unique
- Parametric similar (electrically): `12AU7` / `ECC82` (μ=20) or `12AT7` / `ECC81` (μ=60) — but **not pin-compatible** (different base)
- Used in 1950s European radios and small amplifiers (Telefunken, Philips, Valvo)

## Search log

- ✓ frank.pocnet.net/sheets/046/e/ECC40.pdf — Philips, saved

## Notes

- **Rimlock base** (B8A) is the predecessor to noval — has 8 pins on a small base with one fatter "key" pin for socket orientation. Adapters to noval exist for restoration
- Designed primarily for AC line-operated portable / table radios; replaced by noval ECC81/82/83 family
- Tester compatibility: AVT5229 may not have rimlock socket — requires noval adapter or external pinout breakout
- PDF is image-only scan
