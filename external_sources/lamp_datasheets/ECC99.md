# ECC99 — Datasheet Extract

Primary: `ECC99`
Aliases: none — JJ Electronic's own designation (modern-only design, no NOS equivalent). Per JJ's own datasheet, recommended substitutes (with pinout adjustment): `5687`, `E182CC`, `6840`, `6BL7`
Topology: dual triode (medium-µ, medium-power, line/headphone driver)
Base/Construction: noval 9-pin, all-glass
Compiled: 2026-05-12

## Local copies

| File | Source | Manufacturer | Notes |
|---|---|---|---|
| [ECC99_jj.pdf](ECC99_jj.pdf) | <https://www.jj-electronic.com/images/stories/product/preamplifying_tubes/pdf/ecc99.pdf> | JJ Electronic | scan, ~490 KB |

## Typical operating point

Reference (JJ): Ua = 150 V, Ug1 = −4 V

| Parameter | JJ ECC99 (system I) | JJ ECC99 (system II) | Unit |
|---|---|---|---|
| Ia per section | 18 | 18 | mA |
| S (gm) | 9.5 | 9.5 | mA/V |
| μ | 22 | 22 | — |
| Ri (rp) | 2.3 | 2.3 | kΩ |
| C_g/k (input) | 5.8 | 5.8 | pF |
| C_a (output) | 0.91 | 0.81 | pF |
| C_g/a (feedthrough) | 5.1 | 5.1 | pF |

Values extracted directly from text-readable JJ ECC99 PDF (Adobe InDesign source, 2015-01-19).

## Maximum ratings

| Parameter | JJ ECC99 | Unit |
|---|---|---|
| Ua_max | 400 | V |
| **Wa (Pa_max)** | **3.5** | W |
| Uh | 6.3 V (parallel) / 12.6 V (series) | V |
| Ih @ 6.3V | 800 | mA |
| Ih @ 12.6V | 400 | mA |
| Ik_max per section | 60 | mA |
| Uk/f_max | 200 | V |

**Correction note**: Earlier versions of this extract listed `Pa_max per section = 5 W` and `Pa_max combined = 9 W` — both incorrect. The official JJ Electronic datasheet (text-extracted from PDF metadata, Adobe InDesign source) gives single value `Wa = 3.5 W`. The Ik_max also corrected from 40 mA to 60 mA per section per official datasheet.

## Pinout (noval 9-pin, bottom view)

Same family as 12AX7 / 12AU7 / ECC8x with heater centre-tap option for 6.3/12.6 V switching.

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
| 9 | Heater centre-tap |
| top cap | none |

## Equivalents

- No NOS Western equivalent — ECC99 is **JJ Electronic's modern original design** (introduced ~2000)
- **JJ-recommended substitutes** (per official datasheet, with note: "Outlets on some of these types could have different set-up"): `5687`, `E182CC`, `6840`, `6BL7`
- Functional cousins: `5687WA` (μ=18, gm=11.5, Pa=4.2 W — different pinout!), `6BL7-GTA` (octal, μ=15, Pa=10 W), `12BH7A` (similar μ, lower Ia)
- Soviet rough counterpart: `6Н6П` (6N6P) — μ=20, gm=12, Pa=8 W per section. Different pinout but similar role

## Recommended applications (per JJ datasheet)

Direct quote of recommended uses: "Driver of power triodes such as 300B, 2A3..., Output stage headphone amplifiers, preamplifiers, power stage little P-P triode amplifiers (10W-4xECC99) and parallel voltage power supplies."

## Search log

- ✓ jj-electronic.com/.../ecc99.pdf — JJ Electronic, saved

## Notes

- **JJ-specific modern tube** — there's no NOS supply chain for this designation. If JJ stops production, replacement requires functional substitution (5687, 6N6P / `6Н6П`) with circuit redesign
- Audiophile-popular for headphone amplifiers and line stages — high transconductance + decent power dissipation
- Higher Pa rating "combined" (9 W) than "per section × 2" (10 W) acknowledges thermal interaction — typical of modern dual-triodes
- PDF is image-only scan
