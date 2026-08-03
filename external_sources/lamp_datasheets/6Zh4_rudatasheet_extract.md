# 6Zh4 (`6Ж4`) — parameters (rudatasheet.ru extract)

Source: <https://rudatasheet.ru/tubes/6zh4/>
Fetched: 2026-05-12

Textual extract from rudatasheet.ru (the origin page has no direct PDF —
only a parameter table). Saved for textual citation in code/tests,
since the main PDF sources (`6Zh4_eandc.pdf`, `6AC7_rca.pdf`,
`6AC7_ge.pdf`, `6AC7_rtellason.pdf`) are image-only scans.

## Description

Pentode for voltage amplification at high and intermediate frequencies.
Metal envelope, octal base. Mass 43 g.

## Typical operating point

Conditions: Uh = 6.3 V, Ua = 300 V, Ug3 = 0 V, Rk = 160 Ω

| Parameter | Value | Unit |
|---|---|---|
| Ih (heater current) | 450 ± 25 | mA |
| Ia (anode current) | 10.25 ± 2.25 | mA |
| Ig2 (screen grid current) | 2.2 ± 1.0 | mA |
| S (transconductance) | 9 ± 2 | mA/V |
| Ri (plate resistance) | 0.8 | MΩ |
| C_in (input capacitance) | 8.5 ± 1.5 | pF |
| C_out (output capacitance) | 4.75 ± 1.25 | pF |
| C_pg (grid-plate capacitance) | ≤ 0.015 | pF |

## Maximum ratings

| Parameter | Value | Unit |
|---|---|---|
| Uh (heater voltage range) | 5.7…6.9 | V |
| Ua_max (anode voltage) | 330 | V |
| Ug2_max (screen voltage) | 165 | V |
| Pa_max (anode dissipation) | 3.3 | W |
| Pg2_max (screen dissipation) | 0.45 | W |
| Operating temperature range | −60…+70 | °C |
| Minimum service life | ≥ 2000 | h |

## Equivalents

6AC7, 6F10 (per rudatasheet.ru)

## Pinout

Octal base (8 pins). Diagram is on the origin page (not extracted from HTML).
Cross-reference with `6Zh4_eandc.pdf` or `6AC7_rca.pdf`.

## Notes

- μ (amplification factor) is not listed on the source page
- Cgp ≤ 0.015 pF matches RCA 6AC7 (single-ended screened construction,
  g1 brought to a base pin)
- Pa_max of 3.3 W is higher than the RCA variant (3.0 W).
