# 6N6P — parameters (rudatasheet.ru extract)

Source: <https://rudatasheet.ru/tubes/6n6p/>
Fetched: 2026-05-13

## Description

Dual triode for voltage amplification. Miniature 9-pin glass envelope, 15 g, 22.5 mm diameter.

## Typical operating point

Conditions: Uh=6.3 V

| Parameter | Value | Unit |
|---|---|---|
| Ih (heater current) | 750 ± 60 (900 ± 50 for `6N6P-I` / `6Н6П-И`) | mA |
| Ia | 30 ± 10 | mA |
| S (transconductance) | 11 ± 2.9 | mA/V |
| μ (amplification factor) | 20 ± 4 |  |
| Ig1 reverse | ≤ 1.0 | µA |

## Maximum ratings

| Parameter | Value | Unit |
|---|---|---|
| Uh (heater voltage range) | 5.7…7 | V |
| Ua_max | 300 (450 V locked/cold) | V |
| Operating temperature | −60…+85 | °C |
| Vibration tolerance | up to 6 | g |
| Service life (standard) | ≥ 3000 | h |
| Service life (`6N6P-I` / `6Н6П-И`) | ≥ 500 | h |

## Equivalents

- Western: **`E182CC`** (Philips industrial CC-series)

## Notes

- Per rudatasheet, **E182CC is the Western equivalent** — this updates my earlier 6N6P.md which said "no direct Western pin-compatible equivalent"
- However parametric match isn't perfect: E182CC has μ=25 vs 6N6P (`6Н6П`) μ=20, similar gm
- Often used in audio: cathode-follower output, line driver, headphone amp
