# 6S19P — parameters (rudatasheet.ru extract)

Source: <https://rudatasheet.ru/tubes/6s19p/>
Fetched: 2026-05-13

## Description

Triode for voltage-regulator (series-pass stabiliser) service. Miniature
9-pin base, 22.5 mm diameter, 25 g.

## Typical operating point

Conditions: Uh=6.3 V, Ua=110 V, Rk=130 Ω, Ug1=−7 V

| Parameter | Value | Unit |
|---|---|---|
| If (filament/heater) | 1.0 ± 0.1 | A |
| Ia | 95 ± 15 | mA |
| S (transconductance) | 7.5 ± 1.5 | mA/V |
| Ri (plate resistance) | 400 ± 100 | Ω |

## Maximum ratings

| Parameter | Value | Unit |
|---|---|---|
| Uh (heater range) | 5.7…6.9 | V |
| Ua_max | 350 | V |
| Ug1 range (negative bias) | 1.5…200 | V |
| Pa_max (at Ua ≤ 200 V) | 11 | W |

## Variants

`6S19P` / `6С19П`, `6S19P-V` / `6С19П-В`, `6S19P-VR` / `6С19П-ВР`

## Equivalents

No direct Western pin-compatible equivalent. Functional class:
voltage-regulator pass-element triode.

## Notes

- Very low Ri (400 Ω!) — high gm makes it ideal for **series-pass regulator** service in HV B+ supplies
- Heater current **1 A @ 6.3 V** — heavy draw, need beefy filament supply
- Sometimes paralleled for higher current (datasheet includes parallel-tube calculation table)
- Operating Ug1 range goes to **−200 V** — unusual; useful for wide-range stabilizers
