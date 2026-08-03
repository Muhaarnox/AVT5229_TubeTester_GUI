# Lamp Datasheets

Per-tube datasheets and reference data used for `config/tube_params.json`,
`config/lamp_limits.json`, and `config/lamps.json` entries.

## Layout

Each tube has its own `<TubeName>.md` in `lamp_datasheets/` (sibling of this index).
That file contains:
- list of local copies (PDF and/or markdown extracts) with origin URLs
- typical operating point (cross-manufacturer columns when available)
- maximum ratings (cross-manufacturer columns when available)
- pinout
- equivalents/aliases
- search log (which URLs tried, which succeeded/failed)
- notes (peculiarities, cross-check discrepancies)

Source PDFs share the folder: `<TubeName>_<source>.pdf` (e.g.`EL34_mullard.pdf`). 
Use the Latin transliteration of Soviet tube names
(`6Zh4`) so filenames stay portable. Tubes are normally compiled
under their **Western primary name** (`6AC7`); the alias
lives in the `Aliases:` field of the `.md`.

When a tube has Soviet-only origin (e.g. `6N6P`, `6P1P`, `6S19P`), use the
Latin transliteration as primary: `6N6P.md`, etc.

When no datasheet could be found after 2-3 source attempts, the `<TubeName>.md`
file is still created with a populated `Search log` section. Silent skips
are not allowed.

Auxiliary text extracts (when origin site has no PDF but tabular HTML data
exists) live alongside as `<TubeName>_<source>_extract.md`.

## Index

### Triodes (single)
- [6S2S](lamp_datasheets/6S2S.md) — single octal triode, equivalent to 6J5GT

### Triodes (dual)
- [12AX7](lamp_datasheets/12AX7.md) — ECC83 / 6N2P (`6Н2П`), dual high-µ noval (μ=100)
- [12AT7](lamp_datasheets/12AT7.md) — ECC81, dual medium-µ noval (μ=60)
- [12AU7](lamp_datasheets/12AU7.md) — ECC82, dual low-µ noval (μ=20)
- [6DJ8](lamp_datasheets/6DJ8.md) — ECC88 / 6922, low-noise cascode (μ=33, frame grid)
- [6SN7](lamp_datasheets/6SN7.md) — 6N8S (`6Н8С`), dual medium-µ octal (μ=20)
- [6SL7](lamp_datasheets/6SL7.md) — 6N9S (`6Н9С`), dual high-µ octal (μ=70)
- [6SC7](lamp_datasheets/6SC7.md) — dual high-µ octal, **common cathode**
- [6AS7](lamp_datasheets/6AS7.md) — 6080 / 6N13S (`6Н13С`), dual low-µ high-current
- [6N3P](lamp_datasheets/6N3P.md) — 5670 / 2C51, dual medium-µ noval
- [6N6P](lamp_datasheets/6N6P.md) — Soviet, dual medium-µ medium-power
- [6N30P](lamp_datasheets/6N30P.md) — Soviet "SuperTube", dual high-current (gm=18!)
- [6N7S](lamp_datasheets/6N7S.md) — 6N7, dual class-B PP output (common cathode)
- [ECC85](lamp_datasheets/ECC85.md) — 6AQ8, dual medium-µ FM tuner
- [ECC91](lamp_datasheets/ECC91.md) — 6J6, dual VHF (common cathode, 7-pin)
- [ECC99](lamp_datasheets/ECC99.md) — JJ modern, medium-µ medium-power
- [ECC40](lamp_datasheets/ECC40.md) — early dual (rimlock B8A base — uncommon)
- [E180CC](lamp_datasheets/E180CC.md) — 7062, industrial long-life dual triode
- [E182CC](lamp_datasheets/E182CC.md) — 7119, industrial dual medium-power
- [PCC84](lamp_datasheets/PCC84.md) — 7AN7, dual VHF, **9 V heater**
- [PCC85](lamp_datasheets/PCC85.md) — 9AQ8, dual FM tuner, **9 V heater**
- [6S19P](lamp_datasheets/6S19P.md) — Soviet single regulator triode (noval)

### Triodes (power, testable:false)
- [300B](lamp_datasheets/300B.md) — WE 300B, directly-heated SE audio (Pa=36 W, Uf=5 V)
- [2A3](lamp_datasheets/2A3.md) — directly-heated SE audio (Pa=15 W, Uf=2.5 V)
- [845](lamp_datasheets/845.md) — RCA HV transmitter triode (Pa=75 W, top cap, Uf=10 V)
- [211](lamp_datasheets/211.md) — VT-4C, HV transmitter triode (Pa=75 W, top cap, Uf=10 V)

### Pentodes / Beam tetrodes (small to medium)
- [6AC7](lamp_datasheets/6AC7.md) — 1852 / 6Zh4 (`6Ж4`), sharp-cutoff metal octal RF pentode
- [6SJ7](lamp_datasheets/6SJ7.md) — 6Zh8 (`6Ж8`), octal sharp-cutoff voltage amp
- [6F6](lamp_datasheets/6F6.md) — 6F6S (`6Ф6С`), octal power pentode (Pa=11 W)
- [6V6](lamp_datasheets/6V6.md) — 6P6S (`6П6С`), octal beam tetrode (Pa=14 W)
- [6P1P](lamp_datasheets/6P1P.md) — Soviet noval beam tetrode (Pa=12 W)
- [EF86](lamp_datasheets/EF86.md) — 6267, noval low-noise audio pentode (μ=6000)
- [EL90](lamp_datasheets/EL90.md) — 6AQ5, 7-pin miniature beam tetrode
- [EL95](lamp_datasheets/EL95.md) — 6DL5, 7-pin miniature power pentode

### Pentodes / Beam tetrodes (medium)
- [EL84](lamp_datasheets/EL84.md) — 6BQ5 / 6P14P (`6П14П`), noval power pentode (Pa=12 W)
- [EL34](lamp_datasheets/EL34.md) — 6CA7 / E34L, octal power pentode (Pa=25 W). Includes JJ EL34 II (text-extractable)
- [6L6](lamp_datasheets/6L6.md) — 6L6GC / 6P3S (`6П3С`) / 5881, octal beam tetrode (Pa=30 W)
- [KT66](lamp_datasheets/KT66.md) — Genalex, octal beam tetrode (Pa=25 W)

### Pentodes / Beam tetrodes (large power, testable:false)
- [KT88](lamp_datasheets/KT88.md) — octal beam tetrode (Pa=42 W)
- [6550](lamp_datasheets/6550.md) — octal beam tetrode (Pa=35 W)
- [KT120](lamp_datasheets/KT120.md) — Tung-Sol modern, large octal beam tetrode (Pa=60 W)

### Combo tubes (triode + pentode in one envelope)
- [PCL86](lamp_datasheets/PCL86.md) — 14GW8, **9 V heater**. Splits into `PCL86_triode` + `PCL86_pentode` in `tube_params.json`
- [ECL86](lamp_datasheets/ECL86.md) — 6GW8, 6.3 V heater. Splits into `ECL86_triode` + `ECL86_pentode`
- [ECL82](lamp_datasheets/ECL82.md) — 6BM8, 6.3 V heater. Splits into `ECL82_triode` + `ECL82_pentode`

## Inventory

44 unique tubes covered (47 entries in `tube_params.json` — combo tubes split into 2 entries each, accounted as 3 physical envelopes covering 6 `tube_params.json` keys).

- **Markdown files**: 52 (44 per-tube + 8 auxiliary `_extract.md` for Soviet/text-only sources)
- **Source PDFs**: 102 (multi-manufacturer cross-check where available), all listed in the per-tube "Local copies" tables
- **Reference manuals** (`lamp_datasheets/manuals/`): RCA RC-22 (1963, 43 MB), Sylvania Technical Manual (1970, 30 MB), Sylvania Receiving Tubes (compact, 5 MB), Tung-Sol 1948 (×2 copies, ~26 MB each)
- Sources covered: RCA, General Electric, Sylvania, Brimar, Philips, Mullard, JJ Electronic, Western Electric (1950+2020), Electro-Harmonix, PSVANE, Tung-Sol, GEC, Keith Snook archive, eandc.ru, rudatasheet.ru, rtellason.com, jacmusic.com, TDSL Duncan

### Source coverage breakdown (PDFs + text extracts)

| Tubes with N sources | Count | Examples |
|---|---|---|
| 1 source | 12 | 211, 6DJ8, 6N30P, 6N6P, 6N7S, 6P1P, 6S19P, ECC91, ECC99, EL90, EL95, KT120 (rare/modern-only/single-mfg; the Soviet types add a text `_extract.md`) |
| 2 sources | 16 | 6AS7, 6SC7, 6SL7, 6SN7, 6V6, 845, E180CC, E182CC, ECC40, ECC85, ECL86, EF86, KT66, PCC84, PCC85, PCL86 |
| 3+ sources | 15 | 12AT7, 12AU7, 12AX7, 2A3, 300B, 6550, 6AC7, 6F6, 6L6, 6N3P, 6SJ7, ECL82, EL34, EL84, KT88 (KT88 leads with 5) |

## Findings applied to `config/tube_params.json` (2026-05-12 verification pass)

Verified via rudatasheet.ru + cross-check with text-extractable JJ Electronic datasheets:

- ✓ `6SN7` aliases: added `6N8S`, `6Н8С` (rudatasheet.ru explicitly lists `6N8S → 6SN7`)
- ✓ `6SL7` aliases: added `6N9S`, `6Н9С` (rudatasheet.ru explicitly lists `6N9S → 6SL7`)
- ✓ `12AU7` / `ECC82` aliases: **removed `6N1P`** (rudatasheet.ru gives 6Н1П analogs as `E80CC`, `ECC85` — NOT `ECC82`)
- ⚠ `ECC85` aliases: kept `6N1P` per rudatasheet's explicit claim, but parameters mismatch (μ=35 vs 57 — Russian-convention equivalence, not parametric). Flagged for future review — best fix is a dedicated `6N1P` entry in `tube_params.json`
- ✓ `EL34` aliases: added `E34L` (JJ heavy-plate variant), `7D11`
- ✓ `6L6` aliases: added `6P3S`, `6П3С` (Soviet equivalent)
- ✓ `EL84` aliases: added Cyrillic forms `6П14П` (`6P14P`), `6П18П` (`6P18P`)
- ✓ `6V6` aliases: added Cyrillic form `6П6С` (`6P6S`)
- ✓ `6F6` aliases: added Cyrillic form `6Ф6С` (`6F6S`)
- ✓ `6SJ7` aliases: added `6Zh8`, `6Ж8` (Soviet equivalent)

## ECC99.md correction

Earlier extract overstated power dissipation: had `Pa=5 W per section / 9 W combined`. Text-extracted from official JJ ECC99 PDF (Adobe InDesign source, 2015): **single `Wa = 3.5 W`**. Also corrected `Ik_max` from 40 mA → 60 mA per section. Added JJ-recommended substitutes: `5687`, `E182CC`, `6840`, `6BL7` (per JJ's own datasheet text).

## Text-extractable JJ PDFs

All 11 JJ Electronic PDFs in this collection are **text-extractable** (Adobe InDesign source, not scans):

- 12AT7_jj.pdf, 12AU7_jj.pdf, 300B_jj.pdf, 6550_jj.pdf, 6L6_jj.pdf, ECC99_jj.pdf
- EL34_jj.pdf, EL34_jj_II.pdf, EL84_jj.pdf, KT66_jj.pdf, KT88_jj.pdf

These are the gold-standard reference for modern-production specs. Extract with `pypdf`:
```python
import pypdf
r = pypdf.PdfReader('lamp_datasheets/EL34_jj.pdf')
for p in r.pages: print(p.extract_text())
```

## Full reference manuals (downloaded via curl, exceed WebFetch 10MB limit)

In `lamp_datasheets/manuals/`:

| File | Size | Pages | Source | Notes |
|---|---|---|---|---|
| RCA_RC22_Receiving_Tube_Manual_1963.pdf | 43 MB | 548 | archive.org | Comprehensive RCA reference manual; text partially extractable (OCR), all common receiving tubes |
| Sylvania_Technical_Manual_1970.pdf | 30 MB | 628 | frank.pocnet.net | Fully text-extractable; pin diagrams, ratings, characteristics for every tube |
| TungSol_TechnicalData_1948.pdf | 25 MB | 1028 | w140.com | Mixed-quality OCR; original Tung-Sol catalog |

`worldradiohistory.com` URLs are Cloudflare-protected (blocks even with User-Agent header) — Tung-Sol 1951 manual not downloaded.
