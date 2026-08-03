# SPICE Model Test Data — Sources & Inventory

Reference data for verifying Koren Triode and Koren Pentode model fitting
in LM19 `spice_export.py`.

---

## Directory Structure

```
tests/spice_test_data/
├── SOURCES.md              ← this file
├── raw/                    ← original source files (do not modify)
│   ├── tuparam/            ← Koren's Tuparam Matlab data (legacy copies)
│   │   ├── 12AX7A.m       ← 6 measured points + expected fit
│   │   ├── 7025.m         ← 9 measured points + expected fit (VCT=0.5)
│   │   └── 6550.m         ← 11 measured points (pentode + triode-mode)
│   ├── next_tube/          ← Next-Tube.com empirical data
│   │   ├── download.ps1   ← download script
│   │   ├── *.xls          ← 18 original Excel files
│   │   └── csv/           ← 18 CSV files (LibreOffice conversion)
│   │       └── *.csv
│   ├── loadline_plotter/   ← GitHub CSV datasheet curves
│   │   ├── valves_data.csv ← plate curves for 6 triodes
│   │   └── valves_specs.csv← mu, ra, Pa_max specs
│   └── libs/               ← SPICE model libraries (reference)
│       ├── koren/          ← Norman Koren originals
│       │   ├── Tubemods.zip       ← 88 KB, full PSpice package
│       │   ├── Tubemods/          ← extracted contents
│       │   │   ├── Tube.lib       ← 12.5 KB, main tube model library
│       │   │   ├── tube1.lib      ← 11 KB, additional models (same set)
│       │   │   ├── Tube97.lib     ← 11.5 KB, 1997 revision
│       │   │   ├── errata.txt     ← corrections to published article
│       │   │   ├── psreadme.txt   ← original README
│       │   │   ├── *.sch          ← PSpice schematics (12AX7TST, EL34, etc.)
│       │   │   ├── Pentode.cir    ← pentode test circuit
│       │   │   └── Triode.cir     ← triode test circuit
│       │   ├── Tuparam.zip        ← 9.6 KB, Matlab fitter + all tube data
│       │   └── Tuparam/           ← extracted contents
│       │       ├── TuParam.m      ← 9 KB, main fitting program
│       │       ├── TuCalc.m       ← model calculator
│       │       ├── Optube.m       ← optimizer helper
│       │       └── *.m            ← 15 tube data files (see inventory)
│       ├── duncan/         ← Duncan Munro / Leach models
│       │   ├── spice.zip          ← 13.7 KB
│       │   └── spice/             ← extracted .INC files
│       │       ├── 12AX7A.INC     ← Leach 3/2 power law triode model
│       │       ├── 6L6GC.INC      ← Leach pentode model
│       │       ├── KT88.INC       ← KT88/6550 pentode
│       │       └── ... (18 files total)
│       ├── ltspice_community/ ← LTwiki mirror + EXCEM/Intusoft
│       │   ├── Tube.lib       ← Koren mirror (LTwiki)
│       │   ├── tube1.lib      ← Koren mirror
│       │   ├── Tube97.lib     ← Koren mirror
│       │   ├── psreadme.txt   ← Koren readme
│       │   ├── VACUUM.LIB     ← 19 KB, EXCEM/Intusoft triode+pentode+heater
│       │   └── vacuumnl.lib   ← 15 KB, EXCEM no-labels version
│       └── next_tube/      ← Next-Tube.com Koren-based libs
│           ├── download.ps1   ← download script
│           ├── PSlib.zip      ← 3.6 KB, PSpice library
│           ├── MC7.zip        ← 4.4 KB, Micro-Cap 6,7 library
│           ├── Model.zip      ← 2.1 KB, SEcalcS(i) library
│           ├── Tube_ORCAD.zip ← 39 KB, Extended ORCAD library
│           └── Tube_Msim10.zip← 285 KB, MSIM-10 library
│   ├── curvetracedata/     ← pypsucurvetrace measured data (GitHub: mbrennwa)
│       ├── ECC88/data/     ← 62 specimens (Siemens, Philips, JJ, Mullard, Sovtek, Amperex)
│       ├── EL34/data/      ← 28 specimens (Mullard, Siemens, EHX, AEG, RFT)
│       ├── 300B/data/      ← 16 specimens (EHX, Svetlana, TJ Mesh)
│       ├── KT66/data/      ← 9 specimens
│       ├── 6C33C/data/     ← 19 specimens
│       ├── 6E5P/data/      ← 36 specimens
│       ├── 801_VT62/data/  ← 38 specimens
│       ├── THF51/data/     ← 28 specimens
│       ├── 6N23P_6H23pi/   ← 24 specimens
│       └── ... (22 tube types, 309 .dat files total)
│   └── etracer_samples/    ← eTracer CSV + ETD data (vt52.com)
│       ├── 10Y.csv, 10Y_VT25.etd
│       ├── EL34_triode.csv, EL34_triode.etd
│       ├── VT52.csv, VT52.etd
│       └── ... (35 CSV + 35 ETD files, 36 tube types)
├── converted/              ← unified JSON format for tests (94 files, 27016 points)
│   ├── pentode_6550_tuparam.json     ← 11pts, has expected_params
│   ├── pentode_6550A_tuparam.json    ← 12pts
│   ├── pentode_6550C_tuparam.json    ← 6pts
│   ├── pentode_6L6GB_tuparam.json    ← 13pts
│   ├── pentode_6L6GC_tuparam.json    ← 11pts
│   ├── pentode_EL34_tuparam.json     ← 9pts
│   ├── pentode_KT88_tuparam.json     ← 12pts
│   ├── triode_12AT7_tuparam.json     ← 8pts
│   ├── triode_12AU7_tuparam.json     ← 8pts
│   ├── triode_12AU7A_tuparam.json    ← 8pts
│   ├── triode_12AX7AMitch_tuparam.json ← 6pts, has expected_params
│   ├── triode_12AX7ASYL_tuparam.json ← 8pts, VCT=0.5
│   ├── triode_6DJ8Mitch_tuparam.json ← 8pts, VCT=0.5
│   ├── triode_6SN7Sylv_tuparam.json  ← 11pts
│   ├── triode_7025_tuparam.json      ← 9pts, has expected_params, VCT=0.5
│   ├── triode_12ay7_datasheet.json   ← 96pts (loadline_plotter)
│   ├── triode_e88cc_datasheet.json   ← 48pts (loadline_plotter)
│   ├── triode_ecc81_datasheet.json   ← 103pts (loadline_plotter)
│   ├── triode_ecc82_datasheet.json   ← 95pts (loadline_plotter)
│   ├── triode_ecc83_datasheet.json   ← 85pts (loadline_plotter)
│   ├── triode_ecc85_datasheet.json   ← 86pts (loadline_plotter)
│   ├── triode_6N2P_nexttube.json     ← 30pts, has expected_params (Next-Tube)
│   ├── triode_6N23P_nexttube.json    ← 29pts, has expected_params
│   ├── triode_6N8C_nexttube.json     ← 35pts, has expected_params
│   ├── triode_6N1P_nexttube.json     ← 20pts, has expected_params
│   ├── triode_6N5P_nexttube.json     ← 22pts, has expected_params
│   ├── triode_6N6P_nexttube.json     ← 29pts
│   ├── triode_6N13S_nexttube.json    ← 98pts, has expected_params
│   ├── triode_6N3P_nexttube.json     ← 70pts, has expected_params
│   ├── triode_6N27P_nexttube.json    ← 72pts, has expected_params
│   ├── triode_6S6B_nexttube.json     ← 77pts, has expected_params
│   ├── triode_6S19P_nexttube.json    ← 22pts, has expected_params
│   ├── triode_6C41C_nexttube.json    ← 48pts
│   ├── triode_6S4S_nexttube.json     ← 35pts, has expected_params
│   ├── triode_GU50_T_nexttube.json   ← 35pts
│   ├── triode_6P14P_T_nexttube.json  ← 65pts, has expected_params
│   ├── triode_6P45S_T_nexttube.json  ← 43pts, has expected_params
│   ├── triode_6F5P_T_nexttube.json   ← 46pts, has expected_params, VCT=0.6
│   ├── triode_6F5P_PT_nexttube.json  ← 67pts, has expected_params
│   ├── triode_6P3C_T_nexttube.json   ← 45pts
│   ├── pentode_6P3C_P_nexttube.json  ← 37pts
│   ├── pentode_6F5P_P_nexttube.json  ← 61pts, has expected_params (Kg2=4500)
│   ├── pentode_GU50_P_nexttube.json  ← 37pts
│   ├── triode_ECC88_curvetracedata.json    ← 141pts, 61 specimens (curvetracedata)
│   ├── triode_EL34_curvetracedata.json     ← 493pts, 28 specimens
│   ├── triode_300B_curvetracedata.json     ← 850pts, 16 specimens
│   ├── triode_KT66_curvetracedata.json     ← 483pts, 9 specimens
│   ├── triode_6C33C_curvetracedata.json    ← 320pts, 19 specimens
│   ├── triode_6N23P_curvetracedata.json    ← 291pts, 24 specimens
│   ├── triode_6N30P_curvetracedata.json    ← 703pts, 2 specimens
│   ├── triode_ECC81_curvetracedata.json    ← 240pts, 2 specimens
│   ├── triode_PCC88_curvetracedata.json    ← 500pts, 4 specimens
│   ├── triode_6E5P_curvetracedata.json     ← 870pts, 32 specimens
│   ├── triode_D3A_curvetracedata.json      ← 941pts, 6 specimens
│   ├── triode_807_curvetracedata.json      ← 928pts, 1 specimen
│   ├── triode_DL92_curvetracedata.json     ← 206pts, 1 specimen
│   ├── triode_DL94_curvetracedata.json     ← 258pts, 1 specimen
│   ├── triode_DL96_curvetracedata.json     ← 325pts, 1 specimen
│   ├── triode_DL98_curvetracedata.json     ← 163pts, 1 specimen
│   ├── triode_10_VT25_curvetracedata.json  ← 832pts, 11 specimens
│   ├── triode_20B_curvetracedata.json      ← 817pts, 10 specimens
│   ├── triode_32B_curvetracedata.json      ← 938pts, 4 specimens
│   ├── triode_801_VT62_curvetracedata.json ← 1524pts, 38 specimens
│   ├── triode_841_VT51_curvetracedata.json ← 516pts, 4 specimens
│   ├── triode_THF51_curvetracedata.json    ← 546pts, 28 specimens
│   ├── triode_10Y_etracer.json            ← 311pts (eTracer / vt52.com)
│   ├── triode_4P1L_etracer.json           ← 531pts, triode_connected
│   ├── triode_50_etracer.json             ← 473pts
│   ├── triode_5842Q_etracer.json          ← 630pts
│   ├── triode_VT52_etracer.json           ← 365pts
│   ├── triode_EL84_etracer.json           ← 349pts, triode_connected
│   ├── triode_E180F_etracer.json          ← 645pts
│   ├── triode_EC8010_etracer.json         ← 673pts
│   └── ... (29 eTracer files, 12455 points total)
└── tools/
    ├── convert_loadline_to_lm19.py   ← CSV→JSON converter (loadline_plotter)
    ├── convert_tuparam_to_lm19.py    ← Matlab .m→JSON converter (Tuparam)
    ├── convert_nexttube_to_lm19.py   ← Next-Tube CSV→JSON converter
    ├── convert_curvetracedata_to_lm19.py ← pypsucurvetrace .dat→JSON converter
    ├── convert_etracer_to_lm19.py    ← eTracer CSV→JSON converter
    └── validate_all.py               ← validates all converted JSON files
```

---

## Source 1: Norman Koren — Tuparam

**URL:** https://www.normankoren.com/Audio/Tuparam.zip
**Article:** https://www.normankoren.com/Audio/Tube_params.html
**License:** Free for non-commercial use (academic)
**Format:** Matlab .m files with Vp, Vg, [Vs], Idata arrays
**Date:** 2001

### Data Quality
- **Gold standard** — measured data used by the model author himself
- Points are few (6–11 per tube) but carefully selected across operating range
- For pentodes: includes triode-connected points (Vp = Vs)
- Each file includes expected fit result for direct comparison

### Available Tubes

| File | Type | Topology | Points | Has Ig2? | Has Ug2 variation? |
|------|------|----------|--------|----------|-------------------|
| **12AX7A.m** | 12AX7 | Triode | 6 | — | — |
| **7025.m** | 7025 (~12AX7) | Triode | 9 | — | — |
| **6550.m** | 6550 | **Pentode** | 11 | No | Yes (100–500V) |

### Expected Fit Results (from Tuparam)

| Tube | mu | Ex | Kg1 | Kp | Kvb | Kg2 | VCT |
|------|----|----|-----|----|----|-----|-----|
| 12AX7A | 101.24 | 1.267 | 1002.9 | 699.73 | 300.0 | — | 0.0 |
| 7025 | 103.44 | 1.245 | 1515.4 | 903.23 | 99.2 | — | 0.5 |
| 6550 | 8.45 | 1.247 | 642.5 | 48.92 | 20.9 | 4500* | 0.0 |

*Kg2 was NOT optimized by Tuparam — kept at user-provided value 4500.

### Published Reference Values (Koren tube.lib, Table 1)

| Tube | mu | Ex | Kg1 | Kg2 | Kp | Kvb |
|------|----|----|-----|-----|----|-----|
| 12AX7 | 100 | 1.4 | 1060 | — | 600 | 300 |
| 12AU7 | 21.5 | 1.3 | 1180 | — | 84 | 300 |
| 6DJ8 | 28 | 1.3 | 330 | — | 320 | 300 |
| 6550 | 7.9 | 1.35 | 890 | 4200 | 60 | 24 |
| 6L6GC | 8.7 | 1.35 | 1460 | 4500 | 48 | 12 |
| KT88 | 8.8 | 1.35 | 730 | 4200 | 32 | 16 |
| EL34 | 11.0 | 1.35 | 650 | 4200 | 60 | 24 |

### Test Strategy
- **Triode unit-test:** Fit 12AX7A data → compare mu/ex/kg1/kp/kvb
  with Tuparam expected values. Tolerance: ±10% per parameter.
- **Pentode unit-test:** Fit 6550 data → compare with Tuparam expected
  values. Note: Kg2 not optimized by Tuparam, so we should only compare
  mu/ex/kg1/kp/kvb.
- **Regression test:** Verify Ia prediction at each input point matches
  measured Ia within RMS < 5% of max Ia.

---

## Source 2: Next-Tube.com — Empirical Measurements

**URL:** https://next-tube.com/data.php (Russian version: /ru/data.php)
**License:** Free, community contributions (Eugene V. Karpov, Shevchenko)
**Format:** Excel .xls with data blocks + modeling diagrams
**Date:** 2002–2006
**Downloaded:** Yes (18 XLS files, all converted)

### Data Quality
- **Real measurements** of individual tubes (averaged readings)
- Compatible with Tuparam format (Vp, Vg, Vs, Idata)
- Variable quality: "perspective" tubes measured more carefully
- Contains Koren model fitting results and comparison diagrams
- Most files include published `.SUBCKT` parameters fitted by Karpov

### File Inventory

```
raw/next_tube/
├── download.ps1           ← download script
├── *.xls                  ← 18 original Excel files
└── csv/                   ← LibreOffice headless conversions
    └── *.csv              ← 18 CSV files
```

### Available Tubes — Triodes (14 files)

| File | Type | Western Equiv | Points | Curves | Koren Params? |
|------|------|---------------|--------|--------|---------------|
| 6N2P.xls | 6Н2П | **12AX7** | 30 | 8 | mu=106.0, Kvb=4680.6, VCT=0.3 |
| 6N23P.xls | 6Н23П | **6DJ8 / ECC88** | 29 | 8 | mu=36.21, Kp=171.1 |
| 6N8C.xls | 6Н8С | **6SN7GT / ECC32** | 35 | 7 | mu=20.06, Kp=108.75 |
| 6N1P.xls | 6Н1П | **ECC85 / 6AQ8** | 20 | 5 | mu=34.51, Kp=238.29 |
| 6N5P.xls | 6Н5П | — | 22 | 6 | mu=40.73 (has positive Ug) |
| 6N6P.xls | 6Н6П | — | 29 | 7 | No |
| 6N13S.xls | 6Н13С | **6AS7G** | 98 | 14 | mu=2.35 (low-mu) |
| 6N3P.xls | 6Н3П | **2C51 / 6CC42** | 70 | 7 | mu=34.82, Ex=1.909 |
| 6N27P.xls | 6Н27П | **ECC86 / 6GM8** | 72 | 7 | mu=16.3, Kvb=150 |
| 6S6B.xls | 6С6Б | — | 77 | 10 | mu=25.5, Kp=203.2 |
| 6S19P.xls | 6С19П | — | 22 | 6 | mu=3.71 (low-mu) |
| 6C41C.xls | 6С41С | — | 48 | 13 | No |
| 6S4S.xls | 6С4С | **6A3 / 6A5** | 35 | 7 | mu=4.32 (power triode) |
| GU50.xls | ГУ-50 | **LS50 / P50-2** | 35 | 8 | No (triode mode only) |

### Available Tubes — Pentode→Triode Mode (3 files)

| File | Type | Western Equiv | Points | Curves | Koren Params? |
|------|------|---------------|--------|--------|---------------|
| 6P14P-T.xls | 6П14П | **EL84 / N329** | 65 | 12 | mu=21.49, Kp=136.46 |
| 6P45S-T.xls | 6П45С | — | 43 | 11 | mu=3.96, Kp=44.51 |
| 6F5P.xls (PT) | 6Ф5П | **6GV8 / ECL85** | 67 | 13 | mu=7.49, Kvb=120 |

### Available Tubes — Pentodes (4 files, pentode mode data)

| File | Type | Western Equiv | Points | Curves | Ug2 | Koren Params? |
|------|------|---------------|--------|--------|-----|---------------|
| 6P3C.xls | 6П3С | **6L6G** | 37 | 8 | 200/240V | No |
| 6F5P.xls (P) | 6Ф5П | **6GV8 / ECL85** | 61 | 11 | 200V | mu=11.5, Kg2=4500 |
| GU50.xls (P) | ГУ-50 | **LS50** | 37 | 11 | 220/240V | No |
| 6P3C.xls (T) | 6П3С | **6L6G** | 45 | 9 | triode conn. | No |

### Available Tubes — Multi-Section: 6F5P (Triode-Pentode)

6F5P (6GV8/ECL85) is a composite triode-pentode tube with 3 separate datasets:

| Dataset | JSON File | Topology | Points | Section |
|---------|-----------|----------|--------|---------|
| 6F5P_T | `triode_6F5P_T_nexttube.json` | Triode | 46 | True triode section |
| 6F5P_PT | `triode_6F5P_PT_nexttube.json` | Triode | 67 | Pentode section, triode-connected |
| 6F5P_P | `pentode_6F5P_P_nexttube.json` | Pentode | 61 | Pentode section, Ug2=200V |

### Expected Koren Fit Results (from XLS model headers)

| Tube | mu | Ex | Kg1 | Kp | Kvb | VCT |
|------|----|----|-----|----|----|-----|
| 6N2P | 106.00 | 1.398 | 1326.3 | 415.26 | 4680.6 | 0.30 |
| 6N23P | 36.21 | 1.316 | 1131.7 | 171.10 | 300.0 | — |
| 6N8C | 20.06 | 1.306 | 978.5 | 108.75 | 300.0 | — |
| 6N1P | 34.51 | 1.26 | 2106.9 | 238.29 | 300.0 | — |
| 6N5P | 40.73 | 1.396 | 1362.2 | 245.44 | 300.0 | — |
| 6N13S | 2.35 | 1.247 | 637.1 | 12.40 | 300.0 | — |
| 6N3P | 34.82 | 1.909 | 1445.3 | 171.13 | 300.0 | — |
| 6S6B | 25.50 | 1.361 | 846.1 | 203.20 | 300.0 | — |
| 6N27P | 16.3 | 1.3 | 350.1 | 76.28 | 150.0 | — |
| 6S4S | 4.32 | 1.078 | 523.3 | 43.32 | 300.0 | — |
| 6S19P | 3.71 | 1.000 | 297.0 | 12.17 | 400.0 | — |
| 6P14P (T) | 21.49 | 1.428 | 514.4 | 136.46 | 300.0 | — |
| 6P45S (T) | 3.96 | 1.505 | 522.0 | 44.51 | 300.0 | — |
| 6F5P (T) | 73.16 | 1.668 | 522.9 | 353.07 | 519.0 | 0.60 |
| 6F5P (PT) | 7.49 | 1.622 | 1336.3 | 59.91 | 120.0 | — |

| Tube (Pentode) | mu | Ex | Kg1 | Kg2 | Kp | Kvb |
|----------------|----|----|-----|-----|----|-----|
| 6F5P (P) | 11.5 | 1.204 | 303.0 | 4500 | 34.87 | 20.1 |

### Test Strategy
- **Triode fitting:** All 19 triode datasets → fit and compare with Karpov's params.
  Tolerance: mu ±15%, other params ±25% (empirical data has more variance than Tuparam).
- **Pentode fitting:** 3 pentode datasets (6P3C, 6F5P, GU50) → validate Ia model.
  6F5P has published Kg2=4500 for direct comparison.
- **Cross-validation:** Compare fitted params for 6N2P/12AX7 and 6N8C/6SN7 with
  Koren Tube.lib values for the same tube families.

### How to Download
```powershell
cd tests\spice_test_data\raw\next_tube
powershell -ExecutionPolicy Bypass -File download.ps1
```

### How to Convert
```powershell
# Step 1: Convert XLS to CSV via LibreOffice headless
$soffice = "C:\Program Files\LibreOffice\program\soffice.exe"
Get-ChildItem raw\next_tube\*.xls | ForEach-Object {
    & $soffice --headless --convert-to csv --outdir raw\next_tube\csv $_.FullName
}

# Step 2: Convert CSV to unified JSON
python tools\convert_nexttube_to_lm19.py
```

---

## Source 3: loadline_plotter — Datasheet Curves (GitHub)

**URL:** https://github.com/andmarti1424/loadline_plotter
**License:** GPLv3
**Format:** CSV with (VALVE, GRIDCURVE, X1, Y1, X2, Y2, ...) pairs
**Date:** 2023+

### Data Quality
- **Digitized from datasheets** — not real measurements
- Good density: 6–16 points per curve, 7–12 curves per tube
- Only triodes, no pentodes
- No Ig2 data
- Useful for fitting validation and regression tests

### Available Tubes

| Tube | Aliases | Curves | Total Points | In lamps.json? |
|------|---------|--------|-------------|----------------|
| **ECC83** | 12AX7 | 9 (Ug1: 0 to -4V) | ~80 | Yes |
| **ECC82** | 12AU7 | 10 (Ug1: 0 to -20V) | ~80 | Yes |
| **ECC81** | 12AT7 | 8 (Ug1: 0 to -8V) | ~85 | Yes |
| **E88CC** | 6DJ8, 6922 | 10 (Ug1: 0 to -10V) | ~40 | Yes |
| **ECC85** | 6AQ8 | 7 (Ug1: 0 to -7V) | ~65 | Yes |
| **12AY7** | — | 12 (Ug1: 0 to -12V) | ~80 | No |

### Datasheet Specs (from valves_specs.csv)

| Tube | Pa max (W) | Ua max (V) | mu | ra (Ω) |
|------|-----------|-----------|-----|--------|
| E88CC | 1.5 | 220 | 33 | 2650 |
| ECC85 | 2.5 | 300 | 57 | 9700 |
| ECC82 | 2.75 | 300 | 19 | 7800 |
| ECC83 | 1.0 | 300 | 100 | 62500 |
| ECC81 | 2.5 | 300 | 60 | 11000 |
| 12AY7 | 1.65 | 330 | 44 | 25000 |

---

## Converted JSON Format

All test data is converted to a unified JSON format in `converted/`.

### Naming Convention
```
{topology}_{tube_name}_{source}.json
```
Examples:
- `triode_12AX7_tuparam.json` — Koren's measured data
- `triode_ecc83_datasheet.json` — loadline_plotter digitized curves
- `pentode_6550_tuparam.json` — Koren's measured pentode data

### JSON Schema

```json
{
  "_comment": "description",
  "tube_type": "12AX7",
  "topology": "triode",
  "aliases": ["ECC83"],
  "source": "origin",
  "url": "https://...",

  "expected_params": {
    "mu": 101.24, "ex": 1.267, "kg1": 1002.9,
    "kp": 699.73, "kvb": 300.0
  },

  "points": [
    {"ua": 100, "ug1": 0.0, "ia": 1.94},
    {"ua": 100, "ug1": 0.0, "ug2": 300, "ia": 307.0, "ig2": null}
  ],

  "_units": {"ua": "V", "ug1": "V", "ug2": "V", "ia": "mA", "ig2": "mA"}
}
```

### Field Notes
- `expected_params` — only present for Tuparam data (gold standard)
- `published_params` — Koren tube.lib values (if available)
- `datasheet_specs` — from manufacturer datasheets (if available)
- `ug2`, `ig2` — only for pentodes; `ig2: null` = not measured
- `_triode_connected: true` — pentode point with Ug2 = Ua

---

## Source 4: Norman Koren — Tubemods (SPICE Libraries)

**URL:** https://www.normankoren.com/Audio/Tubemods.zip
**Article:** https://www.normankoren.com/Audio/Tubemodspice_article.html
**License:** Shareware (free for personal use)
**Format:** PSpice .lib + .slb (symbols), .sch (schematics), .cir (circuits)
**Downloaded:** Yes (88 KB zip, 74 files extracted)

### Contents — Tube.lib (main library)

| SUBCKT | Topology | Pins | Parameters |
|--------|----------|------|------------|
| **6550** | Pentode | P G1 C G2 | MU=7.9 EX=1.35 KG1=890 KG2=4200 KP=60 KVB=24 |
| **EL34** | Pentode | P G1 C G2 | MU=11 EX=1.35 KG1=650 KG2=4200 KP=60 KVB=24 |
| **6L6GC** | Pentode | P G1 C G2 | MU=8.7 EX=1.35 KG1=1460 KG2=4500 KP=48 KVB=12 |
| **KT88** | Pentode | P G1 C G2 | MU=8.8 EX=1.35 KG1=730 KG2=4800 KP=32 KVB=16 |
| **6AN8P** | Pentode | P G1 C G2 | MU=45 EX=1.35 KG1=520 KG2=120 KP=120 KVB=18 |
| **12AX7** | Triode | P G C | MU=100 EX=1.4 KG1=1060 KP=600 KVB=300 |
| **12AU7** | Triode | P G C | MU=21.5 EX=1.3 KG1=1180 KP=84 KVB=300 |
| **12AT7** | Triode | P G C | MU=60 EX=1.35 KG1=460 KP=300 KVB=300 |
| **6DJ8** | Triode | P G C | MU=28 EX=1.3 KG1=330 KP=320 KVB=300 |
| **6AN8T** | Triode | P G C | MU=21.2 EX=1.36 KG1=945 KP=84 KVB=300 |
| **2A3** | Triode | P G C | MU=4.2 EX=1.4 KG1=1500 KP=60 KVB=300 |
| **300B** | Triode | P G C | MU=3.95 EX=1.4 KG1=1550 KP=65 KVB=300 |
| **6C33C** | Triode | P G C | MU=3.1 EX=1.4 KG1=163 KP=15 KVB=300 |

### Tube97.lib (1997 revision)

Same tubes, but **6550** and **12AX7** have updated circuit structure:
- 6550: Uses `P G CS` pinout (combined screen), modified equations
- 12AX7: "MODIFIED MODEL 12/97" with different equation structure
- New: **6SN7** triode model added (MU=20, EX=1.3, KG1=1040)

### Tuparam Data Files (15 tubes)

| File | Type | Topology | Points | VCT | Key Notes |
|------|------|----------|--------|-----|-----------|
| **12AX7AMitch.m** | 12AX7A | Triode | 6 | 0 | Tom Mitchell data |
| **12AX7ASYL.m** | 12AX7A | Triode | 8 | 0.5 | Sylvania manual (6AV6) |
| **7025.m** | 7025 | Triode | 9 | 0.5 | Sylvania manual |
| **12AU7.m** | 12AU7 | Triode | 8 | 0 | Sylvania manual |
| **12AU7A.m** | 12AU7A | Triode | 8 | 0 | Tom Mitchell |
| **12AT7.m** | 12AT7 | Triode | 8 | 0 | Tom Mitchell |
| **6SN7Sylv.m** | 6SN7 | Triode | 11 | 0 | Sylvania manual |
| **6DJ8Mitch.m** | 6DJ8 | Triode | 8 | 0.5 | Tom Mitchell |
| **6550.m** | 6550 | Pentode | 11 | 0 | TPC Tung-Sol |
| **6550A.m** | 6550A | Pentode | 12 | 0 | GE (Frank's website) |
| **6550C.m** | 6550C | Pentode | 6 | 0 | Svetlana |
| **6L6GB.m** | 6L6GB | Pentode | 13 | 0 | Sylvania manual |
| **6L6GC.m** | 6L6GC | Pentode | 11 | 0 | Svetlana SV6L6GC |
| **EL34.m** | EL34 | Pentode | 9 | 0 | Svetlana |
| **KT88.m** | KT88 | Pentode | 12 | 0 | M-O Valve Genalex |

### Purpose for LM19

1. **Reference parameters** — Koren's own published values for comparison with our fitted results
2. **SPICE subcircuit template** — exact reference for our `.sub` generator output format
3. **Pentode equation structure** — the G1/G2 current sources in `.lib` confirm the model equations
4. **Tuparam data** — 15 measured datasets (8 triodes + 7 pentodes) for unit testing

---

## Source 5: Duncan Munro — Leach Model Libraries

**URL:** https://duncanamps.com/spicemodels.html
**Article:** W.M. Leach, "SPICE Models for Vacuum-Tube Amplifiers", JAES 1995
**License:** Free, no warranty
**Format:** PSpice .INC files
**Downloaded:** Yes (13.7 KB zip, 18 files + 2 example circuits)

### Model Type

Duncan's models use the **Leach/three-halves power law** approach — *different* from Koren:
- Triode: `Ip = K * (Vp + mu*Vg)^1.5` (simpler than Koren)
- Pentode: More complex, with plate/screen current splitting

### Available Models

| File | Type | Topology | Notes |
|------|------|----------|-------|
| 12AX7A.INC | 12AX7A | Triode | Valid 25–400V, 0 to -3.5V |
| 12AU7A.INC | 12AU7A | Triode | |
| 12AT7.INC | 12AT7 | Triode | |
| 12BH7A.INC | 12BH7A | Triode | |
| 6DJ8.INC | 6DJ8 | Triode | |
| 6SL7GT.INC | 6SL7GT | Triode | |
| 6SN7GTB.INC | 6SN7GTB | Triode | |
| GL211.INC | GL-211 | Triode | Power triode |
| WE300B.INC | WE 300B | Triode | |
| **6L6GC.INC** | 6L6GC | **Pentode** | 0–800V plate, 0–600V screen |
| **6V6GTA.INC** | 6V6GTA | **Pentode** | |
| **KT88.INC** | KT88 | **Pentode** | Also valid for 6550 |
| 5AR4.INC | 5AR4 | Rectifier | |
| 5U4GB.INC | 5U4GB | Rectifier | |
| 5V3A.INC | 5V3A | Rectifier | |
| 5V4GA.INC | 5V4GA | Rectifier | |
| 5Y3GT.INC | 5Y3GT | Rectifier | |
| HEATER.INC | — | Heater model | Thermal delay |

### Purpose for LM19

- **Cross-validation** — compare Koren model output with Leach model at same operating point
- **Alternative model reference** — if users import Leach .INC files, we know the format
- Not directly usable for our SPICE export (different equation structure)

---

## Source 6: EXCEM/Intusoft — VACUUM.LIB

**URL:** LTwiki mirror (originally from Intusoft)
**License:** Copyright Intusoft 1989-1997
**Format:** IsSpice .lib with TRIO1/PENT1/HEAT1 subcircuit primitives
**Downloaded:** Yes (19 KB VACUUM.LIB + 15 KB vacuumnl.lib)

### Model Type

Uses EXCEM's own tube model primitives (TRIO1, PENT1, HEAT1) — *different from both Koren and Leach*:
- Complex thermal heater modeling
- Different parameter naming (SFS, VBIG, VBIA, MU, RMU, VMU, SFMU, K, RK, VK, SFK...)
- Includes emission saturation and contact potential

### Available Tubes

T12AU7A, T6SN7, T12AX7A, T12AX7WA, T12AT7, T6AQ6, T6AV6, T6AQ5-A, T12BY7A,
Old12AU7, GL-807, BW1185J2, BW1608F, BW1609, T6AW8-A(triode+pentode), EL9000

### Purpose for LM19

- **Academic reference** — shows alternative modeling approaches for vacuum tubes
- **Not directly applicable** — incompatible with Koren equation structure
- May be useful if we ever add heater thermal modeling

---

## Source 6b: Next-Tube.com — SPICE Model Libraries

**URL:** https://next-tube.com/zip/
**Author:** A. Karpov (next-tube.com)
**License:** Free (educational)
**Format:** SPICE .lib subcircuits (Koren equations), packaged for different simulators
**Date:** 2002–2006
**Downloaded:** Yes (5 ZIP files)

### Available Libraries

| File | Size | Simulator | Description |
|------|------|-----------|-------------|
| `PSlib.zip` | 3.6 KB | PSpice | Contains `Tube_IM.lib` — same 18 tubes |
| `MC7.zip` | 4.4 KB | Micro-Cap 6/7 | Micro-Cap format models |
| `Model.zip` | 2.1 KB | SEcalcS(i) | Next-Tube's own calculator format |
| `Tube_ORCAD.zip` | 39 KB | ORCAD | Extended library with more subcircuits |
| `Tube_Msim10.zip` | 285 KB | MSIM-10 | Largest — includes schematics/test circuits |

### Purpose for LM19

- **Cross-reference** — Karpov's fitted Koren params for same 18 tubes as empirical data (Source 2)
- **Validation** — compare our fits against independently fitted params
- `Tube_IM.lib` (from PSlib.zip) also stored in `external_sources/data/Tube_IM.lib`

---

## Source 7: pypsucurvetrace — Precision Curve Tracer Data (GitHub)

**URL:** https://github.com/mbrennwa/curvetracedata
**Software:** https://github.com/mbrennwa/pypsucurvetrace
**License:** GPL-3
**Format:** `.dat` files (space-separated: Ua_nom, Ia_max, Ua_meas, Ia_meas, limiter, Ug_nom, Ig_max, Ug_meas, Ig_meas, limiter, T)
**Date:** 2023–2025
**Downloaded:** Yes (309 .dat files, 22 tube types)

### Data Quality
- **Precision instrument measurements** — programmable PSU curve tracer
- High resolution: 5–10V Ua steps, 0.5–5V Ug steps
- Pre-heat stabilization before each test run (100–300 sec)
- Multiple specimens per tube type from different manufacturers
- All pentodes tested in **triode mode only** (G2 = Anode)
- No Ig2 data

### Available Tubes

| Tube | Topology | Points | Curves | Specimens | Manufacturers |
|------|----------|--------|--------|-----------|---------------|
| **ECC88** | Triode | 141 | 7 | **61** | Siemens, Siemens CCa, Philips-SQ, JJ, Mullard, Sovtek, Amperex |
| **EL34** | Triode mode | 493 | 23 | **28** | Mullard-Russia, Siemens, EHX-Russia, AEG, RFT |
| **300B** | Triode | 850 | 23 | **16** | EHX, Svetlana, TJ Mesh |
| **KT66** | Triode mode | 483 | 13 | **9** | — |
| **6C33C** | Triode | 320 | 15 | **19** | — |
| **6E5P** | Triode mode | 870 | 17 | **32** | Soviet (pentode tested as triode) |
| **6N23P** | Triode | 291 | 10 | **24** | — |
| **6N30P** | Triode | 703 | 21 | 2 | — |
| **ECC81** | Triode | 240 | 5 | 2 | — |
| **PCC88** | Triode | 500 | 18 | 4 | — |
| **D3A** | Triode mode | 941 | 21 | 6 | — |
| **807** | Triode mode | 928 | 13 | 1 | — |
| **801/VT62** | Triode | 1524 | 28 | **38** | — |
| **THF51** | Triode | 546 | 15 | **28** | — |
| **10/VT25** | Triode | 832 | 23 | 11 | — |
| **20B** | Triode | 817 | 9 | 10 | — |
| **32B** | Triode | 938 | 24 | 4 | — |
| **841/VT51** | Triode | 516 | 8 | 4 | — |
| DL92 (3S4) | Triode mode | 206 | 16 | 1 | — |
| DL94 (3V4) | Triode mode | 258 | 14 | 1 | — |
| DL96 (3C4) | Triode mode | 325 | 17 | 1 | — |
| DL98 (3B4) | Triode mode | 163 | 12 | 1 | — |

**Totals: 22 tube types, 12 885 points, 303 usable specimens.**

### Key Value for LM19
1. **Multi-specimen matching** — ECC88 (61), EL34 (28), 300B (16) enable matching algorithm validation
2. **Cross-validation** — ECC88 overlaps with Tuparam, loadline_plotter, and Next-Tube data
3. **High-density curves** — 141–1524 points per type (vs 6–11 in Tuparam)
4. **New tube types** — 300B, KT66, 6N30P, D3A, 807, THF51 not in other sources

### Conversion Notes

By default, the converter picks **one representative specimen** per tube type
(first non-defective `.dat` file) for SPICE model fitting. The JSON file
includes `specimens_total`, `specimens_usable`, and `all_specimens` fields
listing all available specimens.

All 309 raw `.dat` files remain in `raw/curvetracedata/` for multi-specimen
matching analysis — load them directly with the parser from the converter script.

### How to Convert
```bash
# Convert one specimen per tube type → converted/*.json
python tools/convert_curvetracedata_to_lm19.py
```

### How to Load All Specimens (for matching)
```python
from pathlib import Path
from tools.convert_curvetracedata_to_lm19 import parse_dat_file, filter_useful_points

data_dir = Path("tests/spice_test_data/raw/curvetracedata/ECC88/data")
for dat_file in sorted(data_dir.glob("*.dat")):
    sample, date, idle, raw_pts = parse_dat_file(dat_file)
    points = filter_useful_points(raw_pts)
    # points: list of {"ua", "ug1", "ia_A"} — Ia in Amps
```

---

## Source 8: eTracer — Impulse Curve Tracer Data (vt52.com)

**URL:** http://www.vt52.com/etracer-files
**Hardware:** eTracer (Essues Technologies, Netherlands) — impulse measurement method
**Format:** CSV v2.0 (6 rows per curve-set: HV1_V, HV1_I, HV2_V, HV2_I, NEGV, SWEEP_SOURCE)
**Date:** 2018
**Downloaded:** Yes (35 CSV + 35 ETD files)

### Data Quality
- **Impulse measurement** — short pulses, less tube heating than DC methods
- Covers rare/vintage tubes not in other sources (VT52, RS241, KC3, 3a-167m)
- Many Russian tubes (4P1L, 6E5P, 6E6P, 6S45P, 6Z9P, etc.)
- All samples are triode or triode-connected; no pentode-mode data in this set
- Ia values systematically lower than DC measurements (~30–50% for same tube type)

### Available Tubes (29 converted, 5 overlap skipped)

| Tube | Topology | Points | Curves | Notes |
|------|----------|--------|--------|-------|
| **10Y** | Triode | 311 | 8 | Power triode, Vh=7.5V |
| **4P1L** | Triode connected | 531 | 9 | Russian pentode in triode mode |
| **50** | Triode | 473 | 9 | Power triode, Vh=7.5V |
| **5842Q** | Triode | 630 | 13 | =417A/5842, Vh=6.3V |
| **VT52** | Triode | 365 | 11 | =2C34, Vh=6.3V |
| **EL84** | Triode connected | 349 | 9 | =6BQ5, Vh=6.3V |
| **E180F** | Triode | 645 | 13 | Mullard, Vh=6.3V |
| **EC8010** | Triode | 673 | 13 | Vh=6.3V |
| ... | ... | ... | ... | 29 types total, 12 455 points |

### Conversion Notes
- Converter: `tools/convert_etracer_to_lm19.py [--include-overlap] [--dry-run]`
- Heater voltages cross-checked against companion .etd files and datasheets
- 5 overlap tubes (D3a, EL34, KT66, 6E5P, 6S19P) skipped — already in curvetracedata
- Triode-connected files include ug2/ig2 in JSON points
- Source raw CSV files: `lm19_app/tests/spice_test_data/raw/etracer_samples/`

---

## Source 9: AVT5229 / LM19 — Own Hardware Measurements

**Hardware:** AVT5229 (LM19) curve tracer, ATmega16-based
**Format:** Native LM19 measurement JSON, stripped of top-level metadata
(timestamp, scan params, srk, …) so only `points` and a small fixture
header remain.
**License:** Project-internal test fixtures (originally measured by project owner)

### Purpose

Real measurements that **stay coupled with the test suite**. The original
JSON lives in `lm19_app/measurements/` (user data — never read by tests).
Test fixtures are stripped copies kept under `converted/`, so the suite is
hermetic and doesn't depend on user data being present.

### File Naming

`{topology}_{tube_type}_{specimen}_real.json` — `_real` suffix marks the
file as project-owned hardware measurement (vs digitised/datasheet/library
sources). When more than one specimen is captured, the specimen tag (`L1`,
`L2`, …) is part of the filename so multi-specimen matching tests can
pick exact pairs.

### Available Specimens

| File | Tube | Topology | Specimen | Points |
|------|------|----------|----------|--------|
| `pentode_EL84_SOVTEK_L1_real.json` | 6P14P-Sovtek (EL84) | Pentode | L1 | ~1840 |
| `pentode_EL84_SOVTEK_L2_real.json` | 6P14P-Sovtek (EL84) | Pentode | L2 | ~1840 |
| `pentode_EL84_ER_L1_real.json` | 6P14P-ER (EL84) | Pentode | L1 | ~2000 |
| `pentode_EL84_ER_L2_real.json` | 6P14P-ER (EL84) | Pentode | L2 | ~2000 |
| `triode_EL84_SOVTEK_L1_real.json` | 6P14P-Sovtek (EL84) | Triode-connected | L1 | ~280 |
| `triode_EL84_ER_L1_real.json` | 6P14P-ER (EL84) | Triode-connected | L1 | ~280 |
| `pentode_6P1P_real.json` | 6P1P | Pentode | — | — |
| `triode_6P1P_real.json` | 6P1P | Triode-connected | — | — |
| `triode_6S19P_real.json` | 6S19P | Triode | — | — |

### Use in Tests

Real-data tests must read from `converted/` only — never glob
`lm19_app/measurements/`. Helpers and named pair constants live in
`tests/_real_data.py`:

```python
from tests._real_data import (
    EL84_PENTODE_FILES, EL84_TRIODE_FILES,
    EL84_SOVTEK_L1_PENT, EL84_SOVTEK_L2_PENT,
    EL84_ER_L1_PENT, EL84_ER_L2_PENT,
    EL84_SOVTEK_L1_TRI,
    converted_path, load_converted, load_points,
)
```

Multi-specimen pairs (`L1`+`L2`) feed `test_tube_matching.py` matching
sanity checks; single-specimen pentodes feed `test_amplifier_real_data.py`
distortion/Pa_avg/headroom physical-bound checks.

### Adding More Specimens

1. Capture in the LM19 GUI; the file lands in `lm19_app/measurements/<TUBE>/`.
2. Copy `points` into a new `converted/{topology}_{tube}_{specimen}_real.json`
   following the schema of an existing `*_real.json` (header keys: `_comment`,
   `tube_type`, `topology`, `source`, `aliases`, `specimen`,
   `measurement_date`, `_units`, `points`).
3. Reference the new fixture from `tests/_real_data.py` and the test
   that needs it.

---

## Summary: Test Coverage Matrix

### Tuparam (Koren)

| JSON File | Topology | Tube | Points | Expected? | VCT | Status |
|-----------|----------|------|--------|-----------|-----|--------|
| `pentode_6550_tuparam.json` | **Pentode** | 6550 | 11 | Yes | 0 | OK |
| `pentode_6550A_tuparam.json` | **Pentode** | 6550A | 12 | No | 0 | OK |
| `pentode_6550C_tuparam.json` | **Pentode** | 6550C | 6 | No | 0 | OK |
| `pentode_6L6GB_tuparam.json` | **Pentode** | 6L6GB | 13 | No | 0 | OK |
| `pentode_6L6GC_tuparam.json` | **Pentode** | 6L6GC | 11 | No | 0 | OK |
| `pentode_EL34_tuparam.json` | **Pentode** | EL34 | 9 | No | 0 | OK |
| `pentode_KT88_tuparam.json` | **Pentode** | KT88 | 12 | No | 0 | OK |
| `triode_12AT7_tuparam.json` | Triode | 12AT7 | 8 | No | 0 | OK |
| `triode_12AU7_tuparam.json` | Triode | 12AU7 | 8 | No | 0 | OK |
| `triode_12AU7A_tuparam.json` | Triode | 12AU7A | 8 | No | 0 | OK |
| `triode_12AX7AMitch_tuparam.json` | Triode | 12AX7A | 6 | **Yes** | 0 | OK |
| `triode_12AX7ASYL_tuparam.json` | Triode | 12AX7A | 8 | No | **0.5** | OK |
| `triode_6DJ8Mitch_tuparam.json` | Triode | 6DJ8 | 8 | No | **0.5** | OK |
| `triode_6SN7Sylv_tuparam.json` | Triode | 6SN7 | 11 | No | 0 | OK |
| `triode_7025_tuparam.json` | Triode | 7025 | 9 | **Yes** | **0.5** | OK |

### Loadline Plotter (Datasheet)

| JSON File | Topology | Tube | Points | Status |
|-----------|----------|------|--------|--------|
| `triode_12ay7_datasheet.json` | Triode | 12AY7 | 96 | OK |
| `triode_e88cc_datasheet.json` | Triode | E88CC | 48 | OK |
| `triode_ecc81_datasheet.json` | Triode | ECC81 | 103 | OK |
| `triode_ecc82_datasheet.json` | Triode | ECC82 | 95 | OK |
| `triode_ecc83_datasheet.json` | Triode | ECC83 | 85 | OK |
| `triode_ecc85_datasheet.json` | Triode | ECC85 | 86 | OK |

### Next-Tube.com (Empirical)

| JSON File | Topology | Tube | Points | Expected? | VCT | Status |
|-----------|----------|------|--------|-----------|-----|--------|
| `triode_6N2P_nexttube.json` | Triode | 6N2P (12AX7) | 30 | **Yes** | **0.3** | OK |
| `triode_6N23P_nexttube.json` | Triode | 6N23P (6DJ8) | 29 | **Yes** | 0 | OK |
| `triode_6N8C_nexttube.json` | Triode | 6N8C (6SN7) | 35 | **Yes** | 0 | OK |
| `triode_6N1P_nexttube.json` | Triode | 6N1P (ECC85) | 20 | **Yes** | 0 | OK |
| `triode_6N5P_nexttube.json` | Triode | 6N5P | 22 | **Yes** | 0 | OK |
| `triode_6N6P_nexttube.json` | Triode | 6N6P | 29 | No | 0 | OK |
| `triode_6N13S_nexttube.json` | Triode | 6N13S (6AS7G) | 98 | **Yes** | 0 | OK |
| `triode_6N3P_nexttube.json` | Triode | 6N3P (2C51) | 70 | **Yes** | 0 | OK |
| `triode_6N27P_nexttube.json` | Triode | 6N27P (ECC86) | 72 | **Yes** | 0 | OK |
| `triode_6S6B_nexttube.json` | Triode | 6S6B | 77 | **Yes** | 0 | OK |
| `triode_6S19P_nexttube.json` | Triode | 6S19P | 22 | **Yes** | 0 | OK |
| `triode_6C41C_nexttube.json` | Triode | 6C41C | 48 | No | 0 | OK |
| `triode_6S4S_nexttube.json` | Triode | 6S4S (6A3) | 35 | **Yes** | 0 | OK |
| `triode_GU50_T_nexttube.json` | Triode | GU50 (triode) | 35 | No | 0 | OK |
| `triode_6P14P_T_nexttube.json` | Triode | 6P14P (EL84-T) | 65 | **Yes** | 0 | OK |
| `triode_6P45S_T_nexttube.json` | Triode | 6P45S (triode) | 43 | **Yes** | 0 | OK |
| `triode_6F5P_T_nexttube.json` | Triode | 6F5P triode | 46 | **Yes** | **0.6** | OK |
| `triode_6F5P_PT_nexttube.json` | Triode | 6F5P pent→tri | 67 | **Yes** | 0 | OK |
| `triode_6P3C_T_nexttube.json` | Triode | 6P3C (6L6-T) | 45 | No | 0 | OK |
| `pentode_6P3C_P_nexttube.json` | **Pentode** | 6P3C (6L6) | 37 | No | 0 | OK |
| `pentode_6F5P_P_nexttube.json` | **Pentode** | 6F5P pentode | 61 | **Yes** | 0 | OK |
| `pentode_GU50_P_nexttube.json` | **Pentode** | GU50 pentode | 37 | No | 0 | OK |

### pypsucurvetrace (Precision Measurements)

| JSON File | Topology | Tube | Points | Specimens | Status |
|-----------|----------|------|--------|-----------|--------|
| `triode_ECC88_curvetracedata.json` | Triode | ECC88 (6DJ8) | 141 | 61 | OK |
| `triode_EL34_curvetracedata.json` | Triode | EL34 (tri mode) | 493 | 28 | OK |
| `triode_300B_curvetracedata.json` | Triode | 300B | 850 | 16 | OK |
| `triode_KT66_curvetracedata.json` | Triode | KT66 (tri mode) | 483 | 9 | OK |
| `triode_6C33C_curvetracedata.json` | Triode | 6C33C | 320 | 19 | OK |
| `triode_6N23P_curvetracedata.json` | Triode | 6N23P (ECC88) | 291 | 24 | OK |
| `triode_6N30P_curvetracedata.json` | Triode | 6N30P | 703 | 2 | OK |
| `triode_ECC81_curvetracedata.json` | Triode | ECC81 (12AT7) | 240 | 2 | OK |
| `triode_PCC88_curvetracedata.json` | Triode | PCC88 | 500 | 4 | OK |
| `triode_6E5P_curvetracedata.json` | Triode | 6E5P (tri mode) | 870 | 32 | OK |
| `triode_D3A_curvetracedata.json` | Triode | D3A (tri mode) | 941 | 6 | OK |
| `triode_807_curvetracedata.json` | Triode | 807 (tri mode) | 928 | 1 | OK |
| `triode_801_VT62_curvetracedata.json` | Triode | 801/VT62 | 1524 | 38 | OK |
| `triode_THF51_curvetracedata.json` | Triode | THF51 | 546 | 28 | OK |
| `triode_10_VT25_curvetracedata.json` | Triode | 10/VT25 | 832 | 11 | OK |
| `triode_20B_curvetracedata.json` | Triode | 20B | 817 | 10 | OK |
| `triode_32B_curvetracedata.json` | Triode | 32B | 938 | 4 | OK |
| `triode_841_VT51_curvetracedata.json` | Triode | 841/VT51 | 516 | 4 | OK |
| `triode_DL92_curvetracedata.json` | Triode | DL92 (tri mode) | 206 | 1 | OK |
| `triode_DL94_curvetracedata.json` | Triode | DL94 (tri mode) | 258 | 1 | OK |
| `triode_DL96_curvetracedata.json` | Triode | DL96 (tri mode) | 325 | 1 | OK |
| `triode_DL98_curvetracedata.json` | Triode | DL98 (tri mode) | 163 | 1 | OK |

**Validation: 94/94 files valid, 0 errors, 0 warnings.**

### Totals

| Category | Count | Notes |
|----------|-------|-------|
| Converted Tuparam triodes | 8 | 12AX7A(x2), 7025, 12AU7(x2), 12AT7, 6SN7, 6DJ8 |
| Converted Tuparam pentodes | 7 | 6550(x3), 6L6(x2), EL34, KT88 |
| Converted loadline_plotter triodes | 6 | ECC83/82/81/85, E88CC, 12AY7 |
| Converted Next-Tube triodes | 19 | 14 pure triodes + 5 pentode→triode |
| Converted Next-Tube pentodes | 3 | 6P3C, 6F5P, GU50 |
| Converted curvetracedata triodes | 22 | 303 specimens, precision measurements |
| Converted eTracer triodes | 29 | impulse measurements, incl. triode_connected |
| Koren Tube.lib reference params | 13 | 8 triodes + 5 pentodes |
| Next-Tube expected params | 16 | 15 triode + 1 pentode (Karpov fits) |
| Duncan Leach .INC reference models | 18 | 9 triodes + 3 pentodes + 5 rect + 1 heater |
| EXCEM VACUUM.LIB reference models | ~20 | Alternative model (TRIO1/PENT1/HEAT1) |
| **Total converted data points** | **27 016** | 26 881 triode + 135 pentode |
| **Total specimens (matching)** | **303** | ECC88(61), EL34(28), 300B(16), 801(38), THF51(28)... |

### Gaps & Mitigations
1. **Pentode with Ig2 data** — no real measurement data available. Mitigated by:
   - `TestPentodeFitterWithIg2Data` uses synthetic Ig2 from known Koren params + noise
   - Validates the combined Ia+Ig2 residual fitting path
   - Real Ig2 data can come from own AVT5229 measurements
2. **Contact potential (VCT)** — 7025 (0.5), 6DJ8 (0.5), 12AX7A-Syl (0.5),
   6N2P (0.3), 6F5P-T (0.6). Good coverage across sources.
3. **Numpy-only pipeline** — scipy is always installed in test env. Individual numpy
   fitters are tested directly; the pipeline dispatch to numpy is not separately tested.
4. **Cross-source validation** — 6N2P↔12AX7, 6N8C↔6SN7, 6N23P↔6DJ8 enable
   comparison of Karpov's fits with Koren's published params for the same tube families.
5. **Multi-specimen matching** — curvetracedata provides 303 individual specimens
   across 22 tube types for matching algorithm validation and scatter analysis.

---

## How to Use in Tests

**Test file:** `tests/test_spice_export.py` (94 tests, all data used)

```bash
py -m unittest tests.test_spice_export -v   # run SPICE tests only
py -m unittest discover -s tests -v          # run all 336 tests
```

### Test Coverage (by class)

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `TestKorenTriodeModel` | 6 | Triode equation vs hand-calculated values |
| `TestKorenPentodeModel` | 6 | Pentode Ia + Ig2 equations, cutoff, knee |
| `TestTriodeFittingScipy` | 14 | Scipy fit on all 14 triode datasets (RMS < 8-10%) |
| `TestTriodeFittingNumpy` | 4 | Numpy fallback fit (RMS < 12-15%) |
| `TestPentodeFittingScipy` | 9 | Scipy fit on all 7 pentode datasets + param check |
| `TestPentodeFittingNumpy` | 3 | Numpy fallback for pentodes |
| `TestTriodeSubcircuit` | 9 | .SUBCKT format, equations, SPICE syntax |
| `TestPentodeSubcircuit` | 8 | 4-pin header, ATAN, G2 source, Koren lib match |
| `TestTriodeSubcircuitWithRef` | 6 | Caps, RGI, VCT, grid stopper in output |
| `TestPentodeSubcircuitWithRef` | 5 | Pentode caps, Ig2 RMS comment |
| `TestSubcircuitWithoutRef` | 2 | No caps/diode when ref=None |
| `TestFullPipelineTriode` | 2 | End-to-end: ECC83/ECC82 → .sub file |
| `TestFullPipelinePentode` | 5 | End-to-end: 6550/6550A/6L6GB/6L6GC/KT88 |
| `TestPipelineTopologyDetection` | 2 | Auto-detect, triode_connected → triode |
| `TestPipelineUnknownTube` | 2 | Unknown tube defaults + explicit pentode |
| `TestPipelineEdgeCases` | 3 | Too few points, zero Ia, missing ug2 |
| `TestFitterWithRefKoren` | 2 | Explicit ref_koren seed, kg2=None fallback |
| `TestPentodeFitterWithIg2Data` | 1 | Combined Ia+Ig2 residual fitting |
| `TestSubcircuitEdgeCases` | 4 | rgi=0 fallback, no caps, no source, no ig2_rms |
| `TestPublishedParamsComparison` | 2 | mu vs Koren tube.lib published values |

### Loading Data in Custom Tests

```python
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "spice_test_data" / "converted"

def load_test_data(filename):
    with open(DATA_DIR / filename) as f:
        return json.load(f)

# Example: triode fit verification
data = load_test_data("triode_12AX7AMitch_tuparam.json")
points = data["points"]
expected = data["expected_params"]

ua  = [p["ua"] for p in points]
ug1 = [p["ug1"] for p in points]
ia  = [p["ia"] for p in points]  # mA

# Fit and compare with expected["mu"], expected["ex"], etc.
```
