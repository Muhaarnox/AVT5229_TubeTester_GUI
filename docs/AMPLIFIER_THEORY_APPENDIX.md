# Amplifier Theory Appendix

This appendix consolidates theory references used by amplifier analysis.
Source entries below are duplicated from `lm19_app/external_sources/SOURCES_INDEX.md`
for convenient traceability in amplifier documentation.

## References (from SOURCES_INDEX)

### Radiotron Designer's Handbook, 4th ed. (Langford-Smith, 1953) [RDH]
- url: <https://worldradiohistory.com/ENCYCLOPEDIAS/Radiotron-Designers-Handbook-4th-Edition-1953.pdf>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Primary reference for 5-point distortion, SE gain/Zout, CF, PP theory.

### ResearchGate — Methods for Computing Harmonic Distortion (paper) [ResearchGate]
- url: <https://www.researchgate.net/publication/242307545_METHODS_FOR_COMPUTING_HARMONIC_DISTORTION_IN_LOW_FREQUENCY_POWER_AMPLIFIER>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Cross-check for selected-ordinate methods (3/5/7/11-point) and HD formulas.

### Terman — Electronic and Radio Engineering, 4th ed. (1955) [Terman]
- url: n/a (book: Terman F.E., McGraw-Hill, 1955)
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Independent reference for selected-ordinate harmonic analysis, IMD, tube small-signal model.

### Valley & Wallman — Vacuum Tube Amplifiers (MIT Rad. Lab. Vol. 18, 1948) [V&W]
- url: <https://archive.org/details/MITRadiationLaboratorySeries>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Foundation for small-signal derivations (Av, Zout, Thevenin/Norton equivalents).

### Blencowe — Designing Tube Preamps for Guitar and Bass, 2nd ed. (2012) [Blencowe]
- url: n/a (book: Merlin Blencowe, 2012)
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Supplemental reference for practical power and cathode-follower interpretations.

### Next Electronics — Harmonic Distortion in Amplifiers Tutorial [NextElectronics]
- url: <https://next.gr/tutorials/audio-electronics/harmonic-distortion-in-amplifiers-tutorial>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: THD/IMD definitions and Fourier/Taylor interpretation.

### TubeCad — Cathode Follower Output Stage & Simple Tube Math [TubeCad]
- url: <https://tubecad.com/2005/June/blog0049.htm>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: CF formulas for gain and output impedance.

### VTADiy — Push-Pull Loadline in Class AB [VTADiy]
- url: <https://www.vtadiy.com/book/chapter-4-integrated-push-pull-vacuum-tube-amplifier/4-1-output-stage-or-power-stage/4-1-6-push-pull-loadline-in-class-ab/>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: PP load-line and output power relationships.

### Vacuum-tube.eu — Push-Pull Transformer Impedance [VacTube]
- url: <https://www.vacuum-tube.eu/www.voltsecond/Push-Pull_Z/Push-Pull_Pri_Z.html>
- type: theory
- used_in:
  - `lm19_app/lm19/amplifier/`
  - `lm19_app/docs/AMPLIFIER_ANALYSIS_PLAN.md`
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Derivation for per-tube impedance scaling (`Ra_per_tube = Ra_aa / 4`).

### DAFx-23 — A Quadric Surface Model of Vacuum Tubes for Virtual Analog Applications [DAFx23]
- url: <https://dafx.de/paper-archive/2023/DAFx23_paper_15.pdf>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Modern virtual-analog tube modeling context (supplementary, not primary for classic closed-form equations).

### Aiken Amps — Designing Common-Cathode Triode Amplifiers [Aiken]
- url: <https://www.aikenamps.com/index.php/designing-common-cathode-triode-amplifiers>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Practical load-line and triode-stage explanations as modern supplementary reference.

### Stanford CCRMA — Real-Time Wave Digital Simulation of Cascaded Vacuum Tube Amplifiers [StanfordWDF]
- url: <https://ccrma.stanford.edu/~jingjiez/publications/Real-Time%20Wave%20Digital%20Simulation%20of%20Cascaded%20Vacuum%20Tube%20Amplifiers%20using%20Modified%20Blockwise%20Method.pdf>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Modern wave-digital simulation approach for virtual-analog tube amplifier modeling.

### IEEE — A Vacuum-Tube Guitar Amplifier Model Using Long/Short-Term Memory Networks [IEEELSTM]
- url: <https://ieeexplore.ieee.org/document/8479039/>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Data-driven modern modeling direction (LSTM-based approximation of tube amplifier behavior).

### Hafler & Keroes — An Ultra-Linear Amplifier (Audio Engineering, 1951) [Hafler&Keroes]
- url: <https://rubli.net/classic_amps/files/articles/ulamp/ulamp.htm>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Original UL paper. Screen tap at ~18.5% plate impedance. IM distortion minimum.

### RCA Receiving Tube Manual, RC-30 (1975) [RCA RC-30]
- url: <https://archive.org/details/RCA_RC-30_1975>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Tube data, noise characteristics, thermal derating, class A/AB/B operation.

### Mullard — Circuits for Audio Amplifiers (1963) [Mullard]
- url: <https://www.worldradiohistory.com/BOOKSHELF-ARH/Circuits-For-Audio-Amplifiers-Mullard-1963%20.pdf>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Phase splitters, UL output stages, NFB design, complete amplifier circuits.

### Valve Wizard — Cathodyne Phase Inverter (Blencowe) [Blencowe cathodyne]
- url: <https://valvewizard.co.uk/cathodyne.html>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Cathodyne gain mu/(mu+2), Zout differential vs common-mode, overdrive behavior.

### Valve Wizard — DC Coupled Long-Tailed Pair (Blencowe) [Blencowe LTP]
- url: <https://www.valvewizard.co.uk/dcltp.html>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: LTP gain A/2 per output, balance via Ra mismatch, tail resistor design.

### VTADiy — Global Negative Feedback §4.4 [VTADiy NFB]
- url: <https://www.vtadiy.com/book/chapter-4-integrated-push-pull-vacuum-tube-amplifier/4-4-global-negative-feedback/>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Afb = A/(1+β*A), THD reduction, stability analysis, step network compensation.

### MIT — Johnson Noise and Shot Noise (Perepelitsa, 2006) [MIT noise]
- url: <https://web.mit.edu/dvp/Public/noise-paper.pdf>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: I_shot = sqrt(2*e*Idc*BW), V_johnson = sqrt(4*k*T*R*BW). Experimental verification.

### Carniti et al. — Rosenstark Extended Method (Electronics, 2025) [Rosenstark2025]
- url: <https://www.mdpi.com/2079-9292/14/8/1558>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Unified gain/impedance/noise in feedback amps. Open-access.

### SB-LAB — NFB Position (Bianchini, 2025) [SB-LAB NFB 2025]
- url: <https://www.sb-lab.eu/en/negative-feedback-and-zero-feedback-in-tube-amplifiers-my-position/>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Practical NFB analysis: myths, moderate feedback benefits, stability criticalities.

### SB-LAB — UL in SE Myth (Bianchini, 2025) [SB-LAB UL 2025]
- url: <https://www.sb-lab.eu/en/the-great-myth-debunked-why-ultralinear-in-single-ended-designs-is-a-mistake/>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: UL effective only in PP; SE+UL increases odd-harmonic distortion.

### DAFx 2022 — Neural WDF Tube Model [DAFx22]
- url: <https://dafx.de/paper-archive/2022/papers/DAFx20in22_paper_13.pdf>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Neural networks in WDF for multiport tube nonlinearities.

### Cascade Tubes — EL84 SE-UL Optimization (2024) [CascadeEL84 2024]
- url: <https://www.cascadetubes.com/2024/04/26/more-thoughts-on-the-el84-se-ul-optimization/>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Experimental EL84 UL data. Beam power vs pentode differences. Max 275V.

### Analog Ethos — Ultralinear Mode Explained (2024) [AnalogEthos UL 2024]
- url: <https://www.analogethos.com/post/screen-ultralinear>
- type: theory
- used_in:
  - `lm19_app/docs/AMPLIFIER_THEORY_APPENDIX.md`
- note: Modern explanation of triode/pentode/UL trade-offs. Screen tap 40-45%.

### Terman Electronic and Radio Engineering metadata pages
- url: <https://books.google.com/books?id=LAgjAAAAMAAJ> ; <https://openlibrary.org/works/OL7456725W/Electronic_and_radio_engineering>
- type: documentation
- note: Bibliographic confirmation only (edition/year/page count). No direct
  chapter text access.

### Terman Radio Engineer's Handbook (1943, Archive copy)
- url: <https://archive.org/download/in.ernet.dli.2015.126496/2015.126496.Radio-Engineers-Handbook.pdf>
- type: theory
- role here: secondary contextual reference for tube-era methods
- note: Downloaded successfully as alternative Terman source; not a direct
  replacement for 1955 E&RE section citations.

### Triodes at Low Voltages (Merlin Blencowe PDF)
- url: <https://valvewizard.co.uk/Triodes_at_low_voltages_Blencowe.pdf>
- type: theory
- role here: context on low-voltage triode behavior
- note: Downloaded successfully from Valve Wizard site; supplemental
  reference.

### Crowhurst — Understanding Hi-Fi Circuits (Gernsback No. 64, 1957)
- url: <https://www.worldradiohistory.com/Archive-All-Audio/Archive-Audio-Collection/Gernsback-64-Understanding-Hi-Fi-Circuits-1957-Crowhurst.pdf>
- type: theory
- role here: candidate supplemental reference
- note: Classic Crowhurst text (Ch. 1 "Special output stages", output-load
  discussion). Establishes the historical practice this optimizer automates:
  optimum load is found by sliding a load line over measured plate curves
  and reading distortion off selected ordinates — i.e. a 1D/2D manual search
  around datasheet operating points, NOT dense 4D enumeration. Supports
  seeding the optimizer from datasheet-typical operating points and treating
  Ub/Ug2 as coarse outer axes.

## Citation convention

Use the following traceability chain for each non-trivial formula:

- source (theory reference tag, for example `[RDH]`, `[Terman]`);
- implemented_in (`lm19_app/lm19/amplifier/` function);
- validated_by (`lm19_app/tests/...` test case).

Minimal template:

**### N) Formula name [tags]**

- **Formula**: `<equation>`
- **Implemented in**: `<module>::<function>`
- **Validated by**: `<test_file>::<test_name>`
- **Sources**: [RDH], [Terman]
- **Notes**: assumptions / limits

## Detailed derivations

### 1) Small-signal tube model [RDH, Terman, V&W]

Triode small-signal parameters:

- `gm = dIa/dVg | Ua=const`
- `ra = dUa/dIa | Vg=const`
- `mu = -dUa/dVg | Ia=const`

Fundamental relationship: `mu = gm * ra`

Derivation:

- `dIa = (dIa/dVg)*dVg + (dIa/dUa)*dUa = gm*dVg + dUa/ra`
- for `dIa = 0`: `dUa/dVg = -gm*ra = -mu`

Implemented in:

- `lm19_app/lm19/amplifier/` -> `_numerical_gm_ra()`, `compute_stage_params()`.

Validated by:

- `lm19_app/tests/test_amplifier_real_data.py`
- `lm19_app/tests/test_amplifier_smoke.py`

### 2) 5-point selected-ordinate distortion [RDH Ch.7, Terman §13.5]

Input signal: `Vg(theta) = V_bias + A*cos(theta)`

Output current approximation: `i(theta) = I0 + B1*cos(theta) + B2*cos(2*theta) + B3*cos(3*theta) + ...`

Sampling points: `I_max`, `I_h+`, `I_q`, `I_h-`, `I_min`.

Resulting coefficients:

- `B1 = (I_max - I_min + I_h+ - I_h-) / 3`
- `B2 = (I_max + I_min - 2*I_q) / 4`
- `B3 = (I_max - I_min - 2*(I_h+ - I_h-)) / 6`

Distortion metrics:

- `HD2 = |B2 / B1| * 100%`
- `HD3 = |B3 / B1| * 100%`
- `THD ~= sqrt(HD2^2 + HD3^2)`

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_distortion()`.

Validated by:

- `lm19_app/tests/test_amplifier.py` -> `TestAnalyticalHD`.
- `lm19_app/tests/test_amplifier.py` -> `TestDistortionGuards`.

Extended details (restored from the previous plan revision):

Trigonometric sampling table for selected ordinates:

| theta | cos(theta) | cos(2*theta) | cos(3*theta) |
|---|---|---|---|
| 0 deg | 1 | 1 | 1 |
| 60 deg | 1/2 | -1/2 | -1 |
| 90 deg | 0 | -1 | 0 |
| 120 deg | -1/2 | -1/2 | 1 |
| 180 deg | -1 | 1 | -1 |

Linear system used in the derivation:

- (1) `I_max = I0 + B1 + B2 + B3`
- (2) `I_h+  = I0 + B1/2 - B2/2 - B3`
- (3) `I_q   = I0 - B2`
- (4) `I_h-  = I0 - B1/2 - B2/2 + B3`
- (5) `I_min = I0 - B1 + B2 - B3`

Auxiliary substitutions:

- `S = I_max - I_min = 2*B1 + 2*B3`
- `H = I_h+ - I_h-   = B1 - 2*B3`

Therefore:

- `B1 = (S + H)/3`
- `B3 = (S - 2*H)/6`
- `B2 = (I_max + I_min - 2*I_q)/4`

Numerical sanity example (quadratic nonlinearity):

- Given: `Ia(Vg) = 10 + 5*Vg + 0.2*Vg^2`, `bias = 0`, `A = 1`
- `I_max = 15.2`, `I_h+ = 12.55`, `I_q = 10`, `I_h- = 7.55`, `I_min = 5.2`
- `B1 = 5.0`, `B2 = 0.1`, `B3 = 0`
- `HD2 = 2.0%`, `HD3 = 0%`

This matches the analytical quadratic estimate `HD2 = |a2*A/(2*a1)|`.

### 3) Output power relation [RDH Ch.7.5]

For sinusoidal current/voltage swing: `P_out = I_pp * U_pp / 8`

Units in implementation: `mA * V = mW`.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_distortion()`, `pp_distortion()`.

### 3b) SE resistive load line [RDH]

The fundamental DC load line for a resistive-loaded stage:

- `Ia = (Ub - Ua) / Ra`

where `Ub` is supply voltage (V), `Ua` is anode voltage (V), `Ra` is load resistance (kOhm),
`Ia` is anode current (mA).

Intersections of measurement curves with this line define the operating points
at each grid voltage `Ug1`. The sign change of `d(Ua) = Ia_curve(Ua) - Ia_load(Ua)`
determines crossing.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `ResistiveLoadLine.ia_at_ua()`, `find_intersections()`.

Validated by:

- `lm19_app/tests/test_amplifier.py`
- `lm19_app/tests/test_amplifier_smoke.py`

### 4) SE gain and output impedance [RDH Ch.5, V&W Ch.2]

Triode stage:

- `|Av| = mu*Ra / (ra + Ra)`
- `Zout = ra || Ra = ra*Ra / (ra + Ra)`

Pentode approximation (`ra >> Ra`):

- `Av ~= gm * Ra`
- `Zout ~= Ra`

Fallback without SRK data:

- `gm` estimated from local slope near `Ua_q`;
- `ra` estimated from local slope near `Vg_bias`.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_stage_params()`, `_numerical_gm_ra()`.

Extended details (restored):

Thevenin equivalent small-signal model at the anode: voltage source `mu*Vgk`
in series with internal resistance `ra`, loaded by `Ra` to ground.

Voltage divider interpretation: `Vout = mu*Vgk * Ra/(ra + Ra)`

### 4b) Transformer AC/DC load line [RDH Ch.6]

Transformer-coupled stage uses different load lines for operating point and signal swing:

DC load line (Q-point): `Ia_dc = (Ub - Ua) / Ra_dc`

where `Ra_dc` is winding resistance (typically much lower than reflected AC load),
so `Ua_q` tends to stay close to `Ub`.

AC load line (signal path through reflected load):

- `Ia_ac = Ia_q - (Ua - Ua_q) / Ra_ac`
- `Ra_ac = n^2 * R_load / 1000` (kOhm)

Meaning:

- DC line defines quiescent operating point.
- AC line passes through Q-point with slope `-1/Ra_ac` and defines distortion/power swing.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `TransformerLoadLine.ia_at_ua()`, `ia_at_ua_ac(ua, q_ua, q_ia)`.

Validated by:

- `lm19_app/tests/test_amplifier.py`
- `lm19_app/tests/test_amplifier_smoke.py`

### 5) Cathode follower formulas [RDH Ch.12, TubeCad]

Gain: `Av_cf = mu*Rk / (ra + (mu + 1)*Rk)`

Output impedance: `Zout_cf = ra / (mu + 1) ~= 1/gm`

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_cf_stage_params()`.

Extended derivation (restored):

- `Vgk = Vin - Vout`
- `Vak = -Vout`
- `Ia = gm*(Vin - Vout) - Vout/ra`
- `Vout = Ia*Rk`
- `Vout*(1 + gm*Rk + Rk/ra) = gm*Rk*Vin`
- `Av_cf = mu*Rk / (ra + (mu + 1)*Rk)`

Output impedance derivation with `Vin = 0` and test cathode voltage `V`:

- `Vgk = -V`, `Vak = -V`
- `I_test = -V*(gm + 1/ra) = -V*(mu + 1)/ra`
- `Zout_cf = V/|I_test| = ra/(mu + 1) ~= 1/gm`

### 6) Push-pull composite behavior [RDH Ch.13, VTADiy, VacTube]

Composite current: `Ia_comp(Vg) = Ia_A(Vg) - Ia_B(2*bias - Vg)`

Matched pair gives odd symmetry around bias, therefore even harmonics cancel.

Per-tube reflected load (Class AB/B): `Ra_per_tube = Ra_aa / 4`

Class A note: effective per-tube load is typically `Ra_aa / 2`.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `composite_characteristic()`, `PushPullLoadLine`, `pp_distortion()`.

Validated by:

- `lm19_app/tests/test_amplifier.py`
- `lm19_app/tests/test_amplifier_smoke.py`
- `lm19_app/tests/test_amplifier_spice_dataset.py`

Additional notes (restored):

- Class AB/B approximation uses `Ra_per_tube = Ra_aa/4`.
- Class A operation is typically approximated by `Ra_per_tube = Ra_aa/2`.
- In `pp_distortion()`, absolute values for odd coefficients are used as a
  protective measure for strongly imbalanced/non-monotonic composite curves.

Implementation note on `abs()` in PP B1/B3: the SE `compute_distortion()` computes
`b1 = (swing + half_diff) / 3` without abs(), which is mathematically correct per
the 5-point theory. The PP `pp_distortion()` uses `abs(swing)` and `abs(half_diff)`,
which prevents negative B1 for pathological composite curves but may produce
incorrect B3 for imbalanced pairs where signs of swing and half_diff differ.
This is a known trade-off for robustness vs mathematical purity.

### 7) IMD model [Terman §13.7]

Polynomial approximation around bias:

`Ia(v) ~= a0 + a1*v + a2*v^2 + a3*v^3`, where `v = Vg - V_bias`

Simplified IMD metrics:

- `IMD2 = |a2 / a1| * 100%`
- `IMD3 = |a3 / a1| * 100%`
- `IMD_total = sqrt(IMD2^2 + IMD3^2)`

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_imd()`.

Validated by:

- `lm19_app/tests/test_amplifier.py` -> `TestIMD`.

Implementation note: coefficients are obtained via `numpy.polyfit` (degree 3, cubic fit
around bias point). The polynomial is fitted to measured `Ia(Ug1)` data within the
signal swing range.

Normalization note: `IMD2 = |a2/a1|` implicitly assumes unit signal amplitude (A=1V).
Standard SMPTE IMD (two-tone test, e.g. 60+7000 Hz, 4:1 ratio) depends on the LF tone
amplitude: `IMD2_SMPTE = |a2 * A_LF / a1|`. The metric here is the normalized
nonlinearity ratio, proportional to SMPTE IMD but independent of test amplitude.

Numerical sanity example:

- Given: `Ia(Vg) = 10 + 5*Vg + 0.3*Vg^2 + 0.05*Vg^3`, `bias = 0`, swing `A = 2V`
- `a1 = 5`, `a2 = 0.3`, `a3 = 0.05`
- `IMD2 = |0.3/5| * 100% = 6.0%`
- `IMD3 = |0.05/5| * 100% = 1.0%`

### 8) Auto-bias cathode resistor estimate

For auto-bias estimate from operating point:

`Rk = |Ug1| / Ia * 1000` (Ohm)

where:

- `Ug1` is bias voltage in volts (typically negative);
- `Ia` is quiescent current in mA.

Unit check: `V / mA = kOhm`, then `*1000 -> Ohm`.

Implemented in:

- `lm19_app/lm19/amplifier/` -> `optimize_bias()` (`rk_auto_bias`).

Validation notes:

- Returned only for meaningful current (`Ia > 0.01 mA`) to avoid unstable values.

Numerical sanity example:

- Given: `Ug1 = -8V`, `Ia = 40 mA`
- `Rk = 8 / 40 * 1000 = 200 Ohm`
- With bypass capacitor, this sets the DC operating point while AC signal
  passes through unimpeded.

### 9) Headroom definition and clipping limits

Headroom is the maximum symmetric input half-swing around selected bias:

- `swing_neg = |Ug1_bias - clip_neg|`
- `swing_pos = |clip_pos - Ug1_bias|`
- `max_swing = min(swing_neg, swing_pos)`

Clipping limits defined by:

- negative-side cutoff (`Ia -> 0`) for `clip_neg`;
- positive-side grid-current boundary (`Ug1 -> 0`) baseline for `clip_pos`;
- optional thermal limit (`Pa > Pa_max`) can reduce `clip_pos`.

Auxiliary operating-point power: `Pa_q = Ua_q * Ia_q / 1000` (W)

Implemented in:

- `lm19_app/lm19/amplifier/` -> `compute_headroom()`.

Validated by:

- `lm19_app/tests/test_amplifier.py` (`TestHeadroom`).

Numerical sanity example:

- Given: 12AX7 with `Ug1_bias = -2.0V`, cutoff at `Ug1 = -4.0V`, grid current at `Ug1 = 0V`
- `swing_neg = |-2.0 - (-4.0)| = 2.0V`
- `swing_pos = |0 - (-2.0)| = 2.0V`
- `max_swing = min(2.0, 2.0) = 2.0V` (symmetric case)
- If `Pa_max` limit reduces `clip_pos` to `Ug1 = -0.5V`: `swing_pos = 1.5V`, `max_swing = 1.5V`

### 10) Negative Feedback (NFB) [RDH Ch.12, VTADiy §4.4]

Closed-loop gain: `Afb = A / (1 + β*A)`

where `A` is open-loop gain, `β` is feedback factor (voltage divider `R2 / (R1 + R2)`).

Derived effects:

- `Zout_fb = Zout / (1 + β*A)` — output impedance reduced by feedback factor
- `THD_fb = THD / (1 + β*A)` — distortion reduced at same output level
- `BW_fb = BW * (1 + β*A)` — bandwidth extended

Amount of feedback in dB: `fb_dB = 20*log10(A / Afb) = 20*log10(1 + β*A)`

Stability criterion: at frequency where phase shift = 180°, loop gain `β*A` must be < 1.
Compensation: step network (`Csn` parallel with `R1`) shifts phase margin.

Implemented in: ✅ `lm19/amplifier/` -> `compute_nfb_effect()`.

Sources: [RDH] Ch.12, [VTADiy] §4.4.

### 11) Frequency response and Miller effect [V&W Ch.2, Terman §7]

High-frequency pole (Miller effect):

- `C_miller = C_gp * (1 + |Av|)` — effective input capacitance
- `f_high = 1 / (2*π * R_source * C_miller)` — dominant pole
- `C_gp` — grid-to-plate capacitance (typically 1-4 pF for triodes)

Low-frequency pole (coupling capacitor):

- `f_low = 1 / (2*π * R_total * C_coupling)`
- `R_total = (Ra || ra) + Rg_next` — series connection: Thevenin resistance of driving stage plus grid leak of next stage
- Since typically `Rg_next >> Ra || ra`: `f_low ≈ 1 / (2*π * Rg_next * C_coupling)`

Bandwidth: `BW = f_high - f_low`

Gain-bandwidth product: `GBW = |Av| * f_high`

For pentodes Miller effect is negligible (`C_gp << 0.1 pF`) due to screen shielding.

Numerical example:
- 12AX7: `gm = 1.6 mA/V`, `ra = 62.5 kΩ`, `C_gp ~ 1.7 pF`, `Ra = 100 kΩ`, `Av ~ 60`
- `C_miller = 1.7 * (1 + 60) = 103.7 pF`
- With `R_source = 50 kΩ`: `f_high = 1 / (2π * 50e3 * 103.7e-12) ~ 30.7 kHz`

Implemented in: planned (`lm19/amplifier/` -> `estimate_bandwidth()`).

Sources: [V&W] Ch.2, [Terman] §7, [RDH] Ch.5.

### 12) Cathode bypass capacitor effect [RDH Ch.12, Blencowe]

Without bypass (local NFB through Rk):

- `Av_unbypassed = mu * Ra / (ra + Ra + (mu+1)*Rk)`

With full bypass (Ck large enough):

- `Av_bypassed = mu * Ra / (ra + Ra)`

Transition frequency (simplified): `f_c = 1 / (2*π * Rk * Ck)`

This is a lower bound. The exact transition frequency accounts for the impedance
looking into the cathode from Ck: `Zk = Rk || ((ra + Ra) / (mu + 1))`. Since
typically `(ra + Ra)/(mu + 1) << Rk`, the actual `f_c` is higher.
Precise formula: `f_c = 1 / (2*π * (Rk || ((ra+Ra)/(mu+1))) * Ck)`.

Below `f_c` — gain is reduced by local NFB. Above `f_c` — full gain restored.

Gain ratio: `Av_bypassed / Av_unbypassed = (ra + Ra + (mu+1)*Rk) / (ra + Ra)`

Numerical example:
- 12AX7: `mu = 100`, `ra = 62.5 kΩ`, `Ra = 100 kΩ`, `Rk = 1.5 kΩ`
- `Av_bypassed = 100*100 / (62.5 + 100) = 61.5`
- `Av_unbypassed = 100*100 / (62.5 + 100 + 101*1.5) = 100*100 / 314 = 31.8`
- Gain loss from cathode NFB: `~6 dB`
- With `Ck = 25 µF`: `f_c = 1 / (2π * 1500 * 25e-6) = 4.2 Hz`
- With `Ck = 1 µF`: `f_c = 106 Hz` — deliberate bass roll-off

Implemented in: planned (`lm19/amplifier/` -> `gain_vs_frequency()`).

Sources: [RDH] Ch.12, [Blencowe].

### 13) PSRR — Power Supply Rejection Ratio [V&W Ch.2, RDH]

For common-cathode SE stage with bypassed cathode, supply ripple `ΔVb` appears at anode:
tube acts as `ra` to ground, `Ra` goes to B+. Voltage divider:

- `PSRR_voltage = ra / (ra + Ra)` — fraction of ripple at output (lower = better)

In dB (rejection): `PSRR_dB = 20*log10((ra + Ra) / ra)` — higher dB = better rejection

With unbypassed cathode resistor Rk:
- Ripple rejection improves by factor `(1 + gm*Rk)` due to local NFB.
- Bypassed Rk provides no additional supply rejection beyond the ra/(ra+Ra) divider.

For cathode follower (exact): `PSRR_cf = 1 / (mu + 1 + ra/Rk)`

Approximation for `Rk >> ra/(mu+1)`: `PSRR_cf ≈ 1/(mu + 1)`

Numerical example:
- SE 12AX7: `ra = 62.5 kΩ`, `Ra = 100 kΩ`
- `PSRR = 62.5 / (62.5 + 100) = 0.385` → 8.3 dB rejection
- CF with `mu = 100`, `ra = 62.5 kΩ`, `Rk = 100 kΩ`:
  `PSRR_cf = 1 / (101 + 62.5/100) = 1/101.625 = 0.0098` → 40.2 dB rejection

Implemented in: planned (`lm19/amplifier/` -> `estimate_psrr()`).

Sources: [V&W] Ch.2, [RDH] Ch.5.

### 14) Phase splitter — Cathodyne and Long-Tail Pair [RDH Ch.12, Blencowe, Mullard]

**Cathodyne (split-load):**

- Gain per output: `A = mu*R / (ra + R*(mu+2))`, approx `mu / (mu+2)` ~ 0.95
- Differential output impedance (balanced): `Zout_diff ~ 2/gm` (both outputs)
- Common-mode: `Zout_k ~ 2/gm`, `Zout_a ~ Ra` (asymmetric)
- 50% internal feedback — extremely linear before clipping

**Long-Tail Pair (LTP):**

- Gain per output: `A/2` (where `A` is normal stage gain)
- Balance: improved by making `Ra2` 10-15% larger than `Ra1`
- Tail resistor: `R_tail = V_tail / (2*Ia_q)`
- CMRR: `CMRR = gm * R_tail` (higher tail R -> better balance)
- Output impedance: `Zout (balanced) = Ra || ra`, `Zout (one loaded) ~ Ra/2`

Design trade-off: tail voltage (25-30% of HT) vs output swing.

Implemented in: planned (`lm19/amplifier/` -> `compute_phase_splitter()`).

Sources: [RDH] Ch.12, [Blencowe cathodyne], [Blencowe LTP], [Mullard].

### 15) Ultralinear mode [Hafler & Keroes 1951, Mullard]

Screen grid connected to tap on output transformer primary.
Tap position: fraction of plate load impedance applied to screen.

Behavior vs tap position:
- 0% tap = pentode (screen to fixed B+)
- ~18-43% tap = ultralinear optimum (varies by tube type)
- 100% tap = triode connected

Effects at UL optimum:
- Internal impedance: sharp drop from pentode value, approaching triode
- Max undistorted output: slight drop from pentode, well above triode
- IM distortion: minimum at optimal tap (lower than both pentode and triode)
- Low-level distortion: decreases and holds level

Effective parameters (approximate interpolation):
- `ra_ul ~ ra_pentode / (1 + screen_coupling_factor * tap)`
- `gm_ul ~ gm_pentode` (nearly unchanged)
- `Gain_ul` between pentode and triode gain
- `Zout_ul` between pentode and triode Zout

Typical tap values: KT88/6550 ~ 43%, EL34 ~ 40%, 6L6 ~ 18.5% (Hafler original).

Implemented in: ✅ `lm19/amplifier/` -> `UltralinearModelWrapper`, `ul_screen_voltage()`. Model wrapper approach: dynamic Ug2(Ua) = Ug2_nom*(1-tap) + Ua*tap instead of interpolation.

Sources: [Hafler & Keroes 1951], [Mullard].

### 16) Efficiency and class of operation [RDH Ch.13, RCA RC-30]

**Efficiency:**

- `η = Pout / (Ub * Ia_q) * 100%`
- SE class A theoretical max: 25% (50% for transformer-coupled)
- PP class A theoretical max: 50%
- PP class AB: up to 60-70% depending on bias point

**PP class-A power threshold:** `P_A = Iq² × Ra_aa / 8` (Iq=A, Ra_aa=Ω, P_A=W). Output power at which one tube reaches cutoff and the amp transitions from class A to class AB. Used in optimizer `class_a_power_mode/value` filter. Source: Aiken «Last Word on Class A», sound-au.com.

**Class of operation (detection criteria):**

| Class | Condition | Characteristics |
|-------|-----------|----------------|
| A | `Ia > 0` throughout full cycle | Low distortion (mostly HD2), low efficiency |
| AB1 | `Ia = 0` for part of cycle, no grid current | Moderate distortion, moderate efficiency |
| AB2 | `Ia = 0` for part of cycle, grid current flows | Higher power, requires low-Z driver |
| B | `Ia = 0` for half cycle | High efficiency, significant crossover distortion |

Detection algorithm: from sweep data, check `ia_min_at_max_swing` and `ug1_max_at_swing`.

Implemented in: ✅ `lm19/amplifier/` -> class detection in `compute_distortion()` (amp_class field), η in `compute_distortion()` (eta_pct field), accurate Pa_avg via `compute_pa_avg()` (model-based numerical integration).

Sources: [RDH] Ch.13, [RCA RC-30].

### 17) Tube noise and SNR [V&W Ch.5, MIT noise paper, RCA RC-30]

**Shot noise** (anode current):

- `I_shot = sqrt(2 * e * Ia * BW)` where `e = 1.6e-19 C`

**Johnson noise** (resistors):

- `V_johnson = sqrt(4 * k * T * R * BW)` where `k = 1.38e-23 J/K`, `T` in Kelvin

**Equivalent input noise voltage** (simplified triode model):

- `V_eq_in ~ sqrt(4*k*T * (2.5/gm + Ra) * BW)` — factor 2.5/gm accounts for partition noise

**SNR estimation:**

- `V_signal_max = Av * half_swing`
- `V_noise = V_eq_in * Av` (referred to output)
- `SNR = 20*log10(V_signal_max / V_noise)`

**Dynamic range:** `DR = SNR + headroom_above_noise_floor`

Numerical example:
- 12AX7: `gm = 1.6 mA/V`, `Ra = 100 kΩ`, `BW = 20 kHz`, `Av = 60`
- `R_eq = 2.5/gm = 2.5/1.6e-3 = 1562.5 Ω` (equivalent noise resistance of triode)
- `V_eq_in ~ sqrt(4 * 1.38e-23 * 300 * (1562.5 + 100e3) * 20e3)`
- `V_eq_in ~ sqrt(4 * 1.38e-23 * 300 * 101563 * 20000) ~ 5.8 µV`
- With `swing = 1V`: `SNR = 20*log10(1 / 5.8e-6) ~ 105 dB`

Implemented in: planned (`lm19/amplifier/` -> `estimate_snr()`).

Sources: [V&W] Ch.5, [MIT noise paper], [RCA RC-30].

### 18) Bias stability and thermal considerations [RDH Ch.7, RCA RC-30]

**Cathode bias (self-biasing):**

- Stability factor: `S = 1 + gm * Rk` — reduces Ia drift by factor S
- `ΔIa_actual = ΔIa_intrinsic / S`
- Trade-off: Rk reduces gain by `(mu+1)*Rk / (ra + Ra + (mu+1)*Rk)` fraction

**Fixed bias:**

- No self-correction — full `ΔIa` from temperature drift
- Thermal runaway condition: `dPa/dIa * dIa/dT > thermal_dissipation_rate`
- Risk increases with: high Pa, high ambient T, poor ventilation
- Mitigation: individual cathode resistors (partial stabilization) or bias servo

**Thermal drift model:**

- Typical `ΔIa/Ia` per °C: 0.5-2% for power tubes (higher for small-signal)
- Operating temperature rise: 50-150°C above ambient (depends on Pa)
- Warm-up time to thermal equilibrium: 10-30 minutes

Implemented in: planned (`lm19/amplifier/` -> `compare_bias_methods()`, `estimate_thermal_drift()`).

Sources: [RDH] Ch.7, [RCA RC-30].

### 19) Multi-stage cascade and impedance loading [V&W Ch.2-3, RDH Ch.5]

Inter-stage gain with impedance loading:

- `Av_loaded = Av * Zin_next / (Zout + Zin_next)`

For N cascaded stages:

- `Total_gain = Π(Av_i * Zin_{i+1} / (Zout_i + Zin_{i+1}))` for `i = 1..N-1`
- Last stage: `Av_N` into final load

Input impedance of tube stage: `Zin = Rg || (1 / (j*w*C_in))`

where `C_in = C_gk + C_miller` (Miller capacitance amplifies `C_gp`).

At low frequencies (capacitive loading negligible): `Zin ~ Rg` (grid leak).

Numerical example (two-stage 12AX7 preamp):
- Stage 1: `Av = 60`, `Zout = Ra || ra = 38.5 kΩ`, `Ra = 100 kΩ`
- Stage 2: `Zin = 1 MΩ` (grid leak), `Av = 60`
- `Loading_loss = 1M / (38.5k + 1M) = 0.963` (negligible)
- `Total_gain = 60 * 0.963 * 60 = 3467` (71 dB)

Implemented in: planned (`lm19/amplifier/` -> `cascade_stages()`).

Sources: [V&W] Ch.2-3, [RDH] Ch.5.

### 20) Damping factor [RDH Ch.13]

`DF = Z_load / Z_out`

For SE triode without NFB: `DF = Z_speaker / (ra || Ra)` — typically 2-5.
For SE pentode: `DF < 1` (ra >> Ra, so `Zout ~ Ra`).
With NFB: `DF_fb = Z_load / (Zout / (1 + β*A))` — greatly improved.

Significance: DF controls transient response of loudspeaker. `DF > 8` considered adequate
for good speaker control. Tube amps with low DF produce characteristic "warm" bass
(underdamped speaker resonance).

Implemented in: ✅ `lm19/amplifier/` -> DF computed in `compute_stage_params()` (df field). DF = Ra_load / Zout.

Sources: [RDH] Ch.13.

## Traceability matrix (formula -> code -> tests -> sources)

| Topic | Formula or claim | Implemented in | Validated by | Sources |
|---|---|---|---|---|
| SE load line | `Ia = (Ub - Ua) / Ra` | `amplifier.py::ResistiveLoadLine.ia_at_ua()` | `test_amplifier.py` | [RDH] |
| 5-point HD | `B1/B2/B3` selected-ordinate equations | `amplifier.py::compute_distortion()` | `test_amplifier.py::TestAnalyticalHD`, `TestDistortionGuards` | [RDH], [Terman] |
| THD approximation | `THD ~= sqrt(HD2^2 + HD3^2)` | `amplifier.py::compute_distortion()` | `test_amplifier.py` | [RDH], [ResearchGate] |
| Output power | `Pout = I_pp * U_pp / 8` | `amplifier.py::compute_distortion()`, `pp_distortion()` | `test_amplifier.py`, `test_amplifier_smoke.py` | [RDH], [Blencowe] |
| SE stage params | `Av`, `Zout`, fallback `gm/ra` slopes | `amplifier.py::compute_stage_params()`, `_numerical_gm_ra()` | `test_amplifier_real_data.py`, `test_amplifier_smoke.py` | [V&W], [RDH] |
| Transformer AC/DC line | `Ia_dc=(Ub-Ua)/Ra_dc`, `Ia_ac=Ia_q-(Ua-Ua_q)/Ra_ac` | `amplifier.py::TransformerLoadLine` | `test_amplifier.py`, `test_amplifier_smoke.py` | [RDH] |
| Cathode follower | `Av_cf`, `Zout_cf` | `amplifier.py::compute_cf_stage_params()` | `test_amplifier.py`, `test_amplifier_smoke.py` | [TubeCad], [RDH] |
| Push-pull composite | odd symmetry suppresses even harmonics | `amplifier.py::composite_characteristic()`, `pp_distortion()` | `test_amplifier.py` | [RDH], [VTADiy] |
| PP reflected load | `Ra_per_tube = Ra_aa / 4` (Class AB/B) | `amplifier.py::PushPullLoadLine.ra_per_tube` | `test_amplifier.py`, `test_amplifier_spice_dataset.py` | [VacTube], [VTADiy] |
| IMD model | cubic fit and `IMD2/IMD3` metrics | `amplifier.py::compute_imd()` | `test_amplifier.py::TestIMD` | [Terman], [NextElectronics] |
| Auto-bias estimate | `Rk = \|Ug1\|/Ia * 1000` | `amplifier.py::optimize_bias()` | `test_amplifier.py` | [RDH] |
| Headroom limits | `max_swing = min(swing_neg, swing_pos)` | `amplifier.py::compute_headroom()` | `test_amplifier.py::TestHeadroom` | [RDH] |
| NFB closed-loop | `Afb = A/(1+β*A)`, `Zout_fb`, `THD_fb` | planned: `compute_nfb_effect()` | planned | [RDH], [VTADiy] |
| Miller bandwidth | `C_miller = C_gp*(1+Av)`, `f_h` | planned: `estimate_bandwidth()` | planned | [V&W], [Terman] |
| Cathode bypass | `Av_unbypassed`, `f_c = 1/(2π*Rk*Ck)` | planned: `gain_vs_frequency()` | planned | [RDH], [Blencowe] |
| PSRR | `PSRR = Ra/(ra+Ra)` | planned: `estimate_psrr()` | planned | [V&W], [RDH] |
| Phase splitter | Cathodyne `mu/(mu+2)`, LTP balance | planned: `compute_phase_splitter()` | planned | [RDH], [Blencowe], [Mullard] |
| Ultralinear mode | Screen tap interpolation | planned: `compute_ul_params()` | planned | [Hafler&Keroes], [Mullard] |
| Efficiency | `η = Pout/(Ub*Ia_q)` | planned: `compute_efficiency()` | planned | [RDH], [RCA RC-30] |
| Class detection | A/AB1/AB2/B from swing | planned: `detect_operation_class()` | planned | [RDH], [RCA RC-30] |
| Tube noise/SNR | Shot + Johnson noise | planned: `estimate_snr()` | planned | [V&W], [MIT], [RCA RC-30] |
| Bias stability | `S = 1+gm*Rk`, thermal | planned: `compare_bias_methods()` | planned | [RDH], [RCA RC-30] |
| Cascade loading | `Av*Zin/(Zout+Zin)` | planned: `cascade_stages()` | planned | [V&W], [RDH] |
| Damping factor | `DF = Z_load/Z_out` | planned: `compute_damping_factor()` | planned | [RDH] |

## Modern supplementary theory (2000+)

These references provide modern context (virtual-analog and data-driven modeling).
They are supplementary and do not replace classic primary derivations used for
closed-form equations in `lm19/amplifier/`.

### DAFx-23 quadric tube model [DAFx23]

- **Topic**: Efficient nonlinear fitting methods for vacuum tube characteristics.
- **Implemented in**: n/a (supplementary context, not directly implemented).
- **Validated by**: n/a.
- **Sources**: [DAFx23]
- **Notes**: Compares quadric surface fits vs traditional Koren-style polynomial models.
  Relevant as context for future model refinement, not for current closed-form analysis.

### Aiken common-cathode triode guide [Aiken]

- **Topic**: Practical modern explanation of load-line workflow for triode stages.
- **Implemented in**: n/a (supplementary pedagogical reference).
- **Validated by**: n/a.
- **Sources**: [Aiken]
- **Notes**: Covers DC load-line construction, bias point selection, and distortion
  estimation in accessible format. Useful for cross-checking methodology.

### Stanford WDF simulation [StanfordWDF]

- **Topic**: Real-time wave digital simulation of cascaded tube amplifier stages.
- **Implemented in**: n/a (future reference for real-time simulation).
- **Validated by**: n/a.
- **Sources**: [StanfordWDF]
- **Notes**: Demonstrates modified blockwise WDF approach for multi-stage tube amps.
  Potential direction for extending analysis beyond static operating points.

### IEEE LSTM tube model [IEEELSTM]

- **Topic**: Data-driven LSTM-based approximation of tube amplifier transfer functions.
- **Implemented in**: n/a (future reference for ML-based modeling).
- **Validated by**: n/a.
- **Sources**: [IEEELSTM]
- **Notes**: Shows neural-network approach to modeling nonlinear tube behavior.
  Complementary to physics-based Koren model used in `tube_sim.py`.

### DAFx-22 neural WDF tube modeling [DAFx22]

- **Topic**: Neural networks inside wave digital filters for multiport tube nonlinearities.
- **Implemented in**: n/a (supplementary modeling context).
- **Validated by**: n/a.
- **Sources**: [DAFx22]
- **Notes**: Trains networks directly in wave domain from Kirchhoff-domain datasets.
  Activation function selection significantly affects distortion characteristics.
  Complementary to DAFx23 quadric model.

### Rosenstark extended method for feedback analysis [Rosenstark2025]

- **Topic**: Unified gain/impedance/noise evaluation for feedback amplifiers (2025).
- **Implemented in**: n/a (methodological reference for NFB analysis).
- **Validated by**: n/a.
- **Sources**: [Rosenstark2025] — Carniti et al., Electronics 2025, 14, 1558.
- **Notes**: Open-access paper extending Rosenstark method. Eliminates need for
  Blackman's theorem; handles multiple feedback loops and parasitic elements.
  Directly applicable to global NFB stability and impedance analysis in tube amps.
  Relevant for planned `compute_nfb_effect()` implementation.

### SB-LAB — NFB vs zero-feedback practical analysis [SB-LAB NFB 2025]

- **Topic**: Practical perspective on negative feedback in tube amplifiers (2025).
- **Implemented in**: n/a (design philosophy reference).
- **Validated by**: n/a.
- **Sources**: [SB-LAB NFB 2025]
- **Notes**: Covers: NFB myths debunking, moderate feedback benefits (DF, noise, linearity),
  stability criticalities (phase margin, bandwidth), practical trade-offs with speaker types.
  Important context for damping factor and NFB design recommendations.

### SB-LAB — UL in SE designs critique [SB-LAB UL 2025]

- **Topic**: Why ultralinear is ineffective in single-ended topologies (2025).
- **Implemented in**: n/a (design caveat for UL mode).
- **Validated by**: n/a.
- **Sources**: [SB-LAB UL 2025]
- **Notes**: Argues UL requires push-pull even-harmonic cancellation to be effective.
  In SE, UL can increase odd-order distortion. Important limitation for
  planned `compute_ul_params()` — should warn user about SE+UL combination.

### Cascade Tubes — EL84 SE-UL optimization data [CascadeEL84 2024]

- **Topic**: Experimental EL84 ultralinear optimization (2024).
- **Implemented in**: n/a (empirical data reference for UL tap optimization).
- **Validated by**: n/a.
- **Sources**: [CascadeEL84 2024]
- **Notes**: Systematic bias/voltage sweep for EL84 in UL mode. Finding:
  beam power tubes vs screen pentodes behave differently in UL.
  EL84 max 275V plate. Edcor GXSE10-5K with 40% tap.

### Analog Ethos — Screen voltage and ultralinear mode [AnalogEthos UL 2024]

- **Topic**: Modern practical explanation of triode/pentode/UL modes.
- **Implemented in**: n/a (pedagogical supplement).
- **Validated by**: n/a.
- **Sources**: [AnalogEthos UL 2024]
- **Notes**: Clear explanation of screen tap mechanism. Distortion curves of
  triode and pentode modes bend in opposite directions; UL finds cancellation point.
  Part of comprehensive "Tube Amplifiers Explained" series.
