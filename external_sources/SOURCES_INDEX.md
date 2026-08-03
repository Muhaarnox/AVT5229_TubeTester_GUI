# Sources Index

Registry of external sources, datasets and theory materials.
Shared across projects: an entry describes the SOURCE and the copy kept in
this folder — never a path into the consuming project.

## Entry format

```md
### <Source name>
- url: <https://...>            # n/a + reason when it cannot be downloaded
- type: data | theory | documentation | tool
- local_copy: <path>            # relative to this file when inside the folder
- status: active | updated | downloaded | not_downloaded | archived
- added_on: YYYY-MM-DD          # plus updated_on when the source is replaced
- role: <why this source is kept>    # optional, when the note does not say it
- note: <what is inside, extracted data, verification status>
```

## Sources

### TDSL (Duncan Tube Data Sheet Locator)
- url: <https://tdsl.duncanamps.com/show.php?des=...>
- type: data
- local_copy: n/a
- status: active
- added_on: 2026-02-26
- note: Primary source for lamp limits and reference values.

### Frank Pöcnet datasheets (fallback)
- url: <https://frank.pocnet.net/>
- type: documentation
- local_copy: n/a
- status: active
- added_on: 2026-02-26
- note: Fallback datasheets when TDSL entries are unavailable.

### JJ Electronic datasheet (ECC99 fallback)
- url: <https://www.jj-electronic.com/images/stories/product/preamplifying_tubes/pdf/ecc99.pdf>
- type: documentation
- local_copy: `lamp_datasheets/ECC99_jj.pdf`
- status: downloaded
- added_on: 2026-02-26
- note: Fallback for ECC99 limits/ratings.

### Norman Koren (model theory and Tuparam data)
- url: <https://www.normankoren.com/Audio/Tube_params.html>
- type: theory
- local_copy: `spice_models/koren/`
- status: active
- added_on: 2026-02-26
- note: Core equations and fitting reference for SPICE export/tests.

### Next-Tube datasets
- url: <https://next-tube.com/data.php>
- type: data
- local_copy: `data/next_tube/`
- status: active
- added_on: 2026-02-26
- note: 18 tubes, empirical measurements converted to unified JSON.

### Next-Tube SPICE libraries
- url: <https://next-tube.com/zip/>
- type: data
- role: reference SPICE models for validation
- local_copy: `spice_models/next_tube/`
- status: downloaded
- added_on: 2026-04-11
- note: |
    5 SPICE model libraries by A. Karpov (next-tube.com):
    PSlib.zip (PSpice), MC7.zip (Micro-Cap), Model.zip (SEcalcS),
    Tube_ORCAD.zip (extended ORCAD), Tube_Msim10.zip (MSIM-10, largest).
    Koren-equation based models for same 18 tubes as the datasets.

### Duncan Munro — Leach Model SPICE Libraries
- url: <https://duncanamps.com/technical/spice.zip>
- type: data
- role: reference SPICE models — alternative Leach 3/2 power law
- local_copy: `spice_models/duncan/`
- status: downloaded
- added_on: 2026-04-11
- note: |
    18 .INC files (12AX7A, 6L6GC, KT88, etc.) using Leach model (3/2 power law).
    Different equation structure from Koren — academic reference only.

### LTspice Community / EXCEM — Tube Libraries
- url: LTwiki mirror (originally from Intusoft/EXCEM)
- type: data
- role: reference — alternative TRIO1/PENT1/HEAT1 model primitives
- local_copy: `spice_models/ltspice_community/`
- status: downloaded
- added_on: 2026-04-11
- note: |
    Tube.lib, Tube97.lib, tube1.lib — Koren mirror from LTwiki.
    VACUUM.LIB, vacuumnl.lib — EXCEM/Intusoft models (19 KB, different equation structure).
    Academic reference for alternative modeling approaches.

### Norman Koren — Tuparam raw data
- url: <https://www.normankoren.com/Audio/Tuparam.zip>
- type: data
- local_copy: `data/tuparam/`
- status: downloaded
- added_on: 2026-04-11
- note: |
    Matlab .m files with measured tube data (12AX7A, 7025, 6550).
    Gold standard — data used by Koren himself for model fitting.
    6–11 points per tube, carefully selected across operating range.

### loadline_plotter datasets
- url: <https://github.com/andmarti1424/loadline_plotter>
- type: data
- local_copy: `data/loadline_plotter/`
- status: active
- added_on: 2026-02-26
- note: Digitized datasheet curves for fitting/regression checks.

### pypsucurvetrace / curvetracedata
- url: <https://github.com/mbrennwa/curvetracedata>
- type: data
- local_copy: n/a (upstream .dat files not archived here; only datasets derived
  from them are kept by the consuming project)
- status: active
- added_on: 2026-02-26
- note: High-density measured curves for test coverage and import validation.

### µTracer User Manual (SRK measurement methodology)
- url: <https://www.dos4ever.com/uTracer3/uTracer3_user_man.pdf>
- type: theory
- local_copy: `theory/utracer_user_manual.pdf`
- status: downloaded
- added_on: 2026-02-28
- note: |
    Section 5.1–5.2: Quick Test methodology for gm, Rp, mu.
    Central difference: measure at Vo−δV and Vo+δV, default δV = 10% of bias voltage.
    Triode: 5 measurements, pentode: 7 measurements.
    Recommends fixed ADC range and high averaging (16×) for high-Rp pentodes.

### Gamma Electronics — Transconductance Tube Tester Test Standard
- url: <https://www.gammaelectronics.xyz/ax_0205_transcond.html>
- type: theory
- local_copy: `theory/gamma_electronics_transconductance_standard.html` (text only, figures not stored)
- status: downloaded
- added_on: 2026-02-28
- note: Accuracy standards for gm measurement; typical 5–10% accuracy for vintage testers.

### Radiotron Designer's Handbook, 4th ed. (Langford-Smith, 1953)
- url: <https://worldradiohistory.com/ENCYCLOPEDIAS/Radiotron-Designers-Handbook-4th-Edition-1953.pdf>
- type: theory
- local_copy: `theory/radiotron_designers_handbook_4ed_1953.pdf`
- status: active
- added_on: 2026-03-05
- note: |
    Primary reference for amplifier theory. Ch. 7: 5-point selected-ordinate method for
    harmonic distortion (B1/B2/B3 derivation). Ch. 5: small-signal gain, Zout.
    Ch. 12: feedback & cathode follower. Ch. 13: push-pull cancellation.
    Full derivation reproduced in AMPLIFIER_THEORY_APPENDIX.md.

### ResearchGate — Methods for Computing Harmonic Distortion (paper)
- url: <https://www.researchgate.net/publication/242307545_METHODS_FOR_COMPUTING_HARMONIC_DISTORTION_IN_LOW_FREQUENCY_POWER_AMPLIFIER>
- type: theory
- local_copy: n/a
- status: active
- added_on: 2026-03-05
- note: |
    Academic paper summarizing selected-ordinate methods (3-, 5-, 7-, 11-point).
    Confirms B1/B2/B3 formulas from RDH. Used as cross-reference during formula audit.

### Terman — Electronic and Radio Engineering, 4th ed. (1955)
- url: n/a (book: Terman F.E., McGraw-Hill, 1955)
- type: theory
- local_copy: `theory/terman_electronic_and_radio_engineering_4ed.pdf`
- status: downloaded
- added_on: 2026-03-05
- note: |
    §13.5: Selected-ordinate method for harmonic analysis — independent derivation
    confirming RDH formulas. §13.7: Intermodulation distortion via Taylor series.
    §7: Tube models (gm, ra, μ). A 1072-page scan of the 4th edition is now archived
    here (Digital Library of India copy), so the section-level citations above can be
    verified against the text instead of from memory.

### Valley & Wallman — Vacuum Tube Amplifiers (MIT Rad. Lab. Vol. 18, 1948)
- url: <https://archive.org/details/MITRadiationLaboratorySeries>
- type: theory
- local_copy: `theory/valley_wallman_vacuum_tube_amplifiers_1948.pdf`
- status: downloaded
- added_on: 2026-03-05
- note: |
    Ch. 1-2: Rigorous small-signal vacuum tube models. Thevenin/Norton equivalents.
    Gain = μ·Ra/(ra+Ra), Zout = ra‖Ra. Foundation for all SE stage parameter formulas.
    766-page scan archived here (Digital Library of India copy); the details page of
    the series collection itself offers no direct download.

### Blencowe — Designing Tube Preamps for Guitar and Bass, 2nd ed. (2012)
- url: n/a (book: Merlin Blencowe, 2012)
- type: theory
- local_copy: n/a
- status: updated
- added_on: 2026-03-15
- note: |
    Practical supplement for tube-stage interpretation; used as secondary reference,
    not as primary source of equations.
    2026-03-15: full book not downloaded; related author materials saved separately.

### Terman Electronic and Radio Engineering metadata pages
- url: <https://books.google.com/books?id=LAgjAAAAMAAJ> ; <https://openlibrary.org/works/OL7456725W/Electronic_and_radio_engineering>
- type: documentation
- local_copy: n/a (catalogue shells; the book itself is stored — see the Terman 4th ed. entry)
- status: not_downloaded
- added_on: 2026-03-15
- note: Bibliographic confirmation only (edition/year/page count). No direct chapter text access.

### Terman Radio Engineer's Handbook (1943, Archive copy)
- url: <https://archive.org/download/in.ernet.dli.2015.126496/2015.126496.Radio-Engineers-Handbook.pdf>
- type: theory
- local_copy: `theory/terman_radio_engineers_handbook.pdf`
- status: active
- added_on: 2026-03-15
- note: Downloaded successfully as alternative Terman source; not a direct replacement for 1955 E&RE section citations.

### Valve Wizard — Cathode Follower (Merlin Blencowe)
- url: <https://www.valvewizard.co.uk/accf.html>
- type: theory
- local_copy: `theory/blencowe_valvewizard_cathode_follower.mht`
- status: downloaded
- added_on: 2026-03-15
- note: Author-maintained explanatory article; used as supplemental intuition, not primary equation source.

### Triodes at Low Voltages (Merlin Blencowe PDF)
- url: <https://valvewizard.co.uk/Triodes_at_low_voltages_Blencowe.pdf>
- type: theory
- local_copy: `theory/blencowe_triodes_low_voltage.pdf`
- status: active
- added_on: 2026-03-15
- note: Downloaded successfully from Valve Wizard site; supplemental reference.

### Next Electronics — Harmonic Distortion in Amplifiers Tutorial
- url: <https://next.gr/tutorials/audio-electronics/harmonic-distortion-in-amplifiers-tutorial>
- type: theory
- local_copy: `theory/next_electronics_harmonic_distortion.html`
- status: updated
- added_on: 2026-03-05
- updated_on: 2026-04-03
- note: |
    THD = sqrt(V2²+V3²+...)/V1. IMD from Taylor series: a2 generates 2nd harmonic,
    a3 generates 3rd harmonic. Push-pull cancels even harmonics by symmetry.
    The stored copy is the page itself (16 sections: THD, IMD, Taylor series,
    measurement techniques, linearization, amplifier topologies); figures are
    referenced, not embedded, so diagrams need the live site.

### TubeCad — Cathode Follower Output Stage & Simple Tube Math
- url: <https://tubecad.com/2005/June/blog0049.htm>
- type: theory
- local_copy: `theory/tubecad_cathode_follower.mht`
- status: downloaded
- added_on: 2026-03-05
- note: |
    CF gain = mu*Rk/(ra+(mu+1)*Rk), Zout ≈ ra/(mu+1) ≈ 1/gm.
    Verified against amplifier.py CF formulas — both correct.

### VTADiy — Push-Pull Loadline in Class AB
- url: <https://www.vtadiy.com/book/chapter-4-integrated-push-pull-vacuum-tube-amplifier/4-1-output-stage-or-power-stage/4-1-6-push-pull-loadline-in-class-ab/>
- type: theory
- local_copy: `theory/vtadiy_push_pull_loadline_class_ab.mht`
- status: downloaded
- added_on: 2026-03-05
- note: |
    Class A: each tube sees Ra_aa/2. Class AB/B: each tube sees Ra_aa/4.
    PP output power: P = V_aa_pk² / (2*Ra_aa). Used to verify PP formulas in amplifier.py.

### VTADiy — Output Stage (Power Stage) of a Vacuum Tube Amplifier §4.1
- url: <https://www.vtadiy.com/book/chapter-4-integrated-push-pull-vacuum-tube-amplifier/4-1-output-stage-or-power-stage/>
- type: theory
- local_copy: `theory/vtadiy_push_pull_output_stage.mht`
- status: downloaded
- added_on: 2026-08-02
- note: |
    Chapter-level page of the already-indexed §4.1.6 «Push-Pull Loadline in Class AB»;
    saved 2026-07-01 during the push-pull working-line work. Covers the output stage
    as a whole: reflected primary impedance, class A vs class AB operation, and why
    each tube sees Ra_aa/2 while both conduct and Ra_aa/4 once the partner cuts off —
    the impedance kink reproduced by `pp_joint_trajectory` / `pp_working_line_ia`.
    Index entry added retroactively 2026-08-02 (file had been downloaded unregistered).

### Vacuum-tube.eu — Push-Pull Transformer Impedance
- url: <https://www.vacuum-tube.eu/www.voltsecond/Push-Pull_Z/Push-Pull_Pri_Z.html>
- type: theory
- local_copy: `theory/vactube_push_pull_impedance.mht`
- status: downloaded
- added_on: 2026-03-05
- note: |
    Impedance scales with N². Half-primary → (N/2)² = N²/4, so Ra_per_tube = Ra_aa/4.
    Verified PushPullLoadLine.ra_per_tube formula.

### DAFx-23 — A Quadric Surface Model of Vacuum Tubes for Virtual Analog Applications
- url: <https://dafx.de/paper-archive/2023/DAFx23_paper_15.pdf>
- type: theory
- local_copy: `theory/modern_dafx23_quadric_vacuum_tube_model.pdf`
- status: active
- added_on: 2026-03-15
- note: Modern digital tube-modeling reference (virtual analog). Supplemental context, not a primary source for classic closed-form SE/PP equations.

### Aiken Amps — Designing Common-Cathode Triode Amplifiers
- url: <https://www.aikenamps.com/index.php/designing-common-cathode-triode-amplifiers>
- type: theory
- local_copy: `theory/modern_aiken_common_cathode_triode_amplifiers.mht`
- status: downloaded
- added_on: 2026-03-15
- note: Practical explanatory material for common-cathode/load-line design; used as modern secondary reference.

### Aiken Amps — The Last Word on Class A
- url: <https://www.aikenamps.com/index.php/the-last-word-on-class-a>
- type: theory
- local_copy: `theory/aiken_last_word_class_a.mht`
- status: downloaded
- added_on: 2026-08-02
- note: |
    Definition used for the optimizer's class-A constraint: class A = plate current in
    every output device flows for the full 360 deg of the cycle at full unclipped output.
    Key claims: PP class A gives exactly 2x the power of SE
    class A under the same plate voltage / bias / effective load; the bias METHOD
    (fixed vs cathode) does not determine the class; dissipation is maximal at idle in
    class A and does not rise with signal (unlike AB/B). Basis for `P_A = Iq^2*Ra_aa/8`.
    Index entry added retroactively 2026-08-02 (downloaded 2026-04-26, unregistered).

### Rod Elliott (ESP) — Class-A Amplifiers Explained
- url: <https://sound-au.com/class-a.htm>
- type: theory
- local_copy: `theory/sound_au_class_a_amplifiers.html` (text only, figures not stored)
- status: active
- added_on: 2026-08-02
- note: |
    Second, independent statement of the class-A boundary used to cross-check the
    Aiken definition: the amplifier stays in class A only while the signal current
    stays below the idle current, so the class-A power ceiling is fixed by Iq and the
    load, and an amplifier leaves class A long before clipping. Efficiency ceiling
    (25% SE / 50% PP theoretical) and heat-at-idle consequences documented there too.
    Index entry added retroactively 2026-08-02 (downloaded 2026-04-26, unregistered).

### Stanford CCRMA — Real-Time Wave Digital Simulation of Cascaded Vacuum Tube Amplifiers
- url: <https://ccrma.stanford.edu/~jingjiez/publications/Real-Time%20Wave%20Digital%20Simulation%20of%20Cascaded%20Vacuum%20Tube%20Amplifiers%20using%20Modified%20Blockwise%20Method.pdf>
- type: theory
- local_copy: `theory/modern_stanford_wdf_cascaded_tube_amplifiers.pdf`
- status: active
- added_on: 2026-03-15
- note: Downloaded successfully on retry; modern wave-digital simulation reference.

### IEEE — A Vacuum-Tube Guitar Amplifier Model Using Long/Short-Term Memory Networks
- url: <https://ieeexplore.ieee.org/document/8479039/>
- type: theory
- local_copy: n/a (landing page only — read online)
- status: not_downloaded
- added_on: 2026-03-15
- note: IEEE landing/metadata page saved locally; full paper PDF access is paywalled in current environment.

### Hafler & Keroes — An Ultra-Linear Amplifier (Audio Engineering, 1951)
- url: <https://rubli.net/classic_amps/files/articles/ulamp/ulamp.htm>
- type: theory
- local_copy: `theory/hafler_keroes_ultralinear_amplifier.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    Original 1951 paper defining ultralinear output stage. Screen tap at ~18.5% of plate
    load impedance gives optimal blend of triode (low Zout) and pentode (high power).
    Key finding: IM distortion minimum at narrow band of screen/plate impedance ratio.
    Patent #2710312 (1955). Foundation for Dynaco Mark II and most UL amplifiers.

### RCA Receiving Tube Manual, RC-30 (1975)
- url: <https://archive.org/details/RCA_RC-30_1975>
- type: theory
- local_copy: `theory/rca_receiving_tube_manual_rc30.pdf`
- status: active
- added_on: 2026-03-20
- note: |
    ~760 pages. Comprehensive tube data, application notes, circuit examples.
    Relevant sections: resistance-coupled amplifier data (gain/bandwidth tables),
    tube noise characteristics, maximum ratings and derating curves,
    class A/AB/B operating conditions, efficiency calculations.

### Mullard — Circuits for Audio Amplifiers (1963)
- url: <https://www.worldradiohistory.com/BOOKSHELF-ARH/Circuits-For-Audio-Amplifiers-Mullard-1963%20.pdf>
- type: theory
- local_copy: `theory/mullard_circuits_for_audio_amplifiers_1963.pdf`
- status: active
- added_on: 2026-03-20
- note: |
    Iconic Mullard application book. Contains 5-10, 5-20, 3-3 amplifier designs.
    Phase splitter circuits (cathodyne, long-tail pair), ultralinear output stages,
    NFB design with stability analysis, practical component selection.
    Primary reference for complete amplifier design workflow.

### Valve Wizard — Cathodyne Phase Inverter (Merlin Blencowe)
- url: <https://valvewizard.co.uk/cathodyne.html>
- type: theory
- local_copy: `theory/blencowe_valvewizard_cathodyne.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    Cathodyne gain: A = mu*R / (ra + R*(mu+2)), approx mu/(mu+2) ~ 0.95.
    Differential Zout ~ 2/gm (balanced), common-mode Zout_a ~ Ra, Zout_k ~ 2/gm.
    Overdrive behavior, grid stopper trick, DC coupling considerations.

### Valve Wizard — DC Coupled Long-Tailed Pair (Merlin Blencowe)
- url: <https://www.valvewizard.co.uk/dcltp.html>
- type: theory
- local_copy: `theory/blencowe_valvewizard_longtail_pair.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    LTP gain = normal stage gain / 2 per output. Balance improved by 10-15% Ra2 > Ra1.
    Tail resistor approximates constant current source. Zout (equal load) = Ra || ra.
    Design procedure: choose tail voltage (25-30% HT), bias, Ra, Rk(tail).

### VTADiy — Global Negative Feedback in Vacuum Tube Amplifier
- url: <https://www.vtadiy.com/book/chapter-4-integrated-push-pull-vacuum-tube-amplifier/4-4-global-negative-feedback/>
- type: theory
- local_copy: n/a (formulas are images — read online)
- status: not_downloaded
- added_on: 2026-03-20
- note: |
    Afb = A / (1 + β*A). THD reduction factor = 1 / (1 + β*A).
    Zout reduction same factor. Bandwidth extension.
    Stability: phase margin, step network compensation (Csn || R1).
    Loop gain A*β < 1 at 180° phase shift frequency.

### MIT — Johnson Noise and Shot Noise (D.V. Perepelitsa, 2006)
- url: <https://web.mit.edu/dvp/Public/noise-paper.pdf>
- type: theory
- local_copy: `theory/mit_noise_johnson_shot.pdf`
- status: active
- added_on: 2026-03-20
- note: |
    Shot noise: I_rms = sqrt(2*e*I_dc*BW). Johnson noise: V^2 = 4*R*k*T*BW.
    Experimental verification of Boltzmann constant and electron charge.
    Used as theoretical foundation for tube amplifier noise floor estimation.

### Carniti et al. — Extending the Rosenstark Method (Electronics, 2025) [Rosenstark2025]
- url: <https://www.mdpi.com/2079-9292/14/8/1558>
- type: theory
- local_copy: `theory/modern_rosenstark_feedback_2025.pdf`
- status: active
- added_on: 2026-03-20
- note: |
    Open-access 2025 paper. Extends Rosenstark method to unify gain, frequency response,
    input/output impedance, and noise evaluation in feedback amplifiers. Eliminates need
    for Blackman's theorem or circuit simplifications. Handles multiple feedback loops
    and parasitic elements. Directly applicable to global NFB analysis in tube amps.

### DAFx 2022 — Neural Network WDF Tube Modeling [DAFx22]
- url: <https://dafx.de/paper-archive/2022/papers/DAFx20in22_paper_13.pdf>
- type: theory
- local_copy: `theory/modern_dafx22_neural_wdf_tube.pdf`
- status: active
- added_on: 2026-03-20
- note: |
    Neural networks inside wave digital filters for multiport tube nonlinearities.
    Activation function selection affects distortion characteristics.
    Complementary to DAFx23 quadric model and Stanford WDF approaches.

### Analog Ethos — Screen Voltage and Ultralinear Mode (2024)
- url: <https://www.analogethos.com/post/screen-ultralinear>
- type: theory
- local_copy: `theory/modern_analogethos_ultralinear.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    Modern practical explanation of triode/pentode/UL modes. Screen tap ~40-45%.
    Part of comprehensive "Tube Amplifiers Explained" series (Parts 1-10+).

### SB-LAB — Negative Feedback and Zero Feedback in Tube Amplifiers (Bianchini, 2025)
- url: <https://www.sb-lab.eu/en/negative-feedback-and-zero-feedback-in-tube-amplifiers-my-position/>
- type: theory
- local_copy: `theory/modern_sblab_nfb_position_2025.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    Experienced practitioner's position on NFB debate. Covers: benefits of moderate NFB
    (damping factor, noise, linearity), zero-NFB myths vs reality, stability criticalities
    (phase margin, bandwidth, compensation), practical trade-offs with speaker types.

### SB-LAB — Why Ultralinear in SE Designs Is a Mistake (Bianchini, 2025)
- url: <https://www.sb-lab.eu/en/the-great-myth-debunked-why-ultralinear-in-single-ended-designs-is-a-mistake/>
- type: theory
- local_copy: `theory/modern_sblab_ul_se_myth_2025.html`
- status: active
- added_on: 2026-03-20
- note: |
    Critical analysis of UL in SE topology. Argues UL is effective only in PP designs
    due to cancellation of even harmonics; in SE, UL can increase odd-order distortion
    without the benefit of push-pull cancellation. Important design caveat.

### Cascade Tubes — EL84 SE-UL Optimization (2024)
- url: <https://www.cascadetubes.com/2024/04/26/more-thoughts-on-the-el84-se-ul-optimization/>
- type: theory
- local_copy: `theory/modern_cascadetubes_el84_ul_2024.mht`
- status: downloaded
- added_on: 2026-03-20
- note: |
    Experimental EL84 UL optimization: optimal plate voltage, tap %, bias.
    Finding: beam power vs screen pentode structural differences significantly
    affect UL behavior. EL84 should not exceed 275V plate. Uses Edcor GXSE10-5K
    with 40% tap. Part of series covering 6V6, 6L6, EL84.

### Dempwolf & Zölzer — A Physically-Motivated Triode Model (DAFx-11, 2011)
- url: <https://dafx.de/paper-archive/2011/Papers/76_e.pdf>
- type: theory
- local_copy: `theory/dempwolf_dafx11_triode_model.pdf`
- status: active
- added_on: 2026-03-24
- note: |
    Original Dempwolf triode model paper. Cathode current Ik = G·(ln(1+exp(C·(Va/µ+Vg)))/C)^γ,
    grid current Ig = Gg·(ln(1+exp(Cg·Vg))/Cg)^ξ + Ig0, plate current Ia = Ik − Ig.
    8 parameters (G, µ, γ, C, Gg, ξ, Cg, Ig0). Published fits for 12AX7 (3 specimens).
    Key advantage over Koren: smooth grid current, current conservation by construction.
    Known limitation: poor Region A behavior at low Va (addressed by a Kvb_t extension in the Dempwolf v2 variant).

### Dunkel et al. — Fender Bassman 5F6-A WDF Case Study (DAFx-16, 2016)
- url: <https://www.dafx.de/paper-archive/2016/dafxpapers/37-DAFx-16_paper_53-PN.pdf>
- type: theory
- local_copy: `theory/dunkel_dafx16_bassman_wdf.pdf`
- status: active
- added_on: 2026-03-24
- note: |
    Wave digital filter implementation of Fender Bassman using Dempwolf triode model.
    Confirms 12AX7 parameters from original Dempwolf paper. Reference C++ implementation
    available in RT-WDF library (github.com/RT-WDF/rt-wdf_lib).

### Reefman — Spice Models for Vacuum Tubes Using the uTracer (Theory.pdf, 2016)
- url: <https://www.dos4ever.com/uTracer3/Theory.pdf>
- type: theory
- local_copy: `theory/reefman_utracer_spice_theory.pdf`
- status: active
- added_on: 2026-03-24
- note: |
    50-page paper by Derk Reefman (2016-01-24). Defines Derk (true pentode) and DerkE (beam tetrode)
    models with physically-motivated current splitting, constant space charge principle, secondary emission
    (Psec), variable-mu pentodes, and hexode/heptode models. Key equations: Ip,Koren with √(Kvb+Vg2²),
    Derk Ig2 = Ip/Kg2·(1+αs/(1+βVa)), DerkE Ig2 = Ip/Kg2·(1+αs·exp(-(βVa)^1.5)).
    Companion software: ExtractModel, utMax. Several ideas adopted into Dempwolf Extended v2.

### Norman Koren — Improved Vacuum Tube Models for SPICE Simulations (1996)
- url: <https://www.normankoren.com/Audio/Tubemodspice_article.html>
- type: theory
- local_copy: `theory/koren_improved_tube_models_article.mht`
- status: downloaded
- added_on: 2026-02-26
- updated_on: 2026-03-24
- note: |
    Full article with Koren model equations. Triode: E1=(Va/Kp)·ln(1+exp(Kp·(1/µ+Vg/√(Kvb+Va²)))),
    Ia=2·E1^Ex/Kg1. Pentode: E1 with Vg2, Ia·arctan(Va/Kvb). Known issues (per Reefman):
    Kg1/Kp correlation, Ig2 independent of Va, pentode-as-triode inconsistency.
    Companion page with parameter tables saved as koren_improved_tube_models.html.

### uTracer page 14 — ExtractModel and SPICE models
- url: <https://www.dos4ever.com/uTracer3/uTracer3_pag14.html>
- type: documentation
- local_copy: n/a (needs the live site — read online)
- status: not_downloaded
- added_on: 2026-03-24
- note: Ronald Dekker's uTracer page with SPICE subcircuits for Koren/Derk/DerkE models. Equations in Theory.pdf.

### TubeLib.inc — ExtractModel SPICE library
- url: <https://www.dos4ever.com/uTracer3/TubeLib.inc>
- type: data
    the planned `data/extractmodel_params.json` extract was never created)
- local_copy: `data/TubeLib.inc`
- status: downloaded
- added_on: 2026-03-24
- note: |
    213 SPICE subcircuits (164 unique tubes) fitted by ExtractModel (Reefman, Jan 2016).
    Subcircuit types (9): TriodeK (98), BTetrodeD (23), BTetrodeDE (18), DiodeK (31),
    PenthodeD (8), PenthodeDE (4), PenthodeVD (3), PenthodeVDE (4), HepthodeD (10).
    Contains Derk/DerkE parameters for EL84, EL34, 6L6GC, KT88, EF86, etc.

### White Cottage — Valve Modelling
- url: <https://whitecottage.org.uk/guitar-and-audio/valve-modelling/>
- type: theory
- local_copy: n/a (equations are images — read online)
- status: not_downloaded
- added_on: 2026-03-24
- note: |
    Detailed presentation of Derk/DerkE equations with extended knee functions:
    g(Va) = exp(-(β(1-αVg1)Va)^γ), h(Va) = exp(-(ρ(1-τVg1)Va)^θ).
    Adds Vg1-dependent knee parameters (α, γ, ρ, τ, θ) not in original Reefman paper.

### Rod Elliott (ESP) — Valves: Distortion and Intermodulation
- url: <https://sound-au.com/valves/thd-imd.html>
- type: theory
- local_copy: `theory/sound_au_valves_thd_imd.mht`
- status: downloaded
- added_on: 2026-04-03
- note: |
    Comprehensive overview of THD and IMD in tube amplifiers. Key findings:
    2nd harmonic never exists in isolation; asymmetrical clipping (SE) produces ~2.5 dB
    more IMD than symmetrical (PP). Practical FFT-based verification methodology using
    two-tone (1 kHz + 1.1 kHz) test. Challenges "even harmonics sound nice" myth.

### DIY Audio Projects — Of Loadlines, Power Output and Distortion
- url: <http://diyaudioprojects.com/mirror/members.aol.com/sbench102/po-dis.html>
- type: theory
- local_copy: `theory/diyaudioprojects_loadlines_power_distortion.mht`
- status: downloaded
- added_on: 2026-04-03
- note: |
    Classic graphical method for deriving distortion from load lines. 5-point
    selected-ordinate method: HD2 = 75*(Ia+Ie-2*Ic)/(Ia+Ib-Id-Ie),
    HD3 = 50*(Ia-2*Ib+2*Id-Ie)/(Ia+Ib-Id-Ie). Power: Po = (Ve-Va)^2/(8*Rload).
    Transfer characteristic construction from tube curves. Confirms RDH formulas.

### Kenny Peng — Chebyshev Polynomials and Harmonics (2022)
- url: <https://kennypeng.com/2022/06/18/chebyshev_harmonics.html>
- type: theory
- local_copy: `theory/kennypeng_chebyshev_harmonics.mht`
- status: downloaded
- added_on: 2026-04-03
- note: |
    Mathematical relationship between Chebyshev polynomials and harmonic distortion.
    Key identity: T_n(cos x) = cos(nx). Recurrence: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x).
    T2 = 2x²-1, T3 = 4x³-3x, T4 = 8x⁴-8x²+1. Waveshaper synthesis via weighted
    Chebyshev sum. Limitation: precise only with cosine inputs in [-1,1].

### Power Electronics News — QSPICE FFT Analysis (Part 10)
- url: <https://www.powerelectronicsnews.com/qspice-fft-analysis-part-10/>
- type: theory
- local_copy: n/a
- status: failed
- added_on: 2026-04-03
- note: |
    How SPICE performs .FOUR analysis — DFT on simulated waveform.
    Download failed: connection timeout on repeated attempts (2026-04-03).

### Lynn Olson — The Sound of the Machine: Hidden Harmonics behind THD (Milbert)
- url: <https://milbert.com/articles/Lynn_Olsen_THD>
- type: theory
- local_copy: `theory/milbert_lynn_olson_thd.mht`
- status: downloaded
- added_on: 2026-04-03
- note: |
    Practical analysis of THD measurement limitations for tube amplifiers.
    Crowhurst analysis: feedback creates harmonics up to 81st order from pure 2nd-harmonic
    source. Triodes have lowest distortion with rapid upper-harmonic fall-off.
    Music genre affects distortion perception (sparse vs dense spectra).
    Transformer coupling gives smoothest harmonic fall-off.

### Cerdeira et al. — Integral Function Method for Harmonic Distortion (2004)
- url: n/a (paper: Solid-State Electronics 48, 2004)
- type: theory
- local_copy: n/a
- status: not_downloaded
- added_on: 2026-04-03
- note: |
    Paywalled academic paper. Integral Function Method (IFM) for determination of
    nonlinear harmonic distortion without requiring explicit Taylor series coefficients.
    Relevant as alternative analytical approach to selected-ordinate methods.

### AES E-Library — Distortion Analysis Using SPICE
- url: n/a (AES E-Library, paywalled)
- type: theory
- local_copy: n/a
- status: not_downloaded
- added_on: 2026-04-03
- note: |
    Paywalled AES paper on SPICE FFT internals for distortion analysis.
    Documents how SPICE .FOUR directive performs DFT on simulated time-domain waveforms
    to extract harmonic amplitudes and compute THD.

### Analog Devices MT-053 — Op Amp Distortion: HD, THD, THD+N, IMD, SFDR
- url: n/a (Analog Devices tutorial MT-053)
- type: theory
- local_copy: n/a
- status: not_downloaded
- added_on: 2026-04-03
- note: |
    Analog Devices tutorial on distortion metrics. Taylor series derivation of
    individual harmonic distortion components: HD2 = (1/2)*a2*A/a1,
    HD3 = (1/4)*a3*A²/a1. Defines THD, THD+N, IMD, SFDR with measurement context.
    Available from Analog Devices website but not downloaded due to access restrictions.

## Tube SPICE Model Sources (reference coefficient sets)

### Koren Original Library (Tube.lib, Tube97.lib, tube1.lib)
- url: <https://www.normankoren.com/Audio/Tubemodspice_article.html> (author page)
- type: data
- local_copy: `spice_models/koren/`
- status: downloaded
- added_on: 2026-04-06
- note: |
    Norman Koren's original tube SPICE models (1996-2001). Contains 300B, 2A3, 6550,
    6L6GC, KT88, EL34, 12AX7, 12AU7, 12AT7, 6SN7, 6SL7, 6AN8, 6C33C, 6DJ8.
    Primary source of published Koren coefficient sets.
    See also errata.txt for corrected capacitance values.
    CAVEAT: the library contains NO EL84 set — an EL84 row attributed to it is a
    misattribution. Any EL84 Koren set claiming this library as its provenance is
    in fact derived (typically templated from EL34), not published here.

### Koren_Tubes extended library — Suusi Malcolm-Brown 2008 (incl. EL84)
- url: https://polonai.se/audiofreaks/Koren_Tubes.txt
- type: data
- local_copy: `spice_models/Koren_Tubes_SuusiMB_2008.txt`
- status: downloaded
- added_on: 2026-07-06
- note: |
    Extension of Norman Koren's library (Suusi Malcolm-Brown, May-June 2008,
    160 subcircuits). The only PUBLISHED Koren EL84 set found so far, credited to
    the Mullard data sheet: MU=21.29 EX=1.240 KG1=401.7 KG2=4500 KP=111.04
    KVB=17.9 VCT=0 RGI=1000. KVB=17.9 agrees independently with a Koren fit to
    measured EL84 curves (17.9) and with datasheet calibration (kvb about 16-24).
    NB the set runs hot in the grid region (Iq(300,-11,300)=39.9 mA vs 36 mA on
    the data sheet; at -14.7 V cutoff 13.3 mA vs 7.5 mA): the knee is closer to
    the data sheet than in templated sets, the grid region is worse.

### Tube_IM.lib — Next-Tube (Eugene Karpov) tube library, 2007 vintage
- url: <https://next-tube.com/libs.php> (author's model-library page; the exact file URL
  used on 2026-04-06 was not captured and the site returns 403 to scripted fetches)
- type: data
- role: reference SPICE models: cross-check of Koren coefficients for Soviet tube types
- local_copy: `data/Tube_IM.lib`
- status: downloaded
- added_on: 2026-08-02
- note: |
    2177-line SPICE library, 116 subcircuits, internally dated 29.03.07 (74 624 B,
    MD5 `7c3204156e25ab5d1bfd18aa580effb1`). Same `Tube_IM.lib` that Next-Tube ships
    inside its own archives — provenance established 2026-08-03 by matching the file
    name AND the five in-file section headers: «*Norman Koren library», «*Duncan Munro
    library», «*Eugene V. Karpov library / For tubes manufactured in USSR», «*Mithat
    Konar library», and models «courtesy of Robert Casey, WA2ISE». The copy kept
    here is an
    intermediate revision of that lineage: PSlib.zip carries the 2001/2002 PSpice
    version (20 241 B, 26 subcircuits), Tube_ORCAD.zip the 2024 version updated by
    GaLeX with Reefman/dos4ever/Bartola models (123 082 B, 210 subcircuits) — both
    kept under `spice_models/next_tube/`, see the «Next-Tube SPICE libraries»
    entry above. NB the file name's «IM» is NOT Adrian Immler: his models
    postdate this file and use his own generic triode model, not Koren's.
    Value over the original Koren set: Soviet/Russian types absent there (6S33S, 6S19P, 6N6P, 6N23P, 6N30P-DR, 6E5P,
    6P14P/EL84, 6P15P, 6P18P, 6F3P, 6F12P, 6J32P, 6J38P, 6P36S, 6S45P-E and others),
    many with separate _T triode-connected variants. Comment header warns that Cyrillic
    «S»/«P» are transliterated (6S33S = 6C33C).
    CAVEAT: the file is community-circulated and its author/publication URL was not
    captured when it was saved; treat parameters as unverified reference material, not
    as a citable source. Index entry added retroactively 2026-08-02.

### chanmix51 LTspice Tube Models (845)
- url: https://gist.github.com/chanmix51/6947361
- local_copy: n/a
- status: referenced
- added_on: 2026-04-06
- note: |
    Community SPICE models for 845, 300B, 2A3 in LTspice format.
    845 model: MU=5.355, EX=1.5, KG1=6323, KP=85.64, KVB=65.8 (Koren-equation fit).
    Used as the 845 reference parameter set.

### Bartola Valves GU-50 SPICE Model
- url: https://www.bartola.co.uk/valves/2023/03/13/gu-50-spice-model/
- local_copy: n/a
- status: referenced
- added_on: 2026-04-06
- note: |
    Ale Moglia (Bartola Valves) GU-50 Koren model, created 4-Jun-2016.
    Pentode mode: MU=6.26, EX=1.296, KG1=380.1, KG2=92411.5, KP=22.3, KVB=437.3.
    Triode mode: MU=6.18, EX=1.38, KG1=448.2, KP=22.3, KVB=700.
    Used as the GU50 reference parameter set. Note: KG2=92411.5 is very high.

### diyAudio Community Models (KT120, KT150, 211)
- url: https://www.diyaudio.com/community/threads/pentode-spice-models-for-kt120-kt150.383522/
- local_copy: `spice_models/diyaudio_kt120_kt150_thread.mht`
- status: downloaded
- added_on: 2026-04-06
- note: |
    Community-fitted Koren pentode models for KT120 and KT150.
    KT120: MU=9.14, EX=1.35, KG1=612.3, KG2=4500, KP=31.40, KVB=20.0.
    KT150: MU=10.77, EX=1.35, KG1=475.4, KG2=886.91, KP=28.6, KVB=14.23.
    211: MU=12.0, EX=1.458, KG1=3928.74, KP=451.14, KVB=249.72 (GE VT4C).
    The thread page is archived here; earlier fetch attempts were refused (403), a
    later one succeeded, so the copy may not be reproducible from that URL.

## Optimization Method Sources (operating-point optimizer research)

### SciPy — differential_evolution reference
- url: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
- type: theory
- local_copy: `optimization/opt_scipy_differential_evolution.html`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Key facts for optimizer redesign: `vectorized=True` passes the WHOLE population
    (S, N) to the objective in one call (numpy batch eval, kills interpreter overhead);
    `workers=-1` parallelizes via multiprocessing; `init='sobol'|'latinhypercube'`
    (Sobol covers space better than LHS per docs); `integrality` mask; constraints via
    `NonlinearConstraint` (feasibility-ranked selection); `callback` returning True or
    raising StopIteration halts the run (maps 1:1 onto a cancellation-polling
    callback);
    `polish=True` runs L-BFGS-B (or trust-constr with constraints) on the best member.
    Default popsize=15 → 15·4=60 members for a 4-parameter problem.

### SciPy — direct (DIRECT / DIRECT_L) reference
- url: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html>
- type: theory
- local_copy: `optimization/opt_scipy_direct.html`
- status: downloaded
- added_on: 2026-06-30
- note: |
    DIRECT = deterministic Lipschitzian global optimization by adaptive hyperrectangle
    subdivision (Jones 1993); scipy default is locally-biased DIRECT_L (Gablonsky &
    Kelley 2001), efficient for problems with few local minima. Box bounds only
    (nonlinear constraints must go through penalty), `callback(xk)` supported but no
    documented early-stop return, maxfun default 1000·N. Deterministic and reproducible —
    attractive replacement for the fixed grid, but single-objective and awkward for
    Pareto-front generation.

### Martins & Ning — Engineering Design Optimization, Multiobjective chapter
- url: <https://mdobook.github.io/html/multiobj/>
- type: theory
- local_copy: `optimization/opt_mdobook_multiobjective.mht`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Textbook (open web edition) guidance: weighted-sum cannot reach nonconvex Pareto
    regions and gives nonuniform spacing; epsilon-constraint reveals nonconvex portions,
    epsilon values map intuitively to objective magnitudes (e.g. "min THD s.t.
    Pout >= P_k" for a ladder of P_k) — directly matches a THD-vs-Pout front;
    NBI gives most uniform spacing at more implementation complexity; evolutionary
    (NSGA-II) recommended when a gradient-free method already fits the single-objective
    problem — whole front in one run, best with many design vars / discrete spaces.

### pymoo — NSGA-II documentation
- url: <https://pymoo.org/algorithms/moo/nsga2.html>
- type: theory
- local_copy: `optimization/opt_pymoo_nsga2.mht`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Reference NSGA-II implementation (non-dominated sorting + crowding-distance
    survival, binary tournament). Handles constraints via feasibility dominance.
    Would replace grid+Pareto-collection in one algorithm, BUT: new heavy dependency
    (pymoo pulls cma, autograd etc.), and for a 4-parameter problem with ms-cheap
    evals a vectorized grid/DE + epsilon-constraint refinement achieves the same
    front without the dependency. Parked unless objective count grows (>2).

### Gavana — Global Optimization Benchmarks (2021)
- url: <http://infinity77.net/go_2021/>
- type: theory
- local_copy: `optimization/opt_gavana_global_optimization_benchmarks.mht`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Large benchmark (thousands of low-dimensional test functions, 2-6 vars — same
    regime as a 4-parameter problem) of 13 global optimizers under hard evaluation
    budgets
    (100..2000, 5000, 10000). At NF=2000: BiteOpt 83.4%, MCS 82.0%, DE 75.4%,
    CRS2 73.4%, DIRECT 72.5%, DualAnnealing 72.4%, SHGO 69.8%, BasinHopping 64.7%.
    Takeaway: scipy DE / DIRECT / DualAnnealing are all competitive; with tiny
    budgets (~100 evals) MCS/SHGO/DualAnnealing/AMPGO solve 35-45% of problems.
    Index page + rankings summarized here; sub-pages not mirrored.

### Brochu, Cora, de Freitas — A Tutorial on Bayesian Optimization of Expensive Cost Functions (2010)
- url: <https://arxiv.org/pdf/1012.2599>
- type: theory
- local_copy: `optimization/opt_brochu_bayesian_optimization_tutorial.pdf`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Canonical BO tutorial. BO's premise: objective is EXPENSIVE (seconds-minutes per
    eval), so it is worth spending O(n^3) GP posterior updates + acquisition-function
    optimization per point to minimize the number of samples. A load-line objective
    costs ~0.1-5 ms (numpy load line + 5-point/Chebyshev HD), so GP overhead per
    suggestion
    (>> objective cost) makes BO strictly slower than vectorized sampling. Registered
    as the standard citation for this negative decision (also applies to Optuna TPE).

### Crowhurst — Understanding Hi-Fi Circuits (Gernsback No. 64, 1957)
- url: <https://www.worldradiohistory.com/Archive-All-Audio/Archive-Audio-Collection/Gernsback-64-Understanding-Hi-Fi-Circuits-1957-Crowhurst.pdf>
- type: theory
- local_copy: `theory/crowhurst_understanding_hifi_circuits_1957.pdf`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Classic Crowhurst text (Ch. 1 "Special output stages", output-load discussion).
    Establishes the historical practice this optimizer automates: optimum load is
    found by sliding a load line over measured plate curves and reading distortion
    off selected ordinates — i.e. a 1D/2D manual search around datasheet operating
    points, NOT dense 4D enumeration. Supports seeding the optimizer from
    datasheet-typical operating points and treating Ub/Ug2 as coarse outer axes.

### VTADiy — Power Stage Loadline Calculator
- url: <https://www.vtadiy.com/loadline-calculators/power-stage-calculator/>
- type: theory
- local_copy: n/a (in-browser calculator — use online)
- status: not_downloaded
- added_on: 2026-06-30
- note: |
    Representative of current DIY/CAD practice (also TubeCad, SE Amp CAD, Glassware):
    interactive manual sliders for V+, screen voltage, bias, load; instant redraw;
    distortion estimated on the single user-chosen operating point. No tool found in
    this class performs automatic multi-parameter optimization — an automated
    optimizer exceeds the state of practice, so there is no off-the-shelf
    tube-CAD algorithm to copy;
    numerical-optimization literature is the right source instead.

### Clark — Nord Modular book, Ch. 12 Distortion Effects (Chebyshev waveshapers)
- url: <https://cim.mcgill.ca/~clark/nordmodularbook/nm_distortion_effects.html>
- type: theory
- local_copy: `theory/clark_nordmodular_ch12_distortion_chebyshev.mht`
- status: downloaded
- added_on: 2026-06-30
- note: |
    Practical waveshaper treatment of Tn(cos θ) = cos(nθ): driving a static transfer
    curve with a cosine and reading harmonics IS Chebyshev decomposition. Consequence
    for implementations: sampling the transfer curve at cosine-spaced points and
    taking a DCT/FFT
    gives all HD2..HDn in O(N log N) with no least-squares polynomial fit — the
    "chebyshev" HD method can be reimplemented as a DFT of y(cos θ) (identical math,
    ~10x cheaper than polyfit). Complements existing Kenny Peng source.

## External Curve Tracer Formats

### eTracer — Digital Tube Curve Tracer (Essues Technologies)
- url: <https://www.essues.com/etracer/>
- type: documentation
- role: reference for potential future import support
- local_copy: n/a (needs the live site — read online)
- status: not_downloaded
- added_on: 2026-04-10
- note: |
    Commercial digital tube curve tracer by Chris Chang (Essues Co. Ltd., Netherlands).
    Impulse measurement method. Software: Python + NumPy/SciPy/Matplotlib (closed source).
    Export formats: CSV, text, PDF, JPEG, BMP. Config files use `.etd` extension.
    CSV format version 2.0. Header comments starting with `#`:
      `# ETRACER_CSV_FORMAT_VERSION:2.0`
      `# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]`
      `# SWEEP_SOURCE types: 0:NONE 1:NEGV 2:HV2`
    Comma-delimited. Col 1 = curve-set index (0-based). Remaining cols = data points, nan-padded.
    6 rows per curve-set:
      Row 1: HV1_V (anode voltage, V), Row 2: HV1_I (anode current, mA),
      Row 3: HV2_V (screen/unit2 voltage), Row 4: HV2_I (screen/unit2 current),
      Row 5: NEGV (grid voltage, V, negative), Row 6: Sweep source indicator.
    Pentode: sweep_src=2 → HV2 used as screen grid. Double triodes: unit2 in rows 3-4.
    Sample data (6C4, 300B, EL156 etc.) available on site but requires registration.
    Converter exists: eTracer → uTracer/ExtractModel format (by Chris Chang).

### VT52.com eTracer Data Collection
- url: <http://www.vt52.com/etracer-files>
- type: data
- role: eTracer format reference and test data
- local_copy: `data/etracer_samples/`
  (35 CSV + 35 ETD files)
- status: downloaded
- added_on: 2026-04-10
- note: |
    ZIP archive (165 KB) with tube.cfg (.etd) and CSV scan files for ~35 tubes:
    Russian (4P1L, 6E5P, 6S45P, 6S4P, 6Z9P, etc.), US (10Y, 71, 801A, VT52, 50),
    European (D3a, E180F, E810F, EC8010, EL34, EL84, KC3, RS241).
    CSV format version 2.0, confirmed header:
      `# ETRACER_CSV_FORMAT_VERSION:2.0`
      `# Each curve-set contains 6 rows: [HV1_V HV1_I HV2_V HV2_I NEGV SWEEP_SOURCE]`
    Comma-delimited, col 1 = curve-set index, remaining = data points, nan-padded.
    ETD files = INI-style tube config (pin assignment, heater, sweep ranges, quick test refs).

### matetracer — GNU Octave eTracer Parser (Matthias Brennwald)
- url: <https://github.com/mbrennwa/matetracer>
- type: data
- role: reference: eTracer CSV format reverse-engineering
- local_copy: `data/matetracer.zip`
- status: downloaded
- added_on: 2026-04-10
- note: |
    GNU Octave scripts for reading eTracer CSV files. Key file: `met_read_tube_data.m`.
    Uses `load(file)` — confirms pure numeric CSV (no headers). 6-row grouping per trace.
    Sweep source field determines grid voltage interpretation.
    Same author as pypsucurvetrace (below).

### pypsucurvetrace — Python PSU Curve Tracer (Matthias Brennwald)
- url: <https://github.com/mbrennwa/pypsucurvetrace>
- type: data
- role: reference: alternative curve tracer data format
- local_copy: `data/pypsucurvetrace.zip`
- status: downloaded
- added_on: 2026-04-10
- note: |
    Repository archived here as a zip (9.6 MB). NB the companion DATA repository
    (curvetracedata) exceeds 120 MB and is deliberately NOT archived.
    Python-based curve tracer using lab PSUs. `.DAT` format: `%`-comment header with metadata,
    10 columns space-delimited:
      1-5: PSU1 (anode): Vnom, Inom, Vmeas, Imeas, limiter
      6-10: PSU2 (grid): Vnom, Inom, Vmeas, Imeas, limiter
    Current in Amps (not mA). Sample files in `examples/curvedata/` (6C4, 300B, etc.).

### uTracer .utd Format (Ronald Dekker, dos4ever.com)
- url: <https://www.dos4ever.com/uTracer3/uTracer3_pag5.html>
- type: documentation
- role: reference for potential future import support
- local_copy: n/a (needs the live site — read online)
- status: not_downloaded
- added_on: 2026-04-10
- note: |
    uTracer tube curve tracer data format. Three sub-formats:
    **Measurement Matrix** (8 columns, tab-separated):
      1: Point# within curve, 2: Curve#, 3: Ia (mA), 4: Is (mA),
      5: Vg (setpoint, V), 6: Va (measured, V), 7: Vs (measured, V), 8: Vh (setpoint, V).
    **Block format** (text, human-readable):
      Header: `Vg (V)  Ia (mA)`, then `Va = NNN V`, then Vg/Ia rows.
      One curve per file, single Va value.
    **List format**: sequential per-curve storage.
    Distinction: Vg/Vh = setpoints, Va/Vs = measured values.
    150+ sample .utd files at github.com/rrMacKinnon/Electrona_uTracer/SampleTubeData/.

### AVT5229 — Original Project Documentation (Elektronika Praktyczna 5/2010)
- url: <https://serwis.avt.pl/manuals/AVT5229.pdf>
- type: documentation
- local_copy: `AVT5229_article_EP_5-2010.pdf`
- status: downloaded
- added_on: 2026-04-19
- note: |
    Original article "Miernik lamp elektronowych AVT 5229" in Elektronika Praktyczna 5/2010,
    p. 39+ (multi-part series). Authors: Adam Tatuś & Tomasz Gumny (AVT Korporacja, Poland).
    Describes schematic, PCB, ATmega16 firmware (VTTester), measurement methodology,
    LCD menu, and the built-in catalog of ~100 tubes.
    Kit still sold at sklep.avt.pl/avt5229.html (product listing) and
    https://ep.com.pl/kity/13706-miernik-lamp-elektronowych-avt5229.

### AVT5229 — Vanilla firmware source, v1.16 (official AVT archive)
- url: <https://serwis.avt.pl/files/AVT5229.zip>
- type: data
- local_copy: `AVT5229_vanilla_1.16.zip` (25 451 B, MD5 `0851e6a2f9d3851809465b24edf35c98`)
  — archive contents (verified 2026-08-03):
  `TTesterLCD.c` (53 432 B, 2012-02-23 18:11, MD5 `e6688b4d75a6b711e8d7f416468cfd39`),
  `TTesterLCD.eep` (1 473 B, EEPROM image with factory tube catalog,
  MD5 `e81b85e4f3ebe763f663ab64f6393198`),
  `TTesterLCD.hex` (43 156 B, Intel-HEX flash image,
  MD5 `20340b1960f4adda2919a3bd08cf3955`).
- status: downloaded
- added_on: 2026-04-19
- note: |
    Official vanilla release of VTTester firmware v1.16 from AVT Korporacja archive.
    No UART/RS-232 protocol and no LM19 temperature-sensor support — both present only
    in the fork kept here. `historia.txt`-style changelog inside the source ends at
    `1.16 23/02/2012 dodano lampę 6F6S`. The same archive is byte-identical to the copy
    posted in forum-trioda.pl thread 12209 (attachment id=59201), which is the only
    other public location. Confirmed via agent search (GitHub, elektroda.pl, forum-trioda,
    radiokot and other RU/UA electronics forums, 2026-04-19) that no public community
    fork with UART or LM19-sensor support exists — those extensions are original
    work of the fork kept here, not derived from an external patched source.

### AVT1694 — Przystawka (Companion Adapter, EP 8/2012)
- url: <https://serwis.avt.pl/manuals/AVT1694.pdf>
- type: documentation
- role: reference for original hardware
- local_copy: `AVT1694_article_EP_8-2012.pdf`
- status: downloaded
- added_on: 2026-04-19
- note: |
    "Przystawka do miernika lamp elektronowych" in Elektronika Praktyczna 8/2012, p. 67+.
    Same authors (Tatuś & Gumny). Double-sided PTH PCB 215x65 mm that replaces the
    point-to-point wired tube-socket module of AVT5229 with a single board (mechanical
    stability, easier assembly). Electrically compatible with AVT5229 — not a redesign.
    Kit page: https://sklep.avt.pl/avt1694.html.

### Electrona uTracer Sample Data (rrMacKinnon)
- url: <https://github.com/rrMacKinnon/Electrona_uTracer>
- type: data
- role: reference: 150+ uTracer .utd sample files of 5749 tubes
- local_copy: `data/electrona_utracer_samples.zip`
- status: downloaded
- added_on: 2026-04-10
- note: |
    GitHub repo with 150+ .utd files (Block format) for 5749 tubes.
    All measured at Va=325V, swept Vg from -50 to -2V.
    Useful for testing uTracer import and tube matching features.

## Tube Health / Cathode Emission / Matching Theory

### Emission Labs TB-08 — "The right way to TEST Electron tubes"
- url: <http://www.emissionlabs.com/Articles/TECH-BULLETIN/TB-08-Testing-Emissionlabs-Tubes/testing-eml-index.html>
- type: theory
- local_copy: `theory/eml_tb08_testing_tubes.mht`
- status: downloaded
- added_on: 2026-08-01
- note: |
    Jac van de Walle (Emission Labs / JAC Music). Core claims:
    (1) wear shows as a CHANGE in gm against the tube's own factory value, not as an
    absolute gm number; (2) gm is a function of plate current, so a valid comparison
    requires setting Ug1 until Ia equals the datasheet/factory current and measuring
    gm THERE — fixed-Ug1 testing can be off by 30%; (3) pulse/curve-tracer testers
    read Ia and gm low because the anode never reaches thermal equilibrium, and
    cathode-leakage / grid-current / drift tests are invalid on a cold anode
    (needs ~5 min at full dissipation, ~30 min for drift).
    Site is HTTP-only and breaks TLS negotiation on https — fetch with curl over http.

### Emission Labs — "About Lifetime of our tubes"
- url: <http://www.emissionlabs.com/html/guarantee/About-Lifetime.html>
- type: theory
- local_copy: `theory/eml_about_lifetime.mht`
- status: downloaded
- added_on: 2026-08-01
- note: |
    Manufacturer's numeric wear scale, used as an external cross-check for the
    Health-tab verdict boundaries: Ia >= 70% of the tube's OWN factory-test value
    = "good" (stated as the level where the 'good' reading begins on most testers);
    40-70% = often still usable depending on the amplifier; below 40% = problems
    likely. Also: factory marks the post-burn-in Ia and the Ug1 it was measured at
    on the box, and a later test is only comparable at that same Ug1. Slow Ia decay
    over hundreds of hours is normal wear, acceleration of the decay signals
    end-of-life. Heater-voltage discipline ("zero tolerance") is given as the
    dominant lifetime factor - over-heating empties the barium depot, under-heating
    stops regeneration.

### Accurate Tube Testing Information (tubetesting.yolasite.com)
- url: <https://tubetesting.yolasite.com/accurate-tube-testing-information.php>
- type: theory
- local_copy: `theory/tubetesting_accurate_testing.html`
- status: downloaded
- added_on: 2026-08-01
- note: |
    Practical bench description of the classic "life test": drop the heater voltage
    ~10% and note how much gm/Ia falls. A healthy cathode is space-charge limited,
    so the reading barely moves; a depleted cathode is emission limited and the
    reading collapses. A tester can push the same probe deeper — e.g. 80% heater
    ratio, scoring Ia80/Ia100 against a nominal near 0.90. Source also documents the
    +/-25% gm / +/-10% mu production spread typical for small-signal tubes.

### Miram curve knee - physical factors (arXiv 2202.08247 / IEEE TED)
- url: <https://arxiv.org/abs/2202.08247>
- type: theory
- local_copy: `theory/miram_knee_shape_arxiv_2202.08247.pdf`
- status: downloaded
- added_on: 2026-08-01
- note: |
    The Miram curve is emitted current vs cathode temperature (in practice:
    vs heater voltage). It has a plateau where emission is space-charge
    limited (current set by Child-Langmuir, nearly independent of cathode
    temperature) and an exponential branch where it is source/temperature
    limited (Richardson-Laue-Dushman). The transition is the "knee".
    Relevance: the single-point Ia80/Ia100 ratio is a
    one-sample probe of this curve. Sweeping the heater and locating the
    knee measures the actual reserve, and the knee voltage is what shifts
    upward as the cathode's emitting material is consumed. Paper also
    explains why real knees are smooth (nonuniform work function / patch
    fields), i.e. why a knee estimator must fit a transition, not look
    for a corner.

### Edge effect on the current-temperature characteristic of thermionic cathodes (arXiv 2106.05311)
- url: <https://arxiv.org/abs/2106.05311>
- type: theory
- local_copy: `theory/miram_curve_edge_effect_arxiv_2106.05311.pdf`
- status: downloaded
- added_on: 2026-08-01
- note: |
    Molecular-dynamics study of Miram-curve knee smoothing on finite-area
    emitters. Companion to arXiv 2202.08247; taken together they are the
    reason to model the reserve as "heater voltage at which Ia leaves the
    plateau by X %" rather than as a geometric corner.

### Tubes 201 - How Vacuum Tubes Really Work (John Harper)
- url: <https://www.john-a-harper.com/tubes201/>
- type: theory
- role: accessible reference for space-charge vs temperature-limited operation
- local_copy: `theory/harper_tubes201.mht`
- status: downloaded
- added_on: 2026-08-01
- note: |
    Receiving-tube-level treatment of the same physics as the Miram papers:
    why a healthy tube's anode current barely responds to heater voltage
    (space-charge limited) and why a depleted cathode's does.

### Dip test: rapid cathode activity evaluation (PTEP 2017/11/113G02)
- url: <https://academic.oup.com/ptep/article/2017/11/113G02/4676043>
- type: theory
- role: method reference for shortened emission-reserve probing
- local_copy: n/a (publisher blocks scripted download; open access in browser)
- status: not_downloaded
- added_on: 2026-08-01
- note: |
    Publisher refuses scripted download (403 behind a bot check); readable in a
    browser. Describes the standard practice of measuring cathode emission current
    as a function of heater voltage (Miram plot) to estimate remaining
    life, and a shortened "dip" variant that drops the heater briefly
    instead of running the full sweep - the trade-off any single-point probe
    already makes. Key extracted claim: the knee moves
    to higher heater voltage as the cathode ages, so the *shift* of the
    knee, not the absolute current, is the wear signal.

### Aiken — The Last Word On Biasing (matched tubes section)
- url: <https://www.aikenamps.com/the-last-word-on-biasing>
- type: theory
- local_copy: `theory/aiken_last_word_on_biasing.mht`
- status: downloaded
- added_on: 2026-08-02
- note: |
    Canonical treatment of PP matching semantics. Matching = same Ia at the
    same grid voltage AND same transconductance. DC balance: mismatched idle
    currents magnetize the gapless PP output transformer core (inductance
    drops, LF suffers). AC balance: gm matching — and gm must be compared AT
    THE IDLE POINT THE AMP USES ("may be matched at one grid voltage/plate
    current but not at another"). Individual bias pots or a bias+balance
    arrangement give perfect DC balance even with unmatched tubes — i.e. the
    matching requirement depends on the amp's bias topology. A shared cathode
    resistor HIDES per-tube mismatch entirely.

### Apex Tube Matching — A Technician's Explanation
- url: <https://www.apexmatching.com/why-should-i-match-my-tubes-technicians-explanation>
- type: theory
- local_copy: `theory/apex_why_match_technician.html` (text only, figures not stored)
- status: downloaded
- added_on: 2026-08-02
- note: |
    Commercial matcher methodology. Tubes are matched so they "can be biased
    TOGETHER" — the shared-bias assumption is explicit. gm varies along the
    transfer curve, so matching is point-specific. Worked example (Fender
    Blues Deluxe, fixed non-adjustable bias): a "cool" matched pair idles at
    9.2 W/tube, a "hot" matched pair at 14.8 W in the SAME amp — absolute
    current class matters, not only pair closeness ("matched to the amp").
    Unmatched hot+cool pair: 20 vs 39 mA — DC imbalance plus the hot tube
    wears out early.

### TubeSound — Tube Matching with a Tube Tester
- url: <https://tubesound.com/tube-matching-with-a-tube-tester/>
- type: theory
- local_copy: `theory/tubesound_matching_with_tester.html`
- status: downloaded
- added_on: 2026-08-02
- note: |
    Practical bench procedure: match by Ip AND Gm at the operating point the
    amp will use; gm divergence grows with drive (7 vs 5 mA/V partners reach
    245 vs 175 mA at zero grid), so gm mismatch shows up mostly at high
    drive as asymmetric clipping.
