# LM19 Plots — The Complete Guide

A detailed description of every plot in the LM19 application: the physics
behind it, how to read it, and how to use it in practical work with
vacuum tubes.

---

## Contents

- [Core concepts](#core-concepts)
- [2D — Ia(Ua): Anode characteristics](#2d--iaua-anode-characteristics)
- [Transfer — Ia(Ug1): Transfer characteristic](#transfer--iaug1-transfer-characteristic)
- [Contour — Ia(Ua, Ug1): Contour map](#contour--iaua-ug1-contour-map)
- [Gm/Rp — Transconductance and plate resistance maps](#gmrp--transconductance-and-plate-resistance-maps)
- [Pa Map — Anode dissipation map](#pa-map--anode-dissipation-map)
- [Curves — Parametric curves](#curves--parametric-curves)
  - [Gm (transconductance)](#gm-transconductance)
  - [Rp (plate resistance)](#rp-plate-resistance)
  - [mu (amplification factor)](#mu-amplification-factor)
  - [Ia (anode current)](#ia-anode-current)
  - [Ig2 (screen grid current)](#ig2-screen-grid-current)
  - [Pa (anode power)](#pa-anode-power)
  - [Pig2 (screen grid power)](#pig2-screen-grid-power)
  - [Ia/Ig2 (current ratio)](#iaig2-current-ratio)
- [Overlays](#overlays)
- [Compare — Tube comparison](#compare--tube-comparison)
- [Amplifier — Amplifier analysis](#amplifier--amplifier-analysis)
- [Practical scenarios](#practical-scenarios)
- [Diagnosing faults from the plots](#diagnosing-faults-from-the-plots)
- [Choosing the operating point: a step-by-step guide](#choosing-the-operating-point-a-step-by-step-guide)

---

## Core concepts

### What the tester measures

The AVT5229 (LM19) tester records the current–voltage (I-V)
characteristics of vacuum tubes — triodes and pentodes. Each point
measures:

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Anode voltage | Ua | V | Voltage on the anode |
| Grid 1 voltage | Ug1 | V | Bias voltage (negative) |
| Grid 2 voltage | Ug2 | V | Screen grid (pentodes) |
| Anode current | Ia | mA | Current through the anode |
| Screen grid current | Ig2 | mA | Current through grid 2 (pentodes) |

### Derived parameters

Computed from the measured data:

| Parameter | Formula | Unit | Physical meaning |
|-----------|---------|------|------------------|
| Gm (transconductance) | ΔIa / ΔUg1 | mA/V | The grid's ability to control the current |
| Rp (plate resistance) | ΔUa / ΔIa | kΩ | AC resistance of the anode–cathode path |
| μ (amplification factor) | Gm × Rp | — | Maximum gain with no load |
| Pa (anode power) | Ua × Ia / 1000 | W | Thermal load on the anode |
| Pig2 (screen power) | Ug2 × Ig2 / 1000 | W | Thermal load on the screen grid |

### How the parameters relate: the Barkhausen equation

The three key tube parameters are bound by a fundamental relation:

**μ = Gm × Rp**

This is not an approximation but an exact identity following from the
definitions of the partial derivatives. If one parameter changes, the
other two remain linked through this equation.

### Triodes and pentodes

- **Triode**: three electrodes (cathode, grid, anode). No Ug2.
  Typical μ = 4–100, Rp from hundreds of ohms (power triodes: 2A3, 300B)
  up to 60+ kΩ (high-μ types: 12AX7). Anode curves have a noticeable slope.

- **Pentode**: five electrodes (adds the screen and suppressor grids).
  Ug2 is either fixed or tracks Ua (triode connection). Rp from 15 kΩ
  (output types: EL34, 6L6) up to megohms (small-signal types: EF86).
  High Gm, nearly flat anode curves.

- **Pentode in triode connection**: Ug2 is tied to the anode. The
  behaviour approaches a triode — Rp drops sharply, μ decreases, and the
  anode curves acquire a visible slope (they stop being flat).

---

## 2D — Ia(Ua): Anode characteristics

### What it shows

The family of curves **Ia = f(Ua)** at various fixed Ug1 values.
This is the primary plot for characterising a tube — the operating point
is chosen and the load line is drawn on it.

### How to read it

- **Each curve** is one Ug1 value (labelled at its right end); for a
  pentode with independent Ug2 every Ug1 has a family of curves, one per
  screen voltage level (see “Ug2 display mode” below).
- **Snap marker**: hovering the cursor snaps a marker to the nearest
  curve and shows the exact point values (Ua, Ug1, Ug2, Ia). Works on
  2D, Transfer and Curves; the heatmaps have their own hover tooltip.
- **Curve steepness** (slope) is the inverse of the plate resistance Rp:
  the steeper the curve, the lower Rp.
- **Spacing between curves** (at equal ΔUg1) is proportional to the
  transconductance Gm: the wider the spacing, the higher Gm.
- **The knee** (the sharp bend at low Ua) is the saturation region where
  the tube is no longer linear. The operating point must sit to the
  right of the knee.

### What the curve shapes mean

| Shape | Tube type | Meaning |
|-------|-----------|---------|
| Gentle, evenly diverging | Triode | Normal behaviour; moderate μ and Rp |
| Nearly horizontal, tightly grouped | Pentode | High Rp, current barely depends on Ua |
| Curves “stick together” at large \|Ug1\| | Any | The tube is entering cutoff |
| Uneven spacing between curves | Any | Gm nonlinearity → distortion |

### Practical use

1. **Choosing the operating point**: the intersection of the load line
   with the curve at the desired Ug1. The point must stay clear of the
   knee and of the Ua/Ia maxima.
2. **Judging linearity**: equal spacing between curves = low distortion.
3. **Tube matching**: two samples with coinciding curves make a good
   push-pull pair.
4. **Health check**: comparison against the datasheet characteristics.

### Ug2 display mode (pentodes)

A pentode scan with independent Ug2 contains a family of curves for each
screen voltage level. Two encoding modes (the **Ug2 mode** radio buttons
in Plot Options):

- **Ug2 as color** (default) — a continuous viridis colour scale: the
  curve colour encodes the Ug2 value, and a colorbar to the right of the
  plot maps colour→volts. The standard output-characteristics view.
- **Ug2 families** — each Ug2 level gets its own discrete colour from
  the palette plus a legend entry “Ug2 = N V”. Convenient when there are
  few levels and they need to be told apart individually.

Both modes only choose the COLOURING: switching them does not regroup
the points into curves. Grouping is determined by **how the measurement
was taken**: a scan with independent Ug2 groups by the (Ug1, Ug2) pair,
while a triode-connected scan (Ug2 = Ua + offset) and a true triode
group by Ug1 alone. The mode is recorded in the measurement itself, so
arming the next scan in the other Ug2 mode (or picking a different tube
in the selector) does not regroup the curves already on the plot.

### Ug2 filter (pentodes)

For pentodes the **Ug2 display** multi-select shows or hides the curves
at particular screen voltage values. It acts on 2D, Transfer and Curves
simultaneously. Switching is instant — every curve is pre-drawn and only
its visibility is toggled.

The separate **Ug2 calc** selector picks the single Ug2 value used by
the computed plots (Contour, Gm/Rp, Pa map).

### Overlays on the 2D plot

- **Working line** (the **Load line** checkbox in Plot Options) — the
  operating-point path under load; its shape depends on the circuit
  selected on the Amplifier panel. See
  [Overlays → Load line](#load-line).
- **Pa max hyperbola** — the maximum anode dissipation boundary.
- **Ua max / Ia max lines** — absolute limits.
- **SRK zone** — the region used to compute transconductance (S), plate
  resistance (R) and amplification factor (K).

---

## Transfer — Ia(Ug1): Transfer characteristic

### What it shows

The dependence **Ia = f(Ug1)** at fixed Ua values. Shows how the control
grid drives the anode current.

Transfer uses the same unified rendering pipeline as Curves (through
`render_curves` with `target_plot=transfer_plot`, `x_param="Ug1"`,
`y_param="Ia"`). Data is grouped by `series_id` (as in Curves), not by
`(lamp_type, lamp_id)` — this removes artefacts when the same tube is
overlaid in different modes (e.g. pentode and triode connection). Load
line intersections are shown on Transfer and on Curves whenever the axes
are X=Ug1, Y=Ia.

### How to read it

- **Curve slope** = the transconductance Gm at that point. The steeper
  the curve, the higher Gm.
- **Curve shape**: an ideal tube follows a smooth power law
  (Ia ~ (Ug1 + const)^(3/2) for triodes). Deviations from the smooth
  curve = structural defects.
- **Cutoff voltage**: the point where the curve reaches zero. For
  class A the operating point must sit well to the right of cutoff
  (at a less negative Ug1).

### Practical use

1. **Finding the usable Ug1 range**: locate the region where the curve
   is most linear (constant transconductance).
2. **Judging half-wave symmetry**: projecting the load line onto the
   transfer characteristic shows whether the signal swings symmetrically
   around the operating point.
3. **Bias calculation**: find the Ug1 at which Ia matches the desired
   operating point for a given Ua.

### Ua slices: “View” presets and the filter

A scan sweeps Ua densely (dozens of values), while a datasheet-style
transfer characteristic is read at 3–5 slices. Two controls sit above
the plot:

- **View** — a preset for the set of visible slices:
  - *All* — every measured Ua (the old behaviour);
  - *Datasheet* (default) — 5 evenly spaced slices including min/max;
  - *Load line* — the slice closest to the amp panel's Ub, ± one
    neighbour;
  - *Custom* — set automatically when the Ua list is edited by hand.
- **Ua** — a slice multi-select (mirror of the Ug2 filter). Switching is
  instant: visibility of pre-built items, no re-render. It combines with
  the Ug2 filter (visibility = AND of both filters).

Accents: the slice closest to Ub is labelled “≈Ub” (only when Ub lies
inside the measured Ua range); with an active working line the slices
outside its Ua span are dimmed; the dynamic intersection curve is drawn
bold with a white underlay. Accents refresh on a full re-render
(Analyze / data change), not on live working-line ticks.

The PDF report (section `plot_transfer`) receives Transfer **WYSIWYG** —
with the active Ua filter and accents exactly as on screen (consistent
with the Ug2 filter). Export does NOT reset the preset to “All”; the
WYSIWYG precondition (hidden slices must not reach the exported pixmap)
is pinned by `tests/test_transfer_ua_filter.py::TestPdfWysiwyg`.

Implementation: presets and the `TRANSFER_VIEW_*` registry —
`app/plot_manager.py`; the visibility store and the combined predicate —
`app/plotting/_plot_2d_mixin.py`; pins — `tests/test_transfer_ua_filter.py`.

### PP composite

With the **Push-Pull** circuit selected, Transfer (and Curves with the
Ia/Ug1 axes) draws the **composite transfer characteristic** on top of
the family: `Ia_comp(Ug1) = Ia_A(Ug1) − Ia_B(2·bias − Ug1)` — the
data-path curve at Ua=Ub, exactly the one analysed by
`pp_distortion`/Chebyshev (a mismatched pair uses tube B's own data).
Its straightness is what determines the PP stage's distortion.

Display — **folded into the positive quadrant** (the tab's Ia axis
starts at zero): the solid magenta curve is the positive half-wave, the
dashed one is the negative half-wave mirrored around the bias with the
sign flipped (`fold_pp_composite`). For a perfectly matched pair the two
branches coincide; **the gap between the solid and dashed branches = the
residual of even harmonics** (pair imbalance) — readable without any
measurement. An S-shape of both branches = odd harmonics.

The curve does not participate in the Ua/Ug2 filters, refreshes on a
full re-render (Analyze / data change) and is cleared when a non-PP
circuit is selected. Pins — `tests/test_pp_composite_transfer.py`.

---

## Contour — Ia(Ua, Ug1): Contour map

### What it shows

A heatmap of the anode current Ia as a function of two variables:
Ua (horizontal axis) and Ug1 (vertical axis). Colour encodes the current
magnitude.

### How to read it

- Bright (hot) regions — high current.
- Dark regions — low current / cutoff.
- Horizontal “bands” of uniform colour — the current does not depend on
  Ua (pentode behaviour, high Rp).
- Diagonal gradients — the current depends on both parameters (triode
  behaviour).

### Practical use

A quick visual survey of the tube's whole operating region. Helps find
the area with the most uniform gradient (the linear zone).

### Colour scale and “Lock scale” (all 5 maps)

Every heatmap (Contour, Gm, Rp, µ, Pa) has a **colorbar** on its right —
a colour→value scale labelled with the unit (mA, mA/V, kΩ, dimensionless
µ, W). The scale refreshes on every render and follows the selected
colormap (the colormap combo). An empty map hides its scale — there is
never a scale “from the previous scan” next to a blank image.

The **“Lock scale”** checkbox (axes settings row) freezes the current
levels of all maps: subsequent renders (another scan, another Ug2 slice)
are drawn in the same colour scale, so two scans become directly
comparable by colour (emission degradation is visible as the map
“cooling down”). Unchecking immediately returns to auto-levels. If the
lock is enabled before any data exists, the first render's levels are
captured.

A note on Rp/µ: the colour scale of these maps reflects the values
**clipped** for contrast (clip p95×1.5 / p97×1.3), as do the maps
themselves; the exact computed value is available in the hover tooltip.

---

## Gm/Rp — Transconductance and plate resistance maps

### What it shows

Two heatmaps:
- **Gm(Ua, Ug1)** — the transconductance map, mA/V.
- **Rp(Ua, Ug1)** — the plate resistance map, kΩ.

The right-hand map is switchable with the **Rp / µ** combo: instead of
plate resistance it can show the amplification factor map
**µ(Ua, Ug1) = Gm × Rp** (dimensionless). For triodes a uniform µ-map
colour across the whole operating region is the sign of a good sample.

### How it is computed

From the measured Ia(Ua, Ug1) grid:
- **Gm = ΔIa / ΔUg1** (at fixed Ua) — finite difference between
  neighbouring Ug1.
- **Rp = ΔUa / ΔIa** (at fixed Ug1) — finite difference between
  neighbouring Ua.

Values are computed at the midpoints between grid nodes.

### How to read it

**The Gm map:**
- Hot regions (high Gm) — the tube is most sensitive to grid control.
- Gm grows with current (less negative Ug1, higher Ua).
- For class A: the operating point should sit in a region of stable Gm
  (uniform colour).

**The Rp map:**
- Triodes: Rp from hundreds of ohms to tens of kΩ, changing smoothly.
- Pentodes: Rp from tens of kΩ to megohms, almost independent of Ua
  (horizontal bands).
- Very high Rp at low currents — the tube is near cutoff.

### Practical use

1. **Uniformity check**: a “blotchy” Gm map points to nonlinearities or
   measurement artefacts.
2. **Operating point choice**: the region with maximal and stable Gm is
   optimal for amplification.
3. **Datasheet comparison**: Gm at the operating point should match the
   reference data (typical tolerance ±20%).

---

## Pa Map — Anode dissipation map

### What it shows

A heatmap of **Pa = Ua × Ia** (in watts) in (Ua, Ug1) coordinates.

### How to read it

- Hot regions — high dissipated power.
- Pa grows towards the top-right corner (high Ua and Ia).
- The Pa max boundary (from the datasheet) — the operating point must
  not leave this region in static operation.

### Practical use

A quick visual check that the operating point is not in the overheating
zone. Especially useful when designing output stages, where Pa max is
the critical parameter.

---

## Curves — Parametric curves

The Curves tab is the universal tool for exploring any dependency
between tube parameters. Available axes:

- **X**: Ua, Ug1, Ug2
- **Y**: Gm, Rp, μ, Ia, Ig2, Pa, Pig2, Ia/Ig2

For pentodes the shared Ug2 display filter instantly shows or hides
curve groups by screen voltage (distinct colours + legend). The filter
is shared across 2D, Transfer and Curves.

With the axes **X=Ug1, Y=Ia** (the transfer characteristic) Curves shows
the load line intersections — same as the Transfer tab.

---

### Gm (transconductance)

**Gm = ΔIa / ΔUg1** — the slope of the characteristic, mA/V.

Shows how many milliamps of anode current change one volt of grid
voltage produces. The main figure of a tube's amplifying ability.

**Gm vs Ua** (at various Ug1):
- Gm grows with Ua until it reaches a “plateau”. The operating point
  should sit on or near the plateau.
- At low Ua (the knee region) Gm drops sharply — the tube loses its
  amplifying properties.

**Gm vs Ug1** (at various Ua):
- Gm is maximal at small \|Ug1\| (high current) and falls monotonically
  towards cutoff as \|Ug1\| grows.
- The shape of Gm(Ug1) is tied to the exponent of the characteristic:
  for the ideal 3/2 law, Gm ~ (Ug1 + const)^(1/2).

**The S nominal reference line**: when the tube's datasheet
transconductance is configured, a horizontal dashed line is drawn.
A deviation of more than 20% from nominal may indicate tube wear.

---

### Rp (plate resistance)

**Rp = ΔUa / ΔIa** — the tube's AC plate resistance, kΩ.

Determines the output impedance of an amplifier stage (without feedback)
and affects load damping.

**Rp vs Ua**:
- Triodes: Rp from hundreds of ohms to tens of kΩ, weakly dependent on
  Ua in the operating region.
- Pentodes: Rp from tens of kΩ to megohms. High Rp — the pentode acts as
  a current source (ideal for maximum gain, poor for damping a
  loudspeaker).

**Rp vs Ug1**:
- Rp decreases with growing current (decreasing \|Ug1\|). At high
  currents Rp stabilises.
- Rp rises sharply near cutoff (large \|Ug1\|) — the tube is “closing”.

**What Rp means in practice:**

| Context | Desired Rp | Why |
|---------|------------|-----|
| Driver with an RC load | Low | Less loss in the Rp/Ra divider |
| SE output stage | Low (triode) | Better loudspeaker damping |
| Pentode output stage | High (normal) | Feedback compensates |
| Cathode follower | Low | Sets the output impedance ≈ 1/Gm |

---

### mu (amplification factor)

**μ = Gm × Rp** — the dimensionless voltage amplification factor at an
infinitely large load resistance.

The maximum gain the tube can deliver under ideal conditions. The real
gain of a stage with anode load Ra:

**A = μ × Ra / (Rp + Ra)**

**mu vs Ua**:
- Triodes: μ is nearly constant (20–100). This is a property of the
  triode — μ is set by the electrode geometry and barely depends on the
  operating regime.
- Pentodes: μ is very high (200–2000+) and unstable, which is why
  pentodes are usually characterised by Gm rather than μ.

**mu vs Ug1**:
- In a triode μ barely depends on Ug1 across the operating region —
  a confirmation of tube quality.
- A significant spread of μ over Ug1 points to nonlinearity or a
  construction defect.

**Practical meaning:**
μ stability is a triode quality metric. A good triode holds μ within
±10% across the whole operating region.

---

### Ia (anode current)

A direct display of the anode current. Essentially duplicates the 2D
tab, but lets you choose the X axis:

- **Ia vs Ua**: the classic anode characteristics.
- **Ia vs Ug1**: the transfer characteristic.
- **Ia vs Ug2**: dependence on the screen voltage (pentodes).

The main advantage is the uniform interface with the Ug2 multi-select
and the shared Curves tab settings.

---

### Ig2 (screen grid current)

The family of curves **Ig2 = f(X)** at various parameters.
Pentodes/tetrodes only.

**Ig2 vs Ua** (at various Ug1):
- High Ig2 at low Ua — in the knee region part of the electron stream is
  intercepted by the screen grid. Normal pentode behaviour, but
  dangerous for the grid during prolonged operation at low Ua.
- Ig2 falls as Ua grows — the higher Ua, the more electrons fly through
  to the anode.
- Negative Ig2 at very low Ua — electrons may return from the anode
  (more pronounced in beam tetrodes such as 6L6, KT88).

**Ig2 vs Ug2**: shows how the screen current depends on the screen
voltage. Useful for judging overload at elevated Ug2.

**Uses:**
1. Keeping Pig2 (Ug2 × Ig2) under the datasheet limit.
2. Operating point choice — avoid the knee region where Ig2 rises
   sharply.
3. Diagnostics — abnormally high Ig2 may indicate emission loss or grid
   deformation.

---

### Pa (anode power)

**Pa = Ua × Ia / 1000** — the power dissipated on the anode, watts.

**Pa vs Ua** (at various Ug1):
- Pa grows with Ua. The curve shape shows where the tube approaches its
  thermal limit.
- Lets you find the maximum Ua for a given Ug1 without exceeding Pa max.

**Pa vs Ug1** (at various Ua):
- Shows how the bias affects the dissipated power. Important for
  class AB: biasing towards lower currents reduces Pa.

**Pa vs Ug2**:
- Pentodes: shows the screen voltage's influence on anode power. Raising
  Ug2 increases Pa.

**Use:** thermal load control. The datasheet Pa max is a limit on the
average dissipated power (set by the anode's heat dissipation). In
class A the maximum Pa occurs at idle (no signal) — with signal applied
part of the power goes into the load and the average Pa drops. In
class AB/B it is the opposite: the average Pa grows with signal
amplitude, so watching Pa max at full power is especially important.

---

### Pig2 (screen grid power)

**Pig2 = Ug2 × Ig2 / 1000** — the power dissipated on the screen grid,
watts. Pentodes only.

The screen grid is the pentode's weak spot. The Pig2 limit is usually
much lower than Pa max (e.g. EL34: Pa max = 25 W, Pig2 max = 8 W).

**Pig2 vs Ua**:
- At low Ua (the knee region) Ig2 rises sharply → Pig2 grows. This is
  the most dangerous regime for the screen grid.
- As Ua grows, Ig2 falls → Pig2 drops.

**Pig2 vs Ug2**:
- Shows the power's dependence on the screen voltage. Lets you determine
  the safe Ug2 range.

**Use:** critical for designing pentode output stages. Especially in the
UL (ultralinear) connection, where Ug2 is not fixed but follows a share
of the anode voltage through the output transformer tap. The
instantaneous Pig2 under drive can substantially exceed the static one.

---

### Ia/Ig2 (current ratio)

**Ia/Ig2** — the dimensionless ratio of anode current to screen grid
current. Pentodes only.

Shows what share of the cathode's electron stream reaches the anode
(useful work) versus being intercepted by the screen grid (loss).

**Ia/Ig2 vs Ua**:
- At low Ua (the knee region): the ratio drops sharply — a large share
  of electrons never reaches the anode.
- At normal Ua: the ratio is stable and high (typically 5–15 for output
  pentodes).

**Ia/Ig2 vs Ug1**:
- At deep bias (low current) the ratio may be unstable.
- In the operating region the ratio is fairly constant.

**Criteria:**

| Ia/Ig2 | Verdict | Comment |
|--------|---------|---------|
| > 10 | Excellent | Minimal screen grid losses |
| 5–10 | Normal | Typical for output pentodes |
| 3–5 | Caution | Elevated grid current, check Pig2 |
| < 3 | Dangerous | High risk of screen grid overheating |

**Uses:**
1. **Operating point choice**: avoid zones with ratio < 5.
2. **Diagnostics**: an abnormally low ratio at normal Ua may indicate
   electrode deformation or inter-electrode leakage.
3. **Sample comparison**: the same ratio under the same conditions —
   a good pair.

---

## Overlays

Overlays are drawn on top of the 2D Ia(Ua) plot and help visually judge
the safety and linearity of the operating point.

### Pa max hyperbola

The curve Ia = Pa_max × 1000 / Ua. The region above the hyperbola is
shaded red — the operating point must not be inside it.

On the **Pa map** the same limit is shown as the Pa = Pa_max isoline
(red dashes over the heatmap): everything “hotter” than the isoline
exceeds the tube's rating. On the **Gm map** the analogous S = S nom
isoline (the datasheet transconductance from the tube card) shows the
locus of regimes where the tube reaches its nominal transconductance.

Hover tooltips of the Rp and µ maps show the **computed** value even
when the colour scale is clipped for contrast (clip p95×1.5 / p97×1.3 —
display only).

### Ua max line

A vertical line at the maximum allowed anode voltage. The region to the
right is shaded red.

### Ia max line

A horizontal line at the maximum allowed anode current. The region above
is shaded red.

### Load line

The **Load line** checkbox draws the working line of **the circuit
selected on the Amplifier panel** and recomputes it on the fly (with a
debounce) when Ub, Ra, Ug1, Swing or the circuit type change — no need
to press Analyze for that.

The base case (SE resistive) is the straight line
**Ia = (Ub − Ua) / Ra**, where:
- **Ub** — supply voltage (V)
- **Ra** — anode load (kΩ)

The other circuits:

| Circuit | What is drawn |
|---|---|
| SE transformer | The AC line through the operating point (the anode swings above Ub) plus a separate DC line for the winding resistance |
| Cathode follower | A line set by Rk/Rl |
| Push-Pull | A polyline with the class-AB kink — see below |

**Push-Pull** — the line is not straight but kinked by class AB
(near Q each tube sees Zaa/2, after the partner cuts off — Zaa/4):
- **without a model** — the analytic polyline (kink at 2·Iq);
- **with a model** (model source or a series fit; matched pair) — the
  exact **joint-solve trajectory** regardless of the selected HD method:
  class-A curvature and the true partner-cutoff point; the intersection
  diamonds and swing markers sit on it, and the **A→AB** point (magenta)
  marks the stage leaving class A. Which of the two lines is drawn is
  visible from the info line note (“joint-solve (model)” /
  “kinked Za-a/2→Za-a/4 (analytic)”).

Marked on the line:
- **Q-point** (operating point): the intersection with the curve at the
  given Ug1.
- **Swing range**: from Ug1 − half_swing to Ug1 + half_swing.
- **Amplitude markers**: show how symmetrically the signal clips.

**The info line under the plot** accompanies the line: the circuit and
line-source label, Q (Ua/Ia), THD with HD2/HD3 **and the computation
method note**, Pout. Failures and caveats go there too, not only to the
log:
- `⚠ intersections at FIXED Ug2` — UL mode is on but no model is
  available for it, and the intersections were taken at fixed Ug2;
- `Working line: no data` — not enough data to build the line;
- `Working line: select a Ug2 level` — a pentode model with no screen
  voltage specified.

### Model overlay

A fitted tube model (any of the three fitters — Koren, Dempwolf,
Reefman) is added to the 2D plot as computed (dashed) curves for
comparison with the measured data. The legend entry carries the model
type and the fit's RMS deviation in mA.

---

## Compare — Tube comparison

### The main plot

An overlay of the anode characteristics of several samples of the same
tube type. Lets you visually judge the spread and select pairs/quads.

### Legend (decoupled semantics)

The two independent encoding axes — colour and line style — have
separate legend entries:

- **Tubes** — one entry per checked record, drawn as a solid line in
  **its colour** (the colour from the table's Color column). Two records
  with the same name (two runs of one tube) get numbered suffixes
  “Name (2)” — every colour on the plot stays named.
- **Ug2 levels** — in neutral grey with the level's **line style**
  (solid/dash/dot…). Shown only with ≥2 levels and only for levels
  enabled in the filter panel — the legend does not advertise
  filtered-out curves.

Measurements with close Ug2 (within the clustering threshold, 2 V by
default — e.g. 250.0 and 251.5 V) fall into one clustered level — they
are drawn, filtered and styled as a single Ug2 level.

### Matching Delta

The dialog shows the ΔIa difference (in mA or %) between two
measurements as a function of Ua. An ideal pair has ΔIa ≈ 0 across the
whole range.

### Aging Trend

Tracking a tube's parameters over time: Ia at the operating point and
transconductance S across several measurements. Lets you estimate
cathode degradation (emission loss) and predict the remaining lifetime.

---

## Amplifier — Amplifier analysis

### THD vs Amplitude

Harmonic distortion (HD2, HD3, THD) and output power versus input signal
amplitude.

**How to read it:**
- THD < 1% is the usual Hi-Fi target.
- The intersection of the THD curve with the 1% line gives the maximum
  undistorted amplitude.
- HD2 >> HD3 — typical of an SE triode (soft clipping, quadratic
  characteristic).
- Noticeable HD3 — typical of an SE pentode (sharper clipping on both
  half-waves). In push-pull HD2 cancels and HD3 dominates for both tube
  types.

### Distortion vs Ra

THD and power versus load resistance at fixed amplitude. Helps choose
the optimal Ra:
- Small Ra → more power but more distortion.
- Large Ra → less distortion but less power.

The usual compromise sits near Ra ≈ 2×Rp for triodes. For pentodes Ra is
always far below Rp — typically Ra ≈ Ua_q/Ia_q (the operating point's
voltage-to-current ratio), which is Rp/5…Rp/10 for output pentodes.

Checkboxes add more curves to the same plot: HD4/HD5, Gain/Zout/Pa.
The min-THD and max-Pout markers show the optimum points.

### Pareto (THD vs Pout)

Replaces the left plot after the Optimizer runs. Shows the search space
(all admissible Ug1 × Ra × Swing combinations) and the Pareto front:
- Grey dots — all valid grid points.
- Green dashes — the grid Pareto front.
- Cyan line — the refined Pareto front (after scipy).
- Yellow/cyan star — the best point (grid/refined).

Clicking a Pareto point applies Ub, Ra, Ug1, Swing, Ug2 and the UL tap
(including resetting it to 0 when the point was computed without a tap).
Hovering shows a tooltip with the point's full parameters.

---

## Practical scenarios

### Scenario 1: Checking a new tube

1. Record a characteristic (Scan).
2. **2D**: compare the curve shapes against the datasheet.
3. **Gm/Rp maps**: confirm that Gm at the operating point matches the
   nominal (±20%).
4. **Curves → Gm**: check against the S nominal reference line.
5. **Ig2** (pentodes): confirm there is no abnormally high screen
   current.

### Scenario 2: Matching a pair for Push-Pull

1. Record characteristics of both tubes.
2. **Compare**: overlay the curves, judge the match visually.
3. **Matching Delta**: quantify ΔIa.
4. **Curves → Gm**: compare Gm at the operating point.
5. Target criteria: ΔIa < 5%, Gm difference < 10%.

### Scenario 3: Designing a stage

1. Record the working tube's characteristic.
2. **2D + Load Line**: set Ub and Ra, find the Q-point.
3. **Curves → Gm, Rp, mu** at the operating point: note the parameters.
4. **Amplifier**: estimate THD and Pout, tune Ra.
5. **Curves → Pa**: confirm Pa < Pa_max at the operating point.
6. **Pig2** (pentodes): confirm Pig2 < Pig2_max.

### Scenario 4: Diagnosing a pentode

1. **Curves → Ia/Ig2**: check the current ratio across the operating
   region. Ratio < 3 — the tube is in the danger zone or faulty.
2. **Curves → Pig2 vs Ua**: find the minimum safe Ua (where
   Pig2 < Pig2_max).
3. **Curves → Ig2 vs Ua**: check the behaviour in the knee region.
4. **Curves → Ia vs Ug2**: judge the dependence on screen voltage.

### Scenario 5: Wear monitoring

1. Periodically record the same tube's characteristic.
2. **Aging Trend**: track Ia and S over time.
3. **Curves → Gm**: compare the current transconductance with the
   original.
4. A Gm drop of > 30% from the initial value — the tube needs replacing.

---

## Diagnosing faults from the plots

Visual analysis of the characteristics is a powerful diagnostic tool.
Below are the typical defects and how they show on the LM19 plots.

### Cathode emission loss

The most common defect of old tubes. The cathode's oxide coating
gradually depletes and the cathode emits fewer electrons.

**On 2D (Ia vs Ua):**
- All curves “sag” downward — currents are below the datasheet values at
  the same Ua and Ug1.
- The curve shapes persist, but the family “compresses” towards the
  X axis.
- Cutoff occurs at a smaller \|Ug1\| than in a new tube.

**On Curves → Gm:**
- Gm is reduced across the whole region. If the datasheet Gm is
  5.5 mA/V and the measured is 3.5 mA/V — about 36% of emission is lost.
- The S nominal reference line makes the deviation obvious.

**On Transfer:**
- The curve is shifted right and down: less current at the same Ug1,
  cutoff comes earlier.

**Criteria:**

| Gm loss | Verdict | Recommendation |
|---------|---------|----------------|
| < 10% | New/excellent | No restrictions |
| 10–20% | Good | Fit for most uses |
| 20–30% | Worn | Works, with degraded parameters |
| > 30% | Unfit | Replacement recommended |

### Gas in the envelope

When the vacuum degrades, residual gas is ionised by the electron
stream. Gas ions create an extra current to the grid and anode.

**On 2D (Ia vs Ua):**
- At high Ua the curves bend upward — the current grows faster than the
  normal characteristic predicts.
- The effect intensifies with Ua, as ionisation increases at higher
  electron energy.
- In severe cases the curves “take off” upward at Ua > 200–300 V.

**On Curves → Rp:**
- Rp is abnormally low at high Ua — the ion current “shunts” the normal
  electron flow.

**On Curves → Gm:**
- Gm may look normal or even elevated at high Ua due to the extra ion
  current.

**Distinguishing from normal wear:** with normal ageing the curves sag
downward uniformly. Gas produces the characteristic upward bend
specifically at high voltages.

### Control grid emission

An overheated grid begins emitting electrons itself. Grid current flows
even at zero or slightly negative Ug1.

**On 2D (Ia vs Ua):**
- The curves at small \|Ug1\| (near 0 V) sit above normal — the grid
  current adds to the anode current.
- The effect disappears at large \|Ug1\|, where the grid field
  suppresses the emission.

**On Transfer:**
- At Ug1 = 0 the current is noticeably above expectation.
- The curve may have a “hump” near Ug1 = 0.

**Consequences:** grid current loads the bias source. In stages with a
large grid leak resistance (Rg > 1 MΩ) this can shift the operating
point and cause thermal runaway.

### Cathode interface (interface resistance)

A high-resistance layer forms between the cathode's metal base and its
oxide coating. Current through this layer produces a voltage drop that
acts as extra negative bias.

**On 2D (Ia vs Ua):**
- The curves at high currents “sag” — the effective Ug1 becomes more
  negative due to the drop across the interface layer.
- At low currents (small Ia) the effect is minimal — the curves are
  normal.
- The telltale sign: the curves “bunch up” at high currents, their
  spacing shrinking.

**On Curves → Gm:**
- Gm falls at high currents instead of growing or stabilising. This is
  anomalous — in a normal tube Gm grows with current.

**On Transfer:**
- The curve deviates downward from the expected power law at high
  currents — “compression from above”.

**Distinguishing from emission loss:** with emission loss all curves are
lowered uniformly. With cathode interface the curves are normal at low
currents and deviate only at high ones.

### Inter-electrode leakage

Contamination or metal deposits on the insulators create a resistive
path between electrodes.

**On 2D (Ia vs Ua):**
- Grid–cathode leakage: all curves are shifted up by a constant amount —
  a current independent of Ua is added.
- Anode–grid leakage: the current grows linearly with Ua even in full
  cutoff (when Ia should be 0).

**On Curves → Ia/Ig2 (pentodes):**
- Anode–screen leakage: the ratio is abnormal and may be unstable.

**The telltale sign:** the current does not reach zero in deep cutoff.
A normal tube has Ia = 0 at a sufficiently large \|Ug1\|.

### Electrode deformation (shorts, partial shorts)

Mechanical damage or thermal deformation brings electrodes closer
together, changing the geometry and the electric fields.

**On 2D (Ia vs Ua):**
- Asymmetry of the curve family: one part of the characteristics is
  normal, another deviates strongly.
- With a full short — the current is not controlled by the grid.

**On Curves → Gm:**
- Sharp jumps or “teeth” on the Gm curve — Gm is unstable because of the
  varying electrode spacing.

**On the Gm/Rp map:**
- “Spots” of abnormally high or low Gm — nonuniformity of the electric
  field.

### Microphonics

Electrode vibration modulates the current. Not visible on static
characteristics (the tester acquires data slowly), but may show as
elevated measurement “noise”.

**On 2D and Curves:**
- Points do not lie on a smooth curve — scatter on repeated measurements
  under identical conditions.
- The Gm/Rp map looks “blotchy” with no discernible pattern.

**Note:** a reliable microphonics diagnosis needs a dynamic test
(inject a signal and listen); static measurements give only indirect
evidence.

### Diagnostics summary table

| Defect | 2D (Ia vs Ua) | Gm | Transfer | Ig2 / Ia/Ig2 |
|--------|---------------|-----|----------|--------------|
| Emission loss | Curves ↓ uniformly | ↓ everywhere | Shift ↓→ | Ratio stable |
| Gas | Upward bend at high Ua | Normal or ↑ | Normal | — |
| Grid emission | ↑ at small \|Ug1\| | ↑ at small \|Ug1\| | Hump near Ug1=0 | — |
| Cathode interface | Bunching at high Ia | ↓ at high Ia | Compression from above | Ratio stable |
| Leakage | Offset, current ≠ 0 in cutoff | Offset | Offset | Ratio unstable |
| Deformation | Asymmetry, jumps | Jumps, “teeth” | Kink | Ratio abnormal |
| Microphonics | Point scatter | Blotchy map | Scatter | — |

---

## Choosing the operating point: a step-by-step guide

Choosing the operating point is the key stage of tube circuit design.
Below is a systematic approach using the LM19 plots.

### General principles

The operating point (Q-point) is defined by three quantities:
- **Ua_q** — anode voltage at idle
- **Ia_q** — anode current at idle
- **Ug1_q** — bias voltage

They must satisfy the constraints:
1. **Pa_q = Ua_q × Ia_q / 1000 ≤ Pa_max** — stay within the thermal
   limit
2. **Ua_q ≤ Ua_max** — stay within the voltage limit
3. **Ia_q ≤ Ia_max** — stay within the current limit
4. **Sufficient distance from the knee** — to ensure linearity
5. **Sufficient distance from cutoff** — to leave headroom for signal
   amplitude

### Example 1: SE triode (12AX7, driver)

Datasheet values: μ = 100, Gm = 1.6 mA/V, Rp = 62.5 kΩ, Pa_max = 1.2 W.

**Step 1. Establish the limits.**
- Open the **2D** plot, enable the Pa max = 1.2 W and Ua max = 300 V
  overlays.
- The whole safe region is visible as the unshaded zone.

**Step 2. Choose the supply voltage and the load.**
- A typical regime: Ub = 250 V, Ra = 100 kΩ (high Ra for maximum gain,
  since A = μ × Ra / (Ra + Rp) = 100 × 100 / 162.5 ≈ 61.5).
- Enable **Load Line**: Ub = 250, Ra = 100.

**Step 3. Find the Q-point on the load line.**
- The load line crosses the curves at various Ug1.
- Set Ug1 = −2 V. Q-point: Ua_q ≈ 175 V, Ia_q ≈ 0.75 mA.
- Pa_q = 175 × 0.75 / 1000 = 0.13 W — far from the 1.2 W limit.

**Step 4. Check linearity.**
- On **2D**: visually judge the uniformity of curve spacing along the
  load line near the Q-point.
- On **Curves → Gm vs Ug1**: Gm should be stable within
  Ug1_q ± the expected amplitude (e.g. ±1 V).
- On **Transfer**: the curve should be roughly linear in the working
  region.

**Step 5. Estimate distortion.**
- On the **Amplifier** tab set half_swing = 1 V.
- Read HD2, HD3, THD from the THD vs Amplitude plot.
- A typical result for a 12AX7 at small signal: THD < 1%.

**Step 6. Record the parameters.**
- On **Curves** read Gm, Rp, μ at Ua ≈ 175 V, Ug1 ≈ −2 V.
- Compare with the datasheet: Gm ≈ 1.6 mA/V, Rp ≈ 62 kΩ, μ ≈ 100.

### Example 2: SE pentode (EL34, output stage)

Datasheet values: Gm = 11 mA/V, Rp = 15 kΩ, Pa_max = 25 W,
Pig2_max = 8 W, Ua_max = 800 V, Ug2 = 250 V (typical regime).

> **Important: transformer load.**
> Output stages use a transformer, not a resistor. The primary winding's
> DC resistance is small, so at idle Ua_q ≈ Ub (the supply voltage).
> “Ra” is the reflected load resistance and only defines the AC
> behaviour. For the LM19 load line to pass through the Q-point
> correctly, set **Ub_eff = Ua_q + Ia_q × Ra**, not the real Ub.

**Step 1. Establish the limits.**
- On **2D**: enable Pa max = 25 W. The hyperbola shows the thermal
  limit.
- Enable Ua max and Ia max (if applicable).

**Step 2. Determine the Q-point.**
- Supply: Ub = 400 V. With a transformer: Ua_q ≈ 380 V (minus the drop
  on the cathode resistor ≈ 20 V).
- At Ug2 = 250 V, Ug1 ≈ −22 V (from the datasheet): Ia_q ≈ 48 mA.
- Pa_q = 380 × 48 / 1000 ≈ 18.2 W — headroom below 25 W.

**Step 3. Set the load line.**
- Typical (reflected) Ra for SE EL34: 3.5 kΩ.
- For LM19: Ub_eff = 380 + 48 × 3.5 = 548 V.
  Set **Load Line**: Ub = 548, Ra = 3.5, Ug1 = −22.
- The load line now passes through the Q-point (380 V, 48 mA) and shows
  the real AC signal path.

**Step 4. Check the screen grid power.**
- On **Curves → Pig2 vs Ua**: find Pig2 at the operating point.
- At Ua_q = 380 V, Ug2 = 250 V: Pig2 = 250 × Ig2 / 1000. A typical
  Ig2 ≈ 10 mA → Pig2 ≈ 2.5 W (headroom below 8 W).
- **Critical**: as the signal swings towards low Ua (the knee region)
  Ig2 rises sharply. On the **Pig2 vs Ua** plot find the minimum Ua at
  which Pig2 is still < Pig2_max.

**Step 5. Check Ia/Ig2.**
- On **Curves → Ia/Ig2 vs Ua**: the ratio should be > 5 in the operating
  region.
- If at the lowest Ua of the dynamic range the ratio is < 3 — reduce the
  amplitude or raise Ub.

**Step 6. Judge linearity and distortion.**
- On **Curves → Gm vs Ua**: Gm should be stable from the minimum of the
  dynamic swing up to Ua_q + half_swing.
- On **Amplifier**: estimate THD. For a pentode SE, THD of 5–10% at full
  power is normal (feedback will bring it down to 1–2%).
- On **Amplifier → Distortion vs Ra**: pick the optimal Ra — the
  power-versus-distortion compromise.

**Step 7. Record the parameters.**
- Gm, Rp at the Q-point (from **Curves**).
- Pout, THD at the design amplitude (from **Amplifier**).
- Pa_q, Pig2 (from **Curves → Pa, Pig2**).

### Example 3: Push-Pull (EL34 pair)

**Step 1. Match the pair.**
- Record both tubes' characteristics under identical conditions.
- On **Compare**: overlay the curves. Target: visual coincidence.
- **Matching Delta**: ΔIa < 5% in the operating region.

**Step 2. Determine the class-AB Q-point.**
- Class AB: Ia_q is lower than for class A. Typically Ia_q ≈ 30–40 mA
  per tube for EL34.
- Ua_q ≈ Ub (transformer coupling). At Ub = 430 V, Ia = 35 mA:
  Pa_q = 430 × 35 / 1000 = 15 W per tube (headroom below 25 W).

**Step 3. Check the pair's symmetry.**
- On **Curves → Gm**: both tubes should have the same Gm at the
  operating point. Difference < 10%.
- On **Transfer**: both tubes' curves should coincide in shape.
  A shape difference (different curvature) is worse than a plain Ia
  offset — it yields incomplete HD2 cancellation.

**Step 4. Estimate distortion.**
- On **Amplifier** (Push-Pull mode, Matched pair): HD2 should be
  strongly suppressed (even-harmonic cancellation in PP). HD3 is the
  main residual distortion component.
- Ra_aa (anode-to-anode) is typically 3–6 kΩ for an EL34 pair.

### Operating point checklist

| Check | Plot | Criterion |
|-------|------|-----------|
| Pa_q < Pa_max | Curves → Pa or 2D + Pa hyperbola | Headroom ≥ 20% |
| Pig2 < Pig2_max | Curves → Pig2 | Headroom ≥ 30% (dynamics included) |
| Ua_q in the safe zone | 2D + Ua max | Well below the limit |
| Gm close to datasheet | Curves → Gm + S nominal | Deviation < 20% |
| Gm stable in the working region | Curves → Gm | Flat curve around the Q-point |
| Ia/Ig2 sufficient | Curves → Ia/Ig2 | > 5 across the dynamic range |
| THD acceptable | Amplifier → THD vs Amplitude | Below target at the required power |
| Amplitude headroom | 2D + Load Line + swing | Symmetric swing without clipping |
