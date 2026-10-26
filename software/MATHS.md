# The maths behind camber-by-steer  (double-validation notes)

Everything the tools compute comes from the five steps below. Read this next to
`steer_camber.py` — the code is a line-for-line copy of these equations.

Frame (right front wheel): **x = forward, y = inboard (toward car centre), z = up.**
Angles: **λ = KPI**, **ν = caster**, **δ = road-wheel steer**.

---

## 1. The kingpin (steering) axis from KPI and caster
The axis is the line LBJ → UBJ. Its unit vector, pointing up:

```
u_raw = ( −tan ν , +tan λ , 1 )
u     = u_raw / |u_raw| ,     |u_raw| = √(1 + tan²ν + tan²λ)
```

* `−tan ν` (x): +caster leans the top **rearward** (−x).
* `+tan λ` (y): +KPI leans the top **inboard** (+y).
* `1` (z): predominantly vertical.

## 2. Wheel spin axis and the definition of camber
Spin axis at zero steer / zero camber points straight outboard:

```
s0 = (0, −1, 0)
```

Camber = tilt of the spin axis out of the horizontal plane:

```
γ = arcsin(−s_z)     (right wheel: top-out/positive camber ⇒ s_z < 0)
```

## 3. Steering rotates the wheel about u  (Rodrigues)
```
s(δ) = s0·cos δ + (u × s0)·sin δ + u·(u·s0)·(1 − cos δ)
```
with
```
u·s0  = −u_y
u × s0 = (u_z, 0, −u_x)
```
so only the vertical component is needed:
```
s_z(δ) = −u_x·sin δ − u_z·u_y·(1 − cos δ)
Δγ(δ)  = arcsin(−s_z)        # γ(0)=0, so this IS the camber change
```
This is **exact** for any δ (that is why it holds across ±45°). It is the whole
of `camber_change()`.

## 4. Splitting into a caster term and a KPI term (the textbook bridge)
For small camber `arcsin(x) ≈ x`, insert `u_x=−tanν/|u|`, `u_y=tanλ/|u|`, `u_z=1/|u|`:

```
Δγ ≈ (tan ν / |u|)·(−sin δ)  +  (tan λ / |u|²)·(1 − cos δ)
```

and for modest angles (`tan≈sin`, `|u|≈1`):

```
Δγ ≈ −sin ν · sin δ  +  sin λ · (1 − cos δ)
       └─ caster ─┘       └──── KPI ────┘
```

* **caster**  `−sin ν·sin δ`  → **odd** in δ (flips with steer direction), **linear**.
  The dominant, FAVOURABLE camber lever (negative = leans into corner).
* **KPI**  `+sin λ·(1 − cos δ) ≈ ½·sin λ·δ²`  → **even** (same sign both ways),
  **second-order**. Small, UNFAVOURABLE, only matters at large steer.

`camber_change_approx()` is this formula. It matches the exact function to
<0.01° at stock — that agreement is the double-check.

## 5. Rack travel → steer angle
Steering arm = perpendicular distance from the tie-rod OUTER ball joint to the
kingpin axis:
```
L = |w × u| / |u| ,   w = (tie-rod outer − LBJ) ,  u = (UBJ − LBJ)
δ ≈ rack / L        (radians; tie rod ≈ perpendicular to the arm)
```
**Corrected by the wishbone solver:** the perpendicular arm (148.6 mm) is NOT the
effective ratio — the tie rod is oblique to the steering arm, so `δ ≈ rack/L⊥`
understates the steer by ~30%. The exact velocity relation
`dδ/d(rack) = (x̂·t̂)/((k̂×r)·t̂)` gives an **effective arm of 114.1 mm**, hence
`20 mm rack ≈ 10.04°` first-order (10.7° exact at the finite travel). Verified
against the full wishbone solver to 1e-5 rad/mm.

---

## 6. Worked example — run it
`python steer_camber.py` prints the whole calculation step by step, with every
intermediate number, for the baseline geometry (KPI −5.27°, caster +1.66°, steer 10°).
The STEP A/B/C/D labels match the comments inside `camber_change()` and section A.6
of the report appendix. To try another case:

```python
from steer_camber import worked_example
worked_example(kpi_deg=7.0, caster_deg=8.0, steer_deg=20.0)
```

It prints: STEP A (build `u`), STEP B (the trig), STEP C (the caster and KPI terms
of `s_z`), STEP D (back to an angle), then cross-checks against the classical
formula, the superposition of the two effects, and how they grow with steer.

## 6.1 The KPI sign, and why it decides the whole story
**Corrected 2026-07-16.** An earlier version of this note claimed the baseline's camber
compensation "collapses at 36°". That was wrong: it came from a hard-coded
`STOCK_KPI = +5.3` in `config.py` while `scrub_trail.py` read the CAD, which says the
axis leans the *other* way. The two halves of the project disagreed about the sign.
The angles are now **derived from the hard points** in one place (`config.stock_geometry_from_csv`)
and asserted equal across modules by `validate_scrub_trail.py`, so it cannot recur.

The CAD puts the UBJ **13.4 mm outboard** of the LBJ (confirmed against the front view),
so this car's **KPI is negative: −5.27°**. Sign convention: KPI > 0 leans *inboard*.

Because the KPI term is `+sin λ (1 − cos δ)`, a **negative** λ makes it negative too —
so it *adds* favourable camber instead of fighting the caster:

| steer δ | caster term | KPI term | total Δγ |
|---|---|---|---|
| 5° | −0.145° | −0.020° | **−0.164°** |
| 10° | −0.288° | −0.080° | **−0.366°** |
| 20° | −0.567° | −0.316° | **−0.881°** |
| 35° | −0.951° | −0.947° | **−1.894°** |

**No collapse, at any caster.** By 35° the negative KPI is contributing as much
favourable camber as the caster is. What the sign is worth:

| | CAD as-measured (KPI −5.27°) | if KPI were +5.27° |
|---|---|---|
| scrub radius | **144.9 mm** | **98.6 mm** |
| Δγ @ 35° | −1.894° | −0.001° |
| collapses? | **never** | at 35° |

### The lesson (and why real cars still run positive KPI)
Negative KPI *flatters* the camber curve — and costs **46.3 mm of extra scrub radius**,
with all the kickback, braking sensitivity and steering effort that brings. That is a
bad trade, and it is exactly why real GT3s run **positive** KPI and simply accept the
small camber penalty (a competitive GT3 runs low double-digit KPI). This car's geometry is
an artistic mesh, not a design.

So the baseline's real problems are **not** camber-by-steer. They are:
1. **scrub radius 144.9 mm** — enormous (and confirmed by the CAD sketch's own 146.213 mm dimension);
2. **mechanical trail reverses beyond +18° of steer** — the steering stops self-centring;
3. **caster only 1.66°** — so almost none of the camber compensation is coming from the
   knob you would actually want to use for it.

## Verification numbers (printed by `python steer_camber.py`)
| case | δ | exact | small-angle | note |
|---|---|---|---|---|
| caster only ν=5, λ=0 | +20° | −1.708° | −1.708° | favourable, linear |
| KPI only λ=8, ν=0 | +20° | +0.476° | +0.481° | unfavourable, 2nd-order |
| caster odd | ±20° | ∓1.708° | — | flips sign with steer |
| KPI even | ±20° | +0.476° | — | same sign both ways |

## 7. Scrub radius & mechanical trail
Covered by `scrub_trail.py` and `Study Guide/Unit 2 - Scrub & Mechanical Trail -
WORKED EXAMPLE.pdf`. Two things there that bite, both learned the hard way:

* **The contact patch is NOT the wheel centre projected down.** The wheel is cambered,
  so it is tilted and touches down `R·sin(γ)` to one side (`R = WC_z/cos γ`). At the
  baseline's −2.7° that is **15.9 mm**, and since scrub is defined *to the patch*,
  ignoring it understated the scrub radius by exactly that: **129.1 → 144.9 mm**.
  Confirmed independently: the user's own CAD front-view sketch dimensions that same
  distance as **146.213 mm** (1.3 mm = the mesh's asymmetry).
* **`d(half-track)/dδ = trail·u_z` is exact; the rest are approximations.** The forms
  `d(wheelbase)/dδ ≈ scrub + R·sin λ` and `d(half-track)/dδ ≈ trail − R·sin ν` assume
  the patch is the projected centre — but steering *induces* camber, which shifts it
  (~9.7 mm/rad of error). Only the half-track identity survives, and it simplifies:
  the `R·sin ν` term cancels.

## How to validate in CAD (Fusion)
1. Place the ball joints so the LBJ→UBJ axis has a chosen KPI & caster
   (front view angle = KPI, side view angle = caster).
2. Rotate the upright + wheel about that axis by δ.
3. Measure the wheel plane's inclination (camber) before and after — the change
   should equal the interactive readout at that (KPI, caster, δ).

Provisional-box sanity (δ = 10.04°): KPI 7°, caster 8° → **Δγ = −1.28°**.
(That box is a placeholder — the agreed plan is to derive the window from first
principles once the wishbone solver gives roll-induced camber.)
