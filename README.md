# GT3 suspension kinematics workbench

An interactive tool for designing and interrogating double-wishbone suspension
geometry, written from first principles and backed by an executable validation
suite of **793 assertions**.

It solves the real 3D linkage. Nothing in it relies on the side-view or
front-view instant-centre constructions that textbooks draw, because those are
projections of a 3D mechanism and are both ill-conditioned — the instant centre
runs off to infinity for perfectly ordinary geometry — and slightly wrong, since
the projection of a 3D rigid motion is not itself a rigid planar motion.

---

## What it does

Load a set of hard points, move one, and watch every derived quantity move with
it, live, on the real mechanism.

| area | what you get |
|---|---|
| **Steering axis** | KPI, caster, scrub radius, mechanical trail, and how each moves with steer and with travel |
| **Camber** | camber vs bump, vs roll, vs steer; camber recovery; the front-view swing arm behind it |
| **Roll centre** | height at ride, height through travel, and lateral migration in roll — the number that actually matters |
| **Bump steer** | toe vs travel, with the slope and the curvature separated |
| **Ackermann** | the axle-level percentage, on the real steer geometry |
| **Anti-geometry** | anti-dive, anti-lift and anti-squat, with the front-view coupling made visible |
| **Springs** | motion ratio solved from the linkage, coil rate, ride frequency on both of its definitions |
| **Anti-roll bars** | blade rates, the drop-link motion ratio solved from the geometry, roll-stiffness distribution |
| **Mechanical balance** | lateral load-transfer distribution and roll-stiffness maps |
| **Geometry 2D / 3D** | front, side and top views with the construction lines drawn on them |

---

## Quick start

```bash
pip install -r requirements.txt
python software/explorer.py
```

Hard points come from `hardpoints_fusion.csv` at the repository root — a plain
`Corner,Point,X,Y,Z` table exported from CAD. Edit it, or move points live in
the tool and save a new one.

Run the validation suite:

```bash
python software/validate_all.py
```

---

## Validation

The suite is the point, not an afterthought. It runs 17 independent files and
asserts physics rather than snapshots: identities that must hold for any
geometry, sign conventions, limiting cases, and agreement between methods that
were derived separately.

```
scrub_trail 33 · wishbone 14 · geom_edit 16 · axle 26 · explorer 91
rear_geometry 19 · rear_bump 14 · rear_roll 31 · load_transfer 50
conventions 29 · ackermann 34 · springs 94 · arb 102 · roll_axis 38
anti_geometry 85 · mech_balance 75 · hardpoint_audit 42
--------------------------------------------- 793 pass, 0 fail, 17 suites
```

A sample of what is actually checked:

- a rigid tyre returns the wheel rate exactly; two equal rates in series halve
- the half-step anti-roll-bar identity is the **harmonic** mean, derived rather
  than assumed, and it beats the arithmetic mean as any series path must
- `K_s = K_w / R_M²` and `K_s = K_w · MR²` agree to machine precision in both
  motion-ratio conventions, and an unknown convention raises rather than being
  silently wrong by a factor of `R_M⁴`
- with both side-view hinge axes parallel at angle *t*, the anti-geometry
  coefficient is `tan t` exactly
- a percentage field never holds a length — written after a bug put a 1404 mm
  kingpin ground track into a field labelled `ackermann_pct`

---

## Some design decisions worth reading the code for

**Anti-geometry without instant centres.** The corner has one degree of freedom,
so apply a force at the contact patch and ask when the spring feels nothing:
`F_y·dP_y + F_z·dP_z = 0`. Everything reduces to one quantity — the fore-aft
motion of a reference point per mm of wheel travel. Exact on the real 3D
mechanism, no singularity, and it collapses to the textbook answer wherever the
textbook answer is valid. See `software/anti_geometry.py`.

**The reference point depends on what reacts the torque, not on what the force
is.** Outboard brakes react on the upright, so the reference is the contact
patch. An inboard differential reacts on the chassis, so the reference is the
wheel centre. A GT3 has both, which means the rear axle has two different anti
numbers from one piece of geometry, differing by exactly one wheel radius of
lever.

**"Ride frequency" means two different things**, and the published benchmark
bands belong to only one of them. The wheel-rate definition leaves the tyre out
entirely; the ride-rate definition puts it in series. On a racing car the wheel
rate is about half the tyre rate, so the tyre is worth roughly 15 % of the
frequency and the two numbers are not interchangeable. Every function returns
both. See `software/springs.py`.

**Motion-ratio convention is a single switch.** Internally `R_M` is spring
travel over wheel travel, because that is what the solver produces. Everything
shown to a human uses the textbook `MR`, wheel travel over spring travel.
`software/springs.py` is the only place the two are ever converted, because
mixing them is an `R_M⁴` error.

**A damper bolted to the wrong arm is a factor of 2.9 on spring rate.** The
solver picks the member the lower mount actually rides on rather than assuming
the lower wishbone, because the motion ratio enters the rate squared.

---

## Layout

```
hardpoints_fusion.csv        default CAD hard points, both corners
software/
  wishbone_solver.py         the 3D linkage solve everything else sits on
  scrub_trail.py             steering axis, contact patch, scrub, trail
  rear_geometry.py           rear corner; the toe link as a fixed tie rod
  anti_geometry.py           anti-dive / anti-lift / anti-squat
  roll_axis.py               roll centres and the roll axis
  springs.py                 motion ratio, coil rate, ride frequency
  arb.py                     anti-roll bars, blades, drop-link ratio
  mech_balance.py            lateral load-transfer distribution
  load_transfer.py           the load-transfer model
  ackermann.py  steer_camber.py  jacking.py  tyre.py  rim_envelope.py
  explorer.py  viewer.py     the application and its 2D / 3D views
  validate_*.py              the suite
  benchmarks.py              optional reference data, see below
suspension_tool/             a validated two-corner axle model, reused by the
                             roll-centre-in-roll view
```

---

## Reference data

This project was developed partly against a manufacturer's performance handbook
and a tyre supplier's data booklet. Both are confidential documents, and **no
figure from either is in this repository.**

Where the code wants a reference car or a measured tyre it asks
`software/benchmarks.py`, which looks for a private module on `GT3_PRIVATE_DIR`
— a folder deliberately kept outside this tree, so publishing the repository
cannot leak anything even if a `.gitignore` rule is edited by mistake.

Without private data installed:

- reference overlays are hidden, and the checks that compare against them are
  skipped; everything else runs identically
- the anti-geometry target falls back to the open literature — Gillespie's 50 %
  practical anti-dive ceiling, and Adams' 5–30 % range with a 25 % limit for an
  independent rear suspension
- the roll-axis reference falls back to the ground-level axis, the classical
  neutral case
- the tyre falls back to a clearly-labelled linear placeholder carcass

`software/benchmarks_private.example.py` documents the shape so anyone can plug
in data they are entitled to use.

---

## Requirements

Python 3.10+, NumPy, Matplotlib, Tkinter. SciPy is optional — one validation
section uses it for a least-squares fit and is skipped without it.

---

## Licence

MIT. See [LICENSE](LICENSE).

The vehicle this geometry was developed for is a real car. The hard points here
are an independent reconstruction, not manufacturer data.
