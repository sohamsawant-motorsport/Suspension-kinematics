# GT3 suspension kinematics workbench

A desktop tool for designing double-wishbone suspension geometry. Load a set of
hard points, move one, and every derived quantity updates on the real linkage.

It solves the corner in 3D. There is no side-view or front-view instant-centre
construction anywhere in the maths, because those are projections of a 3D
mechanism: the instant centre runs off to infinity for perfectly ordinary
geometry, and the projection of a 3D rigid motion is not itself a rigid planar
motion.

793 assertions run against it. `python software/validate_all.py`.

---

## Install and run

```bash
pip install -r requirements.txt
python software/explorer.py
```

Windows users can double-click `run_tool.bat` instead.

Needs Python 3.10 or later, NumPy, Matplotlib and Tkinter. SciPy is optional:
one validation section uses it and is skipped if it is missing.

---

## Workflow

**1. Point it at a set of hard points.**

`hardpoints_fusion.csv` in the repository root is what loads on startup. It is
a plain table:

```
Corner,Point,X_mm,Y_mm,Z_mm
FRONT,Upper wishbone - outer (UBJ),815.936,1222.414,399.588
...
```

`X` is lateral and positive outboard, `Y` is longitudinal and positive forward,
`Z` is vertical. The file holds the front-right and rear-right corners; the left
side is their mirror.

A second set is in `hardpoints/`, the geometry this project ended up with after
a front and rear redesign. Load it to see a worked-through result instead of the
starting point.

**2. Pick a page.** Three across the top: FRONT, REAR, VEHICLE.

**3. Edit points on the Hard points tab.** Type a coordinate, press Enter. Every
other tab recalculates. The reference geometry stays drawn underneath so a
change is visible as a difference rather than a new picture.

**4. Read the numbers, not just the curves.** Each corner page carries a text
readout with the static values: KPI, caster, scrub, trail, camber gain, camber
recovery, bump steer, roll-centre height, motion ratio, steering arm.

**5. Save.** Writes a CSV in the same format, so a saved set loads straight back.

---

## The tabs

### FRONT and REAR pages

| tab | what it shows | what moves it |
|---|---|---|
| **Camber–KPI**, **Camber–caster** | camber change against steer angle, one curve per KPI or caster value | the two ball joints |
| **Camber map** | camber over the KPI–caster plane at a chosen steer angle | - |
| **Patch path** | where the contact patch travels as the wheel steers | scrub and trail |
| **Scrub & trail** | both against steer, with the sign convention drawn | sliding the ball joints along the steering axis |
| **Wheelbase & track** | how much each one changes with steer | scrub sets the curvature, trail sets the slope |
| **Heave & bump steer** | toe against wheel travel | the tie rod or toe link, after the instant centre is settled |
| **Body roll** | camber against body roll, which is what the tyre actually sees | camber recovery, i.e. FVSA |
| **Roll centre** | height against wheel travel | common-mode inner pickup height |
| **RC in roll** | the roll centre's lateral migration, usually the number that matters | the ratio of the two arm lengths |
| **KPI/caster in bump** | how the steering axis itself moves through travel | the side-view hinge-axis angles |
| **Motion ratio** | wheel travel over spring travel, through the stroke | where the damper's lower mount sits, and how far the damper leans |
| **Ackermann** | the axle-level percentage on the real steer geometry (front only) | the tie-rod outer, fore and aft |
| **Geometry 2D / 3D** | front, side and top views with the construction lines on them | - |

### VEHICLE page

| tab | what it shows |
|---|---|
| **Load transfer & g-g** | wheel loads under combined lateral and longitudinal acceleration |
| **Springs** | coil rate from a target ride frequency, motion ratio solved from the linkage, both definitions of ride frequency side by side |
| **Anti-roll bars** | bar and blade rates, the drop-link ratio solved from the geometry, roll-stiffness split |
| **Roll axis** | both roll centres and the axis between them, in side view |
| **Anti geometry** | anti-dive, anti-lift and anti-squat, with what the side-view tilt costs in the front view |
| **Mechanical balance** | lateral load-transfer distribution against spring and bar changes |

---

## Things worth knowing before you change a number

**Motion ratio is shown the textbook way.** `MR` is wheel travel over spring
travel, so the coil rate is `K_s = K_w · MR²`. Internally the solver works in
the reciprocal, because that is what it produces directly. `software/springs.py`
is the only place the two meet. Mixing them is an error to the fourth power.

**Ride frequency means two different things.** One definition leaves the tyre
out, the other puts it in series with the spring. On a racing car the wheel rate
is roughly half the tyre rate, so the two numbers differ by about 15 % and the
published target bands belong to only one of them. Every spring function returns
both and the panel prints both.

**Anti-geometry is computed from patch motion, not from a swing arm.** The
corner has one degree of freedom, so a force at the patch does no work on the
spring when `F_y·dP_y + F_z·dP_z = 0`. That reduces the whole subject to the
fore-aft motion of one reference point per mm of wheel travel. Which point
depends on what reacts the torque: outboard brakes react on the upright, so the
contact patch; an inboard differential reacts on the chassis, so the wheel
centre. A GT3 has both, which is why the rear axle reads two different anti
numbers from one piece of geometry.

**The damper has to be on the right arm.** The solver works out which member the
lower mount actually rides on rather than assuming the lower wishbone. Getting
that wrong on a rear corner where the damper picks up high is a factor of nearly
three on spring rate, because the motion ratio is squared.

---

## Layout

```
hardpoints_fusion.csv    the hard points the tool loads on startup
hardpoints/              a second, redesigned set to load
software/                the solver, the application, the validation suite
studies/                 eight standalone parameter sweeps, with their plots
suspension_tool/         a two-corner axle model the RC-in-roll view reuses
REFERENCES.md            the books and papers behind the maths
```

---

## Reference data

Every benchmark value is read at run time, not written into the code. Where the
tool wants a reference car or a measured tyre it asks `software/benchmarks.py`,
which looks for a private module on `GT3_PRIVATE_DIR`, a folder outside this
tree. Nothing in this repository depends on it being there.

With no private data installed the tool runs the same, minus the comparisons:
anti-geometry targets come from the open literature in
[REFERENCES.md](REFERENCES.md), the roll-axis reference is the ground-level
axis, and the tyre is a labelled linear placeholder.
`software/benchmarks_private.example.py` shows the shape, so you can point it
at measurements of your own.

---

## Licence

MIT, see [LICENSE](LICENSE). The hard points are an independent reconstruction.
