# Hard points

`../hardpoints_fusion.csv` is what the tool loads on startup. It is the starting
geometry, straight out of CAD, and the eight studies in `../studies/` were all
run against it.

`hardpoints_redesigned.csv` here is where that geometry ended up after working
the front axle and then the rear through the tool. Load it on the FRONT page,
then load it again on the REAR page. The file holds both corners and each page
picks out the rows it needs.

## What changed

Both columns below are printed by `verify_table.py` from the two CSV files, not
typed in. Run it to check them:

```bash
python hardpoints/verify_table.py
```

Front static camber −2.7°, rear −1.5°, both columns.

| | starting | redesigned |
|---|---|---|
| **front** | | |
| roll centre | −283.6 mm | **+30.8 mm** |
| bump steer | −0.1169 °/mm | **−0.0028 °/mm** |
| scrub radius | 144.9 mm | **17.3 mm** |
| mechanical trail | 31.1 mm | **30.2 mm** |
| KPI | −5.27 ° | **+7.64 °** |
| caster | +1.66 ° | **+7.99 °** |
| camber gain | −0.023 °/25 mm | **−0.694 °/25 mm** |
| Ackermann, straight ahead | +129.7 % | **+16.4 %** |
| RC lateral at 1° roll | −146 mm | **+50 mm** |
| arm ratio, lower/upper | 1.008 | **1.310** |
| **rear** | | |
| roll centre | −51.6 mm | **+38.0 mm** |
| bump steer | +0.0055 °/mm | **+0.0004 °/mm** |
| scrub radius | 138.4 mm | **7.1 mm** |
| mechanical trail | 52.5 mm | **30.0 mm** |
| KPI | −2.73 ° | **+7.01 °** |
| caster | +8.70 ° | **+3.00 °** |
| camber gain | −0.355 °/25 mm | **−0.787 °/25 mm** |
| camber recovery | 20.4 % | **47.0 %** |
| FVSA | 4290 mm | **1830 mm** |
| RC lateral at 1° roll | −847 mm | **+86 mm** |
| arm ratio, lower/upper | 0.847 | **1.240** |
| **dimensions** | | |
| wheelbase | 2645.0 mm | **2700.0 mm** |
| front track | 1662.4 mm | **1742.0 mm** |
| rear track | 1645.8 mm | **1712.0 mm** |

Three conventions have to be stated, because a before/after table is only
honest if both columns use the same one.

**Static camber is a setup input, not a hard point.** Scrub and roll-centre
height both move with it. Rear scrub moves 6.08 mm per degree of camber, so
quoting the two columns at different camber settings would invent most of a
change that was really a shim. One value per axle, used for both columns.

**Arm ratio** is the perpendicular distance from the ball joint to its inner
hinge *axis*, lower over upper. The arm swings about that axis, so that is the
radius that governs how fast the instant centre sweeps. The distance to a pivot
*point* is a different number and gives a different ratio.

**RC lateral** is how far sideways the roll centre has moved at one degree of
body roll, from the two-corner axle model. It is quoted as a position, not a
gradient, because the migration is not linear and a slope through the origin
overstates what happens in the first degree, which is the only part a stiff GT
car ever sees.

## The line worth pointing at

The roll-centre migration row. It did not come from moving the roll centre. It
came from lengthening both wishbones **along their own lines**, which leaves
every static value identical (the arm's tangent is unchanged, only its radius
moves) while slowing how fast the instant centre sweeps. Front view, static
printout, KPI, caster, scrub, trail: all frozen. Migration drops by a factor of
roughly ten.

## Known gap: the front tie rod

The front Ackermann in the redesigned column is **+16.4 %**, and it should not
be. Moving the tie-rod outer ball joint 20 mm inboard takes it to −5.6 % and
takes front bump steer from −0.0028 to −0.0006 °/mm, while scrub, trail, KPI,
caster and roll-centre height do not move at all. That step is free and it is
not in this file yet, because the tie rod was the last thing worked and the CSV
was exported before it.

It is flagged rather than quietly applied: these are CAD coordinates and they
should be changed in CAD and re-exported, not patched in a text file.

## Format

```
Corner,Point,X_mm,Y_mm,Z_mm
```

`X` lateral, positive outboard. `Y` longitudinal, positive forward. `Z`
vertical, positive up. Millimetres. Both files hold the right-hand corners; the
left side is their mirror. Lines starting with `#` are ignored, so a header
block like the one in `hardpoints_redesigned.csv` is safe to keep.

Saving from the tool writes the same format, so anything you save loads back.
