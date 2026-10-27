# Hard points

`../hardpoints_fusion.csv` is what the tool loads on startup. It is the starting
geometry, straight out of CAD, and the eight studies in `../studies/` were all
run against it.

`hardpoints_redesigned.csv` here is where that geometry ended up after working
the front axle and then the rear through the tool. Load it on the FRONT page,
then load it again on the REAR page — the file holds both corners and each page
picks out the rows it needs.

## What changed

| | starting | redesigned |
|---|---|---|
| **front** roll centre | −283.6 mm | **+30.8 mm** |
| front bump steer | −0.117 °/mm | **−0.0028 °/mm** |
| front scrub | 145 mm | **17.3 mm** |
| front Ackermann | 130 % | **−5.4 %** |
| **rear** roll centre | −41.8 mm | **+38.0 mm** |
| rear camber recovery | 18.9 % | **46.8 %** |
| rear scrub | 145.7 mm | **+7.1 mm** |
| rear bump steer | +0.0076 °/mm | **+0.0003 °/mm toe-IN** |
| rear RC lateral migration | ~534 mm/deg | **86 mm/deg** |
| rear arm length ratio | 0.960 | **1.234** |
| wheelbase / front track / rear track | 2645 / 1662 / 1646 | **2700 / 1742 / 1712** |

The roll-centre migration line is the one I would point at. It did not come from
moving the roll centre — it came from lengthening both wishbones **along their
own lines**, which leaves every static value identical (the arm's tangent is
unchanged, only its radius moves) while slowing how fast the instant centre
sweeps. Front view, static printout, KPI, caster, scrub, trail: all frozen.
Migration drops by a factor of six.

## Format

```
Corner,Point,X_mm,Y_mm,Z_mm
```

`X` lateral, positive outboard. `Y` longitudinal, positive forward. `Z`
vertical, positive up. Millimetres. Both files hold the right-hand corners; the
left side is their mirror. Lines starting with `#` are ignored, so a header
block like the one in `hardpoints_redesigned.csv` is safe to keep.

Saving from the tool writes the same format, so anything you save loads back.
