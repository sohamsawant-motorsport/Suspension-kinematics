# -*- coding: utf-8 -*-
"""
The ROLL AXIS: the line joining the front and rear roll centres.
================================================================

WHY IT IS ONE THING, NOT TWO
----------------------------
"Roll axis behaviour" and "front/rear roll-centre balance" are the same geometry
seen two ways. The two roll centres define a line; where that line sits under the
sprung CG sets how much roll moment the car makes, and how the line is TILTED
sets how the resulting load transfer divides front to rear. So this module
computes both from the same two numbers.

    axis height at any station:  h(u) = h_f + (h_r - h_f) u,  u = 0 front, 1 rear
    inclination:                 atan( (h_r - h_f) / wheelbase )
    roll lever:                  h_CG - h(u_CG),  u_CG = the SPRUNG rear fraction

THE LEVER IS THE POINT
----------------------
Roll moment = sprung weight x lever, and roll angle = moment / roll stiffness. So
the roll axis is not decoration -- it is one of only two things that set the roll
gradient, the other being the springs and bars.

On this car both roll centres are BELOW GROUND (-283.6 front, -52.0 rear), which
puts the axis 151.7 mm below ground under the CG and makes the lever 601.7 mm --
**33.7 % longer than the CG height alone**. That is roll moment the car creates
for free, and then pays to control with springs and bars.

GEOMETRIC vs ELASTIC
--------------------
Lateral load transfer arrives at each axle two ways:

  * GEOMETRIC -- fed straight through the LINKS, proportional to that axle's roll
    centre height. Instantaneous, no roll needed.
  * ELASTIC -- fed through the springs and bars as the body rolls, proportional
    to that axle's share of the roll stiffness.

A roll centre BELOW ground makes its axle's geometric term NEGATIVE: the links
transfer load the WRONG way and the springs have to make up the difference. Ours
does exactly that -- -36.0 % at the front, -17.9 % at the rear.

WHAT THE BALANCE DOES
---------------------
Raising the FRONT roll centre relative to the rear adds geometric transfer at the
front, which is understeer, and tilts the axis nose-up. Raising the REAR does the
opposite. That is the "roll-centre balance" lever, and it is separate from the
roll-stiffness (spring/bar) lever even though both end up moving LLTD.

Everything here takes roll-centre heights as INPUTS, so it is driven by whatever
the FRONT and REAR hard points currently solve to.
"""

import numpy as np

import benchmarks as bm

import load_transfer as ltr

# Benchmark roll-centre heights [mm], read off the reference GT3's published
# kinematic curves at zero wheel travel, when private reference data is
# installed. Reading precision is about +/- 5 mm. Benchmarking only,
# never published; anything public must argue from the physics instead.
#
# The striking part is not either number on its own but that BOTH are ABOVE
# ground and only 6 mm apart, i.e. an essentially LEVEL roll axis sitting just
# over the road. Ours is 5 deg nose-down and 151.7 mm underground.
# With private data installed these are a reference car's measured roll
# centres. Without it they fall back to the GROUND-LEVEL axis -- the classical
# neutral reference, where the geometric share of lateral load transfer is zero
# and the car neither jacks up nor digs in. It is a line to design against, not
# a measurement of any vehicle.
GROUND_LEVEL_REFERENCE_MM = (0.0, 0.0)

BENCHMARK_RC_MM = bm.get("ROLL_CENTRE_MM") or GROUND_LEVEL_REFERENCE_MM
BENCHMARK_SLOPE = bm.get("ROLL_CENTRE_SLOPE") or (0.0, 0.0)
BENCHMARK_IS_MEASURED = bm.get("ROLL_CENTRE_MM") is not None


# ---------------------------------------------------------------------------
#  the axis itself
# ---------------------------------------------------------------------------


def axis_height_at(u, rc_f_mm, rc_r_mm):
    """Roll-axis height [mm] at longitudinal fraction u (0 = front axle, 1 = rear).

    A straight line between the two roll centres. u outside [0, 1] extrapolates,
    which is meaningful -- overhangs are still part of the car.
    """
    return (float(rc_f_mm)
            + (float(rc_r_mm) - float(rc_f_mm)) * np.asarray(u, float))


def inclination_deg(rc_f_mm, rc_r_mm, wheelbase_mm):
    """Roll-axis inclination [deg]. POSITIVE = nose-DOWN (front RC lower).

    Nose-down is the conventional layout: it keeps the front geometric transfer
    low, which is where you usually want the car to be softer, and it puts the
    axis closer to the CG at the rear where the mass is.
    """
    return float(np.degrees(np.arctan2(float(rc_r_mm) - float(rc_f_mm),
                                       float(wheelbase_mm))))


def sprung_rear_fraction(car):
    """The SPRUNG rear mass fraction -- the station the roll lever is measured at.

    Not the total rear fraction: unsprung mass sits AT the axles, so removing it
    shifts the sprung CG. The same correction that fixed the load-transfer moment
    identity and appears in the spring-rate maths.
    """
    Wuf = car["unsprung_f_kg"] * ltr.G
    Wur = car["unsprung_r_kg"] * ltr.G
    Wsf = car["mass_total_kg"] * (1.0 - car["rear_frac"]) * ltr.G - Wuf
    Wsr = car["mass_total_kg"] * car["rear_frac"] * ltr.G - Wur
    return float(Wsr / (Wsf + Wsr))


def state(rc_f_mm, rc_r_mm, car):
    """Everything about the axis at ride height, in one dict."""
    u_cg = sprung_rear_fraction(car)
    h_axis = float(ltr.roll_axis_height(rc_f_mm, rc_r_mm, u_cg))
    lever = float(car["h_cg_mm"]) - h_axis
    return dict(
        rc_f=float(rc_f_mm), rc_r=float(rc_r_mm),
        u_cg=u_cg, h_axis=h_axis, lever=lever,
        incl_deg=inclination_deg(rc_f_mm, rc_r_mm, car["wheelbase_mm"]),
        rise_mm=float(rc_r_mm) - float(rc_f_mm),
        lever_vs_cg=lever / float(car["h_cg_mm"]) - 1.0,
    )


# ---------------------------------------------------------------------------
#  the axis under roll
# ---------------------------------------------------------------------------


def in_roll(roll_f, rcz_f, roll_r, rcz_r, car, rolls=None):
    """The axis re-built at each roll angle from BOTH axles' migrating RCs.

    The two roll centres do not sit still as the car leans -- each is a force-line
    intersection and moves with its own linkage. So the axis height, its
    inclination and the roll lever all change THROUGH the corner, which is a
    large part of why a car can feel different at entry and at mid-corner.

    Front and rear are sampled on their own roll grids, so both are interpolated
    onto a common one before the axis is formed.
    """
    if rolls is None:
        lo = max(np.min(roll_f), np.min(roll_r))
        hi = min(np.max(roll_f), np.max(roll_r))
        rolls = np.linspace(lo, hi, 41)
    rolls = np.asarray(rolls, float)
    hf = np.interp(rolls, np.asarray(roll_f, float), np.asarray(rcz_f, float))
    hr = np.interp(rolls, np.asarray(roll_r, float), np.asarray(rcz_r, float))
    u_cg = sprung_rear_fraction(car)
    h_axis = hf + (hr - hf) * u_cg
    return dict(
        roll=rolls, rc_f=hf, rc_r=hr, h_axis=h_axis,
        lever=float(car["h_cg_mm"]) - h_axis,
        incl_deg=np.degrees(np.arctan2(hr - hf, float(car["wheelbase_mm"]))),
    )


# ---------------------------------------------------------------------------
#  geometric vs elastic
# ---------------------------------------------------------------------------


def split(rc_f_mm, rc_r_mm, car, lat_g=1.0):
    """Per-axle geometric / elastic / unsprung shares of the lateral transfer."""
    t = ltr.transfer(lat_g, rc_f_mm, rc_r_mm, car)
    out = dict(t)
    for tag in ("f", "r"):
        tot = t[f"total_{tag}"]
        out[f"geo_pct_{tag}"] = t[f"geo_{tag}"] / tot * 100.0
        out[f"el_pct_{tag}"] = t[f"el_{tag}"] / tot * 100.0
        out[f"uns_pct_{tag}"] = t[f"uns_{tag}"] / tot * 100.0
    return out


def balance_map(rc_f_grid, rc_r_grid, car, lat_g=1.0):
    """Front LLTD [%] over a grid of front x rear roll-centre heights.

    This is the roll-centre balance lever laid out flat: how much of the car's
    balance you can move purely by where the links put the roll centres, with the
    springs and bars untouched.

    Returns (LLTD, roll_gradient, lever) each shaped (len(rc_r), len(rc_f)).
    """
    F, R = np.meshgrid(np.asarray(rc_f_grid, float),
                       np.asarray(rc_r_grid, float))
    lltd = np.zeros_like(F)
    grad = np.zeros_like(F)
    lev = np.zeros_like(F)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            t = ltr.transfer(lat_g, F[i, j], R[i, j], car)
            lltd[i, j] = t["lltd_f"]
            grad[i, j] = t["roll_deg"] / lat_g
            lev[i, j] = t["lever"]
    return F, R, lltd, grad, lev


if __name__ == "__main__":
    car = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
    st = state(-283.6, -52.0, car)
    print("ROLL AXIS at ride height")
    print(f"   front RC {st['rc_f']:+8.1f}   rear RC {st['rc_r']:+8.1f} mm")
    print(f"   rise front->rear {st['rise_mm']:+.1f} mm  ->  inclination "
          f"{st['incl_deg']:+.3f} deg  ({'nose-down' if st['incl_deg'] > 0 else 'nose-up'})")
    print(f"   sprung rear fraction {st['u_cg']:.4f}")
    print(f"   axis height under the CG {st['h_axis']:+.1f} mm")
    print(f"   ROLL LEVER {st['lever']:.1f} mm   ({st['lever_vs_cg']:+.1%} vs the "
          f"CG height alone)")
    s = split(-283.6, -52.0, car)
    print("\nSHARES at 1 g")
    for tag, nm in (("f", "FRONT"), ("r", "REAR")):
        print(f"   {nm:5s} geometric {s['geo_pct_' + tag]:+7.1f} %   "
              f"elastic {s['el_pct_' + tag]:+7.1f} %   "
              f"unsprung {s['uns_pct_' + tag]:+6.1f} %")
    print(f"   front LLTD {s['lltd_f']:.2f} %")
