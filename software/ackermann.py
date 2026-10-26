"""Ackermann / Jeantaud steering geometry for the front axle.

WHAT THIS ANSWERS
-----------------
When a car turns, the inside front wheel travels a tighter circle than the
outside one, so at zero slip angle it must point further into the corner. The
Ackermann (Jeantaud) condition says that for BOTH front wheels to roll without
scrub about a single turn centre lying on the rear-axle line:

        cot(delta_outer) - cot(delta_inner) = t / L                       (1)

    t = lateral distance between the two STEERING AXES measured at the GROUND
        (NOT the wheel track -- the wheels pivot about their kingpin axes, and
        those pierce the ground inboard/outboard of the tyre centre by the
        scrub radius)
    L = wheelbase

The linkage almost never delivers exactly that, so the industry reports how far
along the way it gets:

        %Ackermann = (delta_inner_actual - delta_outer)
                     / (delta_inner_ideal  - delta_outer) * 100           (2)

      100 %  = true Ackermann          (inner wheel steers MORE)
        0 %  = parallel steering       (both wheels the same angle)
      < 0 %  = anti-Ackermann          (inner wheel steers LESS)

WHY A RACE CAR IS NOT 100 %
---------------------------
Equation (1) is a pure ROLLING condition -- it is derived at zero slip angle, so
it is only correct in a car park. A racing tyre generates lateral force by
running AT a slip angle, and the heavily-loaded outer tyre needs a bigger slip
angle than the lightly-loaded inner one to reach its peak. Steering the inner
wheel further therefore drags it PAST its own peak and it starts scrubbing.
That is why GT and formula cars run near-parallel or even anti-Ackermann.
(For scale: a competitive GT3 front axle runs only a few per cent -- essentially
parallel -- Ackermann is largely a slip-angle question at racing speeds.)

THE VALUE AT ZERO STEER
-----------------------
Both numerator and denominator of (2) vanish at zero steer, so %Ackermann there
is a 0/0 limit -- but a finite one. Expanding (1) for small angles, cot(d) ~ 1/d:

        1/do - 1/di = t/L  =>  (di - do) / (di*do) = t/L
                        =>  di - do  ~  (t/L) * d^2                       (3)

Both the ideal and the actual differences are SECOND order in steer angle, so
their ratio tends to a constant. `zero_steer_limit()` recovers it with a
quadratic fit instead of dividing two vanishing numbers, which is what makes the
curve near the origin usable rather than noise.

Everything here is driven by the SOLVED linkage (explorer.steer_from_rack ->
wishbone_solver.solve_on_circle), never by a small-angle steering-ratio
estimate: the nonlinearity IS the Ackermann.
"""

import numpy as np

from scrub_trail import ground_point

# --------------------------------------------------------------------------
#  geometry of the steering axes
# --------------------------------------------------------------------------


def kingpin_ground_track(pts_R, pts_L, mirror):
    """Lateral distance `t` between the two steering axes AT GROUND LEVEL [mm].

    This -- not the wheel track -- is the `t` in the Ackermann condition. The
    two differ by twice the scrub radius, which on this car is ~145 mm a side,
    so using the wheel track would put the ideal curve out by ~18 %.

    `pts_L` is given in the LEFT corner's own (mirrored) coordinates, so it is
    mapped back into the car frame with `mirror` before measuring.
    """
    from explorer import LBJ_NAME, UBJ_NAME          # local: avoids a cycle

    gR = ground_point(pts_R[LBJ_NAME], pts_R[UBJ_NAME])
    mL = mirror(pts_L)
    gL = ground_point(mL[LBJ_NAME], mL[UBJ_NAME])
    # the left corner lives at -X once mirrored back into the car frame
    return float(abs(gR[0]) + abs(gL[0]))


# --------------------------------------------------------------------------
#  the Jeantaud ideal
# --------------------------------------------------------------------------


def ideal_inner(delta_outer_deg, t_mm, wheelbase_mm):
    """Inner-wheel angle that satisfies the Ackermann condition, from the outer.

    Rearranging (1):   cot(di) = cot(do) - t/L   =>   di = atan(1/that).

    Sign-agnostic: the magnitude of the outer angle is used and the caller's
    sign is restored, so it works for a left or a right turn unchanged. At
    delta_outer = 0 the ideal inner angle is 0 (cot -> inf both sides).
    """
    do = np.asarray(delta_outer_deg, float)
    sgn = np.sign(do)
    mag = np.abs(np.radians(do))

    out = np.zeros_like(mag)
    nz = mag > 0.0
    cot_o = 1.0 / np.tan(mag[nz])
    cot_i = cot_o - float(t_mm) / float(wheelbase_mm)
    # cot_i <= 0 means the ideal inner wheel is at/past 90 deg -- unreachable,
    # which only happens at absurd steer; np.arctan handles the sign correctly.
    out[nz] = np.degrees(np.arctan2(1.0, cot_i))
    return sgn * out


def ackermann_percent(delta_inner_deg, delta_outer_deg, t_mm, wheelbase_mm,
                      min_outer_deg=0.75):
    """%Ackermann from equation (2), NaN where the 0/0 limit makes it meaningless.

    Below `min_outer_deg` of outer steer the denominator is second-order small
    (see (3)) and the quotient is numerical noise -- use `zero_steer_limit()`
    for the value AT the origin instead of trusting this near it.
    """
    di = np.asarray(delta_inner_deg, float)
    do = np.asarray(delta_outer_deg, float)
    ideal = ideal_inner(do, t_mm, wheelbase_mm)

    num = np.abs(di) - np.abs(do)
    den = np.abs(ideal) - np.abs(do)
    out = np.full(di.shape, np.nan)
    ok = (np.abs(do) >= min_outer_deg) & (np.abs(den) > 1e-12)
    out[ok] = num[ok] / den[ok] * 100.0
    return out


def zero_steer_limit(delta_inner_deg, delta_outer_deg, t_mm, wheelbase_mm,
                     fit_below_deg=8.0):
    """%Ackermann AT zero steer, as a limit rather than a quotient of zeros.

    By (3) both differences go as delta^2, so fit
        (di - do)      = a * do^2        and
        (ideal - do)   = b * do^2
    over the small-steer region and return 100*a/b. Least squares through the
    origin, which is exact for a pure quadratic and robust to solver noise.
    """
    di = np.abs(np.asarray(delta_inner_deg, float))
    do = np.abs(np.asarray(delta_outer_deg, float))
    ideal = np.abs(ideal_inner(do, t_mm, wheelbase_mm))

    m = np.isfinite(di) & np.isfinite(do) & (do > 1e-9) & (do <= fit_below_deg)
    if m.sum() < 4:
        return float("nan")
    x = do[m] ** 2
    a = float(np.dot(x, (di - do)[m]) / np.dot(x, x))
    b = float(np.dot(x, (ideal - do)[m]) / np.dot(x, x))
    return 100.0 * a / b if b else float("nan")


# --------------------------------------------------------------------------
#  the sweep
# --------------------------------------------------------------------------


def sweep(pts_R, cam_R, pts_L, cam_L, rack_mm=None, wheelbase_mm=2645.0):
    """Full Ackermann study over a rack sweep.

    Returns a dict with, at every rack position:
        rack, steer_R, steer_L   -- road-wheel HEADINGS in the car frame [deg]
        inner, outer             -- sorted by magnitude, signs preserved
        ideal_inner              -- the Jeantaud target for that outer angle
        pct                      -- %Ackermann, NaN near the origin
        toe_out                  -- inner minus outer, the raw asymmetry [deg]
    plus the scalars `t_mm`, `wheelbase_mm` and `pct_at_zero`.

    The rack drives BOTH wheels off one bar, exactly as the car does, so the
    left/right difference is produced by the linkage rather than imposed.
    """
    from explorer import axle_patch_paths, mirror_pts, RACK_LO, RACK_HI

    if rack_mm is None:
        rack_mm = np.linspace(RACK_LO, RACK_HI, 241)
    rack_mm = np.asarray(rack_mm, float)

    paths = axle_patch_paths(pts_R, cam_R, pts_L, cam_L, rack_mm)
    if paths is None:
        return None

    sR = np.asarray(paths["R"]["steer"], float)
    sL = np.asarray(paths["L"]["steer"], float)
    t = kingpin_ground_track(pts_R, pts_L, mirror_pts)

    # Inner/outer is decided by the TURN DIRECTION, never by which wheel happens
    # to steer more. + heading = pointing toward +X = a RIGHT-hand turn, whose
    # centre lies on the +X side, so the RIGHT wheel is the inner one; the roles
    # swap for a left turn.
    #
    # Picking "inner = whichever steers more" looks equivalent and is NOT: it is
    # only true while the car is pro-Ackermann. Past parallel the inner wheel
    # steers LESS, the magnitude rule then selects the wrong wheel, and the
    # reported percentage folds back through zero and climbs again instead of
    # going negative. An anti-Ackermann design would have been scored positive.
    right_turn = (sR + sL) >= 0.0
    inner = np.where(right_turn, sR, sL)
    outer = np.where(right_turn, sL, sR)

    ideal = ideal_inner(outer, t, wheelbase_mm)
    pct = ackermann_percent(inner, outer, t, wheelbase_mm)

    return dict(
        rack=np.asarray(paths["rack"], float),
        steer_R=sR, steer_L=sL,
        inner=inner, outer=outer,
        ideal_inner=ideal,
        pct=pct,
        toe_out=np.abs(inner) - np.abs(outer),
        t_mm=t, wheelbase_mm=float(wheelbase_mm),
        pct_at_zero=zero_steer_limit(inner, outer, t, wheelbase_mm),
    )


def steering_wheel_angle(road_wheel_deg, ratio=13.0):
    """Convert road-wheel angle to STEERING-WHEEL angle for benchmark overlays.

    `ratio` is an ASSUMPTION -- the Temerario's steering ratio is not in our
    data. 13:1 is mid-range for a GT3. Only ever used to place a reference
    curve on a second axis; every number this module computes is a function of
    road-wheel angle alone and is unaffected by the value chosen.
    """
    return np.asarray(road_wheel_deg, float) * float(ratio)


if __name__ == "__main__":
    import config
    from explorer import mirror_pts

    pR = config._load_front_points()
    s = sweep(pR, -2.7, mirror_pts(pR), -2.7)
    print("ACKERMANN -- Temerario GT3 front axle")
    print(f"  kingpin ground track t   {s['t_mm']:8.1f} mm")
    print(f"  wheelbase L              {s['wheelbase_mm']:8.1f} mm")
    print(f"  t/L                      {s['t_mm']/s['wheelbase_mm']:8.4f}")
    print(f"  %Ackermann at zero steer {s['pct_at_zero']:8.2f} %")
    print()
    print(f"  {'rack':>7} {'outer':>8} {'inner':>8} {'ideal':>8} {'%Ack':>8}")
    for i in range(0, len(s["rack"]), max(1, len(s["rack"]) // 12)):
        print(f"  {s['rack'][i]:7.1f} {s['outer'][i]:8.3f} {s['inner'][i]:8.3f}"
              f" {s['ideal_inner'][i]:8.3f} {s['pct'][i]:8.2f}")
