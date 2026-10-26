# -*- coding: utf-8 -*-
"""
Steering jacking, damper deflection and the roll it forces on the front axle.
=============================================================================

Third core, after steer_camber (how the wheel LEANS) and scrub_trail (where the
patch GOES). This one answers: when you turn the wheel, how far does each corner
move VERTICALLY, what does that do to the dampers, and what roll does it force?

THE MECHANISM
-------------
Steering rotates the wheel about the INCLINED kingpin axis. Because the axis is
tilted, the wheel tries to move vertically. The wheel cannot -- it is on the road
-- so the corner jacks instead. That vertical demand is `corner_state['jack']`,
already validated in scrub_trail.

TWO JACKING EFFECTS -- DO NOT CONFUSE THEM
------------------------------------------
  * STEERING jacking (this module): purely geometric, from the tilted kingpin
    axis. Set by KPI / caster / scrub / trail. Appears the INSTANT you turn the
    wheel, before any lateral force exists.
  * LATERAL-FORCE jacking: lateral force at the patch, reacted through the links,
    makes a vertical force whose size depends on ROLL-CENTRE HEIGHT. Appears as
    grip builds.
Roll centres control the SECOND one. They do not control steering jacking.

ODD / EVEN -- the same structure as camber
------------------------------------------
Under parallel steer the left corner is the mirror of the right, so
    jack_L(delta) = jack_R(-delta)
and the axle response splits cleanly:
    EVEN part  (jack(+d)+jack(-d))/2  -> both corners move together  -> HEAVE  <- KPI
    ODD  part  (jack(+d)-jack(-d))/2  -> one up, one down            -> ROLL   <- CASTER
The two corners are NOT simply opposite: with KPI != 0 there is always a common
heave term. On this car the heave term DOMINATES (-2.08 mm vs -0.85 mm at 35 deg).

ROLL WITHOUT MODELLING THE CHASSIS
----------------------------------
Both wheels stay on the road, so the differential vertical demand between them is
a roll the axle must take:
    roll = atan( (jack_L - jack_R) / track )
No chassis model, no spring rate, no equilibrium solve. It is the KINEMATIC roll
demand -- what the geometry forces. (In reality the springs are compliant, so the
truth is shared between chassis roll and spring deflection; this is the stiff-spring
limit, and the deflection numbers are the locked-chassis limit. State which you mean.)

MOTION RATIO -- read this before trusting it
--------------------------------------------
`motion_ratio()` is FIRST-ORDER. The damper hangs off the LOWER WISHBONE, whose
rotation about its inner axis is exact, so the damper end of the calculation is
sound. The WHEEL end is not: this treats the LBJ's vertical motion as the wheel's,
but the wheel centre sits on the upright, constrained by BOTH wishbones, so it does
not move quite with the LBJ. A rigorous MR needs the wishbone solver.

The estimate comes out ~0.997 (essentially 1:1) because the damper is mounted 9.5 mm
from the ball joint. That is NOT a modelling artefact: the user has confirmed against
the real car that this Lamborghini uses a conventional damper mounted directly on the
lower A-arm, front and rear -- no pushrod, no pullrod, no rocker. So a near-unity
motion ratio is the correct answer for this car, not an error to be corrected.
"""
import numpy as np

from scrub_trail import corner_state


def _perp(vec, axis):
    """Component of vec perpendicular to the unit `axis`."""
    return vec - np.dot(vec, axis) * axis


def motion_ratio(LIF, LIR, LBJ, SL, SU):
    """FIRST-ORDER motion ratio: damper travel per mm of vertical wheel travel.

    The lower wishbone rotates about the axis through its two inner pivots. For a
    small rotation:
        damper travel = |damper_unit . v_shock| * |r_shock|
        wheel travel  = |v_LBJ_z|              * |r_LBJ|
    where v_x = axis x r_x is the velocity direction of each point.

    See the module docstring: the wheel end of this is an approximation until the
    wishbone solver exists.
    """
    LIF, LIR = np.asarray(LIF, float), np.asarray(LIR, float)
    LBJ, SL, SU = np.asarray(LBJ, float), np.asarray(SL, float), np.asarray(SU, float)

    ax = LIR - LIF
    ax = ax / np.linalg.norm(ax)                 # wishbone inner pivot axis

    r_lbj = _perp(LBJ - LIF, ax)                 # radius arms, perpendicular to it
    r_sl = _perp(SL - LIF, ax)

    v_lbj = np.cross(ax, r_lbj)
    v_lbj = v_lbj / np.linalg.norm(v_lbj)        # velocity directions
    v_sl = np.cross(ax, r_sl)
    v_sl = v_sl / np.linalg.norm(v_sl)

    damper = SU - SL
    damper = damper / np.linalg.norm(damper)

    damper_rate = abs(np.dot(damper, v_sl)) * np.linalg.norm(r_sl)
    wheel_rate = abs(v_lbj[2]) * np.linalg.norm(r_lbj)
    return float(damper_rate / wheel_rate)


def axle_state(LBJ, UBJ, WC, steer_deg, camber_deg, MR, track_mm):
    """The FRONT AXLE under steer, from ONE corner + mirror symmetry.

    Under parallel steer the left corner is the mirror of the right, so its
    jacking is the right corner's evaluated at -delta. No second data set needed,
    and it stays inside the one-wheel scope (Ackermann is future work).

    Returns arrays over steer_deg:
        jack_R, jack_L   [mm]  vertical demand at each corner (+ = corner lifts)
        defl_R, defl_L   [mm]  damper deflection = jack * MR (chassis-locked limit)
        heave            [mm]  the EVEN part -- both corners together (KPI)
        roll             [deg] the ODD part as an angle (caster), stiff-spring limit
    """
    steer = np.atleast_1d(np.asarray(steer_deg, float))
    jack_R = corner_state(LBJ, UBJ, WC, steer, camber_deg)["jack"]
    jack_L = corner_state(LBJ, UBJ, WC, -steer, camber_deg)["jack"]   # mirror
    return dict(
        steer=steer,
        jack_R=jack_R, jack_L=jack_L,
        defl_R=jack_R * MR, defl_L=jack_L * MR,
        heave=0.5 * (jack_R + jack_L),
        roll=np.degrees(np.arctan((jack_L - jack_R) / track_mm)),
    )


def front_track(csv_points):
    """Front track [mm] from the wheel centre, assuming a mirrored left corner."""
    return 2.0 * float(csv_points["Wheel centre"][0])


if __name__ == "__main__":
    import config as cfg

    p = cfg._load_front_points()
    LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]
    MR = motion_ratio(p["Lower wishbone - inner front"], p["Lower wishbone - inner rear"],
                      LBJ, p["Lower Shock mount"], p["Upper Shock mount"])
    TRACK = front_track(p)

    print("Jacking core - baseline check")
    print("=" * 66)
    print(f"  KPI {cfg.STOCK_KPI:+.2f}   caster {cfg.STOCK_CASTER:+.2f}   "
          f"camber {cfg.STATIC_CAMBER:+.1f}")
    print(f"  motion ratio  = {MR:.4f}   (FIRST-ORDER - see the module docstring)")
    print(f"  front track   = {TRACK:.2f} mm")
    print(f"  damper mount is {np.linalg.norm(p['Lower Shock mount']-LBJ):.1f} mm "
          f"from the LBJ -> direct-acting, no rocker")
    print()
    st = axle_state(LBJ, UBJ, WC, np.array([-35., -20., 0., 20., 35.]),
                    cfg.STATIC_CAMBER, MR, TRACK)
    print("  steer    jack_R    jack_L    defl_R    defl_L     heave      roll")
    print("   [deg]     [mm]      [mm]      [mm]      [mm]      [mm]     [deg]")
    for i, d in enumerate([-35, -20, 0, 20, 35]):
        print(f"  {d:+6.0f}  {st['jack_R'][i]:+8.3f}  {st['jack_L'][i]:+8.3f}  "
              f"{st['defl_R'][i]:+8.3f}  {st['defl_L'][i]:+8.3f}  "
              f"{st['heave'][i]:+8.3f}  {st['roll'][i]:+8.4f}")
    print()
    print("  NOTE the two corners do NOT go in opposite directions: at +35 deg both")
    print("  drop. The EVEN (heave) term dominates because KPI != 0. The roll comes")
    print("  from the DIFFERENCE between them, which is what caster controls.")
