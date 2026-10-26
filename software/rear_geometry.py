# -*- coding: utf-8 -*-
"""
REAR CORNER GEOMETRY  -  static geometry + the front-view roll centre.
======================================================================

The rear corner is a double wishbone like the front, with TWO differences:

  1  NO STEERING.  There is no rack and no tie rod. The fifth link is a fixed
     TOE LINK. It is not in the CAD yet, so for now we model it as an ideal link
     that holds the wheel at a constant toe (TOE_STATIC, default 0 deg) through
     travel -- i.e. zero bump toe. That assumption is REPLACEABLE: the moment the
     real toe-link points exist, the wishbone solver runs the rear unchanged
     (a rear = a front corner with the rack frozen at 0), and the true bump toe
     drops out. Everything in THIS module is toe-INDEPENDENT anyway (see below).

  2  A DRIVE SHAFT is present (inner + outer). It transmits torque, plunging as
     the wheel moves, so it constrains NOTHING about the upright. It is kept for
     the later anti-squat / traction study, not used here.

WHAT NEEDS NO TOE LINK  (everything in this file)
-------------------------------------------------
The two ball joints LBJ and UBJ lie ON the kingpin axis, so their motion with
wheel travel is fixed by the two A-arms ALONE -- independent of how the upright
twists about that axis (which is what the toe link would set). Therefore the
kingpin-axis position/orientation, the front-view instant centre, the FVSA and
the roll centre are all determined by the wishbones only. Camber-vs-travel and
bump toe are NOT (they need the twist) and wait for the next study.

FRAME (same as the front): X lateral (outboard +), Y longitudinal (forward +),
Z up; the CAD corner is the rear-RIGHT (X > 0).

Static camber is a SETUP input, not derivable from the hard points -- exactly as
on the front. REAR_STATIC_CAMBER below is a PLACEHOLDER to be measured in CAD.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg                                                  # noqa: E402
from scrub_trail import (kpi_caster, ground_point, contact_patch,     # noqa: E402
                         scrub_radius, mechanical_trail, steering_axis,
                         wheel_spin_axis, wheel_radius, rotate_about_line)
from wishbone_solver import (solve_on_circle, corner_geometry,        # noqa: E402
                            solve_sweep, linkage_roll_centre, solve_at)

# ---- setup inputs (NOT derivable from hard points) --------------------------
# PLACEHOLDER -- measure the real rear static camber in CAD, as was done for the
# front (-2.7 deg). Rear GT3 camber is usually a little less than the front.
REAR_STATIC_CAMBER = -2.7
TOE_STATIC = 0.0            # ideal toe link -> constant toe; user's assumption

UBJ = "Upper wishbone - outer (UBJ)"
LBJ = "Lower wishbone - outer (LBJ)"
UIF = "Upper wishbone - inner front"
UIR = "Upper wishbone - inner rear"
LIF = "Lower wishbone - inner front"
LIR = "Lower wishbone - inner rear"
WC = "Wheel centre"
TOE_I = "Toe link - inner"
TOE_O = "Toe link - outer"


def load_rear():
    """The rear-RIGHT corner hard points from the project CSV."""
    return cfg._load_points("REAR")


# ---------------------------------------------------------------- ball joints
def _arm_axis(pi_front, pi_rear):
    """Unit direction of an A-arm's inner hinge line."""
    a = np.asarray(pi_rear, float) - np.asarray(pi_front, float)
    return a / np.linalg.norm(a)


def ball_joints(pts, theta_L_deg):
    """LBJ and UBJ at a lower-arm rotation of theta_L about the lower inner axis.

    The lower arm sets the travel; the upper arm angle then follows from the
    rigid-upright constraint |UBJ - LBJ| = kingpin length -- solved exactly on
    the upper ball joint's circle (wishbone_solver.solve_on_circle). This is the
    same closed form the validated front solver uses, minus the tie-rod step.
    """
    lbj0, ubj0 = np.asarray(pts[LBJ], float), np.asarray(pts[UBJ], float)
    lax = _arm_axis(pts[LIF], pts[LIR])
    uax = _arm_axis(pts[UIF], pts[UIR])
    kp = float(np.linalg.norm(ubj0 - lbj0))

    lbj = rotate_about_line(lbj0, np.asarray(pts[LIF], float), lax,
                            theta_L_deg)[0]
    # UBJ rides its circle about the upper inner axis, at kingpin length from LBJ
    _, ubj = solve_on_circle(ubj0, np.asarray(pts[UIF], float), uax, lbj, kp,
                             guess_deg=0.0)
    return lbj, ubj


def travel_of(pts, theta_L_deg):
    """Wheel-centre rise for a lower-arm angle. The wheel centre is rigid in the
    upright; near static its rise tracks the ball joints, which is all we need to
    map a small perturbation to a travel for the instant-centre derivative."""
    lbj, ubj = ball_joints(pts, theta_L_deg)
    lbj0, ubj0 = np.asarray(pts[LBJ], float), np.asarray(pts[UBJ], float)
    # rigid-body vertical motion of the ball-joint midpoint (first order)
    return 0.5 * ((lbj[2] - lbj0[2]) + (ubj[2] - ubj0[2]))


# ------------------------------------------------------- front-view instant C
def front_view_ic(pts, d_theta=0.25):
    """Front-view (X-Z) instant centre of the upright relative to the chassis.

    Central difference on the two ball joints about static, then intersect the
    perpendiculars to their front-view velocities. Toe-link-INDEPENDENT: the ball
    joints move by the wishbone linkage only. Returns (IC_x, IC_z), or an
    infinite point if the arms are momentarily parallel.
    """
    lp, up = ball_joints(pts, +d_theta)
    lm, um = ball_joints(pts, -d_theta)
    l0 = np.asarray(pts[LBJ], float)
    u0 = np.asarray(pts[UBJ], float)
    vL = (lp - lm)[[0, 2]]
    vU = (up - um)[[0, 2]]
    pL, pU = l0[[0, 2]], u0[[0, 2]]
    nL = np.array([-vL[1], vL[0]])         # perpendicular to each velocity
    nU = np.array([-vU[1], vU[0]])
    A = np.array([[nL[0], -nU[0]], [nL[1], -nU[1]]])
    if abs(np.linalg.det(A)) < 1e-12:
        return np.array([np.inf, np.inf])
    t, _ = np.linalg.solve(A, pU - pL)
    return pL + t * nL


# ---------------------------------------------------------------- statics
def static_geometry(pts=None, static_camber=REAR_STATIC_CAMBER):
    """Everything about the rear corner at ride height. Contains no sweep."""
    if pts is None:
        pts = load_rear()
    lbj, ubj, wc = pts[LBJ], pts[UBJ], pts[WC]
    kpi, cas = kpi_caster(lbj, ubj)
    G = ground_point(lbj, ubj)
    patch = contact_patch(wc, static_camber)
    ic = front_view_ic(pts)

    # single-corner roll centre: where the force line (patch -> IC) crosses the
    # car centreline X = 0 (the symmetric-other-side assumption)
    d = ic - patch[[0, 2]]          # 2-vector in the front view: (dX, dZ)
    rc_h = np.nan
    if abs(d[0]) > 1e-9:
        s = (0.0 - patch[0]) / d[0]
        rc_h = float(patch[2] + s * d[1])
    fvsa = float(abs(ic[0] - patch[0])) if np.all(np.isfinite(ic)) else np.inf

    def arm(outer, pf, pr):
        o = np.asarray(pts[outer], float)
        return 0.5 * (np.linalg.norm(o - pts[pf]) + np.linalg.norm(o - pts[pr]))

    return dict(
        pts=pts, static_camber=static_camber, toe=TOE_STATIC,
        kpi=kpi, caster=cas,
        kingpin_len=float(np.linalg.norm(np.asarray(ubj) - lbj)),
        upper_arm=float(arm(UBJ, UIF, UIR)),
        lower_arm=float(arm(LBJ, LIF, LIR)),
        G=G, patch=patch, IC=ic, fvsa=fvsa, rc_height=rc_h,
        scrub=scrub_radius(lbj, ubj, wc, static_camber),
        trail=mechanical_trail(lbj, ubj, wc, static_camber),
        track=2.0 * float(np.asarray(wc, float)[0]),
        wc_height=float(np.asarray(wc, float)[2]),
        wheelbase=float(cfg._load_points("FRONT")[WC][1] - np.asarray(wc, float)[1]),
    )


# =====================================================================
#  BUMP KINEMATICS  -  now that there is a toe link
# =====================================================================
def has_toe_link(pts):
    return TOE_I in pts and TOE_O in pts


def solver_points(pts):
    """A copy of the rear points with the TOE LINK presented to the solver under
    the tie-rod keys it expects.

    The rear does not steer: its fifth link is a FIXED toe link, geometrically
    identical to a front tie rod whose rack never moves. So the validated
    wishbone_solver runs the rear unchanged -- it treats the toe link exactly as
    a tie rod and we simply never apply any rack travel (rack = 0 always). No new
    kinematics is written; the rear reuses the 27-assertion front solver.
    """
    if not has_toe_link(pts):
        raise ValueError("rear has no toe link yet -- add 'Toe link - inner/outer'")
    out = {k: np.asarray(v, float) for k, v in pts.items()}
    out["Tie rod - inner"] = out[TOE_I]
    out["Tie rod - outer"] = out[TOE_O]
    return out


def travel_limits(pts=None, static_camber=REAR_STATIC_CAMBER, span=45.0):
    """(max droop, max compression) [mm] reachable within a +/-`span` rotation of
    the lower wishbone, on the monotonic branch through ride height.

    IMPORTANT -- this is a SEARCH RANGE, not a hard geometric bind. The travel
    keeps growing with `span` (20 deg -> +69 mm, 28 -> +93, 36 -> +113, 45 -> +131),
    so quoting a small span as "the suspension binds here" is wrong; an early
    version of this study did exactly that and had to be corrected. The TRUE
    geometric limit is where the closed form itself fails, which on this rear is
    a 75 deg wishbone angle = +155 mm compression (droop does not bind at all
    within +/-120 deg). Real travel is limited by dampers and bump stops long
    before any of this. The default span is set wide enough to cover any roll
    angle worth studying.
    """
    if pts is None:
        pts = load_rear()
    thetas = np.linspace(-span, span, 1121)
    g = corner_geometry(solver_points(pts), static_camber)
    trav, thU, thT = [], 0.0, 0.0
    for th in thetas:
        s = solve_at(g, th, 0.0, thU, thT)
        thU, thT = s["thU"], s["thT"]
        trav.append(s["WC"][2] - g["WC"][2])
    trav = np.asarray(trav)
    i0 = int(np.argmin(np.abs(trav)))
    lo, hi = i0, i0
    while lo > 0 and trav[lo - 1] < trav[lo]:
        lo -= 1
    while hi < len(trav) - 1 and trav[hi + 1] > trav[hi]:
        hi += 1
    return float(trav[lo]), float(trav[hi])


def bump_sweep(pts=None, static_camber=REAR_STATIC_CAMBER, droop=-30.0,
               bump=30.0, n=121, span=20.0):
    """Rear kinematics vs WHEEL TRAVEL (heave), from the validated solver.

    Returns arrays over `bump` (wheel-centre rise, + = compression):
        camber, toe [deg]      -> camber gain and BUMP STEER are their slopes
        caster, kpi [deg]      -> how the axis inclination changes with travel
        rc [mm]                -> linkage roll-centre height
    plus the gradients camber_rate, toe_rate (deg/mm) and the values at ride
    height. `rack` is held at 0 throughout: the rear does not steer.
    """
    if pts is None:
        pts = load_rear()
    g = corner_geometry(solver_points(pts), static_camber)
    travel = np.linspace(droop, bump, n)
    sw = solve_sweep(g, travel, rack_mm=0.0, theta_span_deg=span, n_theta=1121)
    rc = linkage_roll_centre(g, travel)
    i0 = int(np.argmin(np.abs(sw["bump"])))
    cam_rate = np.gradient(sw["camber"], sw["bump"])
    toe_rate = np.gradient(sw["toe"], sw["bump"])
    return dict(
        bump=sw["bump"], camber=sw["camber"], toe=sw["toe"],
        caster=sw["caster"], kpi=sw["kpi"], rc=rc,
        camber_rate=cam_rate, toe_rate=toe_rate,
        camber_gain=float(cam_rate[i0]), bump_steer=float(toe_rate[i0]),
        static_toe=float(sw["toe"][i0]), static_camber=float(sw["camber"][i0]),
        rc0=float(rc[i0]),
        toe_link_len=float(np.linalg.norm(pts[TOE_O] - pts[TOE_I])),
    )


# =====================================================================
#  ROLL  -  camber recovery and ROLL STEER (the two-corner cases)
# =====================================================================
def _axle():
    """The validated two-corner model (26 assertions), imported lazily so the
    static/bump studies do not depend on it."""
    _ST = os.path.join(os.path.dirname(_HERE), "suspension_tool")
    if _ST not in sys.path:
        sys.path.insert(0, _ST)
    import axle                                                # noqa: E402
    return axle


def roll_sweep(pts=None, static_camber=REAR_STATIC_CAMBER,
               roll_lo=-5.0, roll_hi=5.0, n=41):
    """Rear axle vs BODY ROLL: camber recovery, ROLL STEER, and RC migration.

    Roll is NOT a rigid rotation of the car. Rolling the body while the tyres
    stay on the road IS equal-and-opposite wheel travel, so axle.axle_state
    converts a roll angle into  bump = +/- (t/2)*tan(phi)  per side. That is the
    validated model (validate_axle.py asserts the two formulations agree).

    THE ONE SUBTLETY -- two different cambers
    -----------------------------------------
    axle_state reports camber relative to the CHASSIS (verified: it equals the
    single-corner heave camber at the same travel, to 0.0000 deg). What the tyre
    actually works at is camber relative to the ROAD, and the rolling body tilts
    the wheel with it:

        gamma_road = gamma_chassis + phi_wheel ,   phi_wheel = atan(bump / half)

    and because bump = +/-(half)*tan(phi), phi_wheel is exactly +phi on the
    compressing side and -phi on the drooping side. Same relation the front
    explorer uses.

    CAMBER RECOVERY is then how much of the body roll the linkage cancels:

        recovery = (1 - d(gamma_road)/d(phi)) * 100 %

    0 % = the wheel leans over exactly with the body (linkage does nothing);
    100 % = gamma_road never changes (perfect).

    ROLL STEER is d(toe)/d(phi) [deg/deg] -- on a rear axle this is THE stability
    parameter: toe-IN on the compressing (outside) wheel steers the rear into the
    corner => understeer => forgiving; toe-OUT => oversteer => can snap.

    NOTE on sides: with axle's sign convention a POSITIVE roll compresses the
    LEFT wheel, so the LEFT is the loaded / outside wheel for phi > 0.
    """
    if pts is None:
        pts = load_rear()
    axle = _axle()
    sp = solver_points(pts)
    L = axle.mirror_points(sp)
    roll = np.linspace(roll_lo, roll_hi, n)
    sw = axle.roll_sweep(sp, L, static_camber, roll)

    half = 0.5 * (axle._static_patch_x(sp, static_camber, False)
                  - axle._static_patch_x(L, static_camber, True))
    phi_L = np.degrees(np.arctan(sw["bump_L"] / half))     # = +roll
    phi_R = np.degrees(np.arctan(sw["bump_R"] / half))     # = -roll
    road_L = sw["camber_L"] + phi_L
    road_R = sw["camber_R"] + phi_R

    i0 = int(np.argmin(np.abs(roll)))
    d_road_L = np.gradient(road_L, roll)
    d_road_R = np.gradient(road_R, roll)
    roll_steer_L = np.gradient(sw["toe_L"], roll)
    roll_steer_R = np.gradient(sw["toe_R"], roll)

    return dict(
        roll=roll, half_track=float(half),
        camber_L=sw["camber_L"], camber_R=sw["camber_R"],      # vs chassis
        road_L=road_L, road_R=road_R,                          # vs ROAD (what grips)
        toe_L=sw["toe_L"], toe_R=sw["toe_R"],
        bump_L=sw["bump_L"], bump_R=sw["bump_R"],
        rc_x=sw["rc_x"], rc_z=sw["rc_z"],
        d_road_L=d_road_L, d_road_R=d_road_R,
        roll_steer_L=roll_steer_L, roll_steer_R=roll_steer_R,
        # headline scalars at ride height (LEFT = the compressing/outside wheel)
        recovery_pct=float((1.0 - d_road_L[i0]) * 100.0),
        roll_steer=float(roll_steer_L[i0]),
        rc0=float(sw["rc_z"][i0]),
        static_camber=static_camber,
    )


# =====================================================================
#  WHERE CAMBER RECOVERY COMES FROM  -  and a fast evaluator for maps
# =====================================================================
#
# THE CHAIN, in maths.
#
#   what the tyre works at :  gamma_road = gamma_chassis + phi
#   so                        d(gamma_road)/d(phi) = d(gamma_chassis)/d(phi) + 1
#   recovery                = 1 - d(gamma_road)/d(phi)  =  -d(gamma_chassis)/d(phi)
#
#   and the body roll reaches the wheel only as travel:  z = (t/2)*tan(phi)
#   so   d(gamma_chassis)/d(phi) = (d(gamma)/dz) * (t/2) * (pi/180)
#
#   ==>   RECOVERY = -(d(gamma)/dz) * (t/2) * (pi/180) * 100 %          [EXACT]
#
# So recovery is made of exactly TWO things: the linkage's CAMBER GAIN dgamma/dz,
# and the HALF-TRACK that converts body roll into wheel travel. Nothing else.
#
# Camber gain in turn is a pure front-view lever: the upright rotates about the
# front-view instant centre, so a wheel-centre rise dz rotates it by dz / L,
# where L is the horizontal distance from the IC to the wheel:
#
#         d(gamma)/dz  ~=  -1 / L        ==>    RECOVERY ~= (t/2) / FVSA
#
# i.e. HALF-TRACK DIVIDED BY SWING-ARM LENGTH. Short swing arm -> strong recovery.
# (Rear: (t/2)/FVSA = 19.1 % vs the exact 20.4 % -- good. The approximation
# degrades when the arms are near-parallel and the IC runs off to infinity, as on
# this car's front, where the instantaneous IC is 12.5 m out and hyper-sensitive.)
#
# WHICH HARD POINTS MOVE IT (measured, one at a time, 20 mm):
#   wishbone points (inboard AND outboard) -- PRIMARY. They set the IC, hence
#       camber gain AND the roll centre. e.g. UBJ +20 mm in Z: recovery 20->48 %,
#       but the roll centre also jumps -52 -> +37 mm. THE TWO ARE COUPLED: both
#       are read off the same instant centre, so you cannot dial one without
#       moving the other. That is the central trade-off of the redesign.
#   toe link -- SECONDARY. It twists the upright about the (tilted) kingpin axis,
#       and twisting about a tilted axis does change camber -- the same
#       camber-by-steer effect studied on the front. It does NOT move the IC at
#       all (FVSA and RC are unchanged by any toe-link move), but it buys camber
#       at the price of large bump-steer changes.
#
def camber_gain_fast(pts, static_camber=REAR_STATIC_CAMBER, dth=0.6):
    """(camber gain, bump steer) [deg/mm] by central difference on the wishbone
    angle -- no travel sweep, no interpolation.

    543x faster than bump_sweep and agrees with it to 0.00 % on camber gain
    (0.03 % on bump steer), which is what makes a 2-D parameter map practical.
    Raises ValueError if the linkage will not assemble.
    """
    g = corner_geometry(solver_points(pts), static_camber)
    sp, sm = solve_at(g, +dth, 0.0), solve_at(g, -dth, 0.0)
    dz = sp["WC"][2] - sm["WC"][2]
    cam = [np.degrees(np.arcsin(np.clip(-s["s"][2], -1.0, 1.0))) for s in (sp, sm)]
    toe = [np.degrees(np.arctan2(s["s"][1], s["s"][0])) for s in (sp, sm)]
    return (cam[0] - cam[1]) / dz, (toe[0] - toe[1]) / dz


def spin_axis(camber_deg, toe_deg=0.0):
    """Wheel spin axis (pointing outboard, +X) for a given static camber AND toe.

        s = (cos g cos t,  cos g sin t,  -sin g)

    which inverts exactly to  camber = arcsin(-s_z),  toe = atan2(s_y, s_x)
    -- the project's standard conventions. + toe = TOE-IN for this right wheel.

    WHY THIS EXISTS -- a conceptual point worth stating plainly:
    STATIC TOE IS A SETUP INPUT, exactly like static camber. It is NOT something
    you get by moving the toe link in this model. `corner_geometry` builds the
    reference wheel orientation from the static camber alone, so the as-built
    state is toe = 0 by construction, and sliding the toe-link ball joint just
    re-defines that reference. (An earlier version of this function tried to
    bisect the toe link's position to "reach" a static toe and correctly found it
    could not -- the bracket returned 0 at both ends.)

    The real division of labour:
        static toe        <- the threaded sleeve = a SETUP number (here)
        bump / roll steer <- WHERE the toe link is = the hard points
    """
    g, t = np.radians(float(camber_deg)), np.radians(float(toe_deg))
    return np.array([np.cos(g) * np.cos(t), np.cos(g) * np.sin(t), -np.sin(g)])


def rear_corner(pts=None, static_camber=REAR_STATIC_CAMBER, static_toe=0.0):
    """Solver geometry for the rear corner, carrying BOTH setup angles."""
    if pts is None:
        pts = load_rear()
    g = corner_geometry(solver_points(pts), static_camber)
    g["s0"] = spin_axis(static_camber, static_toe)      # toe baked into the wheel
    return g


# =====================================================================
#  JACKING  -  the vertical force the roll centre feeds into the body
# =====================================================================
#
# WHAT JACKING IS
# ---------------
# A tyre's lateral force has to get from the contact patch into the body. The
# suspension links can only push or pull along their own directions, so in the
# front view the whole tyre force is carried along the FORCE LINE -- the line
# from the contact patch to that side's instant centre. The roll centre is just
# where those two force lines cross.
#
# That line is TILTED, so the force it carries has a VERTICAL component. That
# vertical component is the jacking force. Its sign is decided entirely by
# whether the roll centre is above or below the ground.
#
# THE DERIVATION (left turn; outside = right wheel; RC on the centreline at h)
# ---------------------------------------------------------------------------
#   right patch at (+t/2, 0), force line to (0, h)  ->  direction (-t/2, h)
#   its X-component must carry the tyre's lateral force F_out:
#         k*(-t/2) = -F_out            =>  k = 2 F_out / t
#         Z-component = k*h            =   2 F_out h / t     (UP for h > 0)
#   left patch at (-t/2, 0), direction (+t/2, h), lateral force F_in:
#         Z-component                  =  -2 F_in  h / t     (DOWN for h > 0)
#
#         NET JACKING FORCE  =  (2 h / t) * (F_out - F_in)
#
# READ THAT CAREFULLY: jacking is driven by the DIFFERENCE in lateral force
# between the two tyres, not by the total. Two equally-loaded tyres jack nothing,
# whatever the roll-centre height. The difference only appears because load
# transfer loads the outside tyre more than the inside.
#
# With lateral force proportional to vertical load (constant mu),
#         F_out - F_in = 2 a dW ,      dW = W_axle a h_CG / t
#   =>    F_jack = 4 a^2 W_axle h_CG h_RC / t^2
#
# so jacking is LINEAR in roll-centre height and QUADRATIC in lateral g.
#
# CAVEAT, stated honestly: real tyres are load-sensitive (mu falls as load rises),
# so the true F_out - F_in is smaller than the constant-mu model and this is an
# UPPER BOUND. The model also assumes both tyres are still on the ground; the
# inside lifts when dW >= W_axle/2, i.e. a >= t/(2 h_CG) ~ 1.8 g for this car.
#
def jacking_force(h_rc_mm, lat_g, axle_weight_N, h_cg_mm, track_mm):
    """Net vertical force [N] the links feed into the sprung mass, + = UP (jacks
    the car up). Pure geometry plus a load-transfer model -- no springs involved."""
    h_rc, h_cg, t = np.asarray(h_rc_mm, float), float(h_cg_mm), float(track_mm)
    return 4.0 * lat_g ** 2 * axle_weight_N * h_cg * h_rc / (t * t)


def jacking_ride_height(h_rc_mm, lat_g, axle_weight_N, h_cg_mm, track_mm,
                        wheel_rate_N_per_mm):
    """Ride-height change [mm] from jacking, + = the car RISES.

    Only the SPRINGS resist it: jacking pushes both corners the same way, and an
    anti-roll bar does nothing against symmetric (heave) motion. So the axle's
    resisting rate is simply twice the wheel rate.
    """
    f = jacking_force(h_rc_mm, lat_g, axle_weight_N, h_cg_mm, track_mm)
    return f / (2.0 * float(wheel_rate_N_per_mm))


def wheel_rate_from_frequency(sprung_corner_kg, ride_freq_hz):
    """Wheel rate [N/mm] from a target ride frequency -- the way race engineers
    actually spec a spring:  f = (1/2pi) sqrt(K/m)."""
    return sprung_corner_kg * (2.0 * np.pi * float(ride_freq_hz)) ** 2 / 1000.0


def recovery_from_gain(camber_gain, half_track):
    """The exact definition: recovery [%] from camber gain [deg/mm] and half-track."""
    return -camber_gain * half_track * np.pi / 180.0 * 100.0


if __name__ == "__main__":
    g = static_geometry()
    print("REAR CORNER - static geometry")
    print("=" * 60)
    print(f"  static camber   {g['static_camber']:+.2f} deg   (PLACEHOLDER - measure in CAD)")
    print(f"  toe             {g['toe']:+.2f} deg   (ideal toe link assumption)")
    print(f"  KPI             {g['kpi']:+.2f} deg")
    print(f"  caster          {g['caster']:+.2f} deg")
    print(f"  kingpin length  {g['kingpin_len']:.1f} mm")
    print(f"  upper / lower   {g['upper_arm']:.1f} / {g['lower_arm']:.1f} mm")
    print(f"  scrub / trail   {g['scrub']:+.1f} / {g['trail']:+.1f} mm")
    print(f"  FVSA            {g['fvsa']:.0f} mm")
    print(f"  roll centre     {g['rc_height']:+.1f} mm")
    print(f"  IC (X, Z)       ({g['IC'][0]:+.0f}, {g['IC'][1]:+.0f}) mm")
    print(f"  track           {g['track']:.1f} mm")
    print(f"  WC height       {g['wc_height']:.1f} mm")
    print(f"  wheelbase       {g['wheelbase']:.1f} mm")
