# -*- coding: utf-8 -*-
"""
TWO-CORNER FRONT AXLE  -  instant centres, the TRUE roll centre, FVSA.
======================================================================

New physics on top of the validated single-corner cores. Everything before this
modelled ONE corner; a roll centre is a property of the AXLE, so it needs both.

WHY ONE CORNER IS NOT ENOUGH
-----------------------------
The roll centre is where the two "force lines" cross: the line from each contact
patch to that side's instant centre. With a symmetric car at zero roll they cross
on the centreline, so a single corner plus the centreline gives the right answer
-- that is the shortcut the earlier modules used. The moment the car ROLLS, or
the geometry is asymmetric, the two lines no longer cross on the centreline and
the roll centre MIGRATES sideways. Only a two-corner model can see that.

    IC_side   : instant centre of the wheel relative to the chassis, front view.
                Found from the actual solved motion: the perpendiculars to the
                LBJ and UBJ velocities intersect there. (The ball joints lie ON
                the kingpin axis, so tie-rod twist does not move them -- their
                motion is pure wishbone linkage. This is the classical
                construction, and validate_wishbone.py already checks it against
                the textbook 2-line recipe to 0.1 mm.)
    FVSA      : front-view swing-arm length = horizontal distance from the
                contact patch to its IC. Short FVSA -> strong camber gain.
                [Gillespie, Fundamentals of Vehicle Dynamics, ch.7 pp.242-245]
    RC        : intersection of (patch_L -> IC_L) and (patch_R -> IC_R).

TRIM STATE (vsusp convention, at the user's request)
-----------------------------------------------------
Three inputs, exactly like vsusp.com:
    left_bump, right_bump   extra wheel travel per side  [mm, + = compression]
    body_roll               chassis roll angle           [deg, + = onto the LEFT
                            i.e. a RIGHT-HAND turn; the LEFT wheels compress]

HOW THEY COMBINE -- and why roll is NOT a rigid rotation
---------------------------------------------------------
Rolling the body while the tyres stay on the road is EXACTLY the same motion as
giving the two wheels equal and opposite travel. So body roll is realised through
the suspension, not by spinning the whole car:

    effective bump L = left_bump  + (t/2)*tan(roll)
    effective bump R = right_bump - (t/2)*tan(roll)

(An earlier version rotated every point -- chassis, wheels AND contact patches --
by the roll angle. That is a rotisserie, not a car: it lifted the tyres off the
road and shrank the measured track by cos(roll). The validation caught it.)

With roll expressed this way the wheels stay on the ground by construction, the
track is invariant, and left_bump/right_bump remain free for pure heave and
single-wheel-bump studies on top of the roll.

MIRRORING
---------
Only the front-RIGHT corner was measured, so the left is normally its mirror
(negate X). Rather than re-derive every sign for a left-handed corner, the left
is solved by MIRRORING ITS POINTS INTO RIGHT-HAND FORM, running the validated
right-corner solver, and mirroring the results back. No new sign conventions,
no new code paths -- the same solver that passes 27 assertions does both sides.

Sign conventions after mirroring back:
    camber > 0  = top of THAT wheel leans away from the car centre (outboard)
    toe    > 0  = that wheel toes IN
Both are per-wheel and therefore symmetric, which is the standard convention.
"""
import os
import sys

import numpy as np

_SW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "software")
if _SW not in sys.path:
    sys.path.insert(0, _SW)

from wishbone_solver import corner_geometry, solve_at, solve_sweep   # noqa: E402
from scrub_trail import kpi_caster, _lowest_point_dir                 # noqa: E402


def _static_patch_x(pts, static_camber, flip):
    """Lateral position of a corner's contact patch at static, in car coords.
    A cambered wheel touches down R*sin(camber) to one side of its centre."""
    from scrub_trail import wheel_radius
    work = mirror_points(pts) if flip else pts
    WC = np.asarray(work["Wheel centre"], float)
    px = WC[0] - wheel_radius(WC, static_camber) * np.sin(np.radians(static_camber))
    return -px if flip else px


def _patch_3d(WC, spin_axis, R_wheel):
    """The wheel's lowest point, WITH its true height.

    scrub_trail.contact_patch() flattens z to 0 because it only ever asked where
    the patch is on the road. Here the height matters: with the chassis held
    fixed, a bumped wheel's lowest point sits ABOVE z = 0, and that offset is
    exactly what tells us how the body must move to put the tyres back on the
    road. Flattening it made body roll a silent no-op -- the validation caught it.
    """
    return np.asarray(WC, float) + R_wheel * _lowest_point_dir(spin_axis)

MIRROR = np.array([-1.0, 1.0, 1.0])          # X negated: right <-> left


def mirror_points(pts):
    """Mirror a corner's hard points across the car centreline."""
    return {k: np.asarray(v, float) * MIRROR for k, v in pts.items()}


# ---------------------------------------------------------------- one corner
def _travel_branch(g, rack_mm=0.0, span=28.0, n=1121):
    """The monotonic wheel-travel -> wishbone-angle branch around ride height.

    The solver is driven by the lower-wishbone angle, so to hit a target travel
    we sweep the angle once and keep the single-valued branch through zero. This
    map does NOT depend on body roll, so a roll sweep builds it ONCE per corner
    and reuses it (see roll_sweep) instead of rebuilding it at every roll angle.
    """
    thetas = np.linspace(-span, span, n)
    trav, thU, thT = [], 0.0, 0.0
    for th in thetas:
        s = solve_at(g, th, rack_mm, thU, thT)
        thU, thT = s["thU"], s["thT"]
        trav.append(s["WC"][2] - g["WC"][2])
    trav = np.asarray(trav)
    i0 = int(np.argmin(np.abs(trav)))
    lo, hi = i0, i0
    while lo > 0 and trav[lo - 1] < trav[lo]:
        lo -= 1
    while hi < len(trav) - 1 and trav[hi + 1] > trav[hi]:
        hi += 1
    return trav[lo:hi + 1], thetas[lo:hi + 1]


def _solve_corner_at_bump(g, bump_mm, rack_mm=0.0, span=28.0, n=1121, branch=None):
    """Solve one (right-handed) corner at a target wheel-centre rise. `branch` is
    an optional precomputed (travel, theta) map from _travel_branch; when omitted
    it is built here, so the default behaviour is byte-identical to before."""
    tr, th_branch = branch if branch is not None else _travel_branch(g, rack_mm, span, n)
    if not (tr.min() - 1e-6 <= bump_mm <= tr.max() + 1e-6):
        raise ValueError(f"bump {bump_mm:+.1f} mm outside the solvable travel "
                         f"[{tr.min():.1f}, {tr.max():.1f}] mm")
    th_target = float(np.interp(bump_mm, tr, th_branch))
    return solve_at(g, th_target, rack_mm), th_target


def _corner_prep(pts, static_camber, flip, rack_mm=0.0, span=28.0, n=1121):
    """Everything about a corner that a roll sweep can reuse unchanged: the
    right-handed working points, its geometry, and the travel->angle branch."""
    work = mirror_points(pts) if flip else dict(pts)
    g = corner_geometry(work, static_camber)
    return dict(work=work, g=g, branch=_travel_branch(g, rack_mm, span, n), flip=flip)


def _instant_centre(g, theta, rack_mm=0.0, d=0.25):
    """Front-view IC of the wheel relative to the chassis, at this wishbone angle.

    Central difference on the two ball joints, then intersect the perpendiculars
    to their velocities. Returns (IC_x, IC_z) in the corner's own frame.
    """
    sp = solve_at(g, theta + d, rack_mm)
    sm = solve_at(g, theta - d, rack_mm)
    s0 = solve_at(g, theta, rack_mm)
    vL = (sp["LBJ"] - sm["LBJ"])[[0, 2]]
    vU = (sp["UBJ"] - sm["UBJ"])[[0, 2]]
    pL = s0["LBJ"][[0, 2]]
    pU = s0["UBJ"][[0, 2]]
    nL = np.array([-vL[1], vL[0]])            # perpendicular to each velocity
    nU = np.array([-vU[1], vU[0]])
    A = np.array([[nL[0], -nU[0]], [nL[1], -nU[1]]])
    if abs(np.linalg.det(A)) < 1e-12:         # parallel arms -> IC at infinity
        return np.array([np.inf, np.inf])
    t, _ = np.linalg.solve(A, pU - pL)
    return pL + t * nL


def _roll_matrix(phi_deg):
    """Rotation of the chassis about the LONGITUDINAL (Y) axis through the ground
    origin.

    SIGN: positive phi leans the body onto its LEFT, which is what happens in a
    RIGHT-HAND turn, and compresses the LEFT suspension (see axle_state, where
    bump_L = +dz_roll). An earlier docstring claimed "onto the RIGHT", which
    contradicted the arithmetic; validate_conventions.py now pins it down."""
    p = np.radians(phi_deg)
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ---------------------------------------------------------------- the axle
def axle_state(pts_R, pts_L, static_camber, left_bump=0.0, right_bump=0.0,
               body_roll=0.0, rack_mm=0.0, prep=None):
    """Solve the whole front axle at a trim state.

    pts_R : front-RIGHT hard points (the measured corner)
    pts_L : front-LEFT hard points, in their own (left-handed) coordinates
    prep  : optional {"R": ..., "L": ...} from _corner_prep, so a roll sweep can
            reuse each corner's travel->angle branch instead of rebuilding it.
            When None (the default) it is built here -- byte-identical to before.
    Returns a dict with, per side and for the axle:
        points, patch, IC, FVSA, camber, toe, caster, kpi, and the true RC.
    """
    # Body roll is realised through the suspension: rolling the car while the
    # tyres stay on the road IS equal-and-opposite wheel travel.
    # The lever arm is the CONTACT-PATCH separation, not the wheel-centre
    # separation -- the wheels tip about where they touch the road. With static
    # camber the two differ by R*sin(camber) per side (16 mm here), which is
    # enough to make "roll" and "differential bump" disagree if you use the
    # wrong one. The validation caught exactly that.
    half = 0.5 * (_static_patch_x(pts_R, static_camber, False)
                  - _static_patch_x(pts_L, static_camber, True))
    dz_roll = half * np.tan(np.radians(body_roll))
    bump_L = left_bump + dz_roll
    bump_R = right_bump - dz_roll

    out = {"trim": dict(left_bump=left_bump, right_bump=right_bump,
                        body_roll=body_roll, rack=rack_mm,
                        eff_bump_L=bump_L, eff_bump_R=bump_R)}

    for side, pts, bump, flip in (("R", pts_R, bump_R, False),
                                  ("L", pts_L, bump_L, True)):
        # solve every corner in RIGHT-handed form using the validated solver
        if prep is not None:
            work, g, branch = prep[side]["work"], prep[side]["g"], prep[side]["branch"]
        else:
            work = mirror_points(pts) if flip else dict(pts)
            g = corner_geometry(work, static_camber)
            branch = None
        st, theta = _solve_corner_at_bump(g, bump, rack_mm, branch=branch)
        ic = _instant_centre(g, theta, rack_mm)
        patch = _patch_3d(st["WC"], st["s"], g["R_wheel"])   # keeps its height

        cam = float(np.degrees(np.arcsin(np.clip(-st["s"][2], -1.0, 1.0))))
        toe = float(np.degrees(np.arctan2(st["s"][1], st["s"][0])))
        kpi, cas = kpi_caster(st["LBJ"], st["UBJ"])

        # mirror the geometry back to the car frame if this was the left corner
        m = MIRROR if flip else np.ones(3)
        pack = {k: st[k] * m for k in ("LBJ", "UBJ", "TRO", "WC", "SL")}
        pack["patch"] = patch * m
        pack["IC"] = np.array([ic[0], patch[1], ic[1]]) * m   # IC at the patch's Y
        # chassis-fixed pickups (they do not move with wheel travel)
        for nm, key in (("UIF", "Upper wishbone - inner front"),
                        ("UIR", "Upper wishbone - inner rear"),
                        ("LIF", "Lower wishbone - inner front"),
                        ("LIR", "Lower wishbone - inner rear"),
                        ("TRI", "Tie rod - inner"),
                        ("SU", "Upper Shock mount")):
            pack[nm] = np.asarray(pts[key], float)

        pack.update(camber=cam, toe=toe, kpi=kpi, caster=cas,
                    bump=bump, theta=theta,
                    fvsa=float(abs(pack["IC"][0] - pack["patch"][0])),
                    patch_height=float(pack["patch"][2]))
        out[side] = pack

    # ---- into the GROUND frame -------------------------------------------
    # The solver holds the CHASSIS fixed, so after travel the contact patches sit
    # at z = bump, not on the road. Real cars do it the other way round: the road
    # is fixed and the body moves. Rotate + translate the whole assembly so both
    # patches land on z = 0; the chassis is then correctly tilted by the roll it
    # has taken, and every height (RC especially) is measured FROM THE ROAD.
    PL, PR = out["L"]["patch"], out["R"]["patch"]
    dx, dz = PR[0] - PL[0], PR[2] - PL[2]
    phi_act = np.arctan2(dz, dx)                    # tilt of the patch-to-patch line
    # _roll_matrix(phi) sends (x, z) -> (c*x + s*z, -s*x + c*z), so rotating by
    # +phi_act (NOT -phi_act) is what drives z to zero on that line. Getting this
    # backwards doubles the tilt instead of removing it -- the validation caught it.
    Rg = _roll_matrix(np.degrees(phi_act))
    pivot = 0.5 * (PL + PR)
    VEC = ("LBJ", "UBJ", "TRO", "WC", "SL", "patch", "IC",
           "UIF", "UIR", "LIF", "LIR", "TRI", "SU")
    for side in ("L", "R"):
        for k in VEC:
            out[side][k] = Rg @ (out[side][k] - pivot)
        out[side]["patch_height"] = float(out[side]["patch"][2])
        # FVSA is a ground-frame length, so recompute it after the transform
        out[side]["fvsa"] = float(abs(out[side]["IC"][0] - out[side]["patch"][0]))
    # + roll = onto the RIGHT; the patch line tilts the opposite way, hence -
    out["roll_actual"] = float(-np.degrees(phi_act))

    out["roll_centre"] = _roll_centre(out["L"], out["R"])
    out["track"] = float(abs(out["R"]["patch"][0] - out["L"]["patch"][0]))
    return out


def _roll_centre(L, R):
    """TRUE roll centre: where the two force lines cross.

    Each side's force line runs from its contact patch to its instant centre.
    Returns (x, z) in the ground frame, or the centreline fallback if the lines
    are parallel (equal-length parallel arms -> IC at infinity).
    """
    pL, pR = L["patch"][[0, 2]], R["patch"][[0, 2]]
    dL = L["IC"][[0, 2]] - pL
    dR = R["IC"][[0, 2]] - pR
    if not (np.all(np.isfinite(dL)) and np.all(np.isfinite(dR))):
        return np.array([0.0, 0.0])
    A = np.array([[dL[0], -dR[0]], [dL[1], -dR[1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return np.array([np.nan, np.nan])
    t, _ = np.linalg.solve(A, pR - pL)
    return pL + t * dL


def roll_sweep(pts_R, pts_L, static_camber, roll_deg, track_mm=None, rack_mm=0.0):
    """Sweep body roll and track where the roll centre goes.

    `axle_state` already converts roll into equal-and-opposite wheel travel, so
    the tyres stay on the road throughout. `track_mm` is accepted for backward
    compatibility and is no longer used. Returns the RC position and each side's
    camber vs roll -- the classic roll-centre-migration plot.
    """
    roll_deg = np.atleast_1d(np.asarray(roll_deg, float))
    # build each corner's travel->angle branch ONCE; it does not move with roll,
    # so this turns an O(rolls x sweep) cost into O(sweep + rolls) -- ~10x faster
    prep = {"R": _corner_prep(pts_R, static_camber, False, rack_mm),
            "L": _corner_prep(pts_L, static_camber, True, rack_mm)}
    rc_x, rc_z, cam_L, cam_R = [], [], [], []
    toe_L, toe_R, bmp_L, bmp_R = [], [], [], []
    for phi in roll_deg:
        try:
            # body_roll alone now produces the equal-and-opposite travel
            st = axle_state(pts_R, pts_L, static_camber, body_roll=phi,
                            rack_mm=rack_mm, prep=prep)
            rc = st["roll_centre"]
            rc_x.append(rc[0])
            rc_z.append(rc[1])
            cam_L.append(st["L"]["camber"])
            cam_R.append(st["R"]["camber"])
            # toe is what ROLL STEER is made of -- on a non-steered (rear) axle it
            # is the stability parameter, so the sweep reports it alongside camber
            toe_L.append(st["L"]["toe"])
            toe_R.append(st["R"]["toe"])
            bmp_L.append(st["trim"]["eff_bump_L"])
            bmp_R.append(st["trim"]["eff_bump_R"])
        except ValueError:
            rc_x.append(np.nan); rc_z.append(np.nan)
            cam_L.append(np.nan); cam_R.append(np.nan)
            toe_L.append(np.nan); toe_R.append(np.nan)
            bmp_L.append(np.nan); bmp_R.append(np.nan)
    return dict(roll=roll_deg, rc_x=np.array(rc_x), rc_z=np.array(rc_z),
                camber_L=np.array(cam_L), camber_R=np.array(cam_R),
                toe_L=np.array(toe_L), toe_R=np.array(toe_R),
                bump_L=np.array(bmp_L), bump_R=np.array(bmp_R))


if __name__ == "__main__":
    import config as cfg

    pts_R = cfg._load_front_points()
    pts_L = mirror_points(pts_R)                       # symmetric car
    cam = cfg.STATIC_CAMBER

    st = axle_state(pts_R, pts_L, cam)
    print("TWO-CORNER FRONT AXLE - static check")
    print("=" * 72)
    print(f"  track (patch to patch) = {st['track']:.1f} mm")
    for s in ("L", "R"):
        d = st[s]
        print(f"  {s}: camber {d['camber']:+7.3f}  toe {d['toe']:+7.3f}  "
              f"KPI {d['kpi']:+7.3f}  caster {d['caster']:+6.3f}  "
              f"FVSA {d['fvsa']:8.1f} mm")
        print(f"      patch ({d['patch'][0]:+8.1f}, {d['patch'][2]:+6.1f})   "
              f"IC ({d['IC'][0]:+9.1f}, {d['IC'][2]:+8.1f})")
    rc = st["roll_centre"]
    print(f"  ROLL CENTRE  x = {rc[0]:+.2f} mm   height = {rc[1]:+.1f} mm")
    print()
    print("  symmetry check (should be mirror-exact at zero roll):")
    print(f"    camber L-R  {st['L']['camber']-st['R']['camber']:+.2e} deg")
    print(f"    FVSA  L-R   {st['L']['fvsa']-st['R']['fvsa']:+.2e} mm")
    print(f"    RC on centreline: x = {rc[0]:+.3e} mm")
    print()

    sw = roll_sweep(pts_R, pts_L, cam, np.linspace(-3, 3, 13), st["track"])
    print("  ROLL-CENTRE MIGRATION")
    print("    roll     RC x      RC z    camber L   camber R")
    for i in range(0, len(sw["roll"]), 2):
        print(f"   {sw['roll'][i]:+6.2f}  {sw['rc_x'][i]:+8.1f}  {sw['rc_z'][i]:+8.1f}  "
              f"{sw['camber_L'][i]:+8.3f}  {sw['camber_R'][i]:+8.3f}")
