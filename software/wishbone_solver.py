# -*- coding: utf-8 -*-
"""
Double-wishbone position solver -- THE TOOL's kinematic core.
=============================================================

Fourth core module (after steer_camber, scrub_trail, jacking). Everything so far
held the wishbones still and rotated the wheel about a FIXED kingpin axis. This
module finally moves the wishbones: it solves the position of the whole corner as
the wheel takes BUMP travel (and, optionally, rack travel), which is what unlocks
camber gain, bump steer, the exact motion ratio and the roll centre.

GEOMETRY-AGNOSTIC BY DESIGN. The solver takes a dict of hard points and knows
nothing about whose car they are. Feed it the Huracan baseline, a clean-sheet
design, or real Temerario data when it exists -- same tool. That is the project's
architecture: the tool is the product; any particular geometry is just an input.

THE MECHANISM (how a double wishbone actually moves)
----------------------------------------------------
Each wishbone is a rigid triangle hinged to the chassis along the axis through
its two inner pivots -- one degree of freedom each, a pure rotation. The upright
is a rigid body carrying UBJ, LBJ, the tie-rod outer and the wheel. With the rack
held, the corner has ONE degree of freedom: bump.

The solve, per position (all three steps are CLOSED FORM -- no iteration):

  STEP 1  pick the lower wishbone angle theta_L. The LBJ (and the lower shock
          mount, which rides on this wishbone) swing on circles about the lower
          inner axis.
  STEP 2  the upright is rigid, so the UBJ must stay at its fixed distance from
          the LBJ -- AND it must stay on its own circle about the upper inner
          axis. A point on a circle at a prescribed distance from an external
          point reduces to  A cos(t) + B sin(t) = C,  solved exactly.
  STEP 3  the upright can still spin about the LBJ-UBJ (kingpin) line. The
          tie rod pins it: the tie-rod outer must stay at tie-rod length from
          the (chassis-fixed) tie-rod inner. Same closed form again.

Three solved points (LBJ, UBJ, TRO) fix the upright's rigid transform; the wheel
centre and spin axis follow. Camber, toe, KPI, caster, scrub, trail, patch
position, damper length -- all drop out at every bump position.

WHY THIS IS ALSO EXACTLY CONSISTENT WITH THE EARLIER MODULES
------------------------------------------------------------
Steering at fixed bump leaves both ball joints where they are, so the upright
rotates EXACTLY about the LBJ-UBJ line -- which is precisely the model
steer_camber and scrub_trail already use. The validation suite exploits this:
the full solver, driven by the rack, must reproduce those modules to machine
precision. Different code path, same physics.

OUTPUT CONVENTIONS (CAD frame: X outboard+, Y forward+, Z up; front-RIGHT corner)
    bump    [mm]  wheel-centre rise relative to the chassis (+ = compression)
    camber  [deg] arcsin(-s_z), + = top of wheel leans outboard
    toe     [deg] atan2(s_y, s_x), + = TOE-IN for this right wheel
    KPI/caster    signed, same convention as everywhere in the project
    MR      [-]   damper compression per mm of bump (exact, from the sweep)
    rc_h    [mm]  front-view roll-centre height, from the numerically-exact
                  instant centre (see roll_centre_curve)
"""
import numpy as np

from scrub_trail import (rotate_about_line, kpi_caster, contact_patch,
                         wheel_spin_axis, wheel_radius)


# ---------------------------------------------------------------- closed forms
def _circle(P0, A, u):
    """The circle a point P0 sweeps when rotating about the line through A with
    unit direction u. Returns (c, r, e1, e2): centre, radius, and in-plane unit
    vectors with P(t) = c + r*(e1*cos t + e2*sin t), P(0) = P0."""
    w = np.asarray(P0, float) - A
    c = A + np.dot(w, u) * u
    r1 = P0 - c
    r = np.linalg.norm(r1)
    e1 = r1 / r
    e2 = np.cross(u, e1)
    return c, r, e1, e2


def solve_on_circle(P0, A, u, T, dist, guess_deg=0.0):
    """Where on its circle about line (A, u) must the point P0 sit so that it is
    exactly `dist` from the external point T?

    Substituting the circle into |P(t) - T|^2 = dist^2 collapses to
        A' cos t + B' sin t = C'
    which has the closed-form solutions  t = phi +/- arccos(C'/K). Of the two
    intersections, the one nearest `guess_deg` is returned (continuation keeps
    the sweep on the physical branch).

    Returns (t_deg, point). Raises if the circle cannot reach that distance --
    which physically means the linkage would have to separate (travel limit).
    """
    A = np.asarray(A, float)
    u = np.asarray(u, float)
    T = np.asarray(T, float)
    c, r, e1, e2 = _circle(P0, A, u)
    d = c - T
    Ac = 2.0 * r * np.dot(d, e1)
    Bc = 2.0 * r * np.dot(d, e2)
    Cc = dist ** 2 - np.dot(d, d) - r ** 2
    K = np.hypot(Ac, Bc)
    x = Cc / K
    if abs(x) > 1.0 + 1e-9:
        raise ValueError(f"linkage cannot reach: |{x:.6f}| > 1 (travel limit)")
    x = np.clip(x, -1.0, 1.0)
    phi = np.arctan2(Bc, Ac)
    g = np.radians(guess_deg)
    best = None
    for t in (phi + np.arccos(x), phi - np.arccos(x)):
        dt = (t - g + np.pi) % (2 * np.pi) - np.pi     # wrapped distance to guess
        if best is None or abs(dt) < abs(best[1]):
            best = (g + dt, dt)
    t = best[0]
    P = c + r * (e1 * np.cos(t) + e2 * np.sin(t))
    return np.degrees(t), P


def _frame(p, q, r):
    """Right-handed orthonormal frame from three non-collinear points."""
    x = q - p
    x = x / np.linalg.norm(x)
    z = np.cross(x, r - p)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rigid_map(p0, q0, r0, p1, q1, r1):
    """The rigid transform (R, t) taking reference points (p0,q0,r0) onto
    (p1,q1,r1). Apply to a point as R @ P + t, to a direction as R @ v."""
    M0 = _frame(np.asarray(p0, float), np.asarray(q0, float), np.asarray(r0, float))
    M1 = _frame(np.asarray(p1, float), np.asarray(q1, float), np.asarray(r1, float))
    R = M1 @ M0.T
    t = np.asarray(p1, float) - R @ np.asarray(p0, float)
    return R, t


# ---------------------------------------------------------------- the corner
def corner_geometry(pts, static_camber_deg, shock_on=None):
    """Package a corner's hard points for the solver. `pts` is the dict from
    config._load_front_points(); nothing car-specific lives in this module.

    `shock_on` says what the damper's LOWER mount is bolted to -- "lower" arm,
    "upper" arm, or the "upright". It matters because the motion ratio is squared
    into the wheel rate. Left as None it is chosen by PROXIMITY: whichever of the
    two ball joints the mount actually sits next to. On this car that picks the
    lower arm at the front (9.5 mm from the LBJ) and the upper arm at the rear
    (21 mm from the UBJ vs 153 mm from the LBJ), which is what the CAD shows.
    Pass it explicitly to override.
    """
    g = dict(
        UIF=np.asarray(pts["Upper wishbone - inner front"], float),
        UIR=np.asarray(pts["Upper wishbone - inner rear"], float),
        LIF=np.asarray(pts["Lower wishbone - inner front"], float),
        LIR=np.asarray(pts["Lower wishbone - inner rear"], float),
        UBJ=np.asarray(pts["Upper wishbone - outer (UBJ)"], float),
        LBJ=np.asarray(pts["Lower wishbone - outer (LBJ)"], float),
        TRI=np.asarray(pts["Tie rod - inner"], float),
        TRO=np.asarray(pts["Tie rod - outer"], float),
        WC=np.asarray(pts["Wheel centre"], float),
        SL=np.asarray(pts["Lower Shock mount"], float),
        SU=np.asarray(pts["Upper Shock mount"], float),
        camber0=float(static_camber_deg),
    )
    g["s0"] = wheel_spin_axis(static_camber_deg)          # upright-fixed direction
    g["R_wheel"] = wheel_radius(g["WC"], static_camber_deg)
    # chassis-fixed hinge axes
    ax_u = g["UIR"] - g["UIF"]
    g["ax_u"] = ax_u / np.linalg.norm(ax_u)
    ax_l = g["LIR"] - g["LIF"]
    g["ax_l"] = ax_l / np.linalg.norm(ax_l)
    # invariant lengths
    g["len_upright"] = np.linalg.norm(g["UBJ"] - g["LBJ"])
    g["len_tierod"] = np.linalg.norm(g["TRO"] - g["TRI"])
    if shock_on is None:
        d_low = np.linalg.norm(g["SL"] - g["LBJ"])
        d_up = np.linalg.norm(g["SL"] - g["UBJ"])
        shock_on = "lower" if d_low <= d_up else "upper"
    g["shock_on"] = shock_on
    return g


def solve_at(g, theta_L_deg, rack_mm=0.0, thU_guess=0.0, thT_guess=0.0):
    """Solve the corner for one lower-wishbone angle. Returns a dict with every
    solved point plus (thU, thT) for sweep continuation. thT is the upright's
    rotation about the kingpin line relative to reference -- at zero bump this
    IS the steer angle."""
    TRI = g["TRI"] + np.array([rack_mm, 0.0, 0.0])       # rack slides laterally

    # STEP 1 - lower wishbone
    LBJ = rotate_about_line(g["LBJ"], g["LIF"], g["ax_l"], theta_L_deg)[0]

    # STEP 2 - upper wishbone angle from the rigid upright length
    thU, UBJ = solve_on_circle(g["UBJ"], g["UIF"], g["ax_u"],
                               LBJ, g["len_upright"], thU_guess)

    # STEP 3 - upright spin about the kingpin line from the tie rod.
    # Seed the TRO's circle with the reference TRO carried over by the minimal
    # (zero-twist) rotation aligning the old kingpin direction onto the new one;
    # the solved angle thT is then the upright's TWIST relative to reference --
    # at zero bump that is exactly the steer angle.
    kp = UBJ - LBJ
    kp = kp / np.linalg.norm(kp)
    thT, TRO = solve_on_circle(_align_seed(g, LBJ, UBJ), LBJ, kp,
                               TRI, g["len_tierod"], thT_guess)

    # rigid transform of the whole upright from its three solved points
    R, t = rigid_map(g["LBJ"], g["UBJ"], g["TRO"], LBJ, UBJ, TRO)
    WC = R @ g["WC"] + t
    s = R @ g["s0"]

    # STEP 4 - carry the damper's lower mount on whatever it is bolted to.
    # This USED to be hard-coded to the lower wishbone. That is right for the
    # front (its mount sits 9.5 mm from the LBJ) but wrong for the rear, whose
    # mount is 21 mm from the UPPER ball joint and 153 mm from the lower --
    # rotating it about the lower arm's axis swung it 10 mm inboard per 12 mm of
    # travel and produced a motion ratio of 1.297, i.e. a damper moving MORE than
    # the wheel. On the correct arm it is 0.764. MR is SQUARED into the spring
    # rate, so the old assumption was a factor of 2.9 on every rear spring.
    mode = g.get("shock_on", "lower")
    if mode == "lower":
        SL = rotate_about_line(g["SL"], g["LIF"], g["ax_l"], theta_L_deg)[0]
    elif mode == "upper":
        SL = rotate_about_line(g["SL"], g["UIF"], g["ax_u"], thU)[0]
    elif mode == "upright":
        SL = R @ g["SL"] + t
    else:
        raise ValueError(f"shock_on must be lower/upper/upright, got {mode!r}")

    return dict(LBJ=LBJ, UBJ=UBJ, TRO=TRO, WC=WC, SL=SL, s=s,
                thU=thU, thT=thT, TRI=TRI)


def _align_seed(g, LBJ, UBJ):
    """Reference TRO carried to the new kingpin line with zero extra twist:
    rotate (TRO - LBJ) by the minimal rotation taking the old kingpin direction
    onto the new one, then anchor at the new LBJ."""
    v0 = g["UBJ"] - g["LBJ"]
    v0 = v0 / np.linalg.norm(v0)
    v1 = UBJ - LBJ
    v1 = v1 / np.linalg.norm(v1)
    c = np.cross(v0, v1)
    s = np.linalg.norm(c)
    if s < 1e-12:                                       # axes already parallel
        return LBJ + (g["TRO"] - g["LBJ"])
    ang = np.degrees(np.arctan2(s, np.dot(v0, v1)))
    return LBJ + rotate_about_line(g["TRO"] - g["LBJ"], np.zeros(3), c / s, ang)[0]


def solve_sweep(g, bump_grid_mm, rack_mm=0.0, theta_span_deg=14.0, n_theta=561):
    """Sweep the lower-wishbone angle, then re-index every output onto a uniform
    BUMP grid (wheel-centre rise, + = compression) by interpolation.

    Returns a dict of arrays over bump_grid_mm:
        camber, toe, kpi, caster [deg] · scrub, trail, patch_x, patch_y [mm]
        damper [mm] · MR [-] · plus the raw solved points for plotting.
    """
    thetas = np.linspace(-theta_span_deg, theta_span_deg, n_theta)
    rows = []
    thU, thT = 0.0, 0.0
    for th in thetas:
        st = solve_at(g, th, rack_mm, thU, thT)
        thU, thT = st["thU"], st["thT"]
        rows.append(st)

    z0 = g["WC"][2]
    travel = np.array([r["WC"][2] - z0 for r in rows])
    if not np.all(np.diff(travel) > 0):
        # keep the monotonic branch around zero travel
        i0 = int(np.argmin(np.abs(travel)))
        lo, hi = i0, i0
        while lo > 0 and travel[lo - 1] < travel[lo]:
            lo -= 1
        while hi < len(travel) - 1 and travel[hi + 1] > travel[hi]:
            hi += 1
        rows = rows[lo:hi + 1]
        travel = travel[lo:hi + 1]

    def series(fn):
        return np.array([fn(r) for r in rows])

    camber = series(lambda r: np.degrees(np.arcsin(np.clip(-r["s"][2], -1, 1))))
    toe = series(lambda r: np.degrees(np.arctan2(r["s"][1], r["s"][0])))
    kpi = series(lambda r: kpi_caster(r["LBJ"], r["UBJ"])[0])
    caster = series(lambda r: kpi_caster(r["LBJ"], r["UBJ"])[1])
    damper = series(lambda r: np.linalg.norm(g["SU"] - r["SL"]))
    wcx = series(lambda r: r["WC"][0])
    wcz = series(lambda r: r["WC"][2])
    sx = series(lambda r: r["s"][0])
    sz = series(lambda r: r["s"][2])

    patch = np.array([contact_patch(r["WC"], spin_axis=r["s"], R=g["R_wheel"])
                      for r in rows])
    # ground point of the (moving!) kingpin axis, for scrub/trail
    Gx, Gy = [], []
    for r in rows:
        d = r["UBJ"] - r["LBJ"]
        t = -r["LBJ"][2] / d[2]
        Gx.append(r["LBJ"][0] + t * d[0])
        Gy.append(r["LBJ"][1] + t * d[1])
    Gx, Gy = np.array(Gx), np.array(Gy)

    out = dict(bump=np.asarray(bump_grid_mm, float))
    b = out["bump"]
    if b.min() < travel.min() - 1e-6 or b.max() > travel.max() + 1e-6:
        raise ValueError(f"bump grid [{b.min():.1f}, {b.max():.1f}] exceeds the "
                         f"solvable travel [{travel.min():.1f}, {travel.max():.1f}] mm")

    def onto(v):
        return np.interp(b, travel, v)

    out["camber"] = onto(camber)
    out["toe"] = onto(toe)
    out["kpi"] = onto(kpi)
    out["caster"] = onto(caster)
    out["scrub"] = onto(patch[:, 0] - Gx)
    out["trail"] = onto(Gy - patch[:, 1])
    out["patch_x"] = onto(patch[:, 0])
    out["patch_y"] = onto(patch[:, 1])
    out["wc_x"] = onto(wcx)
    out["wc_z"] = onto(wcz)
    out["s_x"] = onto(sx)
    out["s_z"] = onto(sz)
    out["damper"] = onto(damper)
    out["MR"] = -np.gradient(out["damper"], b)          # compression per mm bump
    return out


def roll_centre_curve(sw, track_mm):
    """Front-view instant centre and roll-centre height vs bump, computed from
    the solved motion itself (numerically exact for this 3D geometry, no 2D
    swing-arm shortcut).

    In the front view the upright rotates at  omega = d(beta)/d(bump)  where
    beta = atan2(s_z, s_x) is the spin-axis angle, while the wheel centre
    translates at (d wc_x, d wc_z)/d(bump). Rigid-body kinematics then places
    the instant centre at
        IC_x = wc_x - vz/omega ,   IC_z = wc_z + vx/omega
    and the roll centre is where the patch->IC line crosses the car centreline:
        rc_h = IC_z * patch_x / (patch_x - IC_x)
    (track_mm is carried for the caller's plots; the construction itself only
    needs the centreline x = 0.)
    """
    b = sw["bump"]
    beta = np.arctan2(sw["s_z"], sw["s_x"])
    omega = np.gradient(beta, b)                        # rad per mm bump
    vx = np.gradient(sw["wc_x"], b)
    vz = np.gradient(sw["wc_z"], b)                    # ~1 by construction
    with np.errstate(divide="ignore", invalid="ignore"):
        IC_x = sw["wc_x"] - vz / omega
        IC_z = sw["wc_z"] + vx / omega
        rc_h = IC_z * sw["patch_x"] / (sw["patch_x"] - IC_x)
    return dict(IC_x=IC_x, IC_z=IC_z, rc_h=rc_h, omega=omega)


def linkage_roll_centre(g, bump_grid_mm, dtheta=0.25):
    """Roll-centre height vs bump from the LINKAGE-ONLY instant centre.

    The ball joints lie ON the kingpin line, so the tie-rod twist does not move
    them: their motion is pure wishbone linkage. The front-view IC is where the
    perpendiculars to the LBJ and UBJ velocities intersect -- equivalently (for
    the pure-Y hinge axes of this car) where the two projected link lines cross.
    This is the CLASSICAL roll-centre construction, computed from the solver.

    Distinct from roll_centre_curve(), which uses the wheel's ACTUAL front-view
    rotation including the bump-steer twist. On a geometry with large bump steer
    the two differ substantially -- the classical construction quietly assumes
    bump steer is negligible.
    """
    out_h = []
    thU = thT = 0.0
    # map bump -> theta by solving on a fine sweep first
    thetas = np.linspace(-14, 14, 561)
    trav, ths = [], []
    for th in thetas:
        s = solve_at(g, th, 0.0, thU, thT)
        thU, thT = s["thU"], s["thT"]
        trav.append(s["WC"][2] - g["WC"][2])
        ths.append(th)
    trav = np.array(trav)
    ths = np.array(ths)
    for b in np.atleast_1d(bump_grid_mm):
        th_b = np.interp(b, trav, ths)
        sp = solve_at(g, th_b + dtheta)
        sm = solve_at(g, th_b - dtheta)
        s0 = solve_at(g, th_b)
        vL = (sp["LBJ"] - sm["LBJ"])[[0, 2]]          # front-view velocities
        vU = (sp["UBJ"] - sm["UBJ"])[[0, 2]]
        pL = s0["LBJ"][[0, 2]]
        pU = s0["UBJ"][[0, 2]]
        # IC: intersect the lines through each BJ perpendicular to its velocity
        nL = np.array([-vL[1], vL[0]])
        nU = np.array([-vU[1], vU[0]])
        A = np.array([[nL[0], -nU[0]], [nL[1], -nU[1]]])
        rhs = pU - pL
        tpar, _ = np.linalg.solve(A, rhs)
        IC = pL + tpar * nL
        patch = contact_patch(s0["WC"], spin_axis=s0["s"], R=g["R_wheel"])
        px = patch[0]
        d = IC - np.array([px, 0.0])
        out_h.append(0.0 + d[1] * (0.0 - px) / d[0])
    return np.array(out_h)


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    import config as cfg

    p = cfg._load_front_points()
    g = corner_geometry(p, cfg.STATIC_CAMBER)
    bump = np.linspace(-40, 40, 161)
    sw = solve_sweep(g, bump)
    rc = roll_centre_curve(sw, 2 * g["WC"][0])

    print("Wishbone solver - baseline check (front corner, CAD hard points)")
    print("=" * 76)
    i0 = int(np.argmin(np.abs(bump)))
    print(f"  at zero bump: camber {sw['camber'][i0]:+.3f} (static {g['camber0']:+.1f})   "
          f"toe {sw['toe'][i0]:+.4f}   KPI {sw['kpi'][i0]:+.3f}   "
          f"caster {sw['caster'][i0]:+.3f}")
    print(f"  scrub {sw['scrub'][i0]:+.2f} mm   trail {sw['trail'][i0]:+.2f} mm   "
          f"MR {sw['MR'][i0]:.4f}   RC height {rc['rc_h'][i0]:+.1f} mm")
    print()
    print("  bump    camber     toe      caster     KPI      scrub     MR      RC_h")
    print("   [mm]    [deg]    [deg]     [deg]     [deg]      [mm]     [-]      [mm]")
    for bb in (-40, -20, 0, 20, 40):
        i = int(np.argmin(np.abs(bump - bb)))
        print(f"  {bb:+5.0f}  {sw['camber'][i]:+8.3f} {sw['toe'][i]:+8.4f}  "
              f"{sw['caster'][i]:+8.3f}  {sw['kpi'][i]:+8.3f}  {sw['scrub'][i]:+8.2f}  "
              f"{sw['MR'][i]:6.4f}  {rc['rc_h'][i]:+8.1f}")
    print()
    print("  camber gain  d(camber)/d(bump) at 0: "
          f"{np.gradient(sw['camber'], bump)[i0]:+.5f} deg/mm")
    print("  bump steer   d(toe)/d(bump)   at 0: "
          f"{np.gradient(sw['toe'], bump)[i0]:+.5f} deg/mm")
    print()
    # rack check: 20 mm of rack at zero bump vs the first-order steering arm
    g2 = corner_geometry(p, cfg.STATIC_CAMBER)
    st0 = solve_at(g2, 0.0, rack_mm=0.0)
    st20 = solve_at(g2, 0.0, rack_mm=20.0, thT_guess=0.0)
    print(f"  rack +20 mm at zero bump -> upright rotates {st20['thT']-st0['thT']:+.3f} deg "
          f"about the kingpin axis")
    d1, arm = cfg.steer_angle_for_rack(20.0)
    print(f"  first-order estimate (rack/arm)          -> {d1:+.3f} deg  "
          f"(arm {arm:.1f} mm)")
