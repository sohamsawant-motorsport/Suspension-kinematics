# -*- coding: utf-8 -*-
"""
Scrub radius, mechanical trail, and contact-patch motion under steer.
=====================================================================

Companion to steer_camber.py. Where that module answers "how does the wheel LEAN
when I steer", this one answers "where does the contact patch GO when I steer" --
which is what sets scrub radius, mechanical trail, track/wheelbase change and
steering kickback.

FRAME (the project's CAD frame, from hardpoints_fusion.csv)
-----------------------------------------------------------
    X = lateral (outboard +),  Y = longitudinal (forward +),  Z = vertical (up),  mm
    Z = 0 is the ground plane.

DEFINITIONS  (SAE J670, as given in Gillespie, Fundamentals of Vehicle Dynamics,
              Appendix A, sections 5.5.2.4 and 5.5.2.7)
--------------------------------------------------------------------------------
    Steering axis     : the line through LBJ and UBJ.
    G                 : where that line pierces the ground (Z = 0).
    P                 : the contact-patch centre -- SAE "centre of tyre contact".
                        NOT the wheel centre projected straight down! The wheel is
                        CAMBERED, i.e. tilted, so its lowest point is displaced
                        sideways:   P = WC + R * d_hat,  R = WC_z / cos(camber),
                        which for an unsteered wheel gives P_x = WC_x - R*sin(g).
                        On this car (static camber -2.7 deg) that is 15.9 mm, and
                        it moves the scrub radius from 129.1 to 144.9 mm. Confirmed
                        against the user's CAD, whose front-view sketch dimensions
                        the same distance as 146.213 mm (1.3 mm = mesh asymmetry).

    SCRUB RADIUS (= steering-axis offset at ground, kingpin offset at ground)
        scrub = P_x - G_x        + => contact patch is OUTBOARD of the axis
    MECHANICAL TRAIL (= caster offset at ground, caster trail)
        trail = G_y - P_y        + => axis pierces AHEAD of the patch (stabilising,
                                      like a castering trolley wheel)

WHY OFFSETS, NOT ANGLES
-----------------------
Scrub and trail are LENGTHS, and they are set by where the ball joints sit, not
only by how the axis is tilted. Sliding BOTH ball joints together:
    * laterally  (dx) -> changes scrub, leaves the KPI ANGLE untouched
    * longitudinally (dy) -> changes trail, leaves the CASTER ANGLE untouched
and, to the millimetre:
        scrub(dx) = scrub(0) - dx        trail(dy) = trail(0) + dy
because translating the axis translates G one-for-one while P stays put. This is
why angle and offset are independent design variables: caster angle buys camber,
caster trail buys steering torque, and you can tune them separately.

STEERING
--------
Steering rotates the WHOLE wheel about the steering axis -- its centre AND its
spin axis -- so the axis itself (and hence G) does NOT move. Because the spin axis
rotates, the camber CHANGES with steer (exactly what steer_camber computes), and
the patch's sideways camber offset rotates with it, so the patch is recomputed at
every angle rather than projected. Its motion gives directly:
    dP_x -> track width change while turning
    dP_y -> wheelbase change while turning
and the depth the patch would sink below the road is the JACKING (the corner must
lift by that much to keep the wheel down -- it moves wheel load on turn-in).

Run this file for a self-check against the CAD baseline.
"""
import numpy as np


# ---------------------------------------------------------------- axis geometry
def steering_axis(LBJ, UBJ):
    """Return (anchor, unit_direction) of the steering axis, pointing up."""
    LBJ = np.asarray(LBJ, float)
    u = np.asarray(UBJ, float) - LBJ
    return LBJ, u / np.linalg.norm(u)


def ground_point(LBJ, UBJ):
    """G: where the steering axis pierces the ground plane Z = 0."""
    A, u = steering_axis(LBJ, UBJ)
    t = -A[2] / u[2]
    return A + t * u


def wheel_spin_axis(camber_deg):
    """Unit spin axis of the wheel, in the CAD frame, at zero steer.

    Points outboard (+X). Positive camber = top of the wheel leans OUTBOARD, which
    tips the outboard-pointing spin axis DOWN:  s = (cos g, 0, -sin g).
    Consistent with steer_camber, where camber = arcsin(-s_z).
    """
    g = np.radians(float(camber_deg))
    return np.array([np.cos(g), 0.0, -np.sin(g)])


def wheel_radius(WC, camber_deg):
    """Physical wheel radius implied by the wheel-centre height and the static
    camber. A wheel tilted by g sits with its centre at R*cos(g) above the road,
    so R = WC_z / cos(g)."""
    return float(np.asarray(WC, float)[2] / np.cos(np.radians(float(camber_deg))))


def _lowest_point_dir(spin_axis):
    """Unit vector from the wheel centre to the LOWEST point of the wheel: the
    steepest-descent direction within the wheel plane (the plane normal to the
    spin axis). Projects straight-down onto that plane and normalises."""
    s = np.asarray(spin_axis, float)
    down = np.array([0.0, 0.0, -1.0])
    d = down - np.dot(down, s) * s
    n = np.linalg.norm(d)
    return d / n


def contact_patch(WC, camber_deg=0.0, spin_axis=None, R=None):
    """P: the contact-patch centre.

    IMPORTANT -- this is NOT simply the wheel centre projected down. A cambered
    wheel is tilted, so its lowest point is displaced sideways:

        P = WC + R * d_hat        d_hat = steepest descent in the wheel plane

    which for an unsteered wheel reduces to the closed form
        P_x = WC_x - R*sin(g),   P_z = WC_z - R*cos(g)
    i.e. NEGATIVE camber (top leaning inboard) pushes the patch OUTBOARD.

    This matters: scrub radius is defined TO THE CONTACT PATCH, so ignoring camber
    biases it directly. On this car, static camber -2.7 deg moves the patch 15.9 mm
    outboard and the scrub radius from 129.1 to 144.9 mm.

    Returns the point on the GROUND (z forced to 0). The z-component that was
    discarded is the jacking -- see corner_state.

    Pass spin_axis / R explicitly when the wheel has been steered (the spin axis
    rotates with it); otherwise they are built from camber_deg at zero steer.
    """
    WC = np.asarray(WC, float)
    if spin_axis is None:
        spin_axis = wheel_spin_axis(camber_deg)
    if R is None:
        R = wheel_radius(WC, camber_deg)
    P = WC + R * _lowest_point_dir(spin_axis)
    return np.array([P[0], P[1], 0.0])


def kpi_caster(LBJ, UBJ):
    """(KPI, caster) in deg from the ball joints, SIGNED, standard convention:
        KPI    > 0 = axis leans INBOARD at the top   (dX < 0)
        caster > 0 = axis leans REARWARD at the top  (dY < 0)

    This car's KPI comes out NEGATIVE (-5.27 deg): the CAD puts the UBJ outboard
    of the LBJ. That is confirmed against the CAD front view and is an artefact of
    an artistic mesh (real GT3s run positive KPI), but it is reported honestly --
    the sign is worth 144.9 mm vs 98.6 mm of scrub radius, and it flips whether
    camber grows or collapses with steer. Identical formula to config.stock_geometry_from_csv.
    """
    dX, dY, dZ = np.asarray(UBJ, float) - np.asarray(LBJ, float)
    return (float(np.degrees(np.arctan2(-dX, dZ))),
            float(np.degrees(np.arctan2(-dY, dZ))))


def scrub_radius(LBJ, UBJ, WC, camber_deg=0.0):
    """Scrub radius [mm] at zero steer. + => contact patch outboard of the axis.
    Measured TO THE CONTACT PATCH, so it depends on the static camber."""
    return float(contact_patch(WC, camber_deg)[0] - ground_point(LBJ, UBJ)[0])


def mechanical_trail(LBJ, UBJ, WC, camber_deg=0.0):
    """Mechanical trail [mm] at zero steer. + => axis ahead of the patch.
    Camber displaces the patch laterally, not longitudinally, so at zero steer
    trail is unaffected by camber -- but it is affected once steered, because the
    lateral offset rotates with the wheel."""
    return float(ground_point(LBJ, UBJ)[1] - contact_patch(WC, camber_deg)[1])


def offset_joints(LBJ, UBJ, dx=0.0, dy=0.0, dz=0.0):
    """Slide BOTH ball joints together -> moves the axis without rotating it, so
    KPI and caster ANGLES are unchanged. Returns (LBJ', UBJ')."""
    d = np.array([dx, dy, dz], float)
    return np.asarray(LBJ, float) + d, np.asarray(UBJ, float) + d


# ---------------------------------------------------------------- rotation
def rotate_about_line(P, A, u, steer_deg):
    """Rodrigues rotation of point(s) P about the line through A with unit
    direction u, by steer_deg. Vectorised over steer_deg -> shape (N, 3)."""
    P = np.asarray(P, float)
    A = np.asarray(A, float)
    u = np.asarray(u, float)
    d = np.radians(np.atleast_1d(np.asarray(steer_deg, float)))
    v = P - A                                   # position relative to the axis
    c = np.cos(d)[:, None]
    s = np.sin(d)[:, None]
    return A + v * c + np.cross(u, v) * s + u * np.dot(u, v) * (1.0 - c)


def corner_state(LBJ, UBJ, WC, steer_deg, camber_deg=0.0):
    """Everything about the corner as a function of steer angle.

    Steering rotates the WHOLE wheel about the steering axis -- both its centre
    AND its spin axis. G is fixed (the axis is the rotation axis). Because the
    spin axis rotates, the camber changes with steer (exactly what steer_camber
    computes), and the contact patch's camber offset rotates with it. So the patch
    is recomputed from the tilted wheel at every steer angle, not projected.

    Returns a dict of arrays (one entry per steer angle):
        P        (N,3) contact-patch centre, on the ground
        WC_rot   (N,3) rotated wheel centre
        camber   (N,)  camber [deg] at that steer (from the rotated spin axis)
        scrub    (N,)  P_x - G_x
        trail    (N,)  G_y - P_y
        jack     (N,)  how far the corner must lift to keep the wheel on the road
        d_track  (N,)  lateral patch movement from static (= track change)
        d_base   (N,)  longitudinal patch movement from static (= wheelbase change)
    """
    A, u = steering_axis(LBJ, UBJ)
    G = ground_point(LBJ, UBJ)
    WC = np.asarray(WC, float)
    R = wheel_radius(WC, camber_deg)

    WC_rot = rotate_about_line(WC, A, u, steer_deg)
    # the spin axis is a DIRECTION: rotate it about the axis direction only
    s_rot = rotate_about_line(wheel_spin_axis(camber_deg), np.zeros(3), u, steer_deg)

    P_raw = np.empty_like(WC_rot)
    for i in range(len(WC_rot)):
        P_raw[i] = WC_rot[i] + R * _lowest_point_dir(s_rot[i])
    P = np.column_stack([P_raw[:, 0], P_raw[:, 1], np.zeros(len(P_raw))])

    P0 = contact_patch(WC, camber_deg)
    return dict(
        G=G, P=P, WC_rot=WC_rot, R=R,
        camber=np.degrees(np.arcsin(np.clip(-s_rot[:, 2], -1.0, 1.0))),
        scrub=P[:, 0] - G[0],
        trail=G[1] - P[:, 1],
        # the wheel must stay on the road, so a patch that would sink below z=0
        # means the corner lifts by that amount
        jack=-P_raw[:, 2],
        d_track=P[:, 0] - P0[0],
        d_base=P[:, 1] - P0[1],
    )


# ---------------------------------------------------------------- worked example
def worked_example(LBJ, UBJ, WC, steer_deg=20.0, rear_y=None, camber_deg=0.0):
    """Print the scrub / trail / patch-motion calculation step by step, showing
    every intermediate number.

    The STEP labels match the comments in the functions above, the derivation in
    MATHS.md, and the study-guide sheet. Every number is checked by
    validate_scrub_trail.py against the real CAD hard points.
    """
    LBJ = np.asarray(LBJ, float); UBJ = np.asarray(UBJ, float); WC = np.asarray(WC, float)
    L = []
    add = L.append
    add("=" * 72)
    add(f"WORKED EXAMPLE - scrub, trail and patch motion    steer delta = {steer_deg} deg,"
        f"  static camber = {camber_deg:+.1f} deg")
    add("=" * 72)
    add("")
    add("GIVEN (front-right CAD hard points, mm; X outboard+, Y forward+, Z up)")
    add(f"   LBJ = ({LBJ[0]:9.3f}, {LBJ[1]:10.3f}, {LBJ[2]:8.3f})")
    add(f"   UBJ = ({UBJ[0]:9.3f}, {UBJ[1]:10.3f}, {UBJ[2]:8.3f})")
    add(f"   WC  = ({WC[0]:9.3f}, {WC[1]:10.3f}, {WC[2]:8.3f})")

    # STEP A ---------------------------------------------------------------
    d = UBJ - LBJ
    n = np.linalg.norm(d)
    u = d / n
    add("")
    add("STEP A  -  the steering axis direction")
    add(f"   d = UBJ - LBJ = ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})   |d| = {n:.3f} mm")
    add(f"   u = d/|d|     = ({u[0]:+.5f}, {u[1]:+.5f}, {u[2]:+.5f})")
    kpi = np.degrees(np.arctan2(-d[0], d[2]))
    cas = np.degrees(np.arctan2(-d[1], d[2]))
    add(f"   KPI    = atan(-dX/dZ) = atan({-d[0]:+.3f}/{d[2]:.3f}) = {kpi:+.4f} deg   "
        + ("(NEGATIVE: dX>0, so the axis leans OUTBOARD at the top)" if kpi < 0
           else "(positive: the axis leans inboard at the top)"))
    add(f"   caster = atan(-dY/dZ) = atan({-d[1]:+.3f}/{d[2]:.3f}) = {cas:+.4f} deg   "
        f"({'UBJ behind LBJ - positive caster' if d[1] < 0 else 'UBJ ahead of LBJ'})")

    # STEP B ---------------------------------------------------------------
    t = -LBJ[2] / d[2]
    G = LBJ + t * d
    add("")
    add("STEP B  -  walk DOWN the axis to the ground (Z = 0) to find G")
    add(f"   need LBJ_z + t*dZ = 0  ->  t = -{LBJ[2]:.3f}/{d[2]:.3f} = {t:.5f}")
    add(f"   G_x = {LBJ[0]:.3f} + ({t:.5f})({d[0]:+.3f}) = {G[0]:.3f}")
    add(f"   G_y = {LBJ[1]:.3f} + ({t:.5f})({d[1]:+.3f}) = {G[1]:.3f}")
    add(f"   G   = ({G[0]:.3f}, {G[1]:.3f}, 0)      <- the axis pierces the road here")

    # STEP C ---------------------------------------------------------------
    P = contact_patch(WC, camber_deg)
    sc = P[0] - G[0]
    tr = G[1] - P[1]
    Rw = wheel_radius(WC, camber_deg)
    add("")
    add("STEP C  -  the contact patch, and the two lengths")
    add(f"   the wheel is CAMBERED {camber_deg:+.1f} deg, so it is TILTED: its lowest point is")
    add(f"   displaced sideways, NOT straight below the wheel centre.")
    add(f"      R = WC_z/cos(g) = {WC[2]:.3f}/cos({camber_deg:+.1f}) = {Rw:.3f} mm")
    add(f"      P_x = WC_x - R*sin(g) = {WC[0]:.3f} - {Rw:.3f}*sin({camber_deg:+.1f}) = {P[0]:.3f}")
    add(f"   P = ({P[0]:.3f}, {P[1]:.3f}, 0)"
        f"   [straight-down projection would give {WC[0]:.3f} -- out by {abs(P[0]-WC[0]):.1f} mm]")
    add(f"   SCRUB RADIUS = P_x - G_x = {P[0]:.3f} - {G[0]:.3f} = {sc:+.3f} mm"
        f"   ({'patch OUTBOARD of the axis' if sc > 0 else 'patch inboard of the axis'})")
    add(f"   MECH. TRAIL  = G_y - P_y = {G[1]:.3f} - {P[1]:.3f} = {tr:+.3f} mm"
        f"   ({'axis AHEAD of patch - stabilising' if tr > 0 else 'axis BEHIND patch - de-centring'})")
    ang_part = np.tan(np.radians(cas)) * WC[2]
    add(f"   of that trail, tan(caster)*R = tan({cas:.3f})*{WC[2]:.1f} = {ang_part:.2f} mm "
        f"comes from the ANGLE;")
    add(f"   the remaining {tr - ang_part:.2f} mm is pure ball-joint OFFSET.")

    # STEP D ---------------------------------------------------------------
    add("")
    add("STEP D  -  steering rotates the WHEEL CENTRE about the axis (G stays put)")
    v = WC - G
    add(f"   v = WC - G = ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})  = (scrub, -trail, R)")
    add("   Rodrigues about the line through G with direction u:")
    add("      WC(d) = G + v*cos d + (u x v)*sin d + u*(u.v)*(1 - cos d)")
    uxv = np.cross(u, v)
    udv = float(np.dot(u, v))
    add(f"      u x v = ({uxv[0]:+.4f}, {uxv[1]:+.4f}, {uxv[2]:+.4f})     u.v = {udv:+.4f}")
    dd = np.radians(steer_deg)
    WCr = G + v * np.cos(dd) + uxv * np.sin(dd) + u * udv * (1 - np.cos(dd))
    st = corner_state(LBJ, UBJ, WC, np.array([float(steer_deg)]), camber_deg)
    Pr = st["P"][0]
    add(f"      WC({steer_deg:g} deg) = ({WCr[0]:.3f}, {WCr[1]:.3f}, {WCr[2]:.3f})")
    add(f"   the SPIN AXIS rotates too, so the camber changes to {st['camber'][0]:+.3f} deg and the")
    add(f"   patch's sideways offset rotates with it -- P is recomputed, not projected:")
    add(f"      P({steer_deg:g} deg)  = ({Pr[0]:.3f}, {Pr[1]:.3f}, 0)")

    # STEP E ---------------------------------------------------------------
    add("")
    add("STEP E  -  read off what changed")
    add(f"   scrub  = P_x - G_x = {Pr[0]-G[0]:+8.3f} mm   (was {sc:+.3f})")
    add(f"   trail  = G_y - P_y = {G[1]-Pr[1]:+8.3f} mm   (was {tr:+.3f})"
        + ("   <-- NEGATIVE: self-centring has REVERSED" if (G[1] - Pr[1]) < 0 else ""))
    add(f"   half-track change  = {Pr[0]-P[0]:+8.3f} mm")
    add(f"   wheelbase change   = {Pr[1]-P[1]:+8.3f} mm")
    add(f"   jacking (WC lift)  = {WCr[2]-WC[2]:+8.3f} mm")
    if rear_y is not None:
        add(f"   wheelbase (this corner) = P_y - rear_y = {Pr[1]-rear_y:.3f} mm")

    # STEP F ---------------------------------------------------------------
    R = WC[2]
    add("")
    add("STEP F  -  the closed-form gradients (why scrub/trail are the levers)")
    add("   dP/d(delta) = u x v  ->  per degree, multiply by pi/180")
    add(f"   d(wheelbase)/d(delta)  = scrub*u_z - R*u_x")
    add(f"      = {sc:.3f}*{u[2]:.5f} - {R:.1f}*({u[0]:+.5f}) = {sc*u[2]-R*u[0]:+.3f} /rad"
        f"  = {(sc*u[2]-R*u[0])*np.pi/180:+.4f} mm/deg")
    add(f"   d(half-track)/d(delta) = trail*u_z + R*u_y")
    add(f"      = {tr:.3f}*{u[2]:.5f} + {R:.1f}*({u[1]:+.5f}) = {tr*u[2]+R*u[1]:+.3f} /rad"
        f"  = {(tr*u[2]+R*u[1])*np.pi/180:+.4f} mm/deg")
    add("   => SCRUB drives wheelbase change, TRAIL drives track change --")
    add("      but the ANGLES feed back through the R*u terms, because the wheel")
    add("      centre sits R mm ABOVE the ground and the axis is tilted.")
    add("=" * 72)
    text = "\n".join(L)
    print(text)
    return text


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    import config as cfg

    p = cfg._load_front_points()
    worked_example(p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"],
                   steer_deg=20.0, rear_y=cfg.rear_patch_y(),
                   camber_deg=cfg.STATIC_CAMBER)
    print()
    LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]

    kpi, cas = kpi_caster(LBJ, UBJ)
    G = ground_point(LBJ, UBJ)
    P = contact_patch(WC)
    sc = scrub_radius(LBJ, UBJ, WC)
    tr = mechanical_trail(LBJ, UBJ, WC)

    print("Scrub / trail core - baseline check (CAD hard points)")
    print("=" * 62)
    print(f"  KPI {kpi:+.2f} deg   caster {cas:+.2f} deg")
    print(f"  axis ground point G = ({G[0]:.3f}, {G[1]:.3f}, 0)")
    print(f"  contact patch     P = ({P[0]:.3f}, {P[1]:.3f}, 0)")
    print(f"  SCRUB RADIUS   = {sc:+.1f} mm")
    print(f"  MECH. TRAIL    = {tr:+.1f} mm")
    print()
    print("  trail decomposition:")
    ang = np.tan(np.radians(cas)) * WC[2]
    print(f"     from caster ANGLE  = tan({cas:.2f}) x {WC[2]:.1f} = {ang:.1f} mm")
    print(f"     from ball-joint OFFSET = {tr - ang:.1f} mm")
    print()

    print("  offsets move scrub/trail one-for-one (angles untouched):")
    for dx in (0, 40, 80):
        L2, U2 = offset_joints(LBJ, UBJ, dx=dx)
        k2, c2 = kpi_caster(L2, U2)
        print(f"     dx={dx:+4d} mm -> scrub {scrub_radius(L2,U2,WC):+7.1f} mm   "
              f"(KPI {k2:+.2f}, caster {c2:+.2f})")
    for dy in (0, 20, 40):
        L2, U2 = offset_joints(LBJ, UBJ, dy=dy)
        k2, c2 = kpi_caster(L2, U2)
        print(f"     dy={dy:+4d} mm -> trail {mechanical_trail(L2,U2,WC):+7.1f} mm   "
              f"(KPI {k2:+.2f}, caster {c2:+.2f})")
    print()

    st = corner_state(LBJ, UBJ, WC, np.array([-35.0, 0.0, 35.0]))
    print("  contact-patch motion under steer:")
    print("     steer    scrub      trail    d_track   d_base    jacking")
    for i, d in enumerate([-35.0, 0.0, 35.0]):
        print(f"     {d:+6.1f}  {st['scrub'][i]:+8.1f}  {st['trail'][i]:+8.1f}  "
              f"{st['d_track'][i]:+8.2f}  {st['d_base'][i]:+7.2f}  {st['jack'][i]:+8.2f}")
