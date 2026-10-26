# -*- coding: utf-8 -*-
"""
ANTI-GEOMETRY: anti-dive, anti-lift and anti-squat.
===================================================

WHAT "ANTI" IS
--------------
Brake or accelerate and weight transfers longitudinally. That transfer has to
reach the axle somehow. It can go through the SPRINGS -- which means the nose
dips (dive), the tail drops (squat) and the car pitches -- or it can go through
the LINKS, which is instantaneous and costs no ride height at all.

"Anti" is the fraction that goes through the links.

    0 %    all of it through the springs; maximum pitch
  100 %    all of it through the links; the car brakes and accelerates FLAT
  < 0 %    the links push the WRONG way; MORE pitch than with no anti at all
           ("pro-dive" / "pro-squat")

It is a pure KINEMATIC property. It costs nothing, weighs nothing, and is
decided entirely by where the inner pickups sit in SIDE VIEW.

THE ONE FORMULA THAT MATTERS
----------------------------
Every textbook draws a side-view swing arm and an instant centre. That
construction is a 2D approximation, and on a real 3D linkage it is both
ill-conditioned (the instant centre runs off to infinity for a perfectly
ordinary suspension) and slightly wrong (the side-view PROJECTION of a 3D rigid
motion is not itself a rigid planar motion, so the "instant centre" does not
exist exactly).

There is an exact route. The corner has ONE degree of freedom. Apply a force
(F_y, F_z) at the point P where the tyre touches the road. The spring feels
nothing if that force does no work on the suspension's one coordinate:

        F_y dP_y + F_z dP_z = 0        =>        F_z / F_y = - dP_y / dP_z

So define the only quantity this module needs:

        tau  =  d(P_y) / d(bump)      [mm forward per mm of compression]

the FORE-AFT motion of the reference point per mm of wheel travel. That is the
whole of anti-geometry. No instant centre, no singularity, exact for the real 3D
mechanism, and it collapses to the textbook answer whenever the textbook answer
is valid (with both arms parallel in side view at angle t, tau = tan t exactly
-- asserted in the suite).

    FRONT patch moves FORWARD in bump  ->  anti-dive
    REAR  patch moves REARWARD in bump ->  anti-lift
    REAR  wheel centre moves REARWARD  ->  anti-squat

WHICH POINT? THIS IS WHERE PEOPLE GET IT WRONG
-----------------------------------------------
The reference point depends on WHAT REACTS THE TORQUE, not on what the force is:

  * torque reacted on the UPRIGHT (outboard brake calipers) -> CONTACT PATCH
  * torque reacted on the CHASSIS (inboard diff and half-shafts, inboard
    brakes)                                                  -> WHEEL CENTRE

A GT3 has outboard brakes and an inboard differential, so the REAR AXLE USES
BOTH: the contact patch when braking, the wheel centre when accelerating. One
piece of geometry, two different anti numbers, differing by exactly one wheel
radius. That is Gillespie Eq (7.19) versus Eq (7.29), and it is why the
a GT3 can read strong anti-LIFT and near-zero anti-SQUAT off the SAME rear
suspension.

THE PERCENTAGES  (Gillespie Ch.7, Eqs 7.28 / 7.29 / 7.19)
----------------------------------------------------------
Longitudinal transfer is  dW = W a_x h / L  at both axles. Each axle's links
carry  F * tau,  where F is that axle's share of the longitudinal force. So

    anti-dive  (front, braking)   =  +xi     * tau_patch_f * L/h * 100
    anti-lift  (rear,  braking)   =  -(1-xi) * tau_patch_r * L/h * 100
    anti-squat (rear,  traction)  =  -        tau_wc_r     * L/h * 100

with xi the FRONT BRAKE BIAS. Note what that means: anti-dive is PROPORTIONAL TO
BRAKE BIAS. Move the bias bar and you have changed the car's pitch behaviour.
Nothing else in the suspension does that.

L/h is the whole reason anti works at all: h/L is about 0.17 here, so a force
line sloped by just 1 deg (tau = 0.017) already buys 10 % of anti.

THE STEER TERM -- AND WHY THIS CAR HAS A 108 % PROBLEM
-------------------------------------------------------
tau is the motion of the patch, and the patch is dragged fore-and-aft by TWO
things: the wishbones swinging, and the upright TWISTING on the tie rod (bump
steer) with the contact patch out on the end of the scrub radius. This module
splits them by re-solving the corner with the TOE HELD CONSTANT:

    tau_total  =  tau_wishbone  +  tau_steer

tau_wishbone is therefore not a fiction -- it is the anti this suspension would
have once the bump steer is designed out, which is a geometry somebody could
build. (Freezing the tie rod instead is the obvious shortcut and is wrong: see
_corner_at.) Both terms are real -- virtual work does not care where the motion
comes from, and the tie-rod load genuinely reaches the chassis through the rack.
But the steer term is a pathology, not a design feature: it needs the rack to
hold it, it feeds straight back into the steering wheel as brake-induced torque,
and any steering compliance bleeds it away.

The steer term has a closed form, and the suite asserts it:

        tau_steer  =  (bump steer [rad/mm])  x  scrub_radius / cos(KPI)

because steering the wheel about the kingpin swings the contact patch around on
the end of the scrub radius. Measured 145.57 mm/rad against 144.94/cos(5.27 deg)
= 145.56 -- and the ratio holds to 5 figures across every tie-rod position
tested, including the one where the bump steer changes sign.

On this car's FRONT the wishbones are flat in side view, so tau_wishbone is zero
-- and yet tau_total is -0.297, which is -108 % anti-dive: the nose dives more
than twice as far as it would with no anti geometry at all. Every bit of it is
bump steer (-0.117 deg/mm) through a 145 mm scrub radius, and it vanishes when
the bump steer is nulled. It is the same broken front geometry that gives 129.7 %
Ackermann and a -283.6 mm roll centre, seen from a third direction.

WHAT IT COSTS
-------------
Anti is not free performance, or every car would run 100 %:

  1. It stiffens the suspension against LONGITUDINAL inputs. Braking over a bump
     with high anti-dive means the wheel cannot move back out of the way.
  2. Brake hop, if the effective swing arm gets short.
  3. Caster changes under braking -> steering torque changes mid-corner.
  4. It is bias-dependent, so it moves whenever the driver touches the bias bar.
  5. It couples into the FRONT VIEW. Tilting the inner axes changes camber gain,
     roll-centre height and caster gain as well -- quantified by coupling().

Gillespie: "In practice, 100 % anti-dive is rarely used. The maximum anti-dive
seldom exceeds 50 %."

FRAME
-----
CAD frame throughout:  X lateral (outboard +), Y longitudinal (FORWARD +),
Z vertical (up), mm. Side view is the (Y, Z) plane.
"""

import numpy as np

import benchmarks as bm
import wishbone_solver as ws
from scrub_trail import contact_patch

# ---------------------------------------------------------------------------
#  benchmark
# ---------------------------------------------------------------------------
# Optional, and absent from this repository -- see `benchmarks.py`. Both are
# None unless private reference data is installed, and everything downstream
# treats that as "no benchmark to draw", not as an error.
#
# The useful part of a reference car here is the PATTERN rather than any single
# number, and the pattern is worth stating in the open: a competitive GT3 carries
# substantial anti under BRAKING at both ends and essentially none under
# acceleration. That is a car built to hold its aero platform into the corner
# and to let the rear settle on exit for traction.
# With private data installed these are that car's measured curves. Without
# it they fall back to a DESIGN TARGET taken from the open literature -- not a
# measurement of any vehicle:
#
#   Gillespie, Fundamentals of Vehicle Dynamics, Ch.7:
#       "In practice, 100 % anti-dive is rarely used. The maximum anti-dive
#        seldom exceeds 50 %."                      -> stay well under 50
#   Herb Adams, Chassis Engineering, p.59-60:
#       production cars up to 30 % anti-dive; race cars commonly 5-7 %, many
#       run zero because their CG is so low         -> 25 % is a usable middle
#   Adams, Fig.10-2, p.82:
#       independent rear suspension, inboard diff -> practical anti-squat
#       limit 25 %                                  -> 0 % is the neutral choice
#
# The flat-with-travel slope is a target in its own right: the anti a driver
# feels should not move as the car uses its travel.
LITERATURE_TARGET_PCT = dict(anti_dive_f=25.0, anti_lift_r=25.0,
                             anti_squat_r=0.0)
LITERATURE_TARGET_SLOPE = dict(anti_dive_f=0.0, anti_lift_r=0.0,
                               anti_squat_r=0.0)

BENCHMARK_PCT = bm.get("ANTI_PCT") or LITERATURE_TARGET_PCT
BENCHMARK_SLOPE = bm.get("ANTI_SLOPE") or LITERATURE_TARGET_SLOPE
BENCHMARK_IS_MEASURED = bm.get("ANTI_PCT") is not None

# Gillespie, Fundamentals of Vehicle Dynamics, Ch.7: "In practice, 100 %
# anti-dive is rarely used. The maximum anti-dive seldom exceeds 50 %."
GILLESPIE_MAX_ANTIDIVE_PCT = 50.0

# Design bands, in the same spirit as the roll-gradient bands in arb.py.
BANDS = (
    (-1e9, 0.0, "PRO -- the links add pitch instead of resisting it"),
    (0.0, 15.0, "low -- compliant, race-car normal, platform held by springs"),
    (15.0, 35.0, "moderate -- the usual GT / sports-car window"),
    (35.0, 50.0, "high -- flat stops, watch caster change and brake hop"),
    (50.0, 1e9, "above Gillespie's practical ceiling"),
)


def classify(pct):
    """Which design band a percentage falls in."""
    for lo, hi, name in BANDS:
        if lo <= pct < hi:
            return name
    return BANDS[-1][2]


# ---------------------------------------------------------------------------
#  the design knob: side-view tilt of the inner hinge axes
# ---------------------------------------------------------------------------

_AXES = (("Upper wishbone - inner front", "Upper wishbone - inner rear"),
         ("Lower wishbone - inner front", "Lower wishbone - inner rear"))


def axis_tilt_deg(pts):
    """(upper, lower) side-view tilt of the inner hinge axes [deg].

    POSITIVE = the axis RISES going rearward, i.e. the front pivot is the low
    one. That is the sign that produces anti-DIVE at the front; the rear wants
    the opposite sign for anti-squat and anti-lift.

    Both axes on this car read about 0.000 deg: the CAD has every inner pickup
    at a single Z per wishbone, so the car currently has NO wishbone
    anti-geometry at either end. That is the starting point this module exists
    to move off.
    """
    out = []
    for nf, nr in _AXES:
        A = np.asarray(pts[nf], float)
        B = np.asarray(pts[nr], float)
        out.append(float(np.degrees(np.arctan2(B[2] - A[2], -(B[1] - A[1])))))
    return tuple(out)


def tilt_axes(pts, upper_deg, lower_deg):
    """A copy of `pts` with the two inner hinge axes tilted in SIDE VIEW.

    Each axis is rotated about its own midpoint, so the wishbone's mean height
    and its length are untouched and only the side-view angle changes. That
    keeps the front-view geometry as close to unchanged as a 3D linkage allows,
    which is what makes coupling() an honest measurement rather than an artefact
    of moving everything at once.
    """
    p = {k: np.array(v, float) for k, v in pts.items()}
    for ang, (nf, nr) in zip((upper_deg, lower_deg), _AXES):
        y_mid = (p[nf][1] + p[nr][1]) / 2.0
        z_mid = (p[nf][2] + p[nr][2]) / 2.0
        for n in (nf, nr):
            p[n][2] = z_mid - np.tan(np.radians(float(ang))) * (p[n][1] - y_mid)
    return p


# ---------------------------------------------------------------------------
#  tau -- the only kinematic quantity anti-geometry needs
# ---------------------------------------------------------------------------


def _upright(g, LBJ, UBJ, TRO):
    """Place the wheel from the three solved upright points.

    The contact point is returned as a MATERIAL POINT of the upright -- the
    reference patch position carried by the corner's rigid transform -- NOT by
    re-running contact_patch() at the new position.

    That distinction is the whole of Gillespie's patch-versus-wheel-centre rule
    and it is easy to get wrong. contact_patch() returns the LOWEST POINT of the
    wheel, and a wheel is a body of revolution about its spin axis, so pitching
    the upright rotates the wheel on its bearing and moves the lowest point not
    at all. Use it here and tau_patch collapses onto tau_wc exactly, the
    side-view swing arm reads as infinite for every geometry, and anti-squat and
    anti-lift become impossible to set independently.

    With OUTBOARD brakes the caliper locks the wheel to the upright, so wheel and
    upright are one rigid body and the braking force really does act on a point
    that pitches with it. Hence the material point. With an INBOARD diff the
    wheel spins free on its bearing and the force reaches the upright at the
    WHEEL CENTRE instead -- which is the other return value, and needs no such
    care.
    """
    R, t = ws.rigid_map(g["LBJ"], g["UBJ"], g["TRO"], LBJ, UBJ, TRO)
    return R @ g["WC"] + t, R @ g["patch0"] + t, R @ g["s0"]


def prepare(g):
    """Cache the reference contact point on a solver geometry dict."""
    if "patch0" not in g:
        g["patch0"] = contact_patch(g["WC"], spin_axis=g["s0"],
                                    R=g["R_wheel"])
    return g


def _corner_at(g, theta_L_deg, mode="real"):
    """Solve the corner at one lower-arm angle.

    mode="real"    the actual mechanism -- the tie rod twists the upright, so
                   the wheel steers as it travels (bump steer). This is the
                   number physics cares about.

    mode="notoe"   the SAME wishbones, but the upright twisted so that TOE IS
                   HELD CONSTANT. This is not a modelling fiction: it is the
                   corner you get once the bump steer has been designed out, so
                   it answers "what anti do the WISHBONES give?" with a geometry
                   somebody could actually build.

                   Freezing the tie rod instead -- the obvious alternative --
                   looks simpler but is NOT convention-free: the zero-twist
                   transport still lets the upright roll as the kingpin
                   direction changes, which drags the wheel centre fore-and-aft
                   and leaves a spurious tau of -0.0045 at the rear even with
                   both hinge axes dead flat. Holding toe leaves 1e-6.

    Returns (wheel centre, contact patch, spin axis).
    """
    prepare(g)
    LBJ = ws.rotate_about_line(g["LBJ"], g["LIF"], g["ax_l"], theta_L_deg)[0]
    _, UBJ = ws.solve_on_circle(g["UBJ"], g["UIF"], g["ax_u"], LBJ,
                                g["len_upright"], 0.0)
    kp = UBJ - LBJ
    kp = kp / np.linalg.norm(kp)
    seed = ws._align_seed(g, LBJ, UBJ)
    if mode == "real":
        _, TRO = ws.solve_on_circle(seed, LBJ, kp, g["TRI"],
                                    g["len_tierod"], 0.0)
        return _upright(g, LBJ, UBJ, TRO)
    if mode != "notoe":
        raise ValueError("mode must be 'real' or 'notoe'")

    # bisect the twist about the kingpin until the toe angle comes back to its
    # static value. Monotonic over any sane range, so plain bisection is enough
    # and keeps this module on numpy alone, like the rest of the project.
    toe0 = np.degrees(np.arctan2(g["s0"][1], g["s0"][0]))

    def err(psi):
        TRO = LBJ + ws.rotate_about_line(seed - LBJ, np.zeros(3), kp, psi)[0]
        s = _upright(g, LBJ, UBJ, TRO)[2]
        return np.degrees(np.arctan2(s[1], s[0])) - toe0

    lo, hi = -10.0, 10.0
    flo, fhi = err(lo), err(hi)
    if flo * fhi > 0:
        raise ValueError("cannot hold toe: the twist needed is beyond +/-10 deg")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = err(mid)
        if hi - lo < 1e-11:
            break
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    TRO = LBJ + ws.rotate_about_line(seed - LBJ, np.zeros(3), kp,
                                     0.5 * (lo + hi))[0]
    return _upright(g, LBJ, UBJ, TRO)


def tau(g, theta_L_deg=0.0, dtheta=0.35):
    """The fore-aft motion rates at one operating point. Central difference.

    Returns a dict:
        patch, wc                    tau for each reference point, REAL mechanism
        patch_wishbone, wc_wishbone  the same with the tie-rod twist frozen
        patch_steer, wc_steer        the difference -- the bump-steer share
        bump_steer                   [deg/mm], for context
        patch_y, patch_z, wc_y, wc_z the reference points themselves
    """
    Wp, Pp, sp = _corner_at(g, theta_L_deg + dtheta, "real")
    Wm, Pm, sm = _corner_at(g, theta_L_deg - dtheta, "real")
    Wp0, Pp0, _ = _corner_at(g, theta_L_deg + dtheta, "notoe")
    Wm0, Pm0, _ = _corner_at(g, theta_L_deg - dtheta, "notoe")
    W0, P0, _ = _corner_at(g, theta_L_deg, "real")
    dz = Wp[2] - Wm[2]
    dz0 = Wp0[2] - Wm0[2]
    toe_p = np.degrees(np.arctan2(sp[1], sp[0]))
    toe_m = np.degrees(np.arctan2(sm[1], sm[0]))
    out = dict(
        patch=(Pp[1] - Pm[1]) / dz,
        wc=(Wp[1] - Wm[1]) / dz,
        patch_wishbone=(Pp0[1] - Pm0[1]) / dz0,
        wc_wishbone=(Wp0[1] - Wm0[1]) / dz0,
        bump_steer=(toe_p - toe_m) / dz,
        patch_y=float(P0[1]), patch_z=float(P0[2]),
        wc_y=float(W0[1]), wc_z=float(W0[2]),
    )
    out["patch_steer"] = out["patch"] - out["patch_wishbone"]
    out["wc_steer"] = out["wc"] - out["wc_wishbone"]
    # the lever the bump steer acts through: scrub radius / cos(KPI) at the
    # patch, kingpin offset at the wheel centre. Constant for a given geometry,
    # which is exactly why the steer term is proportional to the bump steer.
    bs = np.radians(out["bump_steer"])
    out["steer_lever_patch"] = out["patch_steer"] / bs if abs(bs) > 1e-12 else 0.0
    out["steer_lever_wc"] = out["wc_steer"] / bs if abs(bs) > 1e-12 else 0.0
    return out


def theta_grid(g, span_deg=14.0, n=401):
    """(bump, theta) lookup so tau can be evaluated at a wheel travel."""
    th = np.linspace(-span_deg, span_deg, n)
    trav = []
    for t in th:
        W, _, _ = _corner_at(g, t, "real")
        trav.append(W[2] - g["WC"][2])
    return np.asarray(trav, float), th


# ---------------------------------------------------------------------------
#  the percentages
# ---------------------------------------------------------------------------


def anti_dive_pct(tau_patch_f, car, bias_f=0.62):
    """FRONT anti-dive [%] under braking, outboard calipers.  Gillespie 7.28.

    Vectorised in both tau and bias, so a whole bump sweep or a whole
    brake-bias sweep goes through in one call.
    """
    return (np.asarray(bias_f, float) * np.asarray(tau_patch_f, float)
            * car["wheelbase_mm"] / car["h_cg_mm"] * 100.0)


def anti_lift_pct(tau_patch_r, car, bias_f=0.62):
    """REAR anti-lift [%] under braking, outboard calipers.  Gillespie 7.29.

    Sign: the rear axle is UNLOADING, so the links have to pull it DOWN. That
    needs the rear patch to move REARWARD in bump, i.e. tau < 0 -- hence the
    leading minus. Get this sign wrong and a car that resists lift reads as one
    that adds to it.
    """
    return (-(1.0 - np.asarray(bias_f, float))
            * np.asarray(tau_patch_r, float)
            * car["wheelbase_mm"] / car["h_cg_mm"] * 100.0)


def anti_squat_pct(tau_r, car, drive_share=1.0):
    """REAR anti-squat [%] under acceleration.  Gillespie 7.19.

    Pass tau measured at the WHEEL CENTRE for an inboard differential (any
    modern GT3: the diff is bolted to the chassis and the half-shafts carry only
    force, so the drive torque never reaches the upright). Pass tau at the
    CONTACT PATCH only for a solid axle or a hub motor, where the drive torque
    does react on the upright.
    """
    return (-np.asarray(drive_share, float) * np.asarray(tau_r, float)
            * car["wheelbase_mm"] / car["h_cg_mm"] * 100.0)


def tau_for_anti(pct, car, kind="dive", bias_f=0.62, drive_share=1.0):
    """Inverse of the three above: the tau a target percentage demands."""
    k = car["h_cg_mm"] / car["wheelbase_mm"] / 100.0
    pct = np.asarray(pct, float)
    if kind == "dive":
        return pct * k / np.asarray(bias_f, float)
    if kind == "lift":
        return -pct * k / (1.0 - np.asarray(bias_f, float))
    if kind == "squat":
        return -pct * k / np.asarray(drive_share, float)
    raise ValueError("kind must be dive/lift/squat")


# ---------------------------------------------------------------------------
#  the side-view instant centre  --  for DRAWING only
# ---------------------------------------------------------------------------


def side_view_ic(t, which="wishbone"):
    """Where the two force lines cross, from a tau() dict. For the side-view
    picture only -- none of the percentages above need it.

    The patch line and the wheel-centre line both pass through the side-view
    instant centre, so intersecting them locates it:

        Z = tau_patch (Y_p - Y)          through the patch
        Z = Z_wc + tau_wc (Y_wc - Y)     through the wheel centre

    which="wishbone" (default) uses the toe-held taus, i.e. the instant centre
    the LINKAGE defines -- that is what "side-view swing arm" means and what
    belongs on a drawing. which="real" includes the bump-steer term, which is a
    genuine force path but is not a swing arm and can put the "instant centre"
    somewhere physically meaningless.

    Returns (Y, Z, finite). When the two taus are equal the lines are parallel,
    which is the perfectly ordinary "instant centre at infinity" case -- equal
    tau at both heights means the whole corner TRANSLATES in side view. finite
    is False there and the caller should draw a direction, not a point.
    """
    sfx = "_wishbone" if which == "wishbone" else ""
    a, b = float(t["patch" + sfx]), float(t["wc" + sfx])
    if abs(a - b) < 1e-9:
        return float("inf"), float("inf"), False
    y = (a * t["patch_y"] - b * t["wc_y"] - t["wc_z"]) / (a - b)
    z = a * (t["patch_y"] - y)
    return float(y), float(z), True


def svsa_mm(t, which="wishbone"):
    """Side-view swing-arm length: horizontal patch-to-instant-centre distance.

    Short arms are what cause brake hop, and they make the anti change fast with
    wheel travel. `inf` means parallel side-view arms -- the longest possible
    swing arm and the flattest possible anti curve.
    """
    y, _, finite = side_view_ic(t, which)
    return abs(t["patch_y"] - y) if finite else float("inf")


def patch_wc_gap(t, R_wheel, which="wishbone"):
    """The identity that lets the rear run TWO different antis at once.

        tau_patch - tau_wc  =  R_wheel / SVSA      (up to sign)

    The patch and the wheel centre sit one wheel radius apart on the same force
    line, so the two taus can only differ if that line is TILTED -- which needs a
    finite swing arm. Parallel side-view arms give SVSA -> infinity, hence
    identical taus, hence anti_squat = anti_lift / (1 - bias), locked together.

    That is why strong anti-LIFT alongside near-zero anti-SQUAT is proof that a
    car's side-view arms are NOT parallel: parallel arms at a useful anti-lift
    would force an absurd anti-squat. Solving for a target pair lands on a swing
    arm of a few metres, and the same length falls out of the two percentages
    read independently through Gillespie 7.19 and 7.29 -- two routes, one
    number.

    Returns (measured gap, R/SVSA).
    """
    sfx = "_wishbone" if which == "wishbone" else ""
    s = svsa_mm(t, which)
    return (float(t["patch" + sfx] - t["wc" + sfx]),
            (float(R_wheel) / s if np.isfinite(s) and s else 0.0))


# ---------------------------------------------------------------------------
#  sweeps
# ---------------------------------------------------------------------------


def sweep(g, bump_grid_mm, dtheta=0.35):
    """tau (and its split) versus wheel travel -- the benchmark plot format.

    The published curves are anti-% against wheel travel, so this is what has to
    be produced to compare like with like.
    """
    trav, th = theta_grid(g)
    b = np.atleast_1d(np.asarray(bump_grid_mm, float))
    keys = ("patch", "wc", "patch_wishbone", "wc_wishbone",
            "patch_steer", "wc_steer", "bump_steer")
    out = {k: [] for k in keys}
    out["bump"] = b
    for bb in b:
        t = tau(g, float(np.interp(bb, trav, th)), dtheta)
        for k in keys:
            out[k].append(t[k])
    for k in keys:
        out[k] = np.asarray(out[k], float)
    return out


def front_view_at_ride(g, dtheta=0.35):
    """Camber gain, caster gain and roll-centre height at ride height.

    Deliberately NOT via solve_sweep + linkage_roll_centre: those each run a
    561-point continuation sweep to map bump onto the arm angle, which is 0.5 s
    a call and makes coupling() unusable in a live panel. At ride height the arm
    angle IS zero by definition, so three solve_at calls give the same numbers
    -- asserted against the sweep-based routines in the suite.
    """
    sp = ws.solve_at(g, +dtheta)
    sm = ws.solve_at(g, -dtheta)
    s0 = ws.solve_at(g, 0.0)
    dz = sp["WC"][2] - sm["WC"][2]

    def cam(r):
        return np.degrees(np.arcsin(np.clip(-r["s"][2], -1, 1)))

    from scrub_trail import kpi_caster
    camber_gain = (cam(sp) - cam(sm)) / dz
    caster_gain = (kpi_caster(sp["LBJ"], sp["UBJ"])[1]
                   - kpi_caster(sm["LBJ"], sm["UBJ"])[1]) / dz

    # classical front-view roll centre: intersect the perpendiculars to the two
    # ball joints' front-view velocities, then project patch -> IC to centreline
    vL = (sp["LBJ"] - sm["LBJ"])[[0, 2]]
    vU = (sp["UBJ"] - sm["UBJ"])[[0, 2]]
    pL = s0["LBJ"][[0, 2]]
    pU = s0["UBJ"][[0, 2]]
    nL = np.array([-vL[1], vL[0]])
    nU = np.array([-vU[1], vU[0]])
    A = np.array([[nL[0], -nU[0]], [nL[1], -nU[1]]])
    try:
        tpar = np.linalg.solve(A, pU - pL)[0]
    except np.linalg.LinAlgError:
        return camber_gain, caster_gain, float("nan")
    IC = pL + tpar * nL
    px = contact_patch(s0["WC"], spin_axis=s0["s"], R=g["R_wheel"])[0]
    d = IC - np.array([px, 0.0])
    rc = d[1] * (0.0 - px) / d[0] if abs(d[0]) > 1e-12 else float("nan")
    return float(camber_gain), float(caster_gain), float(rc)


def coupling(pts, static_camber, tilts_deg, which="both", bump=6.0, n=13):
    """What tilting the inner axes does to EVERYTHING, not just the anti.

    This is the panel that answers "does changing anti move the instant
    centres?" Yes -- the side view and the front view are two projections of ONE
    3D linkage, so tilting the hinge axes changes the front-view geometry too.

    For each tilt angle, re-solves the corner and returns the anti-relevant tau
    ALONGSIDE camber gain, roll-centre height and caster gain, so the trade is
    visible in one place instead of being discovered later.

    `which` is "both", "upper" or "lower" -- which axis to tilt.
    """
    tl = np.atleast_1d(np.asarray(tilts_deg, float))
    now = axis_tilt_deg(pts)
    rows = dict(tilt=tl, tau_patch=[], tau_wc=[], tau_wishbone=[],
                tau_wc_wishbone=[], camber_gain=[], rc_h=[], caster_gain=[],
                svsa=[])
    for a in tl:
        up = a if which in ("both", "upper") else now[0]
        lo = a if which in ("both", "lower") else now[1]
        g = ws.corner_geometry(tilt_axes(pts, up, lo), static_camber)
        t = tau(g)
        cg, cast, rc = front_view_at_ride(g)
        rows["tau_patch"].append(t["patch"])
        rows["tau_wc"].append(t["wc"])
        rows["tau_wishbone"].append(t["patch_wishbone"])
        rows["tau_wc_wishbone"].append(t["wc_wishbone"])
        rows["camber_gain"].append(cg)
        rows["caster_gain"].append(cast)
        rows["rc_h"].append(rc)
        rows["svsa"].append(svsa_mm(t))
    for k in rows:
        if k != "tilt":
            rows[k] = np.asarray(rows[k], float)
    return rows


def required_tilt(pts, static_camber, target_pct, car, kind="dive",
                  bias_f=0.62, drive_share=1.0, lo=-12.0, hi=12.0, tol=1e-6):
    """The equal upper/lower axis tilt that hits a target anti percentage.

    Bisection on the WISHBONE tau, deliberately -- the steer term is a fault to
    be fixed, not a design variable to size a suspension against. Returns
    (tilt_deg, achieved_pct), or (nan, nan) if the target is out of range.
    """
    want = tau_for_anti(target_pct, car, kind, bias_f, drive_share)

    def f(a):
        g = ws.corner_geometry(tilt_axes(pts, a, a), static_camber)
        t = tau(g)
        return (t["wc_wishbone"] if kind == "squat"
                else t["patch_wishbone"]) - want

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan"), float("nan")
    mid = 0.5 * (lo + hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            break
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    g = ws.corner_geometry(tilt_axes(pts, mid, mid), static_camber)
    t = tau(g)
    tv = t["wc_wishbone"] if kind == "squat" else t["patch_wishbone"]
    if kind == "squat":
        got = anti_squat_pct(tv, car, drive_share)
    elif kind == "lift":
        got = anti_lift_pct(tv, car, bias_f)
    else:
        got = anti_dive_pct(tv, car, bias_f)
    return float(mid), float(got)


def required_tilt_pair(pts, static_camber, target_lift, target_squat, car,
                       bias_f=0.62, drive_share=1.0, span=14.0, iters=40):
    """The (upper, lower) tilt pair hitting a rear anti-LIFT and anti-SQUAT
    target at the same time.

    Needed because the two targets cannot both be met with equal tilts: parallel
    side-view arms force tau_patch == tau_wc, which locks anti-squat to
    anti_lift/(1 - bias). Splitting them needs a finite swing arm, i.e. the upper
    and lower axes at DIFFERENT angles -- see patch_wc_gap().

    Damped Newton on the 2x2 Jacobian, finite-differenced. Returns
    (upper_deg, lower_deg, achieved_lift, achieved_squat); the angles are nan if
    it does not converge inside +/-span.
    """
    want = np.array([tau_for_anti(target_lift, car, "lift", bias_f),
                     tau_for_anti(target_squat, car, "squat",
                                  drive_share=drive_share)], float)

    def f(v):
        g = ws.corner_geometry(tilt_axes(pts, v[0], v[1]), static_camber)
        t = tau(g)
        return np.array([t["patch_wishbone"], t["wc_wishbone"]], float) - want

    v = np.array([0.0, 0.0])
    r = f(v)
    for _ in range(iters):
        if np.max(np.abs(r)) < 1e-8:
            break
        J = np.empty((2, 2))
        for j in range(2):
            dv = np.zeros(2)
            dv[j] = 1e-3
            J[:, j] = (f(v + dv) - r) / 1e-3
        if abs(np.linalg.det(J)) < 1e-14:
            return float("nan"), float("nan"), float("nan"), float("nan")
        step = np.linalg.solve(J, -r)
        n = np.linalg.norm(step)
        if n > 2.0:                       # damp, the map is mildly nonlinear
            step *= 2.0 / n
        v = np.clip(v + step, -span, span)
        r = f(v)
    if np.max(np.abs(r)) > 1e-5 or np.any(np.abs(v) >= span - 1e-9):
        return float("nan"), float("nan"), float("nan"), float("nan")
    g = ws.corner_geometry(tilt_axes(pts, v[0], v[1]), static_camber)
    t = tau(g)
    return (float(v[0]), float(v[1]),
            anti_lift_pct(t["patch_wishbone"], car, bias_f),
            anti_squat_pct(t["wc_wishbone"], car, drive_share))


# ---------------------------------------------------------------------------
#  what it is FOR: pitch
# ---------------------------------------------------------------------------


def ride_rate(kw, k_tyre):
    """Spring and tyre in series -- what the CHASSIS moves on, per wheel."""
    kw = np.asarray(kw, float)
    return kw * k_tyre / (kw + k_tyre)


def pitch(car, anti_f_pct, anti_r_pct, a_long_g, k_tyre=310.4, mode="brake"):
    """Body attitude under a longitudinal acceleration.

    The point of anti-geometry, stated in millimetres. Each axle's springs carry
    only the share the links did NOT take, so:

        deflection = dW (1 - anti/100) / (axle ride rate)
        pitch      = (front travel + rear travel) / wheelbase

    mode="brake"  front dives, rear lifts, both antis apply
    mode="accel"  rear squats, front lifts; a RWD car has no front longitudinal
                  force at all, so the front CANNOT have anti under power --
                  anti_f_pct is ignored and the front lift is unresisted.

    Returns mm at each axle, the pitch angle [deg], and the same numbers with
    the anti removed so the benefit is explicit.
    """
    W = car["mass_total_kg"] * 9.80665
    L, h = car["wheelbase_mm"], car["h_cg_mm"]
    dW = W * abs(float(a_long_g)) * h / L          # N, transferred axle to axle
    rr_f = 2.0 * float(ride_rate(car["kw_f"], k_tyre))   # N/mm per AXLE
    rr_r = 2.0 * float(ride_rate(car["kw_r"], k_tyre))
    if mode == "brake":
        af, ar = float(anti_f_pct), float(anti_r_pct)
    else:
        af, ar = 0.0, float(anti_r_pct)
    d_f = dW * (1.0 - af / 100.0) / rr_f
    d_r = dW * (1.0 - ar / 100.0) / rr_r
    d_f0, d_r0 = dW / rr_f, dW / rr_r
    ang = np.degrees(np.arctan2(d_f + d_r, L))
    ang0 = np.degrees(np.arctan2(d_f0 + d_r0, L))
    g = max(abs(float(a_long_g)), 1e-9)
    return dict(dW=dW, front_mm=d_f, rear_mm=d_r, pitch_deg=float(ang),
                front_mm_noanti=d_f0, rear_mm_noanti=d_r0,
                pitch_deg_noanti=float(ang0),
                pitch_grad=float(ang) / g, pitch_grad_noanti=float(ang0) / g,
                rr_f=rr_f, rr_r=rr_r, mode=mode)


if __name__ == "__main__":
    import config as cfg
    import rear_geometry as rgeo
    import load_transfer as ltr

    car = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
    pf = cfg._load_front_points()
    pr = rgeo.solver_points(rgeo.load_rear())
    gf = ws.corner_geometry(pf, cfg.STATIC_CAMBER)
    gr = ws.corner_geometry(pr, rgeo.REAR_STATIC_CAMBER)
    tf, tr = tau(gf), tau(gr)

    print("SIDE-VIEW HINGE-AXIS TILT (the only thing that makes wishbone anti)")
    for nm, p in (("FRONT", pf), ("REAR", pr)):
        u, l = axis_tilt_deg(p)
        print(f"   {nm:6s} upper {u:+8.4f} deg   lower {l:+8.4f} deg")

    print("\nTAU  [mm fore-aft per mm of bump]")
    for nm, t in (("FRONT", tf), ("REAR", tr)):
        print(f"   {nm:6s} patch {t['patch']:+9.5f} = wishbone "
              f"{t['patch_wishbone']:+9.5f} + steer {t['patch_steer']:+9.5f}"
              f"   (bump steer {t['bump_steer']:+.5f} deg/mm)")
        print(f"          wc    {t['wc']:+9.5f} = wishbone "
              f"{t['wc_wishbone']:+9.5f} + steer {t['wc_steer']:+9.5f}")

    print("\nANTI  [%]                     ours     wishbone only    benchmark")
    rows = (("front anti-dive", anti_dive_pct(tf["patch"], car),
             anti_dive_pct(tf["patch_wishbone"], car),
             BENCHMARK_PCT["anti_dive_f"]),
            ("rear  anti-lift", anti_lift_pct(tr["patch"], car),
             anti_lift_pct(tr["patch_wishbone"], car),
             BENCHMARK_PCT["anti_lift_r"]),
            ("rear  anti-squat", anti_squat_pct(tr["wc"], car),
             anti_squat_pct(tr["wc_wishbone"], car),
             BENCHMARK_PCT["anti_squat_r"]))
    for nm, a, b, c in rows:
        print(f"   {nm:18s} {a:9.2f}   {b:11.2f}   {c:10.1f}")

    print("\nWHAT IT BUYS IN PITCH  (braking at 1.2 g)")
    p0 = pitch(car, 0.0, 0.0, 1.2)
    p1 = pitch(car, BENCHMARK_PCT["anti_dive_f"], BENCHMARK_PCT["anti_lift_r"],
               1.2)
    print(f"   no anti          front {p0['front_mm']:6.2f} mm   rear "
          f"{p0['rear_mm']:6.2f} mm   pitch {p0['pitch_deg']:.3f} deg")
    print(f"   benchmark anti   front {p1['front_mm']:6.2f} mm   rear "
          f"{p1['rear_mm']:6.2f} mm   pitch {p1['pitch_deg']:.3f} deg"
          f"   ({p1['pitch_deg'] / p0['pitch_deg'] - 1:+.1%})")

    print("\nTILT NEEDED FOR THE BENCHMARK (wishbones only, equal upper/lower)")
    for nm, p, cam, kind, tgt in (
            ("FRONT anti-dive", pf, cfg.STATIC_CAMBER, "dive",
             BENCHMARK_PCT["anti_dive_f"]),
            ("REAR  anti-lift", pr, rgeo.REAR_STATIC_CAMBER, "lift",
             BENCHMARK_PCT["anti_lift_r"]),
            ("REAR  anti-squat", pr, rgeo.REAR_STATIC_CAMBER, "squat",
             BENCHMARK_PCT["anti_squat_r"])):
        a, got = required_tilt(p, cam, tgt, car, kind)
        print(f"   {nm:18s} tilt {a:+7.3f} deg  ->  {got:7.2f} %  "
              f"(target {tgt:.1f})")

    print("\n...but the REAR's two targets cannot both be met with EQUAL tilts:")
    print("   parallel side-view arms force  anti_squat = anti_lift/(1-bias) = "
          f"{BENCHMARK_PCT['anti_lift_r'] / 0.38:.0f} %")
    u, l, gl, gs = required_tilt_pair(pr, rgeo.REAR_STATIC_CAMBER,
                                      BENCHMARK_PCT["anti_lift_r"],
                                      BENCHMARK_PCT["anti_squat_r"], car)
    print(f"   split axes: upper {u:+.3f} deg, lower {l:+.3f} deg  ->  "
          f"anti-lift {gl:.2f} %, anti-squat {gs:.2f} %")
    gg = ws.corner_geometry(tilt_axes(pr, u, l), rgeo.REAR_STATIC_CAMBER)
    tt = tau(gg)
    gap, ident = patch_wc_gap(tt, gg["R_wheel"])
    print(f"   side-view swing arm {svsa_mm(tt):.0f} mm   "
          f"tau gap {gap:+.5f} vs R/SVSA {ident:+.5f}")
