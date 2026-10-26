# -*- coding: utf-8 -*-
"""
Anti-roll bars: sizing a hollow torsion bar from a roll-stiffness target.
=========================================================================

WHAT AN ARB IS
--------------
A spring that only reacts to ROLL. It ties left to right, so in heave both
wheels rise together, the bar rotates bodily, nothing twists, and it makes no
force. In roll the wheels move oppositely, the bar twists along its length, and
it resists. That is what lets you set roll stiffness INDEPENDENTLY of ride
stiffness -- springs set the ride frequency (and therefore the ride height),
the bar sets the roll.

THE CHAIN
---------
    K_t     = G J / L              [Nm/rad]  torque per radian of bar twist
    k_arm   = K_t / a^2            [N/m]     force per metre at the drop link
    k_wheel = k_arm * MR^2         [N/m]     ...referred to the contact patch
    K_roll  = k_wheel * t^2 / 2    [Nm/rad]  roll moment per radian of body roll

`a` is DIVIDED because the bar makes torque and the drop link needs force:
F = T/a while delta = a*phi, so a appears twice. MR is squared for the same
reason. `t^2/2` comes from rolling by phi: each wheel moves (t/2)phi, makes
K(t/2)phi, and acts at (t/2) -- two wheels give K t^2 / 2.

HOLLOW vs SOLID
---------------
J goes as r^4 but mass as r^2, so the core is nearly dead metal: on a 50 mm
solid bar the inner 30 mm carries 13 % of the stiffness for 36 % of the weight.
A tube of the SAME mass as a 25 mm solid bar is 7x stiffer at 50 mm OD.

The limit on going thin is NOT stress. For a given twist,

    tau = T r / J   and   T = G J phi / L    =>    tau = G r phi / L

-- J cancels. Surface shear depends only on OUTER RADIUS and LENGTH; wall
thickness does not appear. Thinning the wall makes the bar softer, not more
stressed. What actually limits you is bigger OD raising stress (tau ~ r), local
buckling at clamps and arm welds, and manufacturability (~2-3 mm minimum wall on
a 40-50 mm bar). That is why real GT3 bars are things like 50 x 3.

THE DESIGN FLOW THIS MODULE IMPLEMENTS
--------------------------------------
    1. targets (roll gradient + LLTD)  ->  required ARB wheel rate per axle
    2. required rate + geometry        ->  required bar torsional stiffness K_t
    3. stress limit at max roll        ->  MAXIMUM usable outer diameter
    4. required K_t at that OD         ->  wall thickness needed
    5. blade in series                 ->  the adjustment range you actually get

LIMITATIONS (documented, not hidden)
------------------------------------
* The CAD has NO anti-roll bar and no drop-link pickups. The pickup is a design
  input; its motion ratio is then SOLVED from the suspension geometry exactly
  like the damper's. Default pickup = the lower damper mount.
* Off-the-shelf BLADE sections are not published by suppliers -- they advertise
  bar diameter and "blade adjustable" but treat the blade geometry as
  proprietary. Bar diameters here are real catalogue sizes; blade dimensions are
  ASSUMED and flagged as such. Swap in real numbers when you have them.
* The bar is modelled as a uniform torsion tube. Real bars have bends, splines
  and transitions that add compliance, so a real bar is a few per cent softer
  than this predicts.
* Bushing and drop-link compliance are not modelled (they only ever soften).
"""

import numpy as np

import load_transfer as ltr

G_STEEL = 80.0e9          # shear modulus, steel [Pa]
E_STEEL = 210.0e9         # Young's modulus, steel [Pa]
RHO_STEEL = 7850.0        # density [kg/m^3]

# Real catalogue bar diameters seen on motorsport/road-race ARBs [mm]. Sources
# in the study doc; these are the sizes suppliers actually list.
CATALOGUE_OD_MM = [22.0, 24.0, 25.0, 28.0, 29.0, 30.0, 32.0, 35.0, 36.0,
                   38.0, 40.0, 45.0, 50.0, 55.0, 60.0]

# ASSUMED blade sections -- suppliers do not publish blade geometry. They
# advertise bar DIAMETER and "blade adjustable" and treat the blade itself as
# proprietary; searched and not found. (root width, tip width, thickness) [mm].
# Equal root and tip = a prismatic rectangle; unequal = linearly tapered.
# The stiffness RANGE a prismatic blade gives is (b/h)^2, so aspect ratio is
# what matters -- these span 3.1:1 to 6.25:1 in aspect, i.e. ~10:1 to ~39:1.
ASSUMED_BLADES_MM = [(25.0, 25.0, 8.0), (30.0, 30.0, 8.0), (40.0, 40.0, 8.0),
                     (40.0, 25.0, 8.0), (50.0, 50.0, 10.0), (50.0, 30.0, 8.0)]
BLADE_DEFAULT = (40.0, 25.0, 8.0, 230.0)     # root, tip, thickness, free length
# Standard tube walls a fabricator can actually buy [mm].
STANDARD_WALL_MM = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

# Allowable surface shear [MPa]. 450 is the UN-peened spring-steel torsional
# endurance limit from first principles (0.577 x 0.5 x UTS ~ 1600 MPa). Real
# ARBs are shot peened as standard, which adds ~25 % -- and published practice
# uses SAE Class 550/700 steels with design simulations reaching ~570 MPa. So
# 550 is the REALISTIC design point and 450 is a margin, not the other way round.
ALLOWABLE_TAU_MPA = [300.0, 450.0, 550.0]   # conservative / un-peened / peened

# Published roll-gradient bands -- OptimumG "Bar talk" (Claude Rouelle, Racecar
# Engineering, May 2021), Figure 1. Degrees of SUSPENDED-MASS roll about its own
# roll axis per g, EXCLUDING tyre deflection and compliance -- which is exactly
# what our load-transfer model computes, so the comparison is like-for-like.
ROLL_GRADIENT_BANDS = [
    ("SUV", 4.00, 8.00),
    ("most passenger cars", 3.00, 6.00),
    ("sports cars", 2.00, 4.00),
    ("GT racecars", 1.00, 3.00),
    ("low downforce racecars", 0.50, 1.50),
    ("high downforce racecars", 0.25, 0.75),
]


def classify_roll_gradient(deg_per_g):
    """Which of OptimumG's Figure-1 bands a roll gradient falls in.

    The bands OVERLAP by design, so a value can sit in more than one; they are
    reported together rather than picking arbitrarily.
    """
    g = float(deg_per_g)
    hits = [n for n, lo, hi in ROLL_GRADIENT_BANDS if lo <= g <= hi]
    if not hits:
        return ("stiffer than any published band" if g < 0.25
                else "softer than any published band")
    return " / ".join(hits)


# ---------------------------------------------------------------------------
#  section properties
# ---------------------------------------------------------------------------


def polar_J(od_mm, wall_mm=None):
    """Polar second moment of area [m^4]. `wall_mm=None` means a SOLID bar.

    Solid:  pi d^4 / 32        Tube:  pi (do^4 - di^4) / 32
    """
    do = np.asarray(od_mm, float) / 1000.0
    if wall_mm is None:
        di = 0.0
    else:
        di = np.maximum(do - 2.0 * np.asarray(wall_mm, float) / 1000.0, 0.0)
    return np.pi * (do ** 4 - di ** 4) / 32.0


def bar_mass_per_m(od_mm, wall_mm=None, rho=RHO_STEEL):
    """Mass per metre [kg/m] -- the whole point of going hollow."""
    do = np.asarray(od_mm, float) / 1000.0
    di = 0.0 if wall_mm is None else np.maximum(
        do - 2.0 * np.asarray(wall_mm, float) / 1000.0, 0.0)
    return np.pi * (do ** 2 - di ** 2) / 4.0 * rho


# ---------------------------------------------------------------------------
#  the chain
# ---------------------------------------------------------------------------


def torsional_stiffness(J_m4, L_mm, G=G_STEEL):
    """K_t = G J / L  [Nm/rad]."""
    return G * np.asarray(J_m4, float) / (np.asarray(L_mm, float) / 1000.0)


def rate_at_arm(Kt, a_mm):
    """Effective rate at the drop link IN ROLL [N/mm], from the bar's K_t.

        k_arm = 2 K_t / a^2

    THE FACTOR 2 IS NOT OPTIONAL, and leaving it out was a real bug here.
    K_t/a^2 is the rate you get with the OTHER end of the bar held still. In
    roll it is not held still: one arm rotates +d/a and the other -d/a, so the
    torsion section sees the RELATIVE twist 2d/a and reacts twice the torque for
    the same per-wheel displacement.

    This is exactly why OptimumG's "Bar talk" (Racecar Engineering, May 2021)
    writes the spring equation with a /2 and the anti-roll bar equation WITHOUT
    one -- Eq.2 vs Eq.5. Same physical asymmetry, stated as a missing divisor
    rather than an extra factor.

    Cross-checked three ways in validate_arb.py: the force chain, strain energy,
    and OptimumG Eq.5 all give MR^2 K_t t^2 / a^2 for the roll stiffness, which
    is what this returns once to_wheel() and roll_stiffness() are applied.
    `bar_twist_rad` already carried the same factor 2 for the stress path, so
    the two halves of the module now agree.
    """
    a = np.asarray(a_mm, float) / 1000.0
    return 2.0 * np.asarray(Kt, float) / a ** 2 / 1000.0


def to_wheel(k_arm_N_mm, mr):
    """Refer the drop-link rate to the contact patch:  k_w = k_arm * MR^2."""
    return np.asarray(k_arm_N_mm, float) * np.asarray(mr, float) ** 2


def roll_stiffness(k_wheel_N_mm, track_mm):
    """Roll moment per radian of BODY roll [N.mm/rad] -- same units as
    load_transfer.roll_stiffness so the two can be added directly."""
    return np.asarray(k_wheel_N_mm, float) * np.asarray(track_mm, float) ** 2 / 2.0


def required_Kt(k_wheel_target_N_mm, a_mm, mr):
    """Invert the chain: the bar stiffness needed for a wheel rate [Nm/rad].

    Carries the same factor 2 as rate_at_arm, halving the bar needed for a given
    wheel rate. A round trip through rate_at_arm -> to_wheel must return the
    input, and the suite asserts exactly that.
    """
    a = np.asarray(a_mm, float) / 1000.0
    return (np.asarray(k_wheel_target_N_mm, float) * 1000.0
            / np.asarray(mr, float) ** 2 * a ** 2 / 2.0)


# ---------------------------------------------------------------------------
#  stress
# ---------------------------------------------------------------------------


def bar_twist_rad(roll_deg, track_mm, a_mm, mr):
    """RELATIVE twist across the bar for a given body roll [rad].

    Body roll phi moves each wheel (t/2)phi, the drop link MR times that, and
    the arm rotates that over `a`. Both ends rotate OPPOSITELY, so the relative
    twist the torsion section sees is twice one end's rotation:

        phi_bar = 2 * MR * (t/2) * phi_body / a = MR * t * phi_body / a
    """
    phi = np.radians(np.asarray(roll_deg, float))
    return (np.asarray(mr, float) * np.asarray(track_mm, float)
            * phi / np.asarray(a_mm, float))


def shear_stress_MPa(od_mm, twist_rad, L_mm, G=G_STEEL):
    """Peak surface shear stress [MPa].

        tau = T r / J,  T = G J phi / L   =>   tau = G r phi / L

    J CANCELS: for a given twist the stress depends only on outer radius and
    torsion length. Wall thickness does not appear -- which is why you cannot
    fix an overstressed bar by making it thinner.
    """
    r = np.asarray(od_mm, float) / 2000.0
    return G * r * np.asarray(twist_rad, float) / (np.asarray(L_mm, float)
                                                   / 1000.0) / 1e6


def max_od_for_stress(tau_allow_MPa, twist_rad, L_mm, G=G_STEEL):
    """Largest outer diameter that stays under an allowable shear [mm].

    Straight inversion of tau = G r phi / L. Independent of wall thickness.
    """
    tau = np.asarray(tau_allow_MPa, float) * 1e6
    L = np.asarray(L_mm, float) / 1000.0
    return 2000.0 * tau * L / (G * np.asarray(twist_rad, float))


def wall_for_stiffness(od_mm, Kt_target, L_mm, G=G_STEEL):
    """Wall thickness [mm] that makes an OD deliver a target K_t.

    Invert J = pi(do^4 - di^4)/32 for di. Returns NaN where the target is
    impossible (a SOLID bar of this OD is already too soft) and clamps to solid
    where the required wall exceeds the radius.
    """
    do = np.asarray(od_mm, float) / 1000.0
    J_need = np.asarray(Kt_target, float) * (np.asarray(L_mm, float) / 1000.0) / G
    di4 = do ** 4 - 32.0 * J_need / np.pi
    out = np.full(np.shape(do), np.nan, float)
    ok = di4 >= 0.0
    out = np.where(ok, (do - np.power(np.maximum(di4, 0.0), 0.25)) / 2.0 * 1000.0,
                   np.nan)
    return np.where(np.asarray(out) > np.asarray(od_mm) / 2.0,
                    np.asarray(od_mm) / 2.0, out)


# ---------------------------------------------------------------------------
#  blade
# ---------------------------------------------------------------------------


def taper_factor(tip_over_root, cube=False, n=4001):
    """Compliance multiplier for a LINEARLY TAPERED cantilever, vs a prismatic
    one of the ROOT width. 1.0 when tip == root.

    Tip deflection by unit load (Castigliano), width b(u) = b_r(1 + (r-1)u):

        weak axis   I ~ b(u)      ->  F(r) = int_0^1 (1-u)^2 / (1+(r-1)u)   du
        strong axis I ~ b(u)^3    ->  G(r) = int_0^1 (1-u)^2 / (1+(r-1)u)^3 du

    both = 1/3 for a rectangle. Integrated NUMERICALLY on purpose: the closed
    forms
        F = [r^2 ln r - 1.5r^2 + 2r - 0.5] / (r-1)^3
        G = [ln r + 1.5 + r^2/2 - 2r]      / (r-1)^3
    are exact but suffer catastrophic cancellation as r -> 1, where a blade
    that is only slightly tapered is exactly the common case. The validation
    suite checks this routine against those closed forms away from r = 1.
    """
    r = float(tip_over_root)
    if r <= 0:
        raise ValueError("tip width must be positive")
    if abs(r - 1.0) < 1e-12:
        return 1.0                      # prismatic: exact, don't integrate it
    if n % 2 == 0:                      # Simpson needs an odd number of samples
        n += 1
    u = np.linspace(0.0, 1.0, n)
    f = (1.0 - u) ** 2 / ((1.0 + (r - 1.0) * u) ** (3 if cube else 1))
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((f * w).sum() * (1.0 / (n - 1)) / 3.0) * 3.0


def blade_principal_rates(b_root_mm, b_tip_mm, h_mm, length_mm, E=E_STEEL):
    """(k_weak, k_strong) tip rates of the blade [N/mm].

    weak  = blade FLAT, bending through its thickness   (I ~ b h^3 / 12)
    strong= blade EDGE-ON, bending through its width    (I ~ h b^3 / 12)

    A taper softens BOTH, but the strong axis far more (it scales with b^3), so
    tapering also shrinks the adjustment RANGE, not just the stiffness. If tip
    width == root width this reduces exactly to the prismatic rectangle.
    """
    b_r = float(b_root_mm) / 1000.0
    b_t = float(b_tip_mm) / 1000.0
    h = float(h_mm) / 1000.0
    ell = float(length_mm) / 1000.0
    r = b_t / b_r
    k_weak = 3.0 * E * (b_r * h ** 3 / 12.0) / ell ** 3 / taper_factor(r, False)
    k_strong = 3.0 * E * (h * b_r ** 3 / 12.0) / ell ** 3 / taper_factor(r, True)
    return k_weak / 1000.0, k_strong / 1000.0


def blade_rate_at_angle(theta_deg, k_weak, k_strong):
    """Blade tip rate at blade rotation theta [N/mm] -- COMPLIANCES combine.

        k(theta) = 1 / ( cos^2(theta)/k_weak + sin^2(theta)/k_strong )

    This is UNSYMMETRICAL BENDING. The drop link pulls vertically; the blade's
    principal axes are at theta to that. Decompose the load onto the two axes,
    deflect each, and take the VERTICAL component of the result:

        d = P[ cos^2/k_weak + sin^2/k_strong ]

    so the COMPLIANCES add as cos^2/sin^2, NOT the stiffnesses. Combining the
    second moments arithmetically -- I_min cos^2 + I_max sin^2 -- is a real and
    common error: at 45 deg on a 25:1 blade it overstates the rate by 6.8x.
    Both forms agree at 0 and 90 deg, so only the intermediate positions are
    wrong, which is precisely where every blade setting lives.
    """
    th = np.radians(np.asarray(theta_deg, float))
    return 1.0 / (np.cos(th) ** 2 / float(k_weak)
                  + np.sin(th) ** 2 / float(k_strong))


def blade_positions(n=6, theta_lo=0.0, theta_hi=90.0):
    """Blade angles [deg] for n detent positions, evenly spaced in ANGLE.

    Position 1 = softest (flat), position n = stiffest (edge-on). The benchmark
    same chart is usually drawn with 6 front positions and 4 rear ones.
    """
    return np.linspace(float(theta_lo), float(theta_hi), int(n))


def series(k1, k2):
    """Two stiffnesses in series -- softer than either:  1/k = 1/k1 + 1/k2."""
    k1, k2 = np.asarray(k1, float), np.asarray(k2, float)
    return 1.0 / (1.0 / k1 + 1.0 / k2)


def rate_vs_position(bars, blade, a_mm, mr, L_mm, n_pos=6,
                     theta_lo=0.0, theta_hi=90.0, G=G_STEEL, E=E_STEEL):
    """ARB WHEEL rate at each blade position, for each bar section.

    This is the usual blade-rate chart: positions along x, rate up y, one
    curve per bar cross-section. The bar and the blade act in SERIES at the drop
    link, so the softer of the two dominates and the curve flattens wherever the
    bar is the limit -- which is why a real 6-position chart spans only about
    2.2-2.8x even though the blade alone is 25:1.

    `bars`  : iterable of (od_mm, wall_mm) -- normally the two LIGHTEST that pass
    `blade` : (b_root_mm, b_tip_mm, h_mm, length_mm)
    Returns (angles, {label: rates}) with rates referred to the WHEEL.
    """
    th = blade_positions(n_pos, theta_lo, theta_hi)
    kw, ks = blade_principal_rates(*blade, E=E)
    k_blade = blade_rate_at_angle(th, kw, ks)

    out = {}
    for od, wall in bars:
        Kt = float(torsional_stiffness(polar_J(od, wall), L_mm, G))
        k_arm = float(rate_at_arm(Kt, a_mm))
        out[f"{od:.0f} x {wall:.2f} mm"] = np.asarray(
            to_wheel(series(k_arm, k_blade), mr), float)
    return th, out


def lightest_passing(rows, n=2):
    """The n lightest bar options that pass, for the position chart.

    Mass is what you minimise once stiffness and stress are already satisfied.
    """
    ok = [r for r in rows if r.get("ok")]
    ok.sort(key=lambda r: r["mass_kg_m"])
    return ok[:n]


def nearest_standard_wall(wall_mm):
    """Snap a computed wall to the nearest tube a fabricator can actually buy."""
    return min(STANDARD_WALL_MM, key=lambda w: abs(w - float(wall_mm)))


def chart_sections(rows, n=2):
    """The sections the position chart should plot -- taken from the TABLE.

    The table's rows each carry their own SOLVED wall, chosen so that bar
    delivers exactly the required K_t. That makes every row rate-IDENTICAL, so
    plotting two of them as-is would draw one curve twice. Snapping each to the
    nearest STANDARD wall fixes that: the diameters stay the ones the table
    recommends, the walls become buildable, and the small rounding either way
    naturally brackets the requirement.

    Returns (od, wall) pairs, softest first.
    """
    ok = [r for r in rows if r.get("ok")][:n]
    out = [(r["od_mm"], nearest_standard_wall(r["wall_mm"])) for r in ok]
    return sorted(out, key=lambda ow: polar_J(ow[0], ow[1]))


def bracket_sections(Kt_need, L_mm, twist_rad, wall_mm=3.0, tau_allow_MPa=550.0,
                     G=G_STEEL, ods=None):
    """A SOFT and a STIFF real tube section straddling the requirement.

    Why not just "the two lightest that pass"? Because every bar in that table
    was sized to deliver the SAME K_t, so they are rate-IDENTICAL and their
    curves would lie on top of each other. A setup sheet's two curves are two
    different pieces of HARDWARE at a common wall (50x3 front, 30x3 and 40x3
    rear) -- a soft option and a stiff one. This reproduces that: fix the wall
    at a standard size, walk the catalogue diameters, and return the nearest
    section below the requirement and the nearest above.

    Returns a list of dicts (1 or 2 of them), softest first.
    """
    cands = []
    for od in (CATALOGUE_OD_MM if ods is None else ods):
        if wall_mm >= od / 2.0:
            continue
        Kt = float(torsional_stiffness(polar_J(od, wall_mm), L_mm, G))
        tau = float(shear_stress_MPa(od, twist_rad, L_mm, G))
        if tau > tau_allow_MPa:
            continue
        cands.append(dict(od_mm=od, wall_mm=float(wall_mm), Kt=Kt, tau_MPa=tau,
                          ratio=Kt / Kt_need,
                          mass_kg_m=float(bar_mass_per_m(od, wall_mm))))
    if not cands:
        return []
    below = [c for c in cands if c["Kt"] <= Kt_need]
    above = [c for c in cands if c["Kt"] > Kt_need]
    out = []
    if below:
        out.append(max(below, key=lambda c: c["Kt"]))
    if above:
        out.append(min(above, key=lambda c: c["Kt"]))
    if len(out) < 2:                       # all on one side -- take the two nearest
        cands.sort(key=lambda c: abs(np.log(c["ratio"])))
        out = sorted(cands[:2], key=lambda c: c["Kt"])
    return out


# ---------------------------------------------------------------------------
#  the drop-link pickup: motion ratio SOLVED from the suspension geometry
# ---------------------------------------------------------------------------


def pickup_motion_ratio(pts, static_camber, pickup_xyz, attach=None,
                        span=6.0, n=61):
    """ARB motion ratio: VERTICAL travel of the drop-link pickup per mm of wheel
    travel, solved from the real linkage.

    The CAD has no ARB, so the pickup is a DESIGN INPUT -- but once you place it,
    its motion ratio is not a guess: the point rides on whichever member it is
    bolted to and the solver carries it through the stroke, exactly as it does
    the damper mount.

    `attach` is "lower" / "upper" / "upright", or None to pick by proximity to
    the two ball joints. A near-vertical drop link is assumed, so the vertical
    component is the whole story; a strongly inclined link would need its own
    axis and would only ever reduce this.
    """
    from scrub_trail import rotate_about_line
    import wishbone_solver as W

    g = W.corner_geometry(pts, static_camber)
    P0 = np.asarray(pickup_xyz, float)
    if attach is None:
        attach = ("lower" if np.linalg.norm(P0 - g["LBJ"])
                  <= np.linalg.norm(P0 - g["UBJ"]) else "upper")

    zs, ps = [], []
    for th in np.linspace(-span, span, n):
        try:
            r = W.solve_at(g, th)
        except Exception:                                     # noqa: BLE001
            continue
        if attach == "lower":
            p = rotate_about_line(P0, g["LIF"], g["ax_l"], th)[0]
        elif attach == "upper":
            p = rotate_about_line(P0, g["UIF"], g["ax_u"], r["thU"])[0]
        else:
            R, t = W.rigid_map(g["LBJ"], g["UBJ"], g["TRO"],
                               r["LBJ"], r["UBJ"], r["TRO"])
            p = R @ P0 + t
        zs.append(r["WC"][2])
        ps.append(p[2])
    if len(zs) < 5:
        return float("nan"), attach
    zs, ps = np.asarray(zs), np.asarray(ps)
    i0 = len(zs) // 2
    mr = (ps[i0 + 1] - ps[i0 - 1]) / (zs[i0 + 1] - zs[i0 - 1])
    return float(mr), attach


# ---------------------------------------------------------------------------
#  targets -> required ARB rates
# ---------------------------------------------------------------------------


def solve_for_targets(car, rc_f, rc_r, roll_grad_target, lltd_front_target,
                      lat_g=1.0, iters=60):
    """ARB wheel rates [N/mm] that hit a roll-gradient and an LLTD target.

    Two unknowns (front and rear ARB rate), two targets. Solved with a damped
    2x2 Newton on the EXISTING validated load-transfer model rather than a
    closed form, so the geometric (roll-centre) share and the unsprung terms are
    all accounted for exactly as the g-g page does them.

    Returns (arb_f, arb_r, info). Rates are clamped at zero -- a target the
    springs alone already overshoot cannot be met by ADDING a bar, and the info
    dict says so.
    """
    car = dict(car)

    def resid(x):
        c = dict(car)
        c["arb_f"], c["arb_r"] = float(max(x[0], 0.0)), float(max(x[1], 0.0))
        c["k_roll_f"] = ltr.roll_stiffness(c["kw_f"], c["arb_f"], c["track_f_mm"])
        c["k_roll_r"] = ltr.roll_stiffness(c["kw_r"], c["arb_r"], c["track_r_mm"])
        t = ltr.transfer(lat_g, rc_f, rc_r, c)
        return np.array([t["roll_deg"] / lat_g - roll_grad_target,
                         t["lltd_f"] - lltd_front_target])

    x = np.array([max(car.get("arb_f", 50.0), 1.0),
                  max(car.get("arb_r", 50.0), 1.0)], float)
    for _ in range(iters):
        r = resid(x)
        if np.max(np.abs(r / np.array([max(roll_grad_target, 1e-6), 1.0]))) < 1e-10:
            break
        Jm = np.zeros((2, 2))
        for j in range(2):
            dx = max(abs(x[j]) * 1e-6, 1e-6)
            xp = x.copy()
            xp[j] += dx
            Jm[:, j] = (resid(xp) - r) / dx
        try:
            step = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            break
        x = np.maximum(x + 0.7 * step, 0.0)

    r = resid(x)
    return float(x[0]), float(x[1]), dict(
        roll_grad_err=float(r[0]), lltd_err=float(r[1]),
        converged=bool(np.max(np.abs(r)) < 1e-6),
        floored=bool(x[0] <= 0.0 or x[1] <= 0.0))


def springs_only(car):
    """Roll stiffness and LLTD with NO bars -- the baseline the ARBs add to."""
    c = dict(car)
    c["arb_f"] = c["arb_r"] = 0.0
    c["k_roll_f"] = ltr.roll_stiffness(c["kw_f"], 0.0, c["track_f_mm"])
    c["k_roll_r"] = ltr.roll_stiffness(c["kw_r"], 0.0, c["track_r_mm"])
    return c


# ---------------------------------------------------------------------------
#  the catalogue
# ---------------------------------------------------------------------------


def size_bar(Kt_target, L_mm, roll_deg, track_mm, a_mm, mr,
             tau_allow_MPa=450.0, G=G_STEEL, ods=None, min_wall_mm=2.0):
    """For each catalogue OD: the wall needed, the stress, and whether it works.

    Returns a list of dicts, one per OD, with `verdict` explaining any failure
    in words rather than leaving a silent NaN.
    """
    twist = bar_twist_rad(roll_deg, track_mm, a_mm, mr)
    out = []
    for od in (CATALOGUE_OD_MM if ods is None else ods):
        wall = float(wall_for_stiffness(od, Kt_target, L_mm, G))
        tau = float(shear_stress_MPa(od, twist, L_mm, G))
        solid_Kt = float(torsional_stiffness(polar_J(od), L_mm, G))
        if np.isnan(wall):
            verdict = f"too soft even solid ({solid_Kt / Kt_target:.2f}x needed)"
        elif tau > tau_allow_MPa:
            verdict = f"OVERSTRESSED {tau:.0f} > {tau_allow_MPa:.0f} MPa"
        elif wall < min_wall_mm:
            verdict = f"wall {wall:.2f} mm below the {min_wall_mm:.0f} mm minimum"
        elif wall >= od / 2.0 - 1e-9:
            verdict = "solid"
        else:
            verdict = "OK"
        out.append(dict(od_mm=od, wall_mm=wall, tau_MPa=tau,
                        J=float(polar_J(od, wall if np.isfinite(wall) else None)),
                        mass_kg_m=float(bar_mass_per_m(
                            od, wall if np.isfinite(wall) else None)),
                        verdict=verdict, ok=(verdict in ("OK", "solid"))))
    return out


if __name__ == "__main__":
    import springs as sp

    car = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
    base = springs_only(car)
    kf = ltr.roll_stiffness(car["kw_f"], 0.0, car["track_f_mm"])
    kr = ltr.roll_stiffness(car["kw_r"], 0.0, car["track_r_mm"])
    print("SPRINGS ALONE")
    print(f"   roll stiffness F {kf / 1000:9.0f}  R {kr / 1000:9.0f} Nm/rad"
          f"   front share {kf / (kf + kr) * 100:.1f} %")
    t0 = ltr.transfer(1.0, -283.6, -52.0, base)
    print(f"   roll gradient {t0['roll_deg']:.3f} deg/g   front LLTD {t0['lltd_f']:.1f} %")

    print("\nTARGETS: 0.60 deg/g, 50 % front LLTD")
    af, ar, info = solve_for_targets(car, -283.6, -52.0, 0.60, 50.0)
    print(f"   required ARB wheel rate  F {af:7.1f}  R {ar:7.1f} N/mm   "
          f"converged={info['converged']}")

    # Defaults sized for a drop-link pickup at the damper mount, which is a long
    # way outboard and so gives a HIGH ARB motion ratio -- lots of drop-link
    # travel, hence a long arm and a long bar to keep the twist (and the stress)
    # sensible. Moving the pickup inboard would let both shrink.
    A_MM, L_MM = 350.0, 1000.0
    MR_F, MR_R = 0.9908, 0.7635          # solved from the hard points
    for tag, k, track, MR in (("FRONT", af, car["track_f_mm"], MR_F),
                              ("REAR", ar, car["track_r_mm"], MR_R)):
        Kt = float(required_Kt(k, A_MM, MR))
        tw = float(bar_twist_rad(3.0, track, A_MM, MR))
        print(f"\n{tag}: need K_t {Kt:.0f} Nm/rad   a {A_MM:.0f} mm, MR {MR:.4f}, "
              f"bar twist at 3 deg roll = {np.degrees(tw):.1f} deg")
        print(f"   {'OD':>6}{'wall':>8}{'tau MPa':>10}{'kg/m':>8}   verdict")
        for row in size_bar(Kt, L_MM, 3.0, track, A_MM, MR):
            w = f"{row['wall_mm']:.2f}" if np.isfinite(row["wall_mm"]) else "  -"
            print(f"   {row['od_mm']:6.0f}{w:>8}{row['tau_MPa']:10.0f}"
                  f"{row['mass_kg_m']:8.2f}   {row['verdict']}")
