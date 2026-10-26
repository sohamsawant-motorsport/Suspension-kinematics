# -*- coding: utf-8 -*-
"""
Camber change of a road wheel steered about its kingpin (steering) axis.
====================================================================

PHYSICS
-------
The kingpin / steering axis is the line through the lower and upper ball joints
(LBJ -> UBJ) of the front upright. It is inclined by two angles:

    KPI    (lambda) : lateral inclination, seen in the FRONT view.
    caster (nu)     : longitudinal inclination, seen in the SIDE view.

Steering rotates the whole upright (and wheel) about this single tilted axis by
the steer angle delta. Because the axis is NOT vertical, the wheel's spin axis
sweeps out of the horizontal plane as it turns -> the camber changes.

We compute this EXACTLY with Rodrigues' rotation formula. That matters here
because the brief sweeps delta over +/-45 deg, where the classic small-angle
formula  dgamma ~= -sin(nu) sin(d) + sin(lam) (1 - cos d)  starts to drift.
(`camber_change_approx` below is that formula, kept for comparison.)

FRAME (self-contained, RIGHT front wheel)
-----------------------------------------
    x = forward (+)      y = inboard / toward car centre (+)      z = up (+)

    Steering-axis unit vector u (LBJ -> UBJ, pointing up):
        KPI > 0     -> top leans inboard   (+y)
        caster > 0  -> top leans rearward  (-x)
      so  u ~ ( -tan(nu), +tan(lam), 1 ), then normalised.

    Wheel spin axis at zero steer & zero camber points straight outboard:
        s0 = (0, -1, 0).

    Steer by delta about u  ->  s(delta) via Rodrigues:
        s = s0 cos d + (u x s0) sin d + u (u . s0)(1 - cos d)
      with  u . s0 = -uy   and   u x s0 = (uz, 0, -ux).
    Only the vertical component sz is needed for camber:
        sz = -ux sin d + uz (-uy)(1 - cos d)
    Camber (inclination) is the tilt of the spin axis out of horizontal:
        gamma = arcsin(-sz)          [right wheel: top-out camber => sz < 0]

SIGN of the result (camber change, deg)
---------------------------------------
    negative  -> wheel gains NEGATIVE camber (top leans toward the car centre)
                 = FAVOURABLE for the loaded outside tyre in a corner.
    positive  -> wheel gains positive camber (leans out) = unfavourable.

    With delta > 0 this wheel turns toward the outside of a corner:
        * caster contributes  -sin(nu) sin(d)  -> negative (favourable), grows
          linearly and is ANTI-symmetric in steer direction.
        * KPI contributes      +sin(lam)(1-cos d) -> positive (unfavourable),
          second-order and SYMMETRIC (same sign whichever way you steer).
    => run enough caster to beat KPI and net a favourable camber gain in steer.

Self-check (`python steer_camber.py`) reproduces the two textbook cases.
"""
import numpy as np


def _axis(kpi_deg, caster_deg):
    """Unit vector of the kingpin axis (pointing up). Broadcasts over inputs."""
    lam = np.radians(np.asarray(kpi_deg, dtype=float))
    nu = np.radians(np.asarray(caster_deg, dtype=float))
    ux = -np.tan(nu)
    uy = np.tan(lam)
    uz = 1.0
    n = np.sqrt(ux * ux + uy * uy + uz * uz)   # broadcasts to shape of ux+uy
    return ux / n, uy / n, uz / n


def camber_change(kpi_deg, caster_deg, steer_deg):
    """Exact camber change [deg] for a wheel steered `steer_deg` about a kingpin
    axis of inclination `kpi_deg` (KPI) and `caster_deg` (caster).

    All three arguments broadcast against each other (scalars or numpy arrays),
    so you can pass a steer sweep, a KPI sweep, or a full KPI x caster grid.

    The four steps below are the same A/B/C/D printed by worked_example(), which
    shows every intermediate number for a concrete case. Run this file to see it.
    """
    # --- STEP A: build the kingpin axis unit vector from the two inclinations
    ux, uy, uz = _axis(kpi_deg, caster_deg)

    # --- STEP B: the steer angle's trig
    d = np.radians(np.asarray(steer_deg, dtype=float))
    c, s = np.cos(d), np.sin(d)

    # --- STEP C: rotate the spin axis s0 = (0,-1,0) about the kingpin axis by d
    #     (Rodrigues). Only the vertical component s_z is needed for camber:
    #        u . s0  = -uy          u x s0 = (uz, 0, -ux)
    #        s_z = -ux*sin d  -  uz*uy*(1 - cos d)
    #             \__caster__/     \_____KPI_____/
    #     caster pushes s_z UP (=> negative/favourable camber);
    #     KPI pulls it DOWN (=> positive/unfavourable camber).
    term_caster = (-ux) * s
    term_kpi = uz * (-uy) * (1.0 - c)
    sz = term_caster + term_kpi

    # --- STEP D: turn the vertical component back into an angle
    return np.degrees(np.arcsin(np.clip(-sz, -1.0, 1.0)))


def camber_change_approx(kpi_deg, caster_deg, steer_deg):
    """Classic small-angle formula, for comparison with the exact solution:
        dgamma ~= -sin(nu) sin(d) + sin(lam) (1 - cos d)
    """
    lam = np.radians(np.asarray(kpi_deg, dtype=float))
    nu = np.radians(np.asarray(caster_deg, dtype=float))
    d = np.radians(np.asarray(steer_deg, dtype=float))
    return np.degrees(-np.sin(nu) * np.sin(d) + np.sin(lam) * (1.0 - np.cos(d)))


def worked_example(kpi_deg=5.3, caster_deg=1.7, steer_deg=10.0):
    """Print the camber-by-steer calculation step by step, showing every
    intermediate number, for one concrete geometry.

    This is the documented reference calculation: the STEP A/B/C/D labels match
    the comments inside camber_change(), the derivation in MATHS.md, and section
    A.6 of the report appendix. Use it to hand-check the code, or to reproduce a
    number in CAD.

    Defaults are the baseline Huracan geometry measured in Fusion.
    """
    lam, nu, d = float(kpi_deg), float(caster_deg), float(steer_deg)
    L = []
    add = L.append
    add("=" * 68)
    add(f"WORKED EXAMPLE   KPI lambda = {lam} deg,  caster nu = {nu} deg,"
        f"  steer delta = {d} deg")
    add("=" * 68)

    # STEP A ---------------------------------------------------------------
    tl, tn = np.tan(np.radians(lam)), np.tan(np.radians(nu))
    ux_r, uy_r, uz_r = -tn, tl, 1.0
    n = np.sqrt(ux_r ** 2 + uy_r ** 2 + uz_r ** 2)
    ux, uy, uz = ux_r / n, uy_r / n, uz_r / n
    add("")
    add("STEP A  -  build the kingpin axis unit vector  u = (-tan nu, +tan lam, 1)/|.|")
    add(f"   tan(nu={nu})   = {tn:.5f}        (how far the axis leans BACK per 1 up)")
    add(f"   tan(lam={lam})  = {tl:.5f}        (how far it leans INBOARD per 1 up)")
    add(f"   u_raw          = ({ux_r:+.5f}, {uy_r:+.5f}, {uz_r:+.5f})")
    add(f"   |u_raw|        = sqrt(1 + tan^2 nu + tan^2 lam) = {n:.5f}")
    add(f"   u              = ({ux:+.5f}, {uy:+.5f}, {uz:+.5f})")

    # STEP B ---------------------------------------------------------------
    cd, sd = np.cos(np.radians(d)), np.sin(np.radians(d))
    add("")
    add("STEP B  -  the steer angle's trig")
    add(f"   sin({d}) = {sd:.5f}     cos({d}) = {cd:.5f}     (1 - cos) = {1-cd:.5f}")

    # STEP C ---------------------------------------------------------------
    t_cast = -ux * sd
    t_kpi = -uz * uy * (1 - cd)
    sz = t_cast + t_kpi
    add("")
    add("STEP C  -  rotate the spin axis s0=(0,-1,0) about u by delta (Rodrigues)")
    add("           only the vertical component s_z matters for camber:")
    add("           s_z = -u_x*sin d  -  u_z*u_y*(1 - cos d)")
    add(f"   caster term = -u_x*sin d           = {t_cast:+.6f}   (pushes s_z UP -> favourable)")
    add(f"   KPI term    = -u_z*u_y*(1 - cos d) = {t_kpi:+.6f}   (pulls s_z DOWN -> unfavourable)")
    add(f"   s_z         = {t_cast:+.6f} {'+' if t_kpi>=0 else '-'} {abs(t_kpi):.6f} = {sz:+.6f}")

    # STEP D ---------------------------------------------------------------
    g = np.degrees(np.arcsin(-sz))
    add("")
    add("STEP D  -  turn the vertical component back into an angle")
    add(f"   d.gamma = arcsin(-s_z) = arcsin({-sz:+.6f}) = {g:+.4f} deg")

    # cross-checks ---------------------------------------------------------
    ap = float(camber_change_approx(lam, nu, d))
    ex = float(camber_change(lam, nu, d))
    add("")
    add("CROSS-CHECKS")
    add(f"   classical formula  -sin(nu)sin(d) + sin(lam)(1-cos d) = {ap:+.4f} deg")
    add(f"   camber_change()                                       = {ex:+.4f} deg")
    add(f"   agreement                                             = {abs(ex-ap):.4f} deg")

    # superposition --------------------------------------------------------
    only_c = float(camber_change(0.0, nu, d))
    only_k = float(camber_change(lam, 0.0, d))
    add("")
    add("SUPERPOSITION  -  the two effects simply add, and caster dominates")
    add(f"   caster only (nu={nu}, lam=0)  -> {only_c:+.4f} deg")
    add(f"   KPI only    (lam={lam}, nu=0) -> {only_k:+.4f} deg")
    add(f"   sum                            -> {only_c+only_k:+.4f} deg")
    add(f"   both together (exact)          -> {ex:+.4f} deg")
    if abs(only_k) > 1e-9:
        add(f"   caster is {abs(only_c/only_k):.1f}x the KPI contribution, opposite sign")

    # growth ---------------------------------------------------------------
    add("")
    add("GROWTH WITH STEER  -  caster ~ linear in d, KPI ~ d^2")
    add("    d      caster term    KPI term     total (exact)")
    for dd in (0, 5, 10, 15, 20):
        add(f"   {dd:3d}      {float(camber_change_approx(0.0,nu,dd)):+8.4f}    "
            f"{float(camber_change_approx(lam,0.0,dd)):+8.4f}     "
            f"{float(camber_change(lam,nu,dd)):+8.4f}")
    add("=" * 68)
    text = "\n".join(L)
    print(text)
    return text


if __name__ == "__main__":
    worked_example()
    print()
    # --- textbook verification -------------------------------------------------
    # 1) Caster only (nu=5, KPI=0), steer this wheel to the outside by 20 deg.
    #    Expect FAVOURABLE (negative) camber ~ -1.71 deg.
    # 2) KPI only (lam=8, caster=0), same steer.
    #    Expect UNFAVOURABLE (positive) camber ~ +0.48 deg, and second-order.
    print("Verification of the camber-by-steer core")
    print("-" * 48)
    for name, kpi, cas in [("caster only (nu=5, KPI=0)", 0.0, 5.0),
                           ("KPI only (KPI=8, nu=0)", 8.0, 0.0)]:
        ex = camber_change(kpi, cas, 20.0)
        ap = camber_change_approx(kpi, cas, 20.0)
        print(f"{name:28s}  d=20deg -> exact {ex:+.3f} deg   approx {ap:+.3f} deg")
    # symmetry: caster is odd, KPI is even in steer direction
    print("-" * 48)
    print(f"caster odd?  d=+20 {camber_change(0,5,20):+.3f}  vs "
          f"d=-20 {camber_change(0,5,-20):+.3f}")
    print(f"KPI even?    d=+20 {camber_change(8,0,20):+.3f}  vs "
          f"d=-20 {camber_change(8,0,-20):+.3f}")
    # exact vs approx spread out at large steer (baseline geometry, CAD-measured)
    print("-" * 48)
    for d in (10, 20, 30, 45):
        ex = camber_change(5.3, 1.7, d)
        ap = camber_change_approx(5.3, 1.7, d)
        print(f"stock 5.3/1.7  d={d:2d}deg -> exact {ex:+.3f}  approx {ap:+.3f}  "
              f"(diff {ex-ap:+.3f})")
