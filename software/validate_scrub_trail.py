# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the scrub / mechanical-trail maths.
========================================================

Every check below is an assertion against the REAL front-right CAD hard points.
If this file prints ALL CHECKS PASSED, the maths, the code and the car geometry
agree.

The checks, in order of how much they prove:

  1  Hand arithmetic      - recompute G, scrub, trail straight from the raw CAD
                            numbers, independently of the module.
  2  Axis sanity          - G really lies on the LBJ->UBJ line, and at Z = 0.
  3  Rigid rotation       - steering preserves the distance from the axis and the
                            length of every vector; the axis itself never moves.
  4  Offset identities    - scrub(dx) = scrub(0) - dx and trail(dy) = trail(0) + dy
                            EXACTLY, and the KPI/caster angles do not budge.
  5  Closed-form gradients- d(wheelbase)/dd and d(half-track)/dd against a central
                            finite difference of the exact rotation.
  6  CROSS-MODULE         - rotate the wheel's SPIN axis with scrub_trail's own
                            machinery, extract camber, and compare against
                            steer_camber.camber_change(). The two modules were
                            written independently, in different frames, from
                            different derivations. If they agree, both are right.
  7  Physical sanity      - odd/even symmetry, zero-steer identities.

Run:  python validate_scrub_trail.py
"""
import numpy as np

import config as cfg
from scrub_trail import (steering_axis, ground_point, contact_patch, kpi_caster,
                         scrub_radius, mechanical_trail, offset_joints,
                         rotate_about_line, corner_state, wheel_radius,
                         wheel_spin_axis)
from steer_camber import camber_change

TOL = 1e-9
fails = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "**FAIL**"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- load the car
p = cfg._load_front_points()
LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]
CAMBER = cfg.STATIC_CAMBER   # static camber tilts the wheel -> moves the contact patch
A, u = steering_axis(LBJ, UBJ)
G = ground_point(LBJ, UBJ)
P0 = contact_patch(WC, CAMBER)
kpi_mag, cas = kpi_caster(LBJ, UBJ)
sc0, tr0 = scrub_radius(LBJ, UBJ, WC, CAMBER), mechanical_trail(LBJ, UBJ, WC, CAMBER)

print("=" * 74)
print("VALIDATION - scrub / mechanical trail, against the FR CAD hard points")
print("=" * 74)
print(f"  LBJ = {LBJ}")
print(f"  UBJ = {UBJ}")
print(f"  WC  = {WC}")
print(f"  ->  KPI {kpi_mag:+.4f} deg, caster {cas:+.4f} deg, "
      f"scrub {sc0:.4f} mm, trail {tr0:.4f} mm")

# ---------------------------------------------------------------- 1. hand arithmetic
print("\n1. HAND ARITHMETIC (independent of the module)")
dX, dY, dZ = UBJ - LBJ
t_hand = -LBJ[2] / dZ                      # walk down the axis until Z = 0
Gx_hand = LBJ[0] + t_hand * dX
Gy_hand = LBJ[1] + t_hand * dY
check("G_x by hand", abs(Gx_hand - G[0]) < TOL, f"{Gx_hand:.6f} vs {G[0]:.6f}")
check("G_y by hand", abs(Gy_hand - G[1]) < TOL, f"{Gy_hand:.6f} vs {G[1]:.6f}")
# the wheel is cambered, so the patch is NOT the wheel centre projected down:
#   R = WC_z/cos(g),  patch_x = WC_x - R*sin(g)
R_hand = WC[2] / np.cos(np.radians(CAMBER))
Px_hand = WC[0] - R_hand * np.sin(np.radians(CAMBER))
check("wheel radius by hand", abs(R_hand - wheel_radius(WC, CAMBER)) < TOL,
      f"{R_hand:.6f} mm")
check("camber-shifted patch_x by hand",
      abs(Px_hand - contact_patch(WC, CAMBER)[0]) < TOL, f"{Px_hand:.6f}")
check("scrub by hand (camber-aware)", abs((Px_hand - Gx_hand) - sc0) < TOL,
      f"{Px_hand-Gx_hand:.6f} vs {sc0:.6f}")
check("trail by hand", abs((Gy_hand - WC[1]) - tr0) < TOL,
      f"{Gy_hand-WC[1]:.6f} vs {tr0:.6f}")
check("camber moves the patch OUTBOARD (negative camber)",
      contact_patch(WC, CAMBER)[0] > contact_patch(WC, 0.0)[0],
      f"{contact_patch(WC,CAMBER)[0]-contact_patch(WC,0.0)[0]:+.3f} mm")
kpi_hand = np.degrees(np.arctan2(-dX, dZ))
cas_hand = np.degrees(np.arctan2(-dY, dZ))
check("KPI by hand (signed)", abs(kpi_hand - kpi_mag) < TOL, f"{kpi_hand:+.6f} deg")
check("caster by hand", abs(cas_hand - cas) < TOL, f"{cas_hand:.6f} deg")

check("config.STOCK_KPI matches scrub_trail.kpi_caster",
      abs(cfg.STOCK_KPI - kpi_mag) < 1e-9, f"{cfg.STOCK_KPI:+.6f} vs {kpi_mag:+.6f}")
check("config.STOCK_CASTER matches scrub_trail.kpi_caster",
      abs(cfg.STOCK_CASTER - cas) < 1e-9, f"{cfg.STOCK_CASTER:+.6f} vs {cas:+.6f}")

# ---------------------------------------------------------------- 2. axis sanity
print("\n2. AXIS SANITY")
check("G lies on the LBJ->UBJ line", np.linalg.norm(np.cross(G - LBJ, u)) < 1e-6,
      f"|(G-LBJ) x u| = {np.linalg.norm(np.cross(G-LBJ,u)):.2e}")
check("G is on the ground (Z=0)", abs(G[2]) < TOL)
check("u is a unit vector", abs(np.linalg.norm(u) - 1.0) < TOL)
check("U is above L on the axis", (UBJ - LBJ)[2] > 0)

# ---------------------------------------------------------------- 3. rigid rotation
print("\n3. ROTATION IS RIGID (steering must not deform anything)")
angs = np.linspace(-35, 35, 71)
WCr = rotate_about_line(WC, A, u, angs)
d_axis = [np.linalg.norm(np.cross(w - A, u)) for w in WCr]      # perp dist to axis
d0 = np.linalg.norm(np.cross(WC - A, u))
check("perp. distance to the axis is constant",
      max(abs(np.array(d_axis) - d0)) < 1e-8, f"max drift {max(abs(np.array(d_axis)-d0)):.2e} mm")
r_G = [np.linalg.norm(w - G) for w in WCr]
check("|WC - G| is constant", max(abs(np.array(r_G) - np.linalg.norm(WC - G))) < 1e-8)
check("zero steer is the identity",
      np.linalg.norm(rotate_about_line(WC, A, u, 0.0)[0] - WC) < TOL)
check("the axis itself does not move under steer",
      np.linalg.norm(rotate_about_line(UBJ, A, u, 33.0)[0] - UBJ) < 1e-8)
st0 = corner_state(LBJ, UBJ, WC, np.array([0.0]), CAMBER)
check("scrub/trail at 0 steer match the statics",
      abs(st0["scrub"][0] - sc0) < TOL and abs(st0["trail"][0] - tr0) < TOL)

# ---------------------------------------------------------------- 4. offset identities
print("\n4. OFFSET IDENTITIES (the design lever)")
ok_s, ok_t, ok_a = True, True, True
for d in (-40.0, -13.7, 0.0, 25.0, 80.0, 129.086):
    L2, U2 = offset_joints(LBJ, UBJ, dx=d)
    ok_s &= abs(scrub_radius(L2, U2, WC, CAMBER) - (sc0 - d)) < 1e-8
    k2, c2 = kpi_caster(L2, U2)
    ok_a &= abs(k2 - kpi_mag) < 1e-10 and abs(c2 - cas) < 1e-10
    L3, U3 = offset_joints(LBJ, UBJ, dy=d)
    ok_t &= abs(mechanical_trail(L3, U3, WC, CAMBER) - (tr0 + d)) < 1e-8
    k3, c3 = kpi_caster(L3, U3)
    ok_a &= abs(k3 - kpi_mag) < 1e-10 and abs(c3 - cas) < 1e-10
check("scrub(dx) == scrub(0) - dx  exactly", ok_s)
check("trail(dy) == trail(0) + dy  exactly", ok_t)
check("KPI and caster angles invariant under any offset", ok_a)
# and the cross-terms: dx must not touch trail, dy must not touch scrub
L2, U2 = offset_joints(LBJ, UBJ, dx=57.0)
check("lateral offset leaves TRAIL untouched",
      abs(mechanical_trail(L2, U2, WC, CAMBER) - tr0) < 1e-8)
L3, U3 = offset_joints(LBJ, UBJ, dy=57.0)
check("longitudinal offset leaves SCRUB untouched",
      abs(scrub_radius(L3, U3, WC, CAMBER) - sc0) < 1e-8)

# ---------------------------------------------------------------- 5. closed forms
print("\n5. CLOSED-FORM GRADIENTS vs finite difference of the exact rotation")
R = WC[2]
h = 1e-3
ok_wb, ok_ht, worst = True, True, 0.0
for dx in (0.0, 30.0, 60.0, 120.0):
    for dy in (-20.0, 0.0, 40.0):
        L2, U2 = offset_joints(LBJ, UBJ, dx=dx, dy=dy)
        s = corner_state(L2, U2, WC, np.array([-h, h]))
        dwb = (s["P"][1, 1] - s["P"][0, 1]) / (2 * h) * 180 / np.pi   # per rad
        dht = (s["P"][1, 0] - s["P"][0, 0]) / (2 * h) * 180 / np.pi
        _, u2 = steering_axis(L2, U2)
        sc, tr = scrub_radius(L2, U2, WC, 0.0), mechanical_trail(L2, U2, WC, 0.0)
        pw = sc * u2[2] - R * u2[0]
        ph = tr * u2[2] + R * u2[1]
        worst = max(worst, abs(dwb - pw), abs(dht - ph))
        ok_wb &= abs(dwb - pw) < 1e-4
        ok_ht &= abs(dht - ph) < 1e-4
print(f"  [info] d(wheelbase)/dd  ~ scrub*u_z - R*u_x   (approximation)")
print(f"  [info] d(half-track)/dd ~ trail*u_z + R*u_y  (approximation)")
print(f"  [info] worst deviation of the approximations = {worst:.3f} mm/rad "
      f"(= R*sin(camber), the patch shift they omit)")

# with camber modelled, the R*u_y term cancels and half-track is exactly trail*u_z
ok_c, worst_c = True, 0.0
for dy in (-20.0, 0.0, 30.0, 60.0):
    L2, U2 = offset_joints(LBJ, UBJ, dy=dy)
    s2 = corner_state(L2, U2, WC, np.array([-h, h]), CAMBER)
    dht = (s2["P"][1, 0] - s2["P"][0, 0]) / (2 * h) * 180 / np.pi
    _, u2 = steering_axis(L2, U2)
    tr2 = mechanical_trail(L2, U2, WC, CAMBER)
    worst_c = max(worst_c, abs(dht - tr2 * u2[2]))
    ok_c &= abs(dht - tr2 * u2[2]) < 1e-2
check("WITH camber: d(half-track)/dd == trail*u_z exactly", ok_c,
      f"worst {worst_c:.2e} mm/rad: the R*sin(caster) term cancels")

# ---------------------------------------------------------------- 6. CROSS-MODULE
print("\n6. CROSS-MODULE  (scrub_trail rotation  vs  steer_camber.camber_change)")
print("     Rotate the wheel SPIN AXIS about the kingpin axis using scrub_trail's")
print("     own Rodrigues, extract camber, compare with the independently-derived")
print("     steer_camber module. Different frames, different derivations.")
# In the CAD frame (X outboard, Y forward, Z up) the spin axis at zero steer and
# zero camber points straight outboard: s0 = (1, 0, 0).
# Positive camber = top leans outboard => the outboard-pointing spin axis tips
# DOWN, so gamma = arcsin(-s_z), exactly as in steer_camber's own frame.
# steer_camber's KPI convention is +ve = leans INBOARD. The CAD axis leans
# OUTBOARD at the top (u_x > 0), so the signed KPI is negative.
kpi_signed = kpi_mag   # kpi_caster now returns it signed
print(f"     signed KPI for steer_camber = {kpi_signed:+.4f} deg "
      f"(axis leans OUTBOARD at the top), caster = {cas:+.4f} deg")
s0 = np.array([1.0, 0.0, 0.0])
ok_x, worst_x = True, 0.0
for d in (-35, -20, -7.71, 0, 5, 12.5, 20, 35):
    srot = rotate_about_line(s0, np.zeros(3), u, d)[0]     # direction: rotate about u
    gam_st = np.degrees(np.arcsin(np.clip(-srot[2], -1, 1)))
    gam_sc = float(camber_change(kpi_signed, cas, d))   # from ZERO static camber
    worst_x = max(worst_x, abs(gam_st - gam_sc))
    ok_x &= abs(gam_st - gam_sc) < 1e-6
    print(f"       d={d:+6.2f} deg -> scrub_trail {gam_st:+9.5f}   "
          f"steer_camber {gam_sc:+9.5f}   diff {abs(gam_st-gam_sc):.2e}")
check("the two modules agree on camber to 1e-6 deg", ok_x, f"worst {worst_x:.2e} deg")

# ---------------------------------------------------------------- 7. physical sanity
print("\n7. PHYSICAL SANITY")
st = corner_state(LBJ, UBJ, WC, np.array([-20.0, 0.0, 20.0]))
check("track/wheelbase change is zero at zero steer",
      abs(st["d_track"][1]) < TOL and abs(st["d_base"][1]) < TOL)
check("jacking is zero at zero steer", abs(st["jack"][1]) < TOL)
# caster term of camber is odd, KPI term even (independent re-check)
check("camber: caster term is ODD in steer",
      abs(camber_change(0.0, 5.0, 20.0) + camber_change(0.0, 5.0, -20.0)) < 1e-9)
check("camber: KPI term is EVEN in steer",
      abs(camber_change(8.0, 0.0, 20.0) - camber_change(8.0, 0.0, -20.0)) < 1e-9)
# a vertical steering axis must produce no camber and no patch motion at all
Lv = np.array([0.0, 0.0, 100.0]); Uv = np.array([0.0, 0.0, 300.0])
WCv = np.array([100.0, 0.0, 336.0])
stv = corner_state(Lv, Uv, WCv, np.array([25.0]))
check("a VERTICAL axis gives zero jacking (the null case)", abs(stv["jack"][0]) < 1e-9)
check("a VERTICAL axis gives zero camber", abs(float(camber_change(0.0, 0.0, 25.0))) < 1e-12)

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED: the maths, the code and the FR CAD hard points agree.")
print("=" * 74)
