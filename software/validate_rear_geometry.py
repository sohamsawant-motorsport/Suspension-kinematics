# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for rear_geometry (rear static geometry + front-view RC).
==========================================================================

Same discipline as the front: cross-check against the validated cores and
against the geometric DEFINITIONS, so the rear numbers are trustworthy before
any study is built on them.

  1  ANGLES        - kpi/caster from rear_geometry == scrub_trail.kpi_caster.
  2  BALL JOINTS   - the 1-DOF ball-joint solve is an identity at theta = 0 and
                     preserves the kingpin length at every angle (rigid upright).
  3  INSTANT C.    - each ball joint's front-view velocity is perpendicular to
                     the line from it to the IC (that IS the definition), and the
                     IC is independent of the perturbation size (a true limit).
  4  ROLL CENTRE   - the RC lies ON the force line (patch -> IC) AND on the car
                     centreline X = 0, and FVSA == |IC_x - patch_x|.
  5  SCRUB/TRAIL   - match scrub_trail, and scrub moves with static camber.
  6  SANITY        - arm lengths, kingpin length and track are physical.

Run:  python validate_rear_geometry.py
"""
import numpy as np

import rear_geometry as rg
from scrub_trail import kpi_caster, scrub_radius, mechanical_trail

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


P = rg.load_rear()
CAM = rg.REAR_STATIC_CAMBER
g = rg.static_geometry(P, CAM)

print("=" * 74)
print("VALIDATION - rear static geometry")
print("=" * 74)

# ------------------------------------------------------------------ 1. angles
print("\n1. ANGLES")
kpi_c, cas_c = kpi_caster(P[rg.LBJ], P[rg.UBJ])
check("kpi == scrub_trail.kpi_caster", abs(g["kpi"] - kpi_c) < 1e-12,
      f"{g['kpi']:+.4f} deg")
check("caster == scrub_trail.kpi_caster", abs(g["caster"] - cas_c) < 1e-12,
      f"{g['caster']:+.4f} deg")

# -------------------------------------------------------------- 2. ball joints
print("\n2. BALL-JOINT SOLVE (two A-arms, rigid upright)")
l0, u0 = rg.ball_joints(P, 0.0)
check("identity at theta = 0: LBJ", np.allclose(l0, P[rg.LBJ], atol=1e-7),
      f"max move {np.max(np.abs(l0 - P[rg.LBJ])):.2e} mm")
check("identity at theta = 0: UBJ", np.allclose(u0, P[rg.UBJ], atol=1e-7),
      f"max move {np.max(np.abs(u0 - P[rg.UBJ])):.2e} mm")
kp0 = float(np.linalg.norm(np.asarray(P[rg.UBJ]) - P[rg.LBJ]))
worst = 0.0
for th in np.linspace(-8, 8, 17):
    lb, ub = rg.ball_joints(P, th)
    worst = max(worst, abs(np.linalg.norm(ub - lb) - kp0))
check("kingpin length preserved over +/-8 deg", worst < 1e-6,
      f"worst |d| {worst:.2e} mm")
# lower arm rides its own circle: LBJ stays its arm-length from the inner axis
lax = rg._arm_axis(P[rg.LIF], P[rg.LIR])
def _dist_to_axis(p, a, u):
    w = np.asarray(p, float) - a
    return np.linalg.norm(w - np.dot(w, u) * u)
r0 = _dist_to_axis(P[rg.LBJ], np.asarray(P[rg.LIF], float), lax)
lb5, _ = rg.ball_joints(P, 5.0)
check("LBJ stays on its arm circle", abs(_dist_to_axis(lb5, np.asarray(P[rg.LIF], float), lax) - r0) < 1e-6)

# ----------------------------------------------------------- 3. instant centre
print("\n3. FRONT-VIEW INSTANT CENTRE")
ic = g["IC"]
# each ball joint's front-view velocity must be perpendicular to (joint -> IC)
lp, up = rg.ball_joints(P, +0.25)
lm, um = rg.ball_joints(P, -0.25)
for nm, p0key, vp, vm in (("LBJ", rg.LBJ, lp, lm), ("UBJ", rg.UBJ, up, um)):
    p0 = np.asarray(P[p0key], float)[[0, 2]]
    vel = (vp - vm)[[0, 2]]
    to_ic = ic - p0
    cosang = abs(np.dot(vel, to_ic) / (np.linalg.norm(vel) * np.linalg.norm(to_ic)))
    check(f"{nm} velocity perpendicular to {nm}->IC", cosang < 1e-6,
          f"cos {cosang:.2e}")
# the IC is a limit: halving the perturbation must not move it
ic2 = rg.front_view_ic(P, d_theta=0.1)
check("IC independent of perturbation size", np.max(np.abs(ic - ic2)) < 1.0,
      f"|d| {np.max(np.abs(ic - ic2)):.3f} mm")

# ------------------------------------------------------------- 4. roll centre
print("\n4. ROLL CENTRE + FVSA")
patch2 = g["patch"][[0, 2]]
d = ic - patch2
rc = np.array([0.0, g["rc_height"]])
cross = abs(d[0] * (rc[1] - patch2[1]) - d[1] * (rc[0] - patch2[0]))
check("RC lies on the force line patch->IC", cross < 1e-6, f"offset {cross:.2e}")
check("RC lies on the centreline X = 0", abs(rc[0]) < 1e-12)
check("FVSA == |IC_x - patch_x|",
      abs(g["fvsa"] - abs(ic[0] - g["patch"][0])) < 1e-9,
      f"{g['fvsa']:.1f} mm")

# ------------------------------------------------------------- 5. scrub/trail
print("\n5. SCRUB / TRAIL")
check("scrub == scrub_trail.scrub_radius",
      abs(g["scrub"] - scrub_radius(P[rg.LBJ], P[rg.UBJ], P[rg.WC], CAM)) < 1e-12)
check("trail == scrub_trail.mechanical_trail",
      abs(g["trail"] - mechanical_trail(P[rg.LBJ], P[rg.UBJ], P[rg.WC], CAM)) < 1e-12)
g0 = rg.static_geometry(P, 0.0)
check("scrub moves with static camber (patch is camber-aware)",
      abs(g["scrub"] - g0["scrub"]) > 5.0,
      f"{g0['scrub']:.1f} at 0 deg -> {g['scrub']:.1f} at {CAM:.1f} deg")

# ---------------------------------------------------------------- 6. sanity
print("\n6. SANITY")
check("kingpin length matches the CAD", abs(g["kingpin_len"] - kp0) < 1e-9)
check("arm lengths physical", 150 < g["lower_arm"] < 350 and 150 < g["upper_arm"] < 350,
      f"upper {g['upper_arm']:.0f}  lower {g['lower_arm']:.0f} mm")
check("track physical", 1400 < g["track"] < 1800, f"{g['track']:.0f} mm")
check("wheelbase physical", 2400 < g["wheelbase"] < 2900, f"{g['wheelbase']:.0f} mm")

print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - rear static geometry agrees with the cores and the "
      "definitions.")
print("=" * 74)
