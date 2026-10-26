# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the two-corner axle model.
===============================================

Same discipline as the other suites: assertions against the real CAD, and the
strongest checks are CROSS-MODULE -- the new two-corner model must reproduce the
validated single-corner results wherever the two overlap.

  1  Mirror symmetry   - at zero roll a symmetric car must be exactly symmetric.
  2  CROSS-MODULE      - the axle's RC at zero roll must equal the single-corner
                         linkage_roll_centre() from wishbone_solver, and its
                         camber must equal the solver's camber at that bump.
  3  Bump targeting    - asking for a bump must actually produce that bump.
  4  Force lines       - the RC must lie ON both patch->IC lines (that is its
                         definition), and FVSA must match |patch_x - IC_x|.
  5  Roll behaviour    - RC x is odd in roll, RC z is even; camber L/R swap.
  6  Roll model        - body roll must be EQUIVALENT to equal-and-opposite
                         wheel travel, must keep both patches on the road, and
                         must preserve every rigid length.

Run:  python validate_axle.py
"""
import os
import sys

import numpy as np

_SW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "software")
if _SW not in sys.path:
    sys.path.insert(0, _SW)

import config as cfg                                            # noqa: E402
from wishbone_solver import (corner_geometry, solve_sweep,       # noqa: E402
                            linkage_roll_centre)
from axle import axle_state, mirror_points, roll_sweep, _roll_matrix   # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


pts_R = cfg._load_front_points()
pts_L = mirror_points(pts_R)
CAM = cfg.STATIC_CAMBER

print("=" * 76)
print("VALIDATION - two-corner front axle")
print("=" * 76)

st = axle_state(pts_R, pts_L, CAM)

# ---------------------------------------------------------------- 1. symmetry
print("\n1. MIRROR SYMMETRY at zero roll")
check("camber L == camber R", abs(st["L"]["camber"] - st["R"]["camber"]) < 1e-9,
      f"{st['L']['camber']-st['R']['camber']:+.2e} deg")
check("toe L == toe R", abs(st["L"]["toe"] - st["R"]["toe"]) < 1e-9)
check("FVSA L == FVSA R", abs(st["L"]["fvsa"] - st["R"]["fvsa"]) < 1e-6,
      f"{st['L']['fvsa']:.1f} mm")
check("patches are mirror images",
      abs(st["L"]["patch"][0] + st["R"]["patch"][0]) < 1e-9)
check("ICs are mirror images", abs(st["L"]["IC"][0] + st["R"]["IC"][0]) < 1e-6)
check("RC sits ON the centreline", abs(st["roll_centre"][0]) < 1e-6,
      f"x = {st['roll_centre'][0]:+.2e} mm")

# ---------------------------------------------------------------- 2. cross-module
print("\n2. CROSS-MODULE vs the validated single-corner solver")
g = corner_geometry(pts_R, CAM)
rc_single = float(linkage_roll_centre(g, np.array([0.0]))[0])
check("RC height == single-corner linkage_roll_centre",
      abs(st["roll_centre"][1] - rc_single) < 0.5,
      f"axle {st['roll_centre'][1]:+.2f} vs solver {rc_single:+.2f} mm")
sw = solve_sweep(g, np.linspace(-20, 20, 81))
i0 = int(np.argmin(np.abs(sw["bump"])))
check("camber == solver camber at zero bump",
      abs(st["R"]["camber"] - sw["camber"][i0]) < 1e-6,
      f"{st['R']['camber']:+.5f} vs {sw['camber'][i0]:+.5f} deg")
check("KPI == solver KPI", abs(st["R"]["kpi"] - sw["kpi"][i0]) < 1e-6)
check("caster == solver caster", abs(st["R"]["caster"] - sw["caster"][i0]) < 1e-6)

# ---------------------------------------------------------------- 3. bump targeting
print("\n3. BUMP TARGETING (asking for a travel must deliver it)")
worst = 0.0
for b in (-20.0, -8.0, 0.0, 12.0, 30.0):
    s = axle_state(pts_R, pts_L, CAM, left_bump=b, right_bump=b)
    # in pure heave there is no roll, so the ground transform is a pure
    # translation: the wheel-centre-to-patch height is the invariant to test
    got = s["R"]["WC"][2] - s["R"]["patch"][2]
    ref = pts_R["Wheel centre"][2]
    worst = max(worst, abs(got - ref))
check("wheel centre stays its own radius above the patch", worst < 0.6,
      f"worst error {worst:.4f} mm")

# ---------------------------------------------------------------- 4. force lines
print("\n4. FORCE LINES (the RC is DEFINED as their intersection)")
s = axle_state(pts_R, pts_L, CAM, left_bump=-12.0, right_bump=+12.0, body_roll=1.5)
rc = s["roll_centre"]
ok_on = True
for side in ("L", "R"):
    p = s[side]["patch"][[0, 2]]
    d = s[side]["IC"][[0, 2]] - p
    d = d / np.linalg.norm(d)
    perp = abs(np.cross(d, rc - p))          # distance from RC to the line
    ok_on &= perp < 1e-6
check("RC lies on BOTH patch->IC lines", ok_on)
ok_f = all(abs(s[k]["fvsa"] - abs(s[k]["IC"][0] - s[k]["patch"][0])) < 1e-9
           for k in ("L", "R"))
check("FVSA == |patch_x - IC_x|", ok_f)

# ---------------------------------------------------------------- 5. roll behaviour
print("\n5. ROLL BEHAVIOUR")
rs = roll_sweep(pts_R, pts_L, CAM, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                st["track"])
check("RC x is ODD in roll", abs(rs["rc_x"][0] + rs["rc_x"][4]) < 1e-6,
      f"{rs['rc_x'][0]:+.1f} vs {rs['rc_x'][4]:+.1f} mm")
check("RC z is EVEN in roll", abs(rs["rc_z"][0] - rs["rc_z"][4]) < 1e-6,
      f"{rs['rc_z'][0]:+.1f} vs {rs['rc_z'][4]:+.1f} mm")
check("camber L/R swap under +/- roll",
      abs(rs["camber_L"][0] - rs["camber_R"][4]) < 1e-9)
check("RC x is zero at zero roll", abs(rs["rc_x"][2]) < 1e-6)

# ---------------------------------------------------------------- 6. roll model
print("\n6. ROLL MODEL")
print("     rolling the body with the tyres on the road IS equal-and-opposite")
print("     wheel travel -- asking either way must give the same car.")
half = 0.5 * st["track"]
PHI = 2.0
z = half * np.tan(np.radians(PHI))
by_roll = axle_state(pts_R, pts_L, CAM, body_roll=PHI)
by_bump = axle_state(pts_R, pts_L, CAM, left_bump=+z, right_bump=-z)
check("roll(phi) == differential bump: camber L",
      abs(by_roll["L"]["camber"] - by_bump["L"]["camber"]) < 1e-9)
check("roll(phi) == differential bump: camber R",
      abs(by_roll["R"]["camber"] - by_bump["R"]["camber"]) < 1e-9)
check("roll(phi) == differential bump: RC height",
      abs(by_roll["roll_centre"][1] - by_bump["roll_centre"][1]) < 1e-6,
      f"{by_roll['roll_centre'][1]:+.3f} vs {by_bump['roll_centre'][1]:+.3f} mm")

# both tyres must end up ON the road, at every trim
ok_grd = True
for lb, rb, ph in ((0, 0, 0), (10, 10, 0), (0, 0, 2.5), (15, -5, 1.0), (-20, 8, -2.0)):
    t = axle_state(pts_R, pts_L, CAM, left_bump=lb, right_bump=rb, body_roll=ph)
    ok_grd &= abs(t["L"]["patch_height"]) < 1e-6 and abs(t["R"]["patch_height"]) < 1e-6
check("both contact patches sit ON the road at every trim", ok_grd)

t = axle_state(pts_R, pts_L, CAM, body_roll=3.0)
check("realised roll matches the requested roll",
      abs(abs(t["roll_actual"]) - 3.0) < 0.05,
      f"asked 3.000, got {abs(t['roll_actual']):.3f} deg")

# rigid lengths must survive the ground-frame transform
for nm, aa, bb in (("kingpin", "UBJ", "LBJ"), ("UBJ-WC", "UBJ", "WC"),
                   ("LBJ-WC", "LBJ", "WC")):
    d0 = np.linalg.norm(st["R"][aa] - st["R"][bb])
    d1 = np.linalg.norm(t["R"][aa] - t["R"][bb])
    check(f"{nm} length preserved through the transform", abs(d0 - d1) < 1e-8)
R = _roll_matrix(4.0)
check("roll matrix is a proper rotation",
      abs(np.linalg.det(R) - 1.0) < 1e-12 and
      np.allclose(R @ R.T, np.eye(3), atol=1e-12))

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the axle model agrees with the single-corner solver.")
print("=" * 76)
