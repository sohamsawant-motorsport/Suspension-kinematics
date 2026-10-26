# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the wishbone solver.
=========================================

Same philosophy as validate_scrub_trail.py: every check is an assertion against
the real front-right CAD hard points, and the strongest checks are CROSS-MODULE
-- the full solver must reproduce the earlier fixed-axis modules exactly, because
steering at fixed bump IS a rotation about the LBJ-UBJ line.

  1  Reference reproduction - at zero bump / zero rack the solver must return
                              the input geometry to machine precision.
  2  Constraints hold       - across the whole sweep: ball joints stay on their
                              circles, the tie rod stays at its length.
  3  Upright is rigid       - all six internal distances constant.
  4  CROSS-MODULE (rack)    - sweeping the rack at zero bump must reproduce
                              scrub_trail.corner_state (camber, scrub, trail)
                              at the solver's own kingpin-twist angle.
  5  Effective steer ratio  - the velocity-relation closed form
                                 d(delta)/d(rack) = (x.t) / ((k x r).t)
                              against a finite difference of the solver.
  6  Motion ratio           - solver MR at zero bump vs the first-order
                              lower-wishbone estimate (must be close, not equal).
  7  Roll centre            - numerical instant-centre construction vs the
                              classical 2D front-view construction (the hinge
                              axes are pure-Y here, so the 2D one is near-exact).

Run:  python validate_wishbone.py
"""
import numpy as np

import config as cfg
from scrub_trail import corner_state, scrub_radius, mechanical_trail
from jacking import motion_ratio
from wishbone_solver import (corner_geometry, solve_at, solve_sweep,
                             roll_centre_curve)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


p = cfg._load_front_points()
g = corner_geometry(p, cfg.STATIC_CAMBER)
LBJ0, UBJ0, WC0 = g["LBJ"], g["UBJ"], g["WC"]

print("=" * 76)
print("VALIDATION - wishbone solver, against the FR CAD hard points")
print("=" * 76)

# ---------------------------------------------------------------- 1. reference
print("\n1. REFERENCE REPRODUCTION (zero bump, zero rack)")
st = solve_at(g, 0.0)
for nm, a, b in [("LBJ", st["LBJ"], LBJ0), ("UBJ", st["UBJ"], UBJ0),
                 ("TRO", st["TRO"], g["TRO"]), ("WC", st["WC"], WC0)]:
    check(f"{nm} returned unchanged", np.linalg.norm(a - b) < 1e-9,
          f"drift {np.linalg.norm(a-b):.2e} mm")
cam0 = np.degrees(np.arcsin(-st["s"][2]))
toe0 = np.degrees(np.arctan2(st["s"][1], st["s"][0]))
check("camber == static camber", abs(cam0 - cfg.STATIC_CAMBER) < 1e-9,
      f"{cam0:+.6f} vs {cfg.STATIC_CAMBER:+.1f}")
check("toe == 0 at reference", abs(toe0) < 1e-9, f"{toe0:+.2e} deg")

# ---------------------------------------------------------------- 2+3. sweep integrity
print("\n2+3. CONSTRAINTS AND RIGIDITY ACROSS THE SWEEP (theta_L -12..+12 deg)")
names = [("UBJ-LBJ", "UBJ", "LBJ"), ("TRO-UBJ", "TRO", "UBJ"),
         ("TRO-LBJ", "TRO", "LBJ"), ("WC-UBJ", "WC", "UBJ"),
         ("WC-LBJ", "WC", "LBJ"), ("WC-TRO", "WC", "TRO")]
ref = {n: np.linalg.norm(
    {"UBJ": UBJ0, "LBJ": LBJ0, "TRO": g["TRO"], "WC": WC0}[a]
    - {"UBJ": UBJ0, "LBJ": LBJ0, "TRO": g["TRO"], "WC": WC0}[b]) for n, a, b in names}
worst_rig, worst_con = 0.0, 0.0
thU = thT = 0.0
for th in np.linspace(-12, 12, 97):
    s = solve_at(g, th, 0.0, thU, thT)
    thU, thT = s["thU"], s["thT"]
    for n, a, b in names:
        worst_rig = max(worst_rig, abs(np.linalg.norm(s[a] - s[b]) - ref[n]))
    for P, Q in [(s["UBJ"], g["UIF"]), (s["UBJ"], g["UIR"]),
                 (s["LBJ"], g["LIF"]), (s["LBJ"], g["LIR"])]:
        pass
    worst_con = max(worst_con,
                    abs(np.linalg.norm(s["UBJ"] - g["UIF"]) - np.linalg.norm(UBJ0 - g["UIF"])),
                    abs(np.linalg.norm(s["UBJ"] - g["UIR"]) - np.linalg.norm(UBJ0 - g["UIR"])),
                    abs(np.linalg.norm(s["LBJ"] - g["LIF"]) - np.linalg.norm(LBJ0 - g["LIF"])),
                    abs(np.linalg.norm(s["LBJ"] - g["LIR"]) - np.linalg.norm(LBJ0 - g["LIR"])),
                    abs(np.linalg.norm(s["TRO"] - s["TRI"]) - g["len_tierod"]))
check("upright rigid (6 internal distances)", worst_rig < 1e-8, f"worst {worst_rig:.2e} mm")
check("wishbone circles + tie-rod length held", worst_con < 1e-8, f"worst {worst_con:.2e} mm")

# ---------------------------------------------------------------- 4. cross-module
print("\n4. CROSS-MODULE - rack sweep at zero bump vs scrub_trail.corner_state")
print("     steering at fixed bump rotates the upright EXACTLY about the LBJ-UBJ")
print("     line, which is scrub_trail's model. Same physics, different code path.")
worst_cam, worst_sc, worst_tr = 0.0, 0.0, 0.0
for rack in (-15.0, -5.0, 5.0, 15.0, 25.0):
    s = solve_at(g, 0.0, rack_mm=rack)
    delta = s["thT"]                       # the upright's twist about the kingpin
    ref_st = corner_state(LBJ0, UBJ0, WC0, np.array([delta]), cfg.STATIC_CAMBER)
    cam_solver = np.degrees(np.arcsin(-s["s"][2]))
    worst_cam = max(worst_cam, abs(cam_solver - ref_st["camber"][0]))
    # patch from the solver's own wheel state
    from scrub_trail import contact_patch, ground_point
    P = contact_patch(s["WC"], spin_axis=s["s"], R=g["R_wheel"])
    G = ground_point(LBJ0, UBJ0)
    worst_sc = max(worst_sc, abs((P[0] - G[0]) - ref_st["scrub"][0]))
    worst_tr = max(worst_tr, abs((G[1] - P[1]) - ref_st["trail"][0]))
check("camber matches corner_state at the solver's twist", worst_cam < 1e-9,
      f"worst {worst_cam:.2e} deg")
check("scrub matches", worst_sc < 1e-8, f"worst {worst_sc:.2e} mm")
check("trail matches", worst_tr < 1e-8, f"worst {worst_tr:.2e} mm")

# ---------------------------------------------------------------- 5. steer ratio
print("\n5. EFFECTIVE STEERING RATIO - velocity relation vs finite difference")
t_hat = (g["TRO"] - g["TRI"]) / np.linalg.norm(g["TRO"] - g["TRI"])
k_hat = (UBJ0 - LBJ0) / np.linalg.norm(UBJ0 - LBJ0)
w = g["TRO"] - LBJ0
ratio_closed = np.dot(np.array([1.0, 0, 0]), t_hat) / np.dot(np.cross(k_hat, w), t_hat)
h = 0.01
sp = solve_at(g, 0.0, rack_mm=+h)
sm = solve_at(g, 0.0, rack_mm=-h)
ratio_fd = np.radians(sp["thT"] - sm["thT"]) / (2 * h)
check("d(delta)/d(rack): closed form vs solver", abs(ratio_closed - ratio_fd) < 1e-5,
      f"{ratio_closed:+.6f} vs {ratio_fd:+.6f} rad/mm")
eff_arm = abs(1.0 / ratio_fd)
print(f"     effective steering arm = {eff_arm:.1f} mm  "
      f"(the perp-distance estimate was {cfg.steering_arm_mm():.1f} mm -- the tie")
print(f"     rod is OBLIQUE to the steering arm, so the true ratio differs; "
      f"20 mm rack = {abs(np.degrees(20*ratio_fd)):.2f} deg first-order)")

# ---------------------------------------------------------------- 6. motion ratio
print("\n6. MOTION RATIO - exact (solver) vs first-order (lower wishbone only)")
bump = np.linspace(-30, 30, 121)
sw = solve_sweep(g, bump)
i0 = int(np.argmin(np.abs(bump)))
mr_solver = sw["MR"][i0]
mr_first = motion_ratio(g["LIF"], g["LIR"], LBJ0, g["SL"], g["SU"])
check("solver MR within 2% of the first-order estimate",
      abs(mr_solver - mr_first) / mr_first < 0.02,
      f"{mr_solver:.4f} vs {mr_first:.4f}")

# ---------------------------------------------------------------- 7. roll centre
print("\n7. ROLL CENTRE - numerical IC vs the classical 2D construction")
print("     both hinge axes here are pure-Y, so the classic front-view recipe")
print("     (intersect the two projected link lines; patch->IC to x=0) is valid.")
rc = roll_centre_curve(sw, 2 * WC0[0])


def line_intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    s_, _ = np.linalg.solve(A, b)
    return p1 + s_ * d1


up_i = np.array([g["UIF"][0], g["UIF"][2]])
lo_i = np.array([g["LIF"][0], g["LIF"][2]])
up_o = np.array([UBJ0[0], UBJ0[2]])
lo_o = np.array([LBJ0[0], LBJ0[2]])
IC2d = line_intersect(up_i, up_o - up_i, lo_i, lo_o - lo_i)
from scrub_trail import contact_patch as cp_
from wishbone_solver import linkage_roll_centre
P0 = cp_(WC0, cfg.STATIC_CAMBER)
patch2d = np.array([P0[0], 0.0])
d_ = IC2d - patch2d
rc2d = patch2d[1] + d_[1] * (0.0 - patch2d[0]) / d_[0]
rc_link0 = float(linkage_roll_centre(g, np.array([0.0]))[0])
check("LINKAGE roll centre matches the textbook 2D construction",
      abs(rc_link0 - rc2d) < 2.0,
      f"solver {rc_link0:+.1f} vs 2D {rc2d:+.1f} mm")
print(f"     WHEEL-motion RC = {rc['rc_h'][i0]:+.1f} mm vs LINKAGE RC = {rc_link0:+.1f} mm.")
print(f"     [info] They differ by {abs(rc['rc_h'][i0]-rc_link0):.0f} mm because this")
print(f"     geometry's bump steer is enormous (~0.12 deg/mm): the wheel's front-view")
print(f"     rotation is DOMINATED by the steer twist, which the classical")
print(f"     construction quietly assumes is negligible. On a sane geometry the")
print(f"     two converge. Both are reported; know which one you are quoting.")
print(f"     (2D IC at x={IC2d[0]:+.0f}, z={IC2d[1]:+.0f} -- the near-parallel arms")
print(f"      put the IC miles inboard and the RC BELOW ground: road-car/mesh traits.)")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the solver, the earlier modules and the CAD agree.")
print("=" * 76)
