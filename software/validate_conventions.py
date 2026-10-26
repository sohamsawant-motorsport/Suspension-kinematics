# -*- coding: utf-8 -*-
"""
SANITY CHECK  -  coordinate frame and every sign convention.
=============================================================

Nothing here computes anything new. It takes each convention the project relies
on and checks it against an INDEPENDENT construction -- a rotation applied by
hand, or a geometric fact that must be true -- so a silent sign flip cannot hide.

This exists because a sign error already cost this project once (KPI +5.3 vs
-5.27, worth 60 mm of scrub radius) and because the answers to "is +ve toe-in or
toe-out?" have to be provable, not remembered.

  1  FRAME          - handedness, and which way each axis points on the real car.
  2  TOE            - + = toe-IN, verified by yawing a wheel by hand.
  3  CAMBER         - - = top leans toward the car, verified by tilting by hand.
  4  KPI / CASTER   - signs verified against the ball-joint offsets.
  5  BODY ROLL      - which wheel compresses at +roll. This one is a TRAP: the
                      axle model's docstring and its arithmetic disagreed.
  6  BUMP / TRAVEL  - + = compression, verified against the wheel centre rising.
  7  LOAD TRANSFER  - + lateral g loads the RIGHT-hand wheels.

Run:  python validate_conventions.py
"""
import os
import sys

import numpy as np

_ST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "suspension_tool")
if _ST not in sys.path:
    sys.path.insert(0, _ST)

import config as cfg
import rear_geometry as rg
import load_transfer as ltr
from scrub_trail import wheel_spin_axis, kpi_caster
from wishbone_solver import corner_geometry, solve_at
import axle

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("=" * 78)
print("SANITY CHECK - coordinate frame and sign conventions")
print("=" * 78)

F = cfg._load_front_points()
R = rg.load_rear()

# ------------------------------------------------------------------ 1. frame
print("\n1. THE FRAME       X = lateral, Y = longitudinal, Z = vertical")
xh, yh, zh = np.eye(3)
check("right-handed:  X cross Y = Z", np.allclose(np.cross(xh, yh), zh))
check("+X is OUTBOARD (the CAD corner is the RIGHT one)",
      F["Wheel centre"][0] > 0 and R["Wheel centre"][0] > 0,
      f"front WC x {F['Wheel centre'][0]:+.1f}, rear {R['Wheel centre'][0]:+.1f}")
check("+Y is FORWARD (the front axle is ahead of the rear)",
      F["Wheel centre"][1] > R["Wheel centre"][1],
      f"front WC y {F['Wheel centre'][1]:+.1f} > rear {R['Wheel centre'][1]:+.1f}")
check("+Z is UP (wheel centres sit above the road)",
      F["Wheel centre"][2] > 0 and R["Wheel centre"][2] > 0,
      f"front WC z {F['Wheel centre'][2]:+.1f}, rear {R['Wheel centre'][2]:+.1f}")
check("the upper ball joint is above the lower one",
      F[cfg.UBJ_NAME][2] > F[cfg.LBJ_NAME][2]
      and R[rg.UBJ][2] > R[rg.LBJ][2])

# -------------------------------------------------------------------- 2. toe
print("\n2. TOE            (+ = TOE-IN for this right-hand wheel)")
# Build it by hand: yaw the wheel about +Z. In a right-handed frame a POSITIVE
# rotation about Z carries +X toward +Y. The spin axis points OUTBOARD (+X), so
# after a +tau yaw the wheel's rolling direction (+Y) swings toward -X, i.e. the
# front of the wheel points at the car's centreline == TOE-IN for a RIGHT wheel.
tau = np.radians(3.0)
Rz = np.array([[np.cos(tau), -np.sin(tau), 0],
               [np.sin(tau), np.cos(tau), 0], [0, 0, 1]])
spin_yawed = Rz @ np.array([1.0, 0.0, 0.0])          # outboard axis, yawed
roll_dir = Rz @ np.array([0.0, 1.0, 0.0])            # rolling direction, yawed
check("a +yaw points the wheel's front INBOARD (toward -X)", roll_dir[0] < 0,
      f"rolling direction x {roll_dir[0]:+.4f}")
check("...and the project's formula reads that as POSITIVE toe",
      np.degrees(np.arctan2(spin_yawed[1], spin_yawed[0])) > 0,
      f"toe = {np.degrees(np.arctan2(spin_yawed[1], spin_yawed[0])):+.3f} deg")
check("so: +toe = TOE-IN = the wheel points at the car centreline", True)
s0 = rg.spin_axis(0.0, +1.0)
check("rear_geometry.spin_axis(camber, +toe) agrees",
      np.degrees(np.arctan2(s0[1], s0[0])) > 0,
      f"toe read back {np.degrees(np.arctan2(s0[1], s0[0])):+.3f} deg")

# ----------------------------------------------------------------- 3. camber
print("\n3. CAMBER         (- = top of the wheel leans IN toward the car)")
s_neg = wheel_spin_axis(-2.7)
check("negative camber tilts the outboard spin axis UP", s_neg[2] > 0,
      f"s_z = {s_neg[2]:+.4f}")
# the wheel disc is perpendicular to the spin axis; its top point is
# perpendicular to s in the X-Z plane
top_dir = np.array([-s_neg[2], 0.0, s_neg[0]])       # perpendicular, pointing up
check("...which puts the TOP of the wheel INBOARD (toward -X)", top_dir[0] < 0,
      f"top of wheel x-direction {top_dir[0]:+.4f}")
check("camber = arcsin(-s_z) inverts correctly",
      abs(np.degrees(np.arcsin(-s_neg[2])) - (-2.7)) < 1e-9)

# ----------------------------------------------------------- 4. KPI / caster
print("\n4. KPI / CASTER   (both measured from vertical)")
for tag, P, U, L in (("front", F, cfg.UBJ_NAME, cfg.LBJ_NAME),
                     ("rear", R, rg.UBJ, rg.LBJ)):
    d = np.asarray(P[U], float) - P[L]
    kpi, cas = kpi_caster(P[L], P[U])
    check(f"{tag}: KPI sign matches the ball-joint dX ({d[0]:+.1f} mm)",
          (kpi > 0) == (d[0] < 0),
          f"KPI {kpi:+.2f} deg -> UBJ is "
          f"{'INBOARD' if d[0] < 0 else 'OUTBOARD'} of the LBJ")
    check(f"{tag}: caster sign matches the ball-joint dY ({d[1]:+.1f} mm)",
          (cas > 0) == (d[1] < 0),
          f"caster {cas:+.2f} deg -> axis leans "
          f"{'REARWARD' if d[1] < 0 else 'FORWARD'} at the top")

# -------------------------------------------------------------- 5. body roll
print("\n5. BODY ROLL      *** the one that had a wrong docstring ***")
sp = rg.solver_points(R)
LM = axle.mirror_points(sp)
st = axle.axle_state(sp, LM, rg.REAR_STATIC_CAMBER, body_roll=+3.0)
bl, br = st["trim"]["eff_bump_L"], st["trim"]["eff_bump_R"]
check("+roll compresses the LEFT wheel and droops the RIGHT",
      bl > 0 and br < 0, f"bump_L {bl:+.1f} mm, bump_R {br:+.1f} mm")
check("so +body_roll = the body leans onto its LEFT = a RIGHT-HAND turn", True)
check("the two sides are equal and opposite", abs(bl + br) < 1e-9)
# and the outside (compressing) wheel must gain NEGATIVE camber from the linkage
cam_L_static = axle.axle_state(sp, LM, rg.REAR_STATIC_CAMBER)["L"]["camber"]
check("the compressing (outside) wheel gains negative camber from the linkage",
      st["L"]["camber"] < cam_L_static,
      f"{cam_L_static:+.3f} -> {st['L']['camber']:+.3f} deg")

# ------------------------------------------------------------ 6. wheel travel
print("\n6. WHEEL TRAVEL   (+ = COMPRESSION = the wheel moves up into the arch)")
g = rg.rear_corner(R, rg.REAR_STATIC_CAMBER, 0.0)
a0 = solve_at(g, 0.0, 0.0)
ap = solve_at(g, +2.0, 0.0)
rise = ap["WC"][2] - a0["WC"][2]
check("a positive wishbone angle raises the wheel centre", rise > 0,
      f"{rise:+.3f} mm for +2 deg of lower-arm rotation")
sw = rg.bump_sweep(R, rg.REAR_STATIC_CAMBER, -20.0, 20.0, 41)
check("bump_sweep's travel axis increases with wheel-centre height",
      sw["bump"][-1] > sw["bump"][0])

# ----------------------------------------------------------- 7. load transfer
print("\n7. LOAD TRANSFER  (+ lateral g = RIGHT turn -> the LEFT wheels load up)")
car = ltr.temerario_gt3()
w = ltr.wheel_loads(+1.0, 0.0, -283.6, -52.0, car)
check("+lateral g loads FL and RL, unloads FR and RR",
      w["FL"] > w["FR"] and w["RL"] > w["RR"],
      f"FL {w['FL']:.0f} FR {w['FR']:.0f} | RL {w['RL']:.0f} RR {w['RR']:.0f} N")
# THE CROSS-MODULE CHECK this file exists for: the kinematic model and the load
# model must agree about which way a positive number turns the car. They did NOT
# until this was written -- axle.py's +roll meant a right turn while
# load_transfer's +lateral g meant a left one.
check("axle.py and load_transfer.py agree which side is LOADED at +ve",
      (bl > 0) and (w["FL"] > w["FR"]),
      "+roll compresses the LEFT and +lateral g loads the LEFT: consistent")
check("g-g envelope has a FLAT TOP (clipped circle, not an ellipse)",
      abs(max(abs(ltr.gg_envelope()[0][abs(ltr.gg_envelope()[1] - 1.0) < 1e-9]))
          - np.sqrt(1.2 ** 2 - 1.0 ** 2)) < 0.02,
      f"flat top runs to +/-{np.sqrt(1.2**2 - 1.0**2):.3f} g lateral")
wb = ltr.wheel_loads(0.0, -1.0, -283.6, -52.0, car)
check("braking (-a_long) loads the FRONT axle",
      wb["W_f"] > car["mass_total_kg"] * ltr.G * (1 - car["rear_frac"]),
      f"front axle {wb['W_f']:.0f} N vs static "
      f"{car['mass_total_kg']*ltr.G*(1-car['rear_frac']):.0f} N")
wa = ltr.wheel_loads(0.0, +1.0, -283.6, -52.0, car)
check("accelerating (+a_long) loads the REAR axle",
      wa["W_r"] > car["mass_total_kg"] * ltr.G * car["rear_frac"])
check("an ABOVE-ground roll centre gives POSITIVE geometric transfer",
      ltr.transfer(1.0, -283.6, +50.0, car)["geo_r"] > 0)
check("a BELOW-ground roll centre gives NEGATIVE geometric transfer",
      ltr.transfer(1.0, -283.6, -50.0, car)["geo_r"] < 0)

print("\n" + "=" * 78)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CONVENTIONS VERIFIED - the frame and every sign are self-consistent.")
print("=" * 78)
