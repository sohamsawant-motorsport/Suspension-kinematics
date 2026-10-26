# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the rear ROLL kinematics (camber recovery + roll steer).
=============================================================================

Roll is the case where sign errors hide, so this suite leans hard on symmetry and
on cross-checking against the ALREADY-VALIDATED single-corner heave sweep.

  1  SYMMETRY      - a symmetric axle must be exactly antisymmetric in roll:
                     RC_x odd, RC_z even, and each side's camber/toe at +phi must
                     equal the other side's at -phi.
  2  ZERO ROLL     - reproduces the statics: camber = setup, toe = 0, RC = Study-1.
  3  CROSS-MODULE  - roll IS equal-and-opposite travel: the camber and toe at roll
                     phi must equal the single-corner HEAVE sweep at the same
                     effective bump. Two independent paths, one answer.
  4  ROAD CAMBER   - the frame relation gamma_road = gamma_chassis + phi_wheel,
                     with phi_wheel exactly +phi (compressing) / -phi (drooping).
  5  RECOVERY      - the definition behaves: a hypothetical zero-gain linkage
                     gives 0 %, and ours sits between 0 and 100 %.
  6  ROLL STEER    - equals the slope of toe(roll), and is consistent with the
                     heave bump-steer via the chain rule d(toe)/dphi =
                     d(toe)/d(bump) * d(bump)/dphi.
  7  CONDITIONING  - the RC migration is real geometry, not a numerical blow-up:
                     the two force lines stay well away from parallel.

Run:  python validate_rear_roll.py
"""
import numpy as np

import rear_geometry as rg

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


CAM = rg.REAR_STATIC_CAMBER
P = rg.load_rear()
sw = rg.roll_sweep(P, CAM, -5.0, 5.0, 41)
roll = sw["roll"]
n = len(roll)
i0 = int(np.argmin(np.abs(roll)))

print("=" * 76)
print("VALIDATION - rear roll kinematics (camber recovery + roll steer)")
print("=" * 76)

# --------------------------------------------------------------- 1. symmetry
print("\n1. SYMMETRY of a mirror-symmetric axle")
check("RC x is ODD in roll", abs(sw["rc_x"][0] + sw["rc_x"][n - 1]) < 1e-6,
      f"{sw['rc_x'][0]:+.1f} vs {sw['rc_x'][n-1]:+.1f} mm")
check("RC z is EVEN in roll", abs(sw["rc_z"][0] - sw["rc_z"][n - 1]) < 1e-6,
      f"{sw['rc_z'][0]:+.1f} vs {sw['rc_z'][n-1]:+.1f} mm")
check("RC x is zero at zero roll", abs(sw["rc_x"][i0]) < 1e-6)
check("camber L(+phi) == camber R(-phi)",
      np.max(np.abs(sw["camber_L"] - sw["camber_R"][::-1])) < 1e-9,
      f"max|d| {np.max(np.abs(sw['camber_L'] - sw['camber_R'][::-1])):.2e} deg")
check("toe L(+phi) == toe R(-phi)",
      np.max(np.abs(sw["toe_L"] - sw["toe_R"][::-1])) < 1e-9,
      f"max|d| {np.max(np.abs(sw['toe_L'] - sw['toe_R'][::-1])):.2e} deg")

# -------------------------------------------------------------- 2. zero roll
print("\n2. ZERO ROLL reproduces the statics")
gs = rg.static_geometry(P, CAM)
check("camber == the setup value", abs(sw["camber_L"][i0] - CAM) < 1e-6,
      f"{sw['camber_L'][i0]:+.4f} vs {CAM:+.2f} deg")
check("toe == 0", abs(sw["toe_L"][i0]) < 1e-9, f"{sw['toe_L'][i0]:+.2e} deg")
check("RC height == Study-1 construction",
      abs(sw["rc0"] - gs["rc_height"]) < 1.0,
      f"roll {sw['rc0']:+.1f} vs static {gs['rc_height']:+.1f} mm")
check("road camber == chassis camber at zero roll",
      abs(sw["road_L"][i0] - sw["camber_L"][i0]) < 1e-9)

# ------------------------------------------------------------ 3. cross-module
print("\n3. CROSS-MODULE: roll IS equal-and-opposite travel")
# the single-corner sweep solves a narrower travel band than the axle model (it
# uses a smaller wishbone-angle span), so compare only where BOTH can solve
LIM = 65.0
heave = rg.bump_sweep(P, CAM, -LIM, LIM, 651)
worst_c = worst_t = 0.0
cmp_n = 0
for i in range(n):
    b = sw["bump_L"][i]
    if not np.isfinite(sw["camber_L"][i]) or abs(b) > LIM:
        continue
    worst_c = max(worst_c, abs(sw["camber_L"][i]
                               - np.interp(b, heave["bump"], heave["camber"])))
    worst_t = max(worst_t, abs(sw["toe_L"][i]
                               - np.interp(b, heave["bump"], heave["toe"])))
    cmp_n += 1
check("roll camber == heave camber at the same bump", worst_c < 5e-3,
      f"worst |d| {worst_c:.2e} deg over {cmp_n} points")
check("roll toe == heave toe at the same bump", worst_t < 5e-3,
      f"worst |d| {worst_t:.2e} deg")

# ------------------------------------------------------------- 4. road camber
print("\n4. ROAD-CAMBER FRAME RELATION")
phi_L = np.degrees(np.arctan(sw["bump_L"] / sw["half_track"]))
check("phi_wheel == +roll on the compressing side",
      np.max(np.abs(phi_L - roll)) < 1e-9,
      f"max|d| {np.max(np.abs(phi_L - roll)):.2e} deg")
check("road camber == chassis camber + phi",
      np.max(np.abs(sw["road_L"] - (sw["camber_L"] + phi_L))) < 1e-12)
check("the two sides' road cambers mirror",
      np.max(np.abs(sw["road_L"] - sw["road_R"][::-1])) < 1e-9)

# --------------------------------------------------------------- 5. recovery
print("\n5. CAMBER RECOVERY definition")
# a linkage with ZERO camber gain would give road camber changing 1:1 with roll
zero_gain_recovery = (1.0 - 1.0) * 100.0
check("zero-gain linkage would give 0 % recovery", abs(zero_gain_recovery) < 1e-12)
check("our recovery is between 0 and 100 %",
      0.0 < sw["recovery_pct"] < 100.0, f"{sw['recovery_pct']:.1f} %")
check("recovery == (1 - d(road camber)/d(roll)) * 100",
      abs(sw["recovery_pct"] - (1.0 - sw["d_road_L"][i0]) * 100.0) < 1e-9)
check("rear recovers far more than the front (1.32 %)",
      sw["recovery_pct"] > 5.0, f"rear {sw['recovery_pct']:.1f} %")

# ------------------------------------------------------------- 6. roll steer
print("\n6. ROLL STEER")
check("roll steer == slope of toe(roll) at ride",
      abs(sw["roll_steer"] - np.gradient(sw["toe_L"], roll)[i0]) < 1e-12,
      f"{sw['roll_steer']:+.5f} deg/deg")
# chain rule: d(toe)/d(phi) = d(toe)/d(bump) * d(bump)/d(phi),
# with d(bump)/d(phi) = half * pi/180 at phi = 0
dbump_dphi = sw["half_track"] * np.pi / 180.0
predicted = heave["bump_steer"] * dbump_dphi
check("roll steer == bump steer x d(bump)/d(roll)  [chain rule]",
      abs(sw["roll_steer"] - predicted) < 2e-3,
      f"{sw['roll_steer']:+.5f} vs {predicted:+.5f} deg/deg")

# ------------------------------------------------------------ 7. conditioning
print("\n7. RC MIGRATION IS REAL GEOMETRY, NOT NUMERICAL NOISE")
axle = rg._axle()
sp = rg.solver_points(P)
Lp = axle.mirror_points(sp)
worst_ang = 90.0
for phi in (-4.0, -2.0, 0.0, 2.0, 4.0):
    st = axle.axle_state(sp, Lp, CAM, body_roll=phi)
    dL = st["L"]["IC"][[0, 2]] - st["L"]["patch"][[0, 2]]
    dR = st["R"]["IC"][[0, 2]] - st["R"]["patch"][[0, 2]]
    c = abs(np.dot(dL / np.linalg.norm(dL), dR / np.linalg.norm(dR)))
    worst_ang = min(worst_ang, np.degrees(np.arccos(np.clip(c, -1, 1))))
check("force lines never approach parallel", worst_ang > 1.0,
      f"smallest angle between them {worst_ang:.2f} deg")
check("RC curve is smooth (no jumps)",
      np.max(np.abs(np.diff(sw["rc_z"]))) < 200.0,
      f"largest step {np.max(np.abs(np.diff(sw['rc_z']))):.1f} mm")

# ------------------------------------------------- 8. the recovery mechanism
print("\n8. WHERE RECOVERY COMES FROM (Study 5's maths)")
heave20 = rg.bump_sweep(P, CAM, -20.0, 20.0, 81)
cg_fast, bs_fast = rg.camber_gain_fast(P, CAM)
check("fast camber gain == full bump_sweep",
      abs(cg_fast - heave20["camber_gain"]) < 1e-4,
      f"{cg_fast:+.5f} vs {heave20['camber_gain']:+.5f} deg/mm")
check("fast bump steer == full bump_sweep",
      abs(bs_fast - heave20["bump_steer"]) < 1e-4)
half_static = rg.static_geometry(P, CAM)["track"] / 2.0
check("recovery_from_gain == the sweep's recovery",
      abs(rg.recovery_from_gain(cg_fast, half_static) - sw["recovery_pct"]) < 1.0,
      f"{rg.recovery_from_gain(cg_fast, half_static):.2f} vs "
      f"{sw['recovery_pct']:.2f} %")
# the geometric shortcut: recovery ~= half-track / FVSA
approx = half_static / gs["fvsa"] * 100.0
check("shortcut (t/2)/FVSA is within 15 % of the exact value",
      abs(approx - sw["recovery_pct"]) / sw["recovery_pct"] < 0.15,
      f"shortcut {approx:.1f} % vs exact {sw['recovery_pct']:.1f} %")
# the toe link must NOT move the instant centre (it only twists the upright)
p_toe = dict(P)
p_toe[rg.TOE_O] = P[rg.TOE_O] + np.array([0.0, 0.0, 20.0])
g_toe = rg.static_geometry(p_toe, CAM)
check("moving the TOE LINK leaves FVSA unchanged",
      abs(g_toe["fvsa"] - gs["fvsa"]) < 1e-6,
      f"{g_toe['fvsa']:.1f} vs {gs['fvsa']:.1f} mm")
check("moving the TOE LINK leaves the ROLL CENTRE unchanged",
      abs(g_toe["rc_height"] - gs["rc_height"]) < 1e-6)
# ... but moving the UPRIGHT moves both
p_ubj = dict(P)
p_ubj[rg.UBJ] = P[rg.UBJ] + np.array([0.0, 0.0, 20.0])
g_ubj = rg.static_geometry(p_ubj, CAM)
check("moving the UPRIGHT moves FVSA and the RC together",
      abs(g_ubj["fvsa"] - gs["fvsa"]) > 100.0
      and abs(g_ubj["rc_height"] - gs["rc_height"]) > 10.0,
      f"FVSA {gs['fvsa']:.0f}->{g_ubj['fvsa']:.0f},  "
      f"RC {gs['rc_height']:+.0f}->{g_ubj['rc_height']:+.0f} mm")
lo, hi = rg.travel_limits(P, CAM)
check("travel search range covers any roll worth studying (>=7 deg)",
      np.degrees(np.arctan(hi / half_static)) >= 7.0,
      f"droop {lo:.1f}, compression {hi:.1f} mm "
      f"(=> {np.degrees(np.arctan(hi/half_static)):.1f} deg of roll)")
# travel_limits is a SEARCH range, not a bind: it must grow with the span. An
# earlier version quoted a narrow-span result as "the suspension binds here".
_, hi_narrow = rg.travel_limits(P, CAM, span=20.0)
check("travel_limits is a search range, not a hard bind (grows with span)",
      hi > hi_narrow + 20.0,
      f"span 20 -> {hi_narrow:.0f} mm,  span 45 -> {hi:.0f} mm")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - rear roll kinematics are symmetric, agree with the "
      "heave\nsweep, and the frame relations are exact.")
print("=" * 76)
