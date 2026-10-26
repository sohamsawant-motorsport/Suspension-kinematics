# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the Ackermann / Jeantaud steering geometry.
================================================================

The whole module rests on one equation, so the checks attack it from angles that
do not share code with the implementation:

  1  JEANTAUD      - ideal_inner() is INVERTED and fed back into the defining
                     condition cot(do) - cot(di) = t/L. If the algebra is wrong
                     the residual will not be ~1e-13.
  2  CALIBRATION   - the percentage formula is fed three constructed inputs whose
                     answers are known by definition: the ideal (100 %), parallel
                     steering (0 %), and the exact reflection of the ideal
                     (-100 %, anti-Ackermann).
  3  ZERO LIMIT    - zero_steer_limit() is fed synthetic data with a KNOWN
                     quadratic ratio; it must recover it. This guards the one
                     piece of numerical trickery in the module.
  4  GROUND TRACK  - the kingpin ground track is cross-checked against the wheel
                     track minus twice the scrub radius, using scrub_trail's
                     INDEPENDENT scrub computation.
  5  SOLVER SWEEP  - straight-ahead is straight, the sweep is symmetric left vs
                     right on a symmetric car, and inner/outer are assigned to
                     the correct physical wheel for each turn direction.
  6  CONSTRUCTION  - the sign and the ">100 %" verdict are confirmed by the
                     drawing-board construction (steering-arm line vs the rear
                     axle), which shares no code with the solver.
  7  BENCHMARK     - guards the headline finding so a regression cannot silently
                     turn the car parallel-steered.

Run:  python validate_ackermann.py
"""
import numpy as np

import ackermann as ak
import config
from explorer import mirror_pts, LBJ_NAME, UBJ_NAME
from scrub_trail import ground_point, corner_state

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


P = config._load_front_points()
CAM = -2.7
WHEELBASE = 2645.0

print("=" * 76)
print("VALIDATION - Ackermann / Jeantaud steering geometry")
print("=" * 76)

# ------------------------------------------------------------ 1. Jeantaud
print("\n1. JEANTAUD CONDITION  cot(do) - cot(di) = t/L")
t_test, L_test = 1400.0, 2645.0
do = np.array([1.0, 3.0, 7.0, 12.0, 20.0, 30.0])
di = ak.ideal_inner(do, t_test, L_test)
resid = (1.0 / np.tan(np.radians(do)) - 1.0 / np.tan(np.radians(di))
         - t_test / L_test)
check("ideal_inner satisfies the defining condition", np.max(np.abs(resid)) < 1e-12,
      f"max residual {np.max(np.abs(resid)):.2e}")
check("ideal inner angle always EXCEEDS the outer", np.all(di > do),
      f"e.g. outer 20.0 -> ideal inner {di[4]:.3f} deg")
check("ideal_inner(0) == 0", abs(float(ak.ideal_inner(0.0, t_test, L_test))) < 1e-12)
check("ideal_inner is sign-antisymmetric",
      np.allclose(ak.ideal_inner(-do, t_test, L_test), -di, atol=1e-12))

# --------------------------------------------------------- 2. calibration
print("\n2. PERCENTAGE CALIBRATION (constructed inputs, answers known by definition)")
pct_ideal = ak.ackermann_percent(di, do, t_test, L_test)
check("feeding the IDEAL returns 100 %", np.allclose(pct_ideal, 100.0, atol=1e-9),
      f"max err {np.max(np.abs(pct_ideal - 100.0)):.2e} %")
pct_par = ak.ackermann_percent(do, do, t_test, L_test)
check("feeding PARALLEL steering returns 0 %", np.allclose(pct_par, 0.0, atol=1e-9))
anti = do - (di - do)                       # reflect the ideal about the outer
pct_anti = ak.ackermann_percent(anti, do, t_test, L_test)
check("feeding ANTI-Ackermann returns -100 %", np.allclose(pct_anti, -100.0, atol=1e-9))
half = do + 0.5 * (di - do)
check("half-way returns 50 %",
      np.allclose(ak.ackermann_percent(half, do, t_test, L_test), 50.0, atol=1e-9))
check("percentage is NaN below the min-outer guard (0/0 region)",
      np.isnan(ak.ackermann_percent(np.array([0.1]), np.array([0.1]),
                                    t_test, L_test))[0])

# ---------------------------------------------------------- 3. zero limit
print("\n3. ZERO-STEER LIMIT (quadratic fit, not a quotient of zeros)")
d_o = np.linspace(0.01, 8.0, 200)
d_i_ideal = ak.ideal_inner(d_o, t_test, L_test)
for want in (100.0, 50.0, 130.0, -100.0):
    synth = d_o + (want / 100.0) * (d_i_ideal - d_o)
    got = ak.zero_steer_limit(synth, d_o, t_test, L_test)
    check(f"recovers a constructed {want:+6.1f} % ratio", abs(got - want) < 0.5,
          f"got {got:+.3f} %")

# -------------------------------------------------------- 4. ground track
print("\n4. KINGPIN GROUND TRACK (cross-checked against scrub_trail)")
t = ak.kingpin_ground_track(P, mirror_pts(P), mirror_pts)
st = corner_state(P[LBJ_NAME], P[UBJ_NAME], P["Wheel centre"], 0.0, CAM)
scrub = float(np.atleast_1d(st["scrub"])[0])
patch_x = float(np.atleast_2d(st["P"])[0, 0])
wheel_track = 2.0 * float(P["Wheel centre"][0])
# NOTE the track here must be the CONTACT-PATCH track, not the wheel-centre one:
# scrub is measured patch -> kingpin, and static camber walks the patch 15.9 mm
# outboard of the wheel centre. Using the wheel track leaves a 31.6 mm error.
check("t = contact-patch track - 2 x scrub radius",
      abs(t - (2 * patch_x - 2 * scrub)) < 1e-6,
      f"t {t:.1f} = 2 x {patch_x:.1f} - 2 x {scrub:.1f}")
check("static camber puts the patch outboard of the wheel centre",
      patch_x > float(P["Wheel centre"][0]),
      f"{patch_x - float(P['Wheel centre'][0]):+.2f} mm at {CAM} deg camber")
check("t is inboard of the wheel track", 0.0 < t < wheel_track,
      f"{t:.1f} < {wheel_track:.1f} mm")
G = ground_point(P[LBJ_NAME], P[UBJ_NAME])
check("t equals twice the kingpin ground X on a symmetric car",
      abs(t - 2.0 * abs(G[0])) < 1e-9, f"2 x {abs(G[0]):.2f}")

# -------------------------------------------------------- 5. solver sweep
print("\n5. SOLVER SWEEP")
s = ak.sweep(P, CAM, mirror_pts(P), CAM, wheelbase_mm=WHEELBASE)
check("the sweep solves", s is not None and len(s["rack"]) > 50,
      f"{len(s['rack'])} rack positions")
i0 = int(np.argmin(np.abs(s["rack"])))
check("straight ahead is straight (both wheels)",
      abs(s["steer_R"][i0]) < 1e-6 and abs(s["steer_L"][i0]) < 1e-6,
      f"R {s['steer_R'][i0]:+.2e}  L {s['steer_L'][i0]:+.2e}")

# a symmetric car steered the other way must mirror exactly
pos = s["rack"] > 0
neg = s["rack"] < 0
pR = np.interp(np.array([10.0, 25.0, 45.0]), s["rack"][pos], s["steer_R"][pos])
nL = np.interp(np.array([10.0, 25.0, 45.0]), -s["rack"][neg][::-1],
               -s["steer_L"][neg][::-1])
check("symmetric car: steer_R(+rack) == -steer_L(-rack)", np.allclose(pR, nL, atol=1e-6),
      f"max diff {np.max(np.abs(pR - nL)):.2e} deg")
ppos = np.interp(np.array([15.0, 30.0, 50.0]), s["rack"][pos], s["pct"][pos])
pneg = np.interp(np.array([15.0, 30.0, 50.0]), -s["rack"][neg][::-1], s["pct"][neg][::-1])
check("%Ackermann is symmetric about zero rack", np.allclose(ppos, pneg, atol=1e-6),
      f"max diff {np.max(np.abs(ppos - pneg)):.2e} %")

# inner/outer must land on the physically correct wheel
j = int(np.argmin(np.abs(s["rack"] - 35.0)))
k = int(np.argmin(np.abs(s["rack"] + 35.0)))
check("RIGHT turn (+steer): the RIGHT wheel is the inner one",
      s["steer_R"][j] > 0 and s["inner"][j] == s["steer_R"][j],
      f"R {s['steer_R'][j]:+.3f} vs L {s['steer_L'][j]:+.3f}")
check("LEFT turn (-steer): the LEFT wheel is the inner one",
      s["steer_L"][k] < 0 and s["inner"][k] == s["steer_L"][k],
      f"L {s['steer_L'][k]:+.3f} vs R {s['steer_R'][k]:+.3f}")
check("|inner| >= |outer| everywhere (this car is pro-Ackermann)",
      np.all(np.abs(s["inner"]) >= np.abs(s["outer"]) - 1e-9))
check("toe_out is >= 0 and grows with steer",
      s["toe_out"][j] > 0 and s["toe_out"][-1] > s["toe_out"][j],
      f"{s['toe_out'][j]:.3f} deg at 35 mm rack")

# REGRESSION GUARD. inner/outer must come from the TURN DIRECTION, not from
# whichever wheel steers more. Moving the tie-rod outer far enough inboard takes
# this car past parallel into genuine anti-Ackermann; the magnitude rule scored
# that as a large POSITIVE percentage (it silently swapped the two wheels), so
# an anti-Ackermann design would have been reported as strongly pro-Ackermann.
anti_pts = {k: np.array(v, float) for k, v in P.items()}
anti_pts["Tie rod - outer"] = anti_pts["Tie rod - outer"] + np.array([-140.0, 0, 0])
s_anti = ak.sweep(anti_pts, CAM, mirror_pts(anti_pts), CAM, wheelbase_mm=WHEELBASE)
check("a past-parallel geometry scores NEGATIVE (anti-Ackermann)",
      s_anti is not None and s_anti["pct_at_zero"] < 0.0,
      f"{s_anti['pct_at_zero']:+.1f} % with the tie-rod outer 140 mm inboard")
ja = int(np.argmin(np.abs(s_anti["rack"] - 35.0)))
check("and its INNER wheel really does steer LESS",
      abs(s_anti["inner"][ja]) < abs(s_anti["outer"][ja]),
      f"inner {s_anti['inner'][ja]:+.3f} vs outer {s_anti['outer'][ja]:+.3f} deg")
check("inner is still the RIGHT wheel in a right turn, even anti-Ackermann",
      s_anti["inner"][ja] == s_anti["steer_R"][ja])

# ------------------------------------------------------- 6. construction
print("\n6. DRAWING-BOARD CONSTRUCTION (shares no code with the solver)")
TRO = P["Tie rod - outer"]
d = np.array([TRO[0] - G[0], TRO[1] - G[1]])
x_rear = G[0] + (-1424.208 - G[1]) / d[1] * d[0]
check("tie rod is AHEAD of the wheel centre (front-steer car)",
      TRO[1] > P["Wheel centre"][1], f"{TRO[1] - P['Wheel centre'][1]:+.1f} mm in Y")
check("steering-arm line crosses the centreline BEFORE the rear axle",
      x_rear < 0.0, f"reaches X = {x_rear:.0f} mm at the rear-axle line")
check("construction agrees with the solver: both say > 100 % Ackermann",
      (x_rear < 0.0) and (s["pct_at_zero"] > 100.0),
      f"solver {s['pct_at_zero']:.1f} %")

# ---------------------------------------------------------- 7. benchmark
print("\n7. HEADLINE FINDING (regression guard)")
check("zero-steer %Ackermann is ~130 %", 125.0 < s["pct_at_zero"] < 135.0,
      f"{s['pct_at_zero']:.2f} %")
check("%Ackermann RISES with steer (gets worse toward lock)",
      s["pct"][-1] > s["pct"][j],
      f"{s['pct'][j]:.1f} % at 35 mm -> {s['pct'][-1]:.1f} % at full rack")
check("this car is nowhere near the GT3 norm of near-parallel steering",
      s["pct_at_zero"] > 50.0,
      f"{s['pct_at_zero']:.1f} % vs a benchmark GT3's single digits")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the Ackermann maths is calibrated against constructed\n"
      "cases and the solver result is corroborated by an independent construction.")
print("=" * 76)
