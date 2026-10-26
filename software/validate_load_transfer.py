# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for load_transfer.py
=====================================

The load-transfer model is the first thing in this project that is not pure
geometry -- it carries mass, spring and CG assumptions -- so it needs asserting
against the physics identities that must hold whatever those assumptions are.

  1  TRIVIAL CASES  - no lateral g, no transfer. RC on the ground, no GEOMETRIC
                      transfer. Both must be exact.
  2  BOOKKEEPING    - the parts must add to the total, and the WHOLE-CAR total
                      must equal the textbook W*a*h_CG/track_equivalent.
  3  SCALING        - every transfer term is linear in lateral g.
  4  ROLL AXIS      - the interpolation must hit the front RC when the CG sits on
                      the front axle and the rear RC when it sits on the rear.
  5  ROLL STIFFNESS - K = (Kw + Karb) * t^2 / 2, and it must add in parallel.
  6  SENSE          - raising the rear RC must raise the rear's geometric share,
                      lower the front LLTD, and stiffen the car in roll.
  7  SANITY         - the Temerario estimates must land in believable territory.

Run:  python validate_load_transfer.py
"""
import numpy as np

import load_transfer as lt

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


CAR = lt.temerario_gt3()
HF, HR = -283.6, -52.0

print("=" * 76)
print("VALIDATION - lateral load transfer")
print("=" * 76)

# ---------------------------------------------------------------- 1. trivial
print("\n1. TRIVIAL CASES")
z = lt.transfer(0.0, HF, HR, CAR)
check("no lateral g -> no transfer at all",
      abs(z["total_f"]) < 1e-12 and abs(z["total_r"]) < 1e-12)
check("no lateral g -> no roll", abs(z["roll_deg"]) < 1e-12)
g0 = lt.transfer(1.0, 0.0, 0.0, CAR)
check("both RCs on the ground -> zero GEOMETRIC transfer",
      abs(g0["geo_f"]) < 1e-12 and abs(g0["geo_r"]) < 1e-12)
check("...but the elastic transfer is still there",
      g0["el_f"] > 0 and g0["el_r"] > 0,
      f"front {g0['el_f']:.0f} N, rear {g0['el_r']:.0f} N")

# ------------------------------------------------------------ 2. bookkeeping
print("\n2. BOOKKEEPING")
r = lt.transfer(1.5, HF, HR, CAR)
check("front parts sum to the front total",
      abs((r["geo_f"] + r["el_f"] + r["uns_f"]) - r["total_f"]) < 1e-9)
check("rear parts sum to the rear total",
      abs((r["geo_r"] + r["el_r"] + r["uns_r"]) - r["total_r"]) < 1e-9)
# WHOLE-CAR IDENTITY. The sum of the axle couples must equal the overturning
# moment of the lateral force about the ground. Note the sprung and unsprung
# masses act at DIFFERENT heights, so it is not simply W_total * a * h_CG -- an
# earlier version of this check made that mistake and was 3 % out.
W_u = (CAR["unsprung_f_kg"] + CAR["unsprung_r_kg"]) * lt.G
W_s = CAR["mass_total_kg"] * lt.G - W_u
moment_model = r["total_f"] * CAR["track_f_mm"] + r["total_r"] * CAR["track_r_mm"]
moment_exact = 1.5 * (W_s * CAR["h_cg_mm"] + W_u * CAR["h_unsprung_mm"])
check("axle couples == sprung + unsprung overturning moment",
      abs(moment_model - moment_exact) / moment_exact < 1e-9,
      f"{moment_model:.5e} vs {moment_exact:.5e} N.mm")

# ---------------------------------------------------------------- 3. scaling
print("\n3. SCALING - everything is linear in lateral g")
r1 = lt.transfer(1.0, HF, HR, CAR)
r2 = lt.transfer(2.0, HF, HR, CAR)
for k in ("geo_f", "geo_r", "el_f", "el_r", "uns_f", "uns_r", "total_f",
          "total_r", "roll_deg"):
    check(f"{k} doubles when g doubles", abs(r2[k] - 2.0 * r1[k]) < 1e-9)
check("front LLTD is INDEPENDENT of g", abs(r2["lltd_f"] - r1["lltd_f"]) < 1e-9,
      f"{r1['lltd_f']:.4f} vs {r2['lltd_f']:.4f} %")

# -------------------------------------------------------------- 4. roll axis
print("\n4. ROLL AXIS")
check("CG on the front axle -> roll axis = front RC",
      abs(lt.roll_axis_height(-50.0, 80.0, 0.0) - (-50.0)) < 1e-12)
check("CG on the rear axle -> roll axis = rear RC",
      abs(lt.roll_axis_height(-50.0, 80.0, 1.0) - 80.0) < 1e-12)
check("CG at mid-wheelbase -> roll axis is the mean",
      abs(lt.roll_axis_height(-50.0, 80.0, 0.5) - 15.0) < 1e-12)
check("equal RCs -> flat roll axis at that height",
      abs(lt.roll_axis_height(40.0, 40.0, 0.56) - 40.0) < 1e-12)

# --------------------------------------------------------- 5. roll stiffness
print("\n5. ROLL STIFFNESS")
t = 1600.0
check("K = (Kw + Karb) * t^2 / 2",
      abs(lt.roll_stiffness(100.0, 50.0, t) - 150.0 * t * t / 2.0) < 1e-9)
check("springs and ARB add in parallel",
      abs(lt.roll_stiffness(100.0, 50.0, t)
          - (lt.roll_stiffness(100.0, 0.0, t) + lt.roll_stiffness(0.0, 50.0, t)))
      < 1e-9)
check("wheel rate from ride frequency inverts",
      abs(lt.wheel_rate_from_frequency(300.0, 3.0)
          - 300.0 * (2 * np.pi * 3.0) ** 2 / 1000.0) < 1e-12)

# ----------------------------------------------------------------- 6. sense
print("\n6. PHYSICAL SENSE - raising the REAR roll centre")
lo = lt.transfer(1.5, HF, -52.0, CAR)
hi = lt.transfer(1.5, HF, +45.0, CAR)
check("raises the rear's GEOMETRIC share",
      hi["geo_frac_r"] > lo["geo_frac_r"],
      f"{lo['geo_frac_r']:+.1f} % -> {hi['geo_frac_r']:+.1f} %")
check("moves transfer to the REAR (front LLTD falls)",
      hi["lltd_f"] < lo["lltd_f"],
      f"{lo['lltd_f']:.2f} % -> {hi['lltd_f']:.2f} %")
check("stiffens the car in roll (roll angle falls)",
      hi["roll_deg"] < lo["roll_deg"],
      f"{lo['roll_deg']/1.5:.3f} -> {hi['roll_deg']/1.5:.3f} deg/g")
check("lifts the roll axis", hi["roll_axis"] > lo["roll_axis"],
      f"{lo['roll_axis']:+.1f} -> {hi['roll_axis']:+.1f} mm")
check("a BELOW-ground RC gives NEGATIVE geometric transfer",
      lo["geo_r"] < 0, f"{lo['geo_r']:+.0f} N at RC -52 mm")
# and the front should react too, because the roll axis moved
check("the FRONT's elastic transfer also changes (whole-car coupling)",
      abs(hi["el_f"] - lo["el_f"]) > 1.0,
      f"{lo['el_f']:.0f} -> {hi['el_f']:.0f} N")

# ---------------------------------------------------------------- 7. sanity
print("\n7. SANITY of the Temerario estimates")
check("front roll-stiffness share is GT-plausible (50-70 %)",
      50.0 < CAR["roll_stiff_front_pct"] < 70.0,
      f"{CAR['roll_stiff_front_pct']:.1f} %")
check("roll gradient is GT-plausible (0.2-1.0 deg/g)",
      0.2 < r1["roll_deg"] < 1.0, f"{r1['roll_deg']:.3f} deg/g")
check("front LLTD is plausible (40-70 %)", 40.0 < r1["lltd_f"] < 70.0,
      f"{r1['lltd_f']:.2f} %")
check("wheel rates are plausible (50-250 N/mm)",
      50 < CAR["kw_f"] < 250 and 50 < CAR["kw_r"] < 250,
      f"F {CAR['kw_f']:.0f}, R {CAR['kw_r']:.0f} N/mm")

# ------------------------------------------- 8. combined g-g / four-wheel loads
print("\n8. COMBINED g-g STATE - four-wheel loads")
W_TOT = CAR["mass_total_kg"] * lt.G

# THE identity: whatever the car is doing, the road pushes up by exactly its
# weight. Checked over the whole envelope, not just a few points.
lat_e, lon_e = lt.gg_envelope()
grid_lat, grid_lon = np.meshgrid(np.linspace(-1.2, 1.2, 25),
                                 np.linspace(-1.2, 1.0, 25))
inside = lt.inside_envelope(grid_lat, grid_lon)
worst = 0.0
for al, ao in zip(np.r_[grid_lat[inside], lat_e], np.r_[grid_lon[inside], lon_e]):
    w = lt.wheel_loads(al, ao, HF, HR, CAR)
    worst = max(worst, abs(w["total"] - W_TOT))
check("FL+FR+RL+RR == total weight everywhere on the g-g diagram",
      worst < 1e-6, f"worst error {worst:.2e} N over "
                    f"{inside.sum() + len(lat_e)} states")

s = lt.wheel_loads(0.0, 0.0, HF, HR, CAR)
check("static: left == right on both axles",
      abs(s["FL"] - s["FR"]) < 1e-9 and abs(s["RL"] - s["RR"]) < 1e-9)
check("static: axle split matches the weight distribution",
      abs(s["W_r"] / W_TOT - CAR["rear_frac"]) < 1e-12,
      f"{s['W_r']/W_TOT*100:.2f} % rear")

# longitudinal, against the lecture's own formula  N = mgL/E -+ m a h / E
print("\n   longitudinal (ENGR7008 wk4 pp.29-32)")
b = lt.wheel_loads(0.0, -1.0, HF, HR, CAR)
exact = W_TOT * 1.0 * CAR["h_cg_mm"] / CAR["wheelbase_mm"]
check("braking moves exactly m*a*h/E to the front",
      abs((b["W_f"] - W_TOT * (1 - CAR["rear_frac"])) - exact) < 1e-9,
      f"{b['W_f'] - W_TOT*(1-CAR['rear_frac']):.1f} vs {exact:.1f} N")
acc = lt.wheel_loads(0.0, +1.0, HF, HR, CAR)
check("accelerating moves the same amount to the REAR",
      abs((acc["W_r"] - W_TOT * CAR["rear_frac"]) - exact) < 1e-9)
check("braking and accelerating are mirror images",
      abs((b["W_f"] - W_TOT * (1 - CAR["rear_frac"]))
          + (acc["W_f"] - W_TOT * (1 - CAR["rear_frac"]))) < 1e-9)

# pure lateral must agree with the dedicated transfer() routine
print("\n   consistency with the pure-lateral model")
p = lt.wheel_loads(1.5, 0.0, HF, HR, CAR)
t15 = lt.transfer(1.5, HF, HR, CAR)
check("lateral transfer at a_long = 0 == transfer() front",
      abs(p["dW_lat_f"] - t15["total_f"]) < 1e-9,
      f"{p['dW_lat_f']:.4f} vs {t15['total_f']:.4f} N")
check("lateral transfer at a_long = 0 == transfer() rear",
      abs(p["dW_lat_r"] - t15["total_r"]) < 1e-9)

# mirror symmetry in lateral g
print("\n   symmetry")
r_pos = lt.wheel_loads(+1.0, -0.5, HF, HR, CAR)
r_neg = lt.wheel_loads(-1.0, -0.5, HF, HR, CAR)
check("reversing lateral g swaps left and right",
      abs(r_pos["FL"] - r_neg["FR"]) < 1e-9
      and abs(r_pos["RL"] - r_neg["RR"]) < 1e-9)
check("the g-g envelope has a FLAT TOP at the traction limit",
      abs(np.max(np.abs(lat_e[np.abs(lon_e - 1.0) < 1e-9]))
          - np.sqrt(1.2**2 - 1.0**2)) < 0.02,
      "clipped circle, not an ellipse")
check("reversing lateral g leaves the AXLE totals alone",
      abs(r_pos["W_f"] - r_neg["W_f"]) < 1e-9)

# the envelope helper itself
print("\n   g-g envelope helper")
check("envelope points are all ON the limit", np.all(lt.inside_envelope(lat_e, lon_e)))
check("the origin is inside", bool(lt.inside_envelope(0.0, 0.0)))
check("beyond the limit is outside",
      not bool(lt.inside_envelope(1.3, 0.0))
      and not bool(lt.inside_envelope(0.0, 1.1)))
check("braking limit is deeper than the traction limit",
      bool(lt.inside_envelope(0.0, -1.15)) and not bool(lt.inside_envelope(0.0, 1.05)))

# and the physically interesting one: a wheel CAN lift, and the model says so
big = lt.wheel_loads(1.6, 0.0, HF, HR, CAR)
check("an extreme lateral g lifts the INSIDE front (now FR: + = right turn)",
      "FR" in big["lifted"] or big["FR"] < 200.0,
      f"FR = {big['FR']:.0f} N at 1.6 g")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - load transfer obeys its identities and behaves sensibly.")
print("=" * 76)
