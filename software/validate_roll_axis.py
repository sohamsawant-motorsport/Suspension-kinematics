# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the roll axis and front/rear roll-centre balance.
======================================================================

  1  GEOMETRY     - the axis is a straight line through the two roll centres:
                    it passes through both, interpolates linearly between them,
                    and its inclination inverts back to the rise.
  2  LEVER        - the roll lever is measured at the SPRUNG rear fraction, not
                    the total one, and a BELOW-ground axis genuinely lengthens
                    it. Cross-checked against load_transfer's own routine.
  3  SHARES       - geometric + elastic + unsprung == 100 % of each axle's
                    transfer, a roll centre AT ground kills its geometric term
                    exactly, and below ground makes it NEGATIVE.
  4  IN ROLL      - the axis is rebuilt from both axles' migrating roll centres;
                    symmetry in +/- roll is asserted, since a symmetric car must
                    have an EVEN axis height in roll.
  5  BALANCE MAP  - raising the front RC moves LLTD forward and raising the rear
                    moves it back, monotonically; and the map agrees with a
                    direct load-transfer call at every sampled point.
  6  CROSS-MODULE - the same roll gradient comes out of this module's lever and
                    load_transfer's, and matches the OptimumG Eq.9 form.
  7  BENCHMARK    - against the reference GT3's published roll centres: BOTH of
                    theirs are above ground and the axis is nearly level, so its
                    lever is SHORTER than the CG height while ours is longer.
                    The shape of that gap is what a redesign has to close.

Run:  python validate_roll_axis.py
"""
import numpy as np

import benchmarks as bm

import load_transfer as ltr
import roll_axis as ra

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


CAR = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
RCF, RCR = -283.6, -52.0

print("=" * 76)
print("VALIDATION - roll axis and F/R roll-centre balance")
print("=" * 76)

# ------------------------------------------------------------- 1. geometry
print("\n1. THE AXIS IS A STRAIGHT LINE THROUGH THE TWO ROLL CENTRES")
check("passes through the FRONT roll centre at u = 0",
      abs(float(ra.axis_height_at(0.0, RCF, RCR)) - RCF) < 1e-12)
check("passes through the REAR roll centre at u = 1",
      abs(float(ra.axis_height_at(1.0, RCF, RCR)) - RCR) < 1e-12)
check("halfway is the mean of the two",
      abs(float(ra.axis_height_at(0.5, RCF, RCR)) - (RCF + RCR) / 2) < 1e-12,
      f"{float(ra.axis_height_at(0.5, RCF, RCR)):+.1f} mm")
u = np.linspace(0, 1, 11)
h = ra.axis_height_at(u, RCF, RCR)
check("linear in between (second difference is zero)",
      np.max(np.abs(np.diff(h, 2))) < 1e-9)
inc = ra.inclination_deg(RCF, RCR, CAR["wheelbase_mm"])
check("inclination inverts back to the rise",
      abs(np.tan(np.radians(inc)) * CAR["wheelbase_mm"] - (RCR - RCF)) < 1e-9,
      f"{inc:+.3f} deg for {RCR - RCF:+.1f} mm of rise")
check("front RC lower than rear reads as NOSE-DOWN (positive)", inc > 0)
check("a level axis has zero inclination",
      abs(ra.inclination_deg(-100.0, -100.0, 2645.0)) < 1e-12)
check("swapping the roll centres flips the sign",
      abs(ra.inclination_deg(RCR, RCF, 2645.0) + inc) < 1e-9)

# ---------------------------------------------------------------- 2. lever
print("\n2. THE ROLL LEVER")
st = ra.state(RCF, RCR, CAR)
check("axis height matches load_transfer's own routine",
      abs(st["h_axis"] - float(ltr.roll_axis_height(RCF, RCR, st["u_cg"]))) < 1e-12,
      f"{st['h_axis']:+.1f} mm")
check("measured at the SPRUNG rear fraction, not the total",
      abs(st["u_cg"] - CAR["rear_frac"]) > 1e-3,
      f"sprung {st['u_cg']:.4f} vs total {CAR['rear_frac']:.4f}")
check("lever = CG height - axis height",
      abs(st["lever"] - (CAR["h_cg_mm"] - st["h_axis"])) < 1e-12,
      f"{st['lever']:.1f} mm")
check("a BELOW-ground axis LENGTHENS the lever past the CG height",
      st["lever"] > CAR["h_cg_mm"], f"{st['lever_vs_cg']:+.1%}")
check("an axis AT ground gives a lever exactly equal to the CG height",
      abs(ra.state(0.0, 0.0, CAR)["lever"] - CAR["h_cg_mm"]) < 1e-9)
check("an ABOVE-ground axis SHORTENS it",
      ra.state(50.0, 50.0, CAR)["lever"] < CAR["h_cg_mm"],
      f"{ra.state(50.0, 50.0, CAR)['lever']:.1f} mm at +50 mm RCs")
# The lever depends only on where the axis passes UNDER THE CG, so a completely
# different pair of roll centres that happens to put it at the same height must
# give an identical lever. Solve for the rear RC that does it rather than
# guessing one -- guessing is what made this check fail first time round.
_rcf2 = -200.0
_rcr2 = _rcf2 + (st["h_axis"] - _rcf2) / st["u_cg"]
_st2 = ra.state(_rcf2, _rcr2, CAR)
check("lever depends on the AXIS, not on either roll centre alone",
      abs(_st2["lever"] - st["lever"]) < 1e-9,
      f"({RCF:.1f}, {RCR:.1f}) and ({_rcf2:.0f}, {_rcr2:.1f}) both give "
      f"{st['lever']:.1f} mm, at {inc:+.2f} vs {_st2['incl_deg']:+.2f} deg of tilt")
check("...even though the two have very different inclinations",
      abs(_st2["incl_deg"] - inc) > 1.0)

# --------------------------------------------------------------- 3. shares
print("\n3. GEOMETRIC / ELASTIC / UNSPRUNG SHARES")
s = ra.split(RCF, RCR, CAR)
for tag, nm in (("f", "FRONT"), ("r", "REAR")):
    tot = s[f"geo_pct_{tag}"] + s[f"el_pct_{tag}"] + s[f"uns_pct_{tag}"]
    check(f"{nm}: the three shares sum to 100 %", abs(tot - 100.0) < 1e-9,
          f"geo {s['geo_pct_' + tag]:+.1f}  el {s['el_pct_' + tag]:+.1f}  "
          f"uns {s['uns_pct_' + tag]:+.1f}")
_g0 = ra.split(0.0, 0.0, CAR)
check("a roll centre AT ground gives exactly ZERO geometric transfer",
      abs(_g0["geo_f"]) < 1e-9 and abs(_g0["geo_r"]) < 1e-9)
check("BELOW ground makes it negative (links transfer the wrong way)",
      s["geo_f"] < 0 and s["geo_r"] < 0,
      f"front {s['geo_f']:.0f} N, rear {s['geo_r']:.0f} N")
_gp = ra.split(60.0, 60.0, CAR)
check("ABOVE ground makes it positive", _gp["geo_f"] > 0 and _gp["geo_r"] > 0)
check("geometric transfer is LINEAR in roll-centre height",
      abs(ra.split(100.0, 0.0, CAR)["geo_f"]
          - 2 * ra.split(50.0, 0.0, CAR)["geo_f"]) < 1e-6)
check("total transfer is conserved however it splits",
      abs((s["total_f"] + s["total_r"])
          - (_g0["total_f"] + _g0["total_r"])) < 1.0,
      "moving the RCs re-routes transfer, it does not create it")

# -------------------------------------------------------------- 4. in roll
print("\n4. THE AXIS UNDER ROLL")
_roll = np.linspace(-5, 5, 11)
_zf = -283.6 + 3.0 * _roll ** 2          # symmetric, rises with |roll|
_zr = -52.0 + 1.0 * _roll ** 2
_ir = ra.in_roll(_roll, _zf, _roll, _zr, CAR)
check("rebuilt on a common roll grid", len(_ir["roll"]) > 10)
check("axis height is EVEN in roll on a symmetric car",
      np.max(np.abs(_ir["h_axis"] - _ir["h_axis"][::-1])) < 1e-9)
check("lever is EVEN in roll too",
      np.max(np.abs(_ir["lever"] - _ir["lever"][::-1])) < 1e-9)
check("at zero roll it reproduces the static state",
      abs(float(np.interp(0.0, _ir["roll"], _ir["h_axis"])) - st["h_axis"]) < 1e-6)
check("front and rear are interpolated independently",
      abs(float(np.interp(0.0, _ir["roll"], _ir["rc_f"])) - RCF) < 1e-6
      and abs(float(np.interp(0.0, _ir["roll"], _ir["rc_r"])) - RCR) < 1e-6)
check("a rising axis SHORTENS the lever through the corner",
      _ir["lever"][0] < _ir["lever"][len(_ir["lever"]) // 2],
      f"{_ir['lever'][len(_ir['lever']) // 2]:.1f} mm static -> "
      f"{_ir['lever'][0]:.1f} mm at full roll")

# ---------------------------------------------------------- 5. balance map
print("\n5. THE F/R ROLL-CENTRE BALANCE MAP")
_f = np.linspace(-300, 60, 7)
_r = np.linspace(-300, 60, 7)
F, R, LLTD, GRAD, LEV = ra.balance_map(_f, _r, CAR)
check("map is the right shape", LLTD.shape == (len(_r), len(_f)))
check("every point agrees with a direct load-transfer call",
      max(abs(LLTD[i, j] - ltr.transfer(1.0, F[i, j], R[i, j], CAR)["lltd_f"])
          for i in range(LLTD.shape[0]) for j in range(LLTD.shape[1])) < 1e-9)
check("raising the FRONT roll centre moves LLTD FORWARD",
      np.all(np.diff(LLTD, axis=1) > 0),
      f"{LLTD[3, 0]:.1f} % at RCf {_f[0]:.0f} -> {LLTD[3, -1]:.1f} % at {_f[-1]:.0f} mm")
check("raising the REAR roll centre moves LLTD REARWARD",
      np.all(np.diff(LLTD, axis=0) < 0),
      f"{LLTD[0, 3]:.1f} % at RCr {_r[0]:.0f} -> {LLTD[-1, 3]:.1f} % at {_r[-1]:.0f} mm")
check("raising BOTH shortens the lever and stiffens the roll gradient",
      GRAD[-1, -1] < GRAD[0, 0] and LEV[-1, -1] < LEV[0, 0],
      f"gradient {GRAD[0, 0]:.3f} -> {GRAD[-1, -1]:.3f} deg/g")
check("the balance lever is worth a real amount of LLTD",
      LLTD.max() - LLTD.min() > 5.0,
      f"{LLTD.min():.1f} to {LLTD.max():.1f} % across the map")

# ------------------------------------------------------------ 6. cross-check
print("\n6. CROSS-MODULE")
_t = ltr.transfer(1.0, RCF, RCR, CAR)
check("our lever == load_transfer's lever", abs(st["lever"] - _t["lever"]) < 1e-9)
check("our axis height == load_transfer's roll_axis",
      abs(st["h_axis"] - _t["roll_axis"]) < 1e-9)
# rebuild the roll gradient from the lever, OptimumG Eq.1/9 style
_Ws = (CAR["mass_total_kg"] - CAR["unsprung_f_kg"] - CAR["unsprung_r_kg"]) * ltr.G
_grad = (_Ws * st["lever"]) / (CAR["k_roll_f"] + CAR["k_roll_r"])
check("roll gradient rebuilt from the lever matches the model",
      abs(np.degrees(_grad) - _t["roll_deg"]) < 1e-9,
      f"{np.degrees(_grad):.4f} deg/g")

if not ra.BENCHMARK_IS_MEASURED:
    print("\n7. AGAINST A REFERENCE ROLL AXIS -- SKIPPED")
    print("   no private reference data installed; see software/benchmarks.py")
else:
    # ------------------------------------------------------- 7. BENCHMARK
    # Reference GT3 roll centres, read off its published kinematic curves
    # (published kinematic curves at zero wheel travel). Private data --
    # benchmarking only. What is asserted here is the SHAPE of the difference, which
    # is what a redesign has to close, not the numbers themselves.
    print("\n7. AGAINST THE BENCHMARK ROLL AXIS")
    _bf, _br = ra.BENCHMARK_RC_MM
    _bm = ra.state(_bf, _br, CAR)
    _sb = ra.split(_bf, _br, CAR)

    check("benchmark has BOTH roll centres above ground", _bf > 0 and _br > 0,
          f"front {_bf:+.0f}, rear {_br:+.0f} mm")
    check("ours has BOTH below ground", RCF < 0 and RCR < 0,
          f"front {RCF:+.1f}, rear {RCR:+.1f} mm")
    check("benchmark axis is nearly LEVEL", abs(_bm["incl_deg"]) < 0.5,
          f"{_bm['incl_deg']:+.2f} deg vs ours {st['incl_deg']:+.2f} deg")
    check("benchmark axis sits ABOVE ground under the CG", _bm["h_axis"] > 0,
          f"{_bm['h_axis']:+.1f} mm vs ours {st['h_axis']:+.1f} mm")
    check("so the benchmark lever is SHORTER than the CG height",
          _bm["lever"] < CAR["h_cg_mm"] < st["lever"],
          f"benchmark {_bm['lever']:.1f} < CG {CAR['h_cg_mm']:.0f} < ours "
          f"{st['lever']:.1f} mm")
    check("the lever gap is worth about 195 mm",
          190.0 < st["lever"] - _bm["lever"] < 200.0,
          f"{st['lever'] - _bm['lever']:.1f} mm, "
          f"{_bm['lever'] / st['lever'] - 1:+.1%}")
    check("benchmark geometric shares are POSITIVE on both axles",
          _sb["geo_pct_f"] > 0 and _sb["geo_pct_r"] > 0,
          f"front {_sb['geo_pct_f']:+.1f} %, rear {_sb['geo_pct_r']:+.1f} % "
          f"(ours {s['geo_pct_f']:+.1f} / {s['geo_pct_r']:+.1f})")
    check("geometry alone would cut the roll gradient by about a third",
          0.60 < _sb["roll_deg"] / s["roll_deg"] < 0.75,
          f"{s['roll_deg']:.3f} -> {_sb['roll_deg']:.3f} deg/g "
          f"({_sb['roll_deg'] / s['roll_deg'] - 1:+.1%}) with the SAME springs and bars")
    check("...while barely touching the balance",
          abs(_sb["lltd_f"] - s["lltd_f"]) < 1.0,
          f"LLTD {s['lltd_f']:.2f} -> {_sb['lltd_f']:.2f} % -- roll centre height is a "
          f"ROLL-MOMENT lever, the F/R SPLIT is the balance lever")
    check("the benchmark point lies inside the balance map's range",
          -320 <= _bf <= 80 and -320 <= _br <= 80)

    print("\n" + "=" * 76)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print("   -", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED - the roll axis is a straight line through the two roll\n"
          "centres, the lever is measured at the sprung station, and the balance map\n"
          "agrees with the load-transfer model at every point.")
    print("=" * 76)
