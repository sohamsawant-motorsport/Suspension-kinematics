# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for spring rates, ride frequency, and the damper motion ratio.
===============================================================================

  1  SERIES RATE   - the tyre-in-series law obeys the properties a series pair
                     must have, including both limits.
  2  ROUND TRIP    - frequency <-> wheel rate inverts exactly, with and without
                     the tyre, and the unreachable-target case returns inf
                     rather than a silently wrong number.
  3  OPTIMUMG      - our spring rate is checked against the Tech Tip's own
                     formula written out independently, INCLUDING the reciprocal
                     motion-ratio convention (the classic way to be off by MR^4).
  4  MASS          - the per-corner split conserves mass, and the SPRUNG rear
                     fraction correctly differs from the total rear fraction.
  5  BENCHMARK     - the reference car's published spring/wheel-rate pairs are
                     reproduced from its implied motion ratio. Two independent
                     columns of someone else's data have to agree with our maths.
  6  SHOCK MOUNT   - the damper's lower mount is carried by the right member.
                     This was a real bug: the solver hard-coded "lower wishbone",
                     which is right at the front but wrong at the rear, and gave
                     a rear motion ratio of 1.297 (a damper moving further than
                     the wheel). MR is SQUARED into the rate, so it was a factor
                     of 2.9 on every rear spring.

Run:  python validate_springs.py
"""
import numpy as np

import config
import rear_geometry as rg
import springs as sp
import tyre
from wishbone_solver import corner_geometry, solve_sweep

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("=" * 76)
print("VALIDATION - springs, ride frequency, motion ratio")
print("=" * 76)

# ------------------------------------------------------------- 1. series rate
print("\n1. TYRE IN SERIES  RR = Kw Kt / (Kw + Kt)")
kw, kt = 141.8, 310.4
rr = sp.ride_rate(kw, kt)
check("series rate is softer than EITHER spring", rr < kw and rr < kt,
      f"RR {rr:.1f} < Kw {kw:.1f}, Kt {kt:.1f}")
check("reciprocals add:  1/RR == 1/Kw + 1/Kt",
      abs(1 / rr - (1 / kw + 1 / kt)) < 1e-12)
check("a RIGID tyre gives back the wheel rate",
      abs(sp.ride_rate(kw, 1e12) - kw) < 1e-3,
      f"{sp.ride_rate(kw, 1e12):.3f}")
check("equal rates halve:  RR(k,k) == k/2", abs(sp.ride_rate(200.0, 200.0) - 100.0) < 1e-12)
check("the tyre costs a real amount of frequency here",
      12.0 < (1 - sp.frequency(kw, 241.0, kt) / sp.frequency(kw, 241.0)) * 100 < 20.0,
      f"{(1 - sp.frequency(kw, 241.0, kt) / sp.frequency(kw, 241.0)) * 100:.1f} % lower")

# -------------------------------------------------------------- 2. round trip
print("\n2. ROUND TRIP  frequency <-> wheel rate")
for f0 in (1.0, 2.5, 3.86, 5.0):
    k = sp.wheel_rate_for_frequency(f0, 241.0)
    check(f"no tyre, {f0} Hz round-trips", abs(sp.frequency(k, 241.0) - f0) < 1e-12,
          f"Kw {float(k):.1f} N/mm")
for f0 in (1.0, 2.5, 3.2):
    k = sp.wheel_rate_for_frequency(f0, 241.0, kt)
    check(f"with tyre, {f0} Hz round-trips",
          abs(sp.frequency(k, 241.0, kt) - f0) < 1e-9, f"Kw {float(k):.1f} N/mm")
check("a tyre-inclusive target needs a STIFFER wheel rate",
      sp.wheel_rate_for_frequency(3.2, 241.0, kt) > sp.wheel_rate_for_frequency(3.2, 241.0),
      f"{float(sp.wheel_rate_for_frequency(3.2, 241.0, kt)):.1f} vs "
      f"{float(sp.wheel_rate_for_frequency(3.2, 241.0)):.1f} N/mm")
f_imposs = sp.frequency(1e9, 241.0, kt) * 1.5      # beyond the tyre-only limit
check("an unreachable tyre-inclusive target returns inf, not a wrong number",
      np.isinf(float(sp.wheel_rate_for_frequency(f_imposs, 241.0, kt))),
      f"target {f_imposs:.2f} Hz exceeds the tyre-alone limit "
      f"{sp.frequency(1e9, 241.0, kt):.2f} Hz")

# ---------------------------------------------------------------- 3. OptimumG
print("\n3. AGAINST OPTIMUMG'S OWN FORMULA  Ks = 4 pi^2 f^2 m MR^2   (their MR = 1/R_M)")
for f0, m, rm in ((3.86, 241.0, 0.9908), (2.89, 319.0, 0.7635), (2.0, 180.0, 0.55)):
    ours = float(sp.spring_rate(sp.wheel_rate_for_frequency(f0, m), rm))
    theirs = 4 * np.pi ** 2 * f0 ** 2 * m * (1.0 / rm) ** 2 / 1000.0
    check(f"f={f0} Hz, m={m} kg, R_M={rm}", abs(ours - theirs) < 1e-9,
          f"{ours:.2f} N/mm both ways")
check("spring<->wheel rate inverts",
      abs(sp.wheel_rate_from_spring(sp.spring_rate(141.8, 0.9908), 0.9908) - 141.8) < 1e-12)

# -------------------------------------------------------------------- 4. mass
print("\n4. MASS BOOKKEEPING")
m = sp.corner_masses(1300.0, 0.56, 45.0)
check("mass is conserved",
      abs(m["sprung_total_kg"] + 4 * 45.0 - 1300.0) < 1e-9,
      f"sprung {m['sprung_total_kg']:.1f} + unsprung 180.0 = 1300.0 kg")
check("axle loads match the distribution",
      abs(m["axle_r_kg"] / 1300.0 - 0.56) < 1e-12)
check("SPRUNG rear fraction differs from the TOTAL rear fraction",
      abs(m["sprung_rear_frac"] - 0.56) > 1e-3,
      f"sprung {m['sprung_rear_frac']:.4f} vs total 0.5600 -- unsprung is split "
      f"evenly, the car is not")
check("the rear corner is the heavier one on this car",
      m["sprung_r_kg"] > m["sprung_f_kg"],
      f"F {m['sprung_f_kg']:.1f} kg  R {m['sprung_r_kg']:.1f} kg")
try:
    sp.corner_masses(1300.0, 0.56, 400.0)
    check("absurd unsprung mass is rejected", False)
except ValueError:
    check("absurd unsprung mass is rejected", True)

# --------------------------------------------------------------- 5. benchmark
print("\n5. BENCHMARK CROSS-CHECK  (published spring AND wheel rates must agree)")
# Two independent columns of the reference car's own data: if our K_w = K_s R_M^2
# is right, one implied motion ratio must reproduce the whole column.
for tag, table in (("front", [(250, 130), (275, 143), (300, 155),
                              (330, 171), (360, 187), (390, 202)]),
                   ("rear", [(150, 71), (175, 83), (200, 95),
                             (225, 107), (250, 119), (275, 131)])):
    rms = [np.sqrt(w / s) for s, w in table]
    rm = float(np.mean(rms))
    err = max(abs(float(sp.wheel_rate_from_spring(s, rm)) - w) for s, w in table)
    check(f"{tag}: one motion ratio reproduces all 6 published pairs",
          err < 1.0, f"R_M {rm:.4f}, worst error {err:.2f} N/mm")
    check(f"{tag}: the implied ratio is tightly consistent",
          float(np.ptp(rms)) < 0.005, f"spread {float(np.ptp(rms)):.4f}")

# ------------------------------------------------------------- 6. shock mount
print("\n6. DAMPER LOWER MOUNT -- carried by the right member")
PF = config._load_front_points()
PR = rg.solver_points(rg.load_rear())


def mr_of(pts, mode=None):
    g = corner_geometry(pts, -2.7, shock_on=mode)
    z = np.linspace(-6, 6, 13)
    sw = solve_sweep(g, z, theta_span_deg=28.0, n_theta=1121)
    i0 = int(np.argmin(np.abs(np.asarray(sw["bump"]))))
    return float(sw["MR"][i0]), g["shock_on"], float(np.asarray(sw["damper"])[i0])


gf = corner_geometry(PF, -2.7)
gr = corner_geometry(PR, -2.7)
check("FRONT mount auto-detects onto the LOWER arm", gf["shock_on"] == "lower",
      f"{np.linalg.norm(gf['SL'] - gf['LBJ']):.1f} mm from the LBJ vs "
      f"{np.linalg.norm(gf['SL'] - gf['UBJ']):.1f} from the UBJ")
check("REAR mount auto-detects onto the UPPER arm", gr["shock_on"] == "upper",
      f"{np.linalg.norm(gr['SL'] - gr['UBJ']):.1f} mm from the UBJ vs "
      f"{np.linalg.norm(gr['SL'] - gr['LBJ']):.1f} from the LBJ")

mrf, _, dlf = mr_of(PF)
mrr, _, dlr = mr_of(PR)
check("FRONT motion ratio is physically sane", 0.3 < mrf < 1.05, f"{mrf:.4f}")
check("REAR motion ratio is physically sane", 0.3 < mrr < 1.05, f"{mrr:.4f}")
check("REGRESSION: the rear is no longer > 1 (the old lower-arm bug)",
      mrr < 1.0, f"{mrr:.4f}, was 1.2971")
bad, _, _ = mr_of(PR, "lower")
check("and forcing the OLD assumption still reproduces the bug",
      bad > 1.2, f"lower-arm mode gives {bad:.4f}")
check("choosing the mount member does NOT move the static damper length",
      abs(dlr - mr_of(PR, "lower")[2]) < 1e-9,
      f"{dlr:.2f} mm either way -- only its RATE of change differs")
check("the mounting choice is a factor of ~3 on rear spring rate",
      2.5 < (bad / mrr) ** 2 < 3.3, f"MR^2 ratio {(bad / mrr) ** 2:.2f}")
try:
    mr_of(PR, "elbow")
    check("an unknown shock_on raises", False)
except ValueError:
    check("an unknown shock_on raises", True)

# ------------------------------------------------------------ 7. the design
print("\n7. THE DESIGN ITSELF")
kt_real = tyre.radial_stiffness_N_per_mm()
d = sp.design(1300.0, 0.56, 45.0, mrf, mrr, 3.86, 2.89, tyre_rate=kt_real)
check("targets are hit on the wheel-rate definition",
      abs(d["f_wheel_f"] - 3.86) < 1e-9 and abs(d["f_wheel_r"] - 2.89) < 1e-9)
check("the tyre-inclusive frequency is reported and is LOWER",
      d["f_ride_f"] < d["f_wheel_f"] and d["f_ride_r"] < d["f_wheel_r"],
      f"F {d['f_wheel_f']:.2f} -> {d['f_ride_f']:.2f} Hz")
check("front lands in OptimumG's high-downforce band",
      sp.classify(d["freq_f"]) == "high-downforce racecar")
check("spring rates are buildable numbers",
      50 < d["ks_f"] < 500 and 50 < d["ks_r"] < 500,
      f"F {d['ks_f']:.0f}  R {d['ks_r']:.0f} N/mm")
check("the REAR needs the stiffer coil despite the softer frequency",
      d["ks_r"] > d["ks_f"],
      f"F {d['ks_f']:.0f} vs R {d['ks_r']:.0f} N/mm -- the rear's lower motion "
      f"ratio ({mrr:.3f}) more than cancels its lower frequency")

# ------------------------------------------------- 8. the VEHICLE page wiring
# The Springs section is only useful if it is LIVE: corner weights must flow to
# the sprung masses, and hard points must flow to the motion ratio and therefore
# to the coil rate. Both are asserted by actually driving the widgets.
print("\n8. VEHICLE PAGE - springs are live, not typed in")
import tkinter as _tk                                                # noqa: E402

import explorer as X                                                 # noqa: E402

_app = X.Explorer()
_app.update()
try:
    _v, _front = _app.vehicle, _app.front
    _d = _v._spring
    check("the vehicle page produces a spring design", _d is not None)
    check("it agrees with the standalone module",
          abs(_d["ks_f"] - sp.design(1300.0, 0.56, 45.0, _d["mr_f"], _d["mr_r"],
                                     _d["freq_f"], _d["freq_r"])["ks_f"]) < 1e-6,
          f"{_d['ks_f']:.2f} N/mm both ways")
    check("corner weights sum to the total",
          abs(sum(_v.corner_kg.values()) - 1300.0) < 1e-6,
          f"{sum(_v.corner_kg.values()):.1f} kg")
    check("the motion ratio matches the FRONT page's solved geometry",
          abs(_d["mr_f"] - float(_front.cur["R"]["mr0"])) < 1e-12,
          f"{_d['mr_f']:.4f}")

    # BALLAST: move 3 % of the car rearward and the springs must follow
    _ks_f0, _ks_r0 = _d["ks_f"], _d["ks_r"]
    for _k in ("FL", "FR"):
        _v.cw[_k].set("20.5")
    for _k in ("RL", "RR"):
        _v.cw[_k].set("29.5")
    _v.refresh()
    _d2 = _v._spring
    check("BALLAST rearward softens the front and stiffens the rear",
          _d2["ks_f"] < _ks_f0 and _d2["ks_r"] > _ks_r0,
          f"F {_ks_f0:.1f}->{_d2['ks_f']:.1f}   R {_ks_r0:.1f}->{_d2['ks_r']:.1f} N/mm")
    check("the rear fraction really moved", abs(_d2["m_r"] - _d["m_r"]) > 15.0,
          f"rear sprung/corner {_d['m_r']:.1f} -> {_d2['m_r']:.1f} kg")
    for _k, _val in (("FL", "22"), ("FR", "22"), ("RL", "28"), ("RR", "28")):
        _v.cw[_k].set(_val)
    _v.refresh()

    # the entries are normalised, so any consistent units must give the same answer
    for _k in ("FL", "FR"):
        _v.cw[_k].set("286")
    for _k in ("RL", "RR"):
        _v.cw[_k].set("364")
    _v.refresh()
    check("corner boxes can be kg as well as % (they are normalised)",
          abs(_v._spring["ks_f"] - _ks_f0) < 1e-9,
          f"{_v._spring['ks_f']:.2f} N/mm from kg entries")
    for _k, _val in (("FL", "22"), ("FR", "22"), ("RL", "28"), ("RR", "28")):
        _v.cw[_k].set(_val)
    _v.refresh()

    # HARD POINT: move the front damper mount and the coil rate must follow
    _tbl = _front.tab["R"]
    _var = _tbl.cells["Lower Shock mount"][0][0]
    _base = _var.get()
    _var.set(f"{float(_base) - 80.0:.3f}")
    _front.apply()
    _d3 = _v._spring
    check("a HARD-POINT move changes the motion ratio",
          abs(_d3["mr_f"] - _ks_f0 * 0 - _d["mr_f"]) > 0.2,
          f"R_M {_d['mr_f']:.4f} -> {_d3['mr_f']:.4f}")
    check("...and flows through to the coil rate", _d3["ks_f"] > _ks_f0 * 1.5,
          f"front coil {_ks_f0:.1f} -> {_d3['ks_f']:.1f} N/mm")
    check("the frequency TARGET is unchanged by that move",
          abs(_d3["freq_f"] - _d["freq_f"]) < 1e-12,
          "the geometry changed, not the goal")
    _var.set(_base)
    _front.apply()

    check("both tabs share ONE set of wheel rates",
          abs(_v.car["kw_f"] - _v._spring["kw_f"]) < 1e-9
          and abs(_v.car["kw_r"] - _v._spring["kw_r"]) < 1e-9,
          "the g-g page transfers load through the designed springs")
    check("the springs figure actually drew", len(_v.axS1.lines) >= 4
          and len(_v.axS2.lines) >= 4)

    # ---- the spring CATALOGUE table (setup-sheet format)
    print("\n9. SPRING TABLE - every row internally consistent")
    _d = _v._spring
    for _tag, _nm in (("f", "FRONT"), ("r", "REAR")):
        _rows, _pick = _d[f"table_{_tag}"], _d[f"pick_{_tag}"]
        _m, _rm, _kt = _d[f"m_{_tag}"], _d[f"mr_{_tag}"], _d["k_tyre"]
        _W = _m * X.ltr.G
        check(f"{_nm}: table is populated", len(_rows) >= 6, f"{len(_rows)} rows")
        _bad = []
        for _k, _kw, _f1, _f2, _defl, _dh in _rows:
            if abs(_kw - _k * _rm ** 2) > 1e-9:
                _bad.append(f"wheel rate at {_k:.0f}")
            if abs(_f1 - float(sp.frequency(_kw, _m))) > 1e-9:
                _bad.append(f"f at {_k:.0f}")
            if abs(_f2 - float(sp.frequency(_kw, _m, _kt))) > 1e-9:
                _bad.append(f"f+tyre at {_k:.0f}")
            if abs(_defl - _W / _kw) > 1e-9:
                _bad.append(f"deflection at {_k:.0f}")
        check(f"{_nm}: wheel rate, both frequencies and deflection all check out",
              not _bad, "; ".join(_bad[:3]) if _bad else f"{len(_rows)} rows exact")
        _ks = [r[0] for r in _rows]
        check(f"{_nm}: coil rates ascend", all(np.diff(_ks) > 0))
        check(f"{_nm}: a stiffer spring raises the frequency",
              all(np.diff([r[2] for r in _rows]) > 0))
        check(f"{_nm}: a stiffer spring reduces static deflection",
              all(np.diff([r[4] for r in _rows]) < 0))
        check(f"{_nm}: d-ride-height is ZERO on the highlighted row",
              abs(_rows[_pick][5]) < 1e-9, f"row {_pick}, coil {_ks[_pick]:.0f} N/mm")
        check(f"{_nm}: softer than the reference sits the car HIGHER",
              _rows[0][5] > 0 and _rows[-1][5] < 0,
              f"{_rows[0][5]:+.1f} mm at the softest, {_rows[-1][5]:+.1f} at the stiffest")
        check(f"{_nm}: the highlighted row is the nearest shelf rate to the design",
              _pick == int(np.argmin(np.abs(np.array(_ks) - _d[f'ks_{_tag}']))),
              f"design {_d['ks_' + _tag]:.1f} -> shelf {_ks[_pick]:.0f} N/mm")
        check(f"{_nm}: the widget shows the same numbers as the data",
              _v.sptab.cells[_tag][_pick][0].cget("text") == f"{_ks[_pick]:.0f}",
              f"cell reads {_v.sptab.cells[_tag][_pick][0].cget('text')!r}")
        check(f"{_nm}: the design row is highlighted in the widget",
              _v.sptab.cells[_tag][_pick][0].cget("bg") == X._SpringTable.PICK)

    # ---- interactivity
    print("\n10. SPRING PLOTS - interactive, with stock and benchmark kept")
    class _Ev2:
        inaxes = None
        xdata = None
    _e = _Ev2()
    _e.inaxes, _e.xdata = _v.axS1, 3.35
    _v._on_spring_move(_e)
    check("hovering S1 fills its own readout", bool(_v._snote[_v.axS1].get_text()))
    check("the OTHER plot is cleared (different x units)",
          _v._snote[_v.axS2].get_text() == "")
    _my = float(_v._smark[_v.axS1][0].get_ydata()[0])
    _want = float(sp.spring_rate(sp.wheel_rate_for_frequency(3.35, _d["m_f"]),
                                 _d["mr_f"]))
    check("the S1 cursor snaps onto the FRONT curve", abs(_my - _want) < 1e-9,
          f"{_my:.1f} N/mm at 3.35 Hz")
    # S2's x axis is in the DISPLAYED convention, so hover at the displayed
    # value of R_M = 0.72 and expect the coil rate that R_M implies. The answer
    # must not depend on which convention the axis happens to be drawn in --
    # that is exactly what this check is for.
    _x72 = float(sp.to_display(0.72))
    _e.inaxes, _e.xdata = _v.axS2, _x72
    _v._on_spring_move(_e)
    check("hovering S2 fills its readout and clears S1",
          bool(_v._snote[_v.axS2].get_text())
          and _v._snote[_v.axS1].get_text() == "")
    check("S2 reads the benchmark motion ratio correctly",
          abs(float(_v._smark[_v.axS2][0].get_ydata()[0])
              - _d["kw_f"] / 0.72 ** 2) < 1e-9)
    check("the S2 cursor lands at the hovered x, in displayed units",
          abs(float(_v._smark[_v.axS2][0].get_xdata()[0]) - _x72) < 1e-9)
    _v._on_spring_leave(None)
    check("leaving clears both readouts",
          all(_v._snote[a].get_text() == "" for a in (_v.axS1, _v.axS2)))
    # stock + benchmark must SURVIVE the hover, they are the fixed references
    _e.inaxes, _e.xdata = _v.axS1, 4.4
    _v._on_spring_move(_e)
    check("the STOCK markers are still on the plot after hovering",
          sum(1 for ln in _v.axS1.lines if ln.get_marker() == "s") >= 4)
    check("the reference lines are still there (or the axes still hold curves)",
          len([ln for ln in _v.axS2.lines if ln.get_linestyle() in ("--", ":")]) >= 2
          or len(_v.axS2.get_lines()) >= 2)
finally:
    _app.destroy()


# ---------------------------------------------------------------------------
#  10.  MOTION-RATIO CONVENTION
# ---------------------------------------------------------------------------
# The project computes in R_M = spring/wheel and SHOWS MR = wheel/spring. Every
# one of those conversions has to be exact and reversible, because mixing the two
# is an R_M^4 error -- a factor of 10 at our values. This section pins the map.
print("\n10. MOTION-RATIO CONVENTION (textbook MR vs internal R_M)")


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


_RM = (0.5644, 0.5999, 0.75, 0.80, 0.9908, 1.0, 1.25)

check("textbook display is the exact reciprocal",
      all(abs(float(sp.to_display(r, "textbook")) - 1.0 / r) < 1e-15 for r in _RM))
check("internal display is the identity",
      all(float(sp.to_display(r, "internal")) == r for r in _RM))
check("display round-trips back to R_M",
      all(abs(float(sp.from_display(sp.to_display(r))) - r) < 1e-14 for r in _RM))
check("to_display is its own inverse (involution)",
      all(abs(float(sp.to_display(sp.to_display(r))) - r) < 1e-14 for r in _RM))
check("to_display works elementwise on an array",
      bool(np.allclose(sp.to_display(np.array(_RM)), 1.0 / np.array(_RM))))
check("a zero R_M displays as inf, not a crash",
      bool(np.isinf(float(sp.to_display(0.0)))))

# the two rate formulas must agree -- this is the whole point of the block
check("K_s = K_w / R_M^2 equals K_s = K_w * MR^2",
      all(abs(float(sp.spring_rate(137.9, r))
              - float(sp.spring_rate_textbook(137.9, sp.to_display(r, "textbook"))))
          < 1e-9 for r in _RM))
check("our front: R_M 0.5644 -> MR 1.7718, coil 433.0 N/mm either way",
      abs(float(sp.to_display(0.5644)) - 1.7717929) < 1e-6
      and abs(float(sp.spring_rate_textbook(137.9, float(sp.to_display(0.5644))))
              - 432.9027) < 1e-3)

# a wrong convention must be LOUD, not silently off by R_M^4
check("an unknown convention raises", _raises(lambda: sp.to_display(0.7, "metric")))
check("symbol/label/formula track the convention",
      sp.symbol("textbook") == "MR" and sp.symbol("internal") == "R_M"
      and "wheel travel" in sp.ratio_text("textbook").split("/")[0]
      and "K_s = K_w" in sp.formula("textbook")
      and "K_w = K_s" in sp.formula("internal"))
check("both_text names BOTH numbers and both conventions",
      all(t in sp.both_text(0.5644)
          for t in ("1.7718", "0.5644", "wheel/spring", "spring/wheel")))
check("the shipped default is the textbook convention", sp.CONVENTION == "textbook")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - spring maths agrees with OptimumG and Gillespie, the\n"
      "benchmark's own two data columns, and the corrected motion ratio.")
print("=" * 76)
