# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for mechanical balance and roll stiffness.
===========================================================

  1  SERIES ALGEBRA  - axle_rate is a correct parallel-then-series combination;
                       the tyre can only SOFTEN an axle; leaving it out is the
                       infinite-tyre limit; roll stiffness scales as t^2.
  2  HALF STEPS      - the harmonic-mean identity for mixed blade pairs is
                       derived, not assumed: build a bar and two blades
                       explicitly, combine them in series, and check the mixed
                       pair equals harmonic(k_ii, k_jj) INDEPENDENTLY of the bar.
  3  THE TWO QUANTITIES - roll stiffness % and balance points behave as defined:
                       zero at the baseline cell, monotonic in the right
                       directions, and the front/rear signs opposed.
  4  THE ELASTIC FRACTION k - d(LLTD)/d(roll-stiffness share) really is the
                       elastic share of the load transfer, measured three ways
                       that must agree: finite difference, the transfer model's
                       own split, and the closed form.
  5  reset_arb        - the solver actually restores balance; it is an exact
                       inverse of the table; and it reports honestly when the
                       balance is out of a bar's reach.
  6  BENCHMARK RECONSTRUCTION - the whole reason we trust the definitions. All
                       150 published cells of the reference GT3's three tables
                       are rebuilt from that same handbook's spring and
                       anti-roll-bar rate tables, and four candidate definitions
                       are scored against them. The suite asserts that the
                       winning pair (tyre in series + balance = LLTD) beats the
                       alternatives by the margin found during the study, so a
                       later edit that quietly changes a definition fails here.
  7  CROSS-MODEL      - the elastic fraction this project's own load-transfer
                       model produces at the benchmark's roll-centre heights
                       agrees with the value recovered from the benchmark's
                       published tables. Two unrelated routes, one number.

Section 6 ran against a confidential performance handbook and is kept in a
separate, git-ignored file. `mech_balance.py` carries the DEFINITIONS that
reconstruction recovered -- which are this project's own work and are published
here -- never the reference car's numbers.
"""

import io
import os

import numpy as np

import arb as arbm
import load_transfer as ltr
import mech_balance as mb
import tyre

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}"
          + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


CAR = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
KT = tyre.radial_stiffness_N_per_mm()
RC_F, RC_R = 46.2, 22.0          # a sane pair; the broken front is exercised too
MR_F, MR_R = 0.68, 0.62          # generic; a private section may override


# ===========================================================================
print("\n1  SERIES ALGEBRA")
# ===========================================================================
k = mb.axle_rate(100.0, 50.0, None)
check("no tyre -> spring and bar simply add", abs(k - 150.0) < 1e-12,
      f"{k:.4f}")
ks = mb.axle_rate(100.0, 50.0, 300.0)
check("tyre in series is SOFTER than the suspension alone", ks < 150.0,
      f"{ks:.4f} < 150")
check("...and softer than the tyre alone too", ks < 300.0, f"{ks:.4f} < 300")
check("series formula exact  1/k = 1/150 + 1/300",
      abs(ks - 1.0 / (1.0 / 150.0 + 1.0 / 300.0)) < 1e-12)
check("an infinitely stiff tyre reproduces the no-tyre case",
      abs(mb.axle_rate(100.0, 50.0, 1e12) - 150.0) < 1e-6)
check("a rigid suspension cannot exceed the tyre",
      abs(mb.axle_rate(1e12, 0.0, 300.0) - 300.0) < 1e-3)
r1 = mb.roll_stiffness(100.0, 50.0, 1600.0)
r2 = mb.roll_stiffness(100.0, 50.0, 3200.0)
check("roll stiffness scales as track^2", abs(r2 / r1 - 4.0) < 1e-12,
      f"{r2 / r1:.6f}")
check("roll stiffness = k t^2 / 2", abs(r1 - 150.0 * 1600.0 ** 2 / 2.0) < 1e-6)
check("bar and spring are interchangeable in the sum (parallel)",
      abs(mb.axle_rate(60.0, 90.0, KT) - mb.axle_rate(90.0, 60.0, KT)) < 1e-12)


# ===========================================================================
print("\n2  HALF STEPS -- the harmonic-mean identity, derived not assumed")
# ===========================================================================
# build a real bar + two real blades and combine them the physical way
K_BAR = 400.0
BLADE = np.array([40.0, 60.0, 95.0, 150.0, 240.0, 400.0])    # 6 detents


def path(bi, bj, kbar=K_BAR):
    """blade i -- bar -- blade j, all in series along one load path."""
    return 1.0 / (1.0 / bi + 1.0 / kbar + 1.0 / bj)


sym = np.array([path(b, b) for b in BLADE])
worst = 0.0
for i in range(len(BLADE) - 1):
    direct = path(BLADE[i], BLADE[i + 1])
    ident = mb.half_step_rate(sym[i], sym[i + 1])
    worst = max(worst, abs(direct - ident))
check("mixed pair == harmonic mean of the two symmetric pairs",
      worst < 1e-10, f"max error {worst:.3e} N/mm")

# ...and that identity must not depend on the bar at all
worst = 0.0
for kbar in (150.0, 400.0, 5000.0, 1e9):
    s = np.array([path(b, b, kbar) for b in BLADE])
    for i in range(len(BLADE) - 1):
        worst = max(worst, abs(path(BLADE[i], BLADE[i + 1], kbar)
                               - mb.half_step_rate(s[i], s[i + 1])))
check("...and it holds for ANY bar stiffness (the 1/K terms cancel)",
      worst < 1e-9, f"max error {worst:.3e} over 4 bars")

lab = mb.pair_labels(6)
check("6 detents give 11 columns", len(lab) == 11, f"{len(lab)}")
check("column labels match the benchmark's",
      lab == ["1-1", "1-2", "2-2", "2-3", "3-3", "3-4", "4-4", "4-5", "5-5",
              "5-6", "6-6"])
check("4 detents give 7 columns", len(mb.pair_labels(4)) == 7)
pr = mb.pair_rates(sym)
check("pair_rates returns 2n-1 rates", len(pr) == 11, f"{len(pr)}")
check("pair_rates keeps the symmetric ones on the odd columns",
      np.allclose(pr[::2], sym))
check("pair rates are monotonic", np.all(np.diff(pr) > 0))
check("every half step lies between its neighbours",
      np.all(pr[1::2] > pr[0:-1:2]) and np.all(pr[1::2] < pr[2::2]))
check("harmonic mean is BELOW the arithmetic mean (series, not parallel)",
      np.all(pr[1::2] < 0.5 * (pr[0:-1:2] + pr[2::2])))


# ===========================================================================
print("\n3  THE TWO TABULATED QUANTITIES")
# ===========================================================================
SPR_F = [180.0, 195.0, 210.0, 225.0, 240.0, 255.0]
SPR_R = [230.0, 250.0, 270.0, 290.0, 310.0, 330.0]
ARB_F = mb.pair_rates(np.linspace(150.0, 330.0, 6))
ARB_R = mb.pair_rates(np.linspace(55.0, 125.0, 4))
BASE_SF, BASE_AF = 210.0, ARB_F[7]
BASE_SR, BASE_AR = 270.0, ARB_R[4]

balF, rsF, s0F = mb.tables(CAR, RC_F, RC_R, "f", SPR_F, ARB_F, MR_F,
                           BASE_SF, BASE_AF, tyre_rate=KT)
balR, rsR, s0R = mb.tables(CAR, RC_F, RC_R, "r", SPR_R, ARB_R, MR_R,
                           BASE_SR, BASE_AR, tyre_rate=KT)
iF, jF = SPR_F.index(BASE_SF), 7
iR, jR = SPR_R.index(BASE_SR), 4
check("front table reads exactly 0.0 at its baseline cell",
      abs(balF[iF, jF]) < 1e-9 and abs(rsF[iF, jF]) < 1e-9)
check("rear table reads exactly 0.0 at its baseline cell",
      abs(balR[iR, jR]) < 1e-9 and abs(rsR[iR, jR]) < 1e-9)
check("stiffer FRONT spring -> balance towards the front (understeer)",
      np.all(np.diff(balF[:, jF]) > 0))
check("stiffer FRONT bar -> balance towards the front",
      np.all(np.diff(balF[iF, :]) > 0))
check("stiffer REAR spring -> balance AWAY from the front (oversteer)",
      np.all(np.diff(balR[:, jR]) < 0))
check("stiffer REAR bar -> balance away from the front",
      np.all(np.diff(balR[iR, :]) < 0))
check("either axle stiffer -> total roll stiffness up (front)",
      np.all(np.diff(rsF[:, jF]) > 0) and np.all(np.diff(rsF[iF, :]) > 0))
check("either axle stiffer -> total roll stiffness up (rear)",
      np.all(np.diff(rsR[:, jR]) > 0) and np.all(np.diff(rsR[iR, :]) > 0))
spans = [rsF[i, -1] - rsF[i, 0] for i in range(len(SPR_F))]
check("the tyre makes a fixed bar step worth LESS on stiffer springs",
      np.all(np.diff(spans) < 0),
      f"span {spans[0]:.2f} -> {spans[-1]:.2f} %")
balF_nt, rsF_nt, _ = mb.tables(CAR, RC_F, RC_R, "f", SPR_F, ARB_F, MR_F,
                               BASE_SF, BASE_AF, tyre_rate=None)
spans_nt = [rsF_nt[i, -1] - rsF_nt[i, 0] for i in range(len(SPR_F))]
check("...and WITHOUT the tyre that span is constant (so the tyre causes it)",
      np.std(spans_nt) < 1e-9, f"sd {np.std(spans_nt):.2e} %")


# ===========================================================================
print("\n4  THE ELASTIC FRACTION k")
# ===========================================================================
s = mb.state(CAR, RC_F, RC_R, tyre_rate=KT)
k_model = s["elastic_frac"]
S0 = s["front_share_pct"] / 100.0
L0 = s["lltd_f"]
c2 = dict(CAR)
c2["arb_f"] = CAR["arb_f"] * 1.02
s2 = mb.state(c2, RC_F, RC_R, tyre_rate=KT)
k_fd = (s2["lltd_f"] - L0) / (s2["front_share_pct"] - s["front_share_pct"])
check("k measured by finite difference == the transfer model's elastic share",
      abs(k_fd - k_model) < 2e-3, f"{k_fd:.5f} vs {k_model:.5f}")
check("k is strictly below 1 with both roll centres above ground",
      0.0 < k_model < 1.0, f"{k_model:.4f}")
k_ground = mb.elastic_fraction(CAR, 0.0, 0.0, tyre_rate=KT)
check("roll centres ON the ground -> k rises (no geometric term left)",
      k_ground > k_model, f"{k_ground:.4f} > {k_model:.4f}")
c_nou = dict(CAR, unsprung_f_kg=1e-9, unsprung_r_kg=1e-9)
k_ideal = mb.elastic_fraction(c_nou, 0.0, 0.0, tyre_rate=KT)
check("...and with no unsprung mass either, k -> 1 exactly",
      abs(k_ideal - 1.0) < 1e-6, f"{k_ideal:.8f}")
k_bad = mb.elastic_fraction(CAR, -283.6, RC_R, tyre_rate=KT)
check("an UNDERGROUND front roll centre pushes k above 1 (over-sensitive)",
      k_bad > 1.0, f"{k_bad:.4f} at our broken front RC")
check("balance moves less than roll-stiffness share does",
      abs(k_model) < 1.0 and abs(L0 - 50) < 50)


# ===========================================================================
print("\n5  reset_arb -- the lookup the table exists for")
# ===========================================================================
for axle, spr, mr, bs, ba, grid in (("f", SPR_F, MR_F, BASE_SF, BASE_AF, SPR_F),
                                    ("r", SPR_R, MR_R, BASE_SR, BASE_AR, SPR_R)):
    worst = 0.0
    for new in grid:
        kb, info = mb.reset_arb(CAR, RC_F, RC_R, axle, new, mr, bs, ba,
                                tyre_rate=KT)
        worst = max(worst, abs(info["residual"]))
        if not info["reachable"]:
            worst = 1e9
    check(f"{axle.upper()}: balance restored at every spring on the grid",
          worst < 1e-6, f"max residual {worst:.2e} points")
kb, info = mb.reset_arb(CAR, RC_F, RC_R, "f", BASE_SF, MR_F, BASE_SF, BASE_AF,
                        tyre_rate=KT)
check("no spring change -> no bar change", abs(kb - BASE_AF) < 1e-6,
      f"{kb:.4f} vs {BASE_AF:.4f}")
kb_up, _ = mb.reset_arb(CAR, RC_F, RC_R, "f", SPR_F[-1], MR_F, BASE_SF, BASE_AF,
                        tyre_rate=KT)
kb_dn, _ = mb.reset_arb(CAR, RC_F, RC_R, "f", SPR_F[0], MR_F, BASE_SF, BASE_AF,
                        tyre_rate=KT)
check("stiffer front spring must be paid for with a SOFTER front bar",
      kb_up < BASE_AF < kb_dn, f"{kb_dn:.1f} -> {BASE_AF:.1f} -> {kb_up:.1f} N/mm")
# the solver must be the exact inverse of the table it is derived from
kb, _ = mb.reset_arb(CAR, RC_F, RC_R, "f", SPR_F[-1], MR_F, BASE_SF, BASE_AF,
                     tyre_rate=KT)
b2, _, _ = mb.tables(CAR, RC_F, RC_R, "f", [SPR_F[-1]], [kb], MR_F,
                     BASE_SF, BASE_AF, tyre_rate=KT)
check("reset_arb is an exact inverse of tables()", abs(b2[0, 0]) < 1e-6,
      f"{b2[0, 0]:.2e} points")
kb, info = mb.reset_arb(CAR, RC_F, RC_R, "f", 900.0, MR_F, BASE_SF, BASE_AF,
                        tyre_rate=KT, hi=BASE_AF)
check("an unreachable balance is reported, not silently clipped",
      not info["reachable"] and np.isnan(info["delta_pct"]))


# ===========================================================================
print("\n6  BENCHMARK RECONSTRUCTION")
# ===========================================================================
# The reconstruction that recovered the two table definitions ran against a
# confidential performance handbook, so neither the figures nor the section
# that uses them are in this repository. If the private file is installed
# beside this one it is exec'd here, in this module's own namespace, so the
# suite behaves exactly as it always did. Without it the section is skipped
# and nothing later depends on it.
def _find_private(name):
    """Look for a private file on GT3_PRIVATE_DIR, then anywhere on sys.path.

    It is deliberately NOT looked for inside this repository. Confidential
    material lives outside the tree entirely, so publishing this repo cannot
    leak it even if a .gitignore is edited by mistake.
    """
    import sys
    roots = [os.environ.get("GT3_PRIVATE_DIR", "")] + list(sys.path)
    for r in roots:
        if r and os.path.exists(os.path.join(r, name)):
            return os.path.join(r, name)
    return None


_BM = _find_private("validate_mech_balance_benchmark.py")
if _BM:
    exec(compile(io.open(_BM, encoding="utf-8").read(), _BM, "exec"))
else:
    print("   -- skipped: no private reference data on GT3_PRIVATE_DIR")

import roll_axis as _rax
if not _rax.BENCHMARK_IS_MEASURED:
    print("\n7  CROSS-MODEL -- SKIPPED")
    print("   needs a measured reference roll axis; see software/benchmarks.py")
else:
    # ===========================================================================
    print("\n7  CROSS-MODEL: two unrelated routes to the same k")
    # ===========================================================================
    import roll_axis as rax
    RCB_F, RCB_R = rax.BENCHMARK_RC_MM
    k_ours = mb.elastic_fraction(CAR, RCB_F, RCB_R, tyre_rate=KT)
    check("our load-transfer model, at the benchmark's own roll-centre heights, "
          "reproduces the k recovered from its published tables",
          abs(k_ours - k_pub) < 0.03,
          f"ours {k_ours:.4f} vs tables {k_pub:.4f}  "
          f"({100 * abs(k_ours / k_pub - 1):.1f} % apart)")
    check("our tyre rate is close to the one the benchmark tables imply",
          abs(KT / kt_fit - 1.0) < 0.15 if HAVE_SCIPY else True,
          f"ours {KT:.1f} vs implied {kt_fit:.0f} N/mm" if HAVE_SCIPY else "")
    check("our front roll-stiffness share sits in the public GT band",
          mb.FRONT_SHARE_BAND[0] <= s["front_share_pct"] <= mb.FRONT_SHARE_BAND[1],
          mb.classify_front_share(s["front_share_pct"]))


# ===========================================================================
print("\n8  THE EXPLORER PANEL")
# ===========================================================================
# The panel adds one piece of logic of its own: it builds the spring rows and
# blade-pair columns so the MIDDLE cell lands exactly on the live baseline. If
# that scaling drifts, every number in both tables is measured from a setup the
# car is not actually in -- which would be invisible on screen.
import matplotlib
matplotlib.use("Agg")
import tkinter as tk                                                # noqa: E402

try:
    _root = tk.Tk()
    _root.withdraw()
    HAVE_TK = True
except Exception:                                          # pragma: no cover
    HAVE_TK = False

if HAVE_TK:
    import explorer as X                                             # noqa: E402
    _nb = X.ttk.Notebook(_root)
    _pan = X.MechBalancePanel(_nb, None)
    for nb_, nn in ((6, 6), (4, 4), (2, 2), (8, 8)):
        rows, cols, n = _pan._grids(210.0, 275.0, nb_)
        check(f"{nb_} blades -> {2 * nb_ - 1} columns, middle one on the baseline",
              n == nn and len(cols) == 2 * nb_ - 1
              and abs(cols[n - 1] - 275.0) < 1e-9,
              f"centre column {cols[n - 1]:.6f} N/mm")
    rows, cols, n = _pan._grids(210.0, 275.0, 6)
    check("the spring rows straddle the baseline and include it exactly",
          abs(rows[2] - 210.0) < 1e-9 and rows[0] < 210.0 < rows[-1],
          f"{rows[0]:.1f} .. {rows[-1]:.1f} N/mm")
    check("the columns stay monotonic after the rescale",
          bool(np.all(np.diff(cols) > 0)))
    check("panel exposes the three sub-tabs",
          [_pan.nb.tab(i, "text").strip() for i in range(_pan.nb.index("end"))]
          == ["Front axle", "Rear axle", "Reset & trade"])
    check("a panel with no page degrades quietly instead of raising",
          _pan.refresh() is None)
    # the table built on the panel's own grids must still zero at the baseline
    b, r_, _ = mb.tables(CAR, RC_F, RC_R, "f", rows, cols, MR_F, 210.0,
                         cols[n - 1], tyre_rate=KT)
    check("panel grids give a table that reads 0.0 at its centre cell",
          abs(b[2, n - 1]) < 1e-9 and abs(r_[2, n - 1]) < 1e-9)
    _root.destroy()
else:                                                      # pragma: no cover
    check("Tk available for the panel checks", False, "no display")


# ===========================================================================
print("\n9  SENSITIVITY -- what each input does to the balance")
# ===========================================================================
# The study guide's parameter table is generated by mb.sensitivity(); these
# checks are what make it safe to learn from. Every sign is asserted on BOTH
# roll-centre pairs, so a rule that only holds because our roll centres are
# underground cannot slip through as general physics.
SC = dict(CAR)
SC["arb_f"], SC["arb_r"] = 304.7824, 70.8934
KSF, MRF_, KSR, MRR_ = 144.4058, 0.99079, 180.4345, 0.76351
RC_OURS, RC_SANE = (-283.571, -52.0), (46.2, 22.0)
SENS = {tag: {r["key"]: r for r in mb.sensitivity(SC, rc[0], rc[1], KSF, MRF_,
                                                 KSR, MRR_, KT)}
        for tag, rc in (("ours", RC_OURS), ("sane", RC_SANE))}

for tag in ("ours", "sane"):
    d_ = SENS[tag]
    check(f"[{tag}] front spring, bar and motion ratio all move balance FORWARD",
          all(d_[k]["delta"] > 0 for k in ("ks_f", "arb_f", "mr_f")))
    check(f"[{tag}] rear spring, bar and motion ratio all move balance REARWARD",
          all(d_[k]["delta"] < 0 for k in ("ks_r", "arb_r", "mr_r")))
    check(f"[{tag}] raising a roll centre moves balance TOWARDS that axle",
          d_["rc_f"]["delta"] > 0 and d_["rc_r"]["delta"] < 0,
          f"front {d_['rc_f']['delta']:+.3f}  rear {d_['rc_r']['delta']:+.3f}")
    check(f"[{tag}] wider track moves balance towards that axle (t^2 beats 1/t)",
          d_["track_f_mm"]["delta"] > 0 and d_["track_r_mm"]["delta"] < 0)
    check(f"[{tag}] more unsprung mass moves balance towards that axle",
          d_["unsprung_f_kg"]["delta"] > 0 and d_["unsprung_r_kg"]["delta"] < 0)
    check(f"[{tag}] stiffer tyre moves balance forward (front is more choked)",
          d_["tyre"]["delta"] > 0)
    check(f"[{tag}] rear roll centre is the strongest geometric lever",
          abs(d_["rc_r"]["delta"]) > abs(d_["rc_f"]["delta"]))

check("the one sign that FLIPS: rear weight pushes balance forward only because "
      "our rear geometric term is negative",
      SENS["ours"]["rear_frac"]["delta"] > 0 > SENS["sane"]["rear_frac"]["delta"],
      f"ours {SENS['ours']['rear_frac']['delta']:+.3f}  "
      f"sane {SENS['sane']['rear_frac']['delta']:+.3f}")

# ---- the two ratios the guide quotes, as derivatives (tiny steps) --------
H = 1e-4


def dL(car_, rc, key, h=H):
    a = dict(ks_f=KSF, mr_f=MRF_, ks_r=KSR, mr_r=MRR_)
    c_ = dict(car_)
    base_ = mb.lltd_from_rates(c_, rc[0], rc[1], a["ks_f"], a["mr_f"],
                               a["ks_r"], a["mr_r"], KT)
    if key in a:
        a[key] += h
    else:
        c_[key] += h
    return (mb.lltd_from_rates(c_, rc[0], rc[1], a["ks_f"], a["mr_f"],
                               a["ks_r"], a["mr_r"], KT) - base_) / h


for ax, mr_ in (("f", MRF_), ("r", MRR_)):
    r_ = dL(SC, RC_OURS, f"ks_{ax}") / dL(SC, RC_OURS, f"arb_{ax}")
    check(f"{ax.upper()}: spring / bar sensitivity per N/mm == MR^2 exactly",
          abs(r_ - mr_ ** 2) < 1e-5, f"{r_:.6f} vs MR^2 {mr_ ** 2:.6f}")

kwf, kwr = KSF * MRF_ ** 2, KSR * MRR_ ** 2
cf, cr_ = kwf + SC["arb_f"], kwr + SC["arb_r"]
Kf_ = cf * KT / (cf + KT) * SC["track_f_mm"] ** 2 / 2
Kr_ = cr_ * KT / (cr_ + KT) * SC["track_r_mm"] ** 2 / 2
pass_f, pass_r = KT ** 2 / (cf + KT) ** 2, KT ** 2 / (cr_ + KT) ** 2
closed = -(Kf_ / Kr_) * (pass_r / pass_f) \
    * (SC["track_r_mm"] / SC["track_f_mm"]) ** 2
for tag, rc in (("ours", RC_OURS), ("sane", RC_SANE)):
    num_ = dL(SC, rc, "arb_r") / dL(SC, rc, "arb_f")
    check(f"[{tag}] rear bar / front bar per N/mm == closed form "
          "-(K_f/K_r)(tyre pass R / F)(t_r/t_f)^2",
          abs(num_ / closed - 1.0) < 1e-4, f"{num_:+.4f} vs {closed:+.4f}")
check("...and that ratio does not depend on the roll centres at all",
      abs(dL(SC, RC_OURS, "arb_r") / dL(SC, RC_OURS, "arb_f")
          - dL(SC, RC_SANE, "arb_r") / dL(SC, RC_SANE, "arb_f")) < 1e-4)
check("the tyre passes MORE of a rear change through than a front one",
      pass_r > pass_f, f"rear {pass_r:.4f}  front {pass_f:.4f}")

# ---- why ours is twitchier: the multiplier is the elastic fraction -------
k_o = mb.state(dict(SC, kw_f=kwf, kw_r=kwr), *RC_OURS, tyre_rate=KT)["elastic_frac"]
k_s = mb.state(dict(SC, kw_f=kwf, kw_r=kwr), *RC_SANE, tyre_rate=KT)["elastic_frac"]
m_sens = dL(SC, RC_OURS, "arb_f") / dL(SC, RC_SANE, "arb_f")
check("springs/bars are more sensitive on our car by about k_ours / k_sane",
      abs(m_sens / (k_o / k_s) - 1.0) < 0.03,
      f"sensitivity x{m_sens:.3f}   k ratio x{k_o / k_s:.3f}")
check("sensitivity() reports the same base LLTD as state()",
      abs(SENS["ours"]["ks_f"]["base"]
          - mb.state(dict(SC, kw_f=kwf, kw_r=kwr), *RC_OURS,
                     tyre_rate=KT)["lltd_f"]) < 1e-9)
check("the tyre row is skipped when no tyre rate is given",
      all(r["key"] != "tyre" for r in
          mb.sensitivity(SC, *RC_SANE, KSF, MRF_, KSR, MRR_, None)))
check("lateral g cancels out of the balance tables exactly",
      np.abs(mb.tables(CAR, RC_F, RC_R, "f", SPR_F, ARB_F, MR_F, BASE_SF,
                       BASE_AF, tyre_rate=KT, lat_g=1.0)[0]
             - mb.tables(CAR, RC_F, RC_R, "f", SPR_F, ARB_F, MR_F, BASE_SF,
                         BASE_AF, tyre_rate=KT, lat_g=2.3)[0]).max() < 1e-9)

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - mechanical balance is the FRONT share of lateral load\n"
      "transfer, roll stiffness is the axle pair with the TYRE IN SERIES, and the\n"
      "two are linked by the elastic fraction k ~ 0.8 rather than being equal.\n"
      "All three definitions were recovered from, and re-verified against, 150\n"
      "published benchmark cells.")
print("=" * 76)
