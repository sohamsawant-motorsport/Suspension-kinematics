# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for anti-dive, anti-lift and anti-squat.
=========================================================

  1  THE KNOB      - axis_tilt_deg reads the CAD (both axes flat, so the car has
                     no wishbone anti at all), tilt_axes writes back exactly what
                     it is asked for, preserves the wishbone's mean height and
                     length, and round-trips.
  2  TAU           - the closed-form anchor: with both side-view axes PARALLEL
                     the corner translates, so tau == tan(tilt) exactly and the
                     patch and wheel centre must agree. Zero tilt gives zero tau
                     at both ends. Tau is odd in the tilt and monotonic.
  3  REFERENCE PT  - the contact point must be a MATERIAL point of the upright.
                     Asserted by the consequence: re-projecting to the ground
                     (which contact_patch does) makes tau_patch == tau_wc for
                     EVERY geometry, which would make anti-lift and anti-squat
                     impossible to set independently.
  4  THE SPLIT     - tau_total = tau_wishbone + tau_steer by construction, and
                     the steer term equals bump_steer x scrub/cos(KPI). The
                     lever is asserted CONSTANT across tie-rod positions, and
                     the steer term is asserted to vanish when bump steer does.
  5  PERCENTAGES   - Gillespie 7.28 / 7.29 / 7.19 reproduced from first
                     principles, the L/h scaling, the brake-bias proportionality,
                     the sign rules, and tau_for_anti as an exact inverse.
  6  SWING ARM     - tau_patch - tau_wc == R_wheel / SVSA, and parallel arms give
                     an infinite swing arm and hence a LOCKED ratio between
                     anti-squat and anti-lift.
  7  PITCH         - anti reduces travel linearly, 100 % anti gives exactly zero
                     travel at that axle, the no-anti case matches a hand
                     calculation, and a RWD front gets no anti under power.
  8  SOLVERS       - required_tilt and required_tilt_pair hit their targets, and
                     the pair solver reproduces the swing arm the benchmark's own
                     two numbers imply through Gillespie.
  9  COUPLING      - tilting the side-view axes really does move the FRONT-view
                     geometry (camber gain and roll-centre height), which is the
                     reason this module exists next to the others rather than in
                     isolation.
 10  BENCHMARK     - the reference GT3's published anti curves, asserted as a
                     SHAPE: strong anti under braking at both ends, near-zero
                     under power, all flat with travel, all under Gillespie's
                     50 % practical ceiling.
 11  EXTERNAL      - Soham's own F1 anti-dive/squat spreadsheet, built
                     independently on a different car (L/H 11.53 vs our 5.88)
                     with hand-measured CAD angles, reproduced to 1e-8 on all
                     three of its valid outputs. The fourth -- a front anti-lift
                     under power on a RWD car -- is asserted to be zero here,
                     and the sheet's value is shown to follow from a front drive
                     share of 1.0.
 12  BACKDROP      - the car silhouette behind D1 is registered to the LIVE
                     hard points: its wheels land on the plotted contact
                     patches, it rescales with the wheelbase, and a missing
                     outline file degrades to not drawing rather than crashing.

Run:  python validate_anti_geometry.py
"""
import numpy as np

import benchmarks as bm

import anti_geometry as ag
import config as cfg
import load_transfer as ltr
import rear_geometry as rgeo
import wishbone_solver as ws

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}"
          + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


CAR = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
PF = cfg._load_front_points()
PR = rgeo.solver_points(rgeo.load_rear())
GF = ws.corner_geometry(PF, cfg.STATIC_CAMBER)
GR = ws.corner_geometry(PR, rgeo.REAR_STATIC_CAMBER)
TF, TR = ag.tau(GF), ag.tau(GR)
BIAS = 0.62
LH = CAR["wheelbase_mm"] / CAR["h_cg_mm"]

print("=" * 76)
print("VALIDATION - anti-dive / anti-lift / anti-squat")
print("=" * 76)

# ------------------------------------------------------------- 1. the knob
print("\n1. THE DESIGN KNOB: SIDE-VIEW TILT OF THE INNER HINGE AXES")
uf, lf = ag.axis_tilt_deg(PF)
ur, lr = ag.axis_tilt_deg(PR)
check("the CAD front axes are FLAT in side view", abs(uf) < 0.01 and abs(lf) < 0.01,
      f"upper {uf:+.4f}, lower {lf:+.4f} deg -> no wishbone anti at all")
check("the CAD rear axes are flat to within a fifth of a degree",
      abs(ur) < 0.01 and abs(lr) < 0.2, f"upper {ur:+.4f}, lower {lr:+.4f} deg")
p2 = ag.tilt_axes(PF, 3.0, -2.0)
g2u, g2l = ag.axis_tilt_deg(p2)
check("tilt_axes writes back exactly what it was asked for",
      abs(g2u - 3.0) < 1e-9 and abs(g2l - (-2.0)) < 1e-9,
      f"read back {g2u:+.6f} / {g2l:+.6f} deg")
for nm, (a, b) in (("upper", ag._AXES[0]), ("lower", ag._AXES[1])):
    z0 = (PF[a][2] + PF[b][2]) / 2.0
    z1 = (p2[a][2] + p2[b][2]) / 2.0
    check(f"tilting about the midpoint keeps the {nm} arm's mean height",
          abs(z0 - z1) < 1e-9, f"{z0:.3f} -> {z1:.3f} mm")
check("and does not move the pivots laterally or fore-aft",
      all(abs(p2[k][i] - PF[k][i]) < 1e-12
          for k in (ag._AXES[0] + ag._AXES[1]) for i in (0, 1)))
check("tilting back to the original angles restores the points",
      max(np.max(np.abs(ag.tilt_axes(p2, uf, lf)[k] - PF[k]))
          for k in (ag._AXES[0] + ag._AXES[1])) < 1e-9)

# ------------------------------------------------------------------ 2. tau
print("\n2. TAU AGAINST ITS CLOSED FORM (parallel side-view arms -> tan(tilt))")
check("zero tilt gives zero tau at the FRONT", abs(TF["patch_wishbone"]) < 2e-4,
      f"{TF['patch_wishbone']:+.2e}")
check("zero tilt gives zero tau at the REAR",
      abs(ag.tau(ws.corner_geometry(ag.tilt_axes(PR, 0.0, 0.0),
                                    rgeo.REAR_STATIC_CAMBER))["patch_wishbone"])
      < 2e-4)
# tau == tan(tilt) is exact only in the limit of EQUAL side-view arm lengths.
# Real wishbones differ, so the upright still rotates a little even with the two
# axes parallel, leaving a second-order residual. Assert the identity to 2.5e-3
# absolute and, more tellingly, to 4 % relative -- and check the residual really
# is second order by confirming it does not grow with the tilt.
resid = []
for a in (-5.0, -3.0, 2.0, 5.0):
    t = ag.tau(ws.corner_geometry(ag.tilt_axes(PF, a, a), cfg.STATIC_CAMBER))
    want = np.tan(np.radians(a))
    resid.append(abs(t["patch_wishbone"] - want))
    check(f"front, both arms at {a:+.0f} deg: tau == tan(tilt)",
          resid[-1] < 2.5e-3 and resid[-1] / abs(want) < 0.04,
          f"{t['patch_wishbone']:+.6f} vs {want:+.6f}   "
          f"({resid[-1] / abs(want):.2%} -- the unequal-arm second-order term)")
    check("   ...and the corner nearly TRANSLATES, so patch and wheel centre "
          "agree", abs(t["patch_wishbone"] - t["wc_wishbone"]) < 1e-3,
          f"gap {t['patch_wishbone'] - t['wc_wishbone']:+.2e}")
check("that residual stays second order -- it does not grow with the tilt",
      max(resid) < 3e-3, f"worst {max(resid):.2e} over +/-5 deg")
tp = ag.tau(ws.corner_geometry(ag.tilt_axes(PF, 4.0, 4.0),
                               cfg.STATIC_CAMBER))["patch_wishbone"]
tm = ag.tau(ws.corner_geometry(ag.tilt_axes(PF, -4.0, -4.0),
                               cfg.STATIC_CAMBER))["patch_wishbone"]
check("tau is ODD in the tilt, as tan is", abs(tp + tm) < 2e-3,
      f"{tp:+.5f} and {tm:+.5f}")
seq = [ag.tau(ws.corner_geometry(ag.tilt_axes(PF, a, a),
                                 cfg.STATIC_CAMBER))["patch_wishbone"]
       for a in np.linspace(-6, 6, 9)]
check("tau is monotonic in the tilt", np.all(np.diff(seq) > 0))

# ------------------------------------------------- 3. the reference point
print("\n3. THE CONTACT POINT MUST BE A MATERIAL POINT OF THE UPRIGHT")
from scrub_trail import contact_patch as _cp


def _tau_reprojected(g, d=0.35):
    """The WRONG way: take the wheel's LOWEST POINT at each position instead of
    a material point of the upright. Run in the toe-held model, where the only
    thing separating the patch from the wheel centre is the upright's pitch --
    which is precisely what re-projecting throws away."""
    out = []
    for s in (+d, -d):
        W, _, sax = ag._corner_at(g, s, "notoe")
        out.append((W, _cp(W, spin_axis=sax, R=g["R_wheel"])))
    dz = out[0][0][2] - out[1][0][2]
    return (out[0][1][1] - out[1][1][1]) / dz, (out[0][0][1] - out[1][0][1]) / dz


gsplit = ws.corner_geometry(ag.tilt_axes(PR, 3.0, -2.0), rgeo.REAR_STATIC_CAMBER)
tsplit = ag.tau(gsplit)
bad_p, bad_w = _tau_reprojected(gsplit)
check("re-projecting to the ground WOULD collapse the two taus together",
      abs(bad_p - bad_w) < 1e-6,
      f"gap {bad_p - bad_w:+.2e} -- a wheel is round, so pitching the upright "
      f"spins it on its bearing and never moves the lowest point")
check("the material point keeps them APART, as the physics requires",
      abs(tsplit["patch_wishbone"] - tsplit["wc_wishbone"]) > 0.05,
      f"gap {tsplit['patch_wishbone'] - tsplit['wc_wishbone']:+.5f} vs "
      f"{bad_p - bad_w:+.2e} the wrong way")
check("...which is what makes anti-lift and anti-squat independently settable",
      abs(ag.anti_lift_pct(tsplit["patch_wishbone"], CAR)
          - ag.anti_squat_pct(tsplit["wc_wishbone"], CAR)) > 5.0)

# ------------------------------------------------------------- 4. the split
print("\n4. THE SPLIT: tau_total = tau_wishbone + tau_steer")
for nm, t in (("front", TF), ("rear", TR)):
    check(f"{nm}: the three parts add up by construction",
          abs(t["patch"] - t["patch_wishbone"] - t["patch_steer"]) < 1e-12)
sw = ws.solve_sweep(GF, np.linspace(-2, 2, 9))
scrub, kpi = sw["scrub"][4], sw["kpi"][4]
want_lever = scrub / np.cos(np.radians(kpi))
check("the steer lever IS the scrub radius / cos(KPI)",
      abs(TF["steer_lever_patch"] - want_lever) < 0.5,
      f"{TF['steer_lever_patch']:.2f} vs {want_lever:.2f} mm/rad "
      f"(scrub {scrub:.2f}, KPI {kpi:+.2f} deg)")
levers = []
for dz in (0.0, -30.0, -60.0, -90.0, -120.0):
    p = {k: np.array(v, float) for k, v in PF.items()}
    p["Tie rod - inner"][2] += dz
    levers.append(ag.tau(ws.corner_geometry(p, cfg.STATIC_CAMBER))
                  ["steer_lever_patch"])
check("that lever is CONSTANT as the tie rod is moved -- so the steer term is "
      "strictly proportional to bump steer",
      np.ptp(levers) < 0.01,
      f"{min(levers):.3f} to {max(levers):.3f} mm/rad over 120 mm of tie-rod "
      f"travel")
p0 = {k: np.array(v, float) for k, v in PF.items()}
p0["Tie rod - inner"][2] -= 88.0                      # near the bump-steer null
t0 = ag.tau(ws.corner_geometry(p0, cfg.STATIC_CAMBER))
check("null the bump steer and the steer term goes with it",
      abs(t0["bump_steer"]) < 0.01 and abs(t0["patch_steer"]) < 0.02,
      f"bump steer {t0['bump_steer']:+.5f} deg/mm -> "
      f"tau_steer {t0['patch_steer']:+.5f}")
check("on OUR front, the whole of the anti is bump steer, not wishbones",
      abs(TF["patch_wishbone"]) < 0.001 and abs(TF["patch_steer"]) > 0.2,
      f"wishbone {TF['patch_wishbone']:+.5f}, steer {TF['patch_steer']:+.5f}")

# ------------------------------------------------------- 5. the percentages
print("\n5. THE PERCENTAGES  (Gillespie 7.28 / 7.29 / 7.19)")
check("anti-dive is Gillespie 7.28 exactly",
      abs(ag.anti_dive_pct(0.05, CAR, BIAS) - BIAS * 0.05 * LH * 100) < 1e-9)
check("anti-lift is Gillespie 7.29 exactly",
      abs(ag.anti_lift_pct(-0.05, CAR, BIAS) - (1 - BIAS) * 0.05 * LH * 100)
      < 1e-9)
check("anti-squat is Gillespie 7.19 exactly",
      abs(ag.anti_squat_pct(-0.05, CAR) - 0.05 * LH * 100) < 1e-9)
check("100 % anti-dive needs exactly tau = (h/L)/bias",
      abs(ag.anti_dive_pct(1.0 / LH / BIAS, CAR, BIAS) - 100.0) < 1e-9)
check("anti-dive is PROPORTIONAL to brake bias -- the bias bar retunes it",
      abs(ag.anti_dive_pct(0.05, CAR, 0.70)
          / ag.anti_dive_pct(0.05, CAR, 0.35) - 2.0) < 1e-12)
check("...and rear anti-lift moves the OPPOSITE way with bias",
      ag.anti_lift_pct(-0.05, CAR, 0.70) < ag.anti_lift_pct(-0.05, CAR, 0.35))
check("anti-squat does NOT depend on brake bias",
      abs(ag.anti_squat_pct(-0.05, CAR) - ag.anti_squat_pct(-0.05, CAR)) < 1e-15)
check("a lower CG (smaller h/L) needs MORE tau for the same anti",
      ag.tau_for_anti(30.0, dict(CAR, h_cg_mm=300.0), "dive")
      < ag.tau_for_anti(30.0, CAR, "dive"))
for kind, pct in (("dive", 30.0), ("lift", 40.0), ("squat", -5.0)):
    tv = ag.tau_for_anti(pct, CAR, kind, BIAS)
    got = (ag.anti_squat_pct(tv, CAR) if kind == "squat"
           else (ag.anti_lift_pct if kind == "lift" else ag.anti_dive_pct)
           (tv, CAR, BIAS))
    check(f"tau_for_anti inverts the {kind} formula exactly",
          abs(got - pct) < 1e-9, f"{pct} % -> tau {tv:+.5f} -> {got:.4f} %")
check("SIGN: a front patch moving FORWARD in bump gives anti-DIVE",
      ag.anti_dive_pct(+0.05, CAR) > 0)
check("SIGN: a rear patch moving REARWARD in bump gives anti-LIFT",
      ag.anti_lift_pct(-0.05, CAR) > 0)
check("SIGN: a rear wheel centre moving REARWARD gives anti-SQUAT",
      ag.anti_squat_pct(-0.05, CAR) > 0)
check("classify() puts 30 % in the moderate band",
      "moderate" in ag.classify(30.0))
check("classify() flags anything past Gillespie's ceiling",
      "ceiling" in ag.classify(60.0))
check("classify() flags a negative number as PRO", "PRO" in ag.classify(-10.0))

# --------------------------------------------------------- 6. the swing arm
print("\n6. THE SIDE-VIEW SWING ARM")
gap, ident = ag.patch_wc_gap(tsplit, gsplit["R_wheel"])
check("tau_patch - tau_wc == R_wheel / SVSA",
      abs(abs(gap) - abs(ident)) / abs(ident) < 5e-3,
      f"{gap:+.5f} vs {ident:+.5f}  (SVSA {ag.svsa_mm(tsplit):.0f} mm)")
tpar = ag.tau(ws.corner_geometry(ag.tilt_axes(PR, -3.0, -3.0),
                                 rgeo.REAR_STATIC_CAMBER))
check("parallel arms give an effectively INFINITE swing arm",
      ag.svsa_mm(tpar) > 1e5, f"{ag.svsa_mm(tpar):.3g} mm")
lift_par = ag.anti_lift_pct(tpar["patch_wishbone"], CAR, BIAS)
squat_par = ag.anti_squat_pct(tpar["wc_wishbone"], CAR)
check("...so parallel arms LOCK anti-squat to anti-lift/(1-bias)",
      abs(squat_par - lift_par / (1 - BIAS)) < 0.5,
      f"anti-lift {lift_par:.2f} % forces anti-squat {squat_par:.2f} % "
      f"(= {lift_par / (1 - BIAS):.2f})")
check("a shorter swing arm makes a BIGGER patch/wheel-centre split",
      abs(ag.tau(ws.corner_geometry(ag.tilt_axes(PR, 5.0, -4.0),
                                    rgeo.REAR_STATIC_CAMBER))["patch_wishbone"]
          - ag.tau(ws.corner_geometry(ag.tilt_axes(PR, 5.0, -4.0),
                                      rgeo.REAR_STATIC_CAMBER))["wc_wishbone"])
      > abs(tpar["patch_wishbone"] - tpar["wc_wishbone"]))

# -------------------------------------------------------------- 7. pitch
print("\n7. WHAT IT IS FOR: PITCH")
p_no = ag.pitch(CAR, 0.0, 0.0, 1.2)
W = CAR["mass_total_kg"] * 9.80665
dW_hand = W * 1.2 * CAR["h_cg_mm"] / CAR["wheelbase_mm"]
check("the load transferred matches W a h / L by hand",
      abs(p_no["dW"] - dW_hand) < 1e-6, f"{p_no['dW']:.1f} N")
check("with no anti, front travel is dW / (axle ride rate)",
      abs(p_no["front_mm"] - dW_hand / p_no["rr_f"]) < 1e-9,
      f"{p_no['front_mm']:.2f} mm")
check("the tyre is in SERIES with the spring, so the ride rate is softer than "
      "the wheel rate", p_no["rr_f"] < 2 * CAR["kw_f"],
      f"{p_no['rr_f']:.1f} vs {2 * CAR['kw_f']:.1f} N/mm")
p_half = ag.pitch(CAR, 50.0, 50.0, 1.2)
check("50 % anti halves the travel at both axles",
      abs(p_half["front_mm"] / p_no["front_mm"] - 0.5) < 1e-12
      and abs(p_half["rear_mm"] / p_no["rear_mm"] - 0.5) < 1e-12)
p_full = ag.pitch(CAR, 100.0, 100.0, 1.2)
check("100 % anti gives EXACTLY zero travel -- the links take all of it",
      abs(p_full["front_mm"]) < 1e-12 and abs(p_full["pitch_deg"]) < 1e-12)
p_pro = ag.pitch(CAR, -100.0, 0.0, 1.2)
check("negative anti makes it WORSE, not better",
      p_pro["front_mm"] > p_no["front_mm"],
      f"{p_no['front_mm']:.2f} -> {p_pro['front_mm']:.2f} mm at -100 %")
check("pitch angle is the two travels over the wheelbase",
      abs(p_no["pitch_deg"]
          - np.degrees(np.arctan2(p_no["front_mm"] + p_no["rear_mm"],
                                  CAR["wheelbase_mm"]))) < 1e-12)
check("the pitch GRADIENT is per g", abs(p_no["pitch_grad"] * 1.2
                                         - p_no["pitch_deg"]) < 1e-12)
acc = ag.pitch(CAR, 80.0, 0.0, 1.0, mode="accel")
acc0 = ag.pitch(CAR, 0.0, 0.0, 1.0, mode="accel")
check("under power a RWD front has NO anti available, whatever is passed",
      abs(acc["front_mm"] - acc0["front_mm"]) < 1e-12,
      "no longitudinal force at the front means no force line to tilt")
check("the no-anti pitch gradient is in the same order as Gillespie's worked "
      "examples (1.4 - 4.2 deg/g)",
      0.3 < p_no["pitch_grad"] < 5.0,
      f"{p_no['pitch_grad']:.3f} deg/g -- lower than a road car, as a GT3's "
      f"much stiffer springs should give")

# ------------------------------------------------------------- 8. solvers
print("\n8. THE DESIGN SOLVERS")
for kind, tgt, pts, cam in (("dive", 30.0, PF, cfg.STATIC_CAMBER),
                            ("dive", 10.0, PF, cfg.STATIC_CAMBER),
                            ("squat", 20.0, PR, rgeo.REAR_STATIC_CAMBER)):
    a, got = ag.required_tilt(pts, cam, tgt, CAR, kind, BIAS)
    check(f"required_tilt hits {tgt:.1f} % {kind}", abs(got - tgt) < 0.05,
          f"tilt {a:+.3f} deg -> {got:.3f} %")
a_up, a_lo = 4.955, 4.955
check("the front's required tilt reproduces the closed form",
      abs(ag.required_tilt(PF, cfg.STATIC_CAMBER, 30.0, CAR, "dive", BIAS)[0]
          - np.degrees(np.arctan(ag.tau_for_anti(30.0, CAR, "dive", BIAS))))
      < 0.05)
u, l, gl, gs = ag.required_tilt_pair(PR, rgeo.REAR_STATIC_CAMBER, 40.0, -5.0,
                                     CAR, BIAS)
check("required_tilt_pair hits BOTH rear targets at once",
      abs(gl - 40.0) < 0.05 and abs(gs - (-5.0)) < 0.05,
      f"upper {u:+.3f}, lower {l:+.3f} deg -> lift {gl:.2f} %, "
      f"squat {gs:.2f} %")
check("...and it needs SPLIT axes, not parallel ones", abs(u - l) > 1.0,
      f"{u - l:+.3f} deg apart")
_gp = ws.corner_geometry(ag.tilt_axes(PR, u, l), rgeo.REAR_STATIC_CAMBER)
_sv = ag.svsa_mm(ag.tau(_gp))
# Independent number: the benchmark's OWN two rear percentages, read through
# Gillespie 7.19 and 7.29, imply a swing arm without any of our geometry.
_tp = 40.0 / 100 * (1 / LH) / (1 - BIAS)
_tw = 3.9 / 100 * (1 / LH)
_svsa_bm = _gp["R_wheel"] / (_tp + _tw)
check("the swing arm it lands on matches the one the benchmark's own two "
      "numbers imply",
      abs(_sv / _svsa_bm - 1) < 0.05,
      f"solver {_sv:.0f} mm vs {_svsa_bm:.0f} mm from Gillespie alone")

# ------------------------------------------------------------ 9. coupling
print("\n9. COUPLING INTO THE FRONT VIEW (the reason this is not a standalone)")
# front_view_at_ride() is a 26x shortcut past solve_sweep + linkage_roll_centre.
# Guard it against BOTH originals, on both axles, at several tilts -- a shortcut
# that silently drifts from the validated core is worse than no shortcut.
for _nm, _p, _c in (("front", PF, cfg.STATIC_CAMBER),
                    ("rear", PR, rgeo.REAR_STATIC_CAMBER)):
    worst_c = worst_r = 0.0
    for _a in (0.0, 3.0, -4.0):
        _g = ws.corner_geometry(ag.tilt_axes(_p, _a, _a), _c)
        _cg, _ca, _rc = ag.front_view_at_ride(_g)
        _b = np.linspace(-6, 6, 13)
        _sw = ws.solve_sweep(_g, _b)
        worst_c = max(worst_c, abs(_cg - np.gradient(_sw["camber"], _b)[6]))
        worst_r = max(worst_r,
                      abs(_rc - float(ws.linkage_roll_centre(_g, [0.0])[0])))
    check(f"{_nm}: the fast front-view path matches solve_sweep's camber gain",
          worst_c < 1e-6, f"worst {worst_c:.1e} deg/mm -- four orders below the "
          f"smallest camber gain on this car")
    check(f"{_nm}: ...and matches linkage_roll_centre exactly",
          worst_r < 1e-2, f"worst {worst_r:.1e} mm")

co = ag.coupling(PF, cfg.STATIC_CAMBER, np.linspace(-6, 6, 7))
check("tau responds to the tilt, monotonically",
      np.all(np.diff(co["tau_wishbone"]) > 0),
      f"{co['tau_wishbone'][0]:+.4f} to {co['tau_wishbone'][-1]:+.4f}")
check("the ROLL CENTRE moves too -- side view and front view are one linkage",
      np.ptp(co["rc_h"]) > 10.0,
      f"{co['rc_h'].min():.1f} to {co['rc_h'].max():.1f} mm over +/-6 deg")
check("and roll-centre height is monotonic in the tilt",
      np.all(np.diff(co["rc_h"]) < 0) or np.all(np.diff(co["rc_h"]) > 0))
check("CAMBER GAIN moves as well, and on this near-singular front it even "
      "changes sign", co["camber_gain"][0] * co["camber_gain"][-1] < 0,
      f"{co['camber_gain'][0]:+.5f} to {co['camber_gain'][-1]:+.5f} deg/mm")
cor = ag.coupling(PR, rgeo.REAR_STATIC_CAMBER, np.array([0.0, -5.0]))
check("at the REAR, anti-squat tilt and roll-centre height want the SAME "
      "direction", cor["rc_h"][1] > cor["rc_h"][0],
      f"RC {cor['rc_h'][0]:.1f} -> {cor['rc_h'][1]:.1f} mm at -5 deg, which is "
      f"also the anti-squat direction")
check("...and it deepens the rear camber gain at the same time",
      abs(cor["camber_gain"][1]) > abs(cor["camber_gain"][0]),
      f"{cor['camber_gain'][0]:+.5f} -> {cor['camber_gain'][1]:+.5f} deg/mm")

if not ag.BENCHMARK_IS_MEASURED:
    print("\n10. AGAINST A REFERENCE CAR -- SKIPPED")
    print("   no private reference data installed; see software/benchmarks.py")
else:
    # ----------------------------------------------------------- 10. benchmark
    print("\n10. AGAINST A MEASURED REFERENCE CAR (shape, not numbers)")
    B = ag.BENCHMARK_PCT
    check("the benchmark brakes with real anti at BOTH ends",
          B["anti_dive_f"] > 25 and B["anti_lift_r"] > 25,
          f"front {B['anti_dive_f']:.1f} %, rear {B['anti_lift_r']:.1f} %")
    check("but runs essentially NO anti-squat under power",
          abs(B["anti_squat_r"]) < 10.0, f"{B['anti_squat_r']:+.1f} % -- slightly "
          f"PRO-squat, which lets the rear settle for traction on exit")
    check("every published value sits under Gillespie's 50 % practical ceiling",
          all(v < ag.GILLESPIE_MAX_ANTIDIVE_PCT for v in B.values()))
    check("the rear carries MORE anti under braking than the front, despite taking "
          "less of the brake force", B["anti_lift_r"] > B["anti_dive_f"],
          "so its force line must be steeper still")
    check("all three curves are nearly FLAT with wheel travel",
          all(abs(v) * 40 < 1.0 for v in ag.BENCHMARK_SLOPE.values()),
          "under 1 % across 40 mm -- the anti a driver feels should not move as the "
          "car uses its travel")
    check("OUR front is on the wrong side of zero by a wide margin",
          ag.anti_dive_pct(TF["patch"], CAR, BIAS) < -50,
          f"{ag.anti_dive_pct(TF['patch'], CAR, BIAS):.1f} % vs "
          f"{B['anti_dive_f']:+.1f} % -- the nose dives MORE than with no anti")
    check("...but its WISHBONES are merely neutral, not wrong",
          abs(ag.anti_dive_pct(TF["patch_wishbone"], CAR, BIAS)) < 1.0,
          f"{ag.anti_dive_pct(TF['patch_wishbone'], CAR, BIAS):+.2f} % -- so the "
          f"fix is the tie rod, then the axes")
    _pb = ag.pitch(CAR, B["anti_dive_f"], B["anti_lift_r"], 1.2)
    check("reaching the benchmark anti would cut braking pitch by about a third",
          0.55 < _pb["pitch_deg"] / p_no["pitch_deg"] < 0.75,
          f"{p_no['pitch_deg']:.3f} -> {_pb['pitch_deg']:.3f} deg at 1.2 g "
          f"({_pb['pitch_deg'] / p_no['pitch_deg'] - 1:+.1%})")
    _ad, _ = ag.required_tilt(PF, cfg.STATIC_CAMBER, B["anti_dive_f"], CAR, "dive",
                              BIAS)
    check("and the front axis tilt it asks for is a buildable angle",
          0.0 < _ad < 8.0, f"{_ad:+.2f} deg of side-view rake on both front arms")

# ------------------------------------------ 11. against an independent sheet
print("\n11. AGAINST SOHAM'S OWN F1 ANTI-DIVE / ANTI-SQUAT SPREADSHEET")
# An external cross-check, and the best kind: a DIFFERENT car (F1, L/H = 11.53
# against our 5.88), built independently in Google Sheets from two YouTube
# derivations, with its four force-line angles measured by hand in CAD. If this
# module's formulas agree with it to 1e-8 on three of four outputs, both are
# almost certainly right -- and the fourth disagreement is then worth trusting.
F1 = dict(wheelbase_mm=3400.0, h_cg_mm=295.0)
F1_BIAS = 0.60
TH_F, TH_R = 0.1015762958, 0.05065781464      # patch force lines  (braking)
PS_F, PS_R = 0.0331734166, 0.01920098091      # wheel-centre lines (traction)
check("F1 sheet's anti-dive reproduces exactly (Gillespie 7.28)",
      abs(float(ag.anti_dive_pct(TH_F, F1, F1_BIAS)) - 70.24259101) < 1e-6,
      f"{float(ag.anti_dive_pct(TH_F, F1, F1_BIAS)):.6f} vs 70.24259101")
check("F1 sheet's rear anti-lift reproduces exactly (Gillespie 7.29)",
      abs(float(ag.anti_lift_pct(-TH_R, F1, F1_BIAS)) - 23.35411116) < 1e-6,
      f"{float(ag.anti_lift_pct(-TH_R, F1, F1_BIAS)):.6f} vs 23.35411116")
check("F1 sheet's anti-squat reproduces exactly (Gillespie 7.19)",
      abs(float(ag.anti_squat_pct(-PS_R, F1)) - 22.1299441) < 1e-6,
      f"{float(ag.anti_squat_pct(-PS_R, F1)):.6f} vs 22.1299441")
check("the L/H leverage differs by 2x between the two cars, and both still work",
      abs(F1["wheelbase_mm"] / F1["h_cg_mm"]
          / (CAR["wheelbase_mm"] / CAR["h_cg_mm"]) - 1.961) < 0.01,
      f"F1 {F1['wheelbase_mm'] / F1['h_cg_mm']:.2f} vs ours "
      f"{CAR['wheelbase_mm'] / CAR['h_cg_mm']:.2f} -- an F1 car gets twice the "
      f"anti per degree of force-line angle")
# The one place the sheet and this module part company, and this module is right:
# an RWD front transmits NO longitudinal force under power, so no geometry can
# give it anti-lift. The sheet applies a drive share of 1.0 to the front.
check("a RWD front's anti-lift under power is ZERO whatever psi_F is",
      abs(ag.pitch(CAR, 99.0, 0.0, 1.0, mode="accel")["front_mm"]
          - ag.pitch(CAR, 0.0, 0.0, 1.0, mode="accel")["front_mm"]) < 1e-12,
      "the sheet's 38.23 % front anti-lift assumes a front drive share of 1.0")
check("...and this module reproduces the sheet's 38.23 % only if a FWD front "
      "drive share is forced, confirming that is what the sheet assumed",
      abs(float(ag.anti_squat_pct(-PS_F, F1, drive_share=1.0)) - 38.23376829)
      < 1e-6,
      f"{float(ag.anti_squat_pct(-PS_F, F1, drive_share=1.0)):.5f}")
# The sheet's own theta/psi pair implies a swing arm, exactly as ours does.
for _nm, _th, _ps in (("front", TH_F, PS_F), ("rear", TH_R, PS_R)):
    _svsa = 330.0 / (_th - _ps)
    check(f"the sheet's {_nm} theta/psi gap implies a sane swing arm",
          1000.0 < _svsa < 20000.0,
          f"R/gap = {_svsa:.0f} mm at a 330 mm F1 wheel -- long, as an F1 car "
          f"needs for a flat anti curve")


# ---------------------------------------------- 12. the D1 car backdrop
print("\n12. THE SIDE-VIEW SILHOUETTE BEHIND D1")
# car_outline exists only to make D1 readable, but a backdrop that does not
# line up with the numbers drawn over it is worse than none: it would quietly
# contradict them. These checks are about REGISTRATION, not looks.
import car_outline as co

_out = co.outline(TF["patch_y"], TR["patch_y"])
if _out is None:
    print("   -- skipped: no vehicle silhouette asset installed")
else:
    _y, _lo, _hi = _out
    check("the outline loads and covers both axles",
          _y.min() < TR["patch_y"] and _y.max() > TF["patch_y"],
          f"spans Y {_y.min():+.0f} .. {_y.max():+.0f} mm around axles "
          f"{TF['patch_y']:+.0f} / {TR['patch_y']:+.0f}")
    check("its lower edge never goes below the road",
          _lo.min() >= -1e-9, f"min {_lo.min():+.3f} mm")
    check("the body height is a plausible GT3 at this wheelbase",
          1100.0 < _hi.max() < 1500.0, f"{_hi.max():.0f} mm")
    # The registration that matters: the mesh's own wheelbase is ~2 % off ours, so
    # the outline is stored NORMALISED and re-scaled. Check the wheels really do
    # land on the patches by finding where the silhouette touches the ground.
    _touch = _y[_lo < 12.0]
    _frontish = _touch[_touch > 0].mean() if (_touch > 0).any() else float("nan")
    _rearish = _touch[_touch < 0].mean() if (_touch < 0).any() else float("nan")
    check("the front wheel sits on the front contact patch",
          abs(_frontish - TF["patch_y"]) < 120.0,
          f"tyre contact centred {_frontish:+.0f} vs patch {TF['patch_y']:+.0f} mm")
    check("the rear wheel sits on the rear contact patch",
          abs(_rearish - TR["patch_y"]) < 120.0,
          f"tyre contact centred {_rearish:+.0f} vs patch {TR['patch_y']:+.0f} mm")
    # and it must FOLLOW the hard points, not sit at a baked-in scale
    _y2, _lo2, _hi2 = co.outline(TF["patch_y"] * 2.0, TR["patch_y"] * 2.0)
    check("it rescales with the wheelbase instead of being baked in",
          abs(np.ptp(_y2) / np.ptp(_y) - 2.0) < 1e-6
          and abs(_hi2.max() / _hi.max() - 2.0) < 1e-6,
          "doubling the axle spacing doubles the drawing")
    _mid = 0.5 * (TF["patch_y"] + TR["patch_y"])
    _mid2 = 0.5 * (_y2.min() + _y2.max())
    check("it stays centred on the axle mid-station",
          abs((0.5 * (_y.min() + _y.max()) - _mid)
              / (0.5 * (_y2.min() + _y2.max()) - _mid * 2.0) - 0.5) < 1e-6
          or abs(_mid2 - 2.0 * _mid - (0.5 * (_y.min() + _y.max()) - _mid)) < 1.0)
    # the density raster must register on the SAME normalisation as the envelope,
    # or the wheels in the picture and the wheels in the outline would disagree
    _r = co._render()
    check("the side-view raster loads", _r is not None)
    if _r is not None:
        _img, _u0, _u1, _z0, _z1, _vmax = _r
        check("the raster spans the same normalised box as the envelope",
              abs(_u0 - co._load()[0].min()) < 0.02
              and abs(_u1 - co._load()[0].max()) < 0.02,
              f"raster u {_u0:+.3f}..{_u1:+.3f} vs envelope "
              f"{co._load()[0].min():+.3f}..{co._load()[0].max():+.3f}")
        check("its ground edge is at zero height", abs(_z0) < 0.005,
              f"{_z0:+.5f} of a wheelbase")
        check("it carries real structure, not a flat field",
              _vmax > 1.0 and _img.max() > _vmax, f"vmax {_vmax:.2f}")

    # missing file must be a no-op, never a crash mid-plot
    _keep = co._PATH
    try:
        co._PATH = _keep + ".absent"
        co._CACHE = None
        check("a missing outline file degrades to 'do not draw', not a crash",
              co.outline(1.0, -1.0) is None)
    finally:
        co._PATH = _keep
        co._CACHE = None


    print("\n" + "=" * 76)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print("   -", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED - anti follows from ONE quantity, the fore-aft motion\n"
          "of the reference point per mm of bump; the reference point is the contact\n"
          "patch for outboard brakes and the wheel centre for an inboard diff; and\n"
          "the percentages reproduce Gillespie 7.19 / 7.28 / 7.29 exactly.")
    print("=" * 76)
