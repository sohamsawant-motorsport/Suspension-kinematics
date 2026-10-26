# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the anti-roll bar model.
=============================================

  1  SECTIONS     - polar J against the closed forms, the solid/tube limits, and
                    the r^4 shell decomposition that justifies going hollow.
  2  CHAIN        - every step inverts exactly; the forward and backward paths
                    agree; and the chain is cross-checked against the INDEPENDENT
                    roll-stiffness routine already in load_transfer.
  3  STRESS       - the identity tau = G r phi / L is asserted the hard way: two
                    bars of the same OD and wildly different wall must show the
                    SAME surface stress. This is the counter-intuitive result the
                    whole hollow-vs-solid argument rests on.
  4  SIZING       - max_od_for_stress and wall_for_stiffness invert their
                    forward functions, and the impossible cases are reported
                    rather than silently returning a wrong number.
  5  BLADE        - the TAPER integrals against their Castigliano closed forms;
                    the prismatic case reduces exactly to 3EI/l^3; and the angle
                    law is the COMPLIANCE blend, not the arithmetic one. The
                    arithmetic blend is asserted to be wrong by ~7x at 45 deg --
                    that was a real bug, and both forms agree at 0 and 90 so only
                    the intermediate blade positions exposed it.
  5b FE + ENERGY  - the tapered blade is re-solved by a FINITE-ELEMENT
                    Euler-Bernoulli beam (2-node elements, I per element), which
                    shares no algebra with the Castigliano integral; and the
                    angle law is re-derived from STRAIN ENERGY. Two independent
                    routes to the same numbers.
  5c POSITIONS    - the chart's sections come from the TABLE's own diameters at
                    their nearest standard wall, and the 6-position span lands in
                    the benchmark's 2.2-2.9x band.
  6  TARGETS      - the Newton solve genuinely hits the roll-gradient and LLTD
                    targets when fed back through the load-transfer model.
  7  HOLLOW GAIN  - the same-mass and same-stiffness comparisons come out the way
                    the physics says they must.

Run:  python validate_arb.py
"""
import numpy as np

import arb
import load_transfer as ltr

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("=" * 76)
print("VALIDATION - anti-roll bars")
print("=" * 76)

# ------------------------------------------------------------- 1. sections
print("\n1. SECTION PROPERTIES")
check("solid J == pi d^4 / 32",
      abs(float(arb.polar_J(50.0)) - np.pi * 0.050 ** 4 / 32) < 1e-18,
      f"{float(arb.polar_J(50.0)):.4e} m^4")
check("tube J == pi(do^4 - di^4)/32",
      abs(float(arb.polar_J(50.0, 3.0))
          - np.pi * (0.050 ** 4 - 0.044 ** 4) / 32) < 1e-18)
check("wall = radius reduces the tube to the SOLID bar",
      abs(float(arb.polar_J(50.0, 25.0)) - float(arb.polar_J(50.0))) < 1e-20)
walls = np.array([2.0, 3.0, 5.0, 10.0, 25.0])
check("J rises monotonically with wall thickness",
      np.all(np.diff(arb.polar_J(50.0, walls)) > 0))
check("a tube is always softer than the solid bar of the same OD",
      float(arb.polar_J(50.0, 3.0)) < float(arb.polar_J(50.0)))
# the r^4 argument, stated as a number
outer_fifth = 1.0 - float(arb.polar_J(40.0)) / float(arb.polar_J(50.0))
check("the outer fifth of a 50 mm bar carries ~59 % of J",
      0.55 < outer_fifth < 0.62, f"{outer_fifth * 100:.1f} %")
core = float(arb.polar_J(30.0)) / float(arb.polar_J(50.0))
check("the inner 30 mm carries only ~13 % of J", core < 0.15,
      f"{core * 100:.1f} % of the stiffness")

# ---------------------------------------------------------------- 2. chain
print("\n2. THE CHAIN, and its inverse")
J = float(arb.polar_J(45.0, 4.0))
L, a, MR, TRACK = 1000.0, 350.0, 0.9908, 1662.4
Kt = float(arb.torsional_stiffness(J, L))
k_arm = float(arb.rate_at_arm(Kt, a))
k_wh = float(arb.to_wheel(k_arm, MR))
check("K_t = G J / L", abs(Kt - arb.G_STEEL * J / (L / 1000.0)) < 1e-6,
      f"{Kt:.1f} Nm/rad")
check("k_arm = 2 K_t / a^2   (DIVIDED by a^2, and the 2 is the both-ends-move "
      "factor)",
      abs(k_arm - 2.0 * Kt / (a / 1000.0) ** 2 / 1000.0) < 1e-9,
      f"{k_arm:.1f} N/mm")
# THE FACTOR 2, proved three ways -- this was a real bug (the roll stiffness came
# out exactly half). Route 1 follows the forces; route 2 is strain energy; route
# 3 is OptimumG "Bar talk" Eq.5, which pointedly has no /2 where Eq.2 (springs)
# does.
_phi = np.radians(1.0)
_Phi = 2.0 * (MR * (TRACK / 2000.0) * _phi) / (a / 1000.0)      # relative twist
_K1 = (MR * (Kt * _Phi / (a / 1000.0)) * (TRACK / 1000.0)) / _phi   # force chain
_K2 = 2.0 * (0.5 * Kt * _Phi ** 2) / _phi ** 2                      # strain energy
_K3 = MR ** 2 * Kt * (TRACK / 1000.0) ** 2 / (a / 1000.0) ** 2      # closed form
_Kc = float(arb.roll_stiffness(arb.to_wheel(k_arm, MR), TRACK)) / 1000.0
check("ARB roll stiffness: force chain == strain energy", abs(_K1 - _K2) < 1e-6,
      f"{_K1:.0f} Nm/rad")
check("...== the closed form MR^2 K_t t^2 / a^2", abs(_K1 - _K3) < 1e-6)
check("...== what the code's chain produces  (the factor-2 bug is gone)",
      abs(_K1 - _Kc) / _K1 < 1e-9, f"code {_Kc:.0f} vs derivation {_K1:.0f}")
# Eq.5's Karb is the ONE-END-FIXED rate K_t/a^2 -- it supplies the both-ends
# factor by omitting the /2 that Eq.2 has. Our k_arm already carries the 2, so
# it is halved here to feed Eq.5 on Eq.5's own terms.
_eq5 = ((TRACK / 1000.0) ** 2 * np.tan(np.radians(1.0))
        * (k_arm / 2.0 * 1000 * MR ** 2))
check("...and matches OptimumG Bar talk Eq.5 (no /2 for the ARB)",
      abs(_eq5 * 180 / np.pi - _K1) / _K1 < 2e-4,
      f"Eq.5 {_eq5:.1f} Nm/deg vs {_K1 * np.pi / 180:.1f}")
check("required_Kt inverts the whole forward chain",
      abs(float(arb.required_Kt(k_wh, a, MR)) - Kt) < 1e-6)
check("roll stiffness matches load_transfer's own routine",
      abs(float(arb.roll_stiffness(k_wh, TRACK))
          - (ltr.roll_stiffness(0.0, k_wh, TRACK))) < 1e-6,
      "cross-module agreement")
check("doubling the arm quarters the wheel rate",
      abs(float(arb.to_wheel(arb.rate_at_arm(Kt, 2 * a), MR)) - k_wh / 4.0) < 1e-9)
check("doubling MR quadruples the wheel rate",
      abs(float(arb.to_wheel(k_arm, 2 * MR)) - 4 * k_wh) < 1e-9)

# --------------------------------------------------------------- 3. stress
print("\n3. STRESS  -  tau = G r phi / L, so J CANCELS")
tw = float(arb.bar_twist_rad(3.0, TRACK, a, MR))
check("bar twist scales linearly with body roll",
      abs(float(arb.bar_twist_rad(6.0, TRACK, a, MR)) - 2 * tw) < 1e-12,
      f"{np.degrees(tw):.1f} deg of bar twist at 3 deg of body roll")

# THE HEADLINE. shear_stress_MPa has no wall argument at all, so asserting it
# against itself would prove nothing. Compute the stress the LONG way instead --
# tau = T r / J with T = G J phi / L, both of which DO depend on wall -- across
# a 10:1 range of wall thickness, and show the answer never moves.
s3 = float(arb.shear_stress_MPa(50.0, tw, L))
long_way = []
for w in (1.0, 2.0, 5.0, 10.0, 25.0):          # 25 mm = solid
    Jw = float(arb.polar_J(50.0, w))
    T = arb.G_STEEL * Jw / (L / 1000.0) * tw   # torque DOES change with wall
    long_way.append((w, T, T * 0.025 / Jw / 1e6))
check("torque genuinely varies with wall (so this is not a trivial check)",
      long_way[-1][1] / long_way[0][1] > 3.0,
      f"{long_way[0][1]:.0f} Nm at 1 mm -> {long_way[-1][1]:.0f} Nm solid")
check("yet SURFACE STRESS is identical across all of them  (J cancels)",
      max(abs(t - s3) for _w, _T, t in long_way) < 1e-6,
      f"{s3:.1f} MPa at every wall from 1 mm to solid")
check("stress rises with OUTER RADIUS",
      float(arb.shear_stress_MPa(60.0, tw, L))
      > float(arb.shear_stress_MPa(40.0, tw, L)))
check("a LONGER bar is less stressed at the same twist",
      float(arb.shear_stress_MPa(50.0, tw, 2 * L)) < s3 / 1.9)

# --------------------------------------------------------------- 4. sizing
print("\n4. SIZING INVERSES")
for tau in arb.ALLOWABLE_TAU_MPA:
    od = float(arb.max_od_for_stress(tau, tw, L))
    check(f"max_od_for_stress inverts at {tau:.0f} MPa",
          abs(float(arb.shear_stress_MPa(od, tw, L)) - tau) < 1e-6,
          f"OD {od:.1f} mm")
for od in (38.0, 45.0, 50.0):
    w = float(arb.wall_for_stiffness(od, 15314.0, L))
    got = float(arb.torsional_stiffness(arb.polar_J(od, w), L))
    check(f"wall_for_stiffness inverts at OD {od:.0f}",
          abs(got - 15314.0) < 1.0, f"wall {w:.2f} mm -> K_t {got:.0f} Nm/rad")
check("an impossible target (solid already too soft) returns NaN",
      not np.isfinite(float(arb.wall_for_stiffness(22.0, 1e9, L))))
rows = arb.size_bar(15314.0, L, 3.0, TRACK, a, MR, tau_allow_MPa=450.0)
check("size_bar explains every rejection in words",
      all(r["verdict"] for r in rows) and any(r["ok"] for r in rows),
      f"{sum(r['ok'] for r in rows)} of {len(rows)} ODs usable")
check("...and the usable ones really are under the stress limit",
      all(r["tau_MPa"] <= 450.0 + 1e-9 for r in rows if r["ok"]))

# ---------------------------------------------------------------- 5. blade
print("\n5. BLADE  -  taper, the angle law, and series")
b, h, ell = 40.0, 8.0, 230.0


def F_cf(r):
    return (r ** 2 * np.log(r) - 1.5 * r ** 2 + 2 * r - 0.5) / (r - 1.0) ** 3


def G_cf(r):
    return (np.log(r) + 1.5 + r ** 2 / 2 - 2 * r) / (r - 1.0) ** 3


for r in (0.25, 0.5, 0.75, 1.5, 2.0, 4.0):
    check(f"taper factor, weak axis, tip/root {r}",
          abs(arb.taper_factor(r, False) - 3.0 * F_cf(r)) < 2e-6,
          f"{arb.taper_factor(r, False):.6f} vs closed form {3 * F_cf(r):.6f}")
    check(f"taper factor, strong axis, tip/root {r}",
          abs(arb.taper_factor(r, True) - 3.0 * G_cf(r)) < 2e-6)
check("tip == root gives exactly the prismatic case (factor 1)",
      abs(arb.taper_factor(1.0, False) - 1.0) < 1e-9
      and abs(arb.taper_factor(1.0, True) - 1.0) < 1e-9)

kw, ks = arb.blade_principal_rates(b, b, h, ell)
kw_txt = 3 * arb.E_STEEL * (0.040 * 0.008 ** 3 / 12) / (ell / 1000) ** 3 / 1000
ks_txt = 3 * arb.E_STEEL * (0.008 * 0.040 ** 3 / 12) / (ell / 1000) ** 3 / 1000
check("prismatic weak axis == 3E(bh^3/12)/l^3", abs(kw - kw_txt) < 1e-6,
      f"{kw:.2f} N/mm")
check("prismatic strong axis == 3E(hb^3/12)/l^3", abs(ks - ks_txt) < 1e-6)
check("prismatic range == (b/h)^2", abs(ks / kw - (b / h) ** 2) < 1e-9,
      f"{(b / h) ** 2:.0f}:1")

kwt, kst = arb.blade_principal_rates(b, 20.0, h, ell)
check("taper softens both axes", kwt < kw and kst < ks)
check("taper hits the STRONG axis harder (b^3 vs b)", (kst / ks) < (kwt / kw),
      f"strong {kst / ks:.3f}x vs weak {kwt / kw:.3f}x of prismatic")
check("...so tapering SHRINKS the adjustment range", (kst / kwt) < (ks / kw),
      f"{ks / kw:.1f}:1 -> {kst / kwt:.1f}:1")

th = np.linspace(0, 90, 19)
kb = arb.blade_rate_at_angle(th, kw, ks)
check("k(0) == k_weak  (blade flat)", abs(float(kb[0]) - kw) < 1e-9)
check("k(90) == k_strong  (edge on)", abs(float(kb[-1]) - ks) < 1e-9)
check("k(theta) is NEVER zero", np.all(kb > 0), f"min {float(kb.min()):.2f} N/mm")
check("k(theta) rises monotonically flat -> edge", np.all(np.diff(kb) > 0))
comp = np.cos(np.radians(th)) ** 2 / kw + np.sin(np.radians(th)) ** 2 / ks
check("COMPLIANCE is exactly cos^2/k_weak + sin^2/k_strong",
      np.max(np.abs(1.0 / kb - comp)) < 1e-12)
arith = np.cos(np.radians(th)) ** 2 * kw + np.sin(np.radians(th)) ** 2 * ks
i45 = int(np.argmin(np.abs(th - 45.0)))
check("the arithmetic blend would overstate 45 deg by ~7x",
      6.0 < arith[i45] / kb[i45] < 8.0,
      f"{arith[i45] / kb[i45]:.1f}x -- agrees at 0 and 90, wrong in between")
check("...and the two agree exactly at BOTH ends",
      abs(arith[0] - kb[0]) < 1e-9 and abs(arith[-1] - kb[-1]) < 1e-9)

pos = arb.blade_positions(6)
check("6 positions span flat to edge-on",
      len(pos) == 6 and abs(pos[0]) < 1e-12 and abs(pos[-1] - 90.0) < 1e-12,
      f"{np.round(pos, 1).tolist()} deg")
check("positions are evenly spaced in angle",
      np.allclose(np.diff(pos), pos[1] - pos[0]))

k1, k2 = 1455.0, 1475.0
check("series is softer than EITHER part", arb.series(k1, k2) < min(k1, k2))
check("series reciprocals add",
      abs(1 / float(arb.series(k1, k2)) - (1 / k1 + 1 / k2)) < 1e-12)
rng = float(arb.series(k_arm, ks) / arb.series(k_arm, kw))
check("a stiff bar COLLAPSES the blade's adjustment range",
      rng < 0.5 * (ks / kw),
      f"blade alone {ks / kw:.0f}:1 -> in series only {rng:.2f}:1")

# --- FE CROSS-CHECK of the tapered blade. Euler-Bernoulli 2-node elements with
# I evaluated per element. Shares no algebra with the Castigliano integral, so
# agreement is evidence rather than a restatement of the same derivation.
def _fe_tip(b_root, b_tip, h_, ell_, cube, n_el=400, E=arb.E_STEEL):
    b_r, b_t, hh, LL = b_root / 1000, b_tip / 1000, h_ / 1000, ell_ / 1000
    le = LL / n_el
    n = 2 * (n_el + 1)
    K = np.zeros((n, n))
    for e in range(n_el):
        bb = b_r + (b_t - b_r) * ((e + 0.5) * le) / LL
        I = (hh * bb ** 3 / 12) if cube else (bb * hh ** 3 / 12)
        c = E * I / le ** 3
        ke = c * np.array([[12, 6 * le, -12, 6 * le],
                           [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                           [-12, -6 * le, 12, -6 * le],
                           [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        d_ = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d_, d_)] += ke
    F = np.zeros(n - 2)
    F[-2] = 1.0
    return 1.0 / np.linalg.solve(K[2:, 2:], F)[-2] / 1000.0     # N/mm


for b_tip in (40.0, 25.0, 20.0, 60.0):
    kw_i, ks_i = arb.blade_principal_rates(b, b_tip, h, ell)
    for nm, got, cube in (("weak", kw_i, False), ("strong", ks_i, True)):
        fe = _fe_tip(b, b_tip, h, ell, cube)
        check(f"FE beam agrees, {nm} axis, tip {b_tip:.0f} mm",
              abs(got - fe) / fe < 2e-3,
              f"{got:.3f} vs FE {fe:.3f} N/mm  ({abs(got - fe) / fe * 100:.3f} %)")

# --- the angle law by STRAIN ENERGY, an independent route to the same law
for th_ in (0.0, 30.0, 45.0, 60.0, 90.0):
    t_ = np.radians(th_)
    U = 0.5 * np.cos(t_) ** 2 / kw + 0.5 * np.sin(t_) ** 2 / ks
    check(f"strain-energy route reproduces k({th_:.0f} deg)",
          abs(1.0 / (2 * U) - float(arb.blade_rate_at_angle(th_, kw, ks))) < 1e-9,
          f"{1.0 / (2 * U):.3f} N/mm")

print("\n5c. POSITION CHART  (the usual blade-rate chart format)")
# DERIVED, not hard-coded: a literal here silently went stale when the factor-2
# fix halved the bar needed, and the spans blew out to 3.4-4.2x before this
# caught it. Anything the chain can compute, the test should compute too.
Kt_need = float(arb.required_Kt(122.7, 350.0, 1.0207))
tw2 = float(arb.bar_twist_rad(3.0, 1662.4, 350.0, 1.0207))
br = arb.bracket_sections(Kt_need, 1000.0, tw2, 3.0, 550.0)
check("two real sections BRACKET the requirement",
      len(br) == 2 and br[0]["Kt"] <= Kt_need <= br[1]["Kt"],
      " and ".join(f"{c['od_mm']:.0f}x{c['wall_mm']:.0f} ({c['ratio']:.2f}x)"
                   for c in br))
check("both bracketing sections pass the stress limit",
      all(c["tau_MPa"] <= 550.0 for c in br))
thp, curves = arb.rate_vs_position([(c["od_mm"], c["wall_mm"]) for c in br],
                                   arb.BLADE_DEFAULT, 350.0, 1.0207, 1000.0, 6)
check("one curve per section, 6 points each",
      len(curves) == 2 and all(len(v) == 6 for v in curves.values()))
for lab, v in curves.items():
    check(f"{lab}: rate rises with position", np.all(np.diff(v) > 0))
    # The span depends on the BLADE, whose dimensions are assumed (suppliers do
    # not publish them), so this is a sanity band rather than a tight assertion:
    # wide enough to admit a plausible blade, tight enough to catch a chain that
    # has gone wrong by a factor.
    check(f"{lab}: span is benchmark-like (2.0-2.9x)", 2.0 <= v[-1] / v[0] <= 2.9,
          f"{v[-1] / v[0]:.2f}x   (benchmark front 2.42x, rear 2.23-2.80x)")
_vals = list(curves.values())
check("the two sections do NOT overlay  (the point of bracketing)",
      abs(_vals[0][-1] - _vals[1][-1]) > 5.0)
check("lightest_passing sorts by mass",
      [r["od_mm"] for r in arb.lightest_passing(
          [dict(ok=True, mass_kg_m=m, od_mm=o)
           for o, m in ((40, 4.6), (38, 5.8), (45, 3.2))], 2)] == [45, 40])


# -------------------------------------------------------------- 6. targets
print("\n6. TARGET SOLVE")
car = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
for grad, lltd in ((0.60, 50.0), (0.80, 55.0), (0.50, 48.0)):
    af, ar, info = arb.solve_for_targets(car, -283.6, -52.0, grad, lltd)
    c = dict(car)
    c["arb_f"], c["arb_r"] = af, ar
    c["k_roll_f"] = ltr.roll_stiffness(c["kw_f"], af, c["track_f_mm"])
    c["k_roll_r"] = ltr.roll_stiffness(c["kw_r"], ar, c["track_r_mm"])
    t = ltr.transfer(1.0, -283.6, -52.0, c)
    check(f"targets {grad} deg/g and {lltd} % LLTD are actually hit",
          abs(t["roll_deg"] - grad) < 1e-6 and abs(t["lltd_f"] - lltd) < 1e-6,
          f"got {t['roll_deg']:.4f} deg/g, {t['lltd_f']:.3f} %  "
          f"(ARB F {af:.1f} R {ar:.1f} N/mm)")
check("a stiffer roll-gradient target needs stiffer bars",
      arb.solve_for_targets(car, -283.6, -52.0, 0.50, 50.0)[0]
      > arb.solve_for_targets(car, -283.6, -52.0, 0.80, 50.0)[0])
check("springs_only really zeroes the bars",
      arb.springs_only(car)["arb_f"] == 0.0
      and arb.springs_only(car)["arb_r"] == 0.0)

# ---------------------------------------------------------- 7. hollow gain
print("\n7. THE HOLLOW ARGUMENT, as numbers")
m_solid25 = float(arb.bar_mass_per_m(25.0))
J_solid25 = float(arb.polar_J(25.0))
# a 50 mm tube of the SAME mass
di = np.sqrt(0.050 ** 2 - 4 * m_solid25 / (arb.RHO_STEEL * np.pi))
w_same_mass = (0.050 - di) / 2 * 1000
gain = float(arb.polar_J(50.0, w_same_mass)) / J_solid25
check("same mass, 50 mm OD tube is ~7x stiffer than a 25 mm solid",
      6.5 < gain < 7.5, f"{gain:.2f}x at {w_same_mass:.2f} mm wall")
check("...and its mass really does match",
      abs(float(arb.bar_mass_per_m(50.0, w_same_mass)) - m_solid25) < 1e-6)
# same stiffness, how much lighter
di2 = (0.050 ** 4 - 0.025 ** 4) ** 0.25
w_same_k = (0.050 - di2) / 2 * 1000
lighter = float(arb.bar_mass_per_m(50.0, w_same_k)) / m_solid25
check("same stiffness, a 50 mm tube weighs ~13 % of the 25 mm solid",
      0.10 < lighter < 0.16, f"{lighter * 100:.1f} % at {w_same_k:.2f} mm wall")
check("...and its stiffness really does match",
      abs(float(arb.polar_J(50.0, w_same_k)) - J_solid25) < 1e-16)

# --------------------------------------------------- 8. OPTIMUMG CROSS-CHECK
# "Bar talk", Claude Rouelle, Racecar Engineering May 2021. An INDEPENDENT
# published method for the same quantity: build the four anti-roll stiffnesses
# from Eq.2-7, sum them (Eq.8), and divide the roll moment (Eq.1) by the total
# (Eq.9). If our load-transfer model agrees, both are right.
print("\n8. OPTIMUMG 'Bar talk' CROSS-CHECK  (Eq.1-9)")

_car = ltr.temerario_gt3(freq_f=3.86, freq_r=2.89)
_af, _ar, _ = arb.solve_for_targets(_car, -283.6, -52.0, 0.60, 50.0)
_car = dict(_car)
_car["arb_f"], _car["arb_r"] = _af, _ar
_car["k_roll_f"] = ltr.roll_stiffness(_car["kw_f"], _af, _car["track_f_mm"])
_car["k_roll_r"] = ltr.roll_stiffness(_car["kw_r"], _ar, _car["track_r_mm"])

_tf, _tr = _car["track_f_mm"] / 1000.0, _car["track_r_mm"] / 1000.0
_tan1 = np.tan(np.radians(1.0))

# Eq.2/3/4 -- springs.  Eq.5/6/7 -- bars. Our arb_* are already WHEEL rates and
# already carry the both-ends factor 2, which is the same factor Eq.5 supplies by
# omitting the /2 that Eq.2 has -- so both terms take the /2 here.
_ARSfsp = _tf ** 2 * _tan1 * (_car["kw_f"] * 1000.0) / 2.0
_ARSrsp = _tr ** 2 * _tan1 * (_car["kw_r"] * 1000.0) / 2.0
_ARSfarb = _tf ** 2 * _tan1 * (_af * 1000.0) / 2.0
_ARSrarb = _tr ** 2 * _tan1 * (_ar * 1000.0) / 2.0
_TARS = _ARSfsp + _ARSrsp + _ARSfarb + _ARSrarb          # Eq.8  [Nm/deg]

check("Eq.2 spring anti-roll stiffness matches ours, FRONT",
      abs(_ARSfsp - ltr.roll_stiffness(_car["kw_f"], 0.0, _car["track_f_mm"])
          / 1000.0 * np.pi / 180) / _ARSfsp < 2e-4,
      f"{_ARSfsp:.1f} Nm/deg")
check("Eq.2 spring anti-roll stiffness matches ours, REAR",
      abs(_ARSrsp - ltr.roll_stiffness(_car["kw_r"], 0.0, _car["track_r_mm"])
          / 1000.0 * np.pi / 180) / _ARSrsp < 2e-4,
      f"{_ARSrsp:.1f} Nm/deg")
check("Eq.8 total matches our k_roll_f + k_roll_r",
      abs(_TARS - (_car["k_roll_f"] + _car["k_roll_r"]) / 1000.0 * np.pi / 180)
      / _TARS < 2e-4,
      f"TARS {_TARS:.0f} Nm/deg")

# Eq.1 roll moment, then Eq.9 roll angle
_Wuf, _Wur = _car["unsprung_f_kg"] * ltr.G, _car["unsprung_r_kg"] * ltr.G
_Wsf = _car["mass_total_kg"] * (1 - _car["rear_frac"]) * ltr.G - _Wuf
_Wsr = _car["mass_total_kg"] * _car["rear_frac"] * ltr.G - _Wur
_Ws = _Wsf + _Wsr
_hra = ltr.roll_axis_height(-283.6, -52.0, _Wsr / _Ws)
_lever = (_car["h_cg_mm"] - _hra) / 1000.0
_SMRM = _Ws * 1.0 * _lever                                # Eq.1  [Nm per g]
_grad_optg = _SMRM / _TARS                                # Eq.9  [deg/g]
_grad_ours = ltr.transfer(1.0, -283.6, -52.0, _car)["roll_deg"]

check("Eq.1 roll moment == ours (W_s x lever)", abs(_SMRM - _Ws * _lever) < 1e-9,
      f"{_SMRM:.0f} Nm/g")
check("Eq.9 ROLL GRADIENT matches our load-transfer model",
      abs(_grad_optg - _grad_ours) / _grad_ours < 5e-4,
      f"OptimumG {_grad_optg:.4f} vs ours {_grad_ours:.4f} deg/g")
check("...and it is the 0.60 deg/g we asked the solver for",
      abs(_grad_ours - 0.60) < 1e-6)

# Figure 1 -- the published roll-gradient bands
check("roll-gradient bands are ordered and non-overlapping in the right way",
      all(arb.ROLL_GRADIENT_BANDS[i][2] >= arb.ROLL_GRADIENT_BANDS[i + 1][1]
          for i in range(len(arb.ROLL_GRADIENT_BANDS) - 1)),
      f"{len(arb.ROLL_GRADIENT_BANDS)} bands, SUV softest to high-downforce stiffest")
_band = arb.classify_roll_gradient(_grad_ours)
check("our 0.60 deg/g target classifies against Figure 1", _band is not None,
      f"'{_band}'")
check("springs alone land in a SOFTER band than the target",
      arb.classify_roll_gradient(1.119) != _band,
      f"1.119 deg/g -> '{arb.classify_roll_gradient(1.119)}'")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the ARB chain inverts, the stress identity holds, and\n"
      "the target solve genuinely hits the roll-gradient and LLTD targets.")
print("=" * 76)
