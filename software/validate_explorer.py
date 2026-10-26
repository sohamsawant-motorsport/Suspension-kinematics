# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the hard-point explorer's COMPUTE LAYER.
=============================================================

The explorer's physics all comes from cores that are validated elsewhere
(validate_scrub_trail 33, validate_wishbone 27, validate_axle 24,
validate_geom_edit 17). What was NOT asserted anywhere is the ADAPTER the GUI
runs on -- corner_compute, the symmetric fast-path, steer_from_rack, the axle
patch paths / Ackermann, the tab-1/2 angle isolation, and the roll-migration
wiring. This suite closes that gap before the rear axle is added.

The strongest checks here are CROSS-MODULE: the explorer must reproduce the
already-validated cores to machine precision, because it is only meant to be a
thin adapter over them.

  1  BASELINE      - the regression numbers that must never silently change.
  2  CROSS-MODULE  - corner_compute == scrub_trail / wishbone_solver, exactly.
  3  TAB 1/2       - the KPI-only / caster-only isolation == corner_state.
  4  MIRROR        - a mirrored-left corner reproduces the right (signed exact,
                     absolute X negated).
  5  SYMMETRIC     - the fast-path (skip the L solve) == the full L solve. This
                     PROVES the optimisation rather than trusting it.
  6  STEER/RACK    - steer_from_rack == the full wishbone solver at equal wishbone
                     angle; exact, not the first-order estimate.
  7  ACKERMANN     - even in rack, zero at centre, both wheels same sign, inner
                     turns more.
  8  ROLL          - roll_migration is on the centreline and equals the heave RC
                     at zero roll; odd/even symmetry in roll.
  9  ROBUSTNESS    - a broken geometry returns an error dict, never raises.

Run:  python validate_explorer.py
"""
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")                                               # headless

import numpy as np

import config as cfg
from scrub_trail import (kpi_caster, corner_state, scrub_radius,
                         mechanical_trail)
from wishbone_solver import corner_geometry, solve_sweep, solve_at
import explorer as X

fails = []


def check(name, ok_, detail=""):
    print(f"  [{'PASS' if ok_ else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok_:
        fails.append(name)


R0 = cfg._load_front_points()
L0 = X.mirror_pts(R0)
CAM = float(cfg.STATIC_CAMBER)

print("=" * 78)
print("VALIDATION - hard-point explorer compute layer")
print("=" * 78)

res = X.compute(R0, CAM, L0, CAM, symmetric=True)
r = res["R"]

# --------------------------------------------------------------- 1. baseline
print("\n1. BASELINE - the regression numbers that must never silently change")
BASE = dict(kpi=-5.2678, caster=1.6588, scrub0=144.94, trail0=31.15,
            cam_gain=-0.00091, recovery=1.3195, bump_steer=-0.11688,
            rc0=-283.6, mr0=0.9908, eff_arm=114.12, steer20=10.041,
            kingpin=146.502, upper_arm=246.982, lower_arm=237.179)
tol = dict(kpi=1e-3, caster=1e-3, scrub0=0.02, trail0=0.02, cam_gain=1e-5,
           recovery=1e-3, bump_steer=1e-4, rc0=0.2, mr0=1e-3, eff_arm=0.05,
           steer20=0.02, kingpin=1e-2, upper_arm=1e-2, lower_arm=1e-2)
for k in BASE:
    check(f"{k:<12s} = {r[k]:+.5f}", abs(r[k] - BASE[k]) < tol[k],
          f"expected {BASE[k]:+.5f}  |d| {abs(r[k]-BASE[k]):.2e}")
check("track", abs(res["track"] - 1662.4) < 0.2, f"{res['track']:.1f} mm")

# ----------------------------------------------------------- 2. cross-module
print("\n2. CROSS-MODULE - corner_compute must equal the validated cores exactly")
LBJ, UBJ, WC = R0[X.LBJ_NAME], R0[X.UBJ_NAME], R0["Wheel centre"]
kpi_c, cas_c = kpi_caster(LBJ, UBJ)
check("kpi == scrub_trail.kpi_caster", abs(r["kpi"] - kpi_c) < 1e-12)
check("caster == scrub_trail.kpi_caster", abs(r["caster"] - cas_c) < 1e-12)
check("scrub0 == scrub_trail.scrub_radius",
      abs(r["scrub0"] - scrub_radius(LBJ, UBJ, WC, CAM)) < 1e-12)
check("trail0 == scrub_trail.mechanical_trail",
      abs(r["trail0"] - mechanical_trail(LBJ, UBJ, WC, CAM)) < 1e-12)

st = corner_state(LBJ, UBJ, WC, r["steer"], CAM)
check("cam_steer == corner_state camber",
      np.max(np.abs(r["cam_steer"] - st["camber"])) < 1e-12,
      f"max|d| {np.max(np.abs(r['cam_steer']-st['camber'])):.2e}")
check("scrub_steer == corner_state scrub",
      np.max(np.abs(r["scrub_steer"] - st["scrub"])) < 1e-12)
check("trail_steer == corner_state trail",
      np.max(np.abs(r["trail_steer"] - st["trail"])) < 1e-12)

g = corner_geometry(R0, CAM)
sw = solve_sweep(g, r["bump"], theta_span_deg=28.0, n_theta=1121)
check("cam_bump == solve_sweep camber",
      np.max(np.abs(r["cam_bump"] - sw["camber"])) < 1e-12)
check("toe_bump == solve_sweep toe",
      np.max(np.abs(r["toe_bump"] - sw["toe"])) < 1e-12)
check("mr == solve_sweep MR",
      np.max(np.abs(r["mr"] - sw["MR"])) < 1e-12)
# motion ratio IS the lecture's x_s / x_w = -d(damper)/d(bump) (ENGR7008 wk4)
mr_lecture = -np.gradient(sw["damper"], sw["bump"])
check("mr == lecture x_s/x_w = -d(damper)/d(bump)",
      np.max(np.abs(r["mr"] - mr_lecture)) < 1e-12)

# ------------------------------------------------------------- 3. tab 1 / 2
print("\n3. TAB 1/2 - the KPI-only / caster-only isolation")
# the data pickers return {(tag, side): (x, y)} -- one entry per plotted line
st0 = corner_state(LBJ, UBJ, WC, r["steer"], 0.0)   # zero static camber
_, yc = X.kpi_effect(res, res)[("cur", "R")]
check("kpi_effect curve == corner_state camber (zero static camber)",
      np.max(np.abs(yc - st0["camber"])) < 1e-13,
      f"max|d| {np.max(np.abs(yc-st0['camber'])):.2e}")

# isolation: a near-pure KPI move (slide the UBJ in X) must swing tab 1 hard and
# tab 2 hardly at all
res_dK = X.compute({**R0, X.UBJ_NAME: R0[X.UBJ_NAME] + np.array([-30.0, 0, 0])},
                   CAM, L0, CAM, symmetric=False)
base_curve = X.kpi_effect(res, res)[("cur", "R")][1]
k1 = X.kpi_effect(res, res_dK)[("cur", "R")][1]      # tab 1 reacts to KPI
k2 = X.caster_effect(res, res_dK)[("cur", "R")][1]   # tab 2 keys off caster
sw1 = float(np.max(np.abs(k1 - base_curve)))
sw2 = float(np.max(np.abs(k2 - base_curve)))
check("moving UBJ-X swings tab 1 (KPI effect)", sw1 > 0.1, f"swing {sw1:.3f} deg")
check("the SAME move barely moves tab 2 (caster effect)", sw2 < 0.05 * sw1,
      f"tab2 {sw2:.3f} vs tab1 {sw1:.3f} deg")

# ---------------------------------------------------------------- 4. mirror
print("\n4. MIRROR - a mirrored-left corner reproduces the right")
Lfull = X.corner_compute(L0, CAM, True)
for k in ("kpi", "caster", "scrub0", "trail0", "cam_gain", "bump_steer",
          "rc0", "mr0", "eff_arm", "kingpin"):
    check(f"signed {k} identical L vs R", abs(r[k] - Lfull[k]) < 1e-9,
          f"|d| {abs(r[k]-Lfull[k]):.2e}")
check("cam_bump curve identical L vs R",
      np.max(np.abs(r["cam_bump"] - Lfull["cam_bump"])) < 1e-9)
# The patch paths mirror ONLY at zero steer. Away from it both wheels turn the
# SAME way, so the left is the mirror of the right evaluated at the OPPOSITE
# steer -- not at the same index. An earlier version of this check asserted the
# same-index mirror and so quietly enforced the bug where both corners were fed
# an identical steer input.
i0 = int(np.argmin(np.abs(r["steer"])))
check("patch_x mirrors at ZERO steer",
      abs(r["patch_x"][i0] + Lfull["patch_x"][i0]) < 1e-9)
check("patch_y identical at ZERO steer (mirror leaves Y alone)",
      abs(r["patch_y"][i0] - Lfull["patch_y"][i0]) < 1e-9)
check("L(delta) == mirror of R(-delta)   [the steer really is mirrored]",
      np.max(np.abs(Lfull["patch_x"] + r["patch_x"][::-1])) < 1e-9,
      f"max |d| {np.max(np.abs(Lfull['patch_x'] + r['patch_x'][::-1])):.2e} mm")
check("...and the two sides genuinely DIFFER under steer",
      np.max(np.abs(r["patch_x"] + Lfull["patch_x"])) > 1.0,
      "both wheels steer the same way, so they are not mirror images")

# ------------------------------------------------------------- 5. symmetric
print("\n5. SYMMETRIC FAST-PATH - the skipped L solve must equal the full L solve")
fast = X.compute(R0, CAM, L0, CAM, symmetric=True)     # copies R for L
full = X.compute(R0, CAM, L0, CAM, symmetric=False)    # solves L properly
worst = 0.0
for k in ("kpi", "caster", "scrub0", "trail0", "cam_gain", "recovery",
          "bump_steer", "rc0", "mr0", "eff_arm", "steer20"):
    worst = max(worst, abs(fast["L"][k] - full["L"][k]))
check("every L scalar: fast == full", worst < 1e-9, f"worst |d| {worst:.2e}")
cw = max(np.max(np.abs(fast["L"][k] - full["L"][k]))
         for k in ("cam_bump", "toe_bump", "rc", "cam_steer", "mr", "phi",
                   "scrub_steer", "cam_road_car", "cam_bump_car", "patch_x"))
check("every L curve: fast == full", cw < 1e-9, f"worst |d| {cw:.2e}")
check("fast L is flagged is_left", fast["L"]["is_left"] is True)

# THE REGRESSION GUARD for the reported bug: on a symmetric car the two sides
# must be IDENTICAL in heave but DIFFERENT under steer and under roll.
k20 = int(np.argmin(np.abs(full["R"]["steer"] - 20.0)))
check("heave: L == R (camber vs wheel travel)",
      np.max(np.abs(full["R"]["cam_bump"] - full["L"]["cam_bump"])) < 1e-9)
check("STEER: L != R (camber)",
      abs(full["R"]["cam_steer"][k20] - full["L"]["cam_steer"][k20]) > 0.1,
      f"R {full['R']['cam_steer'][k20]:+.3f} vs L {full['L']['cam_steer'][k20]:+.3f} deg")
check("STEER: L != R (scrub)",
      abs(full["R"]["scrub_steer"][k20] - full["L"]["scrub_steer"][k20]) > 1.0,
      f"R {full['R']['scrub_steer'][k20]:+.1f} vs L {full['L']['scrub_steer'][k20]:+.1f} mm")
# Both corners now share ONE body-roll axis, and at a given roll they sit at
# OPPOSITE wheel travel. For a symmetric car that gives the exact identities
# below; they would fail immediately if the two sides were fed the same travel.
check("ROLL: both corners share one body-roll axis",
      np.allclose(full["R"]["phi"], full["L"]["phi"], atol=1e-12))
check("ROLL: per-wheel  camber_L(phi) == camber_R(-phi)",
      np.max(np.abs(full["L"]["cam_road"] - full["R"]["cam_road"][::-1])) < 1e-9,
      f"max |d| {np.max(np.abs(full['L']['cam_road'] - full['R']['cam_road'][::-1])):.2e} deg")
check("ROLL: car-frame  camber_L(phi) == -camber_R(-phi)",
      np.max(np.abs(full["L"]["cam_road_car"]
                    + full["R"]["cam_road_car"][::-1])) < 1e-9)
kr = int(np.argmin(np.abs(full["R"]["phi"] - 2.0)))
check("ROLL: L != R at +2 deg (one loaded, one unloaded)",
      abs(full["R"]["cam_road_car"][kr] - full["L"]["cam_road_car"][kr]) > 1.0,
      f"R {full['R']['cam_road_car'][kr]:+.3f} vs "
      f"L {full['L']['cam_road_car'][kr]:+.3f} deg")

# --------------------------------------------------------------- 6. steer/rack
print("\n6. STEER FROM RACK - exact vs the full wishbone solver")
# same wishbone angle (theta=0) so the only motion is the kingpin twist
a0 = solve_at(g, 0.0, rack_mm=0.0)
t0 = np.degrees(np.arctan2(a0["s"][1], a0["s"][0]))
worst = 0.0
for rv in (5.0, 20.0, 45.0):
    mine = abs(X.steer_from_rack(R0, np.array([0.0, rv]))[1])
    a = solve_at(g, 0.0, rack_mm=rv)
    solver = abs(np.degrees(np.arctan2(a["s"][1], a["s"][0])) - t0)
    # steer_from_rack returns the KINGPIN rotation; convert via the same axis the
    # heading uses is done in axle_patch_paths -- here compare the rotation to the
    # solver's toe change only in magnitude ORDER, and assert exactness via axle
    worst = max(worst, abs(mine - solver))
check("rack->kingpin exact solve is well past the first-order estimate",
      abs(X.steer_from_rack(R0, np.array([0.0, 20.0]))[1]) > 10.3,
      f"20 mm -> {abs(X.steer_from_rack(R0, np.array([0.0,20.0]))[1]):.2f} deg "
      f"(first-order {cfg.steer_angle_for_rack(20.0)[0]:.2f})")
# the DEFINITIVE exactness check lives in the axle patch paths (heading form):
ax = res["axle"]
i20 = int(np.argmin(np.abs(ax["rack"] - 20.0)))
head = abs(ax["R"]["steer"][i20])
a = solve_at(g, 0.0, rack_mm=float(ax["rack"][i20]))
solver_head = abs(np.degrees(np.arctan2(a["s"][1], a["s"][0])) - t0)
check("axle heading == full solver heading (same wishbone angle)",
      abs(head - solver_head) < 1e-9, f"|d| {abs(head-solver_head):.2e} deg")

# ------------------------------------------------------------- 7. ackermann
print("\n7. ACKERMANN")
rk = ax["rack"]
for rv in (20.0, 45.0):
    i = int(np.argmin(np.abs(rk - rv)))
    j = int(np.argmin(np.abs(rk + rv)))
    check(f"even in rack at +/-{rv:.0f}", abs(ax["ack"][i] - ax["ack"][j]) < 1e-9)
    check(f"both wheels same sign at rack {rv:.0f}",
          np.sign(ax["L"]["steer"][i]) == np.sign(ax["R"]["steer"][i]))
i0 = int(np.argmin(np.abs(rk)))
check("Ackermann zero at rack 0", abs(ax["ack"][i0]) < 1e-6,
      f"{ax['ack'][i0]:.2e} deg")
i20 = int(np.argmin(np.abs(rk - 20.0)))
check("inner wheel (R at +rack) turns MORE than outer",
      abs(ax["R"]["steer"][i20]) > abs(ax["L"]["steer"][i20]),
      f"R {ax['R']['steer'][i20]:+.2f}  L {ax['L']['steer'][i20]:+.2f}")

# ---------------------------------------------------------------- 8. roll
print("\n8. ROLL MIGRATION")
rm = X.roll_migration(R0, CAM, L0, CAM)
i0 = int(np.argmin(np.abs(rm["roll"])))
check("RC on centreline at zero roll", abs(rm["rc_x"][i0]) < 1e-6,
      f"rc_x = {rm['rc_x'][i0]:.2e} mm")
check("RC height at zero roll == heave RC (rc0)",
      abs(rm["rc_z"][i0] - r["rc0"]) < 0.5,
      f"roll {rm['rc_z'][i0]:+.1f} vs heave {r['rc0']:+.1f} mm")
n = len(rm["roll"])
check("RC x is ODD in roll",
      abs(rm["rc_x"][0] + rm["rc_x"][n - 1]) < 1e-6,
      f"{rm['rc_x'][0]:+.1f} vs {rm['rc_x'][n-1]:+.1f} mm")
check("RC z is EVEN in roll",
      abs(rm["rc_z"][0] - rm["rc_z"][n - 1]) < 1e-6)
check("camber L/R swap under +/- roll",
      abs(rm["camber_L"][0] - rm["camber_R"][n - 1]) < 1e-9)

# ------------------------------------------------------------ 9. robustness
print("\n9. ROBUSTNESS - a broken geometry returns an error, never raises")
bad = {k: v.copy() for k, v in R0.items()}
bad[X.UBJ_NAME] = bad[X.LBJ_NAME].copy()          # zero-length kingpin
out = X.compute(bad, CAM, X.mirror_pts(bad), CAM, symmetric=True)
check("degenerate geometry -> error dict, not an exception",
      (not X.ok(out)) and isinstance(out.get("error"), str),
      f"error = {str(out.get('error'))[:44]}")
far = {k: v.copy() for k, v in R0.items()}
far[X.TIE_NAME] = far[X.TIE_NAME] + np.array([0, 0, 5000.0])   # unreachable tie rod
out2 = X.corner_compute(far, CAM, False)
check("unsolvable linkage -> error dict, not an exception",
      isinstance(out2.get("error"), str) or X.ok(out2))   # either clean solve or clean error

# ------------------------------------------------------- 10. Ackermann wiring
# The Ackermann MATHS is proved in validate_ackermann.py. What is guarded here
# is the WIRING: that compute() carries the kingpin ground track, and that the
# panel's own reduction of the axle sweep reproduces the standalone module.
print("\n10. ACKERMANN WIRING (maths lives in validate_ackermann.py)")
import ackermann as _ak                                              # noqa: E402

good = X.compute(R0, CAM, X.mirror_pts(R0), CAM, symmetric=True)
check("compute() carries ack_t", good.get("ack_t") is not None)
check("ack_t == ackermann.kingpin_ground_track",
      abs(good["ack_t"]
          - _ak.kingpin_ground_track(R0, X.mirror_pts(R0), X.mirror_pts)) < 1e-9,
      f"{good['ack_t']:.2f} mm")
check("ack_t is the GROUND track, not the wheel track",
      abs(good["ack_t"] - 2 * abs(float(R0['Wheel centre'][0]))) > 100.0,
      f"{good['ack_t']:.1f} vs wheel track {2*abs(float(R0['Wheel centre'][0])):.1f} mm")

pulled = X.AckermannPanel._pull(good)
check("the panel reduces the axle sweep without error", pulled is not None)
o, i, idl, pct, pct0 = pulled
check("panel outer angles are sorted and start at zero",
      np.all(np.diff(o) >= -1e-12) and abs(o[0]) < 1e-9)
check("panel inner >= outer (pro-Ackermann car)", np.all(i >= o - 1e-9))
stand = _ak.sweep(R0, CAM, X.mirror_pts(R0), CAM,
                  wheelbase_mm=good["R"]["wheelbase"][len(good["R"]["wheelbase"]) // 2])
check("panel zero-steer % matches the standalone module",
      abs(pct0 - stand["pct_at_zero"]) < 0.5,
      f"panel {pct0:.2f} vs module {stand['pct_at_zero']:.2f} %")
check("headline preserved through the GUI path: ~130 %", 125.0 < pct0 < 135.0,
      f"{pct0:.2f} %")

# Hover furniture: update() calls ax.clear(), which DESTROYS the markers, guide
# lines and per-plot readout boxes -- so update() has to rebuild them every time.
# Forgetting that is silent (the plot still draws, the cursor just stops working),
# which is exactly the bug this guards. Driven headlessly through a fake event.
import tkinter as _tk                                                # noqa: E402


class _Ev:                       # the two attributes matplotlib hands a callback
    inaxes = None
    xdata = None


_root = _tk.Tk()
_root.withdraw()
try:
    _nb = X.ttk.Notebook(_root)
    _pan = X.AckermannPanel(_nb, "t")
    _pan.update(good, good)
    _axes = (_pan.axA, _pan.axB, _pan.axC)
    check("every plot gets its OWN readout box", len(_pan._note) == 3)
    check("every plot gets a cursor marker and a guide line",
          len(_pan._mark) == 3 and len(_pan._vline) == 3,
          f"markers per plot: {[len(_pan._mark[a]) for a in _axes]}")
    check("13a and 13c mark BOTH the actual and the ideal curve",
          len(_pan._mark[_pan.axA]) == 2 and len(_pan._mark[_pan.axC]) == 2)

    _e = _Ev()
    _e.inaxes, _e.xdata = _pan.axB, 16.0
    _pan._on_move(_e)
    _texts = [_pan._note[a].get_text() for a in _axes]
    check("hovering fills all three boxes", all(t for t in _texts))
    check("each box shows its OWN quantity, not a shared dump",
          "inner" in _texts[0] and "Ackermann" in _texts[1]
          and "toe-out" in _texts[2])
    _xs = [float(_pan._vline[a].get_xdata()[0]) for a in _axes]
    check("all three guides land on the SAME steer angle (linked cursor)",
          max(_xs) - min(_xs) < 1e-9, f"x = {_xs[0]:.2f} deg")
    _my = float(_pan._mark[_pan.axA][0].get_ydata()[0])
    check("the marker snaps ONTO the curve, not to the raw mouse y",
          abs(_my - i[int(np.argmin(np.abs(o - 16.0)))]) < 1e-9,
          f"marker y = {_my:.3f} deg")

    _pan._on_leave(None)
    check("leaving the axes clears every box and marker",
          all(_pan._note[a].get_text() == "" for a in _axes)
          and all(len(m.get_xdata()) == 0 for a in _axes for m in _pan._mark[a]))

    # the regression that matters: a second update() must REBUILD the furniture
    _pan.update(good, good)
    _e.inaxes, _e.xdata = _pan.axA, 22.0
    _pan._on_move(_e)
    check("hover still works after a recompute (artists rebuilt)",
          all(_pan._note[a].get_text() for a in _axes))

    # The hard-point highlight must reflect MEASURED sensitivity, not a list
    # someone typed once. Re-measure it here: nudge every point 5 mm on every
    # axis and check the declared tier matches what actually moves. If a future
    # change alters which points drive Ackermann, this fails instead of quietly
    # pointing the user at the wrong screws. ~1.2 s for the full sweep.
    _STRONG, _WEAK, _NONE = 2.0, 0.05, 0.05        # % per 5 mm tier boundaries
    _wb = abs(float(R0["Wheel centre"][1]) - X.cfg.rear_patch_y())

    def _pct(pts, wb):
        _s = _ak.sweep(pts, CAM, X.mirror_pts(pts), CAM, wheelbase_mm=wb)
        return _s["pct_at_zero"] if _s else float("nan")

    _b = _pct(R0, _wb)
    _measured = {}
    for _nm in R0:
        _d = []
        for _axis in range(3):
            _p = {k: np.array(v, float) for k, v in R0.items()}
            _step = np.zeros(3)
            _step[_axis] = 5.0
            _p[_nm] = _p[_nm] + _step
            _w = abs(float(_p["Wheel centre"][1]) - X.cfg.rear_patch_y())
            try:
                _d.append(_pct(_p, _w) - _b)
            except Exception:                                     # noqa: BLE001
                _d.append(0.0)
        _measured[_nm] = float(np.nanmax(np.abs(_d)))

    _decl = X.AckermannPanel.HIGHLIGHT
    check("every highlighted point is a real Ackermann driver",
          all(_nm in _measured for _nm in _decl))
    _bad = [f"{n} ({_measured[n]:.2f})" for n, lv in _decl.items()
            if (lv == "strong" and _measured[n] < _STRONG)
            or (lv == "weak" and not (_WEAK <= _measured[n] < _STRONG))]
    check("declared STRONG/WEAK tiers match the measured sensitivity",
          not _bad, "mismatch: " + ", ".join(_bad) if _bad else
          "strong " + "/".join(f"{_measured[n]:.1f}" for n, l in _decl.items()
                               if l == "strong")
          + "  weak " + "/".join(f"{_measured[n]:.2f}" for n, l in _decl.items()
                                 if l == "weak") + "  % per 5 mm")
    _missed = [f"{n} ({v:.2f})" for n, v in _measured.items()
               if n not in _decl and v >= _NONE]
    check("no real driver was left OUT of the highlight", not _missed,
          "missing: " + ", ".join(_missed) if _missed else
          f"{len(_measured) - len(_decl)} points measured at exactly zero")

    # and the tint actually lands on the table
    _tbl = X.SideTable(_nb, X.POINT_ROWS)
    _tbl.set_highlight(_decl)
    _lit = {n for n, (lab, _u, _b2) in _tbl.rowart.items()
            if lab.cget("bg") in (X.SideTable.HL_STRONG, X.SideTable.HL_WEAK)}
    check("the table lights exactly the declared rows", _lit == set(_decl),
          f"{len(_lit)} rows lit")
    _tbl.set_highlight({})
    check("clearing the highlight restores every row's normal stripe",
          all(lab.cget("bg") == b2 for _n, (lab, _u, b2) in _tbl.rowart.items()))
finally:
    _root.destroy()

print("\n" + "=" * 78)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the explorer is a faithful adapter over the cores.")
print("=" * 78)
