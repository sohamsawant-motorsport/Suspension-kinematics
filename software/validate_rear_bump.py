# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the rear BUMP kinematics (bump steer + camber gain).
=========================================================================

Now that a toe link exists, the rear is solved by the SAME wishbone_solver the
front uses (rack frozen at 0). So the job here is to prove the reuse is faithful
and self-consistent. The strongest checks are CROSS-MODULE: the bump solver, at
ride height, must reproduce the INDEPENDENT static construction from Study 1
(rear_geometry.static_geometry / front_view_ic), which was built by a completely
different method (ball-joint perturbation). Two different code paths, one answer.

  1  ASSEMBLY      - the rear-with-toe-link solves over the whole travel range.
  2  RIGID LENGTHS - kingpin and TOE-LINK lengths are preserved at every travel
                     (the links are rigid; if these drift the solve is wrong).
  3  RIDE HEIGHT   - at bump = 0 the solver reproduces the static geometry: toe 0,
                     camber = the setup value, and KPI/caster == Study 1.
  4  CROSS-MODULE  - the solver's roll-centre height at ride height == the Study-1
                     front-view construction (independent method), to a mm.
  5  GRADIENTS     - bump steer and camber gain equal the slope of the curves and
                     are finite everywhere; camber builds the way the short rear
                     FVSA demands (stronger than the front).
  6  NO STEER      - applying rack travel is a no-op relabelled: the rear is only
                     ever solved at rack = 0 (documents the reuse, not a bug).

Run:  python validate_rear_bump.py
"""
import numpy as np

import rear_geometry as rg
from wishbone_solver import corner_geometry, solve_sweep, solve_at

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


P = rg.load_rear()
CAM = rg.REAR_STATIC_CAMBER
print("=" * 76)
print("VALIDATION - rear bump kinematics (toe link present)")
print("=" * 76)

check("CAD has a toe link", rg.has_toe_link(P))
sp = rg.solver_points(P)
g = corner_geometry(sp, CAM)
sw = rg.bump_sweep(P, CAM, -30.0, 30.0, 121)

# --------------------------------------------------------------- 1. assembly
print("\n1. ASSEMBLY")
check("rear + toe link solves over -30..+30 mm travel",
      np.all(np.isfinite(sw["camber"])) and np.all(np.isfinite(sw["toe"])))

# ----------------------------------------------------------- 2. rigid lengths
print("\n2. RIGID-LINK LENGTHS PRESERVED")
kp0 = float(np.linalg.norm(P[rg.UBJ] - P[rg.LBJ]))
tl0 = float(np.linalg.norm(P[rg.TOE_O] - P[rg.TOE_I]))
wk = wt = 0.0
thU, thT = 0.0, 0.0
for th in np.linspace(-12, 12, 25):     # sweep the wishbone angle directly
    s = solve_at(g, th, 0.0, thU, thT)
    thU, thT = s["thU"], s["thT"]
    wk = max(wk, abs(np.linalg.norm(s["UBJ"] - s["LBJ"]) - kp0))
    wt = max(wt, abs(np.linalg.norm(s["TRO"] - P[rg.TOE_I]) - tl0))
check("kingpin length constant", wk < 1e-6, f"worst |d| {wk:.2e} mm")
check("toe-link length constant", wt < 1e-6, f"worst |d| {wt:.2e} mm")

# ------------------------------------------------------------- 3. ride height
print("\n3. RIDE HEIGHT reproduces the statics")
check("static toe == 0 (reference)", abs(sw["static_toe"]) < 1e-6,
      f"{sw['static_toe']:+.2e} deg")
check("static camber == the setup value", abs(sw["static_camber"] - CAM) < 1e-6,
      f"{sw['static_camber']:+.4f} vs {CAM:+.2f} deg")
gs = rg.static_geometry(P, CAM)
i0 = int(np.argmin(np.abs(sw["bump"])))
check("KPI at ride == Study-1 static KPI", abs(sw["kpi"][i0] - gs["kpi"]) < 1e-6,
      f"{sw['kpi'][i0]:+.4f} vs {gs['kpi']:+.4f} deg")
check("caster at ride == Study-1 static caster",
      abs(sw["caster"][i0] - gs["caster"]) < 1e-6,
      f"{sw['caster'][i0]:+.4f} vs {gs['caster']:+.4f} deg")

# ------------------------------------------------------------- 4. cross-module
print("\n4. CROSS-MODULE: solver RC == Study-1 front-view construction")
check("roll-centre height agrees between the two methods",
      abs(sw["rc0"] - gs["rc_height"]) < 1.0,
      f"solver {sw['rc0']:+.1f} vs construction {gs['rc_height']:+.1f} mm")

# --------------------------------------------------------------- 5. gradients
print("\n5. GRADIENTS (bump steer + camber gain)")
check("bump steer == slope of toe(travel) at ride",
      abs(sw["bump_steer"] - np.gradient(sw["toe"], sw["bump"])[i0]) < 1e-12,
      f"{sw['bump_steer']:+.5f} deg/mm")
check("camber gain == slope of camber(travel) at ride",
      abs(sw["camber_gain"] - np.gradient(sw["camber"], sw["bump"])[i0]) < 1e-12,
      f"{sw['camber_gain']:+.5f} deg/mm")
check("camber gain is finite and non-trivial",
      np.isfinite(sw["camber_gain"]) and abs(sw["camber_gain"]) > 1e-4)
# short rear FVSA (4297 mm) must give more camber gain than the front (13336 mm)
check("rear camber gain stronger than the front's (-0.00091)",
      abs(sw["camber_gain"]) > 0.00091,
      f"rear {sw['camber_gain']:+.5f} deg/mm")

# ---------------------------------------------------------------- 6. no steer
print("\n6. NO STEER (rear reuse is rack = 0)")
s_ride = solve_at(g, 0.0, rack_mm=0.0)
toe_ride = np.degrees(np.arctan2(s_ride["s"][1], s_ride["s"][0]))
check("solver reference toe is zero at rack 0", abs(toe_ride) < 1e-9,
      f"{toe_ride:+.2e} deg")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the rear bump solve reuses the front solver faithfully "
      "and\nagrees with the independent Study-1 construction.")
print("=" * 76)
