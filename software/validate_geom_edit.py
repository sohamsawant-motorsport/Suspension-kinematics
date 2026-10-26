# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for geom_edit  (the KPI/caster -> hard-point inverse).
=======================================================================

The inverse is only worth anything if it is the EXACT inverse of the forward
formula the rest of the project uses. So the strongest checks here are
round-trips through the OTHER module: set an angle with geom_edit, read it back
with scrub_trail.kpi_caster, and demand machine precision.

  1  Identity      - retargeting to the stock angles must return the stock point.
  2  Round-trip    - over a grid of (KPI, caster), read-back must match to 1e-12.
  3  Invariants    - kingpin length, LBJ, wheel centre and every other point fixed.
  4  Sign sense    - more KPI must lean the top of the axis INBOARD, more caster
                     must lean it REARWARD, and both must reduce scrub/raise trail
                     in the direction textbook geometry says.
  5  Downstream    - the retargeted geometry must still solve, and the solver must
                     report the angles we asked for.

Run:  python validate_geom_edit.py
"""
import numpy as np

import config as cfg
from geom_edit import (retarget, ubj_for_kpi_caster, upper_arm_length,
                       UBJ_NAME, LBJ_NAME)
from scrub_trail import kpi_caster, scrub_radius, mechanical_trail
from wishbone_solver import corner_geometry, solve_sweep

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


P = cfg._load_front_points()
LBJ, UBJ, WC = P[LBJ_NAME], P[UBJ_NAME], P["Wheel centre"]
CAM = cfg.STATIC_CAMBER
KPI0, CAS0 = kpi_caster(LBJ, UBJ)
L0 = float(np.linalg.norm(UBJ - LBJ))

print("=" * 76)
print("VALIDATION - geom_edit (design target -> hard points)")
print("=" * 76)
print(f"  stock: KPI {KPI0:+.4f} deg   caster {CAS0:+.4f} deg   "
      f"kingpin {L0:.3f} mm")

# ------------------------------------------------------------------ 1. identity
print("\n1. IDENTITY - retargeting to the stock angles changes nothing")
same = retarget(P, KPI0, CAS0)
check("UBJ returns to itself",
      np.allclose(same[UBJ_NAME], UBJ, atol=1e-9),
      f"max move {np.max(np.abs(same[UBJ_NAME] - UBJ)):.2e} mm")

# ---------------------------------------------------------------- 2. round-trip
print("\n2. ROUND-TRIP through scrub_trail.kpi_caster (the forward formula)")
worst = 0.0
for nu in np.linspace(-12.0, 20.0, 17):
    for lam in np.linspace(-6.0, 14.0, 11):
        q = retarget(P, nu, lam)
        got_nu, got_lam = kpi_caster(q[LBJ_NAME], q[UBJ_NAME])
        worst = max(worst, abs(got_nu - nu), abs(got_lam - lam))
check("read-back == requested over a 17x11 grid", worst < 1e-12,
      f"worst error {worst:.2e} deg")

# ---------------------------------------------------------------- 3. invariants
print("\n3. INVARIANTS - only the UBJ is allowed to move")
q = retarget(P, 7.0, 8.0)
check("kingpin length preserved",
      abs(np.linalg.norm(q[UBJ_NAME] - q[LBJ_NAME]) - L0) < 1e-9,
      f"{np.linalg.norm(q[UBJ_NAME] - q[LBJ_NAME]):.6f} vs {L0:.6f} mm")
check("UBJ stays ABOVE the LBJ", q[UBJ_NAME][2] > q[LBJ_NAME][2])
others = [k for k in P if k != UBJ_NAME]
check("every other hard point untouched",
      all(np.allclose(q[k], P[k], atol=0) for k in others))
check("source dict not mutated", np.allclose(P[UBJ_NAME], UBJ, atol=0))

# ---------------------------------------------------------------- 4. sign sense
print("\n4. SIGN SENSE - the angles must move the axis the way the book says")
lo, hi = retarget(P, 0.0, CAS0), retarget(P, 12.0, CAS0)
check("more KPI leans the axis top INBOARD (dX falls)",
      (hi[UBJ_NAME][0] - hi[LBJ_NAME][0]) < (lo[UBJ_NAME][0] - lo[LBJ_NAME][0]),
      f"dX {lo[UBJ_NAME][0]-lo[LBJ_NAME][0]:+.2f} -> "
      f"{hi[UBJ_NAME][0]-hi[LBJ_NAME][0]:+.2f} mm")
check("more KPI REDUCES scrub radius",
      scrub_radius(hi[LBJ_NAME], hi[UBJ_NAME], WC, CAM)
      < scrub_radius(lo[LBJ_NAME], lo[UBJ_NAME], WC, CAM),
      f"{scrub_radius(lo[LBJ_NAME], lo[UBJ_NAME], WC, CAM):.1f} -> "
      f"{scrub_radius(hi[LBJ_NAME], hi[UBJ_NAME], WC, CAM):.1f} mm")

lo, hi = retarget(P, KPI0, 0.0), retarget(P, KPI0, 10.0)
check("more caster leans the axis top REARWARD (dY falls)",
      (hi[UBJ_NAME][1] - hi[LBJ_NAME][1]) < (lo[UBJ_NAME][1] - lo[LBJ_NAME][1]))
check("more caster INCREASES mechanical trail",
      mechanical_trail(hi[LBJ_NAME], hi[UBJ_NAME], WC, CAM)
      > mechanical_trail(lo[LBJ_NAME], lo[UBJ_NAME], WC, CAM),
      f"{mechanical_trail(lo[LBJ_NAME], lo[UBJ_NAME], WC, CAM):.1f} -> "
      f"{mechanical_trail(hi[LBJ_NAME], hi[UBJ_NAME], WC, CAM):.1f} mm")

# zero KPI must put the axis exactly vertical in front view
z = retarget(P, 0.0, 0.0)
check("KPI = caster = 0 gives a dead-vertical axis",
      abs(z[UBJ_NAME][0] - z[LBJ_NAME][0]) < 1e-9
      and abs(z[UBJ_NAME][1] - z[LBJ_NAME][1]) < 1e-9)

# ---------------------------------------------------------------- 5. downstream
print("\n5. DOWNSTREAM - the retargeted corner must still solve")
q = retarget(P, 7.0, 8.0)
g = corner_geometry(q, CAM)
sw = solve_sweep(g, np.linspace(-20.0, 20.0, 41))
i0 = int(np.argmin(np.abs(sw["bump"])))
check("solver reports the KPI we asked for", abs(sw["kpi"][i0] - 7.0) < 1e-6,
      f"{sw['kpi'][i0]:+.6f} deg")
check("solver reports the caster we asked for", abs(sw["caster"][i0] - 8.0) < 1e-6,
      f"{sw['caster'][i0]:+.6f} deg")
check("solver reports the static camber we asked for",
      abs(sw["camber"][i0] - CAM) < 1e-6, f"{sw['camber'][i0]:+.6f} deg")
check("camber curve is finite over the whole sweep",
      np.all(np.isfinite(sw["camber"])))
check("upper arm length is reported and sane",
      50.0 < upper_arm_length(q) < 800.0,
      f"stock {upper_arm_length(P):.1f} -> {upper_arm_length(q):.1f} mm")

print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the inverse is exact and its downstream is consistent.")
print("=" * 76)
