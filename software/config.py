# -*- coding: utf-8 -*-
"""
Shared inputs for the front-corner kinematics study.

SINGLE SOURCE OF TRUTH
----------------------
The baseline KPI and caster are **derived from the hard points**, not typed in.
An earlier version hard-coded STOCK_KPI = +5.3 while scrub_trail.py read the CAD
(which says -5.27) -- the two halves of the project silently disagreed about the
sign, which is worth 129 mm vs 83 mm of scrub radius. Deriving them here makes
that impossible. Do NOT reintroduce a hard-coded angle.

Hard-point source: `hardpoints_fusion.csv` -- the user's own manual Fusion
extraction, rebased to the true car origin (written by rebase_hardpoints.py).

Frame:  X = lateral (outboard +),  Y = longitudinal (forward +),  Z = vertical (up),  mm.

SIGN CONVENTIONS (standard; both angles measured from vertical)
    KPI    > 0  =>  the axis leans INBOARD at the top   (UBJ_x < LBJ_x, i.e. dX < 0)
    caster > 0  =>  the axis leans REARWARD at the top  (UBJ_y < LBJ_y, i.e. dY < 0)
    so both are   arctan(-component / dZ).

    The CAD model puts the UBJ ~13.4 mm OUTBOARD of the LBJ, so this car's KPI is
    NEGATIVE (-5.27 deg) -- confirmed against the CAD front view by the user. Real
    GT3s run positive KPI (low double digits is typical), so this is
    an artefact of an artistic mesh, not a real design. It is kept as-is: the point
    of the study is to learn how each parameter moves the car, and KPI becomes a
    design variable in the optimisation stage.

NOTE: KPI, caster and the steering arm are all built from DIFFERENCES between
points, so they are invariant to the choice of origin.
"""
import os
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
CSV = os.path.join(PROJ, "hardpoints_fusion.csv")

UBJ_NAME = "Upper wishbone - outer (UBJ)"
LBJ_NAME = "Lower wishbone - outer (LBJ)"
TIE_NAME = "Tie rod - outer"


# ---- hard-point loaders ----------------------------------------------------
def _load_points(corner="FRONT"):
    """{point_name: np.array([X_lateral, Y_longitudinal, Z_vertical])} for one
    corner, mm. NOTE: the CAD data is the FRONT-RIGHT corner (FR in Fusion);
    the left corner is its exact mirror (negate X)."""
    pts = {}
    with open(CSV) as f:
        for row in f:
            if row.startswith("#") or row.startswith("Corner") or not row.strip():
                continue
            c = row.strip().split(",")
            if len(c) < 5 or c[0] != corner:
                continue
            pts[c[1]] = np.array([float(c[2]), float(c[3]), float(c[4])])
    return pts


def _load_front_points():
    """The FRONT corner points."""
    return _load_points("FRONT")


def rear_patch_y():
    """Longitudinal ground position of the REAR contact patch on this side.
    The rear wheel does not steer, so this is the fixed wheelbase datum."""
    return float(_load_points("REAR")["Wheel centre"][1])


def stock_geometry_from_csv():
    """(KPI, caster) in deg, SIGNED, from the baseline LBJ->UBJ axis.
        KPI    > 0 = leans inboard at the top
        caster > 0 = leans rearward at the top
    This is the ONLY place the baseline angles are defined."""
    p = _load_front_points()
    dX, dY, dZ = p[UBJ_NAME] - p[LBJ_NAME]
    return (float(np.degrees(np.arctan2(-dX, dZ))),
            float(np.degrees(np.arctan2(-dY, dZ))))


def steering_arm_mm():
    """Perpendicular distance from the tie-rod OUTER ball joint to the kingpin
    axis (LBJ->UBJ). GEOMETRIC quantity only -- NOT the effective steering ratio,
    because the tie rod is oblique to the steering arm on this car. Kept for
    reference; the ratio comes from steer_angle_for_rack()."""
    p = _load_front_points()
    a = p[LBJ_NAME]
    u = p[UBJ_NAME] - a
    w = p[TIE_NAME] - a
    return float(np.linalg.norm(np.cross(w, u)) / np.linalg.norm(u))


def steer_angle_for_rack(rack_mm=None):
    """Road-wheel steer angle for a rack travel, from the EXACT velocity relation
    of the linkage (rigid tie rod, rack sliding laterally):

        d(delta)/d(rack) = (x_hat . t_hat) / ((k_hat x r) . t_hat)

    with t_hat the tie-rod direction, k_hat the kingpin direction, r the tie-rod
    outer relative to the LBJ. Verified against the full wishbone solver to 1e-5.

    The earlier estimate delta ~ rack / (perpendicular arm) was WRONG on this car
    (7.71 vs the true 10.04 deg for 20 mm): it silently assumes the tie rod is
    perpendicular to the steering arm, and here it is oblique. The perpendicular
    distance is 148.6 mm but the EFFECTIVE arm is ~114 mm.

    Returns (delta_deg magnitude, effective_arm_mm). First-order in rack; the
    full solver adds the nonlinearity (10.7 deg at a finite 20 mm).
    """
    if rack_mm is None:
        rack_mm = RACK_TRAVEL_MM
    p = _load_front_points()
    LBJ, UBJ = p[LBJ_NAME], p[UBJ_NAME]
    TRO, TRI = p[TIE_NAME], p["Tie rod - inner"]
    t = TRO - TRI
    t = t / np.linalg.norm(t)
    k = UBJ - LBJ
    k = k / np.linalg.norm(k)
    ratio = t[0] / float(np.dot(np.cross(k, TRO - LBJ), t))   # rad per mm rack
    eff_arm = 1.0 / abs(ratio)
    return float(np.degrees(abs(ratio) * rack_mm)), eff_arm


# ---- baseline, DERIVED (never hard-code these) ------------------------------
STOCK_KPI, STOCK_CASTER = stock_geometry_from_csv()   # -5.27 deg, +1.66 deg

# ---- static camber: a SETUP input, not a hard-point -------------------------
# Measured in the CAD front view: 2.7 deg with the top of the wheel leaning IN
# toward the car => NEGATIVE camber by the standard convention (+ve = top out).
# It is NOT derivable from the ball joints -- the hard points give the steering
# axis, not the wheel plane -- so it lives here as an explicit input.
# It MATTERS: a cambered wheel is tilted, so its contact patch is displaced
# sideways by R*sin(g) = 15.9 mm, and scrub radius is measured TO THE PATCH.
# Ignoring it understated the scrub radius by that much (129.1 -> 144.9 mm).
STATIC_CAMBER = -2.7

# ---- Task-2 corner assumption ----------------------------------------------
RACK_TRAVEL_MM = 20.0  # "typical corner" rack travel

# ---- camber sweeps (tasks 1.1 / 1.2) ---------------------------------------
STEER_RANGE = (-45.0, 45.0)                    # deg, road-wheel steer
# The brief asked for KPI 0..20 step 2; extended DOWN to -6 so the sweep actually
# brackets this car's negative baseline, and up to 20 to cover a real GT3 (~+12.7).
KPI_SWEEP = np.arange(-6.0, 20.0 + 1e-6, 2.0)
CASTER_SWEEP = np.arange(0.0, 10.0 + 1e-6, 1.0)   # 1.2: caster 0..10 step 1

CASTER_HELD_FOR_KPI_SWEEP = STOCK_CASTER       # 1.1 holds caster constant
KPI_HELD_FOR_CASTER_SWEEP = STOCK_KPI          # 1.2 holds KPI constant

# ---- scrub / mechanical trail (tasks 3, 4 & 5) -----------------------------
# Scrub and trail are LENGTHS: tune them by sliding BOTH ball joints together
# (dx = lateral, dy = longitudinal), which leaves the KPI/caster ANGLES untouched.
#   scrub(dx) = scrub(0) - dx        trail(dy) = trail(0) + dy
STEER_RANGE_PATCH = (-35.0, 35.0)                    # deg, contact-patch sweep
LATERAL_OFFSET_SWEEP = np.arange(0.0, 121.0, 20.0)   # dx [mm], both joints outboard
LONG_OFFSET_SWEEP = np.arange(-20.0, 61.0, 20.0)     # dy [mm], both joints forward

# ---- map extents (task 2) --------------------------------------------------
# Span the negative baseline (-5.27) through to real-GT3 territory (~+12.7).
KPI_MAP_RANGE = (-8.0, 16.0)
CASTER_MAP_RANGE = (0.0, 14.0)

# PROVISIONAL target box only -- a rough Gillespie/Guiggiani estimate. It is NOT
# yet justified: the agreed plan is to derive the window from first principles
# (caster from the camber compensation required, KPI from scrub/jacking) once the
# wishbone solver gives roll-induced camber. Treat it as a placeholder.
KPI_TARGET = (6.0, 8.0)
CASTER_TARGET = (7.0, 9.0)
TARGET_IS_PROVISIONAL = True


if __name__ == "__main__":
    delta, arm = steer_angle_for_rack()
    p = _load_front_points()
    dX, dY, dZ = p[UBJ_NAME] - p[LBJ_NAME]
    print("Baseline geometry — DERIVED from the CAD hard points")
    print("-" * 62)
    print(f"  UBJ - LBJ      : dX {dX:+.3f}  dY {dY:+.3f}  dZ {dZ:+.3f} mm")
    print(f"  KPI            : {STOCK_KPI:+.4f} deg   "
          f"({'leans INBOARD' if STOCK_KPI > 0 else 'leans OUTBOARD — artistic mesh'})")
    print(f"  caster         : {STOCK_CASTER:+.4f} deg  "
          f"({'leans rearward' if STOCK_CASTER > 0 else 'leans forward'})")
    print(f"  steering arm   : {arm:.1f} mm")
    print(f"  rack {RACK_TRAVEL_MM:.0f} mm     : steer delta ~= {delta:.2f} deg")
    print()
    print("  These are computed here and nowhere else — the camber and scrub")
    print("  modules both read them, so they cannot disagree about the sign.")
