# -*- coding: utf-8 -*-
"""
REGENERATE THE COMPARISON TABLE IN README.md
============================================

    python hardpoints/verify_table.py

Prints the before/after table as markdown, computed from the two CSVs this
repository ships. Paste the output into README.md, or just run it to check that
what README.md claims is still what the hard points do.

WHY THIS EXISTS: the table used to be typed by hand from readouts taken at
different times, and it drifted. Three of its rows disagreed with the files it
was describing. A table assembled from snapshots cannot be checked by a reader;
one printed from the CSVs can.

TWO CONVENTIONS HAVE TO BE STATED, because both columns have to use the same one
or the comparison is meaningless:

  STATIC CAMBER is a setup input, not a hard point. Scrub and roll-centre
  height both move with it (rear scrub moves 6.08 mm per degree), so the same
  value is used for both columns: -2.7 deg front, -1.5 deg rear.

  ARM RATIO is the perpendicular distance from the ball joint to its inner
  hinge AXIS, lower over upper. Not the distance to a pivot point -- the arm
  swings about the axis, so that is the radius that matters.

RC LATERAL is how far sideways the roll centre has moved at 1 degree of body
roll, from the two-corner axle model. Not a slope: the migration is not linear
and quoting a gradient overstates the small-angle behaviour.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "software"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ackermann as ack
import config as cfg
import explorer as ex
import rear_geometry as rg

FRONT_CAMBER = -2.7
REAR_CAMBER = -1.5

SETS = (("starting", os.path.join(_ROOT, "hardpoints_fusion.csv")),
        ("redesigned", os.path.join(_HERE, "hardpoints_redesigned.csv")))

UP = ("Upper wishbone - outer (UBJ)", "Upper wishbone - inner front",
      "Upper wishbone - inner rear")
LO = ("Lower wishbone - outer (LBJ)", "Lower wishbone - inner front",
      "Lower wishbone - inner rear")


def arm_radius(pts, trio):
    """Perpendicular distance from the ball joint to the inner hinge axis."""
    B, A, C = (np.asarray(pts[k], float) for k in trio)
    u = (C - A) / np.linalg.norm(C - A)
    v = B - A
    return float(np.linalg.norm(v - np.dot(v, u) * u))


def rc_lateral_at_1deg(pts, cam):
    """Sideways roll-centre movement at 1 deg of body roll, or None."""
    sw = ex.roll_migration(pts, cam, ex.mirror_pts(pts), cam)
    if sw is None:
        return None
    return float(np.interp(1.0, np.asarray(sw["roll"], float),
                           np.asarray(sw["rc_x"], float)))


def measure(csv_path):
    cfg.CSV = csv_path
    front = cfg._load_points("FRONT")
    rear_raw = cfg._load_points("REAR")
    rear = rg.solver_points(rear_raw)

    wheelbase = abs(float(front["Wheel centre"][1])
                    - float(rear["Wheel centre"][1]))
    F = ex.corner_compute(front, FRONT_CAMBER, False)
    R = ex.corner_compute(rear, REAR_CAMBER, False)
    for tag, d in (("FRONT", F), ("REAR", R)):
        if d.get("error"):
            raise SystemExit(f"{tag} did not solve in {csv_path}: {d['error']}")

    A = ack.sweep(front, FRONT_CAMBER, ex.mirror_pts(front), FRONT_CAMBER,
                  wheelbase_mm=wheelbase)
    G = rg.static_geometry(rear_raw, REAR_CAMBER)
    gain = rg.camber_gain_fast(rear_raw, REAR_CAMBER)[0]

    return {
        "f_rc": float(F["rc0"]),
        "f_bump_steer": float(F["bump_steer"]),
        "f_scrub": float(F["scrub0"]),
        "f_trail": float(F["trail0"]),
        "f_kpi": float(F["kpi"]),
        "f_caster": float(F["caster"]),
        "f_gain25": float(F["cam_gain"]) * 25.0,
        "f_ack": float(A["pct_at_zero"]),
        "f_rc_lat": rc_lateral_at_1deg(front, FRONT_CAMBER),
        "f_arm": arm_radius(front, LO) / arm_radius(front, UP),
        "r_rc": float(R["rc0"]),
        "r_bump_steer": float(R["bump_steer"]),
        "r_scrub": float(R["scrub0"]),
        "r_trail": float(R["trail0"]),
        "r_kpi": float(R["kpi"]),
        "r_caster": float(R["caster"]),
        "r_gain25": float(gain) * 25.0,
        "r_recovery": float(rg.recovery_from_gain(gain, G["track"] / 2.0)),
        "r_fvsa": float(G["fvsa"]),
        "r_rc_lat": rc_lateral_at_1deg(rear, REAR_CAMBER),
        "r_arm": arm_radius(rear_raw, LO) / arm_radius(rear_raw, UP),
        "wheelbase": wheelbase,
        "f_track": abs(float(front["Wheel centre"][0])) * 2.0,
        "r_track": abs(float(rear["Wheel centre"][0])) * 2.0,
    }


ROWS = (
    ("FRONT", None, None, None),
    ("roll centre", "f_rc", "{:+.1f} mm", None),
    ("bump steer", "f_bump_steer", "{:+.4f} \u00b0/mm", None),
    ("scrub radius", "f_scrub", "{:.1f} mm", None),
    ("mechanical trail", "f_trail", "{:.1f} mm", None),
    ("KPI", "f_kpi", "{:+.2f} \u00b0", None),
    ("caster", "f_caster", "{:+.2f} \u00b0", None),
    ("camber gain", "f_gain25", "{:+.3f} \u00b0/25 mm", None),
    ("Ackermann, straight ahead", "f_ack", "{:+.1f} %", None),
    ("RC lateral at 1\u00b0 roll", "f_rc_lat", "{:+.0f} mm", None),
    ("arm ratio, lower/upper", "f_arm", "{:.3f}", None),
    ("REAR", None, None, None),
    ("roll centre", "r_rc", "{:+.1f} mm", None),
    ("bump steer", "r_bump_steer", "{:+.4f} \u00b0/mm", None),
    ("scrub radius", "r_scrub", "{:.1f} mm", None),
    ("mechanical trail", "r_trail", "{:.1f} mm", None),
    ("KPI", "r_kpi", "{:+.2f} \u00b0", None),
    ("caster", "r_caster", "{:+.2f} \u00b0", None),
    ("camber gain", "r_gain25", "{:+.3f} \u00b0/25 mm", None),
    ("camber recovery", "r_recovery", "{:.1f} %", None),
    ("FVSA", "r_fvsa", "{:.0f} mm", None),
    ("RC lateral at 1\u00b0 roll", "r_rc_lat", "{:+.0f} mm", None),
    ("arm ratio, lower/upper", "r_arm", "{:.3f}", None),
    ("DIMENSIONS", None, None, None),
    ("wheelbase", "wheelbase", "{:.1f} mm", None),
    ("front track", "f_track", "{:.1f} mm", None),
    ("rear track", "r_track", "{:.1f} mm", None),
)


def show(value, fmt):
    """Format one cell with a typographic minus, so the output can be pasted
    into README.md and compared with it character for character."""
    if value is None:
        return "n/a"
    return fmt.format(value).replace("-", "−")


def main():
    data = {}
    for name, path in SETS:
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")
        data[name] = measure(path)

    print(("Front static camber %+.1f°, rear %+.1f°, both columns.\n"
           % (FRONT_CAMBER, REAR_CAMBER)).replace("-", "−"))
    print("| | starting | redesigned |")
    print("|---|---|---|")
    for label, key, fmt, _ in ROWS:
        if key is None:
            print(f"| **{label.lower()}** | | |")
            continue
        sa = show(data["starting"][key], fmt)
        sb = show(data["redesigned"][key], fmt)
        print(f"| {label} | {sa} | **{sb}** |")


if __name__ == "__main__":
    main()
