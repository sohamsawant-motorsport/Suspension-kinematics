# -*- coding: utf-8 -*-
"""
Single source of truth for the Huracan GT3 hard points, as extracted BY HAND in
Fusion (supersedes the old mesh-centroid CSV for the outboard ball joints).

Frame (as recorded in the sheet):
    X_mm = lateral   ("Width")            outboard +
    Y_mm = longitudinal ("length")        forward +
    Z_mm = vertical  ("vertical Hight")   up +

ORIGIN
------
The points below were recorded against Fusion "origin 2", which is offset from the
true car origin ("origin 1"). Set ORIGIN_OFFSET to the position of origin 1 as read
in the origin-2 frame; every point is then shifted by  new = old - ORIGIN_OFFSET.

    ORIGIN_OFFSET is SET and APPLIED: (-18.43, -5.122, +0.232).
    PROVEN, not just measured: with an FL wheel-centre point placed in CAD, the
    FR/FL pivots read 812.773 / 850.878 about the old origin -> 38.1 mm asymmetric;
    about the true origin they read 831.203 / 832.447 -> 1.24 mm. The offset
    collapses the asymmetry, so it is right. (The 1.24 mm residual is the artistic
    mesh not being perfectly symmetric.) Front track, pivot-to-pivot = 1663.65 mm.

NOTE: a pure translation of the origin does NOT change any angle (KPI, caster),
any length between points, or the steering arm - those are all built from
DIFFERENCES between points, so the offset cancels. It only matters for quantities
referenced to the car datum / ground plane: scrub radius, ride height, track,
wheelbase, roll-centre height.

THIS SCRIPT DOES NOT OVERWRITE THE PROJECT CSV UNLESS YOU ASK IT TO.

    python rebase_hardpoints.py                 print the rebased table
    python rebase_hardpoints.py -o some.csv     write it somewhere new
    python rebase_hardpoints.py --write         overwrite ../hardpoints_fusion.csv

It used to overwrite the project CSV the moment it was run, with no flag and no
confirmation, which is a bad thing for a file every other module reads on
import. Worse, POINTS below is a transcription of the CAD sheet and the sheet
never had the rear toe link, so a stray run silently deleted two rows and left
the rear axle unsolvable with a confusing error several modules downstream.

So: writing is opt-in, and a write CARRIES OVER any row already in the target
file that POINTS does not contain, rather than dropping it. Carried rows are
listed on stdout so nothing moves silently.

Run:  python rebase_hardpoints.py --write   ->  ../hardpoints_fusion.csv
"""
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "hardpoints_fusion.csv")

# ---------------------------------------------------------------------------
# Position of the TRUE car origin (origin 1) as read in the recorded (origin 2)
# frame.  Applied as:  new = old - ORIGIN_OFFSET
#
# From the user's CAD measurement (confirmed 2026-07-16):
#   X: true origin is 18.43 mm to the RIGHT viewed from the FRONT of the car.
#      Facing the car, your right = the car's LEFT = -X here.   -> X = -18.43
#      => new_X = old_X + 18.43
#   Y: true origin is 5.122 mm BEHIND the current origin.       -> Y = -5.122
#      => new_Y = old_Y + 5.122
#   Z: true origin is 0.232 mm ABOVE the current origin.        -> Z = +0.232
#      => new_Z = old_Z - 0.232
ORIGIN_OFFSET = (-18.43, -5.122, 0.232)     # (X lateral, Y longitudinal, Z vertical)
# ---------------------------------------------------------------------------

RIM_OFFSET = {"FRONT": 47.382, "REAR": 42.44}   # mm, for reference

# Transcribed verbatim from the Fusion hard-point sheet ("HURACAN Hardpoints from CAD")
POINTS = [
    # corner, point,                        X_mm,      Y_mm,       Z_mm
    ("FRONT", "Upper wishbone - outer (UBJ)", 720.303, 1235.319,  397.366),
    ("FRONT", "Upper wishbone - inner front", 519.596, 1393.250,  323.599),
    ("FRONT", "Upper wishbone - inner rear",  519.596, 1155.827,  323.599),
    ("FRONT", "Lower wishbone - outer (LBJ)", 706.858, 1239.542,  251.543),
    ("FRONT", "Lower wishbone - inner front", 503.759, 1351.736,  179.064),
    ("FRONT", "Lower wishbone - inner rear",  503.760, 1155.966,  179.061),
    ("FRONT", "Tie rod - outer",              792.941, 1363.086,  321.827),
    ("FRONT", "Tie rod - inner",              418.414, 1319.860,  275.009),
    ("FRONT", "Wheel centre",                 812.773, 1215.671,  336.398),
    ("FRONT", "Lower Shock mount",            712.362, 1240.576,  259.190),
    ("FRONT", "Upper Shock mount",            443.483, 1218.122,  520.492),
    ("REAR",  "Upper wishbone - outer (UBJ)", 696.050, -1443.814, 437.735),
    ("REAR",  "Upper wishbone - inner front", 453.666, -1311.860, 396.853),
    ("REAR",  "Upper wishbone - inner rear",  453.666, -1525.319, 396.852),
    ("REAR",  "Lower wishbone - outer (LBJ)", 687.966, -1417.854, 268.115),
    ("REAR",  "Lower wishbone - inner front", 481.345, -1284.767, 241.175),
    ("REAR",  "Lower wishbone - inner rear",  481.335, -1588.713, 242.174),
    # ("REAR", "Toe link - outer", ...)   <- still blank in the sheet
    # ("REAR", "Toe link - inner", ...)   <- still blank in the sheet
    ("REAR",  "Wheel centre",                 804.460, -1429.330, 347.774),
    ("REAR",  "Lower Shock mount",            691.768, -1431.967, 420.552),
    ("REAR",  "Upper Shock mount",            372.191, -1431.967, 762.929),
    ("REAR",  "Drive shaft - outer",          811.692, -1423.903, 364.883),
    ("REAR",  "Drive shaft - Inner",          429.763, -1423.903, 296.486),
]


def rebased():
    ox, oy, oz = ORIGIN_OFFSET
    for corner, name, x, y, z in POINTS:
        yield corner, name, round(x - ox, 3), round(y - oy, 3), round(z - oz, 3)


def extra_rows(path):
    """Rows in an existing CSV that POINTS does not contain.

    The rear toe link is the live case: it is not in the CAD sheet, so it is not
    in POINTS, but it IS in the project CSV and the rear corner cannot be solved
    without it. Anything else added to the file by hand is preserved the same
    way. Returns a list of raw lines, in file order.
    """
    if not os.path.exists(path):
        return []
    known = {(corner, name) for corner, name, *_ in POINTS}
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("Corner,"):
                continue
            parts = s.split(",")
            if len(parts) < 5:
                continue
            if (parts[0].strip(), parts[1].strip()) not in known:
                out.append(s)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Rebase the CAD hard points onto the true car origin.")
    ap.add_argument("-o", "--out", metavar="PATH",
                    help="write the table to PATH")
    ap.add_argument("--write", action="store_true",
                    help=f"overwrite the project CSV ({os.path.basename(OUT)})")
    args = ap.parse_args()

    target = args.out or (OUT if args.write else None)
    carried = extra_rows(target) if target else []

    lines = ["Corner,Point,X_mm,Y_mm,Z_mm"]
    lines += [f"{c},{n},{x},{y},{z}" for c, n, x, y, z in rebased()]
    lines += carried

    if target is None:
        print("\n".join(lines))
        print(f"\n-- nothing written. Use --write to overwrite {OUT},"
              f" or -o PATH for somewhere else.")
    else:
        with open(target, "w", newline="", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {target}")

    applied = any(v != 0.0 for v in ORIGIN_OFFSET)
    print(f"origin offset applied: {ORIGIN_OFFSET}"
          f"{'' if applied else '   <-- NOT SET: points are as-recorded (origin 2)'}")
    print(f"rows: {len(POINTS)} from the sheet"
          f"{f', {len(carried)} carried over from the existing file' if carried else ''}")
    for row in carried:
        print(f"   carried: {row}")
