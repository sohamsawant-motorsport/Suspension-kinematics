# -*- coding: utf-8 -*-
"""
HOW MUCH OF THE MIGRATION DROP CAME FROM LENGTHENING THE ARMS?
==============================================================

    python hardpoints/verify_lengthening.py

README.md claims that lengthening both wishbones along their own lines cuts
roll-centre migration while leaving every static value alone. The before/after
table cannot show that, because its two columns differ by the whole redesign:
roll-centre height, KPI, caster and scrub all moved as well. So the claim is
checked here instead, by building the intermediate geometry the table does not
contain.

Three geometries per axle:

    A   starting, as shipped in ../hardpoints_fusion.csv
    B   starting with ONLY the arm radii changed to the redesigned values.
        Nothing else is touched.
    C   redesigned, as shipped in hardpoints_redesigned.csv

THE OPERATION. A wishbone's ball joint B traces a circle of radius r about the
arm's inner hinge axis. Translate both inner pickups by -t*d, where d is the
unit radial direction from the axis foot to B:

    the axis DIRECTION is unchanged
    the axis FOOT moves by -t*d, so the radius becomes r + t
    B itself does not move
    B's velocity direction, perpendicular to both the axis and d, does not move

Position and first-order motion are therefore both frozen. The only thing that
changes is the CURVATURE, 1/r. That is the whole trick: migration is a
second-order effect, so it can be bought without paying anything first-order.

WHAT THE RESULT SHOWS. On both axles the lengthening alone is worth MORE than
the net figure in the table: the later redesign steps, which raised the roll
centre and fixed KPI, caster and scrub, gave some migration back. Quoting the
table's column-to-column drop as the value of the lengthening understates it on
the rear and overstates nothing.
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

import config as cfg
import explorer as ex
import rear_geometry as rg

UPPER = ("Upper wishbone - outer (UBJ)", "Upper wishbone - inner front",
         "Upper wishbone - inner rear")
LOWER = ("Lower wishbone - outer (LBJ)", "Lower wishbone - inner front",
         "Lower wishbone - inner rear")

AXLES = (("FRONT", -2.7), ("REAR", -1.5))

STATICS = (("roll centre", "rc", "{:+.1f}"), ("KPI", "kpi", "{:+.2f}"),
           ("caster", "caster", "{:+.2f}"), ("scrub", "scrub", "{:.1f}"),
           ("trail", "trail", "{:.1f}"), ("camber gain /25", "gain25", "{:+.3f}"),
           ("bump steer", "bump_steer", "{:+.5f}"))


def arm(pts, trio):
    """Perpendicular radius from the ball joint to the inner hinge axis, and
    the unit radial direction pointing from the axis foot out to the joint."""
    B, A, C = (np.asarray(pts[k], float) for k in trio)
    u = (C - A) / np.linalg.norm(C - A)
    foot = A + np.dot(B - A, u) * u
    v = B - foot
    r = float(np.linalg.norm(v))
    return r, v / r


def set_arm_radius(pts, trio, r_new):
    """Slide the two inner pickups along the arm line until the radius is
    r_new. The ball joint is not touched."""
    out = {k: np.array(v, float) for k, v in pts.items()}
    r, d = arm(out, trio)
    shift = (r_new - r) * d
    for pickup in trio[1:]:
        out[pickup] = out[pickup] - shift
    return out


def load(csv_path, corner):
    cfg.CSV = csv_path
    pts = cfg._load_points(corner)
    return rg.solver_points(pts) if corner == "REAR" else pts


def measure(pts, camber):
    g = ex.corner_compute(pts, camber, False)
    if g.get("error"):
        raise SystemExit("did not solve: " + g["error"])
    sweep = ex.roll_migration(pts, camber, ex.mirror_pts(pts), camber)
    lateral = None
    if sweep is not None:
        lateral = float(np.interp(1.0, np.asarray(sweep["roll"], float),
                                  np.asarray(sweep["rc_x"], float)))
    return dict(rc=float(g["rc0"]), kpi=float(g["kpi"]),
                caster=float(g["caster"]), scrub=float(g["scrub0"]),
                trail=float(g["trail0"]), gain25=float(g["cam_gain"]) * 25.0,
                bump_steer=float(g["bump_steer"]), rc_lat=lateral)


def main():
    start_csv = os.path.join(_ROOT, "hardpoints_fusion.csv")
    redes_csv = os.path.join(_HERE, "hardpoints_redesigned.csv")
    for path in (start_csv, redes_csv):
        if not os.path.exists(path):
            raise SystemExit("missing: " + path)

    for corner, camber in AXLES:
        A = load(start_csv, corner)
        C = load(redes_csv, corner)

        r_up, _ = arm(C, UPPER)
        r_lo, _ = arm(C, LOWER)
        B = set_arm_radius(set_arm_radius(A, UPPER, r_up), LOWER, r_lo)

        a, b, c = (measure(p, camber) for p in (A, B, C))

        print("=" * 72)
        print(f"{corner}   static camber {camber:+.1f} deg")
        print("=" * 72)

        print(f"  arm radius, perpendicular to the inner hinge axis [mm]")
        for tag, pts in (("A starting", A), ("B lengthened", B), ("C redesigned", C)):
            u, _ = arm(pts, UPPER)
            lo, _ = arm(pts, LOWER)
            print(f"    {tag:<14s} upper {u:7.2f}   lower {lo:7.2f}   "
                  f"lower/upper {lo / u:.4f}")

        print(f"\n  did the lengthening freeze the statics?  B minus A")
        worst = 0.0
        for label, key, fmt in STATICS:
            d = b[key] - a[key]
            worst = max(worst, abs(d))
            print(f"    {label:<16s} A {fmt.format(a[key]):>10s}   "
                  f"B {fmt.format(b[key]):>10s}   delta {d:+.9f}")
        print(f"    worst absolute change: {worst:.2e}")

        if None in (a["rc_lat"], b["rc_lat"], c["rc_lat"]):
            print("\n  roll migration unavailable on this axle\n")
            continue

        aa, bb, cc = (abs(x["rc_lat"]) for x in (a, b, c))
        print(f"\n  roll-centre lateral movement at 1 deg of body roll [mm]")
        print(f"    A starting                     {aa:8.1f}")
        print(f"    B lengthening alone            {bb:8.1f}   "
              f"({aa / bb:.1f}x less than A)")
        print(f"    C redesigned                   {cc:8.1f}   "
              f"({aa / cc:.1f}x less than A)")
        print(f"    the later steps gave back      {cc - bb:8.1f}\n")


if __name__ == "__main__":
    main()
