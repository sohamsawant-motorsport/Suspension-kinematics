# -*- coding: utf-8 -*-
"""
HARDPOINT AUDIT  --  one page of state, and a diff between two of them.
=======================================================================

WHY IT EXISTS
-------------
Moving a hard point in CAD changes a dozen derived numbers at once, and most of
them are on tabs you are not looking at. A caster tweak that fixes the trail can
quietly move the roll centre, which moves the balance, which moves the required
bar. The explorer SHOWS all of that, but only one panel at a time.

This prints the whole car on one page, and -- more usefully -- DIFFS two of
those pages. The workflow it is built for:

    python hardpoint_audit.py --save before.json     <- before touching CAD
    ...edit the CAD, re-export hardpoints_fusion.csv...
    python hardpoint_audit.py --since before.json    <- what actually changed

Anything that moved shows up, including the things you did not expect to move.
That is the whole point: the coupling is the hard part, not the edit.

WHERE THE NUMBERS COME FROM
---------------------------
The explorer itself, driven headlessly. Not a parallel reimplementation -- if
this disagreed with the tool, the audit would be worse than useless. Every value
here is read out of the same computed state the panels draw from.

THE TARGET COLUMN
-----------------
Only PUBLIC, first-principles bands are used (OptimumG's published ranges, the
load-sensitivity neutral-balance derivation, and hard physical limits like "a
roll centre must be above ground"). No benchmark-car figures appear here, so
this output is safe to publish and safe to reason from on its own.

A blank verdict means there is no defensible public target, not that the number
is fine. Judgement still required.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np


# ---------------------------------------------------------------------------
#  public targets only -- see the module docstring
# ---------------------------------------------------------------------------
#  key: (lo, hi, unit, note). None on a side means unbounded.
TARGETS = {
    "front.roll_centre_mm": (0.0, 60.0, "mm", "must be above ground"),
    "rear.roll_centre_mm": (0.0, 80.0, "mm", "must be above ground"),
    "front.bump_steer_deg_mm": (-0.02, 0.02, "deg/mm", "near zero"),
    "rear.bump_steer_deg_mm": (-0.02, 0.02, "deg/mm", "near zero"),
    "front.ackermann_pct": (80.0, 110.0, "%", "parallel to full Ackermann"),
    "vehicle.roll_gradient_deg_g": (1.0, 3.0, "deg/g", "OptimumG GT band"),
    "vehicle.front_roll_stiff_share_pct": (57.0, 66.0, "%", "public GT range"),
    "vehicle.elastic_fraction_k": (0.75, 0.90, "-", "k>1 means an RC underground"),
    "vehicle.front_lltd_pct": (52.0, 58.0, "%", "neutral ~49-51 + understeer bias"),
    "front.ride_freq_hz": (3.0, 5.0, "Hz", "OptimumG high-downforce band"),
    "rear.ride_freq_hz": (3.0, 5.0, "Hz", "OptimumG high-downforce band"),
}


def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def snapshot():
    """Every derived number the explorer computes, flat, from the live CSV."""
    import explorer as X
    import mech_balance as mbal

    app = X.Explorer()
    app.update()
    try:
        out = {}
        # ---- the raw inputs, so a diff can say WHICH point moved -----------
        pts = {}
        for tag, page in (("FRONT", app.front), ("REAR", app.rear)):
            cur = getattr(page, "cur", None)
            if not X.ok(cur):
                continue
            for name, xyz in cur["R"]["pts"].items():
                pts[f"{tag}.{name}"] = [round(float(q), 4) for q in np.asarray(xyz)]
        out["hardpoints"] = pts

        d = {}
        # ---- front axle ---------------------------------------------------
        cf = getattr(app.front, "cur", None)
        if X.ok(cf):
            r = cf["R"]
            d["front.caster_deg"] = _f(r["caster"])
            d["front.kpi_deg"] = _f(r["kpi"])
            d["front.static_camber_deg"] = _f(r["static_camber"])
            d["front.scrub_radius_mm"] = _f(r["scrub0"])
            d["front.mech_trail_mm"] = _f(r["trail0"])
            d["front.camber_gain_deg_mm"] = _f(r["cam_gain"])
            d["front.bump_steer_deg_mm"] = _f(r["bump_steer"])
            d["front.camber_recovery_pct"] = _f(r["recovery"])
            d["front.roll_centre_mm"] = _f(r["rc0"])
            d["front.damper_motion_ratio"] = _f(r["mr0"])
            d["front.upper_arm_mm"] = _f(r["upper_arm"])
            d["front.lower_arm_mm"] = _f(r["lower_arm"])
            d["front.half_track_mm"] = _f(r["halftrack0"])
            d["front.track_mm"] = _f(cf["track"])
            # ack_t is the kingpin GROUND TRACK in mm, not a percentage -- the
            # %Ackermann has to come from the panel's own reduction, which is
            # the only place inner/outer are picked by TURN DIRECTION rather
            # than by magnitude. (An earlier version of this file printed the
            # ground track as a percentage: 1404 "%".)
            d["front.kingpin_ground_track_mm"] = _f(cf["ack_t"])
            got = X.AckermannPanel._pull(cf)
            if got is not None:
                d["front.ackermann_pct"] = _f(got[4])
        # ---- rear axle ----------------------------------------------------
        cr = getattr(app.rear, "cur", None)
        if X.ok(cr):
            r = cr["R"]
            for src, dst in (("static_camber", "static_camber_deg"),
                             ("cam_gain", "camber_gain_deg_mm"),
                             ("bump_steer", "bump_steer_deg_mm"),
                             ("recovery", "camber_recovery_pct"),
                             ("rc0", "roll_centre_mm"),
                             ("mr0", "damper_motion_ratio"),
                             ("upper_arm", "upper_arm_mm"),
                             ("lower_arm", "lower_arm_mm"),
                             ("halftrack0", "half_track_mm")):
                if src in r:
                    d[f"rear.{dst}"] = _f(r[src])
            if "track" in cr:
                d["rear.track_mm"] = _f(cr["track"])
        # ---- whole car ----------------------------------------------------
        v = app.vehicle
        rc_f, rc_r = v.get_rcs()
        sd = v.spring_design()
        car = v.car
        if sd is not None and rc_f is not None and rc_r is not None:
            kt = 310.4
            for tag in ("f", "r"):
                nm = "front" if tag == "f" else "rear"
                d[f"{nm}.spring_rate_N_mm"] = _f(sd[f"ks_{tag}"])
                d[f"{nm}.wheel_rate_N_mm"] = _f(sd[f"kw_{tag}"])
                d[f"{nm}.ride_freq_hz"] = _f(sd[f"freq_{tag}"])
                d[f"{nm}.ride_freq_tyre_hz"] = _f(sd[f"f_tyre_{tag}"])
                d[f"{nm}.arb_wheel_rate_N_mm"] = _f(car[f"arb_{tag}"])
            st = mbal.state(car, rc_f, rc_r, tyre_rate=kt)
            d["vehicle.front_lltd_pct"] = _f(st["lltd_f"])
            d["vehicle.front_roll_stiff_share_pct"] = _f(st["front_share_pct"])
            d["vehicle.roll_gradient_deg_g"] = _f(st["roll_grad"])
            d["vehicle.elastic_fraction_k"] = _f(st["elastic_frac"])
            d["vehicle.roll_axis_height_mm"] = _f(st["roll_axis"])
            d["vehicle.roll_lever_mm"] = _f(st["lever"])
            d["vehicle.roll_stiffness_MNmm_rad"] = _f(st["k_roll_total"]) / 1e6
            d["vehicle.wheelbase_mm"] = _f(car["wheelbase_mm"])
            d["vehicle.cg_height_mm"] = _f(car["h_cg_mm"])
            d["vehicle.mass_kg"] = _f(car["mass_total_kg"])
            d["vehicle.rear_weight_frac"] = _f(car["rear_frac"])
        out["values"] = d
        out["meta"] = dict(when=datetime.now().isoformat(timespec="seconds"),
                           csv=os.path.abspath(
                               os.path.join(os.path.dirname(__file__), "..",
                                            "hardpoints_fusion.csv")))
        return out
    finally:
        app.destroy()


def verdict(key, val):
    """Public-target verdict for one value, or '' if there is no public target."""
    t = TARGETS.get(key)
    if t is None or not np.isfinite(val):
        return ""
    lo, hi, _unit, note = t
    if lo is not None and val < lo:
        return f"LOW  (want {lo:g}..{hi:g}; {note})"
    if hi is not None and val > hi:
        return f"HIGH (want {lo:g}..{hi:g}; {note})"
    return f"ok   ({lo:g}..{hi:g})"


def show(snap):
    d = snap["values"]
    group = None
    print(f"{'parameter':42s}{'value':>14}   verdict (public targets only)")
    print("-" * 100)
    for k in sorted(d):
        g = k.split(".")[0]
        if g != group:
            group = g
            print(f"\n[{g.upper()}]")
        print(f"  {k:40s}{d[k]:14.4f}   {verdict(k, d[k])}")
    bad = [k for k in d if verdict(k, d[k]).startswith(("LOW", "HIGH"))]
    print("\n" + "-" * 100)
    print(f"{len(d)} parameters,  {len(bad)} outside a public target band"
          + (":" if bad else ""))
    for k in bad:
        print(f"   - {k}")


def diff(a, b, tol=1e-9):
    """What changed between two snapshots -- hard points first, then effects."""
    pa, pb = a.get("hardpoints", {}), b.get("hardpoints", {})
    moved = []
    for k in sorted(set(pa) | set(pb)):
        if k not in pa or k not in pb:
            moved.append((k, None, None, None))
            continue
        va, vb = np.asarray(pa[k], float), np.asarray(pb[k], float)
        if np.max(np.abs(vb - va)) > 1e-4:
            moved.append((k, va, vb, float(np.linalg.norm(vb - va))))
    print("HARD POINTS THAT MOVED")
    if not moved:
        print("  (none)")
    for k, va, vb, dist in moved:
        if va is None:
            print(f"  {k:46s}  added or removed")
        else:
            print(f"  {k:46s}  d = {vb - va}   |d| = {dist:.3f} mm")

    da, db = a["values"], b["values"]
    print("\nWHAT THAT DID")
    rows = []
    for k in sorted(set(da) | set(db)):
        x, y = da.get(k), db.get(k)
        if x is None or y is None:
            rows.append((k, x, y, None))
        elif abs(y - x) > tol:
            rows.append((k, x, y, y - x))
    if not rows:
        print("  (nothing changed)")
        return 0
    print(f"  {'parameter':40s}{'before':>13}{'after':>13}{'change':>13}   verdict")
    for k, x, y, dd in rows:
        if dd is None:
            print(f"  {k:40s}{'-' if x is None else x:>13}"
                  f"{'-' if y is None else y:>13}{'new/gone':>13}")
        else:
            print(f"  {k:40s}{x:13.4f}{y:13.4f}{dd:+13.4f}   {verdict(k, y)}")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--save", metavar="FILE", help="write a JSON snapshot")
    ap.add_argument("--since", metavar="FILE",
                    help="diff the CURRENT state against a saved snapshot")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"),
                    help="diff two saved snapshots")
    a = ap.parse_args(argv)

    if a.diff:
        with open(a.diff[0]) as f1, open(a.diff[1]) as f2:
            diff(json.load(f1), json.load(f2))
        return 0

    snap = snapshot()
    if a.since:
        with open(a.since) as f:
            old = json.load(f)
        print(f"BEFORE  {old['meta']['when']}\nAFTER   {snap['meta']['when']}\n")
        diff(old, snap)
    else:
        show(snap)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(snap, f, indent=1)
        print(f"\nsnapshot written to {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
