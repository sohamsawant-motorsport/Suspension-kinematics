# -*- coding: utf-8 -*-
"""
VALIDATION SUITE for the hardpoint audit.
=========================================

The audit is the thing you will trust while editing CAD, so it has to be
trustworthy in a specific way: every number must be the SAME number the tool
shows, under the SAME name.

This suite exists because the first version of `hardpoint_audit.py` printed the
kingpin GROUND TRACK (1404 mm) in a field labelled `ackermann_pct`, and it read
as "1404 % Ackermann". Nothing crashed, nothing was out of range in a way that
looked wrong -- it was just a lie with a plausible shape. So:

  1  STRUCTURE    - the snapshot has the three sections, every value is finite,
                    and the hard points come through as real coordinates.
  2  NOT A LIE    - each value is cross-checked against the module that owns it,
                    computed independently. The Ackermann/ground-track mix-up is
                    asserted against specifically: a percentage may not be a
                    length, and the two fields may not be equal.
  3  INTERNAL     - the relations that must hold between reported values
                    (wheel rate = spring x MR^2, and so on).
  4  VERDICTS     - the public-target classifier is right at and around its
                    edges, and stays silent where there is no public target.
  5  DIFF         - identical snapshots report nothing; a moved hard point is
                    found with the right distance; a changed value is listed;
                    and the whole thing survives a JSON round trip.

No benchmark-car figures are used anywhere here, matching the audit itself.
"""

import copy
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np

import ackermann
import config as cfg
import hardpoint_audit as hpa
import mech_balance as mb
from scrub_trail import kpi_caster

fails = []


def check(name, ok_, detail=""):
    print(f"  [{'PASS' if ok_ else '**FAIL**'}] {name}"
          + (f"   {detail}" if detail else ""))
    if not ok_:
        fails.append(name)


print("\n  (building one snapshot -- this drives the real explorer, ~5 s)")
SNAP = hpa.snapshot()
V = SNAP["values"]
P = SNAP["hardpoints"]


# ===========================================================================
print("\n1  STRUCTURE")
# ===========================================================================
check("snapshot has values, hardpoints and meta",
      all(k in SNAP for k in ("values", "hardpoints", "meta")))
check("a useful number of parameters", len(V) >= 40, f"{len(V)}")
bad = [k for k, x in V.items() if not np.isfinite(x)]
check("every reported value is finite", not bad, str(bad))
check("hard points came through for both corners",
      any(k.startswith("FRONT.") for k in P)
      and any(k.startswith("REAR.") for k in P), f"{len(P)} points")
check("each hard point is an XYZ triple",
      all(len(v) == 3 for v in P.values()))
check("every key is group-qualified",
      all(k.split(".")[0] in ("front", "rear", "vehicle") for k in V))


# ===========================================================================
print("\n2  EVERY VALUE CROSS-CHECKED AGAINST ITS OWNING MODULE")
# ===========================================================================
pf = cfg._load_front_points()
kpi_x, cas_x = kpi_caster(pf["Lower wishbone - outer (LBJ)"],
                          pf["Upper wishbone - outer (UBJ)"])
check("front KPI == scrub_trail.kpi_caster(LBJ, UBJ)",
      abs(V["front.kpi_deg"] - kpi_x) < 1e-6,
      f"{V['front.kpi_deg']:.4f} vs {kpi_x:.4f}")
check("front caster == the same call's second return",
      abs(V["front.caster_deg"] - cas_x) < 1e-6,
      f"{V['front.caster_deg']:.4f} vs {cas_x:.4f}")
check("...and the argument order matters, so this is not the supplement",
      abs(kpi_caster(pf["Upper wishbone - outer (UBJ)"],
                     pf["Lower wishbone - outer (LBJ)"])[0] - kpi_x) > 1.0)

pl = {k: np.array([-v[0], v[1], v[2]]) for k, v in pf.items()}
sw = ackermann.sweep(pf, cfg.STATIC_CAMBER, pl, cfg.STATIC_CAMBER)
check("front kingpin ground track == ackermann.sweep's t_mm",
      abs(V["front.kingpin_ground_track_mm"] - sw["t_mm"]) < 1e-6,
      f"{V['front.kingpin_ground_track_mm']:.4f} mm")
check("front %Ackermann agrees with an independent sweep",
      abs(V["front.ackermann_pct"] - sw["pct_at_zero"]) < 1.0,
      f"{V['front.ackermann_pct']:.3f} vs {sw['pct_at_zero']:.3f} %")

#  the exact failure this suite was written for
check("a PERCENTAGE field is not holding a LENGTH",
      V["front.ackermann_pct"] < 500.0,
      f"{V['front.ackermann_pct']:.1f} % -- a track would read ~1400")
check("ackermann_pct and kingpin_ground_track_mm are different numbers",
      abs(V["front.ackermann_pct"] - V["front.kingpin_ground_track_mm"]) > 1.0)
check("the ground track is a plausible length, not a percentage",
      1000.0 < V["front.kingpin_ground_track_mm"] < 2500.0)

check("front track == 2 x half track",
      abs(V["front.track_mm"] - 2 * V["front.half_track_mm"]) < 1e-6)
check("rear track == 2 x half track",
      abs(V["rear.track_mm"] - 2 * V["rear.half_track_mm"]) < 1e-6)

import load_transfer as ltr                                        # noqa: E402
car = ltr.temerario_gt3()
car["kw_f"] = V["front.wheel_rate_N_mm"]
car["kw_r"] = V["rear.wheel_rate_N_mm"]
car["arb_f"] = V["front.arb_wheel_rate_N_mm"]
car["arb_r"] = V["rear.arb_wheel_rate_N_mm"]
car["h_cg_mm"] = V["vehicle.cg_height_mm"]
car["mass_total_kg"] = V["vehicle.mass_kg"]
car["rear_frac"] = V["vehicle.rear_weight_frac"]
st = mb.state(car, V["front.roll_centre_mm"], V["rear.roll_centre_mm"],
              tyre_rate=310.4)
for key, got, want in (("front LLTD", V["vehicle.front_lltd_pct"], st["lltd_f"]),
                       ("front roll-stiffness share",
                        V["vehicle.front_roll_stiff_share_pct"],
                        st["front_share_pct"]),
                       ("roll gradient", V["vehicle.roll_gradient_deg_g"],
                        st["roll_grad"]),
                       ("elastic fraction k", V["vehicle.elastic_fraction_k"],
                        st["elastic_frac"])):
    check(f"{key} == mech_balance.state recomputed from the reported inputs",
          abs(got - want) < 1e-6, f"{got:.6f} vs {want:.6f}")


# ===========================================================================
print("\n3  INTERNAL CONSISTENCY")
# ===========================================================================
for tag in ("front", "rear"):
    kw = V[f"{tag}.spring_rate_N_mm"] * V[f"{tag}.damper_motion_ratio"] ** 2
    check(f"{tag}: wheel rate == spring rate x MR^2",
          abs(V[f"{tag}.wheel_rate_N_mm"] - kw) < 1e-6,
          f"{V[f'{tag}.wheel_rate_N_mm']:.4f} vs {kw:.4f}")
    check(f"{tag}: the tyre-inclusive ride frequency is the LOWER one",
          V[f"{tag}.ride_freq_tyre_hz"] < V[f"{tag}.ride_freq_hz"])
    check(f"{tag}: motion ratio is physically plausible",
          0.1 < V[f"{tag}.damper_motion_ratio"] <= 1.2)
check("roll axis height lies between the two roll centres",
      min(V["front.roll_centre_mm"], V["rear.roll_centre_mm"])
      <= V["vehicle.roll_axis_height_mm"]
      <= max(V["front.roll_centre_mm"], V["rear.roll_centre_mm"]))
check("roll lever == CG height - roll axis height",
      abs(V["vehicle.roll_lever_mm"]
          - (V["vehicle.cg_height_mm"] - V["vehicle.roll_axis_height_mm"]))
      < 1e-6)
check("front LLTD is a percentage", 0.0 < V["vehicle.front_lltd_pct"] < 100.0)


# ===========================================================================
print("\n4  VERDICTS")
# ===========================================================================
K = "vehicle.roll_gradient_deg_g"          # public band 1.0 .. 3.0
check("below the band reads LOW", hpa.verdict(K, 0.5).startswith("LOW"))
check("inside the band reads ok", hpa.verdict(K, 2.0).startswith("ok"))
check("above the band reads HIGH", hpa.verdict(K, 9.0).startswith("HIGH"))
check("the band edges are inclusive",
      hpa.verdict(K, 1.0).startswith("ok") and hpa.verdict(K, 3.0).startswith("ok"))
check("a key with no public target stays silent",
      hpa.verdict("front.caster_deg", 1.66) == "")
check("a NaN stays silent rather than guessing",
      hpa.verdict(K, float("nan")) == "")
check("every target key actually exists in the snapshot",
      all(k in V for k in hpa.TARGETS), str([k for k in hpa.TARGETS if k not in V]))
check("an underground roll centre is reported as out of band",
      hpa.verdict("front.roll_centre_mm", -283.6).startswith("LOW"))


# ===========================================================================
print("\n5  DIFF")
# ===========================================================================
same = hpa.diff(SNAP, copy.deepcopy(SNAP))
check("a snapshot against itself reports no change", same == 0, f"{same} rows")

moved = copy.deepcopy(SNAP)
key = "FRONT.Lower wishbone - inner front"
moved["hardpoints"][key] = [moved["hardpoints"][key][0],
                            moved["hardpoints"][key][1],
                            moved["hardpoints"][key][2] + 12.0]
moved["values"]["front.roll_centre_mm"] += 31.5
n = hpa.diff(SNAP, moved)
check("a moved hard point and its effect are both reported", n == 1, f"{n} rows")

rt = json.loads(json.dumps(SNAP))
check("a JSON round trip changes nothing", hpa.diff(SNAP, rt) == 0)
fd, path = tempfile.mkstemp(suffix=".json")
os.close(fd)
try:
    with open(path, "w") as f:
        json.dump(SNAP, f)
    with open(path) as f:
        check("a snapshot survives being written and read back",
              hpa.diff(SNAP, json.load(f)) == 0)
finally:
    os.unlink(path)

gone = copy.deepcopy(SNAP)
gone["values"].pop("front.caster_deg")
check("a parameter that disappears is reported, not skipped",
      hpa.diff(SNAP, gone) == 1)


print("\n" + "=" * 76)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f_ in fails:
        print("   -", f_)
    raise SystemExit(1)
print("ALL CHECKS PASSED - the audit reports the tool's own numbers, under names\n"
      "that mean what they say, and the diff finds both the hard point that moved\n"
      "and everything that moved because of it.")
print("=" * 76)
