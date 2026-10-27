# -*- coding: utf-8 -*-
"""
TASK 6  -  Damper deflection and front roll vs steer, swept by SCRUB and TRAIL.
           (static PNG)

    6a  damper deflection vs steer, one curve per SCRUB   (trail constant)
    6b  damper deflection vs steer, one curve per TRAIL   (scrub constant)
    6c  front roll       vs steer, one curve per SCRUB   (trail constant)
    6d  front roll       vs steer, one curve per TRAIL   (scrub constant)

MODEL (see jacking.py for the full statement)
    * Steering jacks each corner vertically. The LEFT corner is the mirror of the
      right, so jack_L(d) = jack_R(-d) -- one corner's data gives the whole axle.
    * Damper deflection = jack * MR, with the chassis held  (locked-chassis limit).
    * Front roll = atan((jack_L - jack_R)/track), no chassis model needed
      (stiff-spring limit). The truth lies between the two limits.
    * MR is held constant across the sweeps: scrub/trail are the design variables
      here, and the damper mount is treated as a separate choice.

Both corners are plotted, because they do NOT move oppositely -- with KPI != 0 there
is always a common heave term, and on this car it dominates.

Output: ../damper_roll_by_offset.png   (project root)
"""

import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_ROOT, "software"), _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_PLOTS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "plots")
_os.makedirs(_PLOTS, exist_ok=True)


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from scrub_trail import offset_joints, scrub_radius, mechanical_trail, kpi_caster
from jacking import motion_ratio, axle_state, front_track

p = cfg._load_front_points()
LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]
CAMBER = cfg.STATIC_CAMBER
kpi, cas = kpi_caster(LBJ, UBJ)
MR = motion_ratio(p["Lower wishbone - inner front"], p["Lower wishbone - inner rear"],
                  LBJ, p["Lower Shock mount"], p["Upper Shock mount"])
TRACK = front_track(p)

steer = np.linspace(cfg.STEER_RANGE_PATCH[0], cfg.STEER_RANGE_PATCH[1], 281)
dxs = [float(v) for v in cfg.LATERAL_OFFSET_SWEEP]
dys = [float(v) for v in cfg.LONG_OFFSET_SWEEP]

fig, axs = plt.subplots(2, 2, figsize=(15.5, 10.5))
cmap, cmap2 = plt.cm.viridis, plt.cm.plasma
lo_y, hi_y = min(dys), max(dys)


def build(dx=0.0, dy=0.0):
    L2, U2 = offset_joints(LBJ, UBJ, dx=dx, dy=dy)
    st = axle_state(L2, U2, WC, steer, CAMBER, MR, TRACK)
    st["sc"] = scrub_radius(L2, U2, WC, CAMBER)
    st["tr"] = mechanical_trail(L2, U2, WC, CAMBER)
    return st


by_dx = {dx: build(dx=dx) for dx in dxs}
by_dy = {dy: build(dy=dy) for dy in dys}


def draw(ax, key, sweep, data, cfn, lfn, title, ylab, show_left=False):
    for v in sweep:
        base = abs(v) < 1e-9
        c = "crimson" if base else cfn(v)
        ax.plot(steer, data[v][key], lw=3.0 if base else 1.5, color=c,
                zorder=6 if base else 2,
                label=lfn(v) + ("  ← baseline" if base else ""))
        if show_left and base:
            ax.plot(steer, data[v]["defl_L"], lw=2.0, color="crimson", ls="--",
                    zorder=6, label="baseline, LEFT corner")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(cfg.STEER_RANGE_PATCH)
    ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)


draw(axs[0, 0], "defl_R", dxs, by_dx, lambda v: cmap(v / max(dxs)),
     lambda v: f"scrub {by_dx[v]['sc']:+.0f} mm",
     f"6a  DAMPER DEFLECTION vs steer — swept by SCRUB\n"
     f"trail held {by_dx[0.0]['tr']:.1f} mm · MR {MR:.3f} · RIGHT corner (dashed = LEFT)",
     "damper deflection  [mm]   (− = corner drops)", show_left=True)
draw(axs[0, 1], "defl_R", dys, by_dy, lambda v: cmap2((v - lo_y) / (hi_y - lo_y)),
     lambda v: f"trail {by_dy[v]['tr']:+.0f} mm",
     f"6b  DAMPER DEFLECTION vs steer — swept by TRAIL\n"
     f"scrub held {by_dy[0.0]['sc']:.1f} mm · MR {MR:.3f} · RIGHT corner",
     "damper deflection  [mm]   (− = corner drops)")
draw(axs[1, 0], "roll", dxs, by_dx, lambda v: cmap(v / max(dxs)),
     lambda v: f"scrub {by_dx[v]['sc']:+.0f} mm",
     "6c  FRONT ROLL vs steer — swept by SCRUB\n"
     "roll = atan((jack_L − jack_R)/track) — the ODD part of the jacking",
     "front roll  [deg]")
draw(axs[1, 1], "roll", dys, by_dy, lambda v: cmap2((v - lo_y) / (hi_y - lo_y)),
     lambda v: f"trail {by_dy[v]['tr']:+.0f} mm",
     "6d  FRONT ROLL vs steer — swept by TRAIL\n"
     "roll is ODD in steer — it reverses with steer direction, as it must",
     "front roll  [deg]")

fig.suptitle("Task 6 — steering jacking: damper deflection & the roll it forces "
             f"(KPI {kpi:+.2f}°, caster {cas:+.2f}°, camber {CAMBER:+.1f}°)\n"
             "deflection = locked-chassis limit · roll = stiff-spring limit · "
             "left corner from mirror symmetry (parallel steer)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(cfg.PROJ, "damper_roll_by_offset.png")
fig.savefig(_os.path.join(_PLOTS, "task6_damper_roll.png"), dpi=135, bbox_inches="tight")
print("saved", out)

# ---- the numbers that matter ------------------------------------------------
print()
print(f"motion ratio {MR:.4f} (first-order)   front track {TRACK:.1f} mm")
print()
print("at 35 deg steer:")
print("   scrub    trail     defl_R    defl_L     heave      roll")
for dx in dxs:
    st = by_dx[dx]
    i = -1
    print(f"  {st['sc']:+7.1f}  {st['tr']:+7.1f}  {st['defl_R'][i]:+8.3f}  "
          f"{st['defl_L'][i]:+8.3f}  {st['heave'][i]:+8.3f}  {st['roll'][i]:+8.4f}")
for dy in dys:
    if abs(dy) < 1e-9:
        continue
    st = by_dy[dy]
    i = -1
    print(f"  {st['sc']:+7.1f}  {st['tr']:+7.1f}  {st['defl_R'][i]:+8.3f}  "
          f"{st['defl_L'][i]:+8.3f}  {st['heave'][i]:+8.3f}  {st['roll'][i]:+8.4f}")
