# -*- coding: utf-8 -*-
"""
TASK 4  -  Scrub radius and mechanical trail vs steer, swept by BALL-JOINT OFFSET.
           (static PNG)

Scrub and trail are lengths, so the design lever is sliding BOTH ball joints
together -- which leaves the KPI and caster ANGLES untouched:
    dx (lateral)      -> scrub(dx) = scrub(0) - dx
    dy (longitudinal) -> trail(dy) = trail(0) + dy

Left  : scrub radius vs steer, one curve per lateral offset dx.
Right : mechanical trail vs steer, one curve per longitudinal offset dy.
The baseline (dx = dy = 0) is highlighted.

Output: ../scrub_trail_by_offset.png   (project root)
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
from scrub_trail import (corner_state, offset_joints, kpi_caster,
                         scrub_radius, mechanical_trail)

p = cfg._load_front_points()
LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]
CAMBER = cfg.STATIC_CAMBER   # static camber tilts the wheel -> moves the contact patch
kpi, cas = kpi_caster(LBJ, UBJ)
steer = np.linspace(cfg.STEER_RANGE_PATCH[0], cfg.STEER_RANGE_PATCH[1], 281)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.5, 6.4))

# ---------------- left: scrub vs steer, swept by lateral offset dx ----------
cmap = plt.cm.viridis
dxs = [float(v) for v in cfg.LATERAL_OFFSET_SWEEP]
for dx in dxs:
    L2, U2 = offset_joints(LBJ, UBJ, dx=dx)
    st = corner_state(L2, U2, WC, steer, CAMBER)
    base = abs(dx) < 1e-9
    axa.plot(steer, st["scrub"], lw=3.0 if base else 1.6,
             color="crimson" if base else cmap(dx / max(dxs)),
             zorder=6 if base else 2,
             label=f"dx {dx:+.0f} mm  (scrub₀ {scrub_radius(L2,U2,WC, CAMBER):+.0f})"
                   + ("  ← baseline" if base else ""))
axa.axhline(0, color="#888", lw=0.9)
axa.axvline(0, color="#888", lw=0.8)
axa.set_xlim(cfg.STEER_RANGE_PATCH)
axa.set_xlabel("steer angle at the road wheel   δ  [deg]")
axa.set_ylabel("scrub radius  [mm]   (+ = patch outboard of axis)")
axa.set_title(f"4a  scrub radius vs steer, by LATERAL ball-joint offset dx\n"
              f"KPI angle held at {kpi:.2f}° throughout — only the axis position moves")
axa.grid(alpha=0.3)
axa.legend(fontsize=8, loc="best", framealpha=0.9)

# ---------------- right: trail vs steer, swept by longitudinal offset dy ----
cmap2 = plt.cm.plasma
dys = [float(v) for v in cfg.LONG_OFFSET_SWEEP]
lo, hi = min(dys), max(dys)
for dy in dys:
    L2, U2 = offset_joints(LBJ, UBJ, dy=dy)
    st = corner_state(L2, U2, WC, steer, CAMBER)
    base = abs(dy) < 1e-9
    axb.plot(steer, st["trail"], lw=3.0 if base else 1.6,
             color="crimson" if base else cmap2((dy - lo) / (hi - lo)),
             zorder=6 if base else 2,
             label=f"dy {dy:+.0f} mm  (trail₀ {mechanical_trail(L2,U2,WC, CAMBER):+.0f})"
                   + ("  ← baseline" if base else ""))
axb.axhline(0, color="#c0392b", lw=1.4, ls="--")
axb.annotate("trail = 0  ·  below this the steering DE-centres",
             (cfg.STEER_RANGE_PATCH[0] + 1, 1.5), fontsize=8.5, color="#c0392b")
axb.axvline(0, color="#888", lw=0.8)
axb.set_xlim(cfg.STEER_RANGE_PATCH)
axb.set_xlabel("steer angle at the road wheel   δ  [deg]")
axb.set_ylabel("mechanical trail  [mm]   (+ = axis ahead of patch = stabilising)")
axb.set_title(f"4b  mechanical trail vs steer, by LONGITUDINAL offset dy\n"
              f"caster angle held at {cas:.2f}° throughout — trail is bought by offset")
axb.grid(alpha=0.3)
axb.legend(fontsize=8, loc="best", framealpha=0.9)

fig.tight_layout()
out = os.path.join(cfg.PROJ, "scrub_trail_by_offset.png")
fig.savefig(_os.path.join(_PLOTS, "task4_scrub_trail_sweeps.png"), dpi=140, bbox_inches="tight")
print("saved", out)

# where does the baseline trail cross zero?
st0 = corner_state(LBJ, UBJ, WC, steer, CAMBER)
neg = np.where(st0["trail"] < 0)[0]
if len(neg):
    print(f"  baseline trail goes NEGATIVE beyond steer = {steer[neg[0]]:+.1f} deg"
          f"  -> steering self-centring reverses")
print(f"  to reach zero scrub you would need dx = {scrub_radius(LBJ,UBJ,WC, CAMBER):.1f} mm"
      f"  (ball joints would sit at X = {LBJ[0]+scrub_radius(LBJ,UBJ,WC, CAMBER):.0f},"
      f" vs wheel centre X = {WC[0]:.0f}) -> not packageable; scrub must stay positive.")
