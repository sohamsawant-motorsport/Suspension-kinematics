# -*- coding: utf-8 -*-
"""
TASK 1.2  -  Camber change vs steer angle, one curve per caster.  (static PNG)

Sweep the road-wheel steer angle from -45 to +45 deg and plot the camber change
for caster = 0,1,...,10 deg (plus the stock caster), with KPI HELD CONSTANT.

Output: ../camber_vs_steer_by_caster.png   (project root)
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
from steer_camber import camber_change, camber_change_approx

steer = np.linspace(cfg.STEER_RANGE[0], cfg.STEER_RANGE[1], 361)
kpi_held = cfg.KPI_HELD_FOR_CASTER_SWEEP

casters = [float(c) for c in cfg.CASTER_SWEEP]
if not any(abs(c - cfg.STOCK_CASTER) < 1e-6 for c in casters):
    casters.append(cfg.STOCK_CASTER)
casters = sorted(casters)

fig, ax = plt.subplots(figsize=(10.5, 6.8))
cmap = plt.cm.plasma
for c in casters:
    stock = abs(c - cfg.STOCK_CASTER) < 1e-6
    ax.plot(steer, camber_change(kpi_held, c, steer),
            lw=3.0 if stock else 1.5,
            color="crimson" if stock else cmap(c / 10.0),
            zorder=6 if stock else 2,
            label=f"caster {c:g}°" + ("  (stock)" if stock else ""))

ax.plot(steer, camber_change_approx(kpi_held, cfg.STOCK_CASTER, steer),
        ls="--", lw=1.2, color="black", alpha=0.7,
        label="small-angle approx (stock)")

ax.axhline(0, color="#888", lw=0.8)
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
ax.set_ylabel("camber change   Δγ  [deg]      (− = leans into corner, favourable)")
ax.set_title("1.2  Camber change vs steer, by caster\n"
             f"KPI held constant = {kpi_held:g}°   ·   exact kingpin-axis rotation")
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=8, loc="upper center", framealpha=0.9)
ax.set_xlim(-45, 45)
fig.tight_layout()

out = os.path.join(cfg.PROJ, "camber_vs_steer_by_caster.png")
fig.savefig(_os.path.join(_PLOTS, "task1_2_camber_vs_caster.png"), dpi=140, bbox_inches="tight")
print("saved", out)
