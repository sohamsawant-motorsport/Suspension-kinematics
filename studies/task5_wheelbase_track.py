# -*- coding: utf-8 -*-
"""
TASK 5  -  Wheelbase and half-track vs steer, swept by SCRUB and by TRAIL.
           (static PNG)

ONE CORNER only (the CAD hard points are the front-RIGHT corner; the left is its
exact mirror). Ackermann and the left/right split are future work, so these are
this corner's contributions:

    wheelbase (this corner) = front patch Y - rear patch Y   (the rear does not steer)
    half-track (this corner) = front patch X                 (lateral position from
                                                              the car centreline)

The two design levers are the ball-joint offsets, which leave the KPI/caster
ANGLES untouched:
    dx (lateral)      -> changes SCRUB only
    dy (longitudinal) -> changes TRAIL only

WHAT TO EXPECT (derived, and confirmed by the sensitivity print at the bottom)
    The wheel is CAMBERED, so the contact patch is NOT the wheel centre projected
    down -- the tilted wheel touches down R*sin(camber) to one side. Steering
    rotates the whole wheel (centre AND spin axis) about the kingpin axis, so the
    camber offset rotates with it and the patch must be recomputed at every angle.

    With that modelled properly, the gradients come out as:
        d(wheelbase)/d(delta)  ~  scrub*u_z - R*u_x     (small residual, ~0.07/rad)
        d(half-track)/d(delta) =  trail*u_z             (EXACT)

    Note the half-track one: with camber handled, the R*sin(caster) term that an
    uncambered model predicts CANCELS -- the camber offset's own rotation kills it.
    So TRAIL alone sets the track-change rate, and SCRUB dominates the wheelbase
    one. Both scrub and trail here are the CAMBER-CORRECTED values.

Output: ../wheelbase_track_by_offset.png   (project root)
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
REAR_Y = cfg.rear_patch_y()

steer = np.linspace(cfg.STEER_RANGE_PATCH[0], cfg.STEER_RANGE_PATCH[1], 281)
dxs = [float(v) for v in cfg.LATERAL_OFFSET_SWEEP]
dys = [float(v) for v in cfg.LONG_OFFSET_SWEEP]

fig, axs = plt.subplots(2, 2, figsize=(15.5, 10.5))
cmap, cmap2 = plt.cm.viridis, plt.cm.plasma
lo_y, hi_y = min(dys), max(dys)


def curves(dx=0.0, dy=0.0):
    L2, U2 = offset_joints(LBJ, UBJ, dx=dx, dy=dy)
    st = corner_state(L2, U2, WC, steer, CAMBER)
    return (st["P"][:, 1] - REAR_Y,            # wheelbase for this corner
            st["P"][:, 0],                     # half-track for this corner
            scrub_radius(L2, U2, WC, CAMBER), mechanical_trail(L2, U2, WC, CAMBER))


# ---- (a) wheelbase vs steer, swept by SCRUB (trail constant) ---------------
ax = axs[0, 0]
for dx in dxs:
    wb, _, sc, tr = curves(dx=dx)
    base = abs(dx) < 1e-9
    ax.plot(steer, wb, lw=3.0 if base else 1.6,
            color="crimson" if base else cmap(dx / max(dxs)),
            zorder=6 if base else 2,
            label=f"scrub {sc:+.0f} mm  (dx {dx:+.0f})" + ("  ← baseline" if base else ""))
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlim(cfg.STEER_RANGE_PATCH)
ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
ax.set_ylabel("wheelbase, this corner  [mm]")
ax.set_title(f"5a  WHEELBASE vs steer — swept by SCRUB\n"
             f"trail held {mechanical_trail(LBJ,UBJ,WC, CAMBER):.1f} mm · KPI angle held {kpi:.2f}°",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

# ---- (b) wheelbase vs steer, swept by TRAIL (scrub constant) ---------------
ax = axs[0, 1]
for dy in dys:
    wb, _, sc, tr = curves(dy=dy)
    base = abs(dy) < 1e-9
    ax.plot(steer, wb, lw=3.0 if base else 1.6,
            color="crimson" if base else cmap2((dy - lo_y) / (hi_y - lo_y)),
            zorder=6 if base else 2,
            label=f"trail {tr:+.0f} mm  (dy {dy:+.0f})" + ("  ← baseline" if base else ""))
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlim(cfg.STEER_RANGE_PATCH)
ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
ax.set_ylabel("wheelbase, this corner  [mm]")
ax.set_title(f"5b  WHEELBASE vs steer — swept by TRAIL\n"
             f"scrub held {scrub_radius(LBJ,UBJ,WC, CAMBER):.1f} mm · caster angle held {cas:.2f}°",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

# ---- (c) half-track vs steer, swept by SCRUB ------------------------------
ax = axs[1, 0]
for dx in dxs:
    _, ht, sc, tr = curves(dx=dx)
    base = abs(dx) < 1e-9
    ax.plot(steer, ht, lw=3.0 if base else 1.6,
            color="crimson" if base else cmap(dx / max(dxs)),
            zorder=6 if base else 2,
            label=f"scrub {sc:+.0f} mm" + ("  ← baseline" if base else ""))
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlim(cfg.STEER_RANGE_PATCH)
ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
ax.set_ylabel("half-track, this corner  [mm]")
ax.set_title("5c  HALF-TRACK vs steer — swept by SCRUB\n"
             "scrub only curves it; the slope at 0° is set by TRAIL", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

# ---- (d) half-track vs steer, swept by TRAIL ------------------------------
ax = axs[1, 1]
for dy in dys:
    _, ht, sc, tr = curves(dy=dy)
    base = abs(dy) < 1e-9
    ax.plot(steer, ht, lw=3.0 if base else 1.6,
            color="crimson" if base else cmap2((dy - lo_y) / (hi_y - lo_y)),
            zorder=6 if base else 2,
            label=f"trail {tr:+.0f} mm" + ("  ← baseline" if base else ""))
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlim(cfg.STEER_RANGE_PATCH)
ax.set_xlabel("steer angle at the road wheel   δ  [deg]")
ax.set_ylabel("half-track, this corner  [mm]")
ax.set_title("5d  HALF-TRACK vs steer — swept by TRAIL\n"
             "trail tilts the curve and breaks its symmetry", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

fig.suptitle("Task 5 — wheelbase & half-track vs steer, one corner (front-right CAD data; "
             "left is the mirror)\nscrub and trail tuned by sliding BOTH ball joints — "
             "KPI and caster angles unchanged throughout", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = os.path.join(cfg.PROJ, "wheelbase_track_by_offset.png")
fig.savefig(_os.path.join(_PLOTS, "task5_wheelbase_track.png"), dpi=135, bbox_inches="tight")
print("saved", out)

# --- confirm the closed form against the exact rotation ---------------------
# closed form:  dP/d(delta) = u x (WC - G),  with v = (scrub, -trail, R)
from scrub_trail import steering_axis, ground_point

print()
print("sensitivity at delta = 0 (per degree of steer) -- exact vs closed form:")
h = 0.5
R = WC[2]
for dx in (0.0, 60.0, 120.0):
    L2, U2 = offset_joints(LBJ, UBJ, dx=dx)
    s = corner_state(L2, U2, WC, np.array([-h, h]), CAMBER)
    dwb = (s["P"][1, 1] - s["P"][0, 1]) / (2 * h)
    _, u = steering_axis(L2, U2)
    sc = scrub_radius(L2, U2, WC, CAMBER)
    pred = (sc * u[2] - R * u[0]) * np.pi / 180.0
    print(f"  scrub {sc:+6.1f} mm -> d(wheelbase)/d(delta) = {dwb:+7.4f} mm/deg"
          f"   [scrub*u_z - R*u_x = {pred:+7.4f}]")
for dy in (-20.0, 0.0, 60.0):
    L2, U2 = offset_joints(LBJ, UBJ, dy=dy)
    s = corner_state(L2, U2, WC, np.array([-h, h]), CAMBER)
    dht = (s["P"][1, 0] - s["P"][0, 0]) / (2 * h)
    _, u = steering_axis(L2, U2)
    tr = mechanical_trail(L2, U2, WC, CAMBER)
    pred = (tr * u[2]) * np.pi / 180.0
    print(f"  trail {tr:+6.1f} mm -> d(half-track)/d(delta) = {dht:+7.4f} mm/deg"
          f"   [trail*u_z = {pred:+7.4f}]")
