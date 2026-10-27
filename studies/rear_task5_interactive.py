# -*- coding: utf-8 -*-
"""
REAR STUDY 5 (INTERACTIVE)  -  camber recovery map, hover for values
=====================================================================

    python rear_task5_interactive.py

Same content as rear_task5_recovery_map.py, but every panel is live: move the
cursor and the readout gives the numbers under it. Hovering the maps also drops
a marker on the OTHER map at the same upright position, so you can see the
recovery / roll-centre trade-off at a glance -- that coupling is the whole point.

  A  camber recovery % over the UBJ plane   (inboard pickups frozen)
  B  roll-centre height over the SAME plane
  C  road camber vs body roll for the point you are hovering, rebuilt live

RECOVERY = -(dgamma/dz) x (t/2) x (pi/180) x 100 %   ~=   half-track / FVSA
"""

import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_ROOT, "software"), _ROOT):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_PLOTS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "plots")
_os.makedirs(_PLOTS, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt

import rear_geometry as rg
from wishbone_solver import solve_at

CAM = rg.REAR_STATIC_CAMBER
P0 = rg.load_rear()
G0 = rg.static_geometry(P0, CAM)
HALF = G0["track"] / 2.0
ROLL_REAL = (0.5, 1.5)
# The reference line is derived, not borrowed. Camber recovery r means the
# outside tyre's camber relative to the ROAD changes by phi*(1 - r) for a body
# roll of phi. Asking the tyre to stay within 1 deg of its static camber at the
# 2 deg of roll a stiff GT car reaches gives
#
#       phi * (1 - r) <= 1 deg   at phi = 2 deg   ->   r >= 50 %
#
# so 50 % is the line worth drawing. It is a criterion you can argue with, not
# a number copied off somebody's data sheet.
TARGET_RECOVERY = 50.0     # %

N = 41
dX = np.linspace(-60.0, 60.0, N)
dZ = np.linspace(-60.0, 60.0, N)
REC = np.full((N, N), np.nan)
RCH = np.full((N, N), np.nan)
FVS = np.full((N, N), np.nan)
BST = np.full((N, N), np.nan)
for j, dz in enumerate(dZ):
    for i, dx in enumerate(dX):
        p = dict(P0)
        p[rg.UBJ] = P0[rg.UBJ] + np.array([dx, 0.0, dz])
        try:
            cg, bs = rg.camber_gain_fast(p, CAM)
            g = rg.static_geometry(p, CAM)
            REC[j, i] = rg.recovery_from_gain(cg, HALF)
            RCH[j, i] = g["rc_height"]
            FVS[j, i] = g["fvsa"]
            BST[j, i] = bs
        except Exception:                                    # noqa: BLE001
            pass
print(f"map: {np.isfinite(REC).sum()} / {N*N} upright positions solve")

roll = np.linspace(0.0, 5.0, 121)
z_roll = HALF * np.tan(np.radians(roll))

_cache = {}


def road_curve(dx, dz, n_theta=161, span=30.0):
    """Road camber vs roll for one upright position -- the version the hover uses.

    SPEED. The obvious implementation calls bump_sweep, which runs 1121 solves and
    then re-indexes onto a uniform bump grid: 1.15 s, far too slow to sit behind a
    mouse-move. This marches the wishbone angle directly over 161 steps with
    continuation and interpolates once -- 0.07 s, a 16x speed-up, and it agrees
    with the full sweep to 1e-4 deg. Results are cached per grid cell, so
    re-hovering a cell you have already visited is free.
    """
    key = (round(dx, 3), round(dz, 3))
    if key in _cache:
        return _cache[key]
    p = dict(P0)
    p[rg.UBJ] = P0[rg.UBJ] + np.array([dx, 0.0, dz])
    g = rg.rear_corner(p, CAM, 0.0)
    th = np.linspace(-span, span, n_theta)
    tr, cm, thU, thT = [], [], 0.0, 0.0
    for a in th:
        s = solve_at(g, float(a), 0.0, thU, thT)
        thU, thT = s["thU"], s["thT"]
        tr.append(s["WC"][2] - g["WC"][2])
        cm.append(np.degrees(np.arcsin(np.clip(-s["s"][2], -1.0, 1.0))))
    tr, cm = np.asarray(tr), np.asarray(cm)
    o = np.argsort(tr)
    out = np.interp(z_roll, tr[o], cm[o]) + roll
    _cache[key] = out
    return out


fig = plt.figure(figsize=(17.2, 7.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.02], wspace=0.30)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
X, Y = np.meshgrid(dX, dZ)

cf = axA.contourf(X, Y, REC, levels=22, cmap="viridis")
axA.contour(X, Y, REC, levels=11, colors="k", linewidths=0.5, alpha=0.4)
fig.colorbar(cf, ax=axA, label="camber recovery [%]")
axA.contour(X, Y, REC, levels=[TARGET_RECOVERY], colors="deepskyblue",
            linewidths=2.6)
axA.plot(0, 0, "*", ms=19, mfc="w", mec="k", mew=1.6, zorder=9)
axA.set_title("A · CAMBER RECOVERY over the upright plane\n"
              f"blue = {TARGET_RECOVERY:.0f} % (1° of road camber at 2° roll)"
              " · ★ stock", fontsize=9.5)

cf2 = axB.contourf(X, Y, RCH, levels=22, cmap="coolwarm")
axB.contour(X, Y, RCH, levels=11, colors="k", linewidths=0.5, alpha=0.4)
fig.colorbar(cf2, ax=axB, label="roll-centre height [mm]")
axB.contour(X, Y, REC, levels=[TARGET_RECOVERY], colors="deepskyblue",
            linewidths=2.6)
axB.contour(X, Y, RCH, levels=[0.0], colors="k", linewidths=1.6, linestyles="--")
axB.plot(0, 0, "*", ms=19, mfc="w", mec="k", mew=1.6, zorder=9)
axB.set_title("B · ROLL-CENTRE HEIGHT, same plane\n"
              f"blue = the {TARGET_RECOVERY:.0f}% recovery line from A · "
              "dash = RC at ground", fontsize=9.5)

for ax in (axA, axB):
    ax.set_xlabel("UBJ move  ΔX  [mm]  (+ = outboard)", fontsize=9)
    ax.set_ylabel("UBJ move  ΔZ  [mm]  (+ = up)", fontsize=9)
    ax.tick_params(labelsize=8)
mA, = axA.plot([], [], "o", ms=11, mfc="none", mec="k", mew=2.0, zorder=12)
mB, = axB.plot([], [], "o", ms=11, mfc="none", mec="k", mew=2.0, zorder=12)

axC.axvspan(*ROLL_REAL, color="#2a8f4a", alpha=0.13)
base = road_curve(0.0, 0.0)
axC.plot(roll, base, color="#c0392b", lw=2.3, label="stock")
live, = axC.plot(roll, base, color="#1f77b4", lw=2.3, ls="--", label="hovered")
axC.plot(roll, CAM + roll, color="#999", lw=1.3, ls=":", label="0 % recovery")
axC.axhline(0, color="#444", lw=1.0)
axC.set_xlabel("body roll  φ  [deg]", fontsize=9)
axC.set_ylabel("outside-wheel camber to the ROAD  [deg]", fontsize=9)
axC.set_title("C · road camber vs roll — rebuilt live from the hovered point\n"
              "flatter = tyre stays planted", fontsize=9.5)
axC.grid(alpha=0.3)
axC.legend(fontsize=8, loc="upper left")
axC.tick_params(labelsize=8)

read = fig.text(0.012, 0.017, "", fontsize=8.8, family="monospace", va="bottom",
                bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))
_last = {"ij": None}

# ---------------------------------------------------------------- BLITTING
# The two maps are 41x41 contourf plots with 22 levels each. Redrawing the whole
# figure on every mouse-move costs ~120 ms even when nothing but a marker moved,
# which is what made this window feel laggy. So the static artwork is rendered
# ONCE into a cached background and only the four moving artists are redrawn on
# top of it -- the standard matplotlib blitting pattern. Hover cost drops to a
# few ms and the maps never re-render.
_ANIM = (mA, mB, live, read)
_bg = {"buf": None}


def _capture(_evt=None):
    """Cache the static background (after a full draw, and after any resize)."""
    for a in _ANIM:
        a.set_animated(True)
    fig.canvas.draw()
    _bg["buf"] = fig.canvas.copy_from_bbox(fig.bbox)


def _blit():
    if _bg["buf"] is None:
        _capture()
    fig.canvas.restore_region(_bg["buf"])
    for a in _ANIM:
        a.axes.draw_artist(a) if getattr(a, "axes", None) else fig.draw_artist(a)
    fig.canvas.blit(fig.bbox)


fig.canvas.mpl_connect("resize_event", lambda e: _capture())


def on_move(ev):
    if ev.inaxes in (axA, axB) and ev.xdata is not None:
        i = int(np.argmin(np.abs(dX - ev.xdata)))
        j = int(np.argmin(np.abs(dZ - ev.ydata)))
        if not np.isfinite(REC[j, i]):
            return
        mA.set_data([dX[i]], [dZ[j]])
        mB.set_data([dX[i]], [dZ[j]])
        if _last["ij"] != (i, j):        # only re-solve when the cell changes
            _last["ij"] = (i, j)
            try:
                live.set_ydata(road_curve(dX[i], dZ[j]))
            except Exception:                                # noqa: BLE001
                pass
        read.set_text(
            f"UBJ  ΔX {dX[i]:+6.1f}  ΔZ {dZ[j]:+6.1f} mm      "
            f"(stock: recovery {REC[N//2, N//2]:.1f} %, RC {RCH[N//2, N//2]:+.0f} mm, "
            f"FVSA {FVS[N//2, N//2]:.0f} mm)\n"
            f"  camber recovery {REC[j, i]:7.1f} %      "
            f"(target {TARGET_RECOVERY:.0f} %)\n"
            f"  roll centre     {RCH[j, i]:+7.1f} mm\n"
            f"  FVSA            {FVS[j, i]:7.0f} mm      "
            f"bump steer {BST[j, i]:+.5f} °/mm")
    elif ev.inaxes is axC and ev.xdata is not None:
        k = int(np.argmin(np.abs(roll - ev.xdata)))
        read.set_text(f"body roll {roll[k]:+5.2f} deg   "
                      f"(loaded wheel {z_roll[k]:+.1f} mm)\n"
                      f"  stock road camber   {base[k]:+7.3f} deg\n"
                      f"  hovered geometry    {live.get_ydata()[k]:+7.3f} deg\n"
                      f"  0 % recovery would be {CAM + roll[k]:+7.3f} deg")
    else:
        return
    _blit()


fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.suptitle("Rear — Study 5 (interactive): WHERE CAMBER RECOVERY COMES FROM     "
             "RECOVERY = −(dγ/dz)·(t/2)·(π/180) ≈ half-track / FVSA     "
             f"stock {REC[N//2, N//2]:.1f} % vs a "
             f"{TARGET_RECOVERY:.0f} % target",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.10, 1, 0.90])
_capture()
print("hover the maps; panel C rebuilds live. Close the window to exit.")
plt.savefig(_os.path.join(_PLOTS, "rear_task5_interactive.png"), dpi=130,
            bbox_inches="tight")
if _os.environ.get("STUDY_NOSHOW") != "1":
    plt.show()
