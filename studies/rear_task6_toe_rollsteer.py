# -*- coding: utf-8 -*-
"""
REAR STUDY 6  -  TOE LINK & ROLL STEER   (interactive)
=======================================================

    python rear_task6_toe_rollsteer.py

Roll steer is the rear's stability parameter: as the car rolls, the loaded
(outside) wheel toes IN or OUT all by itself. Toe-in = the rear steers into the
corner = understeer = forgiving. Toe-out = the rear steers out = oversteer = the
car can snap. This study finds what sets it.

TWO DIFFERENT KNOBS -- the point of the study
---------------------------------------------
  STATIC TOE  is a SETUP number (the threaded sleeve). It slides the whole toe
              curve up or down. It does NOT change the slope: sweeping it 0 -> 1
              deg moves bump steer only +0.00546 -> +0.00547 deg/mm.
  TOE-LINK    POSITION is the hard-point geometry. It sets the SLOPE of that
              curve -- i.e. the bump steer and the roll steer. This is the lever.

  left    map of ROLL STEER over the toe-link OUTER ball joint's plane, inboard
          toe point frozen. Stock marked; the derived target contour drawn.
  middle  toe vs wheel travel for static toe 0.0 -> 1.0 deg (0.2 steps): the
          curves are parallel -- offset, not slope.
  right   toe vs BODY ROLL for the loaded wheel, stock vs candidate toe links,
          with the roll a GT car really sees shaded.

Hover any panel for a live readout.

WHAT TO AIM AT: a small TOE-IN with roll on the loaded wheel is the
stabilising direction, and the useful question is how much. Hold the toe change
under 0.1 deg over the ~1.5 deg of body roll a stiff GT car reaches and you get
0.067 deg of toe per deg of roll -- the blue contour. Zero roll steer, the black
contour, is the other defensible answer. Anything on the red side toes the
loaded wheel OUT in roll, which is the direction that bites.
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
from wishbone_solver import solve_sweep

CAM = rg.REAR_STATIC_CAMBER
P0 = rg.load_rear()
HALF = rg.static_geometry(P0, CAM)["track"] / 2.0
ROLL_REAL = (0.5, 1.5)                 # Seward: 0.6 deg/g x 1.67 g
# Derived reference, not borrowed. A GT car on stiff springs reaches about
# 1.5 deg of body roll. Holding the loaded wheel's toe change under 0.1 deg over
# that roll, in the stabilising (toe-IN) direction, gives
#
#       0.1 deg / 1.5 deg  =  0.067 deg of toe per deg of roll
#
# which in bump-steer terms is roll_steer / (half-track * pi/180).
TARGET_ROLL_STEER = 0.1 / 1.5          # deg toe per deg of body roll

TRAVEL = np.linspace(-80.0, 80.0, 241)   # wide enough for 5 deg of roll


def toe_curve(pts, static_toe=0.0, travel=TRAVEL):
    g = rg.rear_corner(pts, CAM, static_toe)
    sw = solve_sweep(g, travel, 0.0, 28.0, 1121)
    return sw["toe"], sw["camber"]


def rates(pts):
    """(bump steer, camber gain) in deg/mm -- the fast central difference."""
    return rg.camber_gain_fast(pts, CAM)[1], rg.camber_gain_fast(pts, CAM)[0]


# ------------------------------------------------- A: toe-link position map
N = 33
dX = np.linspace(-50.0, 50.0, N)
dZ = np.linspace(-50.0, 50.0, N)
BS = np.full((N, N), np.nan)
for j, dz in enumerate(dZ):
    for i, dx in enumerate(dX):
        p = dict(P0)
        p[rg.TOE_O] = P0[rg.TOE_O] + np.array([dx, 0.0, dz])
        try:
            BS[j, i] = rg.camber_gain_fast(p, CAM)[1]
        except Exception:                                   # noqa: BLE001
            pass
# roll steer [deg toe per deg roll] = bump steer [deg/mm] * (t/2) * pi/180
RS = BS * HALF * np.pi / 180.0
print(f"toe-link map: {np.isfinite(BS).sum()} / {N*N} positions solve")

# ------------------------------------------------- B: static toe family
STATIC_TOES = np.arange(0.0, 1.01, 0.2)
fam = [(t, toe_curve(P0, float(t))[0]) for t in STATIC_TOES]

# ------------------------------------------------- C: candidate toe links
CANDS = [("stock", np.array([0.0, 0.0, 0.0]), "#c0392b", "-"),
         ("toe-outer +25 mm up", np.array([0.0, 0.0, 25.0]), "#2a8f4a", "--"),
         ("toe-outer -25 mm down", np.array([0.0, 0.0, -25.0]), "#7a3fbf", "-."),
         ("toe-outer +25 mm out", np.array([25.0, 0.0, 0.0]), "#d2822a", ":")]
roll = np.linspace(0.0, 5.0, 121)
z_roll = HALF * np.tan(np.radians(roll))
cands = []
for name, d, col, ls in CANDS:
    p = dict(P0)
    p[rg.TOE_O] = P0[rg.TOE_O] + d
    try:
        tc, _ = toe_curve(p, 0.0)
        cands.append((name, np.interp(z_roll, TRAVEL, tc), col, ls,
                      rg.camber_gain_fast(p, CAM)[1], d))
    except Exception as exc:                                # noqa: BLE001
        print("  candidate failed:", name, exc)

# =====================================================================
fig = plt.figure(figsize=(17.4, 7.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.30)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
X, Y = np.meshgrid(dX, dZ)

cf = axA.contourf(X, Y, RS, levels=22, cmap="RdYlGn")
cs = axA.contour(X, Y, RS, levels=11, colors="k", linewidths=0.5, alpha=0.45)
axA.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
fig.colorbar(cf, ax=axA, label="roll steer  [deg toe per deg roll]")
axA.contour(X, Y, RS, levels=[0.0], colors="k", linewidths=2.2)
axA.contour(X, Y, RS, levels=[TARGET_ROLL_STEER], colors="deepskyblue",
            linewidths=2.4)
axA.plot(0, 0, "*", ms=19, mfc="white", mec="k", mew=1.6, zorder=8, label="stock")
for name, _c, col, _ls, bs, d in cands[1:]:
    axA.plot(d[0], d[2], "P", ms=11, mfc=col, mec="k", mew=1.2, zorder=8,
             label=f"{name.split()[1]}{name.split()[2]}")
axA.set_title("A · ROLL STEER over the TOE-LINK outer plane\n"
              f"black = zero roll steer · blue = {TARGET_ROLL_STEER:.3f} ° toe / ° roll\n"
              "green side = toe-IN in roll (stabilising)", fontsize=9.5)
axA.set_xlabel("toe-link outer  ΔX  [mm]  (+ = outboard)", fontsize=9)
axA.set_ylabel("toe-link outer  ΔZ  [mm]  (+ = up)", fontsize=9)
axA.legend(fontsize=7.4, loc="lower left", framealpha=0.92)
axA.tick_params(labelsize=8)

cmap = plt.cm.viridis
for k, (t, c) in enumerate(fam):
    axB.plot(TRAVEL, c, lw=2.1, color=cmap(k / max(len(fam) - 1, 1)),
             label=f"static toe {t:+.1f}°")
axB.axvline(0, color="#888", lw=0.9)
axB.axhline(0, color="#888", lw=0.9)
axB.set_title("B · STATIC TOE only OFFSETS the curve\n"
              "0 → 1° in 0.2° steps · the SLOPE never changes\n"
              "(bump steer +0.00546 → +0.00547 deg/mm)", fontsize=9.5)
axB.set_xlabel("wheel travel  [mm]   (+ = compression)", fontsize=9)
axB.set_ylabel("toe  [deg]   (+ = toe-in)", fontsize=9)
axB.grid(alpha=0.3)
axB.legend(fontsize=7.6, loc="upper left")
axB.tick_params(labelsize=8)

axC.axvspan(*ROLL_REAL, color="#2a8f4a", alpha=0.13, zorder=0)
axC.annotate("roll a GT car\nactually sees", (np.mean(ROLL_REAL), 0.02),
             ha="center", fontsize=7.6, color="#2a6f3a")
for name, c, col, ls, bs, _d in cands:
    axC.plot(roll, c, color=col, ls=ls, lw=2.3,
             label=f"{name}   bump steer {bs:+.5f}°/mm")
axC.axhline(0, color="#444", lw=1.1)
axC.set_title("C · toe of the LOADED wheel vs body roll\n"
              "rising = toe-IN with roll = stabilising", fontsize=9.5)
axC.set_xlabel("body roll  φ  [deg]", fontsize=9)
axC.set_ylabel("toe  [deg]   (+ = toe-in)", fontsize=9)
axC.grid(alpha=0.3)
axC.legend(fontsize=7.4, loc="lower left")
axC.tick_params(labelsize=8)

# ----------------------------------------------------------- interactivity
read = fig.text(0.012, 0.017, "", fontsize=8.6, family="monospace", va="bottom",
                bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))
mk = {}
mk["A"], = axA.plot([], [], "o", ms=10, mfc="none", mec="k", mew=1.8, zorder=11)
mk["B"], = axB.plot([], [], "o", ms=8, mfc="none", mec="k", mew=1.6, zorder=11)
mk["C"], = axC.plot([], [], "o", ms=8, mfc="none", mec="k", mew=1.6, zorder=11)
vB = axB.axvline(np.nan, color="0.35", lw=0.7)
vC = axC.axvline(np.nan, color="0.35", lw=0.7)


def on_move(ev):
    if ev.inaxes is axA and ev.xdata is not None:
        i = int(np.argmin(np.abs(dX - ev.xdata)))
        j = int(np.argmin(np.abs(dZ - ev.ydata)))
        mk["A"].set_data([dX[i]], [dZ[j]])
        v = RS[j, i]
        read.set_text(f"TOE-LINK OUTER   ΔX {dX[i]:+6.1f}   ΔZ {dZ[j]:+6.1f} mm\n"
                      f"  roll steer  {v:+7.4f} deg toe / deg roll"
                      f"   ({'TOE-IN, stabilising' if v > 0 else 'TOE-OUT, can oversteer'})\n"
                      f"  bump steer  {BS[j, i]:+8.5f} deg/mm"
                      f"      (stock {BS[N//2, N//2]:+.5f})")
    elif ev.inaxes is axB and ev.xdata is not None:
        k = int(np.argmin(np.abs(TRAVEL - ev.xdata)))
        vB.set_xdata([TRAVEL[k], TRAVEL[k]])
        mk["B"].set_data([TRAVEL[k]], [fam[0][1][k]])
        txt = [f"travel {TRAVEL[k]:+6.1f} mm      toe by static setting:"]
        txt += ["  " + "  ".join(f"{t:+.1f}°→{c[k]:+.3f}°" for t, c in fam)]
        read.set_text("\n".join(txt))
    elif ev.inaxes is axC and ev.xdata is not None:
        k = int(np.argmin(np.abs(roll - ev.xdata)))
        vC.set_xdata([roll[k], roll[k]])
        mk["C"].set_data([roll[k]], [cands[0][1][k]])
        txt = [f"body roll {roll[k]:+5.2f} deg   "
               f"(loaded wheel at {z_roll[k]:+.1f} mm)"]
        for name, c, _col, _ls, _bs, _d in cands:
            txt.append(f"  {name:<22s} toe {c[k]:+7.4f} deg")
        read.set_text("\n".join(txt))
    else:
        return
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.suptitle("Rear — Study 6: TOE LINK & ROLL STEER      "
             "static toe = OFFSET (setup)   ·   toe-link POSITION = SLOPE "
             "(roll steer)      "
             f"stock roll steer {BS[N//2, N//2]*HALF*np.pi/180:+.4f} °/°",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.10, 1, 0.90])

print("\nSTATIC TOE (setup) -- offset only")
for t, c in fam:
    print(f"  {t:+.1f} deg -> toe at ride {c[len(TRAVEL)//2]:+.4f},  "
          f"at +30 mm {np.interp(30, TRAVEL, c):+.4f} deg")
print("\nTOE-LINK POSITION (geometry) -- sets the slope")
for name, c, _col, _ls, bs, _d in cands:
    print(f"  {name:<24s} bump steer {bs:+9.5f} deg/mm   "
          f"roll steer {bs*HALF*np.pi/180:+7.4f} deg/deg")
print(f"\n  target: toe change under 0.1 deg at 1.5 deg of body roll"
      f"  ->  roll steer {TARGET_ROLL_STEER:+.4f} deg/deg"
      f"  ->  bump steer {TARGET_ROLL_STEER / (HALF * np.pi / 180):+.5f} deg/mm")
plt.savefig(_os.path.join(_PLOTS, "rear_task6_toe_rollsteer.png"), dpi=130,
            bbox_inches="tight")
if _os.environ.get("STUDY_NOSHOW") != "1":
    plt.show()
