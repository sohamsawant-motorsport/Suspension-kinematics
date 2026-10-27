# -*- coding: utf-8 -*-
"""
REAR STUDY 5  -  WHAT MAKES CAMBER RECOVERY?  (upright map + roll behaviour)
============================================================================

    python rear_task5_recovery_map.py

THE QUESTION: what variable, if changed, gives camber recovery?

THE MATHS (derived and validated in rear_geometry.py):

    gamma_road = gamma_chassis + phi
    recovery   = 1 - d(gamma_road)/d(phi) = -d(gamma_chassis)/d(phi)
    body roll reaches the wheel only as travel:  z = (t/2) tan(phi)

    ==>  RECOVERY = -(d gamma/dz) x (t/2) x (pi/180) x 100 %      [EXACT]
    and  d gamma/dz ~= -1 / FVSA     ==>     RECOVERY ~= (t/2) / FVSA

So recovery is made of exactly two things: the linkage's CAMBER GAIN, and the
half-track that turns roll into travel. Track is fixed by the rules and the tyre,
so the ONLY real lever is camber gain -- i.e. the front-view SWING-ARM LENGTH.
Short swing arm -> strong recovery.

THE MAP: the upright in isolation. The four INBOARD pickups are FROZEN and the
upper ball joint is swept in the front-view plane (lateral dX, vertical dZ). The
lower ball joint, wheel centre and toe link stay put, so every point on the map
is the same car with a different upright.

  left    CAMBER RECOVERY % over the UBJ plane, stock marked, target band drawn.
  middle  ROLL-CENTRE HEIGHT over the SAME plane -- the point of the study: the
          two contour sets are not independent. Both are read off the same
          instant centre, so moving the UBJ moves BOTH. You cannot dial recovery
          without accepting what it does to the roll centre.
  right   ROAD CAMBER vs body roll out to 7 deg for stock and for candidate
          uprights, with the roll a GT car actually sees shaded.

ON 7 DEGREES: you are right to doubt it. Seward's GT-car worked example gives a
roll gradient of 0.6 deg/g; at 1.67 g that is ~1.0 deg of body roll. A stiff GT3
lives in roughly 0.5-1.5 deg. 7 deg is a road car on soft springs. The map is
drawn to 7 deg so the trends are visible, but the shaded band is where the car
actually operates -- judge designs there.
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
from matplotlib.patches import Rectangle

import rear_geometry as rg

CAM = rg.REAR_STATIC_CAMBER
P0 = rg.load_rear()
G0 = rg.static_geometry(P0, CAM)
HALF = G0["track"] / 2.0

# realistic GT roll band (Seward: roll gradient 0.6 deg/g, ~1.67 g peak)
ROLL_REAL = (0.5, 1.5)
ROLL_MAX = 7.0

# ---------------------------------------------------------------- the map
N = 41
dX = np.linspace(-60.0, 60.0, N)          # UBJ lateral move  (+ = outboard)
dZ = np.linspace(-60.0, 60.0, N)          # UBJ vertical move (+ = up)
REC = np.full((N, N), np.nan)
RCH = np.full((N, N), np.nan)
BST = np.full((N, N), np.nan)

for j, dz in enumerate(dZ):
    for i, dx in enumerate(dX):
        p = dict(P0)
        p[rg.UBJ] = P0[rg.UBJ] + np.array([dx, 0.0, dz])
        try:
            cg, bs = rg.camber_gain_fast(p, CAM)
            REC[j, i] = rg.recovery_from_gain(cg, HALF)
            BST[j, i] = bs
            RCH[j, i] = rg.static_geometry(p, CAM)["rc_height"]
        except Exception:                                   # noqa: BLE001
            pass                                            # leaves NaN

print(f"map: {np.isfinite(REC).sum()} / {N*N} upright positions solve")

# ------------------------------------------------------- roll curves
def road_camber_curve(pts, roll):
    """Outside-wheel camber vs the ROAD:  travel = (t/2)tan(phi), then the true
    non-linear solver for gamma(travel), plus the body tilt.

    Returns (curve, roll_reach) where roll_reach is the roll covered by the
    solver's search range -- NOT a physical bind (see rear_geometry.travel_limits).
    """
    lo, hi = rg.travel_limits(pts, CAM)             # wide search span (45 deg)
    lo, hi = lo * 0.985, hi * 0.985
    sw = rg.bump_sweep(pts, CAM, lo, hi, 401, span=45.0)
    z = HALF * np.tan(np.radians(roll))
    out = np.where((z >= lo) & (z <= hi),
                   np.interp(z, sw["bump"], sw["camber"]) + roll, np.nan)
    return out, float(np.degrees(np.arctan(hi / HALF)))


roll = np.linspace(0.0, ROLL_MAX, 141)
CANDIDATES = [("stock", np.array([0.0, 0.0, 0.0]), "#c0392b", "-"),
              ("UBJ +20 mm up", np.array([0.0, 0.0, 20.0]), "#2a8f4a", "--"),
              ("UBJ +40 mm up", np.array([0.0, 0.0, 40.0]), "#7a3fbf", "-."),
              ("UBJ -20 mm in", np.array([-20.0, 0.0, 0.0]), "#d2822a", ":")]

curves = []
for name, d, col, ls in CANDIDATES:
    p = dict(P0)
    p[rg.UBJ] = P0[rg.UBJ] + d
    try:
        cg, _ = rg.camber_gain_fast(p, CAM)
        c, lim = road_camber_curve(p, roll)
        curves.append((name, c, col, ls, rg.recovery_from_gain(cg, HALF),
                       rg.static_geometry(p, CAM)["rc_height"], d, lim))
    except Exception as exc:                                # noqa: BLE001
        print("  candidate failed:", name, exc)

# =====================================================================
fig = plt.figure(figsize=(17.2, 7.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.06], wspace=0.28)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
X, Y = np.meshgrid(dX, dZ)

# ---- A: recovery map -------------------------------------------------------
cf = axA.contourf(X, Y, REC, levels=22, cmap="viridis")
cs = axA.contour(X, Y, REC, levels=11, colors="k", linewidths=0.5, alpha=0.45)
axA.clabel(cs, inline=True, fontsize=7, fmt="%.0f%%")
fig.colorbar(cf, ax=axA, label="camber recovery  [%]")
# TARGET: keep the outside tyre within 1 deg of its static camber at 2 deg roll
#   road camber change = phi(1 - r)  <=  1 deg  at phi = 2  ->  r >= 50 %
axA.contour(X, Y, REC, levels=[50.0], colors="lime", linewidths=2.6)
axA.plot(0, 0, "*", ms=20, mfc="white", mec="k", mew=1.6, zorder=8, label="stock")
for name, _c, col, _ls, rec, rc, d, _lim in curves[1:]:
    axA.plot(d[0], d[2], "P", ms=11, mfc=col, mec="k", mew=1.2, zorder=8,
             label=f"{name}  ({rec:.0f}%)")
axA.set_title("A · CAMBER RECOVERY over the upright plane\n"
              "inboard pickups FROZEN · only the UBJ moves\n"
              "lime = 50 % (holds the tyre within 1° at 2° roll) — PROVISIONAL",
              fontsize=9.5)

# ---- B: roll-centre map ----------------------------------------------------
cf2 = axB.contourf(X, Y, RCH, levels=22, cmap="coolwarm")
cs2 = axB.contour(X, Y, RCH, levels=11, colors="k", linewidths=0.5, alpha=0.45)
axB.clabel(cs2, inline=True, fontsize=7, fmt="%.0f")
fig.colorbar(cf2, ax=axB, label="roll-centre height  [mm]")
axB.contour(X, Y, REC, levels=[50.0], colors="lime", linewidths=2.6)
axB.contour(X, Y, RCH, levels=[0.0], colors="k", linewidths=1.8, linestyles="--")
axB.plot(0, 0, "*", ms=20, mfc="white", mec="k", mew=1.6, zorder=8)
for name, _c, col, _ls, rec, rc, d, _lim in curves[1:]:
    axB.plot(d[0], d[2], "P", ms=11, mfc=col, mec="k", mew=1.2, zorder=8)
axB.set_title("B · ROLL-CENTRE HEIGHT over the SAME plane\n"
              "lime = the 50 %-recovery line from A · black dash = RC at ground\n"
              "THE COUPLING: chasing recovery drags the RC with it",
              fontsize=9.5)

for ax in (axA, axB):
    ax.axhline(0, color="w", lw=0.8, alpha=0.5)
    ax.axvline(0, color="w", lw=0.8, alpha=0.5)
    ax.set_xlabel("UBJ move — lateral  ΔX  [mm]   (+ = outboard)", fontsize=9)
    ax.set_ylabel("UBJ move — vertical  ΔZ  [mm]   (+ = up)", fontsize=9)
    ax.tick_params(labelsize=8)
axA.legend(fontsize=7.6, loc="lower left", framealpha=0.92)

# ---- C: road camber vs roll ------------------------------------------------
axC.axvspan(*ROLL_REAL, color="#2a8f4a", alpha=0.13, zorder=0)
axC.annotate("roll a GT car\nactually sees\n(~0.6°/g × 1.67 g)",
             (np.mean(ROLL_REAL), 3.1), ha="center", fontsize=7.6, color="#2a6f3a")
for name, c, col, ls, rec, rc, _d, lim in curves:
    axC.plot(roll, c, color=col, ls=ls, lw=2.3,
             label=f"{name}   recovery {rec:.0f}%,  RC {rc:+.0f} mm")
axC.plot(roll, CAM + roll, color="#999", lw=1.4, ls=":",
         label="0 % recovery (leans with the body)")
axC.axhline(0, color="#444", lw=1.1)
axC.annotate("above: tyre leans OUT of the corner", (0.15, 0.22), fontsize=7.6,
             color="#a33")
axC.set_xlim(0, ROLL_MAX)
axC.set_xlabel("body roll  φ  [deg]", fontsize=9)
axC.set_ylabel("outside-wheel camber to the ROAD  [deg]", fontsize=9.5)
axC.set_title("C · what it MEANS: road camber vs roll, out to 7°\n"
              "flatter = the tyre stays planted", fontsize=9.5)
axC.grid(alpha=0.3)
axC.legend(fontsize=7.6, loc="upper left")
axC.tick_params(labelsize=8)

fig.suptitle(
    "Rear — Study 5: WHERE CAMBER RECOVERY COMES FROM     "
    "RECOVERY = −(dγ/dz)·(t/2)·(π/180)  ≈  half-track / FVSA     "
    f"stock: {rg.recovery_from_gain(rg.camber_gain_fast(P0, CAM)[0], HALF):.1f} %, "
    f"FVSA {G0['fvsa']:.0f} mm, RC {G0['rc_height']:+.0f} mm", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])

print("\nCANDIDATE UPRIGHTS")
print(f"  {'move':<16s}{'recovery':>10s}{'RC':>9s}{'road camber @1°':>17s}"
      f"{'@3°':>9s}{'@7°':>9s}")
for name, c, _col, _ls, rec, rc, _d, lim in curves:
    print(f"  {name:<16s}{rec:>9.1f}%{rc:>8.0f}mm{np.interp(1, roll, c):>16.2f}°"
          f"{np.interp(3, roll, c):>8.2f}°{np.interp(7, roll, c):>8.2f}°")
plt.savefig(_os.path.join(_PLOTS, "rear_task5_recovery_map.png"), dpi=130,
            bbox_inches="tight")
if _os.environ.get("STUDY_NOSHOW") != "1":
    plt.show()
