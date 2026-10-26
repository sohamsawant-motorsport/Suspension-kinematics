# -*- coding: utf-8 -*-
"""
GEOMETRY VIEWERS  -  see where the hard points are, and where the RC goes.
==========================================================================

Two panels for the explorer, both driven by the validated two-corner axle model
(suspension_tool/axle.py, 26 assertions), so what you see is the same geometry the
numbers come from -- not a separate drawing routine that could drift.

  Geometry2D   FRONT view (the VSusp picture: linkage, wheels, contact patches,
               instant centres, force lines, the roll centre where they cross,
               FVSA) plus SIDE and TOP views of one corner.
  Geometry3D   the whole axle in 3-D, rotatable, for spatial sense and packaging.

Both carry BUMP and ROLL sliders. Dragging roll is the point of the exercise: you
watch the instant centres swing and the roll centre run away sideways, which is
the thing that never quite lands from a graph.

The grey ghost is the REFERENCE geometry, so moving a hard point shows you
before-and-after in place.

WHY THE FRONT VIEW IS DRAWN AT CAR SCALE
----------------------------------------
The instant centre is often metres off the car (12.5 m on this front axle), so
auto-scaling to include it would shrink the car to a dot. Instead the view is
held at car scale, the force lines are drawn as far as they need to go, and the
IC position is reported numerically in the corner -- the same compromise VSusp
makes with its IC boxes.

SIGN CONVENTIONS are the project's: X lateral (+ outboard), Y longitudinal
(+ forward), Z up; + roll = the body leans LEFT = a right-hand turn.
See Study Guide/SIGN CONVENTIONS - read me first.md.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

_ST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "suspension_tool")
if _ST not in sys.path:
    sys.path.insert(0, _ST)
import axle                                                          # noqa: E402
import rear_geometry as _rg                                          # noqa: E402


def _spin(d, side):
    """The wheel's spin axis in car coordinates, from the camber and toe the axle
    model reports. axle_state does not expose it, and drawing the rim without it
    would give a wheel that ignores camber entirely."""
    s = _rg.spin_axis(d["camber"], d["toe"])
    return np.array([-s[0], s[1], s[2]]) if side == "L" else s

# link colours, shared by every view
C_UPPER, C_LOWER, C_TIE = "#3477c8", "#c86a34", "#8a5fbf"
C_UPRIGHT, C_WHEEL, C_GHOST = "#1a7", "#333", "0.72"
C_IC, C_RC, C_FORCE = "k", "#d4a017", "#c0392b"

# the links to draw, as (from, to, colour, width)
LINKS = [("UIF", "UBJ", C_UPPER, 2.2), ("UIR", "UBJ", C_UPPER, 2.2),
         ("LIF", "LBJ", C_LOWER, 2.2), ("LIR", "LBJ", C_LOWER, 2.2),
         ("TRI", "TRO", C_TIE, 1.8), ("LBJ", "UBJ", C_UPRIGHT, 3.0),
         ("SL", "SU", "#777", 1.6)]
JOINTS = ["UIF", "UIR", "LIF", "LIR", "UBJ", "LBJ", "TRI", "TRO", "WC", "SL", "SU"]


def solver_pair(d):
    """(pts_R, pts_L) in the form axle.axle_state wants, from a compute() result.

    corner_compute stores each corner's SOLVER-form points; the left corner's are
    mirrored into right-handed form, so they get mirrored back here.
    """
    return d["R"]["work_pts"], axle.mirror_points(d["L"]["work_pts"])


_PREP = {}          # id(compute result) -> (pts_R, pts_L, prep)


def axle_at(d, bump=0.0, roll=0.0):
    """Solve both corners at a trim state, or None if it will not reach.

    SPEED. axle_state rebuilds each corner's travel->angle branch (1121 solves per
    corner) unless it is handed a `prep`. Behind a slider that cost 2.4 s a frame.
    The branch does not depend on the trim, so it is built ONCE per geometry and
    cached here -- the same optimisation roll_sweep already uses. ~100x faster.
    """
    if d is None or d.get("error"):
        return None
    key = id(d)
    try:
        if key not in _PREP:
            pR, pL = solver_pair(d)
            cam = d["R"]["static_camber"]
            # keep the last few: the ghost and the current geometry are queried
            # alternately every frame, so clearing on each miss made them evict
            # one another and the cache never hit.
            while len(_PREP) >= 4:
                _PREP.pop(next(iter(_PREP)))
            _PREP[key] = (pR, pL, {"R": axle._corner_prep(pR, cam, False),
                                   "L": axle._corner_prep(pL, cam, True)})
        pR, pL, prep = _PREP[key]
        return axle.axle_state(pR, pL, d["R"]["static_camber"],
                               left_bump=bump, right_bump=bump, body_roll=roll,
                               prep=prep)
    except Exception:                                        # noqa: BLE001
        return None


def _wheel_outline(WC, spin, R, n=40):
    """The wheel rim as a circle in the plane normal to its spin axis (3-D view)."""
    s = np.asarray(spin, float)
    s = s / np.linalg.norm(s)
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, s)) > 0.95:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(s, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    t = np.linspace(0, 2 * np.pi, n)
    return np.asarray(WC, float)[None, :] + R * (np.cos(t)[:, None] * e1[None, :]
                                                 + np.sin(t)[:, None] * e2[None, :])


def _tyre_section(WC, spin, R, width=325.0):
    """The tyre as seen in the FRONT view: a rectangle of section-width by
    diameter, leaned over by the camber. Projecting the rim circle instead gives
    a thin ellipse that is geometrically right but reads as a mistake."""
    WC = np.asarray(WC, float)[[0, 2]]
    s = np.asarray(spin, float)
    up = np.array([-s[2], s[0]])           # in-plane "up", perpendicular to spin
    up /= np.linalg.norm(up)
    across = np.array([-up[1], up[0]]) * 0.5 * width
    bot, top = WC - R * up, WC + R * up
    return np.array([bot - across, bot + across, top + across, top - across,
                     bot - across])



# ---------------------------------------------------------------------------
#  construction helpers -- the dotted lines an engineer draws by hand
# ---------------------------------------------------------------------------


def _to_ground(P_lo, P_hi, i, j):
    """Where the line through two 3-D points crosses j = 0, in the (i, j) plane.

    Used for the steering axis: in FRONT view (i=0, j=2) the crossing gives the
    scrub reference, in SIDE view (i=1, j=2) it gives the trail reference. Both
    are the same construction seen from two directions, which is exactly the
    point worth showing on a drawing.
    """
    a, b = np.asarray(P_lo, float), np.asarray(P_hi, float)
    dj = b[j] - a[j]
    if abs(dj) < 1e-9:
        return None
    t = (0.0 - a[j]) / dj
    return a[i] + t * (b[i] - a[i])


def _axis_line(P_lo, P_hi, i, j, above=120.0):
    """The steering axis as a drawable 2-D segment: from `above` mm past the
    upper joint, down to the ground."""
    a, b = np.asarray(P_lo, float), np.asarray(P_hi, float)
    g = _to_ground(a, b, i, j)
    if g is None:
        return [], []
    d = b - a
    n = np.hypot(d[i], d[j])
    if n < 1e-9:
        return [], []
    top = b + d / n * above
    return [g, top[i]], [0.0, top[j]]


def _extend(P_in, P_out, target, i, j, past=60.0):
    """A wishbone line extended from its inner pivot out to the instant centre.

    Drawn dotted, because it is a construction line and not a part.
    """
    a = np.asarray(P_in, float)
    t = np.asarray(target, float)
    return [a[i], t[i]], [a[j], t[j]]


def _circle(cx, cy, r, n=72):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return cx + r * np.cos(th), cy + r * np.sin(th)


def _rect(cx, cy, ux, uy, half_long, half_wide):
    """A rectangle centred at (cx, cy); `u` is the LONG axis direction."""
    u = np.array([ux, uy], float)
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    p = [c + half_long * u + half_wide * v, c + half_long * u - half_wide * v,
         c - half_long * u - half_wide * v, c - half_long * u + half_wide * v]
    p.append(p[0])
    p = np.array(p)
    return p[:, 0], p[:, 1]


#  extra colours for the construction layer
C_CONS, C_AXIS, C_DIM = "#8a8a8a", "#1f7a6e", "#b5179e"

#  section width of the fitted tyre, for the TOP view outline
TYRE_WIDTH_MM = 325.0

class _Base(ttk.Frame):
    """Shared plumbing: canvas, bump/roll sliders, reference ghost."""

    def __init__(self, nb, title, figsize):
        super().__init__(nb)
        nb.add(self, text=title)

        # SLIDERS ARE TK WIDGETS, NOT matplotlib ones. A matplotlib Slider calls
        # draw_idle() on every change, which forces a full canvas render and made
        # blitting pointless (it cost MORE: full render + blit). Tk scales fire
        # only our own callback, so the blit path is the only draw that happens.
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        self.v_bump = tk.DoubleVar(value=0.0)
        self.v_roll = tk.DoubleVar(value=0.0)
        self.lbl = ttk.Label(bar, text="", font=("Consolas", 9), width=52)
        for txt, var, lo, hi in (("bump [mm]", self.v_bump, -60.0, 60.0),
                                 ("roll [deg]", self.v_roll, -4.0, 4.0)):
            ttk.Label(bar, text=txt, font=("Segoe UI", 8)).pack(side="left",
                                                                padx=(6, 2))
            ttk.Scale(bar, from_=lo, to=hi, variable=var, length=210,
                      command=lambda _v: self._redraw()).pack(side="left")
        ttk.Button(bar, text="reset", width=6,
                   command=self._reset_trim).pack(side="left", padx=6)
        self.lbl.pack(side="left", padx=10)

        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self._base = self._cur = None

    def _reset_trim(self):
        self.v_bump.set(0.0)
        self.v_roll.set(0.0)
        self._redraw()

    def update(self, base, cur):
        self._base, self._cur = base, cur
        self._redraw()

    def _trim(self):
        return float(self.v_bump.get()), float(self.v_roll.get())

    def _on_resize(self, _e):
        """A resize invalidates any cached blit background."""
        if hasattr(self, "_bg"):
            self._bg = None


# =====================================================================
class Geometry2D(_Base):
    """Front view (large) + side and top views of the right corner.

    WHAT IS DRAWN ON TOP OF THE LINKAGE
    -----------------------------------
    The links alone tell you where the parts are. What an engineer actually
    reads off a suspension drawing is the CONSTRUCTION: the lines you add with a
    straightedge to find the angles. Those are drawn here as dotted lines in a
    separate colour, on all three views:

      FRONT  both wishbones extended to the INSTANT CENTRE, the steering axis
             carried down to the road (so KPI is an angle you can see), the
             wheel's own centre plane (so camber is visible as a lean), and the
             SCRUB marked on the ground between the axis and the patch.
      SIDE   the tyre at true diameter, the steering axis down to the road
             against a vertical reference (CASTER), and the TRAIL marked on the
             ground between that crossing and the patch.
      TOP    the tyre at true width, its centre line, and a car-longitudinal
             reference so TOE reads as the angle between them.

    Everything is drawn from the same `axle_state` the numbers come from, so the
    picture cannot disagree with the readout.

    LAYOUT. Legends and the numeric readout sit OUTSIDE the axes: a compact key
    under each view for that view's own construction, and one shared band across
    the bottom carrying the link colours (identical in all three) plus the live
    numbers.

    PERSISTENT ARTISTS. Every line is created once and only its DATA is updated
    per frame. Clearing and rebuilding the axes each time cost ~190 ms behind the
    slider (the solve itself is only 4 ms); this way a frame is a few ms and the
    slider tracks the mouse. The readout lives in its own invisible axes so it
    can be blitted with the rest -- figure-level text cannot.
    """

    def __init__(self, nb, title="Geometry 2D"):
        #  FIGSIZE IS THE INITIAL WINDOW REQUEST, not a cap: the canvas packs
        #  with expand=True, so maximising the window scales every view up. It
        #  must stay short enough that the bump/roll bar below it is still on
        #  screen -- an 8.6 in figure pushed that bar off the bottom entirely.
        super().__init__(nb, title, (10.9, 7.3))
        #  the right column carries two equal-aspect views, so it is given more
        #  of the width than the front view's 2:1 data actually needs
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.22, 1],
                                   left=0.062, right=0.978, top=0.952,
                                   bottom=0.225, hspace=0.50, wspace=0.22)
        self.axF = self.fig.add_subplot(gs[:, 0])
        self.axS = self.fig.add_subplot(gs[0, 1])
        self.axT = self.fig.add_subplot(gs[1, 1])
        #  the bottom band: shared legend on the right, live numbers on the left
        self.axL = self.fig.add_axes([0.062, 0.010, 0.916, 0.135])
        self.axL.set_axis_off()
        self._built = False
        self._limits = False
        self._bg = None

    # ------------------------------------------------------------ build once
    def _build(self):
        A = {}
        # ---- FRONT view
        ax = self.axF
        ax.axhline(0, color="#444", lw=1.6, zorder=1)
        ax.axvline(0, color="#999", lw=0.9, ls="--", zorder=1)
        A["gh_F"] = {s: [ax.plot([], [], color=C_GHOST, lw=1.4, zorder=2)[0]
                         for _ in LINKS] for s in ("L", "R")}
        A["ln_F"] = {s: [ax.plot([], [], color=c, lw=w, zorder=4)[0]
                         for _a, _b, c, w in LINKS] for s in ("L", "R")}
        A["jt_F"] = {s: ax.plot([], [], "o", ms=4.5, color="#222", ls="none",
                                zorder=6)[0] for s in ("L", "R")}
        A["rim_F"] = {s: ax.plot([], [], color=C_WHEEL, lw=1.6, zorder=3)[0]
                      for s in ("L", "R")}
        A["pa_F"] = {s: ax.plot([], [], "v", ms=9, color=C_FORCE, ls="none",
                                zorder=7)[0] for s in ("L", "R")}
        A["fl_F"] = {s: ax.plot([], [], color=C_FORCE, lw=1.3, alpha=0.85,
                                zorder=5)[0] for s in ("L", "R")}
        # ---- construction layer, BOTH corners
        A["cons_F"] = {s: [ax.plot([], [], ls=":", lw=1.1, color=C_CONS,
                                   zorder=3)[0] for _ in range(2)]
                       for s in ("L", "R")}
        A["ic_F"] = {s: ax.plot([], [], "x", ms=8, mew=1.8, color=C_IC,
                                ls="none", zorder=8)[0] for s in ("L", "R")}
        A["tcl_F"] = {s: ax.plot([], [], ls=":", lw=1.4, color=C_WHEEL,
                                 alpha=0.9, zorder=6)[0] for s in ("L", "R")}
        A["sax_F"] = {s: ax.plot([], [], ls="-.", lw=1.5, color=C_AXIS,
                                 zorder=6)[0] for s in ("L", "R")}
        A["scr_F"] = {s: ax.plot([], [], lw=3.0, color=C_DIM, alpha=0.95,
                                 solid_capstyle="butt", zorder=8)[0]
                      for s in ("L", "R")}
        A["lbl_F"] = {s: ax.text(0, 0, "", fontsize=7.2, color=C_AXIS,
                                 ha="center", va="bottom", zorder=10)
                      for s in ("L", "R")}
        ax.set_xlabel("X — lateral  [mm]   (+ = right)", fontsize=9, labelpad=2)
        ax.set_ylabel("Z — height  [mm]", fontsize=9)
        ax.set_title("FRONT VIEW — linkage, instant centres, KPI, scrub, "
                     "ROLL CENTRE", fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        A["rc"], = ax.plot([], [], "o", ms=13, mfc=C_RC, mec="k", mew=1.6,
                           zorder=9)

        # ---- SIDE (Y-Z) and TOP (X-Y), right corner only
        for tag, ax2, ttl, xl, yl in (("S", self.axS, "SIDE VIEW (right corner) — caster & trail",
                                       "Y — longitudinal [mm]", "Z — height [mm]"),
                                      ("T", self.axT, "TOP VIEW (right corner) — toe",
                                       "X — lateral [mm]", "Y — longitudinal [mm]")):
            A["gh_" + tag] = [ax2.plot([], [], color=C_GHOST, lw=1.3)[0]
                              for _ in LINKS]
            A["ln_" + tag] = [ax2.plot([], [], color=c, lw=w)[0]
                              for _a, _b, c, w in LINKS]
            A["jt_" + tag], = ax2.plot([], [], "o", ms=4, color="#222", ls="none")
            A["pa_" + tag], = ax2.plot([], [], "v", ms=8, color=C_FORCE, ls="none")
            A["tyre_" + tag], = ax2.plot([], [], color=C_WHEEL, lw=1.5, alpha=0.9)
            A["ref_" + tag], = ax2.plot([], [], ls=":", lw=1.2, color="#999")
            A["cl_" + tag], = ax2.plot([], [], ls=":", lw=1.4, color=C_WHEEL)
            A["ax_" + tag], = ax2.plot([], [], ls="-.", lw=1.5, color=C_AXIS)
            A["dim_" + tag], = ax2.plot([], [], lw=3.0, color=C_DIM,
                                        solid_capstyle="butt")
            #  anchored in AXES fractions, not data: a label that tracked the
            #  joints kept sliding under the frame as the tyre resized the view.
            A["lbl_" + tag] = ax2.text(0.025, 0.975, "", fontsize=7.4,
                                       color=C_AXIS, ha="left", va="top",
                                       transform=ax2.transAxes,
                                       bbox=dict(boxstyle="round,pad=0.3",
                                                 fc="#ffffff", ec="0.8",
                                                 alpha=0.9))
            if tag == "S":
                ax2.axhline(0, color="#444", lw=1.3)
            ax2.set_xlabel(xl, fontsize=8)
            ax2.set_ylabel(yl, fontsize=8)
            ax2.set_title(ttl, fontsize=9)
            ax2.set_aspect("equal")
            ax2.grid(alpha=0.25)
            ax2.tick_params(labelsize=7.5)

        # ---- per-view keys, OUTSIDE each axes ----------------------------
        self._keys = []

        def key(ax_, items, ncol, _unused=None):
            #  anchored later, in _place_keys(), against the box that was really
            #  drawn -- see that method for why neither simple anchor works.
            h = [ax_.plot([], [], **kw)[0] for kw, _l in items]
            lg = ax_.legend(h, [l for _kw, l in items], loc="upper center",
                            ncol=ncol, fontsize=6.8, frameon=True,
                            framealpha=0.92, handlelength=2.1, borderpad=0.35,
                            columnspacing=1.1, handletextpad=0.5)
            self._keys.append((ax_, lg))

        key(self.axF, [
            (dict(ls=":", lw=1.1, color=C_CONS), "wishbone → instant centre"),
            (dict(ls="none", marker="x", ms=7, mew=1.8, color=C_IC), "instant centre"),
            (dict(ls="-.", lw=1.5, color=C_AXIS), "steering axis (KPI)"),
            (dict(ls=":", lw=1.4, color=C_WHEEL), "wheel centre plane (camber)"),
            (dict(lw=3.0, color=C_DIM), "scrub radius"),
            (dict(ls="none", marker="o", ms=8, mfc=C_RC, mec="k"), "roll centre"),
        ], 3)
        key(self.axS, [
            (dict(lw=1.5, color=C_WHEEL), "tyre"),
            (dict(ls="-.", lw=1.5, color=C_AXIS), "steering axis (caster)"),
            (dict(ls=":", lw=1.2, color="#999"), "vertical reference"),
            (dict(lw=3.0, color=C_DIM), "mechanical trail"),
        ], 2)
        key(self.axT, [
            (dict(lw=1.5, color=C_WHEEL), "tyre"),
            (dict(ls=":", lw=1.4, color=C_WHEEL), "wheel centre line"),
            (dict(ls=":", lw=1.2, color="#999"), "car longitudinal"),
        ], 3)

        # ---- shared band: link colours + the live numbers ----------------
        shared = [(dict(color=C_UPPER, lw=2.2), "upper wishbone"),
                  (dict(color=C_LOWER, lw=2.2), "lower wishbone"),
                  (dict(color=C_TIE, lw=1.8), "tie rod"),
                  (dict(color=C_UPRIGHT, lw=3.0), "upright"),
                  (dict(color="#777", lw=1.6), "damper"),
                  (dict(color=C_WHEEL, lw=1.6), "wheel"),
                  (dict(color=C_FORCE, lw=1.3), "force line to RC"),
                  (dict(ls="none", marker="v", ms=8, color=C_FORCE), "contact patch"),
                  (dict(color=C_GHOST, lw=1.4), "reference geometry")]
        hs = [self.axL.plot([], [], **kw)[0] for kw, _l in shared]
        self.axL.legend(hs, [l for _kw, l in shared], loc="center right",
                        bbox_to_anchor=(1.0, 0.5), ncol=5, fontsize=6.9,
                        frameon=True, framealpha=0.95, handlelength=2.2,
                        title="links — all three views", title_fontsize=7.4,
                        borderpad=0.5, columnspacing=1.4)
        A["txt"] = self.axL.text(0.0, 0.5, "", transform=self.axL.transAxes,
                                 va="center", ha="left", fontsize=8.2,
                                 family="monospace", zorder=10,
                                 linespacing=1.35,
                                 bbox=dict(boxstyle="round", fc="#f7f7f7",
                                           ec="0.7"))
        self.axL.set_xlim(0, 1)
        self.axL.set_ylim(0, 1)
        self.A = A
        self._built = True

    # ------------------------------------------------------------ per frame
    def _redraw(self):
        if not self._built:
            self._build()
        A = self.A
        bump, roll = self._trim()
        st = axle_at(self._cur, bump, roll)
        gh = axle_at(self._base, bump, roll)

        if st is None:                       # keep the artwork, just blank it
            for k in ("ln_F", "rim_F", "pa_F", "fl_F", "cons_F", "ic_F",
                      "tcl_F", "sax_F", "scr_F"):
                for s in ("L", "R"):
                    v = A[k][s]
                    (v.set_data([], []) if not isinstance(v, list)
                     else [ln.set_data([], []) for ln in v])
            A["rc"].set_data([], [])
            A["txt"].set_text("this geometry does not solve at that trim")
            self.canvas.draw_idle()
            return

        rc = st["roll_centre"]
        for side in ("L", "R"):
            d = st[side]
            for k, (a, b, _c, _w) in enumerate(LINKS):
                A["ln_F"][side][k].set_data([d[a][0], d[b][0]],
                                            [d[a][2], d[b][2]])
                if gh is not None:
                    g = gh[side]
                    A["gh_F"][side][k].set_data([g[a][0], g[b][0]],
                                                [g[a][2], g[b][2]])
                else:
                    A["gh_F"][side][k].set_data([], [])
            A["jt_F"][side].set_data([d[j][0] for j in JOINTS],
                                     [d[j][2] for j in JOINTS])
            R_wh = abs(d["WC"][2] - d["patch"][2])
            sec = _tyre_section(d["WC"], _spin(d, side), R_wh)
            A["rim_F"][side].set_data(sec[:, 0], sec[:, 1])
            A["pa_F"][side].set_data([d["patch"][0]], [d["patch"][2]])
            A["fl_F"][side].set_data([d["patch"][0], rc[0]],
                                     [d["patch"][2], rc[1]])

            # ---- construction: wishbones extended to the instant centre
            ic = d["IC"]
            A["cons_F"][side][0].set_data(*_extend(d["UIF"], d["UBJ"], ic, 0, 2))
            A["cons_F"][side][1].set_data(*_extend(d["LIF"], d["LBJ"], ic, 0, 2))
            A["ic_F"][side].set_data([ic[0]], [ic[2]])

            # ---- the wheel's own centre plane -- camber made visible
            sp = _spin(d, side)
            up = np.array([-sp[2], sp[0]], float)
            up /= np.linalg.norm(up)
            A["tcl_F"][side].set_data(
                [d["WC"][0] - R_wh * up[0], d["WC"][0] + R_wh * up[0]],
                [d["WC"][2] - R_wh * up[1], d["WC"][2] + R_wh * up[1]])

            # ---- steering axis to the road, and the scrub it leaves
            A["sax_F"][side].set_data(*_axis_line(d["LBJ"], d["UBJ"], 0, 2))
            xg = _to_ground(d["LBJ"], d["UBJ"], 0, 2)
            if xg is None:
                A["scr_F"][side].set_data([], [])
                A["lbl_F"][side].set_text("")
            else:
                A["scr_F"][side].set_data([xg, d["patch"][0]], [0.0, 0.0])
                scrub = abs(d["patch"][0] - xg)
                A["lbl_F"][side].set_position((0.5 * (xg + d["patch"][0]), -165.0))
                A["lbl_F"][side].set_text(f"KPI {abs(d['kpi']):.2f}°\n"
                                          f"scrub {scrub:.1f}")
        A["rc"].set_data([rc[0]], [rc[1]])

        d = st["R"]
        g = gh["R"] if gh is not None else None
        for tag, (i, j) in (("S", (1, 2)), ("T", (0, 1))):
            for k, (a, b, _c, _w) in enumerate(LINKS):
                A["ln_" + tag][k].set_data([d[a][i], d[b][i]], [d[a][j], d[b][j]])
                A["gh_" + tag][k].set_data(
                    [g[a][i], g[b][i]] if g else [],
                    [g[a][j], g[b][j]] if g else [])
            A["jt_" + tag].set_data([d[q][i] for q in JOINTS],
                                    [d[q][j] for q in JOINTS])
            A["pa_" + tag].set_data([d["patch"][i]], [d["patch"][j]])

        R_wh = abs(d["WC"][2] - d["patch"][2])
        sp = _spin(d, "R")

        # ---- SIDE: tyre, caster against vertical, trail on the ground
        A["tyre_S"].set_data(*_circle(d["WC"][1], d["WC"][2], R_wh))
        A["cl_S"].set_data([], [])
        A["ax_S"].set_data(*_axis_line(d["LBJ"], d["UBJ"], 1, 2))
        yg = _to_ground(d["LBJ"], d["UBJ"], 1, 2)
        if yg is None:
            A["ref_S"].set_data([], [])
            A["dim_S"].set_data([], [])
            A["lbl_S"].set_text("")
        else:
            A["ref_S"].set_data([yg, yg], [0.0, d["UBJ"][2] + 120.0])
            A["dim_S"].set_data([yg, d["patch"][1]], [0.0, 0.0])
            A["lbl_S"].set_text(f"caster {abs(d['caster']):.2f}°\n"
                                f"trail {abs(d['patch'][1] - yg):.1f}")

        # ---- TOP: tyre at true width, its centre line, car longitudinal
        fwd = np.array([-sp[1], sp[0]], float)
        n = np.linalg.norm(fwd)
        if n > 1e-9:
            fwd /= n
            A["tyre_T"].set_data(*_rect(d["WC"][0], d["WC"][1], fwd[0], fwd[1],
                                        R_wh, 0.5 * TYRE_WIDTH_MM))
            A["cl_T"].set_data([d["WC"][0] - R_wh * fwd[0],
                                d["WC"][0] + R_wh * fwd[0]],
                               [d["WC"][1] - R_wh * fwd[1],
                                d["WC"][1] + R_wh * fwd[1]])
            A["ref_T"].set_data([d["WC"][0], d["WC"][0]],
                                [d["WC"][1] - R_wh, d["WC"][1] + R_wh])
            A["ax_T"].set_data([], [])
            A["dim_T"].set_data([], [])
            w = ("IN" if d["toe"] > 1e-3 else
                 "OUT" if d["toe"] < -1e-3 else "straight ahead")
            A["lbl_T"].set_text(f"toe {d['toe']:+.3f}°  ({w})")

        # LIMITS ARE SET ONCE, not per frame. With equal-aspect axes, changing
        # the limits forces a full re-layout, and autoscaling two more axes on
        # top of it was most of the frame cost. The geometry only moves tens of
        # mm, so fixed limits sized off the static car are correct and free.
        if not self._limits:
            half = max(abs(st["R"]["patch"][0]), abs(st["L"]["patch"][0])) + 280
            #  the IC is the whole point of the construction lines, so widen to
            #  include it -- but only while it is still on a car-sized scale.
            for q in (st["L"]["IC"][0], st["R"]["IC"][0]):
                if np.isfinite(q) and abs(q) < 2600.0:
                    half = max(half, abs(q) + 110.0)
            self.axF.set_xlim(-half, half)
            self.axF.set_ylim(-300, 780)
            #  the tyre is far bigger than the link cluster, so the side and top
            #  views have to be sized to IT or it runs off the frame
            for ax2, (i, j) in ((self.axS, (1, 2)), (self.axT, (0, 1))):
                pts = np.array([d[q] for q in JOINTS] + [d["patch"]])
                lo = np.array([pts[:, i].min(), pts[:, j].min()])
                hi = np.array([pts[:, i].max(), pts[:, j].max()])
                lo = np.minimum(lo, [d["WC"][i] - R_wh, d["WC"][j] - R_wh])
                hi = np.maximum(hi, [d["WC"][i] + R_wh, d["WC"][j] + R_wh])
                pad = 40.0
                ax2.set_xlim(lo[0] - pad, hi[0] + pad)
                ax2.set_ylim(lo[1] - pad, hi[1] + pad)
            self._limits = True

        icL, icR = st["L"]["IC"], st["R"]["IC"]
        A["txt"].set_text(
            f"ROLL CENTRE ({rc[0]:+.0f}, {rc[1]:+.0f}) mm     "
            f"track {st['track']:.1f} mm     roll {st['roll_actual']:+.2f}°\n"
            f"IC L ({icL[0]:+.0f}, {icL[2]:+.0f})  FVSA {st['L']['fvsa']:.0f}"
            f"      IC R ({icR[0]:+.0f}, {icR[2]:+.0f})  FVSA {st['R']['fvsa']:.0f}\n"
            f"KPI {abs(d['kpi']):.2f}°   caster {abs(d['caster']):.2f}°   "
            f"camber {d['camber']:+.2f}°   toe {d['toe']:+.3f}°")
        self.lbl.config(text=f"bump {bump:+5.0f} mm   roll {roll:+5.1f} deg   "
                             f"RC ({rc[0]:+.0f}, {rc[1]:+.0f}) mm")
        self._blit()

    # ------------------------------------------------------------ blitting
    # Even with persistent artists, draw_idle() re-renders the whole canvas
    # (~112 ms). Blitting restores a cached background and repaints only the
    # moving lines, per axes, which is what makes the slider track the mouse.
    def _artists(self):
        A = self.A
        out = [A["rc"], A["txt"]]
        for s in ("L", "R"):
            out += A["gh_F"][s] + A["ln_F"][s] + A["cons_F"][s]
            out += [A["jt_F"][s], A["rim_F"][s], A["pa_F"][s], A["fl_F"][s],
                    A["ic_F"][s], A["tcl_F"][s], A["sax_F"][s], A["scr_F"][s],
                    A["lbl_F"][s]]
        for tag in ("S", "T"):
            out += A["gh_" + tag] + A["ln_" + tag]
            out += [A["jt_" + tag], A["pa_" + tag], A["tyre_" + tag],
                    A["ref_" + tag], A["cl_" + tag], A["ax_" + tag],
                    A["dim_" + tag], A["lbl_" + tag]]
        return out

    def _place_keys(self):
        """Sit each key just under the axes box that was actually rendered.

        `ax.get_position()` returns the GRIDSPEC SLOT until a draw has happened;
        only afterwards does it report the box equal-aspect shrank it to. So this
        runs after the capture draw, and only once -- the boxes do not move again
        unless the window is resized, which drops the cached background anyway.
        """
        for ax_, lg in getattr(self, "_keys", []):
            bb = ax_.get_position()
            lg.set_bbox_to_anchor((bb.x0 + 0.5 * bb.width, bb.y0 - 0.052),
                                  transform=self.fig.transFigure)

    def _capture(self, _e=None):
        for a in self._artists():
            a.set_animated(True)
        self.canvas.draw()
        self._place_keys()
        self.canvas.draw()
        self._bg = {ax: self.canvas.copy_from_bbox(ax.bbox)
                    for ax in (self.axF, self.axS, self.axT, self.axL)}

    def _blit(self):
        # NOTE the missing `return` after _capture(): capturing marks every moving
        # artist animated=True, which EXCLUDES it from the normal draw. If we
        # returned here the very first frame painted nothing but the axes -- which
        # is exactly the empty viewer this had.
        if getattr(self, "_bg", None) is None:
            self._capture()
        for ax in (self.axF, self.axS, self.axT, self.axL):
            self.canvas.restore_region(self._bg[ax])
        for a in self._artists():
            a.axes.draw_artist(a)
        for ax in (self.axF, self.axS, self.axT, self.axL):
            self.canvas.blit(ax.bbox)


# =====================================================================
class Geometry3D(_Base):
    """The whole axle in 3-D. Drag to rotate; the sliders move the suspension."""

    def __init__(self, nb, title="Geometry 3D"):
        super().__init__(nb, title, (10.4, 7.2))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

    def _redraw(self):
        bump, roll = self._trim()
        st = axle_at(self._cur, bump, roll)
        ax = self.ax
        elev, azim = ax.elev, ax.azim          # keep the user's viewpoint
        ax.clear()
        if st is None:
            ax.text2D(0.5, 0.5, "this geometry does not solve at that trim",
                      ha="center", transform=ax.transAxes, color="#a33")
            self.canvas.draw_idle()
            return
        for side in ("L", "R"):
            d = st[side]
            for a, b, col, w in LINKS:
                p, q = d[a], d[b]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=col, lw=w)
            for j in JOINTS:
                ax.scatter([d[j][0]], [d[j][1]], [d[j][2]], s=16, c="#222",
                           depthshade=False)
            rim = _wheel_outline(d["WC"], _spin(d, side),
                                 abs(d["WC"][2] - d["patch"][2]))
            ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=C_WHEEL, lw=1.4)
            ax.scatter([d["patch"][0]], [d["patch"][1]], [d["patch"][2]], s=40,
                       c=C_FORCE, marker="v", depthshade=False)
        rc = st["roll_centre"]
        y = 0.5 * (st["L"]["patch"][1] + st["R"]["patch"][1])
        ax.scatter([rc[0]], [y], [rc[1]], s=110, c=C_RC, edgecolors="k",
                   depthshade=False, label="roll centre")
        for side in ("L", "R"):
            p = st[side]["patch"]
            ax.plot([p[0], rc[0]], [p[1], y], [p[2], rc[1]], color=C_FORCE,
                    lw=1.1, alpha=0.8)
        # a flat road patch for reference
        half = max(abs(st["R"]["patch"][0]), abs(st["L"]["patch"][0])) + 200
        gx, gy = np.meshgrid(np.linspace(-half, half, 2),
                             np.linspace(y - 500, y + 500, 2))
        ax.plot_surface(gx, gy, np.zeros_like(gx), color="#ddd", alpha=0.25)
        ax.set_xlabel("X lateral", fontsize=8)
        ax.set_ylabel("Y longitudinal", fontsize=8)
        ax.set_zlabel("Z up", fontsize=8)
        ax.set_title(f"3-D axle   ·   roll centre ({rc[0]:+.0f}, {rc[1]:+.0f}) mm"
                     f"   ·   drag to rotate", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper right")
        try:                                   # equal aspect where available
            ax.set_box_aspect((2 * half, 1000, 1100))
        except Exception:                      # noqa: BLE001
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=7)
        self.lbl.config(text=f"bump {bump:+5.0f} mm   roll {roll:+5.1f} deg   "
                             f"RC ({rc[0]:+.0f}, {rc[1]:+.0f}) mm")
        self.canvas.draw_idle()
