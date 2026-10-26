# -*- coding: utf-8 -*-
"""
HARD-POINT EXPLORER  -  type coordinates, read the geometry, see the curves.
=============================================================================

Run in VS Code's integrated terminal:
    python explorer.py
or double-click  ..\\RUN TOOL.bat

LEFT PANEL
    "Is Symmetric (RH = LH)" checkbox, then LEFT / RIGHT tabs holding every hard
    point of that corner as editable X / Y / Z boxes, plus that side's static
    camber. Underneath, the geometry those points PRODUCE -- KPI, caster, scrub,
    trail, camber gain, recovery, bump steer, roll-centre height, motion ratio,
    steering ratio, link lengths, packaging.

    Symmetric ticked  -> the RIGHT side is master, the left is its mirror and its
                         boxes are locked. Untick to build an asymmetric car.

RIGHT PANEL
    Ten tabs. Every curve is drawn FOUR times -- reference and current, right and
    left -- so an asymmetric change is visible immediately:

        grey dashed / grey dotted   reference   R / L
        crimson solid / blue dashed current     R / L

    Hover any panel for a reference | current | delta table for both sides.

        1   camber change vs steer  --  KPI effect     (caster held)
        2   camber change vs steer  --  caster effect  (KPI held)
        3   camber map over KPI x caster, both geometries marked
        4   contact-patch path on the ground, both wheels, RACK slider
        5   scrub radius & mechanical trail vs steer
        6   wheelbase & half-track vs steer
        7   camber gain in heave & bump steer
        8   camber vs body roll
        9   roll-centre height vs travel
       10   KPI & caster vs travel

HOW THE LEFT CORNER IS SOLVED
-----------------------------
No left-hand physics is written anywhere. A left corner MIRRORED in X is itself a
valid right corner, so the left side is mirrored, pushed through the SAME
validated right-corner pipeline, and mirrored back. That holds whether or not the
car is symmetric.

Only absolute POSITIONS need un-mirroring (patch X, ground point X). Signed
quantities are already right: scrub is "patch outboard of the axis" and outboard
mirrors with the corner, KPI is "axis leans inboard at the top" and inboard
mirrors too, camber is measured against outboard, and caster/trail/toe live in Y
and Z which the mirror never touches.

Nothing is derived from a typed angle. KPI and caster are computed FROM the ball
joints you entered, exactly the way config.stock_geometry_from_csv() does it --
the project has been bitten once by an angle typed in one file and derived in
another, and the two disagreeing was worth 60 mm of scrub radius. Here there is
only one direction of travel: points in, angles out.

Physics comes entirely from the validated cores -- steer_camber, scrub_trail and
wishbone_solver. This file draws; it does not calculate.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# the validated two-corner axle model lives in the (parked) suspension_tool
# folder; it is a pure physics core with 24 assertions, reused here unchanged for
# the roll-centre-in-roll view. Nothing else in the parked tool is imported.
_ST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "suspension_tool")
if _ST not in sys.path:
    sys.path.insert(0, _ST)
import axle                                                          # noqa: E402
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure                                  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle             # noqa: E402
from matplotlib.widgets import Slider                                 # noqa: E402
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,     # noqa: E402
                                               NavigationToolbar2Tk)

import ackermann                                                      # noqa: E402
import config as cfg                                                  # noqa: E402
import rim_envelope                                                   # noqa: E402
from steer_camber import camber_change                                # noqa: E402
from scrub_trail import (kpi_caster, corner_state, scrub_radius,      # noqa: E402
                         mechanical_trail, wheel_spin_axis, rotate_about_line)
from wishbone_solver import (corner_geometry, solve_sweep,             # noqa: E402
                            linkage_roll_centre, solve_on_circle)

STEER_LO, STEER_HI = cfg.STEER_RANGE_PATCH
DROOP, BUMP = -70.0, 70.0
RACK_LO, RACK_HI, RACK_N = -70.0, 70.0, 281

UBJ_NAME, LBJ_NAME, TIE_NAME = cfg.UBJ_NAME, cfg.LBJ_NAME, cfg.TIE_NAME

# line styles: (colour, linestyle, linewidth, zorder)
STYLE = {
    ("ref", "R"): ("0.45", "--", 1.9, 3),
    ("ref", "L"): ("0.65", ":", 1.9, 2),
    ("cur", "R"): ("crimson", "-", 2.4, 5),
    ("cur", "L"): ("#1f77b4", "--", 1.8, 4),
}
LABEL = {("ref", "R"): "ref  R", ("ref", "L"): "ref  L",
         ("cur", "R"): "cur  R", ("cur", "L"): "cur  L"}
SERIES = [("ref", "R"), ("ref", "L"), ("cur", "R"), ("cur", "L")]

POINT_ROWS = [
    ("Upper wishbone - inner front", "UPPER arm   inner front", True),
    ("Upper wishbone - inner rear",  "UPPER arm   inner rear",  False),
    ("Upper wishbone - outer (UBJ)", "UPPER ball joint  UBJ",   False),
    ("Lower wishbone - inner front", "LOWER arm   inner front", True),
    ("Lower wishbone - inner rear",  "LOWER arm   inner rear",  False),
    ("Lower wishbone - outer (LBJ)", "LOWER ball joint  LBJ",   False),
    ("Tie rod - inner",              "TIE ROD  inner (rack)",   True),
    ("Tie rod - outer",              "TIE ROD  outer",          False),
    ("Wheel centre",                 "WHEEL CENTRE",            True),
    ("Lower Shock mount",            "DAMPER   lower",          True),
    ("Upper Shock mount",            "DAMPER   upper",          False),
]


def mirror_pts(pts):
    """A left corner reflected in X -- which is a perfectly good RIGHT corner."""
    return {k: np.array([-v[0], v[1], v[2]], float) for k, v in pts.items()}


def ok(d):
    return d is not None and not d.get("error")


# =====================================================================
#  one corner
# =====================================================================
def corner_compute(pts_in, static_camber, is_left, droop=DROOP, bump=BUMP,
                   reuse=None):
    """Every curve for ONE corner. `pts_in` is that corner's real hard points;
    if it is the left corner they are mirrored first and the few absolute
    positions are mirrored back at the end."""
    pts = mirror_pts(pts_in) if is_left else pts_in
    try:
        LBJ, UBJ, WC = pts[LBJ_NAME], pts[UBJ_NAME], pts["Wheel centre"]
        TRO, TRI = pts[TIE_NAME], pts["Tie rod - inner"]
    except KeyError as k:
        return dict(error=f"missing hard point {k}")

    kpi, cas = kpi_caster(LBJ, UBJ)

    # DIRECTIONAL INPUTS ARE MIRRORED FOR THE LEFT CORNER.
    # The left corner is solved in mirrored (right-handed) space, so a car steer
    # of +delta is a rotation of -delta about ITS kingpin, and a body roll that
    # compresses the left droops the right. Feeding both corners the same signed
    # input made every steer- and roll-based plot draw two identical curves --
    # which is only true in heave.
    sgn = -1.0 if is_left else 1.0
    steer = np.linspace(STEER_LO, STEER_HI, 281)
    st = corner_state(LBJ, UBJ, WC, sgn * steer, static_camber)

    # exact steering ratio from the velocity relation (config.steer_angle_for_rack)
    t = TRO - TRI
    t = t / np.linalg.norm(t)
    k = UBJ - LBJ
    k = k / np.linalg.norm(k)
    den = float(np.dot(np.cross(k, TRO - LBJ), t))
    ratio = t[0] / den if abs(den) > 1e-12 else np.nan          # rad per mm rack

    # REUSE: on a symmetric car the two corners share the same solver points, so
    # the BUMP sweep (pure heave) is identical and only needs computing once --
    # it is ~95 % of the cost here. The STEER series below is cheap and IS
    # recomputed, because the left sees the mirrored steer.
    if reuse is not None:
        sw, rc = reuse
    else:
        try:
            g = corner_geometry(pts, static_camber)
            travel = np.linspace(droop, bump, 121)
            sw = solve_sweep(g, travel, theta_span_deg=28.0, n_theta=1121)
            rc = linkage_roll_centre(g, travel)
        except Exception as exc:                  # noqa: BLE001 - shown to user
            return dict(error=f"the linkage will not solve: {exc}")

    if not np.all(np.isfinite(sw["camber"])):
        return dict(error="the linkage cannot reach that travel range -- an arm "
                          "is over-rotating or a link is too short")

    i0 = int(np.argmin(np.abs(sw["bump"])))
    cam_rate = np.gradient(sw["camber"], sw["bump"])
    toe_rate = np.gradient(sw["toe"], sw["bump"])
    half = float(WC[0])
    # tilt this wheel gets from the body rolling: +arctan(bump/half) either side
    tilt = np.degrees(np.arctan(sw["bump"] / half))
    recovery = -half * np.radians(1.0) * cam_rate[i0] * 100.0

    # ROLL is put on a COMMON axis for both corners. Deriving phi from each
    # corner's own travel array gave the two sides different (and opposite) roll
    # ranges, so the curves barely overlapped and could not be compared. Instead
    # sweep body roll directly and interpolate each corner's camber at the travel
    # THAT corner sees: at +roll the left compresses and the right droops.
    phi_max = np.degrees(np.arctan(min(abs(DROOP), bump) / half)) * 0.98
    phi = np.linspace(-phi_max, phi_max, 161)
    z_of_phi = half * np.tan(np.radians(phi)) * (1.0 if is_left else -1.0)
    cam_at = np.interp(z_of_phi, sw["bump"], sw["camber"])
    tilt_at = np.degrees(np.arctan(z_of_phi / half))
    cam_road = cam_at + tilt_at

    # CAR-FRAME camber: measured against one absolute vertical for both wheels,
    # + = the top of the wheel leans toward +X (the car's right). The per-wheel
    # convention (+ = leans away from that wheel's own centreline) is kept for
    # every NUMBER in the readout, because that is what setup sheets use; the
    # plots use the car frame so a rolling car reads as one picture.
    car = -1.0 if is_left else 1.0

    s = -1.0 if is_left else 1.0        # un-mirror absolute positions only
    return dict(
        error=None, pts=pts_in, work_pts=pts, is_left=is_left, _sweep=(sw, rc),
        kpi=kpi, caster=cas, static_camber=static_camber,
        scrub0=scrub_radius(LBJ, UBJ, WC, static_camber),
        trail0=mechanical_trail(LBJ, UBJ, WC, static_camber),
        steer=steer, cam_steer=st["camber"],
        scrub_steer=st["scrub"], trail_steer=st["trail"],
        patch_x=s * st["P"][:, 0], patch_y=st["P"][:, 1],
        wheelbase=st["P"][:, 1] - cfg.rear_patch_y(),
        halftrack=st["P"][:, 0],        # magnitude: comparable L vs R
        jack=st["jack"],
        bump=sw["bump"], cam_bump=sw["camber"], toe_bump=sw["toe"],
        kpi_bump=sw["kpi"], caster_bump=sw["caster"], mr=sw["MR"],
        mr_disp=spr.to_display(sw["MR"]),
        rc=rc, phi=phi,
        cam_road=cam_road,                          # per-wheel, vs the road
        cam_road_car=car * cam_road,                # car frame, for plotting
        cam_bump_car=car * sw["camber"],
        cam_steer_car=car * st["camber"],
        toe_bump_car=car * sw["toe"],
        cam_gain=float(cam_rate[i0]), bump_steer=float(toe_rate[i0]),
        recovery=float(recovery), rc0=float(rc[i0]), mr0=float(sw["MR"][i0]),
        mr0_disp=float(spr.to_display(sw["MR"][i0])),
        kingpin=float(np.linalg.norm(UBJ - LBJ)),
        eff_arm=float(1.0 / abs(ratio)) if np.isfinite(ratio) else np.nan,
        steer20=float(np.degrees(abs(ratio) * cfg.RACK_TRAVEL_MM))
        if np.isfinite(ratio) else np.nan,
        upper_arm=float(0.5 * (np.linalg.norm(UBJ - pts["Upper wishbone - inner front"])
                               + np.linalg.norm(UBJ - pts["Upper wishbone - inner rear"]))),
        lower_arm=float(0.5 * (np.linalg.norm(LBJ - pts["Lower wishbone - inner front"])
                               + np.linalg.norm(LBJ - pts["Lower wishbone - inner rear"]))),
        halftrack0=half,
    )


# =====================================================================
#  the two front wheels, driven by the RACK
# =====================================================================
def steer_from_rack(pts, rack_grid, sign=1.0):
    """Upright rotation about the kingpin axis [deg] for each rack displacement.

    The rack is a bar sliding laterally, so it moves the tie-rod INNER point by
    `rack` in X. The tie rod is rigid, so the OUTER point must stay that same
    length away while riding its circle about the kingpin line -- the classic
    `A cos t + B sin t = C`, solved exactly by wishbone_solver.solve_on_circle.

    `sign` is -1 for a corner being solved in MIRRORED space: one rack bar moves
    both inner points the same way in the CAR frame, which is the opposite way in
    the mirrored frame.

    Exact, not the first-order estimate: config.steer_angle_for_rack gives 10.04
    deg at 20 mm, this gives 10.72, and that nonlinearity is why Ackermann exists.
    Swept outward from rack = 0, each solve seeded with the last, so it cannot
    jump to the other (unphysical) intersection. NaN past the travel limit.
    """
    LBJ, UBJ = pts[LBJ_NAME], pts[UBJ_NAME]
    TRO, TRI = pts[TIE_NAME], pts["Tie rod - inner"]
    u = UBJ - LBJ
    u = u / np.linalg.norm(u)
    Lt = float(np.linalg.norm(TRO - TRI))

    out = np.full(len(rack_grid), np.nan)
    i0 = int(np.argmin(np.abs(rack_grid)))
    for order in (range(i0, len(rack_grid)), range(i0, -1, -1)):
        guess = 0.0
        for i in order:
            try:
                t, _ = solve_on_circle(TRO, LBJ, u,
                                       TRI + np.array([sign * rack_grid[i], 0.0, 0.0]),
                                       Lt, guess)
            except ValueError:
                break               # travel limit -- leave the rest NaN
            out[i] = t
            guess = t
    return out


def axle_patch_paths(pts_R, cam_R, pts_L, cam_L, rack):
    """Contact-patch path of BOTH front wheels as the ONE rack bar sweeps.

    The left corner is solved in mirrored space with the rack sign flipped, then
    mirrored back -- so an asymmetric car is handled exactly like a symmetric one.

    Both wheels' heading is reported in a single car frame (positive toward +X),
    so `ack` is a like-for-like Ackermann difference. Note the kingpin ROTATION is
    not the same as the wheel's HEADING when the axis is tilted -- the paths use
    the rotation (what corner_state consumes), the readout uses the heading.
    """
    mR, mL = pts_R, mirror_pts(pts_L)

    dR = steer_from_rack(mR, rack, +1.0)
    dL = steer_from_rack(mL, rack, -1.0)
    good = np.isfinite(dR) & np.isfinite(dL)
    if good.sum() < 5:
        return None
    rack, dR, dL = rack[good], dR[good], dL[good]

    out = dict(rack=rack)
    for key, pts, cam, d, flip in (("R", mR, cam_R, dR, False),
                                   ("L", mL, cam_L, dL, True)):
        LBJ, UBJ, WC = pts[LBJ_NAME], pts[UBJ_NAME], pts["Wheel centre"]
        stt = corner_state(LBJ, UBJ, WC, d, cam)
        G = stt["G"]
        s = -1.0 if flip else 1.0
        i0 = int(np.argmin(np.abs(rack)))

        u = UBJ - LBJ
        u = u / np.linalg.norm(u)
        ax = rotate_about_line(wheel_spin_axis(cam), np.zeros(3), u, d)
        # heading from straight-ahead (+Y), positive toward +X, in the CAR frame;
        # the mirrored corner's rolling direction is -(z x s), hence the sign
        head = np.degrees(np.arctan2((1.0 if flip else -1.0) * ax[:, 1], ax[:, 0]))

        out[key] = dict(
            P=np.column_stack([s * stt["P"][:, 0], stt["P"][:, 1]]),
            G=np.array([s * G[0], G[1]]),
            P0=np.array([s * stt["P"][i0, 0], stt["P"][i0, 1]]),
            scrub=stt["scrub"], trail=stt["trail"], jack=stt["jack"],
            d_track=stt["d_track"], d_base=stt["d_base"], camber=stt["camber"],
            steer=head, kingpin_rot=(-1.0 if flip else 1.0) * d)
    out["ack"] = out["L"]["steer"] - out["R"]["steer"]
    return out


ROLL_LO, ROLL_HI, ROLL_N = -5.0, 5.0, 11


def roll_migration(pts_R, cam_R, pts_L, cam_L):
    """Roll-centre migration vs BODY ROLL, from the validated two-corner axle.

    This is the load case tab 9 (RC vs wheel travel) cannot show. Under HEAVE
    both wheels move the same way and a symmetric RC just slides up and down the
    centreline. Under ROLL the wheels move oppositely, so the two force lines no
    longer cross on the centreline and the RC migrates SIDEWAYS as well -- which
    is the dangerous, sudden movement that upsets a car mid-corner.

    axle.roll_sweep converts each roll angle into equal-and-opposite wheel
    travel (NOT a rigid rotation), keeping the tyres on the road. It takes a
    single static camber; the RIGHT corner's value is used. Hard-point asymmetry
    is fully modelled -- only a per-side static-camber split is deferred.
    """
    try:
        return axle.roll_sweep(pts_R, pts_L, cam_R,
                               np.linspace(ROLL_LO, ROLL_HI, ROLL_N))
    except Exception:                                 # noqa: BLE001
        return None


# =====================================================================
#  both corners
# =====================================================================
def compute(pts_R, cam_R, pts_L, cam_L, symmetric=False):
    R = corner_compute(pts_R, cam_R, False)
    if R.get("error"):
        return dict(error="RIGHT: " + R["error"])
    if symmetric:
        # A symmetric left corner is NOT simply a copy of the right: it sees the
        # MIRRORED steer and roll, so its steer-based curves genuinely differ.
        # What IS shared is the bump sweep (heave is symmetric), which is ~95 %
        # of the cost -- so hand it over and let the cheap steer part redo.
        # (An earlier version copied R wholesale; once mirrored inputs went in
        # that became stale, and validate_explorer caught it.)
        L = corner_compute(pts_L, cam_L, True, reuse=R["_sweep"])
        if L.get("error"):
            return dict(error="LEFT: " + L["error"])
    else:
        L = corner_compute(pts_L, cam_L, True)
        if L.get("error"):
            return dict(error="LEFT: " + L["error"])
    return dict(error=None, R=R, L=L,
                track=abs(float(pts_R["Wheel centre"][0]))
                + abs(float(pts_L["Wheel centre"][0])),
                axle=axle_patch_paths(pts_R, cam_R, pts_L, cam_L,
                                      np.linspace(RACK_LO, RACK_HI, RACK_N)),
                # the `t` in the Ackermann condition: steering-axis separation at
                # GROUND level, not the wheel track. Cheap, so it rides along.
                ack_t=ackermann.kingpin_ground_track(pts_R, pts_L, mirror_pts),
                # roll migration is ~1 s; it is computed LAZILY the first time the
                # RC-in-roll tab is opened (see Explorer.ensure_roll), so a normal
                # Apply that isn't looking at roll stays fast
                roll=None,
                roll_args=(pts_R, cam_R, pts_L, cam_L))


# ---------------------------------------------------------------- data pickers
def keyed(xk, yk):
    """Plot d[side][yk] against d[side][xk], for both geometries and both sides."""
    def f(b, c):
        out = {}
        for tag, d in (("ref", b), ("cur", c)):
            for side in ("R", "L"):
                s = d.get(side) if ok(d) else None
                out[(tag, side)] = (s[xk], s[yk]) if s else (None, None)
        return out
    return f


def _angle_effect(which):
    """Tabs 1 and 2: move ONE angle to the current value and hold the other at
    the reference, so each tab shows that angle's contribution on its own.

    Plots camber CHANGE, not absolute camber. steer_camber rotates a wheel that
    starts at zero camber, so `static + change` is not the same as rotating an
    already-cambered wheel -- they differ ~5e-3 deg at 35 deg. The change form is
    exact and matches corner_state to 4e-16 deg.
    """
    def f(b, c):
        out = {}
        for tag, d in (("ref", b), ("cur", c)):
            for side in ("R", "L"):
                rs = b.get(side) if ok(b) else None
                ds = d.get(side) if ok(d) else None
                if rs is None or ds is None:
                    out[(tag, side)] = (None, None)
                    continue
                st = rs["steer"]
                kpi = ds["kpi"] if which == "kpi" else rs["kpi"]
                cas = rs["caster"] if which == "kpi" else ds["caster"]
                # a car steer of +delta is -delta about the LEFT kingpin, and the
                # result is then reported in the car frame (see corner_compute)
                sg = -1.0 if side == "L" else 1.0
                out[(tag, side)] = (st, sg * camber_change(kpi, cas, sg * st))
        return out
    return f


kpi_effect, caster_effect = _angle_effect("kpi"), _angle_effect("caster")


# =====================================================================
#  a tab of line plots
# =====================================================================
class Panel:
    """One or two axes. Each draws its quantity four times -- reference/current
    x right/left -- with the readout in the margin below the panel."""

    def __init__(self, nb, title, specs, ncols=1):
        self.specs = specs
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)

        n = len(specs)
        # readouts live UNDER each panel and the legend gets its own row, so
        # neither can ever cover the curves
        if n == 1:
            self.fig.subplots_adjust(left=0.105, right=0.97, top=0.90, bottom=0.30)
            read_at = [(0.105, 0.21)]
        else:
            self.fig.subplots_adjust(left=0.075, right=0.98, top=0.885, bottom=0.32,
                                     wspace=0.25)
            read_at = [(0.075, 0.23), (0.555, 0.23)]

        self.axes, self.lines = [], []
        for i, sp in enumerate(specs):
            ax = (self.fig.add_subplot(1, n, i + 1) if ncols > 1
                  else self.fig.add_subplot(n, 1, i + 1))
            ln = {}
            for key in SERIES:
                col, ls, lw, z = STYLE[key]
                ln[key], = ax.plot([], [], ls=ls, lw=lw, color=col, zorder=z,
                                   label=LABEL[key])
            ax.axhline(0, color="#999", lw=0.8)
            ax.axvline(0, color="#999", lw=0.8)
            ax.set_xlabel(sp["xlabel"], fontsize=9)
            ax.set_ylabel(sp["ylabel"], fontsize=9)
            ax.set_title(sp["title"], fontsize=9.5)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            self.axes.append(ax)
            self.lines.append(ln)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).update()

        self.hover, self.xy = [], []
        for i, ax in enumerate(self.axes):
            v = ax.axvline(np.nan, color="0.3", lw=0.7, zorder=9)
            mk = {k: ax.plot([], [], "o", ms=7, mfc="none", mec=STYLE[k][0],
                             mew=1.5, zorder=10)[0] for k in SERIES}
            t = self.fig.text(read_at[i][0], read_at[i][1], "", fontsize=8.5,
                              family="monospace", va="top", ha="left",
                              linespacing=1.5,
                              bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))
            self.hover.append((v, mk, t))
            self.xy.append({k: (None, None) for k in SERIES})

        self.fig.legend([self.lines[0][k] for k in SERIES],
                        [LABEL[k] for k in SERIES], ncol=4, loc="upper center",
                        bbox_to_anchor=(0.5, 0.095), fontsize=8.5, frameon=False)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)

    def update(self, base, cur):
        for i, sp in enumerate(self.specs):
            data = sp["data"](base, cur)
            self.xy[i] = data
            ax = self.axes[i]
            xs, ys = [], []
            for key in SERIES:
                x, y = data[key]
                if x is None or not len(x):
                    self.lines[i][key].set_data([], [])
                else:
                    self.lines[i][key].set_data(x, y)
                    xs.append(x)
                    ys.append(y)
            # scale to the CURVES only: relim() also sees the axhline(0)/axvline(0)
            # reference lines, which dragged every y-axis down to zero and wasted
            # 70% of the scrub panel
            if xs:
                x0 = min(np.nanmin(a) for a in xs)
                x1 = max(np.nanmax(a) for a in xs)
                y0 = min(np.nanmin(a) for a in ys)
                y1 = max(np.nanmax(a) for a in ys)
                px = 0.02 * (x1 - x0) or 1.0
                py = 0.08 * (y1 - y0) or 1.0
                ax.set_xlim(x0 - px, x1 + px)
                ax.set_ylim(y0 - py, y1 + py)
        self.canvas.draw_idle()

    def _on_move(self, event):
        if event.inaxes not in self.axes or event.xdata is None:
            return
        i = self.axes.index(event.inaxes)
        sp = self.specs[i]
        v, mk, t = self.hover[i]
        val = {}
        for key in SERIES:
            x, y = self.xy[i][key]
            if x is None or not len(x):
                mk[key].set_data([], [])
                val[key] = None
                continue
            j = int(np.argmin(np.abs(x - event.xdata)))
            mk[key].set_data([x[j]], [y[j]])
            val[key] = y[j]
        v.set_xdata([event.xdata, event.xdata])

        u = sp["unit"]
        txt = [f"{sp['xshort']:<6s}{event.xdata:+9.2f} {sp['xunit']}",
               f"{'':6s}{'ref':>9s}{'cur':>10s}{'delta':>10s}"]
        for side in ("R", "L"):
            r, c = val[("ref", side)], val[("cur", side)]
            if r is None and c is None:
                continue
            rs = f"{r:+9.3f}" if r is not None else f"{'-':>9s}"
            cs = f"{c:+10.3f}" if c is not None else f"{'-':>10s}"
            ds = (f"{c - r:+10.3f}" if (r is not None and c is not None)
                  else f"{'-':>10s}")
            txt.append(f"  {side:<4s}{rs}{cs}{ds}")
        txt.append(f"{'':6s}{u}")
        t.set_text("\n".join(txt))
        self.canvas.draw_idle()


# =====================================================================
#  tab 4: contact-patch path, BOTH wheels, driven by the rack
# =====================================================================
class PatchPanel:
    """Two panels -- LEFT wheel and RIGHT wheel -- each showing that corner's
    contact-patch path on the ground, with a shared RACK slider.

    Side by side rather than one true-scale plan view because the track is
    1663 mm while each path spans about 30 x 140 mm: at true scale both paths
    would be invisible specks. Each panel keeps aspect='equal' with
    adjustable='datalim', so the SHAPE is honest and the axes still fill the box.

    The slider is RACK TRAVEL, not steer angle. Both wheels hang off the same
    rack bar, so one rack position gives each wheel its own steer angle -- and
    the difference between them IS the Ackermann, printed under the slider.
    """

    PATH, WEDGE_N, WEDGE_P = "#1f3fbf", "#d94f4f", "#4f7fd9"

    def __init__(self, nb, title):
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        self.axL = self.fig.add_subplot(1, 2, 1)
        self.axR = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.085, right=0.985, top=0.895, bottom=0.365,
                                 wspace=0.26)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).update()

        self.read = {
            "L": self.fig.text(0.085, 0.298, "", fontsize=8, family="monospace",
                               va="top", ha="left", linespacing=1.5,
                               bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7")),
            "R": self.fig.text(0.545, 0.298, "", fontsize=8, family="monospace",
                               va="top", ha="left", linespacing=1.5,
                               bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7")),
        }
        self.sax = self.fig.add_axes([0.14, 0.055, 0.72, 0.026])
        self.slider = Slider(self.sax, "rack [mm]", RACK_LO, RACK_HI,
                             valinit=0.0, valstep=0.5)
        self.slider.on_changed(lambda _v: self._show())
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._d = None
        self._art = {}
        self._legend = None

    def update(self, base, cur):
        self._d = cur.get("axle") if ok(cur) else None
        base_ax = base.get("axle") if ok(base) else None
        self._art = {}

        for ax, key, name in ((self.axL, "L", "LEFT wheel"),
                              (self.axR, "R", "RIGHT wheel")):
            ax.clear()
            if base_ax is not None:
                b = base_ax[key]
                ax.plot(b["P"][:, 0], b["P"][:, 1], ls="--", lw=1.6, color="0.45",
                        zorder=4, label="reference path")
            if self._d is None:
                ax.set_title(f"{name} — does not solve", fontsize=9.5)
                continue
            d = self._d[key]
            P, G, rk = d["P"], d["G"], self._d["rack"]

            neg, pos = rk <= 0, rk >= 0
            ax.fill(np.concatenate([[G[0]], P[neg, 0]]),
                    np.concatenate([[G[1]], P[neg, 1]]),
                    color=self.WEDGE_N, alpha=0.16, lw=0, label="rack −")
            ax.fill(np.concatenate([[G[0]], P[pos, 0]]),
                    np.concatenate([[G[1]], P[pos, 1]]),
                    color=self.WEDGE_P, alpha=0.16, lw=0, label="rack +")
            ax.plot(P[:, 0], P[:, 1], color=self.PATH, lw=2.2, zorder=5,
                    label="patch path")
            ax.plot([G[0]], [G[1]], marker="X", ms=12, color="#c0392b", zorder=7,
                    ls="none", label="steering axis @ ground")
            ax.plot([d["P0"][0]], [d["P0"][1]], "o", ms=7, mfc="white", mec="k",
                    mew=1.5, zorder=6, ls="none", label="patch @ rack 0")

            spoke, = ax.plot([], [], ls=":", lw=1.1, color="#444", zorder=8,
                             label="axis → patch")
            dot, = ax.plot([], [], "o", ms=11, mfc="yellow", mec="k", mew=1.6,
                           zorder=10, ls="none", label="patch at the slider")
            self._art[key] = (spoke, dot)

            ax.set_xlabel("X — lateral  [mm]", fontsize=9)
            ax.set_ylabel("Y — longitudinal  [mm]   (+ = forward)", fontsize=9)
            ax.set_title(name, fontsize=10)
            # 'datalim' expands the DATA RANGE to fill the axes box; the default
            # ('box') shrinks the box instead and letterboxed these panels
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)

        if self._legend is not None:
            self._legend.remove()
            self._legend = None
        h, la = self.axR.get_legend_handles_labels()
        if h:
            self._legend = self.fig.legend(h, la, loc="upper center", ncol=len(h),
                                           fontsize=7.5, frameon=False,
                                           bbox_to_anchor=(0.5, 0.175))

        self.fig.suptitle("4 · contact-patch path on the ground — both front wheels\n"
                          "drag the RACK slider (both wheels hang off one bar) or "
                          "hover a path", fontsize=10, y=0.985)
        self._show()

    def _show(self):
        if self._d is None:
            for t in self.read.values():
                t.set_text("")
            self.canvas.draw_idle()
            return
        rk = self._d["rack"]
        i = int(np.argmin(np.abs(rk - self.slider.val)))
        for key, name in (("L", "LEFT"), ("R", "RIGHT")):
            if key not in self._art:
                self.read[key].set_text("")
                continue
            d = self._d[key]
            spoke, dot = self._art[key]
            spoke.set_data([d["G"][0], d["P"][i, 0]], [d["G"][1], d["P"][i, 1]])
            dot.set_data([d["P"][i, 0]], [d["P"][i, 1]])
            warn = "  <-- DE-CENTRING" if d["trail"][i] < 0 else ""
            self.read[key].set_text(
                f"{name}   steer {d['steer'][i]:+7.2f}°   "
                f"kingpin {d['kingpin_rot'][i]:+7.2f}°   "
                f"camber {d['camber'][i]:+6.2f}°\n"
                f"  scrub {d['scrub'][i]:+8.2f}   trail {d['trail'][i]:+8.2f}"
                f"{warn}\n"
                f"  track {d['d_track'][i]:+8.2f}   base  {d['d_base'][i]:+8.2f}"
                f"   jacking {d['jack'][i]:+7.2f}      [mm]")
        a = self._d["ack"][i]
        self.sax.set_xlabel(
            f"rack {rk[i]:+.1f} mm      steer   L {self._d['L']['steer'][i]:+.2f}°"
            f"   ·   R {self._d['R']['steer'][i]:+.2f}°"
            f"      Ackermann (L−R) {a:+.3f}°", fontsize=8.5)
        self.canvas.draw_idle()

    def _on_move(self, event):
        """Hovering a path snaps the slider to the nearest rack position."""
        if self._d is None or event.inaxes not in (self.axL, self.axR):
            return
        if event.xdata is None or event.ydata is None:
            return
        key = "L" if event.inaxes is self.axL else "R"
        P = self._d[key]["P"]
        j = int(np.argmin((P[:, 0] - event.xdata) ** 2 + (P[:, 1] - event.ydata) ** 2))
        self.slider.set_val(float(self._d["rack"][j]))    # fires _show


# =====================================================================
#  tab 3: the camber map over KPI x caster
# =====================================================================
class MapPanel:
    """Contour map of camber CHANGE at the 20 mm-rack corner steer, over the
    whole KPI x caster plane, with every corner marked.

    The surface is a property of the two ANGLES alone, so it does not depend on
    the rest of the hard points -- only the steer angle it is evaluated at does,
    and that comes from the current right-hand tie-rod geometry. Hover to read;
    click to pin.
    """

    def __init__(self, nb, title):
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).update()
        self.cb = None
        self.delta = 0.0
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def update(self, base, cur):
        ax = self.ax
        ax.clear()
        d = cur if ok(cur) else base
        if not ok(d):
            self.canvas.draw_idle()
            return
        self.delta = d["R"]["steer20"]

        kpi = np.linspace(*cfg.KPI_MAP_RANGE, 241)
        cas = np.linspace(*cfg.CASTER_MAP_RANGE, 241)
        K, C = np.meshgrid(kpi, cas)
        Z = camber_change(K, C, self.delta)

        cf = ax.contourf(K, C, Z, levels=24, cmap="RdBu_r")
        cs = ax.contour(K, C, Z, levels=12, colors="k", linewidths=0.5, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        if self.cb is None:
            self.cb = self.fig.colorbar(cf, ax=ax)
            self.cb.set_label("Δγ [deg]   (− = leans into the corner, favourable)",
                              fontsize=8.5)
        else:
            self.cb.update_normal(cf)          # the surface moves with the steer

        kx, cy = cfg.KPI_TARGET, cfg.CASTER_TARGET
        ax.add_patch(Rectangle((kx[0], cy[0]), kx[1] - kx[0], cy[1] - cy[0],
                               fill=False, ec="lime", lw=2.5, zorder=6,
                               label="performance window (PROVISIONAL)"))
        for tag, dd, mkr, ms in (("ref", base, "*", 17), ("cur", cur, "P", 13)):
            if not ok(dd):
                continue
            for side in ("R", "L"):
                s = dd[side]
                ax.plot(s["kpi"], s["caster"], mkr, ms=ms,
                        mfc=("w" if tag == "ref" else
                             STYLE[("cur", side)][0]),
                        mec="k", mew=1.4, zorder=7 + (tag == "cur"), ls="none",
                        label=f"{tag} {side}")
        if ok(base) and ok(cur):
            ax.annotate("", xy=(cur["R"]["kpi"], cur["R"]["caster"]),
                        xytext=(base["R"]["kpi"], base["R"]["caster"]), zorder=7,
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.3,
                                        alpha=0.7))
        ax.set_xlabel("KPI  [deg]", fontsize=9)
        ax.set_ylabel("caster  [deg]", fontsize=9)
        ax.set_title(f"camber change at δ = {self.delta:.2f}°  "
                     f"({cfg.RACK_TRAVEL_MM:.0f} mm rack, right corner)\n"
                     "hover to read  ·  click to pin  ·  green box is a "
                     "PROVISIONAL target, not yet derived", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)
        ax.tick_params(labelsize=8)

        self.v = ax.axvline(np.nan, color="k", lw=0.7, zorder=9)
        self.h = ax.axhline(np.nan, color="k", lw=0.7, zorder=9)
        self.dot, = ax.plot([], [], "o", ms=9, mfc="yellow", mec="k", mew=1.2,
                            zorder=10)
        self.txt = ax.text(0.015, 0.015, "", transform=ax.transAxes, va="bottom",
                           ha="left", fontsize=9, family="monospace", zorder=11,
                           bbox=dict(boxstyle="round", fc="white", ec="0.5",
                                     alpha=0.92))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_move(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if not hasattr(self, "dot"):
            return            # lazy drawing: this tab has not been rendered yet
        k, c = float(event.xdata), float(event.ydata)
        self.dot.set_data([k], [c])
        self.v.set_xdata([k, k])
        self.h.set_ydata([c, c])
        self.txt.set_text(f"KPI     = {k:6.2f} deg\n"
                          f"caster  = {c:6.2f} deg\n"
                          f"steer   = {self.delta:6.2f} deg\n"
                          f"d.gamma = {float(camber_change(k, c, self.delta)):+6.3f} deg")
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if not hasattr(self, "dot"):
            return
        k, c = float(event.xdata), float(event.ydata)
        cam = float(camber_change(k, c, self.delta))
        self.ax.plot([k], [c], "P", ms=12, mfc="magenta", mec="k", mew=1.2, zorder=12)
        self.ax.annotate(f"{cam:+.2f}°", (k, c), textcoords="offset points",
                         xytext=(8, 6), fontsize=8.5, fontweight="bold", zorder=12)
        print(f"PINNED   KPI={k:6.2f}   caster={c:6.2f}   ->   dgamma={cam:+.3f} deg")
        self.canvas.draw_idle()


# =====================================================================
#  the roll-centre-migration tab (an axle property, not per-corner)
# =====================================================================
class RollCentrePanel:
    """Roll-centre migration under BODY ROLL -5..+5 deg.

    Three panels tell the whole story:
        a  RC HEIGHT vs roll      -- how far the RC drops/climbs as the car rolls
        b  RC LATERAL X vs roll   -- the sideways migration heave never shows
        c  the RC PATH in the front view (X vs height), roll as a colour sweep
    Reference (grey) and current (crimson) on every panel; hover a/b to read.
    """

    def __init__(self, nb, title):
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        gs = self.fig.add_gridspec(2, 2)
        self.axH = self.fig.add_subplot(gs[0, 0])
        self.axX = self.fig.add_subplot(gs[1, 0])
        self.axP = self.fig.add_subplot(gs[:, 1])
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.20,
                                 hspace=0.45, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).update()

        self.read = self.fig.text(0.09, 0.11, "", fontsize=8.5, family="monospace",
                                  va="top", ha="left", linespacing=1.5,
                                  bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._b = self._c = None

    def update(self, base, cur):
        self._b = base.get("roll") if ok(base) else None
        self._c = cur.get("roll") if ok(cur) else None
        for ax in (self.axH, self.axX, self.axP):
            ax.clear()

        for tag, d, col, ls in (("ref", self._b, "0.45", "--"),
                                ("cur", self._c, "crimson", "-")):
            if d is None:
                continue
            r = d["roll"]
            self.axH.plot(r, d["rc_z"], ls=ls, lw=2.2, color=col, label=tag)
            self.axX.plot(r, d["rc_x"], ls=ls, lw=2.2, color=col, label=tag)
            self.axP.plot(d["rc_x"], d["rc_z"], ls=ls, lw=2.0, color=col,
                          zorder=3, label=tag)
        if self._c is not None:
            i0 = int(np.argmin(np.abs(self._c["roll"])))
            sc = self.axP.scatter(self._c["rc_x"], self._c["rc_z"],
                                  c=self._c["roll"], cmap="coolwarm", s=22,
                                  zorder=5)
            self.axP.plot([self._c["rc_x"][i0]], [self._c["rc_z"][i0]], "o",
                          ms=9, mfc="yellow", mec="k", mew=1.4, zorder=6)
            if self.cb_ok():
                self.cb = self.fig.colorbar(sc, ax=self.axP, fraction=0.05)
                self.cb.set_label("body roll  [deg]", fontsize=8)

        self.axH.set_title("11a · RC HEIGHT vs body roll", fontsize=9.5)
        self.axH.set_ylabel("RC height  [mm]", fontsize=9)
        self.axH.set_xlabel("body roll  φ  [deg]", fontsize=9)
        self.axX.set_title("11b · RC LATERAL migration vs body roll\n"
                           "how far the roll centre slides SIDEWAYS off the\n"
                           "car centreline as the body leans", fontsize=9)
        self.axX.set_ylabel("RC lateral X  [mm]", fontsize=9)
        self.axX.set_xlabel("body roll  φ  [deg]", fontsize=9)
        self.axP.set_title("11c · RC PATH in the front view\n"
                           "(yellow = static 0°, colour = roll angle)", fontsize=9.5)
        self.axP.set_xlabel("RC lateral X  [mm]", fontsize=9)
        self.axP.set_ylabel("RC height  [mm]", fontsize=9)
        for ax in (self.axH, self.axX, self.axP):
            ax.grid(alpha=0.3)
            ax.axhline(0, color="#999", lw=0.8)
            ax.tick_params(labelsize=8)
            if ax.get_legend_handles_labels()[0]:      # avoid an empty-legend warning
                ax.legend(fontsize=8, loc="best")
        self.axX.axvline(0, color="#999", lw=0.8)
        self.axH.axvline(0, color="#999", lw=0.8)
        self.axP.axvline(0, color="#999", lw=0.8)

        # crosshairs (on the two roll-axis panels) and per-series markers on ALL
        # THREE panels, so hovering any one moves the others in lock-step
        self.vH = self.axH.axvline(np.nan, color="0.3", lw=0.7, zorder=8)
        self.vX = self.axX.axvline(np.nan, color="0.3", lw=0.7, zorder=8)

        def mk(ax):
            return {t: ax.plot([], [], "o", ms=8, mfc="none",
                               mec=("0.45" if t == "ref" else "crimson"),
                               mew=1.7, zorder=12)[0] for t in ("ref", "cur")}
        self.mH, self.mX, self.mP = mk(self.axH), mk(self.axX), mk(self.axP)

        if self._c is None and self._b is None:
            self.axH.text(0.5, 0.5, "open this tab to compute the roll sweep\n"
                          "(≈1 s)  —  it updates on the next Apply",
                          transform=self.axH.transAxes, ha="center", va="center",
                          fontsize=9, color="#888")
            self.read.set_text("")
        else:
            d = self._c if self._c is not None else self._b
            self._pick(int(np.argmin(np.abs(d["roll"]))))   # populate at 0 deg
        self.canvas.draw_idle()

    def cb_ok(self):
        # one colourbar only; matplotlib re-add would stack them
        if getattr(self, "cb", None) is not None:
            try:
                self.cb.remove()
            except Exception:                          # noqa: BLE001
                pass
        return True

    def _pick(self, j):
        """Show the roll index j on all three panels and in the readout. Both
        geometries share the same roll grid, so one index serves both."""
        d0 = self._c if self._c is not None else self._b
        phi = d0["roll"][j]
        self.vH.set_xdata([phi, phi])
        self.vX.set_xdata([phi, phi])
        lines = [f"roll     {phi:+7.2f} deg",
                 f"{'':8s}{'height':>9s}{'lateral':>10s}"]
        for tag, d in (("ref", self._b), ("cur", self._c)):
            if d is None:
                for m in (self.mH, self.mX, self.mP):
                    m[tag].set_data([], [])
                continue
            self.mH[tag].set_data([d["roll"][j]], [d["rc_z"][j]])
            self.mX[tag].set_data([d["roll"][j]], [d["rc_x"][j]])
            self.mP[tag].set_data([d["rc_x"][j]], [d["rc_z"][j]])
            lines.append(f"  {tag:<5s}{d['rc_z'][j]:+9.1f}{d['rc_x'][j]:+10.1f} mm")
        if self._b is not None and self._c is not None:
            lines.append(f"  {'delta':<5s}{self._c['rc_z'][j]-self._b['rc_z'][j]:+9.1f}"
                         f"{self._c['rc_x'][j]-self._b['rc_x'][j]:+10.1f} mm")
        self.read.set_text("\n".join(lines))
        self.canvas.draw_idle()

    def _on_move(self, event):
        d0 = self._c if self._c is not None else self._b
        if d0 is None or event.xdata is None or event.ydata is None:
            return
        if not hasattr(self, "vH"):
            return            # not rendered yet
        if event.inaxes in (self.axH, self.axX):        # x-axis is roll
            j = int(np.argmin(np.abs(d0["roll"] - event.xdata)))
        elif event.inaxes is self.axP:                  # nearest point in the X-Z plane
            sx = max(np.ptp(self.axP.get_xlim()), 1e-9)
            sz = max(np.ptp(self.axP.get_ylim()), 1e-9)
            j = int(np.argmin(((d0["rc_x"] - event.xdata) / sx) ** 2
                              + ((d0["rc_z"] - event.ydata) / sz) ** 2))
        else:
            return
        self._pick(j)


# =====================================================================
#  the tab list, in the order asked for
# =====================================================================
class AckermannPanel:
    """Ackermann / Jeantaud steering geometry, off the SAME rack sweep tab 4 uses.

    Three views:
        a  inner vs outer steer, against the Jeantaud ideal and the parallel line
        b  %Ackermann vs steer -- the industry number, in the usual chart format
        c  toe-out on turns, the raw asymmetry the tyres actually feel

    Nothing here is recomputed: `compute()` already solved both wheels off one
    rack bar, so this panel only re-reads those headings. The Jeantaud target
    needs the KINGPIN GROUND track (steering-axis separation at the ground), not
    the wheel track -- they differ by twice the scrub radius, ~258 mm on this car.

    HIGHLIGHT tells the hard-point table which rows this tab actually responds
    to, so you can tune Ackermann without hunting through points that do nothing.
    The split is MEASURED, not assumed -- each point was nudged 5 mm on every axis
    and the change in %Ackermann recorded (validate_explorer re-measures it and
    fails if this list drifts):

        Lower ball joint  LBJ     5.94 %    strong  (anchors the kingpin axis,
        Tie rod - outer           5.77 %    strong   so it moves BOTH the pivot
        Tie rod - inner (rack)    4.32 %    strong   and the ground track t)
        Upper ball joint  UBJ     0.97 %    weak
        Wheel centre              0.26 %    weak    (only via the wheelbase)

    Everything else -- all four inner wishbone points, both damper mounts -- is
    EXACTLY zero: the rack sweep never touches them.
    """

    HIGHLIGHT = {
        "Lower wishbone - outer (LBJ)": "strong",
        "Tie rod - outer": "strong",
        "Tie rod - inner": "strong",
        "Upper wishbone - outer (UBJ)": "weak",
        "Wheel centre": "weak",
    }

    def __init__(self, nb, title):
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        gs = self.fig.add_gridspec(2, 2)
        self.axA = self.fig.add_subplot(gs[0, 0])
        self.axC = self.fig.add_subplot(gs[1, 0])
        self.axB = self.fig.add_subplot(gs[:, 1])
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.20,
                                 hspace=0.45, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, frame).update()
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self._cur = None
        # per-axes hover furniture, rebuilt by update() because ax.clear() kills it
        self._mark = {}      # ax -> [markers on that axes' curves]
        self._vline = {}     # ax -> the vertical guide
        self._note = {}      # ax -> that axes' OWN readout box

    @staticmethod
    def _pull(d):
        """(outer, inner, ideal, pct, pct0) for one geometry, or None."""
        if not ok(d) or d.get("axle") is None or d.get("ack_t") is None:
            return None
        a = d["axle"]
        sR = np.asarray(a["R"]["steer"], float)
        sL = np.asarray(a["L"]["steer"], float)
        rack = np.asarray(a["rack"], float)
        keep = rack >= 0                       # one direction; symmetric car
        sR, sL = sR[keep], sL[keep]
        # inner/outer by TURN DIRECTION, not by which steers more -- see the note
        # in ackermann.sweep(): the magnitude rule silently mis-scores any
        # anti-Ackermann geometry as positive.
        right_turn = (sR + sL) >= 0.0
        inner = np.where(right_turn, sR, sL)
        outer = np.where(right_turn, sL, sR)
        t = float(d["ack_t"])
        wb = abs(float(d["R"]["wheelbase"][len(d["R"]["wheelbase"]) // 2]))
        ideal = ackermann.ideal_inner(outer, t, wb)
        pct = ackermann.ackermann_percent(inner, outer, t, wb)
        pct0 = ackermann.zero_steer_limit(inner, outer, t, wb)
        o = np.abs(outer)
        srt = np.argsort(o)
        return (o[srt], np.abs(inner)[srt], np.abs(ideal)[srt], pct[srt], pct0)

    def update(self, base, cur):
        b, c = self._pull(base), self._pull(cur)
        self._cur = c
        for ax in (self.axA, self.axB, self.axC):
            ax.clear()

        for tag, d, col, ls in (("ref", b, "0.45", "--"), ("cur", c, "crimson", "-")):
            if d is None:
                continue
            o, i, idl, pct, _ = d
            self.axA.plot(o, i, ls=ls, lw=2.2, color=col, label=tag)
            self.axB.plot(o, pct, ls=ls, lw=2.2, color=col, label=tag)
            self.axC.plot(o, i - o, ls=ls, lw=2.2, color=col, label=tag)

        ref = c or b
        if ref is not None:
            o, _i, idl, _p, pct0 = ref
            self.axA.plot(o, idl, lw=1.6, ls="--", color="#0353a4",
                          label="Jeantaud ideal")
            self.axA.plot(o, o, lw=1.2, ls=":", color="#555", label="parallel")
            self.axC.plot(o, idl - o, lw=1.6, ls="--", color="#0353a4",
                          label="Jeantaud ideal")
            self.axB.set_title(f"13b · %ACKERMANN vs steer\n"
                               f"zero-steer limit {pct0:.0f} %", fontsize=9.5)
        else:
            self.axB.set_title("13b · %ACKERMANN vs steer", fontsize=9.5)

        self.axB.axhline(100.0, lw=1.3, ls="--", color="#0353a4")
        self.axB.axhline(0.0, lw=1.1, ls=":", color="#555")
        self.axA.set_title("13a · INNER vs OUTER steer angle", fontsize=9.5)
        self.axA.set_xlabel("outer wheel steer  [deg]", fontsize=9)
        self.axA.set_ylabel("inner wheel steer  [deg]", fontsize=9)
        self.axB.set_xlabel("outer wheel steer  [deg]", fontsize=9)
        self.axB.set_ylabel("Ackermann  [%]     100 = true, 0 = parallel", fontsize=9)
        self.axC.set_title("13c · TOE-OUT ON TURNS   inner − outer", fontsize=9.5)
        self.axC.set_xlabel("outer wheel steer  [deg]", fontsize=9)
        self.axC.set_ylabel("inner − outer  [deg]", fontsize=9)
        for ax in (self.axA, self.axB, self.axC):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7.5, loc="upper left")

        # ------------------------------------------------------ hover furniture
        # One readout PER PLOT, parked in each axes' own empty lower-right corner,
        # so a number is always next to the curve it belongs to. All three share
        # the same x (outer steer), so hovering any one drives all three markers
        # together -- read one steer angle across all three views at once.
        self._mark, self._vline, self._note = {}, {}, {}
        SERIES = {self.axA: [("cur", "crimson"), ("ideal", "#0353a4")],
                  self.axB: [("cur", "crimson")],
                  self.axC: [("cur", "crimson"), ("ideal", "#0353a4")]}
        for ax in (self.axA, self.axB, self.axC):
            self._vline[ax] = ax.axvline(np.nan, lw=0.9, color="#444", alpha=0.55,
                                         zorder=8)
            self._mark[ax] = [
                ax.plot([], [], "o", ms=7.5, mfc=col, mec="k", mew=1.2,
                        zorder=10, label="_nolegend_")[0]
                for _tag, col in SERIES[ax]]
            # 13b's zero-% reference line runs along the bottom of its axes, so
            # its box is lifted clear of it; the other two have empty corners.
            y = 0.075 if ax is self.axB else 0.035
            self._note[ax] = ax.text(
                0.985, y, "", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.2, family="monospace", linespacing=1.45, zorder=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#c9b458",
                          alpha=0.95))
        self._hide_hover()
        self.canvas.draw_idle()

    def _hide_hover(self):
        for ax in (self.axA, self.axB, self.axC):
            if ax in self._vline:
                self._vline[ax].set_xdata([np.nan, np.nan])
                for m in self._mark[ax]:
                    m.set_data([], [])
                self._note[ax].set_text("")

    def _on_leave(self, _ev):
        self._hide_hover()
        self.canvas.draw_idle()

    def _on_move(self, ev):
        if self._cur is None or not self._note:
            return
        if ev.inaxes not in (self.axA, self.axB, self.axC) or ev.xdata is None:
            return
        o, i, idl, pct, _pct0 = self._cur
        k = int(np.argmin(np.abs(o - float(ev.xdata))))
        x, p = o[k], pct[k]
        pct_txt = "n/a (0/0 at centre)" if not np.isfinite(p) else f"{p:.1f} %"

        for ax in (self.axA, self.axB, self.axC):
            self._vline[ax].set_xdata([x, x])
        self._mark[self.axA][0].set_data([x], [i[k]])
        self._mark[self.axA][1].set_data([x], [idl[k]])
        self._mark[self.axB][0].set_data([x], [p if np.isfinite(p) else np.nan])
        self._mark[self.axC][0].set_data([x], [i[k] - x])
        self._mark[self.axC][1].set_data([x], [idl[k] - x])

        self._note[self.axA].set_text(f"outer  {x:6.2f}°\n"
                                      f"inner  {i[k]:6.2f}°\n"
                                      f"ideal  {idl[k]:6.2f}°")
        self._note[self.axB].set_text(f"outer  {x:6.2f}°\n"
                                      f"Ackermann  {pct_txt}")
        self._note[self.axC].set_text(f"outer     {x:6.2f}°\n"
                                      f"toe-out  {i[k] - x:+6.2f}°\n"
                                      f"ideal    {idl[k] - x:+6.2f}°")
        self.canvas.draw_idle()


def build_panels(nb):
    P = []
    P.append(Panel(nb, "1 · Camber–KPI", [dict(
        title="1 · camber vs steer — KPI EFFECT ONLY  (caster held at the reference)",
        xlabel="steer angle at the road wheel   δ  [deg]",
        ylabel="camber change  Δγ, CAR FRAME  [deg]",
        data=kpi_effect, unit="deg", xshort="steer", xunit="deg")]))

    P.append(Panel(nb, "2 · Camber–caster", [dict(
        title="2 · camber vs steer — CASTER EFFECT ONLY  (KPI held at the reference)",
        xlabel="steer angle at the road wheel   δ  [deg]",
        ylabel="camber change  Δγ, CAR FRAME  [deg]",
        data=caster_effect, unit="deg", xshort="steer", xunit="deg")]))

    P.append(MapPanel(nb, "3 · Camber map"))
    P.append(PatchPanel(nb, "4 · Patch path"))

    P.append(Panel(nb, "5 · Scrub & trail", [
        dict(title="5a · scrub radius vs steer",
             xlabel="steer  δ  [deg]",
             ylabel="scrub  [mm]   (+ = patch outboard of the axis)",
             data=keyed("steer", "scrub_steer"), unit="mm",
             xshort="steer", xunit="deg"),
        dict(title="5b · mechanical trail vs steer   (< 0 de-centres)",
             xlabel="steer  δ  [deg]",
             ylabel="trail  [mm]   (+ = stabilising)",
             data=keyed("steer", "trail_steer"), unit="mm",
             xshort="steer", xunit="deg"),
    ], ncols=2))

    P.append(Panel(nb, "6 · Wheelbase & track", [
        dict(title="6a · wheelbase (this corner) vs steer",
             xlabel="steer  δ  [deg]", ylabel="wheelbase  [mm]",
             data=keyed("steer", "wheelbase"), unit="mm",
             xshort="steer", xunit="deg"),
        dict(title="6b · half-track (this corner) vs steer",
             xlabel="steer  δ  [deg]", ylabel="half-track  [mm]",
             data=keyed("steer", "halftrack"), unit="mm",
             xshort="steer", xunit="deg"),
    ], ncols=2))

    P.append(Panel(nb, "7 · Heave & bump steer", [
        dict(title="7a · camber gain in HEAVE  (both wheels together)",
             xlabel="wheel travel  [mm]   (+ = compression)",
             ylabel="camber, CAR FRAME  [deg]   (+ = top leans toward +X)",
             data=keyed("bump", "cam_bump_car"), unit="deg",
             xshort="travel", xunit="mm"),
        dict(title="7b · BUMP STEER — flat is good, the slope IS the bump steer",
             xlabel="wheel travel  [mm]   (+ = compression)",
             ylabel="toe  [deg]   (+ = toe-in)",
             data=keyed("bump", "toe_bump"), unit="deg",
             xshort="travel", xunit="mm"),
    ], ncols=2))

    P.append(Panel(nb, "8 · Body roll", [dict(
        title="8 · camber SEEN BY THE ROAD in roll  =  linkage gain + body tilt\n"
              "flat means the linkage fully recovers the roll",
        xlabel="body roll  φ  [deg]", ylabel="camber to the ROAD, CAR FRAME  [deg]",
        data=keyed("phi", "cam_road_car"), unit="deg", xshort="roll",
        xunit="deg")]))

    P.append(Panel(nb, "9 · Roll centre", [dict(
        title="9 · roll-centre height vs wheel travel  (linkage construction)",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel="roll-centre height  [mm]   (0 = ground)",
        data=keyed("bump", "rc"), unit="mm", xshort="travel", xunit="mm")]))

    P.append(Panel(nb, "10 · KPI/caster in bump", [
        dict(title="10a · KPI vs wheel travel", xlabel="wheel travel  [mm]",
             ylabel="KPI  [deg]", data=keyed("bump", "kpi_bump"), unit="deg",
             xshort="travel", xunit="mm"),
        dict(title="10b · caster vs wheel travel", xlabel="wheel travel  [mm]",
             ylabel="caster  [deg]", data=keyed("bump", "caster_bump"), unit="deg",
             xshort="travel", xunit="mm"),
    ], ncols=2))

    P.append(viewer.Geometry2D(nb, "★ Geometry 2D"))
    P.append(viewer.Geometry3D(nb, "★ Geometry 3D"))
    P.append(RollCentrePanel(nb, "11 · RC in roll"))

    # Motion ratio = spring travel / wheel travel = -d(damper)/d(bump), which is
    # exactly the lecture's R_M = x_s / x_w (ENGR7008 wk4). Shown as a CURVE over
    # the stroke, so its rise/fall through travel is visible, not just one number.
    P.append(Panel(nb, "12 · Motion ratio", [dict(
        title=f"12 · MOTION RATIO vs wheel travel  ({spr.ratio_text()})\n"
              f"{spr.formula()}",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel=spr.label(),
        data=keyed("bump", "mr_disp"), unit="-",
        xshort="travel", xunit="mm")]))

    # Ackermann is an AXLE property (it only exists as a left/right difference),
    # so it gets its own panel rather than a per-corner curve. Front only -- the
    # rear does not steer.
    P.append(AckermannPanel(nb, "13 · Ackermann"))
    return P


# =====================================================================
#  one side's editable coordinate table
# =====================================================================
# =====================================================================
#  REAR AXLE  -  same machinery, no steering, a fixed toe link
# =====================================================================
import rear_geometry as rgeo                                          # noqa: E402
import arb                                                            # noqa: E402
import roll_axis as ra                                                # noqa: E402
import anti_geometry as ag                                            # noqa: E402
import car_outline                                                    # noqa: E402
import load_transfer as ltr                                           # noqa: E402
import benchmarks as bmk
import springs as spr                                                 # noqa: E402
import mech_balance as mbal                                           # noqa: E402
import viewer                                                         # noqa: E402

REAR_POINT_ROWS = [
    ("Upper wishbone - inner front", "UPPER arm   inner front", True),
    ("Upper wishbone - inner rear",  "UPPER arm   inner rear",  False),
    ("Upper wishbone - outer (UBJ)", "UPPER ball joint  UBJ",   False),
    ("Lower wishbone - inner front", "LOWER arm   inner front", True),
    ("Lower wishbone - inner rear",  "LOWER arm   inner rear",  False),
    ("Lower wishbone - outer (LBJ)", "LOWER ball joint  LBJ",   False),
    ("Toe link - inner",             "TOE LINK  inner",         True),
    ("Toe link - outer",             "TOE LINK  outer",         False),
    ("Wheel centre",                 "WHEEL CENTRE",            True),
    ("Lower Shock mount",            "DAMPER   lower",          True),
    ("Upper Shock mount",            "DAMPER   upper",          False),
    ("Drive shaft - Inner",          "DRIVE SHAFT  inner",      True),
    ("Drive shaft - outer",          "DRIVE SHAFT  outer",      False),
]


def rear_compute(pts_R, cam_R, pts_L, cam_L, symmetric=False):
    """Both rear corners. The rear has no rack, so there is no patch-path sweep
    and no Ackermann; everything else reuses the SAME corner_compute as the front
    by translating the toe link into the solver's tie-rod slots."""
    try:
        R = corner_compute(rgeo.solver_points(pts_R), cam_R, False)
    except Exception as exc:                                   # noqa: BLE001
        return dict(error=f"REAR RIGHT: {exc}")
    if R.get("error"):
        return dict(error="REAR RIGHT: " + R["error"])
    if symmetric:
        L = dict(R)
        L["is_left"] = True
    else:
        try:
            L = corner_compute(rgeo.solver_points(pts_L), cam_L, True)
        except Exception as exc:                               # noqa: BLE001
            return dict(error=f"REAR LEFT: {exc}")
        if L.get("error"):
            return dict(error="REAR LEFT: " + L["error"])
    # keep the REAL (toe-link) points so the table and CSV round-trip unchanged
    R["pts"], L["pts"] = pts_R, pts_L
    return dict(error=None, R=R, L=L, axle=None, roll=None,
                track=abs(float(pts_R["Wheel centre"][0]))
                + abs(float(pts_L["Wheel centre"][0])),
                roll_args=(rgeo.solver_points(pts_R), cam_R,
                           rgeo.solver_points(pts_L), cam_L))


def build_rear_panels(nb):
    """The fast, live-updating rear curves. The heavy 2-D parameter maps
    (recovery map, toe-link map) stay as standalone scripts -- each needs ~1700
    linkage solves and would stall every hard-point edit."""
    P = []
    P.append(Panel(nb, "1 · Camber gain", [dict(
        title="camber vs WHEEL TRAVEL — the gain that fights body roll",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel="camber, CAR FRAME  [deg]", data=keyed("bump", "cam_bump_car"),
        unit="deg", xshort="travel", xunit="mm")]))
    P.append(Panel(nb, "2 · Bump steer", [dict(
        title="toe vs WHEEL TRAVEL — the slope IS the bump steer\n"
              "(rear: this is ROLL STEER, the stability parameter)",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel="toe  [deg]   (+ = TOE-IN, wheel points at the car)",
        data=keyed("bump", "toe_bump"),
        unit="deg", xshort="travel", xunit="mm")]))
    P.append(Panel(nb, "3 · Camber vs roll", [dict(
        title="camber SEEN BY THE ROAD in roll = linkage gain + body tilt\n"
              "flat means the linkage fully recovers the roll",
        xlabel="body roll  φ  [deg]", ylabel="camber to the ROAD, CAR FRAME  [deg]",
        data=keyed("phi", "cam_road_car"), unit="deg", xshort="roll",
        xunit="deg")]))
    P.append(Panel(nb, "4 · Roll centre", [dict(
        title="roll-centre height vs wheel travel (linkage construction)",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel="roll-centre height  [mm]   (0 = ground)",
        data=keyed("bump", "rc"), unit="mm", xshort="travel", xunit="mm")]))
    P.append(RollCentrePanel(nb, "5 · RC in roll"))
    P.append(Panel(nb, "6 · KPI/caster in bump", [
        dict(title="KPI vs wheel travel", xlabel="wheel travel  [mm]",
             ylabel="KPI  [deg]", data=keyed("bump", "kpi_bump"), unit="deg",
             xshort="travel", xunit="mm"),
        dict(title="caster vs wheel travel", xlabel="wheel travel  [mm]",
             ylabel="caster  [deg]", data=keyed("bump", "caster_bump"),
             unit="deg", xshort="travel", xunit="mm")], ncols=2))
    P.append(Panel(nb, "7 · Motion ratio", [dict(
        title=f"MOTION RATIO vs wheel travel  ({spr.ratio_text()})\n"
              f"{spr.formula()}",
        xlabel="wheel travel  [mm]   (+ = compression)",
        ylabel=spr.label(),
        data=keyed("bump", "mr_disp"), unit="-",
        xshort="travel", xunit="mm")]))
    P.append(viewer.Geometry2D(nb, "★ Geometry 2D"))
    P.append(viewer.Geometry3D(nb, "★ Geometry 3D"))
    return P


class SideTable(ttk.Frame):
    """The AVL-VSM-style grid: Name | X | Y | Z | Unit, alternating row shading.

    tk.Entry rather than ttk.Entry because only the tk widget lets us set a
    per-row background, and 'readonly' state is what locks the mirrored side.
    """
    ROW_A, ROW_B, LOCK = "#ffffff", "#eef2f6", "#e9e9e9"
    # Highlight tints for "these are the points the open tab actually responds
    # to". Two strengths, because a flat highlight would put a 0.26 %/5 mm point
    # next to a 5.9 %/5 mm one and send you tuning the wrong screw. The muted
    # LOCK_* variants keep the readonly (mirrored-side) cue visible underneath.
    HL_STRONG, HL_WEAK = "#ffd97d", "#fff3d0"
    HL_STRONG_LOCK, HL_WEAK_LOCK = "#e5c477", "#e8e0c9"
    HL_FG_STRONG, HL_FG_WEAK = "#6b4200", "#7a6220"

    def __init__(self, master, rows=None):
        super().__init__(master, padding=(4, 4))
        POINT_ROWS = rows if rows is not None else globals()["POINT_ROWS"]
        hdr = ("Name", "X", "Y", "Z", "Unit")
        wid = (23, 9, 9, 9, 4)
        for j, (h, w) in enumerate(zip(hdr, wid)):
            tk.Label(self, text=h, font=("Segoe UI", 8, "bold"), width=w,
                     anchor="w" if j == 0 else "center", relief="groove",
                     bg="#dcdcdc").grid(row=0, column=j, sticky="ew")

        self.cells, self.rowart, r = {}, {}, 1
        for idx, (name, label, _brk) in enumerate(POINT_ROWS):
            bg = self.ROW_A if idx % 2 == 0 else self.ROW_B
            lab = tk.Label(self, text=label, font=("Consolas", 8), width=wid[0],
                           anchor="w", bg=bg)
            lab.grid(row=r, column=0, sticky="ew")
            vs = []
            for j in range(3):
                v = tk.StringVar()
                e = tk.Entry(self, textvariable=v, width=wid[j + 1],
                             font=("Consolas", 8), justify="right", bg=bg,
                             relief="solid", borderwidth=1,
                             readonlybackground=self.LOCK)
                e.grid(row=r, column=j + 1, sticky="ew")
                vs.append((v, e))
            unit = tk.Label(self, text="mm", font=("Consolas", 8), width=wid[4],
                            anchor="center", bg=bg)
            unit.grid(row=r, column=4, sticky="ew")
            self.cells[name] = vs
            # keep the row's own widgets and its NORMAL colour so a highlight can
            # be put on and taken off again without guessing the stripe back
            self.rowart[name] = (lab, unit, bg)
            r += 1

        tk.Label(self, text="Static camber", font=("Consolas", 8, "bold"),
                 width=wid[0], anchor="w", bg="#dcdcdc").grid(row=r, column=0,
                                                              sticky="ew")
        self.camber = tk.StringVar()
        self.camber_entry = tk.Entry(self, textvariable=self.camber, width=wid[1],
                                     font=("Consolas", 8), justify="right",
                                     relief="solid", borderwidth=1,
                                     readonlybackground=self.LOCK)
        self.camber_entry.grid(row=r, column=1, sticky="ew")
        tk.Label(self, text="deg", font=("Consolas", 8), width=wid[4],
                 anchor="center", bg="#dcdcdc").grid(row=r, column=4, sticky="ew")

    def bind_return(self, fn):
        for vs in self.cells.values():
            for _v, e in vs:
                e.bind("<Return>", fn)
        self.camber_entry.bind("<Return>", fn)

    def set_locked(self, locked):
        st = "readonly" if locked else "normal"
        for vs in self.cells.values():
            for _v, e in vs:
                e.config(state=st)
        self.camber_entry.config(state=st)

    def set_highlight(self, levels):
        """Tint the rows the open plot tab actually responds to.

        `levels` maps a point name to "strong" or "weak"; anything absent goes
        back to its normal stripe. Idempotent, so it can be called on every tab
        change without tracking what was set last time.
        """
        levels = levels or {}
        for name, (lab, unit, base_bg) in self.rowart.items():
            lvl = levels.get(name)
            if lvl == "strong":
                bg, lock, fg = self.HL_STRONG, self.HL_STRONG_LOCK, self.HL_FG_STRONG
            elif lvl == "weak":
                bg, lock, fg = self.HL_WEAK, self.HL_WEAK_LOCK, self.HL_FG_WEAK
            else:
                bg, lock, fg = base_bg, self.LOCK, "black"
            lab.config(bg=bg, fg=fg)
            unit.config(bg=bg, fg=fg)
            for _v, e in self.cells[name]:
                e.config(bg=bg, readonlybackground=lock)

    def write(self, pts, camber):
        for name, vs in self.cells.items():
            p = pts.get(name)
            for j, (v, e) in enumerate(vs):
                was = e.cget("state")
                e.config(state="normal")
                v.set(f"{float(p[j]):.3f}" if p is not None else "")
                e.config(state=was)
        was = self.camber_entry.cget("state")
        self.camber_entry.config(state="normal")
        self.camber.set(f"{camber:.3f}")
        self.camber_entry.config(state=was)

    def read(self):
        pts, bad = {}, []
        for name, vs in self.cells.items():
            try:
                pts[name] = np.array([float(v.get()) for v, _e in vs])
            except ValueError:
                bad.append(name)
        try:
            cam = float(self.camber.get())
        except ValueError:
            bad.append("static camber")
            cam = 0.0
        return pts, cam, bad


# =====================================================================
#  the window
# =====================================================================
class AxlePage(ttk.Frame):
    """One axle: Left/Right hard-point tables, a derived readout, and a notebook
    of plots. Front and rear differ only by the `spec` handed in, so both axles
    run the same validated machinery and the same UI."""

    def __init__(self, master, spec, on_change=None):
        super().__init__(master)
        self.spec = spec
        self.on_change = on_change
        self.rows = spec["rows"]
        self.compute = spec["compute"]

        self.stock_R = spec["load"]()
        self.stock_L = mirror_pts(self.stock_R)
        self.stock_camber = float(spec["camber"])

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, padding=(8, 6))
        left.pack(side="left", fill="y")
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # ------------------------------------------------ symmetry + side tabs
        top = ttk.Frame(left)
        top.pack(fill="x")
        self.sym = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Is Symmetric (RH = LH)", variable=self.sym,
                        command=self.on_sym).pack(side="left")
        self.side_lbl = ttk.Label(top, text="Right-hand side coordinates",
                                  font=("Segoe UI", 8, "bold"), foreground="#333")
        self.side_lbl.pack(side="right")

        self.side_nb = ttk.Notebook(left)
        self.side_nb.pack(fill="x", pady=(4, 0))
        self.tab = {"L": SideTable(self.side_nb, self.rows),
                    "R": SideTable(self.side_nb, self.rows)}
        self.side_nb.add(self.tab["L"], text="  Left  ")
        self.side_nb.add(self.tab["R"], text="  Right  ")
        self.side_nb.select(self.tab["R"])
        self.side_nb.bind("<<NotebookTabChanged>>", self.on_tab)
        for t in self.tab.values():
            t.bind_return(lambda _e: self.apply())

        # ------------------------------------------------ buttons
        b1 = ttk.Frame(left)
        b1.pack(fill="x", pady=(8, 2))
        ttk.Button(b1, text="Apply", command=self.apply).pack(side="left")
        ttk.Button(b1, text="Reset to stock", command=self.reset).pack(side="left", padx=3)
        ttk.Button(b1, text="Load CSV", command=self.load_csv).pack(side="left")
        ttk.Button(b1, text="Save CSV", command=self.save_csv).pack(side="left", padx=3)
        b2 = ttk.Frame(left)
        b2.pack(fill="x", pady=(0, 4))
        ttk.Button(b2, text="Use CURRENT as the grey reference",
                   command=self.freeze).pack(side="left")
        ttk.Button(b2, text="back to stock", command=self.unfreeze).pack(side="left", padx=3)

        self.out = tk.Text(left, width=54, height=22, font=("Consolas", 8),
                           relief="flat", background="#f7f7f7", wrap="none")
        self.out.pack(fill="both", expand=True)

        # ------------------------------------------------ plots
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.plot_nb = nb
        self.panels = spec["panels"](nb)
        self.roll_tab = next((i for i, p in enumerate(self.panels)
                              if isinstance(p, RollCentrePanel)), -1)
        self._drawn = set()          # panel indices that reflect the current data
        nb.bind("<<NotebookTabChanged>>", self.on_plot_tab)

        self.tab["R"].write(self.stock_R, self.stock_camber)
        self.tab["L"].write(self.stock_L, self.stock_camber)
        self.on_sym()
        self.stock = self.compute(self.stock_R, self.stock_camber,
                                  self.stock_L, self.stock_camber, symmetric=True)
        self.base, self.base_tag = self.stock, "STOCK (CAD)"
        self.cur = self.stock
        self.redraw()

    # ---------------------------------------------------------- symmetry
    def on_sym(self):
        """RIGHT is master -- it is the corner the CAD was measured from."""
        locked = self.sym.get()
        self.tab["L"].set_locked(locked)
        if locked:
            self.mirror_right_to_left()
        self.on_tab()

    def on_tab(self, _e=None):
        try:
            cur = self.side_nb.index(self.side_nb.select())
        except tk.TclError:
            return
        side = "Left" if cur == 0 else "Right"
        note = "  (mirrored, locked)" if (side == "Left" and self.sym.get()) else ""
        self.side_lbl.config(text=f"{side}-hand side coordinates{note}")

    def mirror_right_to_left(self):
        pts, cam, bad = self.tab["R"].read()
        if bad:
            return
        self.tab["L"].write(mirror_pts(pts), cam)

    # ---------------------------------------------------------- actions
    def apply(self):
        if self.sym.get():
            self.mirror_right_to_left()
        pR, cR, badR = self.tab["R"].read()
        pL, cL, badL = self.tab["L"].read()
        bad = [f"R:{b}" for b in badR] + [f"L:{b}" for b in badL]
        self.cur = (dict(error="not a number -- " + ", ".join(bad)) if bad
                    else self.compute(pR, cR, pL, cL, symmetric=self.sym.get()))
        self.redraw()
        if self.on_change:
            self.on_change()

    def reset(self):
        self.sym.set(True)
        self.tab["R"].set_locked(False)
        self.tab["R"].write(self.stock_R, self.stock_camber)
        self.tab["L"].set_locked(False)
        self.tab["L"].write(self.stock_L, self.stock_camber)
        self.on_sym()
        self.apply()

    def freeze(self):
        if ok(self.cur):
            self.base = self.cur
            self.base_tag = (f"KPI {self.cur['R']['kpi']:+.2f} / "
                             f"caster {self.cur['R']['caster']:+.2f}")
            self.redraw()

    def unfreeze(self):
        self.base, self.base_tag = self.stock, "STOCK (CAD)"
        self.redraw()

    # ---------------------------------------------------------- lazy roll
    def ensure_roll(self, d):
        """Compute a result's roll-migration sweep once, on demand, and cache it.
        Returns True if it filled something in (so the caller can refresh)."""
        if not ok(d) or d.get("roll") is not None or "roll_args" not in d:
            return False
        d["roll"] = roll_migration(*d["roll_args"])
        return True

    def on_plot_tab(self, _e=None):
        """Draw whichever tab just became visible, if it is stale."""
        self._draw_visible()
        self._apply_highlight()

    def _apply_highlight(self):
        """Tint the hard points the open tab responds to, on BOTH side tables.

        A panel opts in by declaring a HIGHLIGHT dict; panels that don't simply
        clear it, so the table looks exactly as it always did. Only tab 13
        declares one so far -- the mechanism is here for the rest when wanted.
        """
        try:
            idx = self.plot_nb.index(self.plot_nb.select())
            panel = self.panels[idx]
        except (tk.TclError, IndexError):
            return
        levels = getattr(panel, "HIGHLIGHT", None)
        # a rear panel must not try to tint a front-only row name, and vice versa
        known = set(self.tab["R"].rowart)
        levels = {k: v for k, v in (levels or {}).items() if k in known}
        for t in self.tab.values():
            t.set_highlight(levels)

    def _draw_visible(self):
        """Update ONLY the tab on screen. The other 11 figures are left stale and
        redraw when their tab is next selected -- drawing all 12 on every Apply
        was ~1.8 s of waste for the panels nobody was looking at."""
        if not hasattr(self, "cur"):
            return
        try:
            idx = self.plot_nb.index(self.plot_nb.select())
        except tk.TclError:
            return
        if idx in self._drawn:
            return
        busy = False
        if idx == self.roll_tab:                       # the slow lazy roll sweep
            busy = self.ensure_roll(self.cur) | self.ensure_roll(self.base)
            if busy:
                self.config(cursor="watch")
                self.update_idletasks()
        self.panels[idx].update(self.base, self.cur)
        self._drawn.add(idx)
        if busy:                                       # roll numbers reached the table
            self.out.delete("1.0", "end")
            self.out.insert("end", self.summary())
            self.config(cursor="")

    def load_csv(self):
        fn = filedialog.askopenfilename(
            title="Load hard points", initialdir=cfg.PROJ,
            filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if not fn:
            return
        try:
            tag = self.spec["csv_tag"]
            got = {tag: {}, tag + "_L": {}}
            with open(fn) as f:
                for row in f:
                    c = row.strip().split(",")
                    if len(c) >= 5 and c[0].upper() in got:
                        got[c[0].upper()][c[1]] = np.array(
                            [float(c[2]), float(c[3]), float(c[4])])
            R = got[tag]
            missing = [n for n, _, _ in self.rows if n not in R]
            if missing:
                messagebox.showwarning("Missing points",
                                       "not in that file:\n" + "\n".join(missing))
                return
            self.tab["R"].set_locked(False)
            self.tab["R"].write(R, self.stock_camber)
            L = got[tag + "_L"] or mirror_pts(R)
            self.sym.set(not got[tag + "_L"])
            self.tab["L"].set_locked(False)
            self.tab["L"].write(L, self.stock_camber)
            self.on_sym()
            self.apply()
        except Exception as exc:                                   # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))

    def save_csv(self):
        pR, cR, badR = self.tab["R"].read()
        pL, cL, badL = self.tab["L"].read()
        if badR or badL:
            messagebox.showerror("Cannot save", "fix the highlighted boxes first")
            return
        fn = filedialog.asksaveasfilename(
            title="Save hard points", initialdir=cfg.PROJ, defaultextension=".csv",
            initialfile="hardpoints_candidate.csv", filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        with open(fn, "w") as f:
            f.write(f"# front hard points  ·  static camber R {cR:.3f}  L {cL:.3f} deg\n")
            f.write(f"# symmetric = {self.sym.get()}\n")
            f.write("Corner,Point,X,Y,Z\n")
            for name, _, _ in POINT_ROWS:
                p = pR[name]
                f.write(f"FRONT,{name},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}\n")
            if not self.sym.get():
                for name, _, _ in POINT_ROWS:
                    p = pL[name]
                    f.write(f"FRONT_L,{name},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}\n")
        messagebox.showinfo("Saved", os.path.basename(fn))

    # ---------------------------------------------------------- drawing
    def redraw(self):
        # the data changed, so every panel is now stale; draw only the visible
        # one and let the rest refresh when their tab is selected
        self._drawn = set()
        self.out.delete("1.0", "end")
        self.out.insert("end", self.summary())
        self._draw_visible()

    def summary(self):
        b, c = self.base, self.cur
        L = [f"GREY REFERENCE : {self.base_tag}",
             f"SYMMETRIC      : {'yes  (LH mirrors RH)' if self.sym.get() else 'NO'}",
             ""]
        if not ok(c):
            L += ["*** these hard points do not give a car ***", "",
                  str(c.get("error") if c else "no result"), "",
                  "Fix the box and press Enter. The grey",
                  "reference curves stay on screen."]
            return "\n".join(L)

        sym = self.sym.get()
        L.append(f"{'':20s}{'ref':>9s}{'current':>10s}{'delta':>10s}")
        L.append("-" * 53)

        def row(name, key, unit, dp=2):
            sides = ("R",) if sym else ("R", "L")
            for k, side in enumerate(sides):
                vb, vc = b[side][key], c[side][key]
                tag = f"{name:<18s}{'  ' if sym else side + ' '}" if k == 0 \
                    else f"{'':<18s}{side} "
                L.append(f"{tag}{vb:>9.{dp}f}{vc:>10.{dp}f}"
                         f"{vc - vb:>+10.{dp}f} {unit}")

        L.append("  ANGLES  (derived from the ball joints)")
        row("KPI", "kpi", "deg")
        row("caster", "caster", "deg")
        row("static camber", "static_camber", "deg")
        L.append("")
        L.append("  STEERING AXIS ON THE GROUND")
        row("scrub radius", "scrub0", "mm", 1)
        row("mech. trail", "trail0", "mm", 1)
        L.append("")
        L.append("  MOTION")
        row("camber gain", "cam_gain", "d/mm", 5)
        row("camber recovery", "recovery", "%", 2)
        row("bump steer", "bump_steer", "d/mm", 5)
        row("roll-centre height", "rc0", "mm", 1)
        row(f"motion ratio {spr.symbol()}", "mr0_disp", "-", 4)
        row("   = R_M spring/wheel", "mr0", "-", 4)
        L.append("")
        L.append("  STEERING")
        row("effective arm", "eff_arm", "mm", 1)
        row("steer @ 20mm rack", "steer20", "deg")
        if ok(c) and c.get("axle"):
            a = c["axle"]
            i = int(np.argmin(np.abs(a["rack"] - cfg.RACK_TRAVEL_MM)))
            L.append(f"{'Ackermann @20mm':<20s}{'':>9s}{a['ack'][i]:>10.3f}"
                     f"{'':>10s} deg")

        # roll-centre migration under BODY ROLL (tab 11), summarised at +5 deg
        if c.get("roll") is not None:
            rc = c["roll"]
            i5 = int(np.argmin(np.abs(rc["roll"] - 5.0)))
            i0 = int(np.argmin(np.abs(rc["roll"])))
            L += ["", "  ROLL CENTRE IN ROLL  (tab 11)"]
            L.append(f"    RC height @0°     {rc['rc_z'][i0]:+8.1f} mm")
            L.append(f"    RC height @+5°    {rc['rc_z'][i5]:+8.1f} mm  "
                     f"(Δ {rc['rc_z'][i5]-rc['rc_z'][i0]:+.1f})")
            L.append(f"    RC lateral @+5°   {rc['rc_x'][i5]:+8.1f} mm  "
                     f"(migration)")
        L.append("")
        L.append("  LENGTHS")
        row("kingpin", "kingpin", "mm", 1)
        row("upper arm", "upper_arm", "mm", 1)
        row("lower arm", "lower_arm", "mm", 1)
        L.append(f"{'track':<20s}{b['track']:>9.1f}{c['track']:>10.1f}"
                 f"{c['track'] - b['track']:>+10.1f} mm")

        L += ["", "  PACKAGING  (inboard of the mounting face)"]
        for side in (("R",) if sym else ("R", "L")):
            WC = c[side]["work_pts"]["Wheel centre"]
            rim = rim_envelope.rim_geometry(float(WC[0]))
            rim["wc_y"] = float(WC[1])
            for nm, key in (("UBJ", UBJ_NAME), ("LBJ", LBJ_NAME),
                            ("tie-rod outer", TIE_NAME)):
                ch = rim_envelope.check_point(nm, c[side]["work_pts"][key], rim,
                                              float(WC[2]))
                tag = nm if sym else f"{side} {nm}"
                L.append(f"    {tag:<16s}{ch['lat_to_face']:+8.1f} mm   "
                         f"[{'OK' if ch['ok_lateral'] else 'FAIL'}]")
        return "\n".join(L)


# =====================================================================
#  VEHICLE  -  load transfer and the g-g state, LIVE from both axles
# =====================================================================
class _SpringTable(ttk.Frame):
    """A spring CATALOGUE in the usual setup-sheet format: pick a coil
    rate off the shelf and read what it actually does to the car.

    Two side-by-side tables, front and rear, because the two axles have different
    motion ratios and corner masses and so need completely different coils. The
    row nearest the current design target is highlighted.

    "d ride height" is relative to the highlighted row, as a setup sheet
    quotes it against its reference setup: a softer spring sits the car lower on
    its static load. The tyre's own deflection is common to every row and cancels.
    """

    COLS = [("coil\nN/mm", 8), ("wheel\nN/mm", 8), ("f wheel\nHz", 8),
            ("f +tyre\nHz", 8), ("static\ndefl mm", 8), ("d ride ht\nmm", 9)]
    HDR, ROW_A, ROW_B, PICK = "#dcdcdc", "#ffffff", "#eef2f6", "#ffd97d"
    N_ROWS = 9

    def __init__(self, master):
        super().__init__(master)
        self.cells = {}
        self.caption = {}
        for side, (tag, name) in enumerate((("f", "FRONT AXLE"), ("r", "REAR AXLE"))):
            box = ttk.Frame(self)
            box.grid(row=0, column=side, padx=(0, 14), sticky="nw")
            cap = tk.Label(box, text=name, font=("Segoe UI", 8, "bold"),
                           anchor="w", fg="#333")
            cap.grid(row=0, column=0, columnspan=len(self.COLS), sticky="ew")
            self.caption[tag] = cap
            for j, (h, w) in enumerate(self.COLS):
                tk.Label(box, text=h, font=("Segoe UI", 7, "bold"), width=w,
                         relief="groove", bg=self.HDR, justify="center"
                         ).grid(row=1, column=j, sticky="ew")
            self.cells[tag] = []
            for i in range(self.N_ROWS):
                roww = []
                for j, (_h, w) in enumerate(self.COLS):
                    lb = tk.Label(box, text="", width=w, anchor="e",
                                  font=("Consolas", 8), relief="solid",
                                  borderwidth=1,
                                  bg=self.ROW_A if i % 2 == 0 else self.ROW_B)
                    lb.grid(row=2 + i, column=j, sticky="ew")
                    roww.append(lb)
                self.cells[tag].append(roww)

    def clear(self):
        for tag in ("f", "r"):
            self.caption[tag].config(text=("FRONT AXLE" if tag == "f"
                                           else "REAR AXLE") + "   -")
            for i, row in enumerate(self.cells[tag]):
                for lb in row:
                    lb.config(text="",
                              bg=self.ROW_A if i % 2 == 0 else self.ROW_B)

    def fill(self, tag, rows, pick, header):
        """`rows` is a list of 6-tuples; `pick` is the index to highlight."""
        self.caption[tag].config(text=header)
        for i, cells in enumerate(self.cells[tag]):
            base = self.ROW_A if i % 2 == 0 else self.ROW_B
            if i >= len(rows):
                for lb in cells:
                    lb.config(text="", bg=base)
                continue
            bg = self.PICK if i == pick else base
            fmt = ("{:.0f}", "{:.1f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:+.1f}")
            for lb, val, f in zip(cells, rows[i], fmt):
                lb.config(text=f.format(val), bg=bg,
                          font=("Consolas", 8, "bold") if i == pick
                          else ("Consolas", 8))


class ARBPanel(ttk.Frame):
    """Anti-roll bar design: targets -> required rate -> bar size.

    The CAD has no ARB, so the drop-link PICKUP is a design input -- but once
    placed, its motion ratio is SOLVED from the linkage, not typed in. Default
    pickup is the lower damper mount.

    Four views, then a catalogue table:
      A  required ARB wheel rate vs the roll-gradient target
      B  stress-limited MAX outer diameter vs roll angle, one curve per allowable
      C  wall thickness needed vs outer diameter, with the stress limit shaded
      D  blade angle vs stiffness, showing how much range survives the series
    """

    COLS = [("OD\nmm", 7), ("wall\nmm", 7), ("tau\nMPa", 7),
            ("kg/m", 7), ("verdict", 30)]

    def __init__(self, master, page):
        super().__init__(master)
        self.page = page
        self._rows = {"f": [], "r": []}

        top = ttk.Frame(self, padding=(6, 4))
        top.pack(side="top", fill="x")
        self.v = {}
        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        for key, lab, val, w in (("grad", "target roll gradient [deg/g]", 0.60, 7),
                                 ("lltd", "target front LLTD [%]", 50.0, 7),
                                 ("a", "arm length a [mm]", 350.0, 7),
                                 ("L", "bar torsion length [mm]", 1000.0, 7),
                                 ("roll", "roll for stress [deg]", 3.0, 7),
                                 ("tau", "allowable [MPa]", 550.0, 7),
                                 ("cmp", "compare [deg/g]", 1.2, 7)):
            ttk.Label(row1, text=lab, font=("Consolas", 8)).pack(side="left",
                                                                 padx=(6, 2))
            sv = tk.StringVar(value=f"{val:g}")
            e = ttk.Entry(row1, textvariable=sv, width=w, font=("Consolas", 8),
                          justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self.refresh())
            self.v[key] = sv
        ttk.Button(row1, text="Apply", command=self.refresh).pack(side="left",
                                                                  padx=8)

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(3, 0))
        ttk.Label(row2, text="drop-link pickup  (CAD has none - design input, "
                             "defaults to the lower damper mount):",
                  font=("Segoe UI", 8, "bold"), foreground="#333").pack(side="left")
        self.pick = {}
        for tag, nm in (("f", "FRONT"), ("r", "REAR")):
            ttk.Label(row2, text=f"  {nm}", font=("Consolas", 8)).pack(side="left")
            self.pick[tag] = []
            for _ax in "XYZ":
                sv = tk.StringVar(value="")
                e = tk.Entry(row2, textvariable=sv, width=9, justify="right",
                             font=("Consolas", 8), relief="solid", borderwidth=1)
                e.pack(side="left", padx=1)
                e.bind("<Return>", lambda _e: self.refresh())
                self.pick[tag].append(sv)
        ttk.Button(row2, text="reset to damper mount",
                   command=self._reset_pickups).pack(side="left", padx=6)
        self.note = ttk.Label(top, text="", font=("Consolas", 8),
                              foreground="#555")
        self.note.pack(anchor="w", pady=(3, 0))
        self.cmpnote = ttk.Label(top, text="", font=("Consolas", 8),
                                 foreground="#7a5195")
        self.cmpnote.pack(anchor="w")

        # ------------------------------------------------- BAR | BLADES tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True)
        bar_tab = ttk.Frame(self.nb)
        blade_tab = ttk.Frame(self.nb)
        self.nb.add(bar_tab, text="  Bar  ")
        self.nb.add(blade_tab, text="  Blades  ")
        self._build_blades(blade_tab)

        tblhost = ttk.Frame(bar_tab, padding=(6, 2))
        tblhost.pack(side="bottom", fill="x")
        self.cells = {}
        for side, (tag, nm) in enumerate((("f", "FRONT BAR"), ("r", "REAR BAR"))):
            box = ttk.Frame(tblhost)
            box.grid(row=0, column=side, padx=(0, 14), sticky="nw")
            cap = tk.Label(box, text=nm, font=("Segoe UI", 8, "bold"), anchor="w",
                           fg="#333")
            cap.grid(row=0, column=0, columnspan=len(self.COLS), sticky="ew")
            self.cells.setdefault("cap", {})[tag] = cap
            for j, (h, w) in enumerate(self.COLS):
                tk.Label(box, text=h, font=("Segoe UI", 7, "bold"), width=w,
                         relief="groove", bg="#dcdcdc").grid(row=1, column=j,
                                                             sticky="ew")
            self.cells[tag] = []
            for i in range(8):
                rw = []
                for j, (_h, w) in enumerate(self.COLS):
                    lb = tk.Label(box, text="", width=w,
                                  anchor="w" if j == 4 else "e",
                                  font=("Consolas", 8), relief="solid",
                                  borderwidth=1,
                                  bg="#ffffff" if i % 2 == 0 else "#eef2f6")
                    lb.grid(row=2 + i, column=j, sticky="ew")
                    rw.append(lb)
                self.cells[tag].append(rw)

        # Plain 2x2 with the fourth cell blanked. C used to span the full width
        # and came out badly elongated, so it stays a quarter like A and B.
        self.fig = Figure(figsize=(11.5, 5.4), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.46, wspace=0.26)
        self.ax = [self.fig.add_subplot(gs[0, 0]),
                   self.fig.add_subplot(gs[0, 1]),
                   self.fig.add_subplot(gs[1, 0])]
        self._blank = self.fig.add_subplot(gs[1, 1])
        self._blank.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=bar_tab)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, bar_tab).update()

    # ------------------------------------------------------------- blades tab
    def _build_blades(self, parent):
        """Blade geometry, the angle law, and the position chart.

        Root and tip width are separate inputs: equal means a prismatic
        rectangle, unequal means a linearly tapered blade and the model
        integrates the varying section along the length.
        """
        top = ttk.Frame(parent, padding=(6, 4))
        top.pack(side="top", fill="x")
        r1 = ttk.Frame(top)
        r1.pack(fill="x")
        self.bv = {}
        for key, lab, val in (("broot", "blade ROOT width b [mm]", 40.0),
                              ("btip", "TIP width [mm]  (= root -> rectangle)", 25.0),
                              ("h", "thickness h [mm]", 8.0),
                              ("ell", "free length l [mm]", 230.0),
                              ("npos", "positions", 6.0)):
            ttk.Label(r1, text=lab, font=("Consolas", 8)).pack(side="left",
                                                               padx=(6, 2))
            sv = tk.StringVar(value=f"{val:g}")
            e = ttk.Entry(r1, textvariable=sv, width=7, font=("Consolas", 8),
                          justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self.refresh())
            self.bv[key] = sv
        ttk.Button(r1, text="Apply", command=self.refresh).pack(side="left", padx=8)

        self.bnote = ttk.Label(top, text="", font=("Consolas", 8),
                               foreground="#555", justify="left")
        self.bnote.pack(anchor="w", pady=(3, 0))

        ref = ("BLADE MODEL & LIMITATIONS.  k = 3EI/l^3 per axis;  I_weak = b h^3/12 (flat),  "
               "I_strong = h b^3/12 (edge-on);  range = (b/h)^2.\n"
               "Angle law is UNSYMMETRICAL BENDING -- the COMPLIANCES combine:  "
               "k(th) = 1 / [ cos^2(th)/k_weak + sin^2(th)/k_strong ].  Combining the "
               "second moments arithmetically overstates 45 deg by ~7x.\n"
               "Blade and bar act in SERIES, so the softer one dominates and a stiff bar "
               "collapses the blade's range.  Taper softens the strong axis (b^3) far more "
               "than the weak (b), so it also shrinks the range.\n"
               "Refs: suspensionsecrets.co.uk/anti-roll-bars-2  (blade rotated 90 deg, "
               "stiffest edge-on; taper usually approximated as rectangular) · "
               "justin-lau-portfolio.webflow.io/work/arb  (uniform blade chosen; a tapered "
               "one was abandoned after FEA showed excessive displacement and stress).\n"
               "SUPPLIERS DO NOT PUBLISH BLADE SECTIONS -- searched and not found. Bar "
               "diameters are real catalogue sizes; every blade dimension here is ASSUMED.")
        tk.Label(parent, text=ref, font=("Consolas", 7), justify="left",
                 anchor="w", bg="#f7f7f7", relief="solid", borderwidth=1,
                 wraplength=1330).pack(side="bottom", fill="x", padx=6, pady=(2, 4))

        self.bfig = Figure(figsize=(11.5, 5.6), dpi=100)
        bgs = self.bfig.add_gridspec(2, 2, wspace=0.24, hspace=0.46)
        # [0] the blade drawing   [1] D angle law   [2] E1 front   [3] E2 rear
        self.bax = [self.bfig.add_subplot(bgs[0, 0]),
                    self.bfig.add_subplot(bgs[0, 1]),
                    self.bfig.add_subplot(bgs[1, 0]),
                    self.bfig.add_subplot(bgs[1, 1])]
        self.bcanvas = FigureCanvasTkAgg(self.bfig, master=parent)
        self.bcanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.bcanvas, parent).update()

    def _bnum(self, k, d):
        try:
            return float(self.bv[k].get())
        except ValueError:
            return d

    def _num(self, k, d):
        try:
            return float(self.v[k].get())
        except ValueError:
            return d

    def _reset_pickups(self):
        got = self.page.get_pts() if self.page.get_pts else None
        if not got:
            return
        for tag, pts in (("f", got[0]), ("r", got[2])):
            p = pts["Lower Shock mount"]
            for sv, val in zip(self.pick[tag], p):
                sv.set(f"{float(val):.3f}")
        self.refresh()

    def _pickup(self, tag, pts):
        try:
            v = [float(sv.get()) for sv in self.pick[tag]]
            if any(x != x for x in v):
                raise ValueError
            return np.array(v, float)
        except ValueError:
            return np.asarray(pts["Lower Shock mount"], float)

    def refresh(self):
        got = self.page.get_pts() if self.page.get_pts else None
        car = getattr(self.page, "car", None)
        rc_f, rc_r = self.page.get_rcs()
        for a in self.ax:
            a.clear()
        if got is None or car is None or rc_f is None or rc_r is None:
            self.ax[0].text(0.5, 0.5, "waiting for a solvable geometry",
                            ha="center", va="center", color="#888")
            self.canvas.draw_idle()
            return
        pf, cf, pr, cr = got
        if not self.pick["f"][0].get():
            for tag, pts in (("f", pf), ("r", pr)):
                for sv, val in zip(self.pick[tag], pts["Lower Shock mount"]):
                    sv.set(f"{float(val):.3f}")

        mr_f, att_f = arb.pickup_motion_ratio(pf, cf, self._pickup("f", pf))
        mr_r, att_r = arb.pickup_motion_ratio(pr, cr, self._pickup("r", pr))
        grad, lltd = self._num("grad", 0.60), self._num("lltd", 50.0)
        a_mm, L_mm = self._num("a", 350.0), self._num("L", 1000.0)
        roll, tau = self._num("roll", 3.0), self._num("tau", 550.0)

        af, ar, info = arb.solve_for_targets(car, rc_f, rc_r, grad, lltd)
        cmp_g = self._num("cmp", 1.2)
        cmp_f, cmp_r, _ci = arb.solve_for_targets(car, rc_f, rc_r, cmp_g, lltd)
        self.note.config(
            text=f"ARB motion ratio {spr.symbol()}  "
                 f"F {spr.to_display(mr_f):.4f} (pickup on the {att_f} arm)   "
                 f"R {spr.to_display(mr_r):.4f} (on the {att_r} arm)      required rate  "
                 f"F {af:.1f}  R {ar:.1f} N/mm      {grad:.2f} deg/g is "
                 f"'{arb.classify_roll_gradient(grad)}' (OptimumG Fig.1)"
                 + ("" if info["converged"] else "   [target not reachable]"))
        # what the comparison target would cost, in the same units, so the
        # trade-off is a number rather than an argument
        self.cmpnote.config(
            text=f"COMPARE {cmp_g:.2f} deg/g -> "
                 f"'{arb.classify_roll_gradient(cmp_g)}':   required rate "
                 f"F {cmp_f:.1f}  R {cmp_r:.1f} N/mm"
                 f"   (vs F {af:.1f}  R {ar:.1f} at {grad:.2f})"
                 + (f"   -> front bar needs "
                    f"{float(arb.required_Kt(cmp_f, a_mm, mr_f)):.0f} vs "
                    f"{float(arb.required_Kt(af, a_mm, mr_f)):.0f} Nm/rad"
                    if cmp_f > 0 else "   -> springs alone already reach it"))

        AX = dict(f=dict(k=af, mr=mr_f, track=car["track_f_mm"], col="#c1121f",
                         nm="FRONT"),
                  r=dict(k=ar, mr=mr_r, track=car["track_r_mm"], col="#0353a4",
                         nm="REAR"))
        for tag, d in AX.items():
            d["Kt"] = float(arb.required_Kt(d["k"], a_mm, d["mr"]))
            # sized once, here, so the table below and the position charts on the
            # Blades tab are guaranteed to be talking about the SAME bars
            d["rows"] = arb.size_bar(d["Kt"], L_mm, roll, d["track"], a_mm,
                                     d["mr"], tau_allow_MPa=tau)

        # ---- A: required rate vs roll-gradient target
        a0 = self.ax[0]
        gs_ = np.linspace(max(0.30, grad * 0.5), grad * 2.0, 25)
        cf_, cr_ = [], []
        for g in gs_:
            x, y, _ = arb.solve_for_targets(car, rc_f, rc_r, float(g), lltd)
            cf_.append(x)
            cr_.append(y)
        a0.plot(gs_, cf_, lw=2.2, color="#c1121f", label="FRONT")
        a0.plot(gs_, cr_, lw=2.2, color="#0353a4", label="REAR")
        for tag, d in AX.items():
            a0.plot([grad], [d["k"]], "s", ms=9, mfc=d["col"], mec="w", mew=1.8,
                    zorder=6)
        a0.axvline(grad, ls="-", lw=1.2, color="#333", alpha=0.7)
        # A SECOND target drawn alongside, so the cost of the choice is visible
        # rather than argued about: 0.60 deg/g sits below OptimumG's GT band
        # (1.0-3.0) and the comparison shows what moving into it would buy.
        a0.axvline(cmp_g, ls=":", lw=1.4, color="#7a5195", alpha=0.9)
        for _k, _c in ((cmp_f, "#c1121f"), (cmp_r, "#0353a4")):
            a0.plot([cmp_g], [_k], "s", ms=9, mfc="none", mec=_c, mew=1.8,
                    zorder=6)
        a0.annotate(f"compare {cmp_g:.2f}", (cmp_g, a0.get_ylim()[1] * 0.55),
                    fontsize=6.5, color="#7a5195", rotation=90, va="top",
                    ha="right")
        # OptimumG "Bar talk" Fig.1 bands, so the target can be judged against
        # published practice rather than picked in a vacuum. They overlap, so
        # they are drawn as stacked strips along the top rather than shaded.
        _ymax = a0.get_ylim()[1]
        for _i, (_nm, _lo, _hi) in enumerate(arb.ROLL_GRADIENT_BANDS):
            if _hi < gs_[0] or _lo > gs_[-1]:
                continue
            _y = _ymax * (0.97 - 0.045 * _i)
            a0.plot([max(_lo, gs_[0]), min(_hi, gs_[-1])], [_y, _y], lw=4,
                    solid_capstyle="butt", color="#8a8a8a", alpha=0.45)
            a0.text(min(_hi, gs_[-1]), _y, f" {_nm}", fontsize=5.8,
                    color="#555", va="center", ha="left")
        a0.set_xlabel("target roll gradient  [deg/g]", fontsize=8.5)
        a0.set_ylabel("required ARB rate  [N/mm]", fontsize=8.5)
        a0.set_title(f"A · required ARB rate   (front LLTD held at {lltd:.0f} %)",
                     fontsize=9)
        a0.legend(fontsize=7.5)

        # ---- B: stress-limited max OD vs roll angle, per allowable
        a1 = self.ax[1]
        rolls = np.linspace(0.5, 5.0, 60)
        for tag, d in AX.items():
            tw = arb.bar_twist_rad(rolls, d["track"], a_mm, d["mr"])
            for ta, ls in zip(arb.ALLOWABLE_TAU_MPA, (":", "-", "--")):
                a1.plot(rolls, arb.max_od_for_stress(ta, tw, L_mm), ls=ls,
                        lw=1.8, color=d["col"],
                        label=f"{d['nm']} {ta:.0f} MPa" if tag == "f" or ls == "-"
                        else None)
        a1.axvline(roll, ls="--", lw=1.0, color="#666", alpha=0.6)
        a1.set_ylim(0, 90)
        a1.set_xlabel("body roll  [deg]", fontsize=8.5)
        a1.set_ylabel("max usable OD  [mm]", fontsize=8.5)
        a1.set_title("B · stress-limited OD   (wall does NOT affect this)",
                     fontsize=9)
        a1.legend(fontsize=6.5, ncol=2)

        # ---- C: wall needed vs OD
        a2 = self.ax[2]
        ods = np.linspace(20.0, 70.0, 200)
        for tag, d in AX.items():
            w = arb.wall_for_stiffness(ods, d["Kt"], L_mm)
            a2.plot(ods, w, lw=2.2, color=d["col"], label=d["nm"])
            tw = float(arb.bar_twist_rad(roll, d["track"], a_mm, d["mr"]))
            od_max = float(arb.max_od_for_stress(tau, tw, L_mm))
            a2.axvline(od_max, ls="--", lw=1.2, color=d["col"], alpha=0.7)
            a2.text(od_max, a2.get_ylim()[1] * 0.95, f" {d['nm']} limit",
                    fontsize=6.5, color=d["col"], rotation=90, va="top")
        a2.axhline(2.0, ls=":", lw=1.2, color="#666")
        # anchored in AXES coords: the x-limits autoscale, so a data-space x
        # here lands outside the plot and the text spills into the y-label
        a2.text(0.985, 2.0, "2 mm manufacturing floor ", fontsize=6.5,
                color="#666", va="bottom", ha="right",
                transform=a2.get_yaxis_transform())
        a2.set_ylim(0, 14)
        a2.set_xlabel("outer diameter  [mm]", fontsize=8.5)
        a2.set_ylabel("wall needed  [mm]", fontsize=8.5)
        a2.set_title(f"C · wall for the required stiffness   "
                     f"(dashed = {tau:.0f} MPa limit at {roll:.1f} deg)", fontsize=9)
        a2.legend(fontsize=7.5)

        for a in self.ax:
            a.grid(alpha=0.3)
            a.tick_params(labelsize=7.5)
        self.fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.10)
        self.canvas.draw_idle()
        self._draw_blades(AX, a_mm, L_mm, roll, tau)

        # ---- the catalogue table
        for tag, d in AX.items():
            rows = d["rows"]
            ok_rows = [r for r in rows if r["ok"]] or rows[-8:]
            self._rows[tag] = ok_rows[:8]
            self.cells["cap"][tag].config(
                text=f"{d['nm']} BAR   need K_t {d['Kt']:.0f} Nm/rad   "
                     f"MR {d['mr']:.4f}   twist "
                     f"{np.degrees(arb.bar_twist_rad(roll, d['track'], a_mm, d['mr'])):.1f} deg"
                     f" at {roll:.1f} deg roll")
            for i, cellrow in enumerate(self.cells[tag]):
                base = "#ffffff" if i % 2 == 0 else "#eef2f6"
                if i >= len(self._rows[tag]):
                    for lb in cellrow:
                        lb.config(text="", bg=base)
                    continue
                r = self._rows[tag][i]
                bg = "#d7f0d7" if r["ok"] else "#f7d7d7"
                w = f"{r['wall_mm']:.2f}" if np.isfinite(r["wall_mm"]) else "solid"
                for lb, txt in zip(cellrow, (f"{r['od_mm']:.0f}", w,
                                             f"{r['tau_MPa']:.0f}",
                                             f"{r['mass_kg_m']:.2f}",
                                             r["verdict"])):
                    lb.config(text=txt, bg=bg)
    def _blade_drawing(self, ax, b_r, b_t, h, ell):
        """Schematic of the bladed cantilever, laid out like the hand sketch: the
        ARB tube cross-section on the left, then the blade in PLAN view (root
        width b tapering to the tip) and in EDGE view (thickness h).

        A legend for the dimension boxes above -- drawn to scale in the BLADE's
        own proportions, not the bar's.
        """
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        scale = max(b_r, b_t, h, 1e-6)
        L = ell / scale
        BR, BT, H = b_r / scale, b_t / scale, h / scale

        # --- ARB tube cross-section (hatched annulus), left of the blade
        ro = 0.42 * max(BR, 1.0)
        cx, cy = -ro - 0.45, 0.0
        ax.add_patch(Circle((cx, cy), ro, fill=False, lw=1.6, color="#333",
                            hatch="////"))
        ax.add_patch(Circle((cx, cy), ro * 0.42, fc="white", ec="#333", lw=1.4))
        ax.annotate("ARB", (cx, cy + ro), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color="#333")

        # --- PLAN view: the taper, root width BR at x=0 -> tip width BT at x=L
        yo = 0.62 * max(BR, 1.0)
        ax.add_patch(Polygon([(0, yo + BR / 2), (L, yo + BT / 2),
                              (L, yo - BT / 2), (0, yo - BR / 2)],
                             closed=True, fc="#cfe8e3", ec="#0f6b5f", lw=1.6))
        ax.annotate("", (0, yo + BR / 2), (0, yo - BR / 2),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color="#0f6b5f"))
        ax.text(-0.05 * L, yo, f"b\n{b_r:.0f}", ha="right", va="center",
                fontsize=8, color="#0f6b5f", fontweight="bold")
        ax.annotate("", (L, yo + BT / 2), (L, yo - BT / 2),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color="#0f6b5f"))
        ax.text(L * 1.03, yo, f"tip\n{b_t:.0f}", ha="left", va="center",
                fontsize=8, color="#0f6b5f", fontweight="bold")
        ax.text(L / 2, yo + max(BR, BT) / 2 + 0.20, "PLAN", ha="center",
                fontsize=7, color="#666")

        # --- EDGE view: constant thickness H
        y2 = -0.62 * max(BR, 1.0)
        ax.add_patch(Rectangle((0, y2 - H / 2), L, H, fc="#cfe8e3",
                               ec="#0f6b5f", lw=1.6))
        # h is tiny beside the length, so a dimension arrow across it collapses
        # into a blob -- a leader line reads at any aspect ratio
        ax.annotate(f"h = {h:.0f}", xy=(0.04 * L, y2),
                    xytext=(-0.16 * L, y2 - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#0f6b5f"),
                    fontsize=8, color="#0f6b5f", fontweight="bold",
                    ha="center", va="top")
        ax.text(L / 2, y2 + H / 2 + 0.20, "EDGE", ha="center", fontsize=7,
                color="#666")

        # --- free length, under everything
        yb = y2 - max(H / 2, 0.3) - 0.42
        ax.annotate("", (0, yb), (L, yb),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color="#333"))
        ax.text(L / 2, yb - 0.18, f"free length  l = {ell:.0f} mm", ha="center",
                va="top", fontsize=8, color="#333")

        shape = "prismatic rectangle" if abs(b_t - b_r) < 1e-9 else "tapered"
        ax.set_title(f"bladed cantilever  -  {shape}\n"
                     f"(schematic, blade proportions to scale)", fontsize=9)
        ax.set_xlim(cx - ro - 0.30, L * 1.18)
        ax.set_ylim(yb - 0.50, yo + max(BR, BT) / 2 + 0.50)

    def _draw_blades(self, AX, a_mm, L_mm, roll, tau):
        """D: the blade's own angle law.  E1/E2: rate vs position, per axle."""
        draw, d, e1, e2 = self.bax
        for a in (d, e1, e2):
            a.clear()
        b_r = self._bnum("broot", 40.0)
        b_t = self._bnum("btip", 25.0)
        h = self._bnum("h", 8.0)
        ell = self._bnum("ell", 230.0)
        npos = max(2, int(self._bnum("npos", 6.0)))
        shape = ("rectangle" if abs(b_t - b_r) < 1e-9
                 else f"tapered {b_r:.0f}->{b_t:.0f}")

        try:
            kw, ks = arb.blade_principal_rates(b_r, b_t, h, ell)
            self._blade_drawing(draw, b_r, b_t, h, ell)
        except ValueError:
            draw.clear()
            draw.axis("off")
            for a in (d, e1, e2):
                a.text(0.5, 0.5, "blade dimensions must be positive",
                       ha="center", va="center", color="#888")
            self.bcanvas.draw_idle()
            return

        # ---- D: the blade ALONE. The bar/series behaviour is what E1 and E2
        # show; overlaying it here made the panel hard to read.
        th = np.linspace(0.0, 90.0, 181)
        kb = arb.blade_rate_at_angle(th, kw, ks)
        d.plot(th, kb, lw=2.4, color="#2a9d8f", label="blade (compliance law)")
        wrong = (np.cos(np.radians(th)) ** 2 * kw
                 + np.sin(np.radians(th)) ** 2 * ks)
        d.plot(th, wrong, lw=1.2, ls=":", color="#999",
               label="arithmetic I  (wrong)")
        # a marked, labelled point at every detent so the numbers can be read off
        # without a cursor, and the x-axis pinned to the blade's full 0-90 range
        pang = arb.blade_positions(npos)
        pk = arb.blade_rate_at_angle(pang, kw, ks)
        for t in pang:
            d.axvline(t, lw=0.7, color="#bbb", zorder=0)
        d.plot(pang, pk, "o", ms=6.5, mfc="#2a9d8f", mec="k", mew=1.2, zorder=6)
        for i, (t, k) in enumerate(zip(pang, pk), 1):
            d.annotate(f"{k:.0f}", (t, k), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=7,
                       color="#0f6b5f", fontweight="bold")
            d.annotate(f"P{i}", (t, k), textcoords="offset points",
                       xytext=(0, -14), ha="center", fontsize=6.5, color="#777")
        d.set_yscale("log")
        d.set_xlim(0, 90)
        d.set_xticks(np.round(pang, 1))
        d.set_xlabel("blade angle  [deg]   (0 = flat, 90 = edge-on)", fontsize=8.5)
        d.set_ylabel("blade rate at its tip  [N/mm]", fontsize=8.5)
        d.set_title(f"D · blade angle law  -  BLADE ALONE\n{shape}, {h:.0f} mm "
                    f"thick, {ell:.0f} mm long   range {ks / kw:.1f}:1", fontsize=9)
        d.legend(fontsize=7, loc="upper left")

        # ---- E1 / E2: ARB rate vs blade position, one panel per axle
        notes = []
        for ax_, tag, panel in ((e1, "f", "E1"), (e2, "r", "E2")):
            dd = AX[tag]
            # Use the TABLE's own top-2 diameters, each snapped to the nearest
            # standard wall. The table's solved walls make every row deliver the
            # same K_t, so plotting two of them raw would draw one curve twice;
            # the standard-wall rounding separates them and brackets the target,
            # while keeping the chart and the table talking about the same bars.
            secs = arb.chart_sections(dd["rows"], 2)
            if not secs:
                ax_.text(0.5, 0.5, "no section in the table passes",
                         ha="center", va="center", color="#888", fontsize=9)
            else:
                _thp, curves = arb.rate_vs_position(
                    secs, (b_r, b_t, h, ell), a_mm, dd["mr"], L_mm, npos)
                pos = np.arange(1, npos + 1)
                # the two sections differ by only a few per cent (they are the
                # same requirement rounded two ways), so their labels collide --
                # push the softer one below the curve and the stiffer one above
                for i, ((lab, rates), col) in enumerate(
                        zip(curves.items(), ("#0353a4", "#c1121f"))):
                    ax_.plot(pos, rates, "-o", lw=2.0, ms=5.5, color=col,
                             label=lab)
                    dy = -12 if i == 0 else 7
                    for x, y in zip(pos, rates):
                        ax_.annotate(f"{y:.1f}", (x, y),
                                     textcoords="offset points", xytext=(0, dy),
                                     ha="center", fontsize=6, color=col)
                ax_.axhline(dd["k"], ls="--", lw=1.3, color="#444")
                ax_.text(0.985, dd["k"], f"required {dd['k']:.1f} ", fontsize=7,
                         va="center", ha="right", color="#444",
                         transform=ax_.get_yaxis_transform(),
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                   ec="none", alpha=0.85))
                ax_.set_xticks(pos)
                ax_.set_xlim(0.5, npos + 0.5)
                ax_.legend(fontsize=6.5, loc="upper left")
                notes.append(f"{dd['nm']} " + " / ".join(
                    f"{v[-1] / v[0]:.2f}x" for v in curves.values()))
            ax_.set_xlabel("ARB position", fontsize=8.5)
            ax_.set_ylabel("ARB rate at the wheel  [N/mm]", fontsize=8.5)
            secname = (" and ".join(f"{o:.0f}x{w:.0f}" for o, w in secs)
                       if secs else "-")
            ax_.set_title(f"{panel} · {dd['nm']} ARB rate vs blade position   "
                          f"(ARB chart format)\nthe TABLE's top two ODs at "
                          f"their nearest standard wall:  {secname}", fontsize=9)

        self.bnote.config(
            text=f"blade {shape}, {h:.0f} x {ell:.0f} mm:  k_weak {kw:.1f}  "
                 f"k_strong {ks:.1f} N/mm  ->  {ks / kw:.1f}:1 alone."
                 f"   In series the {npos}-position span is   "
                 + "   ".join(notes)
                 + f"      (benchmark front spans 2.42x, rear 2.23-2.80x)")
        for a in (d, e1, e2):
            a.grid(alpha=0.3)
            a.tick_params(labelsize=7.5)
        self.bfig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.09)
        self.bcanvas.draw_idle()
class RollAxisPanel(ttk.Frame):
    """The roll axis, and the front/rear roll-centre balance.

    Both list items are the same geometry: two roll centres define a line, where
    that line sits under the sprung CG sets the roll LEVER, and how it is tilted
    sets how the transfer divides front to rear.

      R1  side view -- the axis, the roll centres, the CG and the lever
      R2  the axis THROUGH a corner, rebuilt from both migrating roll centres
      R3  the balance map -- front LLTD over a grid of front x rear RC height
      R4  geometric / elastic / unsprung split per axle vs roll-centre height
    """

    def __init__(self, master, page):
        super().__init__(master)
        self.page = page

        top = ttk.Frame(self, padding=(6, 4))
        top.pack(side="top", fill="x")
        self.note = ttk.Label(top, text="", font=("Consolas", 8),
                              foreground="#555", justify="left")
        self.note.pack(anchor="w")
        ttk.Label(top, text="roll centres are LIVE from the FRONT and REAR hard "
                            "points - the map shows what MOVING them would do",
                  font=("Segoe UI", 8), foreground="#888").pack(anchor="w")

        self.fig = Figure(figsize=(11.5, 6.4), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.44, wspace=0.26)
        self.ax = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1]),
                   self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self).update()
        self._cb = None

    def refresh(self):
        car = getattr(self.page, "car", None)
        rc_f, rc_r = self.page.get_rcs()
        for a in self.ax:
            a.clear()
        if car is None or rc_f is None or rc_r is None:
            self.ax[0].text(0.5, 0.5, "waiting for a solvable geometry",
                            ha="center", va="center", color="#888")
            self.canvas.draw_idle()
            return

        st = ra.state(rc_f, rc_r, car)
        sp = ra.split(rc_f, rc_r, car)
        wb = car["wheelbase_mm"]

        # ---------------------------------------------------- R1  side view
        a = self.ax[0]
        a.plot([0, wb], [rc_f, rc_r], "-", lw=2.4, color="#c1121f",
               label="roll axis")
        a.plot([0], [rc_f], "o", ms=9, mfc="#c1121f", mec="k", mew=1.2)
        a.plot([wb], [rc_r], "o", ms=9, mfc="#0353a4", mec="k", mew=1.2)
        a.annotate(f"front RC\n{rc_f:+.0f}", (0, rc_f), textcoords="offset points",
                   xytext=(6, -18), fontsize=7.5, color="#c1121f")
        a.annotate(f"rear RC\n{rc_r:+.0f}", (wb, rc_r), textcoords="offset points",
                   xytext=(-6, 10), ha="right", fontsize=7.5, color="#0353a4")
        x_cg = st["u_cg"] * wb
        a.plot([x_cg], [car["h_cg_mm"]], "*", ms=15, color="#111", zorder=6)
        a.annotate("sprung CG", (x_cg, car["h_cg_mm"]),
                   textcoords="offset points", xytext=(8, 4), fontsize=7.5)
        a.annotate("", (x_cg, car["h_cg_mm"]), (x_cg, st["h_axis"]),
                   arrowprops=dict(arrowstyle="<->", lw=1.6, color="#2a9d8f"))
        a.text(x_cg + 45, (car["h_cg_mm"] + st["h_axis"]) / 2,
               f"roll lever\n{st['lever']:.0f} mm", fontsize=8, color="#0f6b5f",
               fontweight="bold", va="center")
        # the benchmark axis, for scale -- both its roll centres sit ABOVE ground
        bf, br = ra.BENCHMARK_RC_MM
        bst = ra.state(bf, br, car)
        a.plot([0, wb], [bf, br], "--", lw=1.8, color="#7a5195",
               label="benchmark axis")
        a.plot([0, wb], [bf, br], "o", ms=5, mfc="#7a5195", mec="none")
        a.annotate(f"benchmark  lever {bst['lever']:.0f} mm",
                   (wb * 0.55, (bf + br) / 2), textcoords="offset points",
                   xytext=(0, 10), fontsize=7, color="#7a5195", ha="center")
        a.axhline(0, lw=1.4, color="#333")
        a.text(wb * 0.98, 6, "ground", fontsize=7, color="#333", ha="right")
        a.plot([0, wb], [0, 0], "|", ms=10, color="#333")
        a.set_xlabel("distance from the front axle  [mm]", fontsize=8.5)
        a.set_ylabel("height  [mm]", fontsize=8.5)
        a.set_title(f"R1 · the roll axis in side view\ninclination "
                    f"{st['incl_deg']:+.2f}° "
                    f"({'nose-down' if st['incl_deg'] > 0 else 'nose-up'})",
                    fontsize=9)
        a.legend(fontsize=7, loc="lower right")

        # ------------------------------------------------- R2  through a corner
        a = self.ax[1]
        mf, mr = self.page.get_roll() if self.page.get_roll else (None, None)
        if mf is None or mr is None:
            a.text(0.5, 0.5, "roll sweep not available", ha="center",
                   va="center", color="#888")
        else:
            ir = ra.in_roll(mf["roll"], mf["rc_z"], mr["roll"], mr["rc_z"], car)
            # The sweep runs to +/-5 deg, but a near-ground roll centre is
            # hyper-sensitive there (nearly parallel force lines), so quoting the
            # full-range swing badly overstates what the car actually lives with.
            # Shade the band the car REALLY sees -- its own roll gradient times
            # the g-g envelope -- and quote the lever change over THAT.
            band = abs(ltr.transfer(1.0, rc_f, rc_r, car)["roll_deg"]
                       * self.page.LAT_MAX)
            a.axvspan(-band, band, color="#2a9d8f", alpha=0.12, zorder=0)
            a.plot(ir["roll"], ir["rc_f"], lw=2.0, color="#c1121f",
                   label="front RC")
            a.plot(ir["roll"], ir["rc_r"], lw=2.0, color="#0353a4", label="rear RC")
            a.plot(ir["roll"], ir["h_axis"], lw=2.4, ls="--", color="#2a9d8f",
                   label="axis under the CG")
            a2 = a.twinx()
            a2.plot(ir["roll"], ir["lever"], lw=1.6, ls=":", color="#7a5195",
                    label="roll lever")
            a2.set_ylabel("roll lever  [mm]", color="#7a5195", fontsize=8.5)
            a2.tick_params(axis="y", colors="#7a5195", labelsize=7.5)
            a.axhline(0, lw=1.0, color="#999")
            a.legend(fontsize=6.5, loc="lower center", ncol=3)
            lev_b = float(np.interp(band, ir["roll"], ir["lever"]))
            lev_0 = float(np.interp(0.0, ir["roll"], ir["lever"]))
            a.text(0.0, a.get_ylim()[0], f"  the car only reaches ±{band:.1f}°",
                   fontsize=6.5, color="#0f6b5f", va="bottom", ha="center")
            a.set_title(f"R2 · through a corner   lever {lev_0:.0f} → {lev_b:.0f} mm "
                        f"({lev_b / lev_0 - 1:+.1%}) over ±{band:.1f}° of real roll\n"
                        f"(the ±5° sweep is shown, but a near-ground RC is "
                        f"hyper-sensitive out there)", fontsize=8.5)
        a.set_xlabel("body roll  [deg]", fontsize=8.5)
        a.set_ylabel("height  [mm]", fontsize=8.5)

        # ------------------------------------------------- R3  balance map
        a = self.ax[2]
        gf = np.linspace(-320, 80, 26)
        gr = np.linspace(-320, 80, 26)
        F, R, LLTD, GRAD, LEV = ra.balance_map(gf, gr, car)
        cf = a.contourf(F, R, LLTD, levels=18, cmap="RdBu_r")
        if self._cb is None:
            self._cb = self.fig.colorbar(cf, ax=a, fraction=0.05, pad=0.02)
            self._cb.set_label("front LLTD  [%]", fontsize=8)
        else:
            self._cb.update_normal(cf)
        cs = a.contour(F, R, LLTD, levels=8, colors="k", linewidths=0.6,
                       alpha=0.5)
        a.clabel(cs, fontsize=6, fmt="%.0f")
        a.plot([rc_f], [rc_r], "o", ms=11, mfc="yellow", mec="k", mew=1.6,
               zorder=6)
        a.annotate("ours", (rc_f, rc_r), textcoords="offset points",
                   xytext=(9, 6), fontsize=8, fontweight="bold")
        a.plot([bf], [br], "D", ms=9, mfc="#7a5195", mec="k", mew=1.4, zorder=6)
        a.annotate("benchmark", (bf, br), textcoords="offset points",
                   xytext=(-9, 6), ha="right", fontsize=8, fontweight="bold",
                   color="#7a5195")
        a.axhline(0, lw=0.9, ls=":", color="#333")
        a.axvline(0, lw=0.9, ls=":", color="#333")
        a.plot([-320, 80], [-320, 80], lw=0.9, ls="--", color="#555", alpha=0.6)
        a.text(60, 62, "level axis", fontsize=6, color="#555", rotation=45,
               ha="right")
        a.set_xlabel("FRONT roll-centre height  [mm]", fontsize=8.5)
        a.set_ylabel("REAR roll-centre height  [mm]", fontsize=8.5)
        a.set_title("R3 · F/R roll-centre BALANCE map\n"
                    "what moving the roll centres alone does to LLTD", fontsize=9)

        # ------------------------------------- R4  geometric / elastic split
        a = self.ax[3]
        hs = np.linspace(-320, 80, 60)
        for tag, nm, col, other in (("f", "FRONT", "#c1121f", rc_r),
                                    ("r", "REAR", "#0353a4", rc_f)):
            geo = []
            for h in hs:
                s = ra.split(h, other, car) if tag == "f" else ra.split(other, h, car)
                geo.append(s[f"geo_pct_{tag}"])
            a.plot(hs, geo, lw=2.2, color=col, label=f"{nm} axle")
        a.axhline(0, lw=1.2, color="#333")
        a.axvline(0, lw=0.9, ls=":", color="#555")
        for h, tag, col in ((rc_f, "f", "#c1121f"), (rc_r, "r", "#0353a4")):
            a.plot([h], [sp[f"geo_pct_{tag}"]], "o", ms=9, mfc=col, mec="k",
                   mew=1.2, zorder=6)
            a.annotate(f"{sp['geo_pct_' + tag]:+.0f} %", (h, sp[f"geo_pct_{tag}"]),
                       textcoords="offset points", xytext=(7, -4), fontsize=7.5,
                       color=col, fontweight="bold")
        a.text(-310, 4, "below ground: links transfer the WRONG way",
               fontsize=6.5, color="#666", va="bottom")
        a.set_xlabel("that axle's roll-centre height  [mm]", fontsize=8.5)
        a.set_ylabel("GEOMETRIC share of that axle's transfer  [%]", fontsize=8.5)
        a.set_title("R4 · geometric share vs roll-centre height\n"
                    "the rest goes through the springs and bars", fontsize=9)
        a.legend(fontsize=7.5, loc="upper left")

        for a_ in self.ax:
            a_.grid(alpha=0.3)
            a_.tick_params(labelsize=7.5)
        self.fig.subplots_adjust(left=0.075, right=0.95, top=0.90, bottom=0.09)
        self.canvas.draw_idle()

        self.note.config(
            text=f"front RC {rc_f:+.1f}   rear RC {rc_r:+.1f} mm   "
                 f"rise {st['rise_mm']:+.1f} -> inclination {st['incl_deg']:+.2f} deg "
                 f"({'nose-down' if st['incl_deg'] > 0 else 'nose-up'})      "
                 f"axis under the CG {st['h_axis']:+.1f} mm   "
                 f"ROLL LEVER {st['lever']:.1f} mm ({st['lever_vs_cg']:+.1%} vs the "
                 f"CG height)\n"
                 f"geometric share of transfer:  FRONT {sp['geo_pct_f']:+.1f} %   "
                 f"REAR {sp['geo_pct_r']:+.1f} %      front LLTD {sp['lltd_f']:.2f} %")





class AntiGeometryPanel(ttk.Frame):
    """Anti-dive, anti-lift and anti-squat -- the LONGITUDINAL half of the car.

    Everything else in this tab is about cornering. This one is about braking
    and traction: how much of the longitudinal load transfer goes through the
    LINKS instead of the springs, and therefore how much the car pitches.

    The CAD has all eight inner pickups at a single height per wishbone, so the
    car as drawn has NO wishbone anti-geometry at either end. The side-view tilt
    of each hinge axis is therefore a DESIGN INPUT here, exactly as the ARB
    pickup is on the bar tab -- type an angle and the whole chain re-solves.

      Dive sub-tab   (braking, outboard calipers, reference = contact patch)
        D1  the side-view construction, both axles, with the 100 % loci
        D2  anti-dive and anti-lift vs wheel travel -- the benchmark's format
        D3  brake-bias sweep: anti-dive is PROPORTIONAL to bias
        D4  what it buys, in millimetres of pitch

      Squat sub-tab  (traction, inboard diff, reference = WHEEL CENTRE)
        S1  the same rear corner, showing why the reference point moves
        S2  anti-squat vs wheel travel
        S3  the design map -- anti-lift and anti-squat over both tilt angles
        S4  the coupling: what the tilt does to camber gain and roll centre

    Recomputing is ~2 s, so the panel is LAZY: refresh() only marks it dirty and
    the work happens when the tab is actually looked at.
    """

    IN = (("bias", "front brake bias [%]", 62.0),
          ("fu", "FRONT upper tilt [deg]", 0.0),
          ("fl", "FRONT lower tilt [deg]", 0.0),
          ("ru", "REAR upper tilt [deg]", 0.0),
          ("rl", "REAR lower tilt [deg]", 0.0),
          ("gb", "braking [g]", 1.2),
          ("ga", "traction [g]", 1.0))

    def __init__(self, master, page):
        super().__init__(master)
        self.page = page
        self._dirty = True
        self._key = None

        top = ttk.Frame(self, padding=(6, 4))
        top.pack(side="top", fill="x")
        r1 = ttk.Frame(top)
        r1.pack(fill="x")
        self.v = {}
        for key, lab, val in self.IN:
            ttk.Label(r1, text=lab, font=("Consolas", 8)).pack(side="left",
                                                               padx=(6, 2))
            sv = tk.StringVar(value=f"{val:g}")
            e = ttk.Entry(r1, textvariable=sv, width=7, font=("Consolas", 8),
                          justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self._force())
            self.v[key] = sv
        ttk.Button(r1, text="Apply", command=self._force).pack(side="left",
                                                               padx=(8, 2))
        ttk.Button(r1, text="Solve for benchmark",
                   command=self._solve).pack(side="left", padx=2)
        ttk.Button(r1, text="Zero", command=self._zero).pack(side="left", padx=2)

        self.note = ttk.Label(top, text="", font=("Consolas", 8),
                              foreground="#555", justify="left")
        self.note.pack(anchor="w", pady=(3, 0))
        ttk.Label(top, text="the CAD has every inner pickup at ONE height per "
                            "wishbone, so the car as drawn has no wishbone anti "
                            "at all - these angles are a DESIGN INPUT",
                  font=("Segoe UI", 8), foreground="#888").pack(anchor="w")

        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True)
        dive = ttk.Frame(self.nb)
        squat = ttk.Frame(self.nb)
        self.nb.add(dive, text="  Dive  (braking)  ")
        self.nb.add(squat, text="  Squat  (traction)  ")

        self.dtab = self._make_table(dive, "BRAKING")
        self.dfig = Figure(figsize=(11.5, 5.6), dpi=100)
        gsd = self.dfig.add_gridspec(2, 2, hspace=0.48, wspace=0.28)
        self.dax = [self.dfig.add_subplot(gsd[i, j])
                    for i in (0, 1) for j in (0, 1)]
        self.dcan = FigureCanvasTkAgg(self.dfig, master=dive)
        self.dcan.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.dcan, dive).update()

        self.stab = self._make_table(squat, "TRACTION", rows=5)
        self.sfig = Figure(figsize=(11.5, 5.6), dpi=100)
        gss = self.sfig.add_gridspec(2, 2, hspace=0.48, wspace=0.28)
        self.sax = [self.sfig.add_subplot(gss[i, j])
                    for i in (0, 1) for j in (0, 1)]
        self.scan = FigureCanvasTkAgg(self.sfig, master=squat)
        self.scan.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.scan, squat).update()
        self._scb = None

        pts_tab = ttk.Frame(self.nb)
        self.nb.add(pts_tab, text="  Hard points  ")
        self._build_points(pts_tab)

        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._maybe())

    # ------------------------------------------------------- hard-point tab
    ARMS = (("FRONT", "upper", "fu"), ("FRONT", "lower", "fl"),
            ("REAR", "upper", "ru"), ("REAR", "lower", "rl"))

    def _grid(self, parent, cap, cols, nrows, sub=""):
        """One read-only table. Returns its list of row-lists of Labels."""
        box = ttk.Frame(parent, padding=(8, 4))
        box.pack(side="top", fill="x", anchor="w")
        tk.Label(box, text=cap, font=("Segoe UI", 9, "bold"),
                 fg="#222").grid(row=0, column=0, columnspan=len(cols),
                                 sticky="w")
        if sub:
            tk.Label(box, text=sub, font=("Segoe UI", 7), fg="#888",
                     justify="left").grid(row=1, column=0, columnspan=len(cols),
                                          sticky="w", pady=(0, 2))
        for j, (h, w) in enumerate(cols):
            tk.Label(box, text=h, font=("Segoe UI", 7, "bold"), width=w,
                     relief="groove", bg="#dcdcdc").grid(row=2, column=j,
                                                         sticky="ew")
        out = []
        for i in range(nrows):
            rw = []
            for j, (_h, w) in enumerate(cols):
                lb = tk.Label(box, text="", width=w,
                              anchor="w" if j < 3 else "e",
                              font=("Consolas", 8), relief="solid",
                              borderwidth=1,
                              bg="#ffffff" if i % 2 == 0 else "#eef2f6")
                lb.grid(row=3 + i, column=j, sticky="ew")
                rw.append(lb)
            out.append(rw)
        return out

    def _build_points(self, parent):
        """The hard points the tilt boxes actually move, and the reference
        points the anti percentages are actually measured from.

        Mirrors the coordinate block in Soham's F1 sheet, with two additions
        that sheet does not have: the DESIGNED Z next to the CAD Z (so the tilt
        angle turns back into numbers you can type into CAD), and the reference
        points, which are what the physics uses and are not obtainable from the
        inner pivots alone.

        NOTE THE FRAME. This project uses X lateral / Y longitudinal / Z up;
        Soham's F1 sheet uses x longitudinal / y lateral / z up. Same points,
        first two columns swapped. Columns here are labelled explicitly so the
        two can be compared without a transcription error.
        """
        ttk.Label(parent, text="read-only - hard points are edited on the FRONT "
                               "and REAR tabs; the four tilt boxes above move "
                               "only the Z of the inner pivots",
                  font=("Segoe UI", 8), foreground="#888").pack(anchor="w",
                                                                padx=8, pady=(6, 0))
        self.pt_rows = self._grid(
            parent, "Inner pivots  -  the eight points the tilt boxes move",
            [("axle", 8), ("arm", 8), ("leg", 8), ("X lateral", 12),
             ("Y longitud.", 12), ("Z  CAD", 11), ("Z  designed", 12),
             ("change", 10)], 8,
            sub="In the F1 sheet's frame these are (x, y, z) = (Y longitudinal, "
                "X lateral, Z) - the first two columns are swapped.")
        self.ax_rows = self._grid(
            parent, "Side-view axis of each wishbone  -  what those Z values add up to",
            [("axle", 8), ("arm", 8), ("", 8), ("span in Y", 12),
             ("dZ  CAD", 12), ("dZ designed", 11), ("tilt CAD", 12),
             ("tilt designed", 14)], 4,
            sub="tilt = atan(dZ / span).  POSITIVE = the axis rises going "
                "rearward, i.e. the front pivot is the low one.")
        self.rf_rows = self._grid(
            parent, "Reference points  -  where the anti is actually measured from",
            [("axle", 8), ("point", 18), ("used when", 20), ("Y longitud.", 12),
             ("Z", 11), ("tau [mm/mm]", 13), ("-> anti [%]", 12)], 4,
            sub="tau is the fore-aft motion of THAT point per mm of bump. The "
                "F1 sheet calls these tan(theta) (patch) and tan(psi) (wheel "
                "centre), measured by hand in CAD; here they are solved.")

    # ------------------------------------------------------------- plumbing
    def _make_table(self, parent, cap, rows=4):
        host = ttk.Frame(parent, padding=(6, 2))
        host.pack(side="bottom", fill="x")
        tk.Label(host, text=cap, font=("Segoe UI", 8, "bold"),
                 fg="#333").grid(row=0, column=0, columnspan=6, sticky="w")
        cols = [("", 26), ("tau\n[mm/mm]", 11), ("wishbone\n[%]", 11),
                ("with bump\nsteer [%]", 12), ("benchmark\n[%]", 11),
                ("verdict", 42)]
        for j, (h, w) in enumerate(cols):
            tk.Label(host, text=h, font=("Segoe UI", 7, "bold"), width=w,
                     relief="groove", bg="#dcdcdc").grid(row=1, column=j,
                                                         sticky="ew")
        cells = []
        for i in range(rows):
            rw = []
            for j, (_h, w) in enumerate(cols):
                lb = tk.Label(host, text="", width=w,
                              anchor="w" if j in (0, 5) else "e",
                              font=("Consolas", 8), relief="solid",
                              borderwidth=1,
                              bg="#ffffff" if i % 2 == 0 else "#eef2f6")
                lb.grid(row=2 + i, column=j, sticky="ew")
                rw.append(lb)
            cells.append(rw)
        return cells

    @staticmethod
    def _svsa_txt(v):
        """A swing arm of 80 million mm is a parallel-arm suspension, not a
        number anybody wants to read."""
        return "effectively infinite" if (not np.isfinite(v) or v > 2e4)             else f"{v:,.0f} mm"

    def _num(self, k, d):
        try:
            return float(self.v[k].get())
        except ValueError:
            return d

    def _force(self):
        self._key = None
        self._dirty = True
        self._maybe()

    def _zero(self):
        for k in ("fu", "fl", "ru", "rl"):
            self.v[k].set("0")
        self._force()

    def refresh(self):
        """Called by the page. Cheap on purpose -- see _maybe()."""
        self._dirty = True
        self._maybe()

    def _maybe(self):
        """Do the work only if this panel is actually on screen and something
        it depends on has changed. Two full bump sweeps plus a 7x7 design map
        is a couple of seconds; paying that on every hard-point edit while the
        user is looking at a different tab would make the whole tool feel slow.
        """
        if not self._dirty or not self.winfo_ismapped():
            return
        got = self.page.get_pts() if self.page.get_pts else None
        car = getattr(self.page, "car", None)
        if got is None or car is None:
            return
        key = (tuple(round(self._num(k, d), 6) for k, _l, d in self.IN),
               tuple(round(float(x), 4) for p in (got[0], got[2])
                     for v in p.values() for x in v),
               round(car["wheelbase_mm"], 3), round(car["h_cg_mm"], 3),
               round(car["kw_f"], 3), round(car["kw_r"], 3))
        if key == self._key:
            self._dirty = False
            return
        self._key = key
        self._dirty = False
        try:
            self._compute(got, car)
        except Exception as exc:                              # noqa: BLE001
            self.note.config(text=f"anti geometry unavailable: {exc}")

    # -------------------------------------------------------------- solving
    def _solve(self):
        """Fill the tilt boxes with the angles that reach the benchmark.

        Front: one target (anti-dive), so equal tilts on both arms are enough.
        Rear: TWO targets off ONE geometry, which parallel arms cannot do -- see
        anti_geometry.patch_wc_gap -- so the pair solver splits the axes.
        """
        got = self.page.get_pts() if self.page.get_pts else None
        car = getattr(self.page, "car", None)
        if got is None or car is None:
            return
        pf, cf, pr, cr = got
        bias = self._num("bias", 62.0) / 100.0
        B = ag.BENCHMARK_PCT
        try:
            a, _ = ag.required_tilt(pf, cf, B["anti_dive_f"], car, "dive", bias)
            if a == a:
                self.v["fu"].set(f"{a:.3f}")
                self.v["fl"].set(f"{a:.3f}")
            u, l, _gl, _gs = ag.required_tilt_pair(pr, cr, B["anti_lift_r"],
                                                   B["anti_squat_r"], car, bias)
            if u == u:
                self.v["ru"].set(f"{u:.3f}")
                self.v["rl"].set(f"{l:.3f}")
        except Exception as exc:                              # noqa: BLE001
            self.note.config(text=f"solver failed: {exc}")
            return
        self._force()

    # ------------------------------------------------------------ the work
    def _compute(self, got, car):
        pf, cf, pr, cr = got
        bias = self._num("bias", 62.0) / 100.0
        gf = corner_geometry(ag.tilt_axes(pf, self._num("fu", 0.0),
                                          self._num("fl", 0.0)), cf)
        gr = corner_geometry(ag.tilt_axes(pr, self._num("ru", 0.0),
                                          self._num("rl", 0.0)), cr)
        tf, tr = ag.tau(gf), ag.tau(gr)
        B = ag.BENCHMARK_PCT

        vals = dict(
            dive=ag.anti_dive_pct(tf["patch"], car, bias),
            dive_w=ag.anti_dive_pct(tf["patch_wishbone"], car, bias),
            lift=ag.anti_lift_pct(tr["patch"], car, bias),
            lift_w=ag.anti_lift_pct(tr["patch_wishbone"], car, bias),
            squat=ag.anti_squat_pct(tr["wc"], car),
            squat_w=ag.anti_squat_pct(tr["wc_wishbone"], car))

        self._draw_dive(gf, gr, tf, tr, car, bias, vals, B)
        self._draw_squat(pr, cr, gr, tr, car, bias, vals, B)
        self._fill_points(pf, pr, gf, gr, tf, tr, car, bias, vals)

        sv_f, sv_r = ag.svsa_mm(tf), ag.svsa_mm(tr)
        self.note.config(
            text=f"FRONT axes  upper {ag.axis_tilt_deg(pf)[0]:+.2f} -> "
                 f"{self._num('fu', 0):+.2f} deg   lower "
                 f"{ag.axis_tilt_deg(pf)[1]:+.2f} -> {self._num('fl', 0):+.2f} "
                 f"deg    side-view swing arm {self._svsa_txt(sv_f)}     bump steer "
                 f"{tf['bump_steer']:+.4f} deg/mm through a "
                 f"{tf['steer_lever_patch']:.0f} mm lever\n"
                 f"REAR  axes  upper {ag.axis_tilt_deg(pr)[0]:+.2f} -> "
                 f"{self._num('ru', 0):+.2f} deg   lower "
                 f"{ag.axis_tilt_deg(pr)[1]:+.2f} -> {self._num('rl', 0):+.2f} "
                 f"deg    side-view swing arm {self._svsa_txt(sv_r)}     bump steer "
                 f"{tr['bump_steer']:+.4f} deg/mm through a "
                 f"{tr['steer_lever_patch']:.0f} mm lever")

    def _fill_points(self, pf, pr, gf, gr, tf, tr, car, bias, vals):
        """Numbers for the hard-point tab."""
        des = {"FRONT": ag.tilt_axes(pf, self._num("fu", 0.0),
                                     self._num("fl", 0.0)),
               "REAR": ag.tilt_axes(pr, self._num("ru", 0.0),
                                    self._num("rl", 0.0))}
        cad = {"FRONT": pf, "REAR": pr}
        i = 0
        for axle, arm, key in self.ARMS:
            names = ag._AXES[0] if arm == "upper" else ag._AXES[1]
            for leg, nm in zip(("front", "rear"), names):
                c, d = cad[axle][nm], des[axle][nm]
                for j, txt in enumerate((axle if leg == "front" else "",
                                         arm if leg == "front" else "", leg,
                                         f"{c[0]:.2f}", f"{c[1]:.2f}",
                                         f"{c[2]:.2f}", f"{d[2]:.2f}",
                                         f"{d[2] - c[2]:+.2f}")):
                    self.pt_rows[i][j].config(text=txt)
                i += 1
        for i, (axle, arm, key) in enumerate(self.ARMS):
            nf, nr = ag._AXES[0] if arm == "upper" else ag._AXES[1]
            span = -(cad[axle][nr][1] - cad[axle][nf][1])
            dz_c = cad[axle][nr][2] - cad[axle][nf][2]
            dz_d = des[axle][nr][2] - des[axle][nf][2]
            t_c = ag.axis_tilt_deg(cad[axle])[0 if arm == "upper" else 1]
            for j, txt in enumerate((axle, arm, "", f"{span:.2f}",
                                     f"{dz_c:+.3f}", f"{dz_d:+.3f}",
                                     f"{t_c:+.4f} deg",
                                     f"{self._num(key, 0.0):+.4f} deg")):
                self.ax_rows[i][j].config(text=txt)
        rows = (("FRONT", "contact patch", "braking (outboard caliper)",
                 tf["patch_y"], 0.0, tf["patch"], vals["dive"]),
                ("FRONT", "wheel centre", "nothing - RWD, no drive force",
                 tf["wc_y"], tf["wc_z"], tf["wc"], 0.0),
                ("REAR", "contact patch", "braking (outboard caliper)",
                 tr["patch_y"], 0.0, tr["patch"], vals["lift"]),
                ("REAR", "wheel centre", "traction (diff on the chassis)",
                 tr["wc_y"], tr["wc_z"], tr["wc"], vals["squat"]))
        for i, (axle, nm, use, y, z, tv, pct) in enumerate(rows):
            for j, txt in enumerate((axle, nm, use, f"{y:.2f}", f"{z:.2f}",
                                     f"{tv:+.5f}", f"{pct:+.2f}")):
                self.rf_rows[i][j].config(text=txt)

    # --------------------------------------------------------- DIVE sub-tab
    def _draw_dive(self, gf, gr, tf, tr, car, bias, vals, B):
        for a in self.dax:
            a.clear()
        L, h = car["wheelbase_mm"], car["h_cg_mm"]

        # ---- D1  the side-view construction --------------------------------
        # Gillespie's Fig 7.13, drawn for THIS car. Each axle's force line
        # starts at its own contact patch and runs toward the other axle; where
        # it crosses that axle's vertical, measured against h/bias, IS the anti
        # percentage. No wheels drawn -- an equal-aspect wheel is unreadable at
        # a 4 m by 1 m scale, and the construction never needed one.
        a = self.dax[0]
        yf, yr = tf["patch_y"], tr["patch_y"]
        # Ghost the real body in first, under everything. It is scaled onto the
        # LIVE axle stations, so its wheels always land on the plotted contact
        # patches -- see car_outline for why that matters.
        body = car_outline.draw(a, yf, yr)
        a.axhline(0, lw=1.6, color="#333", zorder=1)
        a.axvline(yf, lw=0.9, ls=":", color="#999")
        a.axvline(yr, lw=0.9, ls=":", color="#999")
        x_cg = yf - (1.0 - car["rear_frac"]) * L
        a.plot([x_cg], [h], "*", ms=15, color="#111", zorder=7)
        a.annotate(f"CG  {h:.0f} mm", (x_cg, h), textcoords="offset points",
                   xytext=(0, 12), fontsize=7.5, ha="center",
                   bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff",
                             ec="none", alpha=0.82), zorder=8)
        full_f = h / max(bias, 1e-6)
        full_r = h / max(1.0 - bias, 1e-6)
        # `at` is where each line carries its own label, as a fraction along
        # from its own patch: the two are deliberately different so the labels
        # separate in BOTH axes instead of meeting in the middle.
        for y, tv, col, sgn, full, nm, pct, at in (
                (yf, tf["patch_wishbone"], "#c1121f", -1.0, full_f,
                 "anti-dive", vals["dive_w"], 0.22),
                (yr, tr["patch_wishbone"], "#0353a4", +1.0, full_r,
                 "anti-lift", vals["lift_w"], 0.30)):
            a.plot([y], [0], "o", ms=7, mfc=col, mec="k", mew=1.0, zorder=6)
            a.plot([y, y + sgn * L], [0, full], ls="--", lw=1.2, color=col,
                   alpha=0.55)
            a.plot([y + sgn * L, y], [-sgn * L * tv, 0], lw=2.6, color=col,
                   zorder=5)
            # Label each force line ON the line, a quarter of the way along
            # from its OWN patch. Over a car silhouette there is no corner a
            # legend can sit in without covering a patch or a locus, and the
            # far end of each line runs into the CG marker.
            a.annotate(f"{nm}  {pct:+.1f} %",
                       (y + sgn * at * L, -sgn * at * L * tv),
                       textcoords="offset points", xytext=(0, 9),
                       fontsize=8, fontweight="bold", color=col,
                       ha="center", zorder=6)
            a.annotate(f"100 %\n{full:.0f} mm", (y + sgn * L, full),
                       textcoords="offset points", xytext=(-8 * sgn, 2),
                       fontsize=6.5, color=col,
                       ha="right" if sgn > 0 else "left")
        if body is not None:
            lo_y, hi_y, top = body
            a.set_xlim(hi_y + 70, lo_y - 70)
            a.set_ylim(-110, max(full_f, full_r, top) * 1.10)
        else:
            a.set_xlim(max(yf, yr) + 0.14 * L, min(yf, yr) - 0.14 * L)
            a.set_ylim(-90, max(full_f, full_r) * 1.30)
        a.set_xlabel("longitudinal  [mm]   (nose to the LEFT)", fontsize=8.5)
        a.set_ylabel("height  [mm]", fontsize=8.5)
        a.set_title("D1 - the side-view construction\nsolid = this car's force "
                    "line;  dashed = where 100 % anti would put it", fontsize=9)


        # ---- D2  vs wheel travel, the benchmark's own format ---------------
        a = self.dax[1]
        grid = np.linspace(-25, 25, 13)
        try:
            swf, swr = ag.sweep(gf, grid), ag.sweep(gr, grid)
        except Exception:                                     # noqa: BLE001
            a.text(0.5, 0.5, "travel beyond the solvable range", ha="center",
                   va="center", color="#888", transform=a.transAxes)
        else:
            a.plot(grid, ag.anti_dive_pct(swf["patch"], car, bias), lw=2.2,
                   color="#c1121f", label="front anti-dive")
            a.plot(grid, ag.anti_lift_pct(swr["patch"], car, bias), lw=2.2,
                   color="#0353a4", label="rear anti-lift")
            a.plot(grid, ag.anti_dive_pct(swf["patch_wishbone"], car, bias),
                   lw=1.3, ls=":", color="#c1121f", label="front, wishbones only")
            a.plot(grid, ag.anti_lift_pct(swr["patch_wishbone"], car, bias),
                   lw=1.3, ls=":", color="#0353a4", label="rear, wishbones only")
        a.axhline(B["anti_dive_f"], lw=1.1, ls="--", color="#7a5195")
        a.axhline(B["anti_lift_r"], lw=1.1, ls="--", color="#7a5195")
        a.text(25, B["anti_lift_r"], " benchmark", fontsize=6.5, color="#7a5195",
               va="bottom", ha="right")
        a.axhline(0, lw=1.0, color="#333")
        a.axhspan(0, ag.GILLESPIE_MAX_ANTIDIVE_PCT, color="#2a9d8f", alpha=0.07)
        a.set_xlabel("wheel travel  [mm]   (+ = bump)", fontsize=8.5)
        a.set_ylabel("anti  [%]", fontsize=8.5)
        a.set_title("D2 - anti vs wheel travel\nshaded: 0 to Gillespie's 50 % "
                    "practical ceiling", fontsize=9)
        a.legend(fontsize=6.5, loc="best")

        # ---- D3  brake-bias sweep ------------------------------------------
        a = self.dax[2]
        bb = np.linspace(0.40, 0.85, 61)
        a.plot(bb * 100, ag.anti_dive_pct(tf["patch"], car, bb), lw=2.2,
               color="#c1121f", label="front anti-dive")
        a.plot(bb * 100, ag.anti_lift_pct(tr["patch"], car, bb), lw=2.2,
               color="#0353a4", label="rear anti-lift")
        a.axvline(bias * 100, lw=1.2, ls="--", color="#333")
        for val, col in ((vals["dive"], "#c1121f"), (vals["lift"], "#0353a4")):
            a.plot([bias * 100], [val], "o", ms=8, mfc=col, mec="k", mew=1.1,
                   zorder=6)
            a.annotate(f"{val:+.1f} %", (bias * 100, val),
                       textcoords="offset points", xytext=(8, 0), fontsize=7.5,
                       color=col, fontweight="bold")
        a.axhline(0, lw=1.0, color="#333")
        a.set_xlabel("front brake bias  [%]", fontsize=8.5)
        a.set_ylabel("anti  [%]", fontsize=8.5)
        a.set_title("D3 - anti is PROPORTIONAL to brake bias\nmoving the bias "
                    "bar retunes the pitch behaviour too", fontsize=9)
        a.legend(fontsize=7, loc="best")

        # ---- D4  what it buys, in millimetres ------------------------------
        # A sweep here is redundant: with the rear held, pitch is an affine
        # function of front travel, so the two curves lie exactly on top of one
        # another. Three concrete cases side by side say the same thing without
        # the duplication, and put the benchmark next to it for scale.
        a = self.dax[3]
        gb = self._num("gb", 1.2)
        cases = (("no anti", 0.0, 0.0, "#999999"),
                 ("ours", vals["dive"], vals["lift"], "#c1121f"),
                 ("benchmark", B["anti_dive_f"], B["anti_lift_r"], "#7a5195"))
        xs = np.arange(len(cases))
        wdt = 0.36
        fr = [ag.pitch(car, d, l, gb)["front_mm"] for _n, d, l, _c in cases]
        re = [ag.pitch(car, d, l, gb)["rear_mm"] for _n, d, l, _c in cases]
        pi = [ag.pitch(car, d, l, gb)["pitch_deg"] for _n, d, l, _c in cases]
        a.bar(xs - wdt / 2, fr, wdt, color="#c1121f", alpha=0.85,
              label="front dive")
        a.bar(xs + wdt / 2, re, wdt, color="#0353a4", alpha=0.85,
              label="rear lift")
        for i, (v1, v2, pv) in enumerate(zip(fr, re, pi)):
            a.text(i - wdt / 2, v1 + 0.4, f"{v1:.1f}", ha="center",
                   fontsize=7, color="#c1121f")
            a.text(i + wdt / 2, v2 + 0.4, f"{v2:.1f}", ha="center",
                   fontsize=7, color="#0353a4")
            a.text(i, max(v1, v2) + 2.4, f"pitch {pv:.3f} deg", ha="center",
                   fontsize=7.5, fontweight="bold", color="#111")
        a.set_xticks(xs)
        a.set_xticklabels([f"{n}\n{d:+.0f} / {l:+.0f} %"
                           for n, d, l, _c in cases], fontsize=7.5)
        a.set_ylim(0, max(max(fr), max(re)) * 1.30)
        a.set_ylabel("suspension travel  [mm]", fontsize=8.5)
        a.set_title(f"D4 - what it buys, braking at {gb:.2f} g\n"
                    f"labels are the front / rear anti used", fontsize=9)
        a.legend(fontsize=7, loc="upper right")

        for a_ in self.dax:
            a_.grid(alpha=0.3)
            a_.tick_params(labelsize=7.5)
        self.dfig.subplots_adjust(left=0.075, right=0.94, top=0.90, bottom=0.09)
        self.dcan.draw_idle()

        rows = (("FRONT anti-dive", tf["patch"], vals["dive_w"], vals["dive"],
                 B["anti_dive_f"]),
                ("REAR anti-lift", tr["patch"], vals["lift_w"], vals["lift"],
                 B["anti_lift_r"]))
        for i, (nm, tv, w, tot, bm) in enumerate(rows):
            for j, txt in enumerate((nm, f"{tv:+.5f}", f"{w:+.2f}",
                                     f"{tot:+.2f}", f"{bm:+.1f}",
                                     ag.classify(w))):
                self.dtab[i][j].config(text=txt)
        pnow = ag.pitch(car, vals["dive"], vals["lift"], self._num("gb", 1.2))
        pno = ag.pitch(car, 0.0, 0.0, self._num("gb", 1.2))
        for j, txt in enumerate(("pitch at the braking g", "",
                                 f"{pno['pitch_deg']:.3f}",
                                 f"{pnow['pitch_deg']:.3f}", "",
                                 f"deg: no anti -> ours   "
                                 f"({pnow['pitch_deg'] / pno['pitch_deg'] - 1:+.1%})")):
            self.dtab[2][j].config(text=txt)
        for j, txt in enumerate(("front dive / rear lift", "",
                                 f"{pno['front_mm']:.2f}",
                                 f"{pnow['front_mm']:.2f}", "",
                                 f"mm at the front; rear {pno['rear_mm']:.2f} "
                                 f"-> {pnow['rear_mm']:.2f} mm")):
            self.dtab[3][j].config(text=txt)

    # -------------------------------------------------------- SQUAT sub-tab
    def _draw_squat(self, pr, cr, gr, tr, car, bias, vals, B):
        for a in self.sax:
            a.clear()
        L, h = car["wheelbase_mm"], car["h_cg_mm"]
        # solved up front: S3 marks the crossing point, and the table repeats it
        u, l, gl_, gs_ = ag.required_tilt_pair(pr, cr, B["anti_lift_r"],
                                               B["anti_squat_r"], car, bias)

        # ---- S1  where each anti is measured FROM ---------------------------
        # The panel answers ONE question: why does the same rear suspension
        # give two different anti numbers? Because braking and traction pick
        # up the force at points one wheel radius apart.
        #
        # It is drawn as a MEASURING device. x runs from the rear contact patch
        # to the front axle; each force line climbs across that span, and how
        # high it gets at the right-hand edge IS the percentage -- the dashed
        # line of the same colour marks where 100 % would land. So the reader
        # compares two heights instead of trusting a number in a legend.
        a = self.sax[0]
        R, wz = gr["R_wheel"], tr["wc_z"]
        full_l = h / max(1.0 - bias, 1e-6)      # 100 % anti-LIFT, from the patch
        full_s = wz + h                          # 100 % anti-SQUAT, from the WC
        end_l = -L * tr["patch_wishbone"]
        end_s = wz - L * tr["wc_wishbone"]
        a.axhline(0, lw=1.6, color="#333", zorder=2)
        a.axvline(L, lw=1.3, color="#555", zorder=2)
        a.text(L, -150, "front axle", fontsize=7.5, color="#555", ha="center")

        for full, end, col, nm, pct in (
                (full_l, end_l, "#c1121f", "BRAKING", vals["lift_w"]),
                (full_s, end_s, "#2a9d8f", "TRACTION", vals["squat_w"])):
            a.plot([L, L], [0 if col == "#c1121f" else wz, full], lw=6,
                   color=col, alpha=0.13, solid_capstyle="butt", zorder=1)
            a.plot([L - 0.022 * L, L + 0.022 * L], [full, full], lw=1.8,
                   color=col, alpha=0.75, zorder=4)
            a.annotate(f"100 %", (L + 0.028 * L, full), fontsize=7,
                       color=col, va="center", ha="left")
            a.plot([L], [end], "o", ms=8, mfc=col, mec="k", mew=1.1, zorder=7)
            a.annotate(f"{pct:+.1f} %", (L + 0.028 * L, end), fontsize=9,
                       color=col, fontweight="bold", va="center", ha="left")

        a.plot([0, L], [0, full_l], ls="--", lw=1.2, color="#c1121f", alpha=0.45)
        a.plot([0, L], [wz, full_s], ls="--", lw=1.2, color="#2a9d8f", alpha=0.45)
        a.plot([0, L], [0, end_l], lw=2.6, color="#c1121f", zorder=5,
               label=f"BRAKING - measured from the CONTACT PATCH")
        a.plot([0, L], [wz, end_s], lw=2.6, color="#2a9d8f", zorder=5,
               label=f"TRACTION - measured from the WHEEL CENTRE")

        # the wheel-radius gap, which is the entire reason the two differ
        a.plot([0], [0], "o", ms=10, mfc="#c1121f", mec="k", mew=1.2, zorder=8)
        a.plot([0], [wz], "o", ms=10, mfc="#2a9d8f", mec="k", mew=1.2, zorder=8)
        a.annotate("", (0, 0), (0, wz), zorder=6,
                   arrowprops=dict(arrowstyle="<->", lw=1.8, color="#7a5195"))
        a.annotate(f"R = {R:.0f} mm\nthe two start points are\none wheel radius "
                   f"apart -\nthat gap is the WHOLE\ndifference between them",
                   (0.035 * L, wz * 0.52), fontsize=7, color="#5b3b7a",
                   va="center", ha="left", zorder=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#e0d5ea"))
        a.set_xlim(-0.05 * L, 1.30 * L)
        a.set_ylim(-200, max(full_l, full_s) * 1.16)
        a.set_xlabel("distance ahead of the rear contact patch  [mm]",
                     fontsize=8.5)
        a.set_ylabel("height of the force line  [mm]", fontsize=8.5)
        a.set_title("S1 - where each anti is measured FROM\n"
                    "how high each line climbs by the front axle IS its "
                    "percentage", fontsize=9)
        a.legend(fontsize=6.8, loc="upper left", framealpha=0.93)

        # ---- S2  anti-squat vs wheel travel --------------------------------
        a = self.sax[1]
        grid = np.linspace(-25, 25, 13)
        try:
            sw = ag.sweep(gr, grid)
        except Exception:                                     # noqa: BLE001
            a.text(0.5, 0.5, "travel beyond the solvable range", ha="center",
                   va="center", color="#888", transform=a.transAxes)
        else:
            a.plot(grid, ag.anti_squat_pct(sw["wc"], car), lw=2.2,
                   color="#2a9d8f", label="anti-squat")
            a.plot(grid, ag.anti_squat_pct(sw["wc_wishbone"], car), lw=1.4,
                   ls=":", color="#2a9d8f", label="wishbones only")
            a.plot(grid, ag.anti_lift_pct(sw["patch"], car, bias), lw=1.6,
                   color="#0353a4", alpha=0.6,
                   label="anti-lift (same geometry)")
        a.axhline(B["anti_squat_r"], lw=1.1, ls="--", color="#7a5195",
                  label="benchmark anti-squat")
        a.axhline(0, lw=1.0, color="#333")
        a.set_xlabel("wheel travel  [mm]   (+ = bump)", fontsize=8.5)
        a.set_ylabel("anti  [%]", fontsize=8.5)
        a.set_title("S2 - anti-squat vs wheel travel\nthe benchmark runs it at "
                    "essentially ZERO", fontsize=9)
        a.legend(fontsize=6.5, loc="best")

        # ---- S3  the design map --------------------------------------------
        # Two targets, two knobs, one picture. Everywhere on this map is a
        # possible pair of axis tilts; the colour says what anti-LIFT you would
        # get there and the green lines say what anti-SQUAT. The benchmark wants
        # a particular value of each, so it is two contours -- and the ONE place
        # they cross is the only tilt pair that satisfies both.
        a = self.sax[2]
        n = 7
        gu = np.linspace(-6, 6, n)
        gl = np.linspace(-6, 6, n)
        LI = np.zeros((n, n))
        SQ = np.zeros((n, n))
        for ii, lo in enumerate(gl):
            for jj, up in enumerate(gu):
                t = ag.tau(corner_geometry(ag.tilt_axes(pr, up, lo), cr))
                LI[ii, jj] = ag.anti_lift_pct(t["patch_wishbone"], car, bias)
                SQ[ii, jj] = ag.anti_squat_pct(t["wc_wishbone"], car)
        cf_ = a.contourf(gu, gl, LI, levels=16, cmap="Blues")
        if self._scb is None:
            self._scb = self.sfig.colorbar(cf_, ax=a, fraction=0.05, pad=0.02)
        else:
            self._scb.update_normal(cf_)
        self._scb.set_label("background shade = anti-LIFT  [%]", fontsize=7.5)
        # force SOLID: matplotlib dashes negative contours by default, which
        # would collide with the dashed purple benchmark line
        cs = a.contour(gu, gl, SQ, levels=8, colors="#1f7a6e", linewidths=1.0,
                       alpha=0.85, linestyles="solid")
        a.clabel(cs, fontsize=7, fmt="%.0f")
        a.contour(gu, gl, LI, levels=[B["anti_lift_r"]], colors="#7a5195",
                  linewidths=2.4)
        a.contour(gu, gl, SQ, levels=[B["anti_squat_r"]], colors="#7a5195",
                  linewidths=2.4, linestyles="--")
        a.plot([-6, 6], [-6, 6], lw=1.0, ls=":", color="#555", alpha=0.7)
        a.text(5.7, 5.2, "equal tilts", fontsize=6.5, color="#555", rotation=42,
               ha="right")
        # where the two benchmark contours cross -- the answer the solver finds
        if u == u:
            a.plot([u], [l], "*", ms=17, mfc="#7a5195", mec="w", mew=1.2,
                   zorder=8)
            a.annotate("target", (u, l), textcoords="offset points",
                       xytext=(9, -12), fontsize=8, fontweight="bold",
                       color="#5b3b7a", zorder=8)
        a.plot([self._num("ru", 0.0)], [self._num("rl", 0.0)], "o", ms=10,
               mfc="yellow", mec="k", mew=1.5, zorder=9)
        # a real legend -- the four things on the map, named
        a.plot([], [], color="#7a5195", lw=2.4,
               label=f"anti-LIFT = {B['anti_lift_r']:.0f} % (benchmark)")
        a.plot([], [], color="#7a5195", lw=2.4, ls="--",
               label=f"anti-SQUAT = {B['anti_squat_r']:.0f} % (benchmark)")
        a.plot([], [], color="#1f7a6e", lw=1.0,
               label="anti-SQUAT contours [%]")
        a.plot([], [], "o", ms=7, mfc="yellow", mec="k", mew=1.2,
               label="your tilts", ls="none")
        a.legend(fontsize=6.3, loc="lower left", framealpha=0.92,
                 handlelength=1.7, borderpad=0.35, labelspacing=0.3)
        a.set_xlabel("UPPER axis tilt  [deg]", fontsize=8.5)
        a.set_ylabel("LOWER axis tilt  [deg]", fontsize=8.5)
        a.set_title("S3 - the rear design map\nthe two purple lines cross at the "
                    "ONLY pair that hits both targets", fontsize=9)

        # ---- S4  the coupling ----------------------------------------------
        a = self.sax[3]
        tl = np.linspace(-6, 6, 13)
        co = ag.coupling(pr, cr, tl)
        sq = ag.anti_squat_pct(co["tau_wc_wishbone"], car)
        a.plot(sq, co["camber_gain"], "-o", ms=3.5, lw=2.0, color="#c1121f",
               label="camber gain")
        a.set_xlabel("anti-squat from an equal tilt on both arms  [%]",
                     fontsize=8.5)
        a.set_ylabel("camber gain  [deg/mm]", color="#c1121f", fontsize=8.5)
        a.tick_params(axis="y", colors="#c1121f", labelsize=7.5)
        a3 = a.twinx()
        a3.plot(sq, co["rc_h"], "-s", ms=3.5, lw=2.0, color="#0353a4")
        a3.set_ylabel("roll-centre height  [mm]", color="#0353a4", fontsize=8.5)
        a3.tick_params(axis="y", colors="#0353a4", labelsize=7.5)
        a3.axhline(0, lw=1.0, ls=":", color="#0353a4", alpha=0.6)
        a.axvline(vals["squat_w"], lw=1.2, ls="--", color="#333")
        a.set_title("S4 - the COUPLING: side view and front view are ONE "
                    "linkage\nadding rear anti-squat also lifts the roll centre "
                    "and deepens camber gain", fontsize=9)
        a.legend(fontsize=7, loc="best")

        for a_ in self.sax:
            a_.grid(alpha=0.3)
            a_.tick_params(labelsize=7.5)
        self.sfig.subplots_adjust(left=0.075, right=0.93, top=0.90, bottom=0.09)
        self.scan.draw_idle()

        rows = (("REAR anti-squat", tr["wc"], vals["squat_w"], vals["squat"],
                 B["anti_squat_r"], ag.classify(vals["squat_w"])),
                ("REAR anti-lift", tr["patch"], vals["lift_w"], vals["lift"],
                 B["anti_lift_r"], ag.classify(vals["lift_w"])))
        for i, (nm, tv, w, tot, bm, verdict) in enumerate(rows):
            for j, txt in enumerate((nm, f"{tv:+.5f}", f"{w:+.2f}",
                                     f"{tot:+.2f}", f"{bm:+.1f}", verdict)):
                self.stab[i][j].config(text=txt)
        sv = ag.svsa_mm(tr)
        for j, txt in enumerate(("side-view swing arm", "", "", "", "",
                                 f"{self._svsa_txt(sv)} - short arms cause brake hop; "
                                 f"parallel arms give infinity")):
            self.stab[2][j].config(text=txt)
        msg = (f"upper {u:+.2f} deg, lower {l:+.2f} deg  ->  lift {gl_:.1f} %, "
               f"squat {gs_:.1f} %" if u == u else
               "no tilt pair inside +/-14 deg reaches both targets")
        for j, txt in enumerate(("to reach BOTH benchmarks", "", "", "", "",
                                 msg)):
            self.stab[3][j].config(text=txt)
        # What the traction-g box is for. mode="accel" ignores any front anti --
        # a RWD front carries no longitudinal force, so it has none to give.
        ga = self._num("ga", 1.0)
        pa0 = ag.pitch(car, 0.0, 0.0, ga, mode="accel")
        pa = ag.pitch(car, 0.0, vals["squat"], ga, mode="accel")
        for j, txt in enumerate((f"rear squat at {ga:.2f} g", "",
                                 f"{pa0['rear_mm']:.2f}", f"{pa['rear_mm']:.2f}",
                                 "", f"mm: no anti -> ours    nose-up pitch "
                                     f"{pa0['pitch_deg']:.3f} -> "
                                     f"{pa['pitch_deg']:.3f} deg")):
            self.stab[4][j].config(text=txt)


class MechBalancePanel(ttk.Frame):
    """Mechanical balance and roll stiffness -- the setup lookup table.

    This is the usual mechanical-balance table format, rebuilt for our
    car: for one axle, a grid of SPRING RATE (rows) against ANTI-ROLL-BAR BLADE
    PAIR (columns), printed twice -- once as the change in mechanical balance and
    once as the change in total roll stiffness, both relative to the baseline
    setup in the middle.

    WHY THE TABLE IS THE RIGHT SHAPE FOR THIS
    -----------------------------------------
    You do not choose springs and bars independently. You change springs for
    RIDE, and that change drags the balance with it whether you wanted it or
    not. The table's whole purpose is the diagonal: read across to find the bar
    move that CANCELS the balance shift your spring change caused, so you keep
    the car you had. The `Reset` sub-tab does that lookup exactly rather than by
    eye.

    TWO DEFINITIONS THAT ARE NOT INTERCHANGEABLE
    --------------------------------------------
      Mechanical Balance FA   the front axle's share of LATERAL LOAD TRANSFER
                              (LLTD), in percentage points. + = understeer.
      Roll Stiffness          the axle pair's total, with the TYRE IN SERIES.

    Balance is NOT the front roll-stiffness share: only the elastic part of the
    load transfer goes through springs at all, so a 1-point move in share moves
    balance by k ~ 0.83 points. Both definitions were recovered by reconstructing
    150 published benchmark cells and are re-asserted by
    validate_mech_balance.py; see mech_balance.py for the derivation.

    Everything here is LIVE: the springs come from the Springs tab's motion
    ratios and corner weights, the roll centres from the FRONT and REAR pages.
    """

    IN = (("dspr", "spring step [%]", 7.0),
          ("darb", "bar span +/- [%]", 30.0),
          ("nfb", "front blades", 6),
          ("nrb", "rear blades", 4),
          ("kt", "tyre rate [N/mm]", 310.4))

    # No lateral-g box: every load-transfer route is proportional to lateral g,
    # so g cancels out of LLTD exactly and the roll gradient is already per g.
    # A box that changes nothing on screen teaches the wrong thing.
    LAT_G = 1.0

    def __init__(self, master, page):
        super().__init__(master)
        self.page = page
        self._dirty = True
        self._key = None

        top = ttk.Frame(self, padding=(6, 4))
        top.pack(side="top", fill="x")
        r1 = ttk.Frame(top)
        r1.pack(fill="x")
        self.v = {}
        for key, lab, val in self.IN:
            ttk.Label(r1, text=lab, font=("Consolas", 8)).pack(side="left",
                                                               padx=(6, 2))
            sv = tk.StringVar(value=f"{val:g}")
            e = ttk.Entry(r1, textvariable=sv, width=7, font=("Consolas", 8),
                          justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self._force())
            self.v[key] = sv
        ttk.Button(r1, text="Apply", command=self._force).pack(side="left",
                                                               padx=(8, 2))

        self.note = ttk.Label(top, text="", font=("Consolas", 8),
                              foreground="#555", justify="left")
        self.note.pack(anchor="w", pady=(3, 0))
        ttk.Label(top, text="springs and motion ratios come LIVE from the "
                            "Springs tab, roll centres from the FRONT and REAR "
                            "pages - the baseline is the middle cell, and every "
                            "number is a CHANGE from it",
                  font=("Segoe UI", 8), foreground="#888").pack(anchor="w")

        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True)
        self.fig, self.ax, self.can = {}, {}, {}
        for tag, cap in (("f", "  Front axle  "), ("r", "  Rear axle  "),
                         ("x", "  Reset & trade  ")):
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text=cap)
            fig = Figure(figsize=(11.5, 5.6), dpi=100)
            if tag == "x":
                gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.28)
                ax = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1)]
            else:
                gs = fig.add_gridspec(1, 2, wspace=0.16)
                ax = [fig.add_subplot(gs[0, j]) for j in (0, 1)]
            can = FigureCanvasTkAgg(fig, master=fr)
            can.get_tk_widget().pack(side="top", fill="both", expand=True)
            NavigationToolbar2Tk(can, fr).update()
            self.fig[tag], self.ax[tag], self.can[tag] = fig, ax, can

        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._maybe())

    # ------------------------------------------------------------- plumbing
    def _num(self, k, d):
        try:
            return float(self.v[k].get())
        except ValueError:
            return d

    def _force(self):
        self._key = None
        self._dirty = True
        self._maybe()

    def refresh(self):
        self._dirty = True
        self._maybe()

    def _maybe(self):
        if not self._dirty or not self.winfo_ismapped():
            return
        car = getattr(self.page, "car", None)
        rc_f, rc_r = self.page.get_rcs()
        d = self.page.spring_design()
        if car is None or rc_f is None or rc_r is None or d is None:
            return
        key = (tuple(round(self._num(k, v), 6) for k, _l, v in self.IN),
               round(rc_f, 4), round(rc_r, 4), round(d["ks_f"], 4),
               round(d["ks_r"], 4), round(d["mr_f"], 6), round(d["mr_r"], 6),
               round(car["arb_f"], 4), round(car["arb_r"], 4),
               round(car["mass_total_kg"], 3), round(car["h_cg_mm"], 3))
        if key == self._key:
            self._dirty = False
            return
        self._key = key
        self._dirty = False
        try:
            self._compute(car, rc_f, rc_r, d)
        except Exception as exc:                              # noqa: BLE001
            self.note.config(text=f"mechanical balance unavailable: {exc}")

    # ---------------------------------------------------------- the grids
    def _grids(self, base_spring, base_arb, n_blades):
        """Spring rows and ARB columns, built so the MIDDLE cell is the baseline.

        The bar columns are blade PAIRS, so n detents give 2n-1 of them (see
        mech_balance.pair_rates). The whole pair set is then scaled so the
        centre column lands exactly on the typed baseline -- otherwise the table
        would not read 0.0 anywhere and the reference would be a fiction.
        """
        step = self._num("dspr", 7.0) / 100.0
        rows = np.array([base_spring * (1.0 + step * i) for i in (-2, -1, 0, 1, 2, 3)])
        n = max(2, int(round(n_blades)))
        span = self._num("darb", 30.0) / 100.0
        full = np.linspace(1.0 - span, 1.0 + span, n)
        cols = mbal.pair_rates(full)
        cols = cols * (base_arb / cols[n - 1])
        return rows, cols, n

    @staticmethod
    def _heat(a, M, rows, labels, title, unit, cmap):
        """One balance table: numbers in cells, colour behind them.

        Colour is symmetric about zero on purpose -- the baseline must read as
        the neutral shade, or the eye reads the wrong cell as 'normal'.
        """
        v = float(np.nanmax(np.abs(M))) or 1.0
        a.imshow(M, cmap=cmap, vmin=-v, vmax=+v, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                       fontsize=7.2, color="#111")
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, fontsize=7)
        a.set_yticks(range(len(rows)))
        a.set_yticklabels([f"{r:.0f}" for r in rows], fontsize=7.5)
        a.set_xlabel("anti-roll bar blade pair", fontsize=8.5)
        a.set_ylabel("spring rate  [N/mm]", fontsize=8.5)
        a.set_title(f"{title}\n{unit}", fontsize=9)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0)

    def _axle_tab(self, tag, car, rc_f, rc_r, d, base_spr, base_arb, n_bl, nm):
        rows, cols, n = self._grids(base_spr, base_arb, n_bl)
        mr = d["mr_f"] if tag == "f" else d["mr_r"]
        kt = self._num("kt", 310.4)
        lat = self.LAT_G
        bal, rs, s0 = mbal.tables(car, rc_f, rc_r, tag, rows, cols, mr,
                                  base_spr, base_arb, tyre_rate=kt, lat_g=lat)
        labels = mbal.pair_labels(n)
        for a in self.ax[tag]:
            a.clear()
        self._heat(self.ax[tag][0], bal, rows, labels,
                   f"{nm} - Mechanical Balance FA",
                   "change in FRONT share of load transfer  [points]   "
                   "+ = understeer", "RdYlGn_r")
        self._heat(self.ax[tag][1], rs, rows, labels,
                   f"{nm} - Roll Stiffness",
                   "change in TOTAL roll stiffness  [%]   + = stiffer, "
                   "less roll", "PuOr_r")
        # ring the baseline cell so the reference is never in doubt
        i0, j0 = 2, n - 1
        for a in self.ax[tag]:
            a.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                                  ec="#111", lw=2.0, zorder=5))
        self.fig[tag].suptitle(
            f"{nm} AXLE   baseline {base_spr:.0f} N/mm spring on blade "
            f"{labels[j0]}   (ringed cell = 0.0 by definition)",
            fontsize=9.5, y=0.99)
        self.can[tag].draw_idle()
        return rows, cols, labels, n, s0

    # ------------------------------------------------------------- compute
    def _compute(self, car, rc_f, rc_r, d):
        kt = self._num("kt", 310.4)
        lat = self.LAT_G
        bs_f, bs_r = d["ks_f"], d["ks_r"]
        ba_f, ba_r = car["arb_f"], car["arb_r"]
        s0 = mbal.state(car, rc_f, rc_r, tyre_rate=kt, lat_g=lat)
        self.note.config(
            text=f"BASELINE   front LLTD {s0['lltd_f']:6.2f} %      "
                 f"roll-stiffness share {s0['front_share_pct']:6.2f} %      "
                 f"roll gradient {s0['roll_grad']:.3f} deg/g      "
                 f"elastic fraction k {s0['elastic_frac']:.4f}\n"
                 f"           springs F {bs_f:.0f} / R {bs_r:.0f} N/mm     "
                 f"bars F {ba_f:.1f} / R {ba_r:.1f} N/mm at the wheel     "
                 f"{mbal.classify_front_share(s0['front_share_pct'])}")

        rows_f, cols_f, lab_f, nf, _ = self._axle_tab(
            "f", car, rc_f, rc_r, d, bs_f, ba_f, self._num("nfb", 6), "FRONT")
        rows_r, cols_r, lab_r, nr, _ = self._axle_tab(
            "r", car, rc_f, rc_r, d, bs_r, ba_r, self._num("nrb", 4), "REAR")

        # ================================================== X: reset & trade
        for a in self.ax["x"]:
            a.clear()

        # ---- X1  the lookup the table exists for -------------------------
        a = self.ax["x"][0]
        lims = []
        for tag, rows, cols, mrv, bs, ba, col, nm in (
                ("f", rows_f, cols_f, d["mr_f"], bs_f, ba_f, "#c1121f", "FRONT"),
                ("r", rows_r, cols_r, d["mr_r"], bs_r, ba_r, "#0353a4", "REAR")):
            need = []
            for sp in rows:
                kb, info = mbal.reset_arb(car, rc_f, rc_r, tag, sp, mrv, bs, ba,
                                          tyre_rate=kt, lat_g=lat)
                need.append(100.0 * (kb / ba - 1.0) if info["reachable"]
                            else np.nan)
            a.plot(100.0 * (rows / bs - 1.0), need, "o-", lw=2.2, ms=5,
                   color=col, label=f"{nm} bar")
            lims.append((100.0 * (cols.min() / ba - 1.0),
                         100.0 * (cols.max() / ba - 1.0)))
        lo = max(l[0] for l in lims)
        hi = min(l[1] for l in lims)
        a.axhspan(lo, hi, color="#2a9d8f", alpha=0.08, zorder=0)
        for yv in (lo, hi):
            a.axhline(yv, ls=":", lw=1.1, color="#1f7a6e", alpha=0.8)
        a.annotate("what the blades can actually reach", (0.02, hi),
                   xycoords=("axes fraction", "data"), ha="left", va="top",
                   fontsize=6.8, color="#1f7a6e")
        a.axhline(0, lw=1.0, color="#555")
        a.axvline(0, lw=1.0, color="#555")
        a.set_xlabel("spring change on that axle  [%]", fontsize=8.5)
        a.set_ylabel("bar change needed to hold balance  [%]", fontsize=8.5)
        a.set_title("X1 - keep the balance you had\nstiffer spring is paid for "
                    "with a SOFTER bar", fontsize=9)
        a.legend(fontsize=7, loc="best")
        a.grid(alpha=0.25)

        # ---- X2  the trade: you cannot move one without the other --------
        a = self.ax["x"][1]
        for tag, rows, cols, mrv, bs, ba, col, nm in (
                ("f", rows_f, cols_f, d["mr_f"], bs_f, ba_f, "#c1121f", "FRONT"),
                ("r", rows_r, cols_r, d["mr_r"], bs_r, ba_r, "#0353a4", "REAR")):
            b, r_, _ = mbal.tables(car, rc_f, rc_r, tag, rows, cols, mrv, bs, ba,
                                   tyre_rate=kt, lat_g=lat)
            a.scatter(r_.ravel(), b.ravel(), s=13, color=col, alpha=0.55,
                      label=f"{nm} axle", edgecolors="none")
        a.plot([0], [0], "*", ms=17, mfc="#f5c000", mec="k", mew=1.1, zorder=6)
        a.annotate("baseline", (0, 0), textcoords="offset points",
                   xytext=(8, -12), fontsize=8, fontweight="bold")
        a.axhline(0, lw=1.0, color="#555")
        a.axvline(0, lw=1.0, color="#555")
        a.set_xlabel("roll stiffness change  [%]", fontsize=8.5)
        a.set_ylabel("balance change  [points]", fontsize=8.5)
        a.set_title("X2 - every setup this axle can reach\nfront and rear pull "
                    "the balance OPPOSITE ways for the same stiffening",
                    fontsize=9)
        a.legend(fontsize=7, loc="best")
        a.grid(alpha=0.25)

        # ---- X3  where the balance actually comes from -------------------
        a = self.ax["x"][2]
        t = ltr.transfer(lat, rc_f, rc_r, s0["car"])
        tot = t["total_f"] + t["total_r"]
        parts = [("elastic\n(springs + bars)", t["el_f"], t["el_r"], "#2a9d8f"),
                 ("geometric\n(roll centres)", t["geo_f"], t["geo_r"], "#7a5195"),
                 ("unsprung", t["uns_f"], t["uns_r"], "#c1121f")]
        y = np.arange(len(parts))
        a.barh(y - 0.18, [100.0 * p[1] / tot for p in parts], 0.34,
               color=[p[3] for p in parts], alpha=0.95, label="front")
        a.barh(y + 0.18, [100.0 * p[2] / tot for p in parts], 0.34,
               color=[p[3] for p in parts], alpha=0.45, label="rear")
        for i, p in enumerate(parts):
            for v, off in ((100.0 * p[1] / tot, -0.18), (100.0 * p[2] / tot, 0.18)):
                a.text(v, i + off, f"{v:.1f} " if v < 0 else f" {v:.1f}",
                       va="center", ha="right" if v < 0 else "left",
                       fontsize=7.5)
        a.margins(x=0.22)
        a.axvline(0, lw=1.0, color="#555")
        a.set_yticks(y)
        a.set_yticklabels([p[0] for p in parts], fontsize=7.5)
        a.set_xlabel("share of TOTAL lateral load transfer  [%]", fontsize=8.5)
        a.set_title(f"X3 - only the green part answers to a spring\n"
                    f"elastic fraction k = {s0['elastic_frac']:.3f}: 1 point of "
                    f"roll-stiffness share buys {s0['elastic_frac']:.2f} of balance",
                    fontsize=9)
        a.legend(fontsize=7, loc="best", framealpha=0.9)
        a.grid(alpha=0.25, axis="x")

        # ---- X4  the tyre is why the rows are not parallel ---------------
        a = self.ax["x"][3]
        sweep = np.linspace(0.5 * bs_f, 2.0 * bs_f, 60)
        for ktv, ls, lab in ((kt, "-", f"with the tyre ({kt:.0f} N/mm)"),
                             (None, "--", "tyre left out")):
            span = []
            for sp in sweep:
                b, r_, _ = mbal.tables(car, rc_f, rc_r, "f", [sp],
                                       [cols_f[0], cols_f[-1]], d["mr_f"],
                                       bs_f, ba_f, tyre_rate=ktv, lat_g=lat)
                span.append(r_[0, 1] - r_[0, 0])
            a.plot(sweep, span, ls, lw=2.2, color="#0353a4", label=lab)
        a.axvline(bs_f, lw=1.2, color="#555", ls=":")
        a.text(bs_f, 0.04, " our front spring", rotation=90, fontsize=7,
               color="#555", va="bottom", ha="left",
               transform=a.get_xaxis_transform())
        a.set_ylim(0.0, None)
        a.set_xlabel("front spring rate  [N/mm]", fontsize=8.5)
        a.set_ylabel("roll stiffness the full bar range buys  [%]", fontsize=8.5)
        a.set_title("X4 - why a bar step is worth less on stiff springs\n"
                    "the tyre is in SERIES, so the axle saturates towards it",
                    fontsize=9)
        a.legend(fontsize=7, loc="best")
        a.grid(alpha=0.25)

        self.fig["x"].suptitle("", fontsize=1)
        self.can["x"].draw_idle()


class VehiclePage(ttk.Frame):
    """Whole-car results. It owns no geometry of its own: the roll centres come
    LIVE from whatever the FRONT and REAR pages currently hold, so a hard-point
    edit on either axle flows straight through to the tyre loads."""

    LAT_MAX, ACC_MAX, BRK_MAX = 1.2, 1.0, 1.2
    WHEELS = ["FL", "FR", "RL", "RR"]

    def __init__(self, master, get_rcs, get_mrs=None, get_pts=None,
                 get_roll=None):
        super().__init__(master)
        self.get_rcs = get_rcs
        self.get_roll = get_roll
        self.get_mrs = get_mrs or (lambda: (None, None))
        self.get_pts = get_pts
        self.car = ltr.temerario_gt3()
        self.W = self.car["mass_total_kg"] * ltr.G
        self.rc_f = self.rc_r = None

        left = ttk.Frame(self, padding=(8, 6))
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="VEHICLE  -  load transfer & g-g",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left, text="roll centres are read LIVE from the FRONT and REAR\n"
                             "tabs - edit a hard point there and this updates",
                  font=("Segoe UI", 8), foreground="#666",
                  justify="left").pack(anchor="w", pady=(0, 6))

        self.vars = {}

        # ---------------------------------------------------- corner weights
        # AVL-VSM layout: relative corner weights [%] are the INPUT, total static
        # weight scales them, and the kg per corner is derived. Entering it this
        # way means a ballast change is one number, and an asymmetric or
        # cross-weighted car is expressible -- a single "rear fraction" is not.
        ttk.Label(left, text="Corner weight  [% of total]", font=("Segoe UI", 8, "bold"),
                  foreground="#333").pack(anchor="w", pady=(4, 1))
        self.cw = {}
        grid = ttk.Frame(left)
        grid.pack(anchor="w")
        for i, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
            for j, k in enumerate((a, b)):
                v = tk.StringVar(value="22" if i == 0 else "28")
                e = tk.Entry(grid, textvariable=v, width=9, justify="right",
                             font=("Consolas", 8), relief="solid", borderwidth=1)
                e.grid(row=i, column=j, padx=1, pady=1)
                e.bind("<Return>", lambda _e: self.refresh())
                self.cw[k] = v
        self.cw_sum = ttk.Label(left, text="", font=("Consolas", 7),
                                foreground="#666")
        self.cw_sum.pack(anchor="w")

        ttk.Label(left, text="Calculated corner weight  [kg]",
                  font=("Segoe UI", 8, "bold"), foreground="#333").pack(anchor="w",
                                                                        pady=(4, 1))
        self.cwkg = {}
        grid2 = ttk.Frame(left)
        grid2.pack(anchor="w")
        for i, (a, b) in enumerate((("FL", "FR"), ("RL", "RR"))):
            for j, k in enumerate((a, b)):
                lb = tk.Label(grid2, text="-", width=9, anchor="e", bg="#e9e9e9",
                              font=("Consolas", 8), relief="solid", borderwidth=1)
                lb.grid(row=i, column=j, padx=1, pady=1)
                self.cwkg[k] = lb

        # ------------------------------------------------------ scalar inputs
        for key, lab, val in (("mass", "total weight static [kg]", 1300.0),
                              ("unsprung", "unsprung / corner [kg]", 45.0),
                              ("h_cg", "CG height [mm]", 450.0),
                              ("wb", "wheelbase [mm]", 2645.0),
                              ("freq_f", "front ride freq [Hz]", 3.86),
                              ("freq_r", "rear ride freq [Hz]", 2.89),
                              ("k_tyre", "tyre rate [N/mm]", 310.4),
                              ("a_lat", "lateral g", 0.52),
                              ("a_long", "longitudinal g", -1.08)):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=lab, width=24, font=("Consolas", 8),
                      anchor="w").pack(side="left")
            v = tk.StringVar(value=f"{val:g}")
            e = ttk.Entry(row, textvariable=v, width=10, font=("Consolas", 8),
                          justify="right")
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self.refresh())
            self.vars[key] = v
        ttk.Button(left, text="Apply", command=self.refresh).pack(anchor="w",
                                                                  pady=(8, 4))
        ttk.Label(left, text="ASSUMPTIONS, not measurements - every number\n"
                             "below scales with the boxes above.",
                  font=("Segoe UI", 7), foreground="#888",
                  justify="left").pack(anchor="w")
        self.out = tk.Text(left, width=54, height=26, font=("Consolas", 8),
                           relief="flat", background="#f7f7f7", wrap="none")
        self.out.pack(fill="both", expand=True, pady=(6, 0))

        self.rnb = ttk.Notebook(right)
        self.rnb.pack(fill="both", expand=True)
        f_gg = ttk.Frame(self.rnb)
        f_sp = ttk.Frame(self.rnb)
        self.rnb.add(f_gg, text="  Load transfer & g-g  ")
        self.rnb.add(f_sp, text="  Springs  ")
        self.arbp = ARBPanel(self.rnb, self)
        self.rnb.add(self.arbp, text="  Anti-roll bars  ")
        self.rollp = RollAxisPanel(self.rnb, self)
        self.rnb.add(self.rollp, text="  Roll axis  ")
        self.antip = AntiGeometryPanel(self.rnb, self)
        self.rnb.add(self.antip, text="  Anti geometry  ")
        self.mbp = MechBalancePanel(self.rnb, self)
        self.rnb.add(self.mbp, text="  Mechanical balance  ")

        self.fig = Figure(figsize=(11.5, 7.0), dpi=100)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.30)
        self.axA = self.fig.add_subplot(gs[0, 0])
        self.axB = self.fig.add_subplot(gs[0, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=f_gg)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, f_gg).update()
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._cb = None

        # table goes in FIRST, packed to the bottom, so the figure above it keeps
        # every pixel that is left rather than squeezing the table off-screen
        tbl_host = ttk.Frame(f_sp, padding=(6, 2))
        tbl_host.pack(side="bottom", fill="x")
        self.sptab = _SpringTable(tbl_host)
        self.sptab.pack(fill="x")

        self.sfig = Figure(figsize=(11.5, 4.6), dpi=100)
        sgs = self.sfig.add_gridspec(1, 2, wspace=0.28)
        self.axS1 = self.sfig.add_subplot(sgs[0, 0])
        self.axS2 = self.sfig.add_subplot(sgs[0, 1])
        self.scanvas = FigureCanvasTkAgg(self.sfig, master=f_sp)
        self.scanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.scanvas, f_sp).update()
        self.scanvas.mpl_connect("motion_notify_event", self._on_spring_move)
        self.scanvas.mpl_connect("axes_leave_event", self._on_spring_leave)
        self._smark = {}
        self._svline = {}
        self._snote = {}

        self.refresh()

    def _num(self, k, d):
        try:
            return float(self.vars[k].get())
        except ValueError:
            return d

    def corner_pct(self):
        """The four corner weights as fractions, NORMALISED to sum to 1.

        Normalising means the boxes can be typed in any units -- percent, kg,
        raw scale readings straight off the pads -- and the split is still read
        correctly. The raw sum is reported next to the table so a typo is
        visible rather than silently rescaled.
        """
        raw = {}
        for k, v in self.cw.items():
            try:
                raw[k] = max(0.0, float(v.get()))
            except ValueError:
                raw[k] = 0.0
        tot = sum(raw.values())
        if tot <= 0:
            raw = {k: 25.0 for k in raw}
            tot = 100.0
        return {k: raw[k] / tot for k in raw}, tot

    def _rebuild_car(self):
        pct, raw_sum = self.corner_pct()
        mass = self._num("mass", 1300.0)
        rear_frac = pct["RL"] + pct["RR"]
        self.cw_sum.config(
            text=f"sum {raw_sum:g}  ->  {rear_frac * 100:.2f} % rear"
                 + ("   (normalised)" if abs(raw_sum - 100.0) > 0.01 else ""))
        for k, lb in self.cwkg.items():
            lb.config(text=f"{pct[k] * mass:.2f}")
        self.corner_kg = {k: pct[k] * mass for k in pct}

        self.car = ltr.temerario_gt3(freq_f=self._num("freq_f", 3.86),
                                     freq_r=self._num("freq_r", 2.89))
        self.car["mass_total_kg"] = mass
        self.car["rear_frac"] = rear_frac
        self.car["h_cg_mm"] = self._num("h_cg", 450.0)
        self.car["wheelbase_mm"] = self._num("wb", 2645.0)
        self.W = mass * ltr.G

    def spring_design(self):
        """Spring rates from the corner weights and the LIVE motion ratios.

        Nothing here is a typed-in constant except the targets: the motion ratio
        comes from whatever the FRONT and REAR hard points currently solve to,
        so moving a damper mount changes the required coil rate immediately.
        Returns None until both axles have a solvable geometry.
        """
        mr_f, mr_r = self.get_mrs()
        if mr_f is None or mr_r is None:
            return None
        kt = self._num("k_tyre", 310.4)
        u = self._num("unsprung", 45.0)
        ck = getattr(self, "corner_kg", None) or {}
        # springs are sized per AXLE (you fit a matched pair), so use the axle's
        # mean corner mass; the per-corner spread is reported separately
        mf = (ck.get("FL", 0.0) + ck.get("FR", 0.0)) / 2.0 - u
        mr = (ck.get("RL", 0.0) + ck.get("RR", 0.0)) / 2.0 - u
        if mf <= 0 or mr <= 0:
            return None
        ff, fr = self._num("freq_f", 3.86), self._num("freq_r", 2.89)
        d = dict(mr_f=mr_f, mr_r=mr_r, k_tyre=kt, m_f=mf, m_r=mr,
                 freq_f=ff, freq_r=fr)
        for tag, f, m, rm in (("f", ff, mf, mr_f), ("r", fr, mr, mr_r)):
            kw = float(spr.wheel_rate_for_frequency(f, m))
            d[f"kw_{tag}"] = kw
            d[f"ks_{tag}"] = float(spr.spring_rate(kw, rm))
            d[f"rr_{tag}"] = float(spr.ride_rate(kw, kt))
            d[f"f_tyre_{tag}"] = float(spr.frequency(kw, m, kt))
        d["cross_kg"] = abs((ck.get("FL", 0.0) + ck.get("RR", 0.0))
                            - (ck.get("FR", 0.0) + ck.get("RL", 0.0))) / 2.0
        for tag in ("f", "r"):
            d[f"table_{tag}"], d[f"pick_{tag}"] = self._catalogue(d, tag)
        return d

    @staticmethod
    def _catalogue(d, tag):
        """A shelf of coil rates around the design point, with what each does.

        Steps are snapped to a round number so the rows read like a parts list
        you could actually order from, rather than the arbitrary output of a
        calculation.
        """
        ks0 = d[f"ks_{tag}"]
        m, rm, kt = d[f"m_{tag}"], d[f"mr_{tag}"], d["k_tyre"]
        step = 25.0 if ks0 >= 150.0 else (10.0 if ks0 >= 60.0 else 5.0)
        base = round(ks0 / step) * step
        pick = 3
        ks = [base + step * (i - pick) for i in range(_SpringTable.N_ROWS)]
        ks = [k for k in ks if k > 0]
        pick = min(range(len(ks)), key=lambda i: abs(ks[i] - ks0))

        W = m * ltr.G                       # sprung corner weight [N]
        kw_pick = spr.wheel_rate_from_spring(ks[pick], rm)
        rows = []
        for k in ks:
            kw = float(spr.wheel_rate_from_spring(k, rm))
            rows.append((k, kw,
                         float(spr.frequency(kw, m)),
                         float(spr.frequency(kw, m, kt)),
                         W / kw,                       # static deflection
                         W / kw - W / float(kw_pick)))  # vs the highlighted row
        return rows, pick

    def _draw_springs(self, d):
        """Two views: what the target costs in spring rate, and how brutally the
        answer depends on the motion ratio (it enters SQUARED)."""
        a1, a2 = self.axS1, self.axS2
        a1.clear()
        a2.clear()
        self._smark, self._svline, self._snote = {}, {}, {}
        if d is None:
            for a in (a1, a2):
                a.text(0.5, 0.5, "waiting for a solvable geometry\non both axles",
                       ha="center", va="center", fontsize=10, color="#888")
                a.set_xticks([])
                a.set_yticks([])
            self.sptab.clear()
            self.scanvas.draw_idle()
            return

        # --- A: coil rate needed vs target ride frequency
        f = np.linspace(1.0, 5.5, 200)
        for tag, nm, col, m, rm in (("f", "FRONT", "#c1121f", d["m_f"], d["mr_f"]),
                                    ("r", "REAR", "#0353a4", d["m_r"], d["mr_r"])):
            ks = spr.spring_rate(spr.wheel_rate_for_frequency(f, m), rm)
            a1.plot(f, ks, lw=2.2, color=col,
                    label=f"{nm}  ({spr.symbol()} {spr.to_display(rm):.3f})")
            # STOCK marker: a square with a white ring, so it stays distinct from
            # the round cursor dot that moves with the mouse
            a1.plot([d[f"freq_{tag}"]], [d[f"ks_{tag}"]], "s", ms=10, mfc=col,
                    mec="w", mew=2.0, zorder=6)
            a1.plot([d[f"freq_{tag}"]], [d[f"ks_{tag}"]], "s", ms=12, mfc="none",
                    mec=col, mew=1.4, zorder=6)
            a1.annotate(f"stock {d['ks_' + tag]:.0f}", (d[f"freq_{tag}"], d[f"ks_{tag}"]),
                        textcoords="offset points", xytext=(-8, 12), fontsize=8,
                        ha="right", color=col, fontweight="bold")
        for nm, lo, hi in spr.BANDS:
            a1.axvspan(lo, hi, color="#999", alpha=0.10, zorder=0)
        a1.text(4.0, a1.get_ylim()[1] * 0.02, "high-downforce band", fontsize=7,
                color="#666", ha="center")
        a1.set_xlabel("target ride frequency  [Hz]   (wheel-rate definition)", fontsize=9)
        a1.set_ylabel("coil rate required  [N/mm]", fontsize=9)
        a1.set_title("S1 · what the frequency target costs in spring rate", fontsize=9.5)
        # S1's curves sweep bottom-left -> top-right, so the UPPER LEFT is the
        # empty corner and the legend moves out of it to make room for the readout
        a1.legend(fontsize=8, loc="lower right")

        # --- B: sensitivity to the motion ratio  (K_s = K_w / R_M^2)
        rm = np.linspace(0.45, 1.15, 200)
        x = spr.to_display(rm)
        for tag, nm, col in (("f", "FRONT", "#c1121f"), ("r", "REAR", "#0353a4")):
            a2.plot(x, d[f"kw_{tag}"] / rm ** 2, lw=2.2, color=col, label=nm)
            xi = spr.to_display(d[f"mr_{tag}"])
            a2.plot([xi], [d[f"ks_{tag}"]], "s", ms=10, mfc=col,
                    mec="w", mew=2.0, zorder=6)
            a2.plot([xi], [d[f"ks_{tag}"]], "s", ms=12, mfc="none",
                    mec=col, mew=1.4, zorder=6)
        _ref = bmk.get("MOTION_RATIO")
        if _ref:
            for _rm, ls in zip(_ref, ("--", ":")):
                a2.axvline(spr.to_display(_rm), ls=ls, lw=1.1, color="#666")
            a2.text(spr.to_display(_ref[0]), a2.get_ylim()[1] * 0.92,
                    f" reference {spr.symbol()}", fontsize=7,
                    color="#666", rotation=90, va="top")
        a2.set_xlabel(spr.label(), fontsize=9)
        a2.set_ylabel("coil rate required  [N/mm]", fontsize=9)
        a2.set_title(f"S2 · {spr.symbol()} enters SQUARED — this is why the "
                     f"mount matters\n(dots = our live geometry)", fontsize=9.5)
        a2.legend(fontsize=8, loc="upper right")

        # benchmark reference lines -- STATIC, they stay put while the cursor moves
        for f_bm, col in ((3.86, "#c1121f"), (2.89, "#0353a4")):
            a1.axvline(f_bm, ls="--", lw=1.0, color=col, alpha=0.45)
        a1.text(3.86, a1.get_ylim()[1] * 0.98, " benchmark f", fontsize=7,
                color="#666", rotation=90, va="top")

        # hover furniture, per plot (ax.clear() above destroyed the last set)
        for a in (a1, a2):
            self._svline[a] = a.axvline(np.nan, lw=0.9, color="#444", alpha=0.55,
                                        zorder=8)
            self._smark[a] = [a.plot([], [], "o", ms=7.5, mfc=c, mec="k", mew=1.2,
                                     zorder=11, label="_nolegend_")[0]
                              for c in ("#c1121f", "#0353a4")]
            _x, _ha = (0.015, "left") if a is a1 else (0.985, "right")
            self._snote[a] = a.text(
                _x, 0.975, "", transform=a.transAxes, ha=_ha, va="top",
                fontsize=8.2, family="monospace", linespacing=1.45, zorder=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#c9b458",
                          alpha=0.95))
            a.grid(alpha=0.3)
            a.tick_params(labelsize=8)
        self._hide_spring_hover()

        self.sfig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.14)
        self.scanvas.draw_idle()

        for tag, nm in (("f", "FRONT"), ("r", "REAR")):
            self.sptab.fill(
                tag, d[f"table_{tag}"], d[f"pick_{tag}"],
                f"{nm} AXLE   {spr.symbol()} "f"{spr.to_display(d['mr_' + tag]):.4f}   "
                f"sprung/corner {d['m_' + tag]:.1f} kg   "
                f"design {d['ks_' + tag]:.1f} N/mm @ {d['freq_' + tag]:.2f} Hz "
                f"(nearest shelf rate highlighted)")

    def _hide_spring_hover(self):
        for a, ln in self._svline.items():
            ln.set_xdata([np.nan, np.nan])
            for m in self._smark[a]:
                m.set_data([], [])
            self._snote[a].set_text("")

    def _on_spring_leave(self, _ev):
        if self._snote:
            self._hide_spring_hover()
            self.scanvas.draw_idle()

    def _on_spring_move(self, ev):
        d = getattr(self, "_spring", None)
        if d is None or not self._snote or ev.xdata is None:
            return
        if ev.inaxes not in (self.axS1, self.axS2):
            return
        a = ev.inaxes
        for other in (self.axS1, self.axS2):     # only the hovered plot responds:
            if other is not a:                   # the two have different x units
                self._svline[other].set_xdata([np.nan, np.nan])
                for m in self._smark[other]:
                    m.set_data([], [])
                self._snote[other].set_text("")
        self._svline[a].set_xdata([ev.xdata, ev.xdata])

        if a is self.axS1:                                    # x = ride frequency
            f = float(np.clip(ev.xdata, 1.0, 5.5))
            txt = [f"target f {f:5.2f} Hz"]
            for i, (tag, nm) in enumerate((("f", "FRONT"), ("r", "REAR"))):
                kw = float(spr.wheel_rate_for_frequency(f, d[f"m_{tag}"]))
                ks = float(spr.spring_rate(kw, d[f"mr_{tag}"]))
                self._smark[a][i].set_data([f], [ks])
                txt.append(f"{nm:5s} coil {ks:6.1f}  wheel {kw:6.1f} N/mm")
            self._snote[a].set_text("\n".join(txt))
        else:                                                 # x = motion ratio
            lo, hi = sorted(float(spr.to_display(v)) for v in (0.45, 1.15))
            xd = float(np.clip(ev.xdata, lo, hi))
            rm = float(spr.from_display(xd))
            txt = [f"motion ratio {spr.symbol()} {xd:5.3f}"]
            for i, (tag, nm) in enumerate((("f", "FRONT"), ("r", "REAR"))):
                ks = d[f"kw_{tag}"] / rm ** 2
                self._smark[a][i].set_data([xd], [ks])
                txt.append(f"{nm:5s} coil {ks:6.1f} N/mm"
                           f"  ({ks / d['ks_' + tag]:.2f}x ours)")
            self._snote[a].set_text("\n".join(txt))
        self.scanvas.draw_idle()

    def refresh(self):
        """Recompute from the live roll centres. Called on Apply and whenever
        either axle page reports a geometry change."""
        self._rebuild_car()
        self._spring = self.spring_design()
        # Keep the two tabs on ONE set of wheel rates. temerario_gt3() derives
        # its own from a fixed mass model; once the corner-weight table drives
        # the sprung masses those can disagree, and the g-g page would then be
        # transferring load through springs the Springs page never designed.
        if self._spring is not None:
            self.car["kw_f"] = self._spring["kw_f"]
            self.car["kw_r"] = self._spring["kw_r"]
            self.car["arb_f"] = self.car["kw_f"] * 2.15      # benchmark ARB/spring
            self.car["arb_r"] = self.car["kw_r"] * 0.674     # ratios, as before
            tf, tr = self.car["track_f_mm"], self.car["track_r_mm"]
            self.car["k_roll_f"] = ltr.roll_stiffness(self.car["kw_f"],
                                                      self.car["arb_f"], tf)
            self.car["k_roll_r"] = ltr.roll_stiffness(self.car["kw_r"],
                                                      self.car["arb_r"], tr)
            tot = self.car["k_roll_f"] + self.car["k_roll_r"]
            self.car["roll_stiff_front_pct"] = self.car["k_roll_f"] / tot * 100.0
        self._draw_springs(self._spring)
        if getattr(self, 'arbp', None) is not None:
            self.arbp.refresh()
        if getattr(self, 'rollp', None) is not None:
            self.rollp.refresh()
        if getattr(self, 'antip', None) is not None:
            self.antip.refresh()
        if getattr(self, 'mbp', None) is not None:
            self.mbp.refresh()
        rc_f, rc_r = self.get_rcs()
        if rc_f is None or rc_r is None:
            self.out.delete("1.0", "end")
            self.out.insert("end", "waiting for a solvable geometry on both axles")
            return
        self.rc_f, self.rc_r = rc_f, rc_r

        N = 91
        gl = np.linspace(-self.LAT_MAX, self.LAT_MAX, N)
        go = np.linspace(-self.BRK_MAX, self.ACC_MAX, N)
        GL, GO = np.meshgrid(gl, go)
        w = ltr.wheel_loads(GL, GO, rc_f, rc_r, self.car)
        minw = np.minimum(np.minimum(w["FL"], w["FR"]), np.minimum(w["RL"], w["RR"]))
        minw = np.where(ltr.inside_envelope(GL, GO, self.LAT_MAX, self.ACC_MAX,
                                            self.BRK_MAX), minw, np.nan)

        ax = self.axA
        ax.clear()
        cf = ax.contourf(GL, GO, minw, levels=20, cmap="viridis")
        if self._cb is None:
            self._cb = self.fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02,
                                         label="minimum wheel load [N]")
        else:
            self._cb.update_normal(cf)
        ax.contour(GL, GO, minw, levels=[0.0], colors="red", linewidths=2.2)
        le, lo = ltr.gg_envelope(self.LAT_MAX, self.ACC_MAX, self.BRK_MAX)
        ax.plot(le, lo, color="k", lw=1.8)
        ax.axhline(0, color="#888", lw=0.7)
        ax.axvline(0, color="#888", lw=0.7)
        ax.set_xlabel("lateral acceleration  [g]", fontsize=9)
        ax.set_ylabel("longitudinal acceleration  [g]", fontsize=9)
        ax.set_title(f"g-g shaded by the LOWEST wheel load\n"
                     f"front RC {rc_f:+.0f} / rear RC {rc_r:+.0f} mm (live) - "
                     "hover to pick", fontsize=9.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        self.mark, = ax.plot([], [], "o", ms=12, mfc="none", mec="k", mew=2.2,
                             zorder=10)

        bx = self.axB
        bx.clear()
        xp = np.arange(4)
        self.bars = bx.bar(xp, [0] * 4, 0.55, color="#2171b5", edgecolor="k")
        bx.axhline(self.W / 4, color="#888", lw=1.1, ls="--")
        bx.axhline(0, color="k", lw=1.2)
        bx.set_xticks(xp)
        bx.set_xticklabels(self.WHEELS)
        bx.set_ylabel("vertical tyre load  [N]", fontsize=9)
        bx.set_title("four-wheel loads at the selected g-g point\n"
                     "below zero = the wheel has LIFTED", fontsize=9.5)
        bx.grid(alpha=0.3, axis="y")
        bx.tick_params(labelsize=8)
        # a colourbar Axes is not tight_layout-compatible; fix the margins instead
        self.fig.subplots_adjust(left=0.085, right=0.965, top=0.88, bottom=0.10,
                                 wspace=0.30)
        self.show(self._num("a_lat", 0.52), self._num("a_long", -1.08))

    def show(self, al, ao):
        d = ltr.wheel_loads(al, ao, self.rc_f, self.rc_r, self.car)
        self.mark.set_data([al], [ao])
        vals = [float(d[k]) for k in self.WHEELS]
        for r, v in zip(self.bars, vals):
            r.set_height(v)
            r.set_color("#c0392b" if v <= 0 else "#2171b5")
        self.axB.set_ylim(min(-400.0, min(vals) - 200), max(6000.0, max(vals) * 1.12))
        self.canvas.draw_idle()

        t = ltr.transfer(al if al != 0 else 1e-9, self.rc_f, self.rc_r, self.car)
        L = ["ROLL CENTRES (live from the geometry)",
             f"   front {self.rc_f:+8.1f} mm      rear {self.rc_r:+8.1f} mm",
             f"   roll axis at the CG {float(t['roll_axis']):+.1f} mm", "",
             f"g-g STATE   a_lat {al:+.2f} g   a_long {ao:+.2f} g",
             f"   longitudinal transfer {float(d['dW_long']):+9.0f} N",
             f"   axle weights   front {float(d['W_f']):8.0f}   "
             f"rear {float(d['W_r']):8.0f} N", "",
             "WHEEL LOADS [N]"]
        for k, v in zip(self.WHEELS, vals):
            L.append(f"   {k}   {v:9.0f}")
        L.append(f"   sum  {float(d['total']):9.0f}   (weight {self.W:.0f})")
        if d["lifted"]:
            L.append(f"   *** LIFTED: {', '.join(d['lifted'])} ***")
        L += ["", "STEADY-STATE LATERAL (at this a_lat)",
              f"   front LLTD      {float(t['lltd_f']):8.2f} %",
              f"   roll gradient   {float(t['roll_deg'])/max(abs(al),1e-9):8.3f} deg/g",
              f"   rear geometric  {float(t['geo_frac_r']):8.1f} % of rear transfer",
              f"   front transfer  {float(t['total_f']):8.0f} N",
              f"   rear transfer   {float(t['total_r']):8.0f} N", "",
              "CAR (assumptions)",
              f"   wheel rate  F {self.car['kw_f']:6.1f}  R {self.car['kw_r']:6.1f} N/mm",
              f"   ARB rate    F {self.car['arb_f']:6.1f}  R {self.car['arb_r']:6.1f} N/mm",
              f"   front roll stiffness {self.car['roll_stiff_front_pct']:.1f} %"]

        d = getattr(self, "_spring", None)
        if d is not None:
            L += ["", "SPRINGS  (motion ratio is LIVE from the hard points)",
                  f"   motion ratio {spr.symbol():>3s}"f"   F {spr.to_display(d['mr_f']):7.4f}"f"  R {spr.to_display(d['mr_r']):7.4f}   ({spr.ratio_text()})",
                  f"      = R_M        F {d['mr_f']:7.4f}"f"  R {d['mr_r']:7.4f}   (spring travel / wheel travel)",
                  f"   sprung/corner  F {d['m_f']:7.1f}  R {d['m_r']:7.1f} kg",
                  f"   wheel rate     F {d['kw_f']:7.1f}  R {d['kw_r']:7.1f} N/mm",
                  f"   COIL RATE      F {d['ks_f']:7.1f}  R {d['ks_r']:7.1f} N/mm",
                  "",
                  f"   ride freq, wheel-rate defn   F {d['freq_f']:5.2f}  "
                  f"R {d['freq_r']:5.2f} Hz",
                  f"   ride freq, with tyre in series F {d['f_tyre_f']:4.2f}  "
                  f"R {d['f_tyre_r']:4.2f} Hz",
                  f"   (tyre {d['k_tyre']:.0f} N/mm costs "
                  f"{(1 - d['f_tyre_f'] / d['freq_f']) * 100:.0f} % of the front)"]
            if d["cross_kg"] > 0.5:
                L += [f"   CROSS-WEIGHT   {d['cross_kg']:7.1f} kg diagonal split"]
        self.out.delete("1.0", "end")
        self.out.insert("end", "\n".join(L))

    def _on_move(self, ev):
        if ev.inaxes is not self.axA or ev.xdata is None:
            return
        if not ltr.inside_envelope(ev.xdata, ev.ydata, self.LAT_MAX,
                                   self.ACC_MAX, self.BRK_MAX):
            return
        self.show(float(ev.xdata), float(ev.ydata))


# =====================================================================
#  the application  -  FRONT | REAR | VEHICLE
# =====================================================================
FRONT_SPEC = dict(rows=POINT_ROWS, load=cfg._load_front_points,
                  camber=cfg.STATIC_CAMBER, compute=compute,
                  panels=build_panels, csv_tag="FRONT", name="FRONT")
REAR_SPEC = dict(rows=REAR_POINT_ROWS, load=rgeo.load_rear,
                 camber=rgeo.REAR_STATIC_CAMBER, compute=rear_compute,
                 panels=build_rear_panels, csv_tag="REAR", name="REAR")


class Explorer(tk.Tk):
    """Top level: one page per axle plus the whole-car page."""

    def __init__(self):
        super().__init__()
        self.title("Temerario GT3 - suspension explorer")
        self.geometry("1720x980")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.front = AxlePage(nb, FRONT_SPEC, on_change=self._axle_changed)
        self.rear = AxlePage(nb, REAR_SPEC, on_change=self._axle_changed)
        nb.add(self.front, text="   FRONT   ")
        nb.add(self.rear, text="   REAR   ")
        self.vehicle = VehiclePage(nb, self._roll_centres, self._motion_ratios,
                                   self._axle_points, self._roll_migration)
        nb.add(self.vehicle, text="   VEHICLE   ")
        self.nb = nb

    def _roll_centres(self):
        """Current front and rear roll-centre heights, or None if unsolvable."""
        out = []
        for page in (getattr(self, "front", None), getattr(self, "rear", None)):
            d = getattr(page, "cur", None) if page is not None else None
            out.append(float(d["R"]["rc0"]) if ok(d) else None)
        return out[0], out[1]

    def _axle_points(self):
        """(front pts, front camber, rear pts, rear camber) as the axle pages
        currently hold them -- so the ARB pickup's motion ratio is solved against
        the live geometry, not the CSV. None until both axles solve."""
        f, r = getattr(self, "front", None), getattr(self, "rear", None)
        if f is None or r is None:
            return None
        try:
            pf, cf, badf = f.tab["R"].read()
            pr, cr, badr = r.tab["R"].read()
        except Exception:                                     # noqa: BLE001
            return None
        if badf or badr:
            return None
        return pf, cf, rgeo.solver_points(pr), cr

    def _roll_migration(self):
        """(front, rear) roll-migration sweeps, computing them on demand.

        Each axle's roll centre MOVES as the car leans, so the roll axis has to
        be rebuilt at every roll angle rather than assumed fixed. The sweep is
        ~1 s per axle, so it stays lazy -- the same cache the RC-in-roll tabs use.
        """
        out = []
        for page in (getattr(self, "front", None), getattr(self, "rear", None)):
            d = getattr(page, "cur", None) if page is not None else None
            if not ok(d):
                out.append(None)
                continue
            page.ensure_roll(d)
            out.append(d.get("roll"))
        return out[0], out[1]

    def _motion_ratios(self):
        """Current front and rear damper motion ratios, or None if unsolvable.

        Same live wiring as the roll centres: R_M is SOLVED from the hard points
        on every recompute, so moving a damper mount (or anything that changes
        how the arm swings) flows straight through to the spring rate. It is not
        a constant typed in anywhere.
        """
        out = []
        for page in (getattr(self, "front", None), getattr(self, "rear", None)):
            d = getattr(page, "cur", None) if page is not None else None
            out.append(float(d["R"]["mr0"]) if ok(d) else None)
        return out[0], out[1]

    def _axle_changed(self):
        if hasattr(self, "vehicle"):
            self.vehicle.refresh()


if __name__ == "__main__":
    print("Suspension explorer opening:  FRONT | REAR | VEHICLE")
    print("  edit a hard point on either axle and the VEHICLE tab follows.")
    Explorer().mainloop()
