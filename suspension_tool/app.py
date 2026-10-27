# -*- coding: utf-8 -*-
"""
SUSPENSION EXPLORER  -  Temerario GT3 front-corner design tool
==============================================================

Type hard points on the left, get every kinematic graph on the right, live.

    cd suspension_tool
    python app.py

Everything is computed by the validated cores in ../software/ (see engine.py) --
this file is only the interface. Nothing here reimplements physics.

TABS
  Bump          camber / toe (bump steer) / caster & KPI vs wheel travel
  Heave & Roll  camber in heave, and camber vs body roll with recovery %
  Scrub & Trail vs wheel travel and vs steer
  RC & Damper   roll-centre height (both constructions) + motion ratio
  Steer         camber, scrub, trail and the contact-patch path vs steer
  Geometry      front and side views of the linkage as you have entered it
  Checks        tyre ride-height and rim-packaging validation

Every plot has a hover readout in the toolbar area (matplotlib's own coordinate
display) plus a crosshair; the Bump and Heave/Roll tabs add a snapping readout.

Buttons: Recalculate · Reset to stock · Load CSV · Save CSV · Export PNGs
"""
import os
import sys
import csv
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


class Tab:
    """A notebook page holding one matplotlib figure."""

    def __init__(self, nb, title, nrows=1, ncols=1, figsize=(11.5, 7.0)):
        self.frame = ttk.Frame(nb)
        nb.add(self.frame, text=title)
        self.fig = Figure(figsize=figsize, dpi=100)
        self.axes = self.fig.subplots(nrows, ncols, squeeze=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.frame).update()
        self.readout = None

    def ax(self, r=0, c=0):
        return self.axes[r][c]

    def clear(self):
        for row in self.axes:
            for a in row:
                a.clear()

    def draw(self):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def add_readout(self):
        """A monospace readout box pinned to the figure, updated on hover."""
        self.readout = self.fig.text(0.005, 0.005, "", fontsize=8.5,
                                     family="monospace", va="bottom", ha="left",
                                     bbox=dict(boxstyle="round", fc="white",
                                               ec="0.6", alpha=0.92))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Suspension Explorer — Temerario GT3 front corner")
        root.geometry("1620x950")

        self.pts = engine.default_points()
        self.static_camber = engine.default_static_camber()
        self.vars = {}
        self.sw = None

        self._build_left(root)
        self._build_right(root)
        self.recalculate()

    # ------------------------------------------------------------ left panel
    def _build_left(self, root):
        left = ttk.Frame(root, padding=8)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="HARD POINTS  [mm]", font=("Segoe UI", 11, "bold")
                  ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 2))
        ttk.Label(left, text="X = lateral (outboard +) · Y = longitudinal (fwd +) · Z = up",
                  foreground="#555").grid(row=1, column=0, columnspan=4, sticky="w",
                                          pady=(0, 6))
        for j, h in enumerate(("point", "X", "Y", "Z")):
            ttk.Label(left, text=h, font=("Segoe UI", 9, "bold")).grid(
                row=2, column=j, sticky="w", padx=(0, 4))

        r = 3
        for name in engine.POINT_NAMES:
            if name not in self.pts:
                continue
            ttk.Label(left, text=name, width=30).grid(row=r, column=0, sticky="w")
            row_vars = []
            for k in range(3):
                v = tk.StringVar(value=f"{self.pts[name][k]:.3f}")
                e = ttk.Entry(left, textvariable=v, width=10, justify="right")
                e.grid(row=r, column=1 + k, padx=1, pady=1)
                e.bind("<Return>", lambda _e: self.recalculate())
                row_vars.append(v)
            self.vars[name] = row_vars
            r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=4,
                                                      sticky="ew", pady=7)
        r += 1
        ttk.Label(left, text="SETUP", font=("Segoe UI", 10, "bold")).grid(
            row=r, column=0, sticky="w")
        r += 1
        ttk.Label(left, text="static camber [deg]").grid(row=r, column=0, sticky="w")
        self.camber_var = tk.StringVar(value=f"{self.static_camber:.2f}")
        e = ttk.Entry(left, textvariable=self.camber_var, width=10, justify="right")
        e.grid(row=r, column=1)
        e.bind("<Return>", lambda _e: self.recalculate())
        r += 1
        ttk.Label(left, text="droop / bump [mm]").grid(row=r, column=0, sticky="w")
        self.droop_var = tk.StringVar(value="-30")
        self.bump_var = tk.StringVar(value="70")
        for k, vv in enumerate((self.droop_var, self.bump_var)):
            e = ttk.Entry(left, textvariable=vv, width=10, justify="right")
            e.grid(row=r, column=1 + k)
            e.bind("<Return>", lambda _e: self.recalculate())
        r += 1
        ttk.Label(left, text="corner load [kg]").grid(row=r, column=0, sticky="w")
        self.load_var = tk.StringVar(value="304")
        e = ttk.Entry(left, textvariable=self.load_var, width=10, justify="right")
        e.grid(row=r, column=1)
        e.bind("<Return>", lambda _e: self.recalculate())
        r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=4,
                                                      sticky="ew", pady=7)
        r += 1
        bar = ttk.Frame(left)
        bar.grid(row=r, column=0, columnspan=4, sticky="w")
        for txt, cmd in (("Recalculate", self.recalculate),
                         ("Reset to stock", self.reset),
                         ("Load CSV", self.load_csv),
                         ("Save CSV", self.save_csv),
                         ("Export PNGs", self.export)):
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=2)
        r += 1

        ttk.Separator(left, orient="horizontal").grid(row=r, column=0, columnspan=4,
                                                      sticky="ew", pady=7)
        r += 1
        ttk.Label(left, text="DERIVED", font=("Segoe UI", 10, "bold")).grid(
            row=r, column=0, sticky="w")
        r += 1
        self.summary = tk.Text(left, width=46, height=15, font=("Consolas", 9),
                               relief="solid", borderwidth=1, wrap="none")
        self.summary.grid(row=r, column=0, columnspan=4, sticky="w", pady=(2, 0))

    # ------------------------------------------------------------ right panel
    def _build_right(self, root):
        nb = ttk.Notebook(root)
        nb.pack(side="right", fill="both", expand=True)
        self.t_bump = Tab(nb, "Bump", 1, 3)
        self.t_roll = Tab(nb, "Heave & Roll", 1, 2)
        self.t_scrub = Tab(nb, "Scrub & Trail", 1, 2)
        self.t_rc = Tab(nb, "RC & Damper", 1, 2)
        self.t_steer = Tab(nb, "Steer", 1, 3)
        self.t_geom = Tab(nb, "Geometry", 1, 2)
        self.t_check = Tab(nb, "Checks", 1, 1)
        for t in (self.t_bump, self.t_roll):
            t.add_readout()
        self.t_bump.canvas.mpl_connect("motion_notify_event", self._hover_bump)
        self.t_roll.canvas.mpl_connect("motion_notify_event", self._hover_roll)

    # ------------------------------------------------------------ data in/out
    def _read_points(self):
        pts = {}
        for name, vv in self.vars.items():
            pts[name] = np.array([float(v.get()) for v in vv], float)
        return pts

    def _write_points(self):
        for name, vv in self.vars.items():
            for k in range(3):
                vv[k].set(f"{self.pts[name][k]:.3f}")

    def reset(self):
        self.pts = engine.default_points()
        self.static_camber = engine.default_static_camber()
        self.camber_var.set(f"{self.static_camber:.2f}")
        self._write_points()
        self.recalculate()

    def load_csv(self):
        fn = filedialog.askopenfilename(
            initialdir=PROJ, title="Load hard points",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not fn:
            return
        try:
            got = {}
            with open(fn) as f:
                for row in csv.reader(f):
                    if len(row) < 5 or row[0].strip().lower() in ("corner", ""):
                        continue
                    if row[0].strip().upper() not in ("FRONT", ""):
                        continue
                    got[row[1].strip()] = np.array([float(row[2]), float(row[3]),
                                                    float(row[4])])
            missing = [n for n in self.vars if n not in got]
            if missing:
                messagebox.showwarning("Missing points",
                                       "Not found, keeping current:\n" + "\n".join(missing))
            for n in self.vars:
                if n in got:
                    self.pts[n] = got[n]
            self._write_points()
            self.recalculate()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def save_csv(self):
        fn = filedialog.asksaveasfilename(
            initialdir=PROJ, defaultextension=".csv", title="Save hard points",
            filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        pts = self._read_points()
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Corner", "Point", "X_mm", "Y_mm", "Z_mm"])
            for n in engine.POINT_NAMES:
                if n in pts:
                    w.writerow(["FRONT", n] + [f"{v:.3f}" for v in pts[n]])
        messagebox.showinfo("Saved", fn)

    def export(self):
        d = filedialog.askdirectory(initialdir=PROJ, title="Export PNGs to…")
        if not d:
            return
        for nm, t in (("bump", self.t_bump), ("heave_roll", self.t_roll),
                      ("scrub_trail", self.t_scrub), ("rc_damper", self.t_rc),
                      ("steer", self.t_steer), ("geometry", self.t_geom)):
            t.fig.savefig(os.path.join(d, f"tool_{nm}.png"), dpi=140)
        messagebox.showinfo("Exported", f"6 PNGs written to\n{d}")

    # ------------------------------------------------------------ the compute
    def recalculate(self):
        try:
            self.pts = self._read_points()
            self.static_camber = float(self.camber_var.get())
            droop = float(self.droop_var.get())
            bump = float(self.bump_var.get())
            load = float(self.load_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "All fields must be numbers.")
            return

        try:
            s = engine.statics(self.pts, self.static_camber)
            self.sw = engine.bump_sweep(self.pts, self.static_camber, droop, bump)
            self.rollc = engine.roll_curve(self.pts, self.static_camber,
                                           self.sw, s["track"])
            self.st = engine.steer_sweep(self.pts, self.static_camber)
            self.checks = engine.run_checks(self.pts, self.static_camber, load)
        except Exception as e:
            messagebox.showerror(
                "Solver failed",
                f"{e}\n\nUsually this means the linkage cannot reach the requested "
                f"travel — try a smaller bump/droop range.\n\n"
                f"{traceback.format_exc(limit=2)}")
            return

        self._fill_summary(s)
        self._plot_bump(s)
        self._plot_roll()
        self._plot_scrub()
        self._plot_rc()
        self._plot_steer(s)
        self._plot_geom()
        self._plot_checks()

    def _fill_summary(self, s):
        sw, i0 = self.sw, int(np.argmin(np.abs(self.sw["bump"])))
        txt = (
            f"KPI            {s['kpi']:+9.3f} deg\n"
            f"caster         {s['caster']:+9.3f} deg\n"
            f"static camber  {s['static_camber']:+9.3f} deg\n"
            f"scrub radius   {s['scrub']:+9.2f} mm\n"
            f"mech. trail    {s['trail']:+9.2f} mm\n"
            f"kingpin length {s['kingpin_len']:9.2f} mm\n"
            f"tie-rod length {s['tierod_len']:9.2f} mm\n"
            f"damper length  {s['damper_len']:9.2f} mm\n"
            f"eff steer arm  {s['eff_steer_arm']:9.1f} mm\n"
            f"  ({s['deg_per_mm_rack']:.3f} deg per mm of rack)\n"
            f"track          {s['track']:9.1f} mm\n"
            f"wheel c height {s['wc_height']:9.2f} mm\n"
            f"---- from the sweep ----\n"
            f"camber gain    {sw['camber_rate'][i0]:+9.5f} deg/mm\n"
            f"BUMP STEER     {sw['toe_rate'][i0]:+9.5f} deg/mm\n"
            f"motion ratio   {sw['MR'][i0]:9.4f}\n"
            f"RC (linkage)   {sw['rc_linkage'][i0]:+9.1f} mm\n"
            f"camber recovery{self.rollc['recovery_pct']:8.1f} %"
        )
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", txt)

    # ------------------------------------------------------------ plots
    def _plot_bump(self, s):
        t = self.t_bump
        t.clear()
        sw, b = self.sw, self.sw["bump"]
        for a, key, col, lab, ttl in (
                (t.ax(0, 0), "camber", "#1f77b4", "camber [deg]", "CAMBER vs travel"),
                (t.ax(0, 1), "toe", "#d62728", "toe [deg]  (+ = toe-in)",
                 "TOE = BUMP STEER")):
            a.plot(b, sw[key], lw=2.3, color=col)
            a.axvline(0, color="#888", lw=0.8)
            a.set_xlabel("wheel travel [mm]")
            a.set_ylabel(lab)
            a.grid(alpha=0.3)
        i0 = int(np.argmin(np.abs(b)))
        t.ax(0, 0).set_title(f"CAMBER vs travel — gain {sw['camber_rate'][i0]:+.4f} °/mm")
        t.ax(0, 1).set_title(f"BUMP STEER — {sw['toe_rate'][i0]:+.4f} °/mm at 0")
        t.ax(0, 1).axhspan(-0.4, 0.4, color="#2ca02c", alpha=0.10)
        a = t.ax(0, 2)
        a.plot(b, sw["caster"], lw=2.3, color="#C77D11", label="caster")
        a.plot(b, sw["kpi"], lw=2.3, color="#7C4DBE", label="KPI")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("wheel travel [mm]")
        a.set_ylabel("angle [deg]")
        a.set_title("CASTER & KPI vs travel")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        t.draw()

    def _plot_roll(self):
        t = self.t_roll
        t.clear()
        sw, rc = self.sw, self.rollc
        a = t.ax(0, 0)
        a.plot(sw["bump"], sw["camber"], lw=2.4, color="#1f77b4")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("wheel travel [mm]")
        a.set_ylabel("camber to road [deg]")
        a.set_title(f"CAMBER in HEAVE — gain {rc['gain0']:+.5f} °/mm\n"
                    "chassis does not rotate, so this is camber-to-road directly",
                    fontsize=10)
        a.grid(alpha=0.3)
        a = t.ax(0, 1)
        a.plot(rc["phi"], rc["cam_road"], lw=2.6, color="#d62728",
               label="camber TO ROAD = linkage + φ")
        a.plot(rc["phi"], rc["cam_body"], lw=1.9, ls="--", color="#1f77b4",
               label="linkage only (to chassis)")
        a.axhline(0, color="#c0392b", lw=1.1, ls="--")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("body roll  φ [deg]   (+ = rolls onto this wheel)")
        a.set_ylabel("camber [deg]")
        a.set_title(f"CAMBER in ROLL — recovery {rc['recovery_pct']:.1f} %\n"
                    "γ_road = γ_linkage(z) + φ,  z = (t/2)·tanφ", fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="best")
        t.draw()

    def _plot_scrub(self):
        t = self.t_scrub
        t.clear()
        sw = self.sw
        a = t.ax(0, 0)
        a.plot(sw["bump"], sw["scrub"], lw=2.3, color="#2ca02c", label="scrub")
        a.plot(sw["bump"], sw["trail"], lw=2.3, color="#17becf", label="trail")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("wheel travel [mm]")
        a.set_ylabel("length [mm]")
        a.set_title("SCRUB & TRAIL vs wheel travel")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a = t.ax(0, 1)
        st = self.st
        a.plot(st["steer"], st["scrub"], lw=2.3, color="#2ca02c", label="scrub")
        a.plot(st["steer"], st["trail"], lw=2.3, color="#17becf", label="trail")
        a.axhline(0, color="#c0392b", lw=1.0, ls="--")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("steer angle [deg]")
        a.set_ylabel("length [mm]")
        a.set_title("SCRUB & TRAIL vs steer  (trail < 0 ⇒ de-centring)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        t.draw()

    def _plot_rc(self):
        t = self.t_rc
        t.clear()
        sw = self.sw
        a = t.ax(0, 0)
        a.plot(sw["bump"], sw["rc_linkage"], lw=2.4, color="#1f77b4",
               label="linkage RC (classical)")
        a.plot(sw["bump"], sw["rc_wheel"], lw=2.0, ls="--", color="#d62728",
               label="wheel-motion RC")
        a.axhline(0, color="#444", lw=1.1)
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("wheel travel [mm]")
        a.set_ylabel("roll-centre height [mm]")
        a.set_title("ROLL CENTRE vs travel   (0 = ground)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a = t.ax(0, 1)
        a.plot(sw["bump"], sw["damper"], lw=2.3, color="#333")
        a.set_xlabel("wheel travel [mm]")
        a.set_ylabel("damper length [mm]")
        a2 = a.twinx()
        a2.plot(sw["bump"], sw["MR"], lw=2.3, color="#e377c2")
        a2.set_ylabel("motion ratio [-]", color="#e377c2")
        a2.tick_params(axis="y", labelcolor="#e377c2")
        a.axvline(0, color="#888", lw=0.8)
        a.set_title("DAMPER LENGTH & MOTION RATIO")
        a.grid(alpha=0.3)
        t.draw()

    def _plot_steer(self, s):
        t = self.t_steer
        t.clear()
        st = self.st
        a = t.ax(0, 0)
        a.plot(st["steer"], st["camber"], lw=2.3, color="#1f77b4")
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("steer [deg]")
        a.set_ylabel("camber [deg]")
        a.set_title("CAMBER vs steer (kingpin axis)")
        a.grid(alpha=0.3)
        a = t.ax(0, 1)
        a.plot(st["steer"], st["d_track"], lw=2.2, color="#1f77b4", label="track")
        a.plot(st["steer"], st["d_base"], lw=2.2, color="#2ca02c", label="wheelbase")
        a.plot(st["steer"], st["jack"], lw=2.2, color="#d62728", label="jacking")
        a.axhline(0, color="#888", lw=0.8)
        a.axvline(0, color="#888", lw=0.8)
        a.set_xlabel("steer [deg]")
        a.set_ylabel("movement [mm]")
        a.set_title("PATCH MOTION & JACKING vs steer")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a = t.ax(0, 2)
        G, P = st["G"], st["P"]
        a.plot(P[:, 0], P[:, 1], lw=2.2, color="#1f3fbf")
        a.plot([G[0]], [G[1]], "X", ms=12, color="#c0392b")
        a.set_xlabel("X lateral [mm]")
        a.set_ylabel("Y longitudinal [mm]")
        a.set_title("CONTACT-PATCH PATH  (X = axis at ground)")
        a.set_aspect("equal")
        a.grid(alpha=0.3)
        t.draw()

    def _plot_geom(self):
        t = self.t_geom
        t.clear()
        p = self.pts
        pairs_f = [("Upper wishbone - inner front", "Upper wishbone - outer (UBJ)"),
                   ("Upper wishbone - inner rear", "Upper wishbone - outer (UBJ)"),
                   ("Lower wishbone - inner front", "Lower wishbone - outer (LBJ)"),
                   ("Lower wishbone - inner rear", "Lower wishbone - outer (LBJ)"),
                   ("Tie rod - inner", "Tie rod - outer"),
                   ("Lower Shock mount", "Upper Shock mount"),
                   ("Upper wishbone - outer (UBJ)", "Lower wishbone - outer (LBJ)")]
        cols = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728", "#333", "#7C4DBE"]
        for idx, (a, ix, iz, xl, yl, ttl) in enumerate((
                (t.ax(0, 0), 0, 2, "X lateral [mm]", "Z up [mm]", "FRONT VIEW"),
                (t.ax(0, 1), 1, 2, "Y longitudinal [mm]", "Z up [mm]", "SIDE VIEW"))):
            for (n1, n2), c in zip(pairs_f, cols):
                a.plot([p[n1][ix], p[n2][ix]], [p[n1][iz], p[n2][iz]], lw=2.0, color=c)
            for n in engine.POINT_NAMES:
                a.plot([p[n][ix]], [p[n][iz]], "o", ms=5, color="k")
            wc = p["Wheel centre"]
            a.plot([wc[ix]], [wc[iz]], "o", ms=9, mfc="none", mec="k", mew=1.6)
            a.axhline(0, color="#444", lw=1.2)
            a.set_xlabel(xl)
            a.set_ylabel(yl)
            a.set_title(ttl + "  (purple = kingpin axis, red = tie rod)")
            a.set_aspect("equal")
            a.grid(alpha=0.3)
        t.draw()

    def _plot_checks(self):
        t = self.t_check
        t.clear()
        a = t.ax(0, 0)
        a.axis("off")
        y = 0.95
        a.text(0.02, y, "VALIDATION CHECKS", fontsize=13, fontweight="bold",
               transform=a.transAxes)
        y -= 0.07
        for lvl, msg in self.checks:
            col = {"OK": "#1a6b1a", "WARN": "#b06000", "FAIL": "#b00000"}[lvl]
            a.text(0.02, y, f"[{lvl}]", fontsize=10.5, color=col, family="monospace",
                   fontweight="bold", transform=a.transAxes)
            a.text(0.11, y, msg, fontsize=10.5, transform=a.transAxes)
            y -= 0.055
        y -= 0.03
        a.text(0.02, y, "Tyre: " + f"{engine.tyre.NAME} on {engine.tyre.RIM}, "
               f"free radius {engine.tyre.FREE_RADIUS_MM:.1f} mm, "
               f"radial stiffness {engine.tyre.radial_stiffness_N_per_mm():.0f} N/mm",
               fontsize=9.5, color="#444", transform=a.transAxes)
        y -= 0.05
        a.text(0.02, y, "Physics from ../software/, validated by validate_scrub_trail.py "
               "(33 checks) and validate_wishbone.py (27 checks).",
               fontsize=9.5, color="#444", style="italic", transform=a.transAxes)
        t.draw()

    # ------------------------------------------------------------ hover
    def _hover_bump(self, ev):
        if ev.inaxes is None or ev.xdata is None or self.sw is None:
            return
        b = self.sw["bump"]
        i = int(np.argmin(np.abs(b - ev.xdata)))
        self.t_bump.readout.set_text(
            f"travel {b[i]:+7.1f} mm | camber {self.sw['camber'][i]:+7.3f}° "
            f"| toe {self.sw['toe'][i]:+7.3f}° | caster {self.sw['caster'][i]:+6.3f}° "
            f"| KPI {self.sw['kpi'][i]:+6.3f}° | scrub {self.sw['scrub'][i]:+7.2f} "
            f"| MR {self.sw['MR'][i]:.4f}")
        self.t_bump.canvas.draw_idle()

    def _hover_roll(self, ev):
        if ev.inaxes is None or ev.xdata is None or self.sw is None:
            return
        rc = self.rollc
        if ev.inaxes is self.t_roll.ax(0, 0):
            b = self.sw["bump"]
            i = int(np.argmin(np.abs(b - ev.xdata)))
            txt = (f"travel {b[i]:+7.1f} mm | camber {self.sw['camber'][i]:+7.3f}°")
        else:
            i = int(np.argmin(np.abs(rc["phi"] - ev.xdata)))
            txt = (f"roll {rc['phi'][i]:+6.2f}° | to-road {rc['cam_road'][i]:+7.3f}° "
                   f"| linkage {rc['cam_body'][i]:+7.3f}° "
                   f"| recovery {rc['recovery_pct']:.1f} %")
        self.t_roll.readout.set_text(txt)
        self.t_roll.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
