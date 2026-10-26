# -*- coding: utf-8 -*-
"""
Engine layer for the Suspension Explorer tool.
==============================================

The tool does NOT reimplement any physics. Every number it draws comes from the
validated cores in `../software/`:

    steer_camber.py     camber vs steer about the kingpin axis   (validated)
    scrub_trail.py      contact patch, scrub, trail, jacking     (33 assertions)
    wishbone_solver.py  full double-wishbone position solve      (27 assertions)
    tyre.py             tyre geometry and radial stiffness
    rim_envelope.py     packaging limits                         (CAD + tyre)

This module is a thin adapter: it takes a dict of hard points from the GUI,
runs the cores, and hands back plain arrays for plotting. If a result here ever
disagreed with `software/validate_*.py`, the validation suites are the truth.

Hard-point keys are exactly the CSV column names, so a geometry loaded in the GUI
can be written straight back out to `hardpoints_fusion.csv` format.
"""
import os
import sys

import numpy as np

_SW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "software")
if _SW not in sys.path:
    sys.path.insert(0, _SW)

import tyre                                                    # noqa: E402
import rim_envelope                                            # noqa: E402
from scrub_trail import (kpi_caster, scrub_radius, mechanical_trail,   # noqa: E402
                         corner_state, contact_patch, ground_point)
from wishbone_solver import (corner_geometry, solve_sweep,      # noqa: E402
                            roll_centre_curve, linkage_roll_centre, solve_at)

# the point names the engine needs, in the order the GUI shows them
POINT_NAMES = [
    "Upper wishbone - inner front",
    "Upper wishbone - inner rear",
    "Upper wishbone - outer (UBJ)",
    "Lower wishbone - inner front",
    "Lower wishbone - inner rear",
    "Lower wishbone - outer (LBJ)",
    "Tie rod - inner",
    "Tie rod - outer",
    "Wheel centre",
    "Lower Shock mount",
    "Upper Shock mount",
]

UBJ_NAME = "Upper wishbone - outer (UBJ)"
LBJ_NAME = "Lower wishbone - outer (LBJ)"
TIE_NAME = "Tie rod - outer"


def default_points():
    """The stock (Huracan CAD) front corner, straight from the project CSV."""
    import config as cfg
    p = cfg._load_front_points()
    return {k: np.asarray(p[k], float).copy() for k in POINT_NAMES if k in p}


def default_static_camber():
    import config as cfg
    return float(cfg.STATIC_CAMBER)


# ---------------------------------------------------------------- statics
def statics(pts, static_camber):
    """Everything that does not need a sweep: the angles and the lengths."""
    LBJ, UBJ, WC = pts[LBJ_NAME], pts[UBJ_NAME], pts["Wheel centre"]
    TRO, TRI = pts[TIE_NAME], pts["Tie rod - inner"]
    kpi, cas = kpi_caster(LBJ, UBJ)
    G = ground_point(LBJ, UBJ)
    P = contact_patch(WC, static_camber)

    # exact steering ratio (velocity relation; see config.steer_angle_for_rack)
    t = TRO - TRI
    t = t / np.linalg.norm(t)
    k = UBJ - LBJ
    k = k / np.linalg.norm(k)
    denom = float(np.dot(np.cross(k, TRO - LBJ), t))
    ratio = t[0] / denom if abs(denom) > 1e-12 else np.nan      # rad per mm rack

    return dict(
        kpi=kpi, caster=cas, static_camber=static_camber,
        scrub=scrub_radius(LBJ, UBJ, WC, static_camber),
        trail=mechanical_trail(LBJ, UBJ, WC, static_camber),
        G=G, patch=P,
        kingpin_len=float(np.linalg.norm(UBJ - LBJ)),
        tierod_len=float(np.linalg.norm(TRO - TRI)),
        damper_len=float(np.linalg.norm(pts["Upper Shock mount"] - pts["Lower Shock mount"])),
        eff_steer_arm=float(1.0 / abs(ratio)) if np.isfinite(ratio) else np.nan,
        deg_per_mm_rack=float(np.degrees(abs(ratio))) if np.isfinite(ratio) else np.nan,
        track=2.0 * float(WC[0]),
        wc_height=float(WC[2]),
    )


# ---------------------------------------------------------------- sweeps
def bump_sweep(pts, static_camber, droop=-30.0, bump=70.0, n=161):
    """Full kinematic sweep vs wheel travel. Returns None on a linkage that
    cannot reach the requested travel (the GUI shows the reason)."""
    g = corner_geometry(pts, static_camber)
    travel = np.linspace(droop, bump, n)
    sw = solve_sweep(g, travel, theta_span_deg=28.0, n_theta=1121)
    rc = roll_centre_curve(sw, 2.0 * g["WC"][0])
    sw["rc_wheel"] = rc["rc_h"]
    sw["rc_linkage"] = linkage_roll_centre(g, travel)
    sw["camber_rate"] = np.gradient(sw["camber"], travel)
    sw["toe_rate"] = np.gradient(sw["toe"], travel)
    return sw


def roll_curve(pts, static_camber, sw, track_mm):
    """Camber vs BODY ROLL for this (outside) wheel.

        z(phi) = (t/2) tan(phi)          suspension compression from roll
        gamma_road = gamma_linkage(z) + phi

    The body roll adds directly: the whole suspension tilts with the car.
    'Camber recovery' is the fraction the linkage cancels.
    """
    half = track_mm / 2.0
    z = sw["bump"]
    phi = np.degrees(np.arctan(z / half))
    cam_body = sw["camber"]
    i0 = int(np.argmin(np.abs(z)))
    gain0 = sw["camber_rate"][i0]
    recovery = -half * np.radians(1.0) * gain0 * 100.0
    return dict(phi=phi, cam_body=cam_body, cam_road=cam_body + phi,
                recovery_pct=recovery, gain0=gain0)


def steer_sweep(pts, static_camber, lo=-35.0, hi=35.0, n=281):
    """Camber / scrub / trail / patch motion vs steer, at static ride height."""
    LBJ, UBJ, WC = pts[LBJ_NAME], pts[UBJ_NAME], pts["Wheel centre"]
    steer = np.linspace(lo, hi, n)
    st = corner_state(LBJ, UBJ, WC, steer, static_camber)
    st["steer"] = steer
    return st


# ---------------------------------------------------------------- checks
def run_checks(pts, static_camber, corner_load_kg=304.0):
    """Tyre + packaging validation. Returns a list of (level, message)."""
    out = []
    WC = pts["Wheel centre"]

    c = tyre.check_ride_height(float(WC[2]), corner_load_kg)
    lvl = "WARN" if abs(c["delta_mm"]) > 1.0 else "OK"
    out.append((lvl,
                f"Ride height: wheel centre {c['wc_height']:.1f} mm vs tyre's "
                f"{c['radius_expected']:.1f} mm at {corner_load_kg:.0f} kg "
                f"({c['delta_mm']:+.1f} mm; implies {c['load_implied']:.0f} kg)"))

    rim = rim_envelope.rim_geometry(float(WC[0]))
    rim["wc_y"] = float(WC[1])
    for nm, key in [("UBJ", UBJ_NAME), ("LBJ", LBJ_NAME), ("tie-rod outer", TIE_NAME)]:
        ch = rim_envelope.check_point(nm, pts[key], rim, float(WC[2]))
        lvl = "OK" if ch["ok_lateral"] else "FAIL"
        out.append((lvl, f"Packaging {nm}: {ch['lat_to_face']:+.1f} mm inboard of the "
                         f"mounting face (need > {rim_envelope.MIN_CLEARANCE_MM:.0f})"))

    s = statics(pts, static_camber)
    out.append(("OK" if s["scrub"] > 0 else "WARN",
                f"Scrub radius {s['scrub']:+.1f} mm"))
    out.append(("OK" if s["trail"] > 0 else "WARN",
                f"Mechanical trail {s['trail']:+.1f} mm"))
    return out
