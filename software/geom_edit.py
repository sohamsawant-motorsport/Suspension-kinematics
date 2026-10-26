# -*- coding: utf-8 -*-
"""
GEOMETRY EDITS  -  turn a DESIGN TARGET into hard points.
==========================================================

Every other module in this project reads hard points and reports angles. For
design work we need the inverse: "I want KPI = 7 deg and caster = 8 deg -- where
does the ball joint have to go?"

The rule used here
------------------
    LBJ is FIXED.  The kingpin LENGTH is FIXED.  Only the UBJ moves.

Why that rule and not another:

  * The LBJ is the most constrained point on the corner. It is jammed between the
    rim's inner face, the brake disc and the lower arm; you do not move it for
    free. The UBJ has room above the disc.
  * Holding |UBJ - LBJ| fixed keeps the upright the same SIZE, so the change is a
    re-orientation of the steering axis rather than a whole new upright casting.
  * The wheel centre does NOT move. Ride height and track are set by the wheel and
    the tyre, not by the kingpin, so they must stay put while we tune the axis.
    The consequence is that the UPRIGHT reshapes -- the UBJ-to-wheel-centre arm
    changes -- which is exactly what a real upright redesign does.

Everything downstream then follows automatically: the upper wishbone gets longer
or shorter (reported, because it is a real packaging consequence), scrub and trail
move because the axis pierces the ground somewhere new, and the bump curves change
because the solver sees a different upright.

The angle convention is the ONE used everywhere in this project
(config.stock_geometry_from_csv and scrub_trail.kpi_caster), with d = UBJ - LBJ:

    KPI    nu     = atan2(-dX, dZ)      > 0 = axis leans INBOARD at the top
    caster lambda = atan2(-dY, dZ)      > 0 = axis leans REARWARD at the top

Inverting it is closed form -- no iteration. Write t_nu = tan(nu), t_l = tan(lam):

    dX = -dZ * t_nu ,   dY = -dZ * t_lam                                    (1)

so the length constraint |d| = L gives

    L^2 = dZ^2 (1 + t_nu^2 + t_lam^2)
    dZ  = L / sqrt(1 + t_nu^2 + t_lam^2)                                    (2)

taking the positive root because the UBJ is always ABOVE the LBJ. Substituting (2)
back into (1) finishes the job. Round-tripping this through kpi_caster() returns
the requested angles to machine precision -- asserted in validate_geom_edit.py.
"""
import numpy as np

UBJ_NAME = "Upper wishbone - outer (UBJ)"
LBJ_NAME = "Lower wishbone - outer (LBJ)"


def ubj_for_kpi_caster(LBJ, UBJ_ref, kpi_deg, caster_deg):
    """Where the UBJ must sit for this (KPI, caster), keeping the kingpin length.

    LBJ, UBJ_ref : the CURRENT ball joints. Only UBJ_ref's DISTANCE from the LBJ
                   is used -- its direction is replaced by the requested angles.
    Returns the new UBJ as a length-3 array.
    """
    LBJ = np.asarray(LBJ, float)
    L = float(np.linalg.norm(np.asarray(UBJ_ref, float) - LBJ))
    t_nu = np.tan(np.radians(float(kpi_deg)))
    t_lam = np.tan(np.radians(float(caster_deg)))
    dZ = L / np.sqrt(1.0 + t_nu * t_nu + t_lam * t_lam)      # eq (2), + root
    return LBJ + np.array([-dZ * t_nu, -dZ * t_lam, dZ])     # eq (1)


def retarget(pts, kpi_deg, caster_deg):
    """A COPY of the hard-point dict with the UBJ moved to hit (KPI, caster).

    Nothing else is touched, so the result can be handed straight to
    engine.statics / engine.bump_sweep / wishbone_solver.corner_geometry.
    """
    out = {k: np.asarray(v, float).copy() for k, v in pts.items()}
    out[UBJ_NAME] = ubj_for_kpi_caster(pts[LBJ_NAME], pts[UBJ_NAME],
                                       kpi_deg, caster_deg)
    return out


def upper_arm_length(pts):
    """Mean UBJ distance to the two upper inner pickups.

    Moving the UBJ re-lengths the upper wishbone. That is a real fabrication and
    packaging consequence of a KPI change, so the tool reports it rather than
    hiding it.
    """
    U = np.asarray(pts[UBJ_NAME], float)
    a = np.linalg.norm(U - np.asarray(pts["Upper wishbone - inner front"], float))
    b = np.linalg.norm(U - np.asarray(pts["Upper wishbone - inner rear"], float))
    return float(0.5 * (a + b))


def ubj_travel(pts_a, pts_b):
    """How far the UBJ actually moved between two geometries [mm], and the
    per-axis breakdown -- the number you take to CAD to check clearance."""
    d = np.asarray(pts_b[UBJ_NAME], float) - np.asarray(pts_a[UBJ_NAME], float)
    return float(np.linalg.norm(d)), d
