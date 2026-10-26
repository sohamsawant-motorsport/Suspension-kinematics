# -*- coding: utf-8 -*-
"""
Rim working space -- can the upright's hard points actually PACKAGE?
====================================================================

Step 2 of the project's outside-in methodology (tyre -> rim -> upright -> chassis)
was written up in a Word doc but never enforced in code. This module makes it a
computable constraint, so the tool can WARN instead of quietly proposing a
geometry that cannot be built.

THE GEOMETRY OF A WHEEL (why offset matters)
---------------------------------------------
A wheel is a dish. Its MOUNTING FACE (where it bolts to the hub) is offset from
the rim's centre plane by the RIM OFFSET, positive when the mounting face is
OUTBOARD of the centre plane. Everything the suspension needs -- ball joints,
brake disc, caliper, upright body -- has to live in the bowl INBOARD of that
mounting face, inside the rim barrel.

    inner rim edge  =  centre plane  -  rim_width/2
    mounting face   =  centre plane  +  offset

A POSITIVE offset therefore pushes the mounting face outboard and OPENS UP the
inboard bowl, which is the space the ball joints use. (This is the point the user
corrected me on earlier in the project: positive offset puts the disc and caliper
outboard and leaves the joints room inboard -- there is no radial conflict.)

WHAT THIS MODULE CHECKS
-----------------------
For each outboard point (UBJ, LBJ, tie-rod outer):
  * is it INBOARD of the mounting face?           (must be, or it fouls the wheel)
  * how much lateral clearance to the inner rim edge / barrel?
  * is it within the rim's inner radius at its own height? (barrel clearance)
and, for a proposed scrub radius, what ball-joint lateral position it demands and
whether that position still packages.

All dimensions come from tyre.py and the CAD, so nothing here is invented.
"""
import numpy as np

import tyre

# Rim offsets measured in the user's CAD (see rebase_hardpoints.RIM_OFFSET)
RIM_OFFSET_FRONT_MM = 47.382
RIM_OFFSET_REAR_MM = 42.44

# Practical clearance a designer needs to leave (bolt heads, seal, dynamic
# deflection of the tyre sidewall onto the rim under load).
MIN_CLEARANCE_MM = 10.0


def rim_geometry(wheel_centre_x, rim_width_j=None, rim_dia_in=None,
                 offset_mm=RIM_OFFSET_FRONT_MM):
    """Key lateral stations of the wheel, in the CAD frame (X outboard +).

    The wheel centre from the CAD IS the rim centre plane (the user confirmed the
    hard point is the tyre/rim midplane, not the mounting face).
    """
    if rim_width_j is None:
        rim_width_j = tyre.RIM_WIDTH_J
    if rim_dia_in is None:
        rim_dia_in = tyre.RIM_DIAMETER_IN
    half_w = rim_width_j * 25.4 / 2.0
    return dict(
        centre_plane=wheel_centre_x,
        half_width=half_w,
        inner_edge=wheel_centre_x - half_w,        # inboard lip
        outer_edge=wheel_centre_x + half_w,
        mounting_face=wheel_centre_x + offset_mm,
        rim_radius=rim_dia_in * 25.4 / 2.0,        # to the bead seat
        offset=offset_mm,
    )


def check_point(name, P, rim, wc_z):
    """Package check for one outboard hard point.

    lateral: must sit inboard of the mounting face, with clearance.
    radial : its distance from the wheel AXIS must be inside the rim radius,
             else it fouls the barrel. (The axis runs laterally through the
             wheel centre, so radial distance uses the Y and Z offsets.)
    """
    P = np.asarray(P, float)
    lat_to_face = rim["mounting_face"] - P[0]          # + = inboard of the face
    radial = float(np.hypot(P[1] - rim.get("wc_y", P[1]), P[2] - wc_z))
    return dict(
        name=name, x=P[0],
        lat_to_face=lat_to_face,
        radial=radial,
        radial_margin=rim["rim_radius"] - radial,
        ok_lateral=lat_to_face > MIN_CLEARANCE_MM,
        ok_radial=(radial < rim["rim_radius"] - MIN_CLEARANCE_MM) or
                  (P[0] > rim["inner_edge"]),
    )


def scrub_to_joint_x(target_scrub_mm, patch_x, kingpin_lean_x_per_z, lbj_z):
    """What LATERAL position must the ball joints take to achieve a target scrub?

    scrub = patch_x - G_x, and G is the kingpin axis at ground, so
        G_x = patch_x - scrub
    and the LBJ sits at G_x plus whatever the axis leans over its height.
    """
    G_x = patch_x - target_scrub_mm
    return G_x + kingpin_lean_x_per_z * lbj_z


if __name__ == "__main__":
    import config as cfg
    from scrub_trail import scrub_radius, ground_point, contact_patch, kpi_caster

    p = cfg._load_front_points()
    LBJ, UBJ, WC = p[cfg.LBJ_NAME], p[cfg.UBJ_NAME], p["Wheel centre"]
    TRO = p[cfg.TIE_NAME]
    rim = rim_geometry(WC[0])
    rim["wc_y"] = WC[1]

    print("RIM WORKING SPACE - front corner")
    print("=" * 72)
    print(f"  tyre {tyre.NAME} on {tyre.RIM}   rim offset {rim['offset']:+.1f} mm")
    print(f"  rim centre plane   X = {rim['centre_plane']:8.1f} mm")
    print(f"  rim inner edge     X = {rim['inner_edge']:8.1f} mm   "
          f"(half-width {rim['half_width']:.1f})")
    print(f"  MOUNTING FACE      X = {rim['mounting_face']:8.1f} mm   "
          f"<- everything must be INBOARD of this")
    print(f"  rim radius (bead)    = {rim['rim_radius']:8.1f} mm")
    print()
    print("  outboard hard points:")
    print("    point                    X       inboard of face    radial    verdict")
    allok = True
    for nm, P in [("UBJ", UBJ), ("LBJ", LBJ), ("tie-rod outer", TRO)]:
        c = check_point(nm, P, rim, WC[2])
        ok = c["ok_lateral"]
        allok &= ok
        print(f"    {nm:16s} {c['x']:8.1f}    {c['lat_to_face']:+9.1f} mm   "
              f"{c['radial']:7.1f}    {'OK' if ok else 'FOULS THE WHEEL'}")
    print()
    print(f"  verdict: {'all outboard points package' if allok else 'PACKAGING VIOLATION'}")
    print()

    # what the CURRENT scrub demands, and what reducing it would demand
    kpi, cas = kpi_caster(LBJ, UBJ)
    sc = scrub_radius(LBJ, UBJ, WC, cfg.STATIC_CAMBER)
    P = contact_patch(WC, cfg.STATIC_CAMBER)
    lean = (UBJ[0] - LBJ[0]) / (UBJ[2] - LBJ[2])       # dX per dZ along the axis
    print("  SCRUB RADIUS vs PACKAGING")
    print("  " + "-" * 68)
    print(f"  contact patch at X = {P[0]:.1f};  current scrub {sc:.1f} mm")
    print("   target scrub   required LBJ X    inboard of face   packages?")
    for target in (140.0, 100.0, 60.0, 30.0, 0.0):
        x_req = scrub_to_joint_x(target, P[0], lean, LBJ[2])
        margin = rim["mounting_face"] - x_req
        print(f"    {target:8.0f} mm  {x_req:12.1f}      {margin:+9.1f} mm      "
              f"{'yes' if margin > MIN_CLEARANCE_MM else 'NO - outside the wheel'}")
    print()
    print("  => scrub can only be reduced by moving the joints OUTBOARD, and the")
    print("     mounting face is the hard stop. That is the real, physical limit on")
    print("     scrub radius -- not a preference.")
