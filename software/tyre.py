# -*- coding: utf-8 -*-
"""
Tyre geometry: free radius, loaded radius and radial stiffness.
===============================================================

WHAT THIS IS FOR
----------------
Two numbers from the tyre reach almost everything else in the tool:

  * **loaded radius** sets the wheel-centre height, which is the datum every
    kinematic quantity is measured from -- scrub, trail, roll centre, FVSA.
  * **radial stiffness** sits in SERIES with the wheel rate, so it sets the
    difference between the "ride frequency" a spring calculator reports and the
    frequency the body actually bounces at. On a racing car the wheel rate is
    roughly half the tyre rate, so the tyre is worth 15 % of the frequency and
    cannot be ignored the way a road-car textbook ignores it.

WHERE THE NUMBERS COME FROM
---------------------------
The fitment (`325/680-18`) is a published size: 680 mm nominal overall
diameter, hence a 340 mm free radius.

The load/deflection behaviour is NOT published, and no measured set is in this
repository. Instead the module ships a clearly-labelled **placeholder**:
a linear carcass of `PLACEHOLDER_STIFFNESS_N_PER_MM`, which is the right order
for an 18-inch racing slick and is enough for every routine here to run.

Install real data by filling `TYRE_*` in `benchmarks_private.py` (see
`benchmarks_private.example.py`). `MEASURED` tells you which you are running on,
and the tool prints it rather than letting you assume.

FRAME: all radii in mm, loads in kg of vertical wheel load.
"""

import numpy as np

import benchmarks as bm

# --------------------------------------------------------- published fitment
NAME = bm.get("TYRE_NAME", "325/680-18")
RIM = '18" x 12J'
RIM_DIAMETER_IN = 18.0
RIM_WIDTH_J = 12.0

# 680 mm nominal overall diameter -> 340 mm free radius. Replaced by the
# measured free rolling radius when private data is installed.
FREE_RADIUS_MM = float(bm.get("TYRE_FREE_RADIUS_MM", 340.0))

# ------------------------------------------------- load / deflection behaviour
# PLACEHOLDER. A linear carcass at this stiffness, not a measurement. It is the
# right order of magnitude for an 18-inch racing slick and nothing more; every
# number this module derives inherits that caveat.
PLACEHOLDER_STIFFNESS_N_PER_MM = 300.0

_load = bm.get("TYRE_LOAD_KG")
_radius = bm.get("TYRE_LOADED_RADIUS_MM")
MEASURED = _load is not None and _radius is not None

if MEASURED:
    LOAD_KG = np.asarray(_load, float)
    LOADED_RADIUS_MM = np.asarray(_radius, float)
else:
    LOAD_KG = np.array([150.0, 300.0, 450.0, 600.0, 750.0])
    LOADED_RADIUS_MM = (FREE_RADIUS_MM
                        - LOAD_KG * 9.81 / PLACEHOLDER_STIFFNESS_N_PER_MM)

SOURCE = "measured (private data installed)" if MEASURED else "PLACEHOLDER"


def loaded_radius(load_kg):
    """Loaded radius [mm] at a vertical wheel load [kg]. Linear interpolation
    of the table; extrapolation is clamped to its range."""
    return float(np.interp(load_kg, LOAD_KG, LOADED_RADIUS_MM))


def load_for_radius(radius_mm):
    """Inverse: what vertical load [kg] puts the wheel centre at this height?
    Useful for asking 'what load does the CAD ride height correspond to?'."""
    return float(np.interp(radius_mm, LOADED_RADIUS_MM[::-1], LOAD_KG[::-1]))


def radial_stiffness_N_per_mm():
    """Vertical (radial) tyre stiffness [N/mm] from the load-deflection slope.
    A straight-line fit over the whole range. See SOURCE for its provenance."""
    defl = LOADED_RADIUS_MM[0] - LOADED_RADIUS_MM          # mm of deflection
    force = (LOAD_KG - LOAD_KG[0]) * 9.81                  # N
    return float(np.polyfit(defl, force, 1)[0])


def deflection_mm(load_kg):
    """Carcass deflection from free radius at a given load."""
    return FREE_RADIUS_MM - loaded_radius(load_kg)


def check_ride_height(wc_height_mm, corner_load_kg):
    """Cross-check the CAD wheel-centre height against the measured tyre.

    Returns a dict: what the tyre says the radius should be at that load, what
    the CAD height implies about the load, and the mismatch.
    """
    r_expected = loaded_radius(corner_load_kg)
    load_implied = load_for_radius(wc_height_mm)
    return dict(
        wc_height=wc_height_mm,
        corner_load=corner_load_kg,
        radius_expected=r_expected,
        delta_mm=wc_height_mm - r_expected,
        load_implied=load_implied,
        load_error_kg=load_implied - corner_load_kg,
    )


if __name__ == "__main__":
    import config as cfg

    p = cfg._load_front_points()
    wc_z = float(p["Wheel centre"][2])
    MASS_KG, FRONT_FRAC = 1415.0, 0.43
    corner = MASS_KG * FRONT_FRAC / 2.0

    print(f"TYRE  {NAME}   data source: {SOURCE}")
    print(f"  free radius            {FREE_RADIUS_MM:.1f} mm")
    print(f"  radial stiffness       {radial_stiffness_N_per_mm():.0f} N/mm")
    print(f"  loaded radius @{corner:.0f} kg  {loaded_radius(corner):.1f} mm")
    r = check_ride_height(wc_z, corner)
    print(f"  CAD wheel centre       {wc_z:.1f} mm "
          f"({r['delta_mm']:+.1f} mm vs the tyre)")
    if not MEASURED:
        print("\n  NOTE: placeholder tyre. Install private data for real "
              "numbers -- see benchmarks_private.example.py")
