# -*- coding: utf-8 -*-
"""Template for the git-ignored `benchmarks_private.py`.

Copy to `benchmarks_private.py` and fill in from a reference car you are
entitled to use. Nothing here is required: leave an entry out and the code
hides that comparison rather than failing.

    cp benchmarks_private.example.py benchmarks_private.py
"""

import numpy as np

# --------------------------------------------------------------- anti geometry
# Read off the reference car's published kinematic curves at zero wheel travel.
#   anti_dive_f   front, braking, contact-patch reference  [%]
#   anti_lift_r   rear,  braking, contact-patch reference  [%]
#   anti_squat_r  rear,  traction, WHEEL-CENTRE reference (inboard diff)  [%]
ANTI_PCT = None            # e.g. dict(anti_dive_f=30.0, anti_lift_r=40.0,
#                                      anti_squat_r=-4.0)
ANTI_SLOPE = None          # d(anti)/d(wheel travel), [% per mm], same keys

# ------------------------------------------------------------------- roll axis
ROLL_CENTRE_MM = None      # (front, rear) roll-centre height at ride  [mm]
ROLL_CENTRE_SLOPE = None   # (front, rear) d(RC height)/d(wheel travel)  [-]

# --------------------------------------------------------------------- steering
ACKERMANN_PCT = None       # (low, high) the reference axle's Ackermann band [%]

# ------------------------------------------------------------------------ tyre
# Measured load/deflection from a tyre data booklet. Supply all three or none.
TYRE_NAME = None           # e.g. "325/680-18"
TYRE_LOAD_KG = None        # e.g. np.array([150., 300., 450., 600., 750.])
TYRE_LOADED_RADIUS_MM = None   # loaded radius at each of those loads  [mm]
TYRE_FREE_RADIUS_MM = None     # free rolling radius  [mm]

# ------------------------------------------------- damper motion ratios, R_M
MOTION_RATIO = None        # (front, rear), spring travel / wheel travel

# --------------------------------------------------- target ride frequencies
RIDE_FREQ_HZ = None        # (front, rear), wheel-rate definition  [Hz]
