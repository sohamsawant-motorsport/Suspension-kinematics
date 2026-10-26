# -*- coding: utf-8 -*-
"""
Spring rates from ride frequency -- and the two definitions of "ride frequency".
================================================================================

THE THING THAT CONFUSES EVERYONE
--------------------------------
"Ride frequency" is used for TWO different quantities, and the published
benchmark bands belong to only one of them.

1. WHEEL-RATE frequency (OptimumG, "Springs & Dampers Tech Tip 1", Giaraffa).
   The tyre is not in the model at all:

        K_s = 4 pi^2 f^2 m_sm MR^2          (their MR = wheel travel / spring travel)

   Their published bands are on THIS definition:
        0.5 - 1.5 Hz   passenger cars
        1.5 - 2.0 Hz   sedan racecars, moderate-downforce formula cars
        3.0 - 5.0+ Hz  high-downforce racecars

2. RIDE-RATE frequency (Gillespie, "Fundamentals of Vehicle Dynamics", Ch.5).
   The tyre is a spring in SERIES with the suspension:

        RR = K_w K_t / (K_w + K_t)          f = sqrt(RR/m) / 2 pi

Gillespie's own worked example is a road car -- tyre 1198 lb/in against a
143 lb/in suspension -- so he concludes the suspension "predominates". That
conclusion does NOT carry over to a racing car. Here the wheel rate is about
HALF the tyre rate, so the tyre costs 14-16 % of the frequency. Quoting a
tyre-inclusive number against OptimumG's band compares two different things.

PROJECT RULE: design and quote targets on definition (1) so the benchmarks are
comparable, and ALWAYS print (2) next to it so the real body-bounce frequency is
never hidden. Every function here returns both.

NOTE ON MOTION-RATIO CONVENTION. This project uses R_M = spring travel / wheel
travel, so K_w = K_s * R_M^2. OptimumG's MR is the reciprocal. Same physics,
inverted symbol -- mixing them squares the error.
"""

import numpy as np

# ---------------------------------------------------------------------------
#  the two frequencies
# ---------------------------------------------------------------------------


def ride_rate(wheel_rate, tyre_rate):
    """Suspension wheel rate in SERIES with the tyre [N/mm]  (Gillespie 5.4)."""
    kw = np.asarray(wheel_rate, float)
    kt = float(tyre_rate)
    return kw * kt / (kw + kt)


def frequency(wheel_rate, sprung_corner_kg, tyre_rate=None):
    """Undamped natural frequency [Hz] of the sprung corner.

    Without `tyre_rate` this is the WHEEL-RATE frequency (OptimumG). With it,
    the tyre goes in series and you get the RIDE-RATE frequency (Gillespie).
    """
    k = np.asarray(wheel_rate, float)
    if tyre_rate is not None:
        k = ride_rate(k, tyre_rate)
    return np.sqrt(k * 1000.0 / float(sprung_corner_kg)) / (2.0 * np.pi)


def wheel_rate_for_frequency(freq_hz, sprung_corner_kg, tyre_rate=None):
    """Wheel rate [N/mm] needed to hit a target frequency.

    With `tyre_rate` the target is the TYRE-INCLUSIVE frequency, so the wheel
    rate has to be stiffer to make up for the tyre's compliance:

        RR = m (2 pi f)^2        then      K_w = RR K_t / (K_t - RR)

    Returns inf if the target cannot be reached at any wheel rate -- once the
    required ride rate meets the tyre rate, the tyre alone is too soft and no
    spring can get you there. That limit is real and worth seeing.
    """
    m = float(sprung_corner_kg)
    rr = m * (2.0 * np.pi * np.asarray(freq_hz, float)) ** 2 / 1000.0
    if tyre_rate is None:
        return rr
    kt = float(tyre_rate)
    return np.where(rr >= kt, np.inf, rr * kt / np.maximum(kt - rr, 1e-12))


# ---------------------------------------------------------------------------
#  MOTION-RATIO CONVENTION  --  the only place the two forms are converted
# ---------------------------------------------------------------------------
# There are two motion ratios in the world and they are reciprocals:
#
#   INTERNAL  R_M = spring travel / wheel travel   K_w = K_s R_M^2    ours
#   TEXTBOOK  MR  = wheel travel / spring travel   K_s = K_w MR^2     OptimumG,
#                                                                    Milliken,
#                                                                    most courses
#
# Everything that COMPUTES in this project uses R_M, because that is the form the
# solver produces (-d(damper)/d(bump)) and the form in which the wheel rate falls
# out directly. Everything that is SHOWN to a human uses whichever convention
# CONVENTION names, defaulting to the textbook one so a reader who has never seen
# this project reads a number in the units they already know.
#
# Mixing the two is an R_M^4 error -- a factor of 10 at our values -- so no other
# module is allowed to write "1/mr" by hand. Call these.

CONVENTION = "textbook"          # "textbook" (wheel/spring) or "internal"

SYMBOL = {"textbook": "MR", "internal": "R_M"}
RATIO_TEXT = {"textbook": "wheel travel ÷ spring travel",
              "internal": "spring travel ÷ wheel travel"}
FORMULA = {"textbook": "MR = x_w / x_s   ·   spring rate  K_s = K_w · MR²",
           "internal": "R_M = x_s / x_w   ·   wheel rate  K_w = K_s · R_M²"}


def _conv(convention=None):
    c = CONVENTION if convention is None else convention
    if c not in ("textbook", "internal"):
        raise ValueError("convention must be 'textbook' or 'internal'")
    return c


def to_display(r_m, convention=None):
    """Internal R_M -> the number a human should be shown.

    Reciprocal for the textbook convention, identity for the internal one.
    Works on scalars and arrays; a zero R_M gives inf, which is the honest
    answer (a spring that never moves has an infinite textbook MR).
    """
    r = np.asarray(r_m, float)
    if _conv(convention) == "internal":
        return r if r.ndim else float(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(r != 0.0, 1.0 / np.where(r == 0.0, 1.0, r), np.inf)
    return out if out.ndim else float(out)


def from_display(mr, convention=None):
    """The inverse of `to_display` -- a shown number back to internal R_M."""
    return to_display(mr, convention)          # both maps are involutions


def symbol(convention=None):
    """'MR' or 'R_M', for axis labels and readout rows."""
    return SYMBOL[_conv(convention)]


def ratio_text(convention=None):
    """'wheel travel / spring travel' or its reverse, spelled out."""
    return RATIO_TEXT[_conv(convention)]


def formula(convention=None):
    """The one-line rate formula in the displayed convention, for plot titles."""
    return FORMULA[_conv(convention)]


def label(convention=None):
    """A full axis label: 'motion ratio  MR  [-]   (wheel travel / spring travel)'."""
    c = _conv(convention)
    return f"motion ratio  {SYMBOL[c]}  [-]   ({RATIO_TEXT[c]})"


def both_text(r_m):
    """'MR 1.772 (wheel/spring)  =  R_M 0.564 (spring/wheel)' -- for the readout,
    where showing BOTH removes the ambiguity rather than hiding half of it."""
    r = float(r_m)
    d = float(to_display(r, "textbook"))
    return f"MR {d:.4f} (wheel/spring) = R_M {r:.4f} (spring/wheel)"


def spring_rate_textbook(wheel_rate, mr):
    """Coil rate from the TEXTBOOK motion ratio:  K_s = K_w * MR^2.

    Provided so code that has a textbook MR in hand never has to invert it by
    hand. Identical result to `spring_rate(K_w, 1/MR)`, asserted in the suite.
    """
    return np.asarray(wheel_rate, float) * float(mr) ** 2


def spring_rate(wheel_rate, motion_ratio):
    """Coil rate [N/mm] from wheel rate:  K_s = K_w / R_M^2  (R_M = spring/wheel)."""
    return np.asarray(wheel_rate, float) / float(motion_ratio) ** 2


def wheel_rate_from_spring(spring_rate_N_mm, motion_ratio):
    """K_w = K_s * R_M^2 -- the inverse of `spring_rate`."""
    return np.asarray(spring_rate_N_mm, float) * float(motion_ratio) ** 2


# ---------------------------------------------------------------------------
#  mass bookkeeping
# ---------------------------------------------------------------------------


def corner_masses(total_kg, rear_frac, unsprung_corner_kg):
    """Split the car into per-corner sprung and unsprung mass.

    The weight distribution applies to the WHOLE car, so the unsprung mass has
    to come off each axle before the sprung mass per corner is known -- that is
    what sets the front/rear spring split, because a heavier corner needs a
    stiffer spring for the same frequency.
    """
    m, rf, u = float(total_kg), float(rear_frac), float(unsprung_corner_kg)
    axle_f, axle_r = m * (1.0 - rf), m * rf
    sf = (axle_f - 2.0 * u) / 2.0
    sr = (axle_r - 2.0 * u) / 2.0
    if sf <= 0 or sr <= 0:
        raise ValueError("unsprung mass exceeds the axle load -- check the inputs")
    return dict(axle_f_kg=axle_f, axle_r_kg=axle_r,
                sprung_f_kg=sf, sprung_r_kg=sr,
                unsprung_corner_kg=u,
                sprung_total_kg=2.0 * (sf + sr),
                sprung_rear_frac=2.0 * sr / (2.0 * (sf + sr)))


# ---------------------------------------------------------------------------
#  the design
# ---------------------------------------------------------------------------


def design(total_kg, rear_frac, unsprung_corner_kg, mr_f, mr_r,
           freq_f, freq_r, tyre_rate=None):
    """Full spring design from target WHEEL-RATE frequencies.

    Targets are on definition (1); the tyre-inclusive frequency of the result is
    reported alongside so both numbers are always on the table together.
    """
    m = corner_masses(total_kg, rear_frac, unsprung_corner_kg)
    out = dict(m)
    out.update(freq_f=float(freq_f), freq_r=float(freq_r),
               mr_f=float(mr_f), mr_r=float(mr_r), tyre_rate=tyre_rate)
    for tag, f, ms, mr in (("f", freq_f, m["sprung_f_kg"], mr_f),
                           ("r", freq_r, m["sprung_r_kg"], mr_r)):
        kw = float(wheel_rate_for_frequency(f, ms))       # target = definition (1)
        out[f"kw_{tag}"] = kw
        out[f"ks_{tag}"] = float(spring_rate(kw, mr))
        out[f"f_wheel_{tag}"] = float(frequency(kw, ms))
        if tyre_rate is not None:
            out[f"rr_{tag}"] = float(ride_rate(kw, tyre_rate))
            out[f"f_ride_{tag}"] = float(frequency(kw, ms, tyre_rate))
    out["freq_split_pct"] = (float(freq_f) / float(freq_r) - 1.0) * 100.0
    out["kw_split_pct"] = (out["kw_f"] / out["kw_r"] - 1.0) * 100.0
    return out


# OptimumG's published bands, for classifying a result (Tech Tip 1)
BANDS = [("passenger car", 0.5, 1.5),
         ("sedan racecar / moderate downforce", 1.5, 2.0),
         ("high-downforce racecar", 3.0, 5.0)]


def classify(freq_hz):
    """Which of OptimumG's bands a wheel-rate frequency falls in."""
    f = float(freq_hz)
    for name, lo, hi in BANDS:
        if lo <= f <= hi:
            return name
    if f < 0.5:
        return "below any published band"
    if 2.0 < f < 3.0:
        return "between the sedan and high-downforce bands"
    return "above the published high-downforce band"


if __name__ == "__main__":
    import config
    import rear_geometry as rg
    import tyre
    from wishbone_solver import corner_geometry, solve_sweep

    def mr_at_ride(pts):
        g = corner_geometry(pts, -2.7)
        z = np.linspace(-6, 6, 13)
        sw = solve_sweep(g, z, theta_span_deg=28.0, n_theta=1121)
        i0 = int(np.argmin(np.abs(np.asarray(sw["bump"]))))
        return float(sw["MR"][i0]), g["shock_on"]

    mrf, onf = mr_at_ride(config._load_front_points())
    mrr, onr = mr_at_ride(rg.solver_points(rg.load_rear()))
    kt = tyre.radial_stiffness_N_per_mm()

    d = design(1300.0, 0.56, 45.0, mrf, mrr, 3.86, 2.89, tyre_rate=kt)
    print("SPRING DESIGN  -  Temerario GT3   (targets: benchmark-led)")
    print(f"  motion ratio   F {mrf:.4f} (damper on the {onf} arm)"
          f"   R {mrr:.4f} (on the {onr} arm)")
    print(f"  tyre rate      {kt:.0f} N/mm")
    print(f"  sprung/corner  F {d['sprung_f_kg']:.1f} kg   R {d['sprung_r_kg']:.1f} kg")
    print()
    print(f"  {'':6s}{'target f':>10}{'K wheel':>10}{'K spring':>10}"
          f"{'ride rate':>11}{'f w/ tyre':>11}")
    for tag, nm in (("f", "FRONT"), ("r", "REAR")):
        print(f"  {nm:6s}{d['freq_'+tag]:9.2f} {d['kw_'+tag]:9.1f} {d['ks_'+tag]:9.1f}"
              f" {d['rr_'+tag]:10.1f} {d['f_ride_'+tag]:10.2f}")
    print()
    print(f"  frequency split F/R  {d['freq_split_pct']:+.1f} %")
    print(f"  wheel-rate split F/R {d['kw_split_pct']:+.1f} %")
    print(f"  front classifies as: {classify(d['freq_f'])}")
