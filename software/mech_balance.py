# -*- coding: utf-8 -*-
"""
MECHANICAL BALANCE & ROLL STIFFNESS  --  the setup lookup table.
=================================================================

WHAT "MECHANICAL BALANCE" MEANS
-------------------------------
A cornering car transfers load from the inside wheels to the outside ones. How
much of that transfer lands on the FRONT axle rather than the rear is the single
number that decides whether the car understeers or oversteers, because tyres are
LOAD-SENSITIVE: a tyre pushed harder gains grip, but not proportionally. Take
1000 N off the inside front and put it on the outside front and the front axle
ends up with LESS total grip than it started with. So:

    more of the transfer at the FRONT -> the front gives up first -> UNDERSTEER
    more of the transfer at the REAR  -> the rear gives up first  -> OVERSTEER

That fraction is the lateral load-transfer distribution, LLTD. "Mechanical
balance" is LLTD -- balance produced by springs, bars and geometry, as opposed to
aero balance or the differential.

It is called MECHANICAL because you set it with hardware that does not care how
fast you are going. Aero balance moves with speed; this does not.

THE TWO NUMBERS THE BENCHMARK TABULATES
---------------------------------------
A GT3 setup handbook prints two tables per axle, both as CHANGES from a
baseline setup, over a grid of spring rate x anti-roll-bar position:

  Mechanical Balance FA [%]   the change in FRONT LLTD, in percentage points.
                              + = more transfer to the front = more understeer.
  Roll Stiffness [%]          the change in the car's TOTAL roll stiffness,
                              as a percentage. + = stiffer, so less body roll.

Those definitions are not stated anywhere -- they were RECOVERED by
reconstructing every published cell from a reference car's own spring and
anti-roll-bar rate tables. `validate_mech_balance.py` documents that
reconstruction and re-runs it as a test. Two results from it matter here:

  1. THE TYRE IS IN SERIES. Reconstructions that treat the axle as
     (spring + bar) alone miss by 2.5-2.7 points rms; putting the tyre in series
     with them drops that to 0.4. That is a factor of six, so it is not a
     modelling preference -- the benchmark's own numbers require it.

  2. BALANCE IS LLTD, NOT ROLL-STIFFNESS SHARE. Those two get used
     interchangeably and they are NOT equal: only part of the load transfer goes
     through the springs at all. The rest goes straight up the links through the
     roll centres (geometric) and through the unsprung mass, and neither of those
     cares what spring you fitted. So a 1-point move in roll-stiffness share
     moves LLTD by only k points, where

            k = (elastic transfer) / (total transfer)   ~ 0.8

     Three independent routes agree on k for the benchmark car: 0.812 from the
     published tables alone with no model at all, 0.828 from the full
     reconstruction fit, and 0.807 from THIS project's load-transfer model run at
     the benchmark's own roll-centre heights.

WHY THE TABLE EXISTS  (what a race engineer actually does with it)
------------------------------------------------------------------
You change springs for RIDE -- kerbs, platform control, aero platform, driver
feel. That change is not free: a stiffer front spring also stiffens the front in
roll, which pushes balance towards understeer. The table lets you look up the bar
move that CANCELS that balance shift, so you get the ride change you wanted and
keep the car you had. `reset_arb()` is that lookup done exactly.

WHY THE CURVES ARE NOT STRAIGHT
-------------------------------
Because of the tyre. Springs and bar act in parallel at the wheel, and that
combination sits in SERIES with the tyre's own vertical stiffness:

    K_axle = (K_spring + K_bar) * K_tyre / (K_spring + K_bar + K_tyre)

Series means the SOFTER element dominates. Once the suspension is a good fraction
of the tyre rate, adding more suspension stiffness buys less and less -- the axle
saturates towards the tyre. That is why a fixed bar step is worth less on stiff
springs than on soft ones, and it shows up in the benchmark tables as rows whose
span shrinks as the spring rate climbs.

HALF-STEP BLADE POSITIONS
-------------------------
Blade-adjustable bars carry a blade at each end, so you can set them differently
and get a half step: "3-4" means one blade on 3 and the other on 4. The two
blades and the bar all sit in SERIES along one path, which makes the pair rate
the HARMONIC mean of the two symmetric settings (algebra in `half_step_rate`).
Checked against the benchmark's own published pair rates it holds to 1-3 %, so n
blade detents give 2n-1 usable columns, not n.

UNITS: rates N/mm at the WHEEL, track mm, roll stiffness N.mm/rad, LLTD %.
"""

import numpy as np

import load_transfer as ltr

#  Roll-stiffness distribution GT cars actually run, front share [%]. From the
#  OptimumG seminar notes and the public Segers GT setup work -- not from any
#  private reference data, so this one stands on its own.
FRONT_SHARE_BAND = (57.0, 66.0)


# ---------------------------------------------------------------------------
#  axle stiffness, with the tyre where it belongs
# ---------------------------------------------------------------------------


def axle_rate(wheel_rate, arb_rate, tyre_rate=None):
    """Effective vertical rate of one corner [N/mm].

    Spring and bar act in PARALLEL at the wheel (they add). That sum is in
    SERIES with the tyre carcass, because the load path is
    ground -> tyre -> wheel -> suspension -> body, and a series chain is only as
    stiff as its softest link.

    `tyre_rate=None` leaves the tyre out, which is the convention this project's
    older modules use. Pass a rate to include it.
    """
    k = np.asarray(wheel_rate, float) + np.asarray(arb_rate, float)
    if tyre_rate is None:
        return k
    kt = float(tyre_rate)
    return k * kt / (k + kt)


def roll_stiffness(wheel_rate, arb_rate, track_mm, tyre_rate=None):
    """Axle roll stiffness [N.mm/rad], tyre optionally in series.

    Rolling by phi moves each wheel (t/2)phi; each makes K(t/2)phi acting at
    (t/2) from the centreline, so M = K t^2 phi / 2.
    """
    return axle_rate(wheel_rate, arb_rate, tyre_rate) * float(track_mm) ** 2 / 2.0


def half_step_rate(k_ii, k_jj):
    """Rate of the mixed blade pair (i,j) from the two symmetric pairs.

    One blade at each end, both in series with the bar:

        1/k(i,j) = 1/b_i + 1/K + 1/b_j
        1/k(i,i) = 2/b_i + 1/K       ->     1/b_i = (1/k(i,i) - 1/K) / 2

    Substituting, the 1/K terms cancel EXACTLY and

        1/k(i,j) = ( 1/k(i,i) + 1/k(j,j) ) / 2

    -- the harmonic mean, independent of the bar stiffness. That independence is
    what makes it usable: you never need the blade/bar split, only the symmetric
    rates you can measure.
    """
    a, b = np.asarray(k_ii, float), np.asarray(k_jj, float)
    return 2.0 / (1.0 / a + 1.0 / b)


def pair_labels(n_positions):
    """The benchmark's column headings: 1-1, 1-2, 2-2, ... n-n  (2n-1 of them)."""
    out = []
    n = int(n_positions)
    for i in range(1, n + 1):
        out.append(f"{i}-{i}")
        if i < n:
            out.append(f"{i}-{i + 1}")
    return out


def pair_rates(full_rates):
    """Expand n symmetric blade rates into the 2n-1 half-step column rates."""
    k = np.asarray(full_rates, float)
    out = []
    for i in range(len(k)):
        out.append(float(k[i]))
        if i < len(k) - 1:
            out.append(float(half_step_rate(k[i], k[i + 1])))
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
#  one setup
# ---------------------------------------------------------------------------


def state(car, rc_f, rc_r, kw_f=None, arb_f=None, kw_r=None, arb_r=None,
          tyre_rate=None, lat_g=1.0):
    """Everything about one setup: roll stiffness, its split, and the balance.

    Any of the four rates left as None keeps whatever is already in `car`.
    Returns the two tabulated quantities as ABSOLUTE values -- the tables
    themselves are differences of these.
    """
    c = dict(car)
    for key, val in (("kw_f", kw_f), ("arb_f", arb_f),
                     ("kw_r", kw_r), ("arb_r", arb_r)):
        if val is not None:
            c[key] = float(val)
    c["k_roll_f"] = roll_stiffness(c["kw_f"], c["arb_f"], c["track_f_mm"], tyre_rate)
    c["k_roll_r"] = roll_stiffness(c["kw_r"], c["arb_r"], c["track_r_mm"], tyre_rate)
    t = ltr.transfer(lat_g, rc_f, rc_r, c)
    k_tot = c["k_roll_f"] + c["k_roll_r"]
    return dict(
        k_roll_f=c["k_roll_f"], k_roll_r=c["k_roll_r"], k_roll_total=k_tot,
        front_share_pct=100.0 * c["k_roll_f"] / k_tot,
        lltd_f=t["lltd_f"], roll_grad=t["roll_deg"] / lat_g,
        elastic_frac=(t["el_f"] + t["el_r"]) / (t["total_f"] + t["total_r"]),
        roll_axis=t["roll_axis"], lever=t["lever"], car=c)


def elastic_fraction(car, rc_f, rc_r, tyre_rate=None, lat_g=1.0):
    """k -- the share of total lateral load transfer that goes through the springs.

    This is the conversion between the two things people both call "balance":

        d(front LLTD)  =  k * d(front roll-stiffness share)

    k < 1 always, because the geometric (roll-centre) and unsprung routes carry
    transfer that no spring or bar can touch. k -> 1 only if both roll centres sit
    on the ground AND the unsprung mass is zero.

    A roll centre BELOW ground makes its geometric term negative, which can push
    k above 1. That is not a bug -- it is the underground-roll-centre pathology
    showing up as an over-sensitive balance, and it is worth being able to see.
    """
    return state(car, rc_f, rc_r, tyre_rate=tyre_rate, lat_g=lat_g)["elastic_frac"]


# ---------------------------------------------------------------------------
#  the tables
# ---------------------------------------------------------------------------


def tables(car, rc_f, rc_r, axle, spring_grid, arb_grid, motion_ratio,
           baseline_spring, baseline_arb, tyre_rate=None, lat_g=1.0):
    """The benchmark's two tables for one axle.

    spring_grid    SPRING rates [N/mm] -- the rows, as they come off a spring
                   rack, not wheel rates.
    arb_grid       ARB WHEEL rates [N/mm], one per column, already expanded to
                   half steps by `pair_rates`.
    motion_ratio   spring travel / wheel travel, so wheel rate = k * MR^2.
    baseline_*     the setup the tables print 0.0 against. Must be ON the grid,
                   or the zero cell will not come out exactly zero.

    Returns (balance, roll_stiff, baseline_state); the two arrays are
    [len(spring_grid) x len(arb_grid)], in points and percent relative to the
    baseline.
    """
    mr2 = float(motion_ratio) ** 2
    kw_key, arb_key = f"kw_{axle}", f"arb_{axle}"

    base = dict(car)
    base[kw_key] = float(baseline_spring) * mr2
    base[arb_key] = float(baseline_arb)
    s0 = state(base, rc_f, rc_r, tyre_rate=tyre_rate, lat_g=lat_g)

    bal = np.zeros((len(spring_grid), len(arb_grid)))
    rs = np.zeros_like(bal)
    for i, spr in enumerate(spring_grid):
        for j, kb in enumerate(arb_grid):
            c = dict(base)
            c[kw_key] = float(spr) * mr2
            c[arb_key] = float(kb)
            s = state(c, rc_f, rc_r, tyre_rate=tyre_rate, lat_g=lat_g)
            bal[i, j] = s["lltd_f"] - s0["lltd_f"]
            rs[i, j] = 100.0 * (s["k_roll_total"] / s0["k_roll_total"] - 1.0)
    return bal, rs, s0


def reset_arb(car, rc_f, rc_r, axle, new_spring, motion_ratio,
              baseline_spring, baseline_arb, tyre_rate=None, lat_g=1.0,
              lo=0.0, hi=None):
    """THE POINT OF THE WHOLE TABLE.

    You changed one axle's spring for ride reasons. What ARB wheel rate on that
    same axle puts the mechanical balance back exactly where it was?

    Solved by bisection on the balance error rather than in closed form, so the
    geometric and unsprung terms stay exact. Returns (arb_rate, info); if the
    balance cannot be restored within the bar's reach, `info["reachable"]` is
    False and the rate returned is the nearest end stop -- an honest "a bar
    cannot get you there, change the OTHER axle" rather than a silent clip.
    """
    mr2 = float(motion_ratio) ** 2
    kw_key, arb_key = f"kw_{axle}", f"arb_{axle}"
    base = dict(car)
    base[kw_key] = float(baseline_spring) * mr2
    base[arb_key] = float(baseline_arb)
    target = state(base, rc_f, rc_r, tyre_rate=tyre_rate, lat_g=lat_g)["lltd_f"]

    if hi is None:
        hi = max(10.0 * float(baseline_arb), 50.0)

    def err(kb):
        c = dict(base)
        c[kw_key] = float(new_spring) * mr2
        c[arb_key] = float(kb)
        return state(c, rc_f, rc_r, tyre_rate=tyre_rate,
                     lat_g=lat_g)["lltd_f"] - target

    e_lo, e_hi = err(lo), err(hi)
    if e_lo * e_hi > 0.0:
        end = lo if abs(e_lo) < abs(e_hi) else hi
        return float(end), dict(reachable=False, residual=float(err(end)),
                                target_lltd=float(target), delta_pct=float("nan"))
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if err(a) * err(m) <= 0.0:
            b = m
        else:
            a = m
    kb = 0.5 * (a + b)
    return float(kb), dict(reachable=True, residual=float(err(kb)),
                           target_lltd=float(target),
                           delta_pct=100.0 * (kb / float(baseline_arb) - 1.0))


def classify_front_share(pct, band=FRONT_SHARE_BAND):
    """Where a front roll-stiffness share sits against the public GT range."""
    lo, hi = band
    if pct < lo:
        return f"{pct:.1f} % -- below the {lo:.0f}-{hi:.0f} % GT range (rear-stiff)"
    if pct > hi:
        return f"{pct:.1f} % -- above the {lo:.0f}-{hi:.0f} % GT range (front-stiff)"
    return f"{pct:.1f} % -- inside the {lo:.0f}-{hi:.0f} % GT range"


# ---------------------------------------------------------------------------
#  sensitivity -- what each input does to the balance
# ---------------------------------------------------------------------------

#  (key, label, group, step, unit). Steps are sized to be a realistic single
#  change on a GT3, so the numbers can be compared with each other directly.
SENSITIVITY_STEPS = (
    ("ks_f", "front spring", "front", 10.0, "+10 N/mm"),
    ("arb_f", "front bar", "front", 10.0, "+10 N/mm"),
    ("mr_f", "front motion ratio", "front", 0.05, "+0.05"),
    ("track_f_mm", "front track", "front", 10.0, "+10 mm"),
    ("unsprung_f_kg", "front unsprung", "front", 5.0, "+5 kg"),
    ("rc_f", "front roll centre", "front", 10.0, "+10 mm"),
    ("ks_r", "rear spring", "rear", 10.0, "+10 N/mm"),
    ("arb_r", "rear bar", "rear", 10.0, "+10 N/mm"),
    ("mr_r", "rear motion ratio", "rear", 0.05, "+0.05"),
    ("track_r_mm", "rear track", "rear", 10.0, "+10 mm"),
    ("unsprung_r_kg", "rear unsprung", "rear", 5.0, "+5 kg"),
    ("rc_r", "rear roll centre", "rear", 10.0, "+10 mm"),
    ("h_cg_mm", "CG height", "car", 10.0, "+10 mm"),
    ("rear_frac", "rear weight", "car", 0.01, "+1 pt"),
    ("tyre", "tyre rate", "car", 10.0, "+10 N/mm"),
)


def lltd_from_rates(car, rc_f, rc_r, ks_f, mr_f, ks_r, mr_r, tyre_rate=None):
    """Front LLTD [%] from SPRING rates and motion ratios rather than wheel rates.

    The tables and the explorer think in spring rates (what you fit) while the
    load-transfer model thinks in wheel rates; this is the one place the two are
    joined, so the motion ratio can be perturbed on its own.
    """
    c = dict(car)
    c["kw_f"] = float(ks_f) * float(mr_f) ** 2
    c["kw_r"] = float(ks_r) * float(mr_r) ** 2
    return state(c, rc_f, rc_r, tyre_rate=tyre_rate)["lltd_f"]


def sensitivity(car, rc_f, rc_r, ks_f, mr_f, ks_r, mr_r, tyre_rate=None,
                steps=SENSITIVITY_STEPS):
    """Change in FRONT LLTD [points] for one step of each input, all else held.

    A finite step, not a derivative, on purpose: these are the numbers you would
    actually see on the car after making that change. `car` must carry arb_f and
    arb_r as WHEEL rates. The tyre row is skipped when tyre_rate is None.

    Returns a list of dicts: key, label, group, step, unit, base, delta.
    """
    base_in = dict(ks_f=float(ks_f), mr_f=float(mr_f), ks_r=float(ks_r),
                   mr_r=float(mr_r), rc_f=float(rc_f), rc_r=float(rc_r),
                   tyre=tyre_rate)

    def run(key=None, step=0.0):
        c = dict(car)
        a = dict(base_in)
        if key is not None:
            if key in a:
                a[key] = a[key] + step
            else:
                c[key] = float(c[key]) + step
        return lltd_from_rates(c, a["rc_f"], a["rc_r"], a["ks_f"], a["mr_f"],
                               a["ks_r"], a["mr_r"], a["tyre"])

    base = run()
    out = []
    for key, label, group, step, unit in steps:
        if key == "tyre" and tyre_rate is None:
            continue
        out.append(dict(key=key, label=label, group=group, step=step, unit=unit,
                        base=base, delta=run(key, step) - base))
    return out
