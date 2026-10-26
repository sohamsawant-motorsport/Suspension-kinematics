# -*- coding: utf-8 -*-
"""
LATERAL LOAD TRANSFER  -  geometric vs elastic, for the whole car.
==================================================================

The standard axle-by-axle model (Seward's load-transfer notes; Milliken;
ENGR7008 wk4). It exists to answer one question: **as the rear roll centre moves,
how does the load transfer split change, and what does that do to the balance?**

THE THREE ROUTES A CORNERING FORCE TAKES INTO THE TYRES
-------------------------------------------------------
  1  GEOMETRIC  - straight up the suspension LINKS, through that axle's roll
                  centre. Instant: it does not wait for the body to roll.
                      dW_geo = W_sprung_axle * a * h_RC / t
  2  ELASTIC    - the body rolls about the ROLL AXIS, and the springs/ARBs
                  resist. Delayed by however long the body takes to lean.
                      M_roll = W_s * a * (h_CG - h_RA)
                      dW_el  = M_roll * (this axle's share of roll stiffness) / t
  3  UNSPRUNG   - the wheels/uprights themselves, which no suspension carries.
                      dW_u = W_unsprung * a * h_unsprung / t

THE ROLL AXIS is the line joining the front and rear roll centres. Its height
BELOW THE CG is the lever the roll moment acts on:

    h_RA(at the CG) = h_RCf + (h_RCr - h_RCf) * l_f / L

so **raising the rear roll centre lifts the roll axis, shortens that lever, and
reduces the roll moment for the WHOLE car** -- which is why rear RC height changes
the elastic transfer at the FRONT as well. That coupling is the whole point; a
rear-axle-in-isolation model cannot see it.

BALANCE. What actually decides understeer/oversteer is the FRACTION of the total
transfer taken by the front axle (the lateral load-transfer distribution, LLTD).
Because tyres are load-sensitive, the axle taking MORE transfer loses relatively
more grip:
        front LLTD up   -> more understeer
        front LLTD down -> more oversteer

SIGN CONVENTION (matches axle.py, asserted by validate_conventions.py):
    + lateral g      = a RIGHT-HAND turn; the LEFT wheels are the loaded pair
    + longitudinal g = accelerating (load to the rear); - = braking
    + body roll      = the body leans onto its LEFT (the same right-hand turn)

UNITS: forces in N, lengths in mm, `a` in g, roll stiffness in N.mm/rad.
"""
import numpy as np

G = 9.81


def roll_stiffness(wheel_rate_N_per_mm, arb_rate_N_per_mm, track_mm):
    """Roll stiffness [N.mm/rad] of one axle from its wheel rate and ARB rate.

    Roll by phi lifts one wheel and drops the other by (t/2)*phi, so each spring
    makes a force K*(t/2)*phi acting at (t/2) from the centreline:
        M = 2 * K*(t/2)*phi * (t/2) = K * t^2 * phi / 2
    The ARB adds in parallel (it only reacts to the ANTI-symmetric part, which is
    exactly what roll is).
    """
    return (float(wheel_rate_N_per_mm) + float(arb_rate_N_per_mm)) * track_mm ** 2 / 2.0


def roll_axis_height(h_rc_f, h_rc_r, rear_weight_frac):
    """Roll-axis height [mm] at the CG's longitudinal station.

    The CG sits `l_f` behind the front axle, and l_f/L == the REAR weight
    fraction (mass close to the rear axle means l_f is large).
    """
    return h_rc_f + (np.asarray(h_rc_r, float) - h_rc_f) \
        * np.asarray(rear_weight_frac, float)


def transfer(a_g, h_rc_f, h_rc_r, car):
    """Lateral load transfer for both axles at `a_g` lateral g.

    `car` is a dict of: mass_total_kg, rear_frac, unsprung_f_kg, unsprung_r_kg,
    h_cg_mm, h_unsprung_mm, track_f_mm, track_r_mm, k_roll_f, k_roll_r.

    Returns a dict with the geometric / elastic / unsprung split per axle, the
    totals, the roll angle, and the front LLTD.
    """
    m = car["mass_total_kg"]
    rf = car["rear_frac"]
    W_r_total = m * rf * G                      # rear axle weight, N
    W_f_total = m * (1.0 - rf) * G
    W_u_f, W_u_r = car["unsprung_f_kg"] * G, car["unsprung_r_kg"] * G
    W_s_f, W_s_r = W_f_total - W_u_f, W_r_total - W_u_r      # sprung, per axle
    W_s = W_s_f + W_s_r
    t_f, t_r = car["track_f_mm"], car["track_r_mm"]
    k_f, k_r = car["k_roll_f"], car["k_roll_r"]

    # The roll axis must be sampled at the SPRUNG mass's longitudinal station,
    # not the total mass's. They differ because the unsprung mass sits AT the
    # axles, so it shifts the total CG without shifting the sprung CG. Using the
    # total fraction here breaks the whole-car moment identity by ~0.4 % -- the
    # validation suite caught exactly that.
    sprung_rear_frac = W_s_r / W_s
    h_ra = roll_axis_height(h_rc_f, h_rc_r, sprung_rear_frac)
    lever = car["h_cg_mm"] - h_ra               # roll moment arm
    M_roll = W_s * a_g * lever                  # N.mm
    phi = M_roll / (k_f + k_r)                  # rad
    share_f = k_f / (k_f + k_r)

    dW_geo_f = W_s_f * a_g * h_rc_f / t_f
    dW_geo_r = W_s_r * a_g * np.asarray(h_rc_r, float) / t_r
    dW_el_f = M_roll * share_f / t_f
    dW_el_r = M_roll * (1.0 - share_f) / t_r
    dW_u_f = W_u_f * a_g * car["h_unsprung_mm"] / t_f
    dW_u_r = W_u_r * a_g * car["h_unsprung_mm"] / t_r

    tot_f = dW_geo_f + dW_el_f + dW_u_f
    tot_r = dW_geo_r + dW_el_r + dW_u_r
    return dict(
        geo_f=dW_geo_f, geo_r=dW_geo_r, el_f=dW_el_f, el_r=dW_el_r,
        uns_f=dW_u_f, uns_r=dW_u_r, total_f=tot_f, total_r=tot_r,
        roll_deg=np.degrees(phi), roll_axis=h_ra, lever=lever,
        lltd_f=tot_f / (tot_f + tot_r) * 100.0,
        geo_frac_r=dW_geo_r / tot_r * 100.0,
        W_s_r=W_s_r, W_s_f=W_s_f)


def wheel_rate_from_frequency(sprung_corner_kg, ride_freq_hz):
    """Wheel rate [N/mm] from a target ride frequency: f = (1/2pi) sqrt(K/m)."""
    return sprung_corner_kg * (2.0 * np.pi * float(ride_freq_hz)) ** 2 / 1000.0


# =====================================================================
#  LONGITUDINAL transfer, and the COMBINED (g-g) four-wheel load state
# =====================================================================
#
# LONGITUDINAL  [ENGR7008 wk4 pp.29-32, using their symbols: E = wheelbase,
# L1/L2 = CG to front/rear axle, h = CG height]
#
#       N_front = m g L2 / E  +  m a h / E        (braking loads the front)
#       N_rear  = m g L1 / E  -  m a h / E
#
# so the transfer term is simply
#
#       dW_long = m * a_x * h_CG / wheelbase
#
# with the sign convention used here: a_x > 0 = ACCELERATING, which moves load
# to the REAR; a_x < 0 = braking, which moves it to the FRONT.
#
# COMBINING the two is not just adding them. Longitudinal transfer changes how
# much weight each AXLE carries, and the lateral transfer at that axle then works
# on the NEW axle weight -- in particular the geometric term W_s_axle*a_y*h_RC/t
# scales with the axle's sprung weight. That coupling is why a car behaves
# differently trail-braking into a corner than it does at steady state.
#
# The one identity that must always hold, whatever the accelerations:
#       FL + FR + RL + RR = total weight
# (the road can only push up by as much as gravity pulls down). validate_load_
# transfer.py asserts exactly that across the whole g-g envelope.
#
def longitudinal_transfer(a_x_g, car):
    """Load moved from front to rear axle [N]. + = rearward (accelerating)."""
    return (car["mass_total_kg"] * G * np.asarray(a_x_g, float) * car["h_cg_mm"]
            / car["wheelbase_mm"])


def wheel_loads(a_lat_g, a_long_g, h_rc_f, h_rc_r, car):
    """The four tyre vertical loads [N] at a point on the g-g diagram.

    a_lat_g  > 0 : a RIGHT-HAND turn -> load moves to the LEFT-hand wheels
    a_long_g > 0 : accelerating (load to the rear);  < 0 : braking (to the front)

    Returns per-wheel loads, the axle sub-totals, the lateral split at each axle,
    and flags any wheel that has lifted.
    """
    m, rf = car["mass_total_kg"], car["rear_frac"]
    W_tot = m * G
    dW_x = longitudinal_transfer(a_long_g, car)

    # dynamic axle weights after longitudinal transfer
    W_f = W_tot * (1.0 - rf) - dW_x
    W_r = W_tot * rf + dW_x

    # lateral transfer, evaluated on the DYNAMIC axle weights
    W_u_f, W_u_r = car["unsprung_f_kg"] * G, car["unsprung_r_kg"] * G
    W_s_f, W_s_r = W_f - W_u_f, W_r - W_u_r
    W_s = W_s_f + W_s_r
    t_f, t_r = car["track_f_mm"], car["track_r_mm"]
    k_f, k_r = car["k_roll_f"], car["k_roll_r"]

    sprung_rear_frac = np.where(W_s != 0.0, W_s_r / np.where(W_s == 0.0, 1.0, W_s), rf)
    h_ra = roll_axis_height(h_rc_f, h_rc_r, sprung_rear_frac)
    M_roll = W_s * a_lat_g * (car["h_cg_mm"] - h_ra)
    share_f = k_f / (k_f + k_r)

    dW_f = (W_s_f * a_lat_g * h_rc_f / t_f            # geometric
            + M_roll * share_f / t_f                  # elastic
            + W_u_f * a_lat_g * car["h_unsprung_mm"] / t_f)   # unsprung
    dW_r = (W_s_r * a_lat_g * np.asarray(h_rc_r, float) / t_r
            + M_roll * (1.0 - share_f) / t_r
            + W_u_r * a_lat_g * car["h_unsprung_mm"] / t_r)

    # SIGN: + lateral g = a RIGHT-HAND TURN. The inertial force then acts to the
    # LEFT, so the LEFT-hand wheels are the loaded (outside) pair. This matches
    # axle.axle_state, where + body_roll leans the car onto its LEFT and
    # compresses the left suspension -- the two modules used to disagree about
    # which way a positive number turned the car, which validate_conventions.py
    # now asserts against.
    FL, FR = W_f / 2.0 + dW_f, W_f / 2.0 - dW_f
    RL, RR = W_r / 2.0 + dW_r, W_r / 2.0 - dW_r
    return dict(FL=FL, FR=FR, RL=RL, RR=RR,
                W_f=W_f, W_r=W_r, dW_lat_f=dW_f, dW_lat_r=dW_r, dW_long=dW_x,
                total=FL + FR + RL + RR, roll_axis=h_ra,
                lifted=[n for n, v in (("FL", FL), ("FR", FR),
                                       ("RL", RL), ("RR", RR))
                        if np.any(np.asarray(v) <= 0.0)])


def gg_envelope(a_lat_max=1.2, a_accel_max=1.0, a_brake_max=1.2, n=361):
    """The g-g (friction) envelope as (a_lat, a_long) arrays.

    The real shape is a FRICTION CIRCLE of radius a_lat_max, CLIPPED FLAT at
    +a_accel_max because forward acceleration is limited by power/traction rather
    than by grip. That clip is what gives the reference plot its flat top, which
    runs out to  lat = sqrt(a_lat_max^2 - a_accel_max^2)  on each side
    (0.66 g for a 1.2 g circle clipped at 1.0 g). Braking is grip-limited, so the
    bottom stays round at -a_brake_max.

    An earlier version drew a quarter-ellipse per quadrant, which rounded the top
    off instead of clipping it -- it did not match the reference diagram.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n)
    lat = a_lat_max * np.cos(th)
    lon = np.where(np.sin(th) >= 0, a_lat_max, a_brake_max) * np.sin(th)
    lon = np.minimum(lon, a_accel_max)               # flat top: the traction cap
    return lat, lon


def inside_envelope(a_lat, a_long, a_lat_max=1.2, a_accel_max=1.0,
                    a_brake_max=1.2):
    """True where (a_lat, a_long) is inside the friction envelope.

    The same CLIPPED CIRCLE gg_envelope() draws: inside the grip circle AND under
    the traction cap. Tolerance is 1e-9, not 1e-12 -- points generated exactly ON
    the boundary land at 1.0 +/- a few ulp, and 1e-12 rejected half of them.
    """
    lat, lon = np.asarray(a_lat, float), np.asarray(a_long, float)
    rad = np.where(lon >= 0, a_lat_max, a_brake_max)
    return ((lat / a_lat_max) ** 2 + (lon / rad) ** 2 <= 1.0 + 1e-9) \
        & (lon <= a_accel_max + 1e-9)


# ------------------------------------------------------------------ the car
def temerario_gt3(k_arb_scale_f=2.15, k_arb_scale_r=0.674,
                  freq_f=3.4, freq_r=3.2, track_f=1662.4, track_r=1645.8):
    """Temerario GT3 ESTIMATES -- stated assumptions, not measured data.

    Mass/CG: FIA GT3 minimum class 1300 kg, 56 % rear (mid-engine), 90 kg
    unsprung per axle, CG 450 mm.
    Springs: from target ride frequencies (front 3.4 Hz, rear 3.2 Hz -- typical
    GT3, rear a little softer for traction).
    ARBs: scaled from the PUBLIC Segers GT setup sheet, which runs ARB/spring
    ratios of 60.21/28 = 2.15 front and 21.57/32 = 0.674 rear. Using the RATIO
    rather than the absolute rates keeps the benchmark's balance philosophy while
    staying consistent with our own spring derivation. It lands at ~62 % front
    roll-stiffness distribution, in line with the 57-62 % these cars run.
    """
    mass, rf, uns = 1300.0, 0.56, 90.0
    sprung_f = (mass * (1 - rf) - uns) / 2.0
    sprung_r = (mass * rf - uns) / 2.0
    kw_f = wheel_rate_from_frequency(sprung_f, freq_f)
    kw_r = wheel_rate_from_frequency(sprung_r, freq_r)
    car = dict(mass_total_kg=mass, rear_frac=rf,
               unsprung_f_kg=uns, unsprung_r_kg=uns,
               h_cg_mm=450.0, h_unsprung_mm=340.0, wheelbase_mm=2645.0,
               track_f_mm=track_f, track_r_mm=track_r,
               kw_f=kw_f, kw_r=kw_r,
               arb_f=kw_f * k_arb_scale_f, arb_r=kw_r * k_arb_scale_r)
    car["k_roll_f"] = roll_stiffness(car["kw_f"], car["arb_f"], track_f)
    car["k_roll_r"] = roll_stiffness(car["kw_r"], car["arb_r"], track_r)
    car["roll_stiff_front_pct"] = (car["k_roll_f"]
                                   / (car["k_roll_f"] + car["k_roll_r"]) * 100.0)
    return car


if __name__ == "__main__":
    c = temerario_gt3()
    print("TEMERARIO GT3 estimates")
    print(f"  wheel rate  F {c['kw_f']:6.1f}   R {c['kw_r']:6.1f}  N/mm")
    print(f"  ARB rate    F {c['arb_f']:6.1f}   R {c['arb_r']:6.1f}  N/mm")
    print(f"  roll stiffness front share {c['roll_stiff_front_pct']:.1f} %")
    for hf, tag in ((-283.6, "our broken front"), (46.2, "sorted front")):
        print(f"\n  {tag}  (h_RCf = {hf:+.0f} mm)")
        print("    rear RC   roll[deg/g]  rear geo%   front LLTD%   dW_rear[N]")
        for hr in (-52.0, 0.0, 22.0, 45.0):
            r = transfer(1.0, hf, hr, c)
            print(f"     {hr:+6.1f}     {r['roll_deg']:8.3f}    {r['geo_frac_r']:7.1f}"
                  f"      {r['lltd_f']:7.2f}    {r['total_r']:8.0f}")
