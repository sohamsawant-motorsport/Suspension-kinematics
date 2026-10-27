# References

What the maths in this repository is built on. Everything listed here is a
published book, paper or technical note. Where a specific equation or figure is
used, the file that uses it names it in a comment.

---

## Primary — equations used directly

**Gillespie, T. D. — *Fundamentals of Vehicle Dynamics*** (SAE International)

The main source. Used throughout:

| where | what |
|---|---|
| Ch. 5, Eq. 5.4 | ride rate with the tyre in series, `RR = K_w·K_t / (K_w + K_t)` — `software/springs.py` |
| Ch. 6, Eq. 6.58–6.60 | lateral-force compliance steer and the understeer gradient it produces |
| Ch. 6, Eq. 6.61 | aligning-torque understeer, and why it needs cornering stiffness |
| Ch. 6, Fig. 6.18 | the yaw-centre sign rule, and his own caveat that independent suspensions need the full linkage analysis |
| Ch. 7, Eq. 7.19 | anti-squat, wheel-centre reference, for a chassis-mounted differential |
| Ch. 7, Eq. 7.28 | anti-dive, contact-patch reference, outboard brakes |
| Ch. 7, Eq. 7.29 | anti-lift, contact-patch reference |
| Ch. 7, p. 205 | "In practice, 100 % anti-dive is rarely used. The maximum anti-dive seldom exceeds 50 %", plus the eight reasons why — including caster change under braking and brake hop on a short swing arm |

`software/anti_geometry.py` reproduces 7.19, 7.28 and 7.29 from first principles
and asserts agreement with the published forms, rather than coding them up and
trusting them.

**Adams, H. — *Chassis Engineering*** (HPBooks)

| where | what |
|---|---|
| p. 59–60 | anti-dive in practice: 5–7 % on a race suspension, up to 30 % on production cars, many race cars at zero because the CG is so low |
| Fig. 10-2, p. 82 | **independent rear suspension has a practical anti-squat limit around 25 %** — an inboard differential removes the torque-reaction path, so only the control-arm angles contribute. This is the limit that applies to a GT3, not Gillespie's 50 % |

**OptimumG — *Springs & Dampers* technical tip series** (Giaraffa)

Tech Tip 1 for the wheel-rate definition of ride frequency,
`K_s = 4π²f²·m·MR²`, and its published bands: 0.5–1.5 Hz passenger cars,
1.5–2.0 Hz sedan racers and moderate-downforce formula cars, 3.0–5.0+ Hz
high-downforce racers. Those bands belong to *this* definition, with the tyre
left out — which is why `software/springs.py` returns both frequencies and
labels which is which.

The anti-roll bar tips are used as an independent cross-check on the bar rate
derivation in `software/arb.py`.

---

## Secondary — background, conventions and cross-checks

**Milliken, W. F. & Milliken, D. L. — *Race Car Vehicle Dynamics*** (SAE)
The standard axle-by-axle load-transfer model in `software/load_transfer.py`
follows the usual treatment given here, and the textbook motion-ratio
convention the tool displays (`MR` = wheel travel / spring travel) is theirs.

**Blundell, M. & Harty, D. — *The Multibody Systems Approach to Vehicle
Dynamics*** (Elsevier)
Compliance steer done properly for an independent suspension, which is what
Gillespie explicitly says his solid-axle sign rule does not cover.

**Guiggiani, M. — *The Science of Vehicle Dynamics: Handling, Braking and Ride
of Road and Race Cars*** (Springer)
A rigorous treatment of handling and the understeer gradient.

**Pacejka, H. B. — *Tyre and Vehicle Dynamics*** (Butterworth-Heinemann)
Pneumatic trail and aligning torque, which is what Gillespie's Eq. 6.61 needs.

**Genta, G. — *Motor Vehicle Dynamics: Modeling and Simulation*** (World
Scientific, Series on Advances in Mathematics for Applied Sciences vol. 43)
An alternative derivation of the handling equations.

**Seward, D. — *Race Car Design*** (Red Globe Press)
Practical race-car suspension layout.

**Puhn, F. — *How to Make Your Car Handle*** (HPBooks)
Older but direct on scrub, bump steer and the measurement methods behind them.

**Segers, J. — *Analysis Techniques for Racecar Data Acquisition*** (SAE)
Used for setup-sheet conventions and the published GT roll-stiffness
distribution range.

---

## Course material

**ENGR7008 Vehicle Dynamics**, weeks 2, 4, 5 and 6.
Week 4 is the source of the motion-ratio notation used internally
(`R_M = x_s / x_w`) and of the roll-centre-in-roll construction the
`RC in roll` tab is built on.

---

## Regulations

**IMSA GTD technical regulations** (2025 technical regulations, blackline
revision) and the **FIA GT3 regulations**, for the class constraints the
geometry has to live inside — minimum mass, wheel and tyre dimensions, and what
is free versus homologated.

---

## Not used as a benchmark

This project was developed alongside a manufacturer's performance handbook and a
tyre supplier's data booklet. Both are confidential documents. **Neither is cited
here and no figure from either is in this repository.** Where the code wants a
reference to compare against it reads it from a private module outside the tree;
with no private data installed it falls back to targets derived from the
published sources above, and the derivation is written next to the constant.
