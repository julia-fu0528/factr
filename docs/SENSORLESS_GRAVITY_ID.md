# FACTR hold-and-capture gravity identification

`identify_gravity.py` now defaults to the B601-method **HOLD/DRAG → CAPTURE → DRAG**
workflow, implemented in `gravity_capture.py`. It does not switch torque off
between capture directions or require manual reproduction of seven joint angles.
This is new hardware-control code tested in software and with a synthetic plant;
it has not yet been validated physically on FACTR.

## Start

Stop the old controller. Secure the base and keep a close catch ready. Support
FACTR in a comfortable pose away from mechanical stops and cable tension.

```bash
.venv/bin/python identify_gravity.py --output outputs/diagnostics/hold_capture_01
```

Type `START` when ready. Support the arm during the two-second current ramp.
When HOLD appears it maintains the captured pose with bounded current/PD control.
All seven joints receive pose support and oscillate together during capture, as
in B601. Optional `--joint 4` restricts motion to J4; this is a FACTR extension.

Current limits default to 350 mA or the hardware limit, whichever is lower.
`--joint4-max-current-ma 450` permits a bounded J4-only headroom trial; it is
not an official FACTR recommendation or a validated continuous rating. The
option accepts 1–900 mA and never increases the stored motor Current Limit.
If the controller cannot support the pose, physically support it and release;
do not rely on a hold that is drifting or saturated.

## Commands: type the letter and press Enter

| Command | Effect |
|---|---|
| `c` | Capture the current pose: 3 seconds to unload the hand, then all joints move smoothly ±0.05 rad (2.86°), two 8-second periods. Returns to DRAG; be ready to support/reposition. |
| `h` | Freeze at the current pose; cancel/reject an in-progress capture. |
| `d` | Gravity-only DRAG so you can support and reposition the arm slowly. |
| `r` | With the arm supported, fade torque off over 3 seconds and exit. |
| `q!` (or `q`) | Immediately disable torque and exit; the arm may fall. |

Ctrl-C during capture/drag/ramp changes to HOLD; Ctrl-C in HOLD starts release;
Ctrl-C during release disables immediately. Communication, thermal, speed, and
hard tracking faults disable torque immediately. Keep a catch ready throughout.
The controller keeps servicing the bus/watchdog while waiting for commands;
there is no blocking terminal prompt while powered.

Release retains damping. If speed exceeds 0.12 rad/s or the pose changes over
2 degrees during the fade, release pauses and HOLD resumes at the present pose.
Fully support the arm and type `r` again. The separate 0.5 rad/s hard speed guard
still disables torque if a sample already exceeds it. This pause cannot guarantee
support through a communication or hardware fault.

During CAPTURE, do not push/support the moving assembly and do not let it rest
on the catch: external forces contaminate the measurement. The arm must carry
its own weight. Contact cannot be detected reliably from these measurements.
Use `h` to reject a capture if you touch it or it catches on anything.

Start with **one capture**. After 19 seconds it returns to DRAG, not HOLD.
Be ready to support/reposition then, or type `h` to hold. Use `r` once supported
to finish. Collect several spread poses in one session (for example 5–10), with
repeat captures to check consistency. No automatic return to a distant pose.

## What is measured

This retains the estimator from `b601/gravity_drag.py` but adapts motion to FACTR:
three seconds settling, then two symmetric 8-second cycles of ±0.05 rad. Each
quarter-cycle is a quintic move: center → positive endpoint → center → negative
endpoint → center, with zero target velocity and acceleration at each junction.
All joints move together. Unlike the original B601 triangle, there is no startup
position step or instantaneous reversal. Peak target speed is 0.046875 rad/s.

The PD residual `KP * (target - q) + KD * (desired_velocity - velocity)` is averaged
over both complete periods, including reversals. The final pose, raw model gravity,
feedforward, and average residual are saved. Fitting uses
`feedforward + mean_PD_residual = scale * raw_gravity (+ optional_bias)`.
Scale-only least squares is the default, matching `b601/scripts/fit_gravity.py`.

Measured current remains in the diagnostic log; it is no longer the gravity
estimator. There is no matched-angle bin or measured-velocity gate. Stuck motion
is recorded with warnings because static friction can bias the mean residual.
A saved capture or a low fitting residual does not demonstrate accurate gravity.

The hardware implementation necessarily differs: FACTR uses a 250 Hz target host PD loop
and Dynamixel current commands, rather than B601's motor-native MIT PD. FACTR's
own stiffness gains, 0.85 feedforward, 350 mA total-current caps, and watchdog
remain. J3 damping is increased from 0.08 to 0.12 Nm·s/rad as a conservative
starting adjustment; it is not a hardware-validated tuning result. Unlike B601, any clipped measurement invalidates a capture: the requested
PD correction was not delivered. B601's absolute torque/friction thresholds are
not transferred to these much smaller motors. No inertia shaping or friction
compensation is enabled. This revised capture needs physical validation.

If measured speed exceeds 0.20 rad/s during capture, the capture is rejected
and HOLD resumes at the current pose to remove the moving target. Support the
arm and use `r` to finish. The independent hard 0.5 rad/s cutoff still runs first;
this earlier cancellation cannot guarantee recovery from an unstable motion.

If the sweep saturates for more than 150 ms or any joint moves over 5° from the capture pose, it
rejects the capture and attempts HOLD at the present pose. Hard speed (>0.5 rad/s),
tracking (>8°), joint-limit, temperature (60 C), stale-state (100 ms), or watchdog
faults abort to torque off. These are software abort criteria, not guaranteed
mechanical travel limits. Guard against a fall independently.

## Files and fitting

`metadata.json`, `samples.csv`, `captures.jsonl`, `calib.csv`, and `report.json` are directly
overwritten when starting another session in the same output folder, without
backups. Each `c` within one session appends a capture. Other files are untouched.
`fault.json` records the actual triggering state on a fault, including the joint
velocities and preceding command; it too is replaced on a fresh session.

```bash
.venv/bin/python identify_gravity.py --output outputs/diagnostics/hold_capture_01 --analyze-only
```

Analysis never opens the serial port. `calib.csv` uses B601's columns, extended
to seven joints: `q1..q7`, `graw1..graw7`, `ff1..ff7`, `resid1..resid7`.
`report.json` contains the proposed per-joint scales and fit errors. Add `--bias`
to the analysis command to also fit a constant offset. A single pose can produce
a scale estimate, but several spread poses are needed to assess it. Rank-deficient
scale/bias fits and effectively zero gravity signals are skipped. Small gravity
signals can still produce unreliable scales; inspect motion warnings and repeatability.
The fitted scale is the total coefficient on raw gravity (it already includes
the current 0.85 feedforward), not a multiplier to apply on top of 0.85.

A fit is never automatically applied. Existing captures from the old current-bin
method are explicitly skipped, not reinterpreted as B601 measurements. New captures
are required. Encoder offsets and active gravity parameters are unchanged.

The older manual-reset current-threshold experiment remains available only with
`--thresholds`; it is no longer the default workflow. Its analyze-only mode reads
`trials.jsonl`, whereas the new default reads `captures.jsonl`.

## Official-source audit (2026-10-01)

Upstream revision `7a07ab3629af03a91c0198df7c44a0082dca77ab` explicitly warns
that its host PD tuning requires at least 200 Hz:
[FACTR controller](https://github.com/JasonJZLiu/FACTR_Teleop/blob/7a07ab3629af03a91c0198df7c44a0082dca77ab/src/factr_teleop/factr_teleop/factr_teleop.py#L283).
The capture controller now targets 250 Hz and checks achieved rate each second,
stopping below 200 Hz. This does not make Python a real-time controller or
establish stability of our different gains. The previous 50 Hz capture loop
was inconsistent with this upstream guidance.

The [upstream driver](https://github.com/JasonJZLiu/FACTR_Teleop/blob/7a07ab3629af03a91c0198df7c44a0082dca77ab/src/factr_teleop/factr_teleop/dynamixel/driver.py#L207)
clips to ±900 raw register units. On J4's XM430 this is ±2421 mA, not ±900 mA.
The [ROBOTIS manual](https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/#current-limit38)
specifies 2.69 mA per unit and a separate stored Current Limit. Neither value
establishes a suitable continuous current for this assembled arm. Our J4 trial
option of 450 mA is a local engineering choice, not an upstream calibrated value.

The checked official controller/configuration does not document or validate
B601's PD-residual gravity identification method; no prohibition of that method
was found there. Matching conversion constants and signs does not independently
validate physical motor order, offsets, or assembly.

The connected arm's Return Delay registers were found at 250 on all seven
motors on 2026-10-01. With torque off, state reads took approximately 6 ms.
Setting Return Delay to 0 as required by upstream setup reduced median reads
to approximately 1 ms; all seven torque-enable registers remained 0. This is a
persistent communication setting. Capture startup now checks Return Delay and
benchmarks state reads before enabling torque. This preflight does not replace
the achieved-loop-rate monitor during the actual session.
