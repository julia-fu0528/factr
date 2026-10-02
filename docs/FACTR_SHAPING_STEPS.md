# FACTR identification and seven-DOF shaping

The contact point is the last-link handle (confirmed by the operator). Its exact
grip-point translation relative to `link_7`, and the handle-axis orientation,
remain to be measured. Do not substitute the simulated Franka wrist offset.

## 1. Verify the model and joint mapping — current step

Offline consistency check (no motor connection):

```bash
.venv/bin/python factr_diagnostics.py audit
```

The report checks gravity against the potential-energy gradient, positive link
inertia, and Jacobian rank/null space at HOME, the saved calibration pose, and
20 nearby configurations. This tests mathematical consistency of the URDF, not
the physical accuracy of its masses, COMs, motor signs, or encoder offsets.
Output files are never overwritten: use `--output` for another audit.

For the guided measurement, stop any teleop/gravity controller normally first
and support the arm. The recorder refuses torque-enabled motors; it will never
disable or enable torque itself. Use the project terminal:

```bash
.venv/bin/python factr_diagnostics.py record --guided \
  --output outputs/diagnostics/kinematics_01
```

1. Support the arm in `docs/calibration_pose.png`, press Enter, and hold still
   for five seconds. Reproducing the reference must be checked physically.
2. Each subsequent prompt names a joint, numbered from base (1) to wrist (7).
   Support the other links. Press Enter, then gently move the named joint out
   and back twice for 12 seconds. Use a comfortable small range; do not pull
   against cable tension or mechanical stops.
3. Stop with Ctrl-C at any point; already flushed CSV samples remain on disk.

Each segment records calibrated angles, raw encoder angles, signed velocities,
present current, model gravity, and read timing at a requested 50 Hz. Currents
are electrical telemetry, **not measured joint torque or force**. The generated
analysis reports joint excursions and reference deviations. Dominant joint
matching screens motor IDs; it does not validate signs or prove assembly is
correct. Sign validation additionally needs the observed physical direction.

Reanalyze a partial recording:

```bash
.venv/bin/python factr_diagnostics.py analyze outputs/diagnostics/kinematics_01
```

## 2. Verify physical gravity compensation — pending measurements

After reviewing stage 1, compare supported static holds at several poses and
slow motion in both directions, with compensation off/on. Record actual current,
commanded current before/after limits, q, velocity, and timing from the SAME
controller process. A second program must not open the controller's serial port.
An independent calibrated load/handle F/T measurement can validate physical
holding effort. Alternatively, the sensorless hold-and-capture procedure
in [SENSORLESS_GRAVITY_ID.md](SENSORLESS_GRAVITY_ID.md) estimates holding current
under explicit quasi-static, symmetric-friction assumptions. The default workflow
now maintains HOLD between directional measurements; the old manual-reset
threshold experiment is available only with `--thresholds`. An F/T sensor is
not mandatory for that procedure. Supported torque-off motion alone cannot fit gravity parameters.
Keep position offsets separate from mass/COM and current-to-torque calibration.
In particular, investigate joint 4's previously reported directional resistance.

The first guided recording `kinematics_01` passed the dominant-motor check for
all seven joints. Its reference segment moved over the full five seconds, but
contains a one-second window with at most 0.352 degrees excursion per joint.
This supports using the recording; it does not validate the absolute zeros or
physical rotation signs. Recorded frequency was approximately 49.8–49.9 Hz.

Next supported check, using the existing compensation settings:

```bash
.venv/bin/python gravity_comp.py run --calibration calibration.json \
  --duration 30 --gain 0.85 --damping 0.01 \
  --log outputs/diagnostics/gravity_j4_01.csv
```

Begin in the reference pose, support the arm throughout, and allow the two-second
ramp to finish. Hold roughly still for five seconds, then gently bend/unbend
joint 4 by about 10–20 degrees, twice, supporting the other links. Finish with
a supported hold. Do not let go to test whether it stays up. Support it before
Ctrl-C or the automatic timeout, both of which disable torque. Stop early for
unexpected pulling or vibration. Report which direction feels harder and
whether it is hard at startup or during steady motion.

The optional log records predicted gravity, requested torque, saturated raw
current command, current limits, electrical current telemetry, and timing. The
current sample is obtained with the state read BEFORE the new command is sent;
analyzer comparisons must account for this delay. Measured current is not an
independent force/torque measurement and cannot validate physical gravity alone.
This check does not change calibration or add shaping/friction compensation.

### Joint 4 reduction withdrawn after sagging

The trial scale 0.30/0.85 made lowering easier but subsequently caused reported
sagging. It has been removed: all model correction scales are 1. Normal gain
0.85 now reproduces the earlier gravity command for all seven joints, including
in preview and teleop. Restart the controller and omit any joint-4 override:

```bash
.venv/bin/python gravity_comp.py run --gain 0.85 --duration 30 \
  --log outputs/diagnostics/gravity_restored_01.csv
```

Support the arm before stopping/starting and begin in the reference pose.
A drop in downstream world position does not establish that the wrist joints
are rotating: joint 4 alone can carry the whole downstream assembly downward.
Compare individual q4–q7 changes and current limits before tuning. Neither easier
lowering nor less sagging proves the model correct; hand support, friction,
encoder geometry and current-to-torque conversion still need separating.
Encoder calibration and the URDF have not been modified. The optional log
retains correction-factor columns (now all 1) to distinguish these runs from
the earlier reduced-compensation recordings.

## 3. Identify friction and apparent inertia — pending stage 2

Separate startup force, slow-motion effort, and acceleration effort. Repeat
both directions at multiple postures; measure wrist roll, pitch, yaw separately.
Use independent known effort or an F/T reference for quantitative identification.
Include actuator/reflected inertia: the rigid-link URDF is not a measurement of
the geared arm's total inertia. Do not copy Piper/B601 friction or inertia tables.

## 4. Validate a passive external-torque observer — pending identification

Run the observer with shaping output disabled. Include the actual saturated
motor command/validated current conversion, gravity, and identified losses.
Test hands-off, sustained hand loading, and both motion directions against the
reference sensor. Rest alone must not authorize bias learning: a stationary arm
can still be pushed. Static friction creates unobservable force intervals.
Only estimate a single handle wrench when its known-contact assumption holds.

## 5. Add seven-DOF shaping — pending observer validation

Use a 6x7 Jacobian at the physical grip point, and explicitly retain/shape the
one-dimensional null-space inertia. J.T Lambda_d J alone is rank deficient.
Start with natural null-space inertia and modest null damping. Identify separate
translation/rotation targets, define which axes stay base-aligned, and use a
dimensionless scaled Jacobian for singularity gating. Retain current, speed,
temperature, timing, and watchdog limits; establish loop stability with FACTR's
measured timing rather than copying B601's gain ceiling.

No shaping controller has been enabled by these diagnostic tools.
