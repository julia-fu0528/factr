# FACTR 7-DOF gravity compensation

## Franka simulation teleoperation

Use your saved calibration; you do not need to recalibrate every time.
The commands below use `calibration_reference.json`. If you calibrated into
`calibration.json` or another file, substitute that filename in Terminal 2.
Stop any earlier leader/gravity controller before starting.

Terminal 1 starts the Isaac Sim window (the first launch may download Franka assets):

```sh
cd /home/yuchen/projects/factr
bash run_isaac_teleop.sh
```

Wait for `READY: Franka simulation`. Return the leader to its saved reference
pose (each joint must be within approximately 20 degrees) and support it.
Terminal 2 starts **teleop with leader gravity compensation** for 10 minutes:

```sh
cd /home/yuchen/projects/factr
# After reconnecting USB, set latency (use the resolved device if not ttyUSB0):
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
.venv/bin/python factr_teleop.py --calibration calibration_reference.json --gravity-comp --duration 600
```

This loads the saved calibration; it does not recalibrate. Compensation ramps
up over two seconds. Once running, move the leader to control the simulated arm.
Support the leader before pressing Ctrl-C in Terminal 2 or before the 600-second
timer ends, since compensation switches off. Then close the simulation window.

Support and move the physical arm; the simulated Franka follows the seven calibrated
joint angles. This is direct joint mapping, so calibration and motor order/signs
determine the simulated posture. The gripper stays open in this initial version.
The simulator clamps to Franka joint limits and limits target motion to 1.5 rad/s.
It holds the last commanded target when no fresh packet arrives for 250 ms.

The simulated wrist roll (joint 7) has a **+45-degree mapping offset**, applied
to the initial pose and incoming leader angles before joint-limit clamping.
This does not change encoder calibration or the leader gravity URDF. Restart
the simulator to load it. If the correction is in the opposite direction, use:

```sh
bash run_isaac_teleop.sh --wrist-roll-offset-deg -45
```

Use `--wrist-roll-offset-deg 0` to restore the original direct joint mapping.

For an optional read-only test with no leader gravity compensation, use this
instead of the Terminal 2 teleop command and support the full arm weight by hand:

```sh
.venv/bin/python factr_teleop.py --calibration calibration_reference.json
```

Do not run `gravity_comp.py run` at the same time: one process owns the serial bus.
The existing FTDI latency requirement (1 ms), calibration checks, current limits,
and shutdown handling apply to `--gravity-comp`. Support the arm at shutdown.
Closing the simulator does not stop leader compensation; stop Terminal 2 with
Ctrl-C as well. Neither process sends commands to a real Franka.

If startup reports `already torque-enabled`, first stop any running leader
controller with Ctrl-C. If none is running, support/rest the arm and run:

```sh
.venv/bin/python gravity_comp.py stop
```

This disables torque on arm IDs 1–7, clears their watchdog/goal-current registers,
and checks that they report `torque=0`. It leaves the gripper unchanged. Then
return to the reference pose and restart Terminal 2. This recovers leftover
motor state; it does not identify why an earlier process failed to clean up.

`run_isaac_teleop.sh` uses the installed Isaac Sim 6.0 environment
at `/home/yuchen/miniforge3/envs/isaaclab`. Set `ISAAC_PYTHON` to override its Python
path. The launcher supplies the missing `libxml2.so.2` from the existing `tacex50`
environment using a process-local preload; override with `ISAAC_XML_LIBRARY` if
needed. It does not modify either environment or the system GPU driver.
The leader uses this project's `.venv`; UDP on localhost connects the two.
The simulator never opens the motor port, and rendering runs separately from
the gravity loop. No extra packages are needed for the stream.

To test the simulator without any physical robot:

```sh
bash run_isaac_teleop.sh --demo
# Short headless test with tracking statistics:
bash run_isaac_teleop.sh --headless --demo --steps 360 --report /tmp/factr-sim.json
```

Use `--speed`, `--timeout`, or `--udp-port` on the simulator to change its defaults.
If changing the UDP port, pass the same `--udp-port` to `factr_teleop.py`.
The viewer uses Isaac Sim's bundled Franka and articulation controller APIs.

The older installed Isaac Sim 5.0 runtime passed physics tests but crashed in
its RTX renderer with this machine's driver. The launcher therefore uses 6.0;
its windowed Franka demo has been tested successfully.

Validation (2026-09-30): 12 unit tests passed. Isaac Sim 6 completed a 240-step
windowed demo and a 1,800-step headless run with eight seconds of real FACTR
input. The latter switched to holding its target after input stopped and exited
cleanly. Leader torque was never enabled during these tests. The simulation
uses a physics position controller, so tracking can have lag and gravity-induced
position error; it does not teleport the robot to each encoder sample.

## Standalone gravity compensation

Standalone Python, no ROS or follower arm required. Targets the original upright
FACTR Franka leader: IDs 1–7, XC330-T288 except XM430-W210 on joints 2 and 4.
The gripper (ID 8) is inspected but never commanded. Other leader geometries or
servo arrangements require a matching model and conversion constants.

## Setup

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python gravity_comp.py inspect
```

The default port is the connected FTDI adapter `FTAKRKBU`, at 4 Mbps.
Override with `--port /dev/serial/by-id/... --baud 4000000` as needed.
USB powers the adapter; the motors also need their external power supply.
`inspect` only pings and reads motor registers. It never enables/disables torque.
If no motors respond, check power and try `--baud 1000000` or `--baud 57600`.
Use Dynamixel Wizard to configure IDs/baud if necessary; this script does not
automatically rewrite bus settings.

## Calibrate, then run

Secure the base upright and manually place the arm in this reference pose:

![Reference pose](docs/calibration_pose.png)

```sh
.venv/bin/python gravity_comp.py calibrate
.venv/bin/python gravity_comp.py preview
```

Calibration uses the original FACTR method: encoder offsets are snapped to
multiples of 90 degrees, with a 20-degree per-joint reference-pose tolerance.
It assumes the original motor indexing, joint order, and signs.
An arbitrary pose cannot establish absolute joint angles.
Check that preview angles correspond to physical motion before enabling torque.
Calibration is stored in `calibration.json`; recalibrate after mechanical changes.
Encoder whole-turn resets are handled automatically.

For a new calibration file after reassembly, use:

```sh
.venv/bin/python gravity_comp.py calibrate --calibration calibration_gravity_new.json
.venv/bin/python gravity_comp.py preview --calibration calibration_gravity_new.json
.venv/bin/python gravity_comp.py run --calibration calibration_gravity_new.json --duration 30
```

Calibration averages approximately one second of stationary encoder samples.
It does not write motor encoder/homing registers. The continuous-offset mode
and its `--reference-pose` flag have been removed. Existing calibration files
are preserved and remain readable; they are not converted by this code change.
Use the same new `--calibration` path for preview, gravity runs, and teleop if
you want those commands to use your recalibration. Successful calibration
atomically overwrites the selected JSON (default: `calibration.json`). A failed
capture leaves the previous file intact. Passing the indexing check does not independently verify joint
order, direction, or the physical pose.

Set FTDI latency to 1 ms, replacing ttyUSB0 if your port resolves elsewhere:

```sh
readlink -f /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRKBU-if00-port0
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

Return the arm to the reference pose, support its weight, and start:

```sh
.venv/bin/python gravity_comp.py run --duration 30
```

The default is 85% modeled gravity compensation, 0.01 N m s/rad damping, a
two-second ramp, 200 Hz, and a 350 mA cap per joint. These initial limits can
undercompensate the arm. Keep it supported during initial tuning and at shutdown.
Use `--gain`, `--max-current-ma`, and `--duration` to tune after validating the
pose/signs. There is no posture servo, joint-limit repulsion, friction dither,
or follower communication in this initial controller.

`run` switches the arm motors to current mode and leaves that mode configured
after exit. It checks model numbers and motor faults, refuses already-enabled
motors, and sets a 200 ms bus watchdog. Ctrl-C/SIGTERM and failures trigger
best-effort zero current and torque disable on every arm motor. A USB/power
fault can prevent cleanup; the watchdog is not a substitute for accessible
motor power cutoff. Abrupt torque loss lets the arm fall under its own weight.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests check dynamics against the potential-energy gradient, encoder wrapping,
current saturation/signs, and cleanup after partial startup/disconnection.
These are software tests, not physical validation of calibration or balance.

Initial read-only inspection (2026-09-20): all IDs 1–8 responded at 4 Mbps;
models matched the expected arrangement, firmware was 53 (XC330) / 50 (XM430),
all hardware error registers were zero, torque was disabled, and bus voltage
was 12.2 V. All five software tests passed. Powered compensation has not yet
been tested; manual reference-pose calibration remains required.

## Train a simulated Franka to pick up a cube

This uses Isaac Lab 2.2 / Isaac Sim 5 headless physics in the existing `tacex50`
environment, and RSL-RL PPO 2.3.3 installed in the project-local `.training_deps`.
It does not connect to the FACTR hardware. The policy observes simulation state
(including cube position), not camera images. Its eight actions control seven
joint position targets and the gripper. Arm gains are Kp 400 / Kd 40, at 50 Hz.
The standard Isaac Lab cube-lift task randomizes the cube's position on the table.

The local training dependencies are already installed. To recreate that overlay:

```bash
uv pip install --python /home/yuchen/miniforge3/envs/tacex50/bin/python \
  --target .training_deps --no-deps -r requirements-training.txt
```

Train (2048 environments, 1500 PPO iterations):

```bash
bash run_cube_policy.sh --output outputs/cube_pick
```

Checkpoints, task/agent configurations, metadata, and TensorBoard logs are saved
in that directory. The final checkpoint is `model_final.pt`; the exported
inference-only TorchScript model is `exported/policy.pt` (36 inputs, 8 outputs).
Joint actions are `default_joint_pos + 0.5 * action[:7]`; negative gripper action
closes the fingers, nonnegative opens them. See `metadata.json` for observation
order and joint defaults. This policy needs cube/target state as well as joints;
it is not yet a controller for the physical leader.

Evaluate on 384 fresh episodes and save one episode for replay:

```bash
bash run_cube_policy.sh --eval --checkpoint outputs/cube_pick/model_final.pt \
  --num-envs 128 --eval-rounds 3 --seed 1000 --record \
  --output outputs/cube_pick_eval
```

`outputs/cube_pick_eval/evaluation.json` reports successes. Success requires the
cube center above 15 cm for half a second, within 8 cm of the end effector, and
moving slower than 0.25 m/s. This is stricter than the training reward.

Watch the recorded evaluation with the working Isaac Sim 6 renderer:

```bash
bash replay_cube_policy.sh
```

This is a replay of recorded simulation states, **not live policy inference**.
It selects a successful episode if one exists, otherwise the highest lift, and
prints the seed, environment index, and success flag. Close the window to stop.
Training/evaluation stays headless because the installed Sim 5 renderer is
incompatible with this machine's current graphics driver.

Resume from a checkpoint for additional iterations in a new output directory:

```bash
bash run_cube_policy.sh --checkpoint outputs/cube_pick/model_final.pt \
  --iterations 500 --output outputs/cube_pick_continued
```

The baseline task and PPO settings come from the installed
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) cube-lift environment.

Completed run: 1500 iterations (73,728,000 transitions), seed 42. The final
checkpoint passed 384/384 evaluation episodes across seeds 1000–1002 with the
criterion above. This result covers the task's configured cube-position range,
not unseen objects, camera observations, or transfer to hardware.

## FACTR inertia-shaping measurements

The staged identification procedure is in
[docs/FACTR_SHAPING_STEPS.md](docs/FACTR_SHAPING_STEPS.md).
The B601-method sensorless hold-and-capture workflow is documented in
[docs/SENSORLESS_GRAVITY_ID.md](docs/SENSORLESS_GRAVITY_ID.md). It requires an
operator and an unloaded catch and does not automatically change calibration.
Run `identify_gravity.py --output outputs/diagnostics/hold_capture_01`:
`c` oscillates all seven joints and returns to DRAG, `d` lets you reposition, `h`
holds, and `r` releases once supported. No manual return to seven target angles.
Capture uses smooth ±0.05 rad motion with 8-second periods and B601’s average
PD-residual estimator. Three seconds settling plus two periods takes 19 seconds.
The capture loop targets 250 Hz (minimum achieved 200 Hz). Optional
`--joint4-max-current-ma 450` allows a J4-only headroom trial; all other joints
stay capped at 350 mA and stored motor limits remain enforced. This is an
experimental setting, not an official FACTR current recommendation.
Fit with `.venv/bin/python identify_gravity.py --output outputs/diagnostics/hold_capture_01 --analyze-only`
(add `--bias` for scale plus offset). Fits remain provisional and are not applied automatically.
Start with `factr_diagnostics.py audit`, then the guided torque-off joint check.
These diagnostics do not change calibration or motor commands and do not yet
identify friction, external force, or physical inertia.

The trial joint-4 gravity reduction was withdrawn after reported sagging.
All gravity-model scales are now 1; `--gain 0.85` restores the earlier
compensation for every joint, including in teleop. Omit `--joint4-gain 0.3`.
Restart a running controller to pick up the change. Encoder calibration and
URDF parameters are unchanged; physical gravity accuracy is still unverified.

### J4 breakaway measurement: first step

Stop other arm controllers, support the arm in a comfortable pose away from
joint limits, and run:

```bash
.venv/bin/python j4_breakaway.py --output outputs/diagnostics/j4_breakaway_01
```

Press **Enter** to enable HOLD and support the two-second startup ramp. Once
`HOLD ready` appears:

1. Type `+` and Enter. Unload your hand during the three-second settling period,
   then keep a close catch **without touching or pushing** throughout the sweep.
2. Wait for the result and automatic small return motion to finish, until
   `HOLD ready` appears again. Type `-` and Enter for the opposite direction.
3. Repeat `+`, then `-` once if comfortable. Support the arm before `q` and Enter
   to disable torque. The session also ends automatically after 180 seconds.

`+` increases the calibrated J4 angle; `-` decreases it. These are model
directions, not labels for lifting/lowering. `x` and Enter cancels a measurement
and returns to HOLD; `q`, Ctrl-C, or a fault disables torque immediately.

Inertia shaping is off. All joints use the existing 0.85 gravity setting with
PD pose support during HOLD. During a sweep, J4's holding current is frozen and
stepped by 2.69 mA every 0.5 seconds until small sustained motion is detected;
the other six joints retain pose support. The sweep stops at onset or its
current/motion bounds, then brakes and slowly returns through the measured
displacement. Total current stays capped at 350 mA (or the lower stored motor
limit). The positive sweep can reach that total-current ceiling; the negative
sweep retains its maximum 120 mA offset from J4's baseline. The CLI prints the
bound for each trial. No threshold within those bounds is recorded as `censored`,
with the limiting bound named, not a successful measurement.

The output folder contains `samples.csv`, `trials.jsonl`, `report.json`, and
`metadata.json`. Reusing the folder directly overwrites these files. Inspect
the report without connecting to the arm with:

```bash
.venv/bin/python j4_breakaway.py --output outputs/diagnostics/j4_breakaway_01 --analyze-only
```

Pairs require successful opposite-direction trials starting within one degree
on every joint. The measured thresholds include gravity bias, friction, and
onset dynamics; they do not directly measure hand force or isolate a friction
coefficient. This step changes neither calibration nor the assistance
controller. The state machine has software tests; hardware behavior still
requires an operator-supervised trial.

## Sources

Model, reference image, motor conversion, and joint conventions come from
[FACTR_Teleop](https://github.com/JasonJZLiu/FACTR_Teleop), revision
`7a07ab3629af03a91c0198df7c44a0082dca77ab` (Jason Jingzhou Liu and Yulong Li,
Apache-2.0; see `models/LICENSE`). The URDF is copied unchanged; visual mesh
files are omitted because only the inertial/kinematic model is loaded.
Register definitions follow the ROBOTIS manuals for
[XC330-T288](https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/) and
[XM430-W210](https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/).

## Inertia shaping: offline observer first

`inertia_shadow.py` replays a `gravity_comp.py --log` CSV without opening the
motor port or sending commands. It uses the previous capped motor command,
model momentum and gravity/Coriolis terms to estimate unexplained effort.
That residual includes friction and gravity/model error; it is not measured
hand force. It never auto-zeroes a stationary residual.

Recording with `gravity_comp.py run --log PATH` directly overwrites an existing
CSV at that path, with no backup or timestamped copy.

```bash
.venv/bin/python inertia_shadow.py outputs/diagnostics/gravity_sag_01.csv
```

The output folder defaults to `outputs/diagnostics/inertia_shadow_01`;
`shadow.csv` and `report.json` are directly overwritten on each analysis.
A candidate task-inertia reduction factor of 1.25 preserves modeled nullspace
inertia in the seventh DOF. Assistance is gated near geometric singularities
and for two seconds after observer initialization/reset. This proposal is for
offline inspection only: it does not implement an active actuator controller,
power/passivity limits, identified rotor inertia, or a calibrated grip frame.

For a new comparison, keep gravity gains fixed, log quiet supported pauses,
slow J4 motion and gentle starts/stops through the same small range. Mark or
report which periods involved hand support. Do not fit physical inertia from
unknown hand force alone. Hardware assistance remains disabled until observer
behavior and a bounded controller have been evaluated.

Offline sensitivity comparison (does not change motor gains):

```bash
.venv/bin/python inertia_shadow.py outputs/diagnostics/inertia_baseline_01.csv \
  --model-gravity-scale 0.85 --output outputs/diagnostics/inertia_baseline_shadow_g085
```

A nominal full-gravity observer paired with a 0.85 gravity motor command reports
approximately 0.15 times modeled gravity as residual even while stationary.
Changing the assumed gravity scale demonstrates this sensitivity; it does not
prove which model is physically correct or calibrate the arm. Never interpret
the smaller residual by itself as validated hand-force sensing.

## Experimental live inertia assistance

The live adapter is `factr_inertia.py`. Unlike the offline commands below,
**this command enables motor torque after you press Enter at the startup prompt**. Stop the existing
controller, support the arm in the reference pose, and keep a catch ready.

```bash
.venv/bin/python factr_inertia.py --controller balanced --duration 120 \
  --target-mass 0.15 0.15 0.15 \
  --target-inertia 0.0002 0.0005 0.00002 \
  --output outputs/diagnostics/balanced_drag_01
```

It starts gravity-only at 0.85 on all seven joints, damping 0.01, and total
current caps of 350 mA (or the stored motor limit if lower). The existing
encoder calibration and gravity model files are not modified. Return Delay 0,
USB latency 1 ms, correct motor IDs/models, torque-off startup, and bus read
speed are checked before enabling. Support the two-second startup ramp.

Commands require Enter and are serviced without a blocking prompt once powered:

| Command | Behavior |
|---|---|
| `a` | Toggle assistance ON/OFF after observer warmup; no three-second timer. Every toggle prints the new state. |
| `q` / Ctrl-C | Disable all arm torque and exit. Support the arm first. |

Keep holding the handle while testing. Use small comfortable movements; do not
force J4 through its resistance to obtain a smooth trace.

**Balanced drag is now the CLI default.** It implements B601-style target
Cartesian inertia with a seven-joint completion, rather than a uniform
lightening factor. `--target-mass MX MY MZ` specifies translational masses in kg;
`--target-inertia IX IY IZ` specifies rotational inertias in kg m². Axes are
the URDF `link_7` local x/y/z axes, not world axes. The default modeled handle
point is the `link_7` origin. Set `--handle-offset X Y Z` in meters, expressed
in those same axes, if the grip point differs; its physical offset has not
been measured. The shifted Jacobian includes the angular-velocity contribution
to linear velocity at that point.

The provisional defaults are masses `[0.15, 0.15, 0.15]` and rotational inertias
`[0.0002, 0.0005, 0.00002]`, chosen on the scale of this 0.67 kg link-only URDF,
not copied from B601's much heavier model. They are targets, not measured hand
inertias or a guarantee of achieved feel. At the reference pose, the natural
operational inertia diagonal is approximately `[0.297, 0.111, 0.129]` kg and
`[0.000256, 0.001026, 0.0000125]` kg m²; off-diagonal coupling also matters.
The controller can assist directions heavier than the target and resist
directions lighter than it. `--kappa 0.5` caps positive task-mode gains;
`--resist 0.25` caps negative gains. Live ranges are 0..1 and 0..0.5,
respectively, as experimental engineering limits, not proven stability bounds.
These are task eigenmode gains, not per-joint gains or the old `--factor`.

The extra internal degree of freedom is handled explicitly. In inertia-whitened
coordinates, let `A = J M^(-1/2)`, let `Q` span the six task directions, and let
`N = I - Q Qᵀ` project onto the internal direction. With `B = A Q`, the full
target is

```text
H_d = Q (Bᵀ Lambda_d B) Qᵀ + N
M_d = M^(1/2) H_d M^(1/2)
S_task = clip_eigenvalues((Bᵀ Lambda_d B)^(-1) - I, -resist, kappa)
tau_shape = M^(1/2) Q S_task Qᵀ M^(-1/2) residual_used
```

Unlike `Jᵀ Lambda_d J` alone, this target is full rank for seven joints. The
internal direction keeps its natural modeled inertia with zero shaping gain:
there is no added posture spring or automatic elbow positioning. At deficient
task rank, shaping goes to zero. Geometry screening scales translation by a
0.2 m characteristic length and fades shaping as the minimum/maximum Jacobian
singular-value ratio falls from 0.04 to 0.02. The observer's internal torque
component is removed before applying a one-current-step numerical deadband;
this deadband is not a friction calibration or hand-force identification.

Balanced output is signed: resisting torque and torque on currently stationary
coupled joints are allowed. The old per-joint velocity-direction gate and
80 ms dwell do not apply. A two-second activation ramp starts on enabling
and restarts after a geometry gate, but does not restart at each reversal.
Torque caps, slew bounds, speed fade, current headroom, and positive-power
limits use one common scale for the entire torque vector. This preserves the
task/internal separation before current quantization. Quantization perturbs
that separation; `nullspace_output_nm_*` logs the resulting component. No
friction compensation, learned static bias, or breakaway feed-forward is
silently imported from B601. The modeled inertia excludes identified motor
rotor/gear inertia. The target law and software bounds do not establish
physical stability or eliminate the previously observed J4 resistance.

`--torque-rise-rate` controls the additional torque magnitude growth bound
(default 0.04 Nm/s, configurable above zero through 0.4). This differs from
the two-second activation ramp: it can restrict output continuously as the
proposed torque changes. Startup and metadata report the selected value.
Larger values allow faster torque changes while preserving the torque, current,
and instantaneous power caps; they have not been validated as hardware-safe
settings. To compare the evaluated 0.16 Nm/s setting while keeping all targets
fixed and saving a separate recording:

```bash
.venv/bin/python factr_inertia.py --controller balanced --duration 120 \
  --torque-rise-rate 0.16 --output outputs/diagnostics/balanced_drag_rate016
```

The offline `balanced_replay.py` accepts the same option. Without an override it
uses the recording's rate, or 0.04 for older recordings without that metadata.

To bypass the experimental **additional torque, positive-power, and torque-rise
caps**, balanced mode supports an explicit `--assist-limits current-only`
profile. It keeps the total-current envelope (350 mA or the motor's lower stored
limit), current quantization, two-second activation ramp, task eigenvalue limits
(`--kappa`/`--resist`), singularity/speed gates, and fault shutdowns. It does not
make the controller unbounded, remove target-gain clipping, or add friction
compensation. The default remains `--assist-limits bounded`.

```bash
.venv/bin/python factr_inertia.py --controller balanced \
  --assist-limits current-only --duration 120 \
  --output outputs/diagnostics/balanced_current_only
```

This changes the possible motor output, not just the logging. No physical
stability validation has been performed for this profile. Startup names the
disabled limits; metadata records the profile and sets the three inactive cap
values to `null`. Bypassed trace stages pass through their input unchanged.
`--torque-rise-rate` has no effect with this profile. The same profile option
is available in `balanced_replay.py` for fixed-recording comparisons without
motor access.

The earlier controller is retained for comparison:

```bash
.venv/bin/python factr_inertia.py --controller uniform --factor 1.25 --duration 120
```

`--factor` is rejected in balanced mode to avoid silently ignoring old commands.
For an offline comparison on an existing recording (no hardware access):

```bash
.venv/bin/python balanced_replay.py outputs/diagnostics/inertia_live_01 \
  --output outputs/diagnostics/balanced_replay_01
```

Replay holds recorded motion, observer residuals, baseline commands, and operator
enable transitions fixed. It verifies command bounds and reports per-joint
output, task gains, and internal torque leakage; it is not a closed-loop
prediction of new arm motion. The same target/handle-offset CLI options apply.

Assistance stays enabled until toggled off, a protection disables it, or the
session ends. Live assistance has no cumulative energy cutoff. Added positive
work is logged for analysis; it does not disable assistance. Each periodic
status line reports `ASSISTANCE ON` or `ASSISTANCE OFF`, along with the output
reason and extra current. ON means enabled, not necessarily nonzero output. Timing or
state faults in the assistance limiter lock it off for the rest of the session
and retain gravity compensation. The loop targets 250 Hz; measured windows
below 200 Hz also lock assistance off. Hardware faults, communication failure,
nonfinite states, joint-limit approach, speed over the hard cutoff (default 4 rad/s, matching `gravity_comp.py`), or stalls over
100 ms exit with torque-off cleanup. A 200 ms motor watchdog is configured.
These are experimental software precautions, not a stability guarantee.

Live assistance begins fading at **2 rad/s** and switches assistance OFF at
**3 rad/s**, retaining gravity compensation. After slowing down, another explicit
`a` enables assistance again. This is separate
from the **4 rad/s** hard torque-off cutoff, which applies in either mode.
These are local engineering settings, not manufacturer ratings or validated
stability boundaries. Configure them with `--assist-fade-speed`,
`--assist-max-speed`, and `--max-joint-speed`; require
`0.035 < fade < assist cutoff <= hard cutoff <= 4` (rad/s). Startup prints the
values and metadata records them. The total-current envelope remains active in
both profiles; additional torque and instantaneous power caps apply in `bounded` mode.


The session ends automatically after the selected duration and disables torque;
it prints a warning ten seconds before timeout. Default output is
`outputs/diagnostics/inertia_live_01`: `metadata.json`, `samples.csv`, and
`report.json` are directly overwritten on a new session. Logs include actual
combined commands, baseline commands, assistance, current telemetry, reason
codes, and cumulative `sampled_assist_work_j` from successfully sent assistance.
Schema version 4 retains the replacement of the old `energy_remaining_j` CSV column with this
work counter; metadata records `energy_budget_enabled: false` and
`energy_budget_j: null`. The observer uses the previous
successfully sent combined command, including assistance. Its gravity scale is
an explicit 0.85 model assumption, not a physical calibration.

Every live sample also records `candidate_assist_nm_1`–`7` before the limiter,
then mode-specific gates, ramp and speed-fade factors,
and torque after each stage: torque cap, motion ramp, speed fade, slew limit,
direction gate, current headroom, power limit, and current quantization.
The historical `power_energy_effort_nm` column name is retained; live runs now
apply only the instantaneous power limit at that stage.
The `*_magnitude_nm` columns are unsigned; `*_effort_nm` columns use model joint
signs. For balanced mode, `motion_ramp` is the global activation ramp,
`direction_effort_nm` is the signed output with no direction gate, and the
per-joint moving/direction/dwell fields are not applicable (`NaN`).
`raw_task_gain` and `clipped_task_gain` entries 1–6 are task eigenmodes, not
joint indices; entry 7 is the zero internal gain. `residual_used_nm`,
`nullspace_input_nm`, `nullspace_output_nm`, rank, null dimension, and geometry
fade are also logged. `negative_power_w` records the nonpositive power of
resisting outputs. `NaN` otherwise means the stage was bypassed by an earlier gate, identified by
`reason`; it does not mean the motor was sent an invalid command. Warmup,
geometry validity, Jacobian rank, minimum joint-limit margin, and baseline
saturation are recorded separately. Invalid geometry zeros the candidate.
`assist_nm_1`–`7` remain the final sent extra torque, including any late fault
override after limiter evaluation. Existing recordings cannot recover
these intermediate values.

Software tests cover startup failure cleanup, explicit ON/OFF toggling, persistence beyond three seconds,
gravity-only fallback, hard fault shutdown, current bounds, and observer command
alignment, handle-offset finite differences, signed balanced output, full-rank
target construction, and natural internal inertia. The balanced controller has
not yet been validated on the physical arm.

## Bounded inertia-assistance development

`inertia_bounded.py` adds a seven-joint assistance limiter plus offline replay
and synthetic closed-loop tests. The offline tool has no hardware launch mode (use the separate experimental
`factr_inertia.py` adapter above for hardware). It does
not open a serial port or modify calibration, gains, or motor registers.

Run the software validation:

```bash
.venv/bin/python inertia_bounded.py validate
```

Replay an existing recording through the proposed observer and limiter:

```bash
.venv/bin/python inertia_bounded.py replay \
  outputs/diagnostics/inertia_baseline_02.csv \
  --output outputs/diagnostics/inertia_bounded_02
```

Outputs are directly overwritten within the selected output directory. The
validation command writes twelve runs (six synthetic conditions, assistance
off/on). The replay writes `bounded.csv`, `bounded_report.json`, and the
unbounded proposal in its `unbounded` subdirectory. Log replay preserves the
recorded motion; it cannot predict what the arm would do with added torque.
Synthetic tests feed the baseline plus assistance back into the observer, but
use independent joint inertias/friction and are not an identified FACTR model.

The controller requires explicit operator enable. A joint receives no assist
while stationary, during direction changes or during an 80 ms motion dwell.
Assistance ramps in over 250 ms and only adds effort in the direction of motion.
The experimental maximum added efforts (Nm) are
`[0.01, 0.02, 0.01, 0.02, 0.005, 0.005, 0.003]`; rise rate is 0.04 Nm/s.
Combined sampled positive power is capped at 0.01 W. The original offline replay
and synthetic fixtures retain a 0.02 J cumulative work budget; the live runner
explicitly disables that budget and continues logging cumulative work. It allocates only
current headroom left by the gravity baseline, rounding added current toward
zero. These are trial software limits, not validated physical ratings or a
continuous-time passivity guarantee.

The original offline synthetic fixture retains its deliberately low speed
settings: motion above 0.20 rad/s reduces assistance; any joint reaching 0.35 rad/s,
invalid state, or an update gap over 10 ms latches assistance off until a new
controller is explicitly created. Geometry/observer gates and operator disable
also remove assistance. The ideal full-matrix proposal retains natural modeled
nullspace inertia, but per-joint motion gates/limits alter that exact property.
This is bounded assistance research, not a claim of exact Cartesian inertia.
The live adapter uses the independently configurable speed settings described
above; the offline fixture defaults are not the live hardware cutoffs.

The current log has approximately 23–26 ms once-per-second gaps. Exact replay
therefore disables assistance at the first gap, before reaching motion; this
is an expected rejection, not a successful hardware-assistance result.
The gravity controller's status reads have now been staggered (one register
every 30 ms rather than 21 together), covering each register approximately every
0.63 seconds. Hardware error, watchdog and temperature checks remain; their
new timing has unit coverage but has not been measured on hardware yet.

Sticking means no joint motion despite an applied load. Breakaway/release means
motion begins; this is not the operator releasing the handle. Static friction,
kinetic friction and inertia have different effects. This controller deliberately
cannot help initiate motion against static friction without an observable intent
signal. Do not increase observer gain to manufacture one. The recorded residual
also contains gravity/model error and cannot be treated as measured hand torque.
