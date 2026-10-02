"""Guided, bounded joint-4 breakaway-current identification. No automatic calibration writes.

Only model-signed current thresholds are identified directly. Their midpoint is
a gravity-current estimate only under quasi-static, symmetric-friction and
zero-external-load assumptions. Requires an operator and a close, unloaded catch.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from factr_diagnostics import DEFAULT_PORT, load_calibration

J = 3


@dataclass(frozen=True)
class Limits:
    period: float = .02
    stall_s: float = .10
    max_speed: float = .30
    max_travel: float = np.deg2rad(2)
    other_drift: float = np.deg2rad(2)
    start_tolerance: float = np.deg2rad(1)
    step_raw: int = 2          # 5.38 mA on joint 4
    max_offset_raw: int = 45  # 121.05 mA either side of the holding command
    dwell_s: float = .5
    max_current_ma: float = 350


def check_motion(q, v, anchor, model, limits):
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise RuntimeError('Nonfinite state')
    if np.max(np.abs(v)) > limits.max_speed:
        raise RuntimeError('Speed bound exceeded; trial rejected')
    if np.any(q < model.lowerPositionLimit + .02) or np.any(q > model.upperPositionLimit - .02):
        raise RuntimeError('Near a modeled joint limit; trial rejected')
    error = np.abs(q - anchor)
    if error[J] > limits.max_travel:
        raise RuntimeError('Joint 4 moved more than 2 degrees; trial rejected')
    if np.max(np.delete(error, J)) > limits.other_drift:
        raise RuntimeError('Another joint moved more than 2 degrees; trial rejected')


def support_command(gravity, q, v, target, caps):
    kp = np.array([1., 2., 1., 3., .4, .4, .2])
    kd = np.full(7, .08)
    tau = .85*gravity + kp*(target-q) - kd*v
    if np.any(np.abs(tau*gc.TORQUE_TO_RAW) > caps):
        raise RuntimeError('Pose support would saturate a motor; choose a different pose')
    return gc.current_command(tau, caps)


class Onset:
    def __init__(self, q_start, direction):
        self.start = float(q_start)
        self.direction = direction
        self.count = 0

    def update(self, q, v):
        # Require both cumulative displacement and sustained velocity. Encoder
        # resolution is ~0.088 deg; a single tick must not be a breakaway event.
        displacement = self.direction * (q - self.start)
        if displacement < -np.deg2rad(.4):
            raise RuntimeError('Motion opposite the sweep; no valid stationary bracket')
        moving = displacement >= np.deg2rad(.25) and self.direction * v >= .025
        self.count = self.count + 1 if moving else 0
        return self.count >= 3


def pair_thresholds(positive, negative):
    if positive['status'] != 'onset' or negative['status'] != 'onset':
        raise ValueError('Both directions need measured onset; censored trials cannot be fitted')
    if positive['direction'] != 1 or negative['direction'] != -1:
        raise ValueError('Wrong threshold directions')
    qpos, qneg = np.array(positive['q_start']), np.array(negative['q_start'])
    if np.max(np.abs(qpos-qneg)) > np.deg2rad(1):
        raise ValueError('Opposite sweeps began more than 1 degree apart')
    plo, phi = sorted(positive['bracket_model_raw'])
    nlo, nhi = sorted(negative['bracket_model_raw'])
    if plo <= nhi:
        raise ValueError('Thresholds overlap or are reversed')
    midpoint = (plo+phi+nlo+nhi)/4
    return {
        'pose': positive['pose'], 'repeat': positive['repeat'],
        'q_rad': ((qpos+qneg)/2).tolist(),
        'gravity_model_signed_ma_estimate': midpoint * gc.MA_PER_RAW[J],
        'gravity_model_signed_ma_interval': [(.5*(plo+nlo))*gc.MA_PER_RAW[J],
                                             (.5*(phi+nhi))*gc.MA_PER_RAW[J]],
        'friction_half_width_ma_estimate': (plo+phi-nlo-nhi)/4 * gc.MA_PER_RAW[J],
        'model_gravity_nm': .5*(positive['model_gravity_nm']+negative['model_gravity_nm']),
        'note': 'Interval is sweep resolution only, NOT total uncertainty; friction asymmetry and hand force bias it.',
    }


def summarize(results):
    pairs, rejected = [], []
    keys = sorted(set((r['pose'], r['repeat']) for r in results))
    for key in keys:
        group = {r['direction']: r for r in results if (r['pose'], r['repeat']) == key}
        try:
            if set(group) != {-1, 1}:
                raise ValueError('Missing direction')
            pairs.append(pair_thresholds(group[1], group[-1]))
        except ValueError as exc:
            rejected.append({'pose': key[0], 'repeat': key[1], 'reason': str(exc)})
    report = {'accepted_pairs': pairs, 'rejected_pairs': rejected,
              'applied_to_controller': False,
              'interpretation': 'Raw-current midpoint under symmetric quasi-static friction; not an independent Nm measurement.'}
    if len(pairs) >= 6 and len(set(p['pose'] for p in pairs)) >= 3:
        x = np.array([p['model_gravity_nm'] for p in pairs])
        y = np.array([p['gravity_model_signed_ma_estimate'] for p in pairs])
        if np.ptp(x) >= .04:
            A = np.column_stack([x, np.ones(len(x))])
            slope, bias = np.linalg.lstsq(A, y, rcond=None)[0]
            residual = y-A@np.array([slope,bias])
            report['candidate_current_fit_NOT_APPLIED'] = {
                'slope_ma_per_model_nm': float(slope), 'bias_ma': float(bias),
                'rms_residual_ma': float(np.sqrt(np.mean(residual**2))),
                'model_gravity_nm_range': [float(x.min()), float(x.max())],
                'note': 'Review repeatability, contact, dynamic creep and friction symmetry before use. Do not extrapolate.',
            }
    return report


def hardware_check(bus, motor, enabled):
    if bus.read(motor, 70):
        raise RuntimeError(f'ID {motor}: hardware fault')
    if bus.read(motor, 146) >= 60:
        raise RuntimeError(f'ID {motor}: temperature >=60 C')
    if enabled and (bus.read(motor, 64) != 1 or bus.read(motor, 98) != 10):
        raise RuntimeError(f'ID {motor}: torque/watchdog state changed')


def enable(bus):
    # Caller owns unconditional stop on partial failure.
    for motor in gc.IDS:
        hardware_check(bus, motor, False)
        if bus.read(motor, 6) < 38:
            raise RuntimeError('Firmware does not support watchdog')
        bus.write(motor, 98, 0)
        if bus.read(motor, 11) != 0:
            bus.write(motor, 11, 0)
        if bus.read(motor, 11) != 0:
            raise RuntimeError('Current mode verification failed')
        bus.write(motor, 102, 0, 2)
    for motor in gc.IDS:
        bus.write(motor, 64, 1)
        bus.write(motor, 98, 10)
        hardware_check(bus, motor, True)


def trial(bus, offsets, hw_limits, target, pose, repeat, direction, writer, limits=Limits()):
    model, data = gc.model()
    caps = np.minimum(hw_limits, np.floor(limits.max_current_ma/gc.MA_PER_RAW))
    # Soft, bounded pose support. These terms are REMOVED from J4 during the
    # sweep; the tested joint receives only the specified current.
    kd = np.full(7, .08)
    anchor = target.copy()
    phase, phase_start = 'supported_ramp', time.monotonic()
    start = previous = phase_start
    stable_since = None
    result = None
    baseline = last_still = level = None
    q_start = None
    last_status = phase_start
    status_motor = 0
    previous_command = np.zeros(7, dtype=int)
    print('POWER ON: support arm for 2 s. Then unload your hand, with a close catch ready. '
          'If it cannot hold, catch it and press Ctrl-C.', flush=True)
    try:
        enable(bus)
        start = previous = phase_start = time.monotonic()
        while True:
            tick = time.monotonic()
            if tick-previous > limits.stall_s:
                raise RuntimeError('Controller delayed more than 100 ms')
            previous = tick
            raw, raw_v = bus.state()
            if time.monotonic()-tick > limits.stall_s:
                raise RuntimeError('Stale telemetry')
            q, v = gc.angles(raw, offsets), gc.SIGNS*raw_v
            measured = bus.present_current_ma()
            check_motion(q, v, anchor, model, limits)
            gravity = np.array(pin.computeGeneralizedGravity(model, data, q))
            # During a sweep, exclude J4's unused PD term from saturation tests.
            # J4 is tested against its own current bound below.
            support_target, support_v = target.copy(), v.copy()
            if phase in ('baseline', 'sweep'):
                support_target[J] = q[J]
                support_v[J] = 0
            command = support_command(gravity, q, support_v, support_target, caps)
            elapsed = tick-phase_start
            if phase == 'supported_ramp':
                command = np.rint(command*min(elapsed/2, 1)).astype(int)
                if elapsed >= 2:
                    phase, phase_start = 'unload_hand', tick
                    print('Unload your hand now; keep the catch close without touching. Settling 4 seconds.', flush=True)
            elif phase == 'unload_hand':
                if elapsed >= 4:
                    phase, phase_start = 'settle', tick
                    print('Checking stationary state...', flush=True)
            elif phase == 'settle':
                still = np.max(abs(v)) < .025 and np.max(abs(q-anchor)) < limits.start_tolerance
                stable_since = (tick if stable_since is None else stable_since) if still else None
                if stable_since is not None and tick-stable_since >= 1:
                    q_start = q.copy()
                    anchor = q.copy()
                    baseline = int(gc.SIGNS[J]*command[J])
                    level = last_still = baseline
                    detector = Onset(q[J], direction)
                    phase, phase_start = 'baseline', tick
                    print(f'Sweep {direction:+d} about model coordinates; freeze J4 at {baseline*gc.MA_PER_RAW[J]:.1f} model-signed mA.', flush=True)
                elif elapsed > 5:
                    raise RuntimeError('Could not establish stationary unloaded baseline')
            elif phase in ('baseline', 'sweep'):
                command[J] = int(gc.SIGNS[J]*level)
                if abs(level) > caps[J]:
                    raise RuntimeError('Current bound reached; no threshold estimate')
                moving = detector.update(q[J], v[J])
                if moving:
                    if phase == 'baseline':
                        raise RuntimeError('Baseline itself moves; no stationary bracket')
                    result = {'status': 'onset', 'pose': pose, 'repeat': repeat, 'direction': direction,
                              'q_start': q_start.tolist(), 'q_onset': q.tolist(),
                              'bracket_model_raw': sorted([last_still, level]),
                              'baseline_model_raw': baseline,
                              'model_gravity_nm': float(pin.computeGeneralizedGravity(model, data, q_start)[J]),
                              'measured_electrical_current_ma_at_onset': float(measured[J])}
                    # Catch at the detected pose, not a sudden return to the start.
                    target = q.copy()
                    phase, phase_start = 'catch', tick
                    command = gc.current_command(.85*gravity-kd*v, caps)
                    print('ONSET FOUND. Support the arm NOW; torque turns off in 3 seconds.', flush=True)
                elif elapsed >= limits.dwell_s:
                    if np.max(abs(v)) >= .025 or np.max(abs(q-q_start)) > np.deg2rad(.2):
                        raise RuntimeError('Creep or coupled motion: cannot mark this current as stationary')
                    last_still = level
                    next_level = level + direction*limits.step_raw
                    if abs(next_level-baseline) > limits.max_offset_raw or abs(next_level) > caps[J]:
                        result = {'status': 'censored', 'pose': pose, 'repeat': repeat, 'direction': direction,
                                  'reason': 'No onset inside current bounds', 'q_start': q_start.tolist()}
                        phase, phase_start = 'catch', tick
                        target = q.copy()
                        command = gc.current_command(.85*gravity-kd*v, caps)
                        print('NO THRESHOLD within bounds. Support NOW; torque off in 3 seconds.', flush=True)
                    else:
                        level = next_level
                        phase, phase_start = 'sweep', tick
                        command[J] = int(gc.SIGNS[J]*level)
            elif phase == 'catch' and elapsed >= 3:
                break
            bus.currents(command)
            # Current telemetry belongs to the preceding command, not this one.
            writer.writerow([time.monotonic(), pose, repeat, direction, phase, *q, *v,
                             *gravity, *previous_command, *command, *measured])
            previous_command = command.copy()
            if tick-last_status >= .1:
                hardware_check(bus, gc.IDS[status_motor], True)
                status_motor = (status_motor+1) % 7
                last_status = tick
            time.sleep(max(0, limits.period-(time.monotonic()-tick)))
        return result
    finally:
        bus.stop()


def prepare_output(directory):
    """Reuse the requested run directory, replacing this tool's old files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = [directory/name for name in ('metadata.json', 'samples.csv', 'trials.jsonl', 'report.json')]
    existing = [path for path in files if path.exists()]
    if any(not path.is_file() for path in existing):
        raise ValueError('An identification output filename is occupied by a directory')
    for path in existing:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=DEFAULT_PORT)
    parser.add_argument('--calibration', type=Path, default=gc.ROOT/'calibration.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='Run directory; previous identification files are overwritten automatically')
    parser.add_argument('--analyze-only', action='store_true', help='Analyze existing trials.jsonl; no hardware')
    args = parser.parse_args()
    if args.analyze_only:
        rows = [json.loads(line) for line in (args.output/'trials.jsonl').read_text().splitlines()]
        print(json.dumps(summarize(rows), indent=2))
        return
    saved, offsets = load_calibration(args.calibration, args.port)
    prepare_output(args.output)
    (args.output/'metadata.json').write_text(json.dumps({
        'calibration': saved, 'joint': 4, 'limits': Limits().__dict__,
        'urdf_sha256': hashlib.sha256((gc.ROOT/'models/factr_teleop_franka.urdf').read_bytes()).hexdigest(),
        'assumptions': ['no hand/contact load during sweep', 'quasi-static onset', 'approximately symmetric friction'],
        'active_mode': 'bounded current sweep on J4, soft position support on others',
        'calibration_writes': False,
    }, indent=2))
    bus = None
    results = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print('Use a secured base and a close catch under the downstream assembly. '
          'No contact with the catch during a valid measurement. Ctrl-C disables torque. '
          'Abort/fault also disables torque: be ready to catch. No other controller may run.', flush=True)
    try:
        bus = gc.Bus(args.port, 4000000)
        hw_limits = bus.validate()
        latency = Path('/sys/bus/usb-serial/devices')/Path(args.port).resolve().name/'latency_timer'
        if not latency.exists() or latency.read_text().strip() != '1':
            raise RuntimeError('Set USB latency_timer to 1 before identification')
        with (args.output/'samples.csv').open('x', newline='', buffering=1) as log, \
             (args.output/'trials.jsonl').open('x', buffering=1) as outcomes:
            writer = csv.writer(log)
            header = ['monotonic_s', 'pose', 'repeat', 'direction', 'phase']
            for label in ('q_rad', 'dq_rad_s', 'model_gravity_nm', 'previous_command_raw', 'command_raw', 'measured_current_ma'):
                header.extend(f'{label}_{i}' for i in gc.IDS)
            writer.writerow(header)
            for pose in range(1, 4):
                target = None
                for repeat in range(1, 3):
                    # Alternate order to expose hysteresis/order effects.
                    for direction in ((1, -1) if repeat == 1 else (-1, 1)):
                        if target is None:
                            instruction = f'Choose comfortable pose {pose}/3 with a different elbow bend (about 15–25 deg apart).'
                        else:
                            instruction = 'Return to this pose in degrees: '+str(np.round(np.degrees(target), 1))
                        print('\nTORQUE OFF. '+instruction, flush=True)
                        while True:
                            # Confirm torque-off BEFORE waiting for terminal input.
                            bus.validate()
                            response = input('Support arm, prepare catch. Type START to enable this one trial, or q to finish: ')
                            if response.strip().lower() == 'q':
                                return
                            if response.strip() != 'START':
                                print('No motors enabled. Enter START or q.', flush=True)
                                continue
                            bus.validate()
                            raw, raw_v = bus.state()
                            q = gc.angles(raw, offsets)
                            if target is None:
                                target = q.copy()
                            if np.max(abs(q-target)) <= np.deg2rad(1):
                                break
                            print('Still torque OFF. Pose errors in degrees:', np.round(np.degrees(q-target), 1), flush=True)
                        model, data = gc.model()
                        check_motion(q, gc.SIGNS*raw_v, target, model, Limits())
                        if np.max(abs(raw_v)) > .025:
                            raise RuntimeError('Arm moving before start. No motors enabled.')
                        try:
                            result = trial(bus, offsets, hw_limits, target, pose, repeat, direction, writer)
                        except BaseException as exc:
                            result = {'status': 'rejected', 'pose': pose, 'repeat': repeat,
                                      'direction': direction, 'reason': str(exc) or type(exc).__name__}
                            results.append(result)
                            outcomes.write(json.dumps(result)+'\n')
                            raise
                        log.flush()
                        results.append(result)
                        outcomes.write(json.dumps(result)+'\n')
                        print('Saved trial:', result['status'], flush=True)
                        (args.output/'report.json').write_text(json.dumps(summarize(results), indent=2))
    finally:
        if bus is not None:
            # trial() covers powered shutdown; outside a trial no torque was enabled.
            bus.port.closePort()
        (args.output/'report.json').write_text(json.dumps(summarize(results), indent=2))
        print('Saved', args.output/'report.json', flush=True)


if __name__ == '__main__':
    try:
        import sys
        if '--thresholds' in sys.argv:
            sys.argv.remove('--thresholds')
            main()
        else:
            from gravity_capture import main as hold_capture_main
            hold_capture_main()
    except KeyboardInterrupt:
        print('Interrupted. Torque-off cleanup attempted; partial data retained.')
    except RuntimeError as exc:
        print(f'Stopped: {exc}', file=sys.stderr)
        sys.exit(1)
