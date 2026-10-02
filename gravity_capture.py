"""FACTR hold/capture/drag calibration. Captures are provisional; never auto-applied."""
import argparse
import csv
import json
from pathlib import Path
import select
import signal
import sys
import time

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from factr_diagnostics import DEFAULT_PORT, load_calibration
from identify_gravity import enable, hardware_check

KP = np.array([1., 2., 1., 3., .4, .4, .2])
# Conservative starting adjustment for the observed J3 overshoot; hardware
# validation is still required. Do not increase stiffness to force tracking.
KD = np.array([.08, .08, .12, .08, .08, .08, .08])
AMP = .05  # B601 calibration amplitude (radians)
PERIOD = 8.
CYCLES = 2
SETTLE = 3.
CAPTURE_SPEED_LIMIT = .20
CONTROL_HZ = 250.  # Headroom above upstream PD minimum of 200 Hz.
METHOD = 'b601_pd_residual_v1'


def trajectory(elapsed, anchor, joint=None):
    """FACTR motion adapter: smooth symmetric cycles, B601 residual estimator.

    Each quarter-cycle uses a quintic move with zero velocity/acceleration
    at both ends: center -> +AMP -> center -> -AMP -> center. There is no
    position step at startup or velocity jump at a reversal.
    """
    target, velocity = anchor.copy(), np.zeros(7)
    if elapsed < SETTLE:
        return target, velocity, False, False
    if elapsed >= SETTLE + PERIOD*CYCLES:
        return target, velocity, True, False
    quarter = ((elapsed-SETTLE) % PERIOD)/(PERIOD/4)
    segment = min(int(quarter), 3)
    u = quarter-segment
    blend = 10*u**3-15*u**4+6*u**5
    rate = (30*u**2-60*u**3+30*u**4)/(PERIOD/4)
    points = (0., AMP, 0., -AMP, 0.)
    delta = points[segment+1]-points[segment]
    indices = slice(None) if joint is None else joint
    target[indices] += points[segment]+delta*blend
    velocity[indices] = delta*rate
    return target, velocity, False, True


def summarize_capture(samples, anchor, joint=None):
    """B601 estimator: mean commanded PD residual across complete cycles.

    Measured current and measured velocity are diagnostics, not an alternative
    torque estimator or a matched-angle acceptance filter.
    """
    record = {'method': METHOD, 'joint': None if joint is None else joint+1,
              'anchor_q': anchor.tolist(), 'samples': len(samples)}
    if not samples:
        return dict(record, status='rejected', reason='No complete capture samples')
    residual = np.mean([s['resid'] for s in samples], axis=0)
    last = samples[-1]
    record.update(q=last['q'], g_raw=last['g_raw'], ff=last['ff'], resid=residual.tolist())
    # The session replaces these with the completion state, matching B601.
    q = np.array([s['q'] for s in samples])
    v = np.array([s['v'] for s in samples])
    record['motion_span_deg'] = np.rad2deg(np.ptp(q, axis=0)).tolist()
    record['zero_velocity_fraction'] = np.mean(abs(v) < 1e-6, axis=0).tolist()
    record['warnings'] = [f'J{i+1}: little measured motion; static friction may bias residual'
                          for i in range(7) if np.ptp(q[:, i]) < AMP]
    if any(s['saturated'] for s in samples):
        return dict(record, status='rejected', reason='Current clipping: requested PD effort was not delivered')
    return dict(record, status='recorded', reason='B601 average PD residual saved; fit is provisional')


def fit_captures(captures, bias=False):
    report = {'method': METHOD, 'captures': captures, 'applied': False,
              'fit_bias': bias, 'candidate_fits': {}, 'skipped_legacy_captures': 0}
    selected = []
    for c in captures:
        if c.get('method') != METHOD:
            report['skipped_legacy_captures'] += 1
        elif c['status'] == 'recorded':
            selected.append(c)
    if not selected:
        return report
    for joint in range(7):
        x = np.array([c['g_raw'][joint] for c in selected])
        y = np.array([c['ff'][joint]+c['resid'][joint] for c in selected])
        A = np.column_stack([x, np.ones(len(x))]) if bias else x[:, None]
        if not np.isfinite(A).all() or not np.isfinite(y).all():
            continue
        if np.max(abs(x)) < 1e-6 or np.linalg.matrix_rank(A) < A.shape[1]:
            continue
        solution = np.linalg.lstsq(A, y, rcond=None)[0]
        report['candidate_fits'][str(joint+1)] = {
            'scale': float(solution[0]),
            'bias_equivalent_nm': float(solution[1]) if bias else 0.,
            'rms_equivalent_nm': float(np.sqrt(np.mean((A@solution-y)**2))),
            'captures': len(selected), 'g_range_nm': [float(x.min()), float(x.max())],
            'note': 'Provisional B601 ff + mean(PD) fit, not applied. Use several spread poses; '
                    'small gravity signals, poor motion and contact can make fits unreliable.'}
    return report


def prepare_output(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ('metadata.json', 'samples.csv', 'captures.jsonl', 'report.json', 'fault.json', 'calib.csv'):
        destination = path/name
        if destination.exists() and not destination.is_file():
            raise ValueError(f'{destination} is not a file')
    for name in ('metadata.json', 'samples.csv', 'captures.jsonl', 'report.json', 'fault.json', 'calib.csv'):
        (path/name).unlink(missing_ok=True)


def check_bus_timing(bus):
    """Read-only preflight; caller validates torque OFF before calling."""
    delays = [bus.read(motor, 9) for motor in gc.IDS]
    if any(delays):
        raise RuntimeError(f'Return Delay must be 0 on every motor (FACTR setup); found {delays}. '
                           'Set register 9 to 0 with torque off before retrying.')
    durations = []
    for _ in range(100):
        started = time.monotonic()
        bus.state()
        durations.append(time.monotonic()-started)
    median, p95 = np.percentile(durations, [50, 95])
    print(f'Torque-OFF bus preflight: median {median*1000:.2f} ms, '
          f'p95 {p95*1000:.2f} ms per state read.', flush=True)
    if np.mean(durations) >= .004:
        raise RuntimeError('State reads leave insufficient headroom for >=200 Hz PD; torque remains off')
    return {'state_read_median_ms': float(median*1000), 'state_read_p95_ms': float(p95*1000)}


def capture_current_caps(hardware_limits, joint4_ma=350.):
    if not np.isfinite(joint4_ma) or not 1 <= joint4_ma <= 900:
        raise ValueError('J4 current cap must be finite and within 1..900 mA')
    requested_ma = np.full(7, 350.)
    requested_ma[3] = joint4_ma
    return np.minimum(hardware_limits, np.floor(requested_ma/gc.MA_PER_RAW))


def check_control_rate(intervals, elapsed):
    if elapsed >= 1.:
        hz = intervals/elapsed
        if hz < 200:
            raise RuntimeError(f'PD loop achieved only {hz:.1f} Hz; requires >=200 Hz; torque-off cleanup')
        return hz
    return None


def bounded_command(gravity, q, v, target, desired_v, phase, fraction, caps):
    if phase == 'DRAG':
        requested = .85*gravity-.01*v
    elif phase == 'RELEASE':
        # Keep braking active while gradually removing weight support.
        requested = fraction*(.85*gravity+KP*(target-q))-KD*v
    else:
        requested = .85*gravity+KP*(target-q)+KD*(desired_v-v)
    if phase != 'RELEASE':
        requested *= fraction
    saturated = bool(np.any(abs(requested*gc.TORQUE_TO_RAW) > caps))
    return gc.current_command(requested, caps), saturated


def run_session(bus, offsets, caps, joint, writer, outcomes, captures, report_path, bias=False):
    model, data = gc.model()
    raw, raw_v = bus.state()
    q = gc.angles(raw, offsets)
    if np.max(abs(raw_v)) > .025:
        raise RuntimeError('Start with the supported arm stationary')
    if np.any(q < model.lowerPositionLimit+.15) or np.any(q > model.upperPositionLimit-.15):
        raise RuntimeError('Starting pose is too close to a modeled joint limit')
    target = q.copy()
    phase = 'RAMP'
    started = previous = phase_start = time.monotonic()
    prior_command = np.zeros(7, dtype=int)
    samples = []
    anchor = target.copy()
    capture_started = None
    saturation_start = None
    last_status, status_id = started, 0
    interrupts = []
    fault_state = {}
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: interrupts.append(True))

    def save_capture(reason=None):
        nonlocal samples
        if reason is None:
            result = summarize_capture(samples, anchor, joint)
        else:
            result = {'method': METHOD, 'joint': None if joint is None else joint+1, 'anchor_q': anchor.tolist(), 'status': 'rejected',
                      'reason': reason, 'samples': len(samples)}
        if result['status'] == 'recorded':
            # Match B601: final state/feedforward, period-averaged PD residual.
            result.update(q=q.tolist(), g_raw=gravity.tolist(), ff=(.85*gravity).tolist())
            csv_path = report_path.parent/'calib.csv'
            new = not csv_path.exists()
            with csv_path.open('a', newline='') as stream:
                csv_writer = csv.writer(stream)
                if new:
                    csv_writer.writerow([f'{name}{i}' for name in ('q', 'graw', 'ff', 'resid') for i in gc.IDS])
                csv_writer.writerow([value for key in ('q', 'g_raw', 'ff', 'resid') for value in result[key]])
            print('Mean PD residual (Nm):', np.round(result['resid'], 4), flush=True)
            for warning in result['warnings']:
                print(warning, flush=True)
        captures.append(result)
        outcomes.write(json.dumps(result)+'\n')
        outcomes.flush()
        report_path.write_text(json.dumps(fit_captures(captures, bias), indent=2))
        print('Capture saved:', result['status'], result.get('reason', ''), flush=True)
        samples = []

    try:
        enable(bus)
        started = previous = phase_start = time.monotonic()
        rate_started, rate_count = started, 0
        print(f'PD target rate: {CONTROL_HZ:.0f} Hz; minimum measured rate: 200 Hz.', flush=True)
        print('RAMP: support arm during the 2-second ramp. Then HOLD stays active.', flush=True)
        while True:
            tick = time.monotonic()
            if tick-previous > .1:
                raise RuntimeError('Loop delay >100 ms')
            rate_count += 1
            measured_hz = check_control_rate(rate_count, tick-rate_started)
            if measured_hz is not None:
                rate_started, rate_count = tick, 0
            previous = tick
            raw, raw_v = bus.state()
            if time.monotonic()-tick > .1:
                raise RuntimeError('Stale state read')
            q, v = gc.angles(raw, offsets), gc.SIGNS*raw_v
            measured = bus.present_current_ma()
            fault_state = {'phase': phase, 'time_s': tick-started,
                           'q_rad': q.tolist(), 'velocity_rad_s': v.tolist(),
                           'previous_command_raw': prior_command.tolist(),
                           'current_cap_raw': caps.tolist()}
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise RuntimeError('Nonfinite state')
            if np.max(abs(v)) > .5:
                fastest = int(np.argmax(abs(v)))
                raise RuntimeError(f'J{fastest+1} speed {v[fastest]:+.3f} rad/s exceeds 0.5; torque-off cleanup')
            if np.any(q < model.lowerPositionLimit+.02) or np.any(q > model.upperPositionLimit-.02):
                raise RuntimeError('Joint limit margin reached')
            gravity = np.array(pin.computeGeneralizedGravity(model, data, q))
            if phase == 'CAPTURE' and np.max(abs(v)) > CAPTURE_SPEED_LIMIT:
                fastest = int(np.argmax(abs(v)))
                save_capture(f'J{fastest+1} speed {v[fastest]:+.3f} rad/s exceeded '
                             f'capture limit {CAPTURE_SPEED_LIMIT}; capture cancelled')
                phase, phase_start, target = 'HOLD', tick, q.copy()
                print('CAPTURE CANCELLED: excess speed; HOLD at current pose. '
                      'Support arm, then r to finish. Capture not fitted.', flush=True)
            if phase == 'RELEASE' and (np.max(abs(v)) > .12 or np.max(abs(q-target)) > np.deg2rad(2)):
                fastest = int(np.argmax(abs(v)))
                phase, phase_start, target = 'HOLD', tick, q.copy()
                print(f'RELEASE PAUSED: motion detected (J{fastest+1}: {v[fastest]:+.3f} rad/s). '
                      'HOLD restored. Support the arm fully, then type r again.', flush=True)
            # Critical overload: bounded currents cannot hold this pose; catch is required.
            if phase != 'DRAG' and np.max(abs(q-target)) > np.deg2rad(8):
                raise RuntimeError('Unable to hold pose within 8 degrees; support/catch arm')
            command_text = None
            if interrupts:
                interrupts.clear()
                command_text = 'r' if phase == 'HOLD' else ('q!' if phase == 'RELEASE' else 'h')
            elif select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    raise RuntimeError('Terminal closed')
                command_text = line.strip().lower()
            if command_text:
                if command_text in ('q!', 'q'):
                    if phase == 'CAPTURE':
                        save_capture('Emergency exit')
                    break
                if command_text in ('h', 'hold'):
                    if phase == 'CAPTURE':
                        save_capture('Interrupted by operator')
                    phase, phase_start, target = 'HOLD', tick, q.copy()
                    print('HOLD: c=capture, d=drag, r=release after supporting.', flush=True)
                elif command_text in ('d', 'drag') and phase == 'HOLD':
                    phase = 'DRAG'
                    print('DRAG: support and slowly reposition; h to hold, c to capture.', flush=True)
                elif command_text in ('c', 'capture') and phase in ('HOLD', 'DRAG'):
                    if np.max(abs(v)) > .05:
                        print('Stop moving before capture.', flush=True)
                    elif np.any(q-AMP < model.lowerPositionLimit+.1) or np.any(q+AMP > model.upperPositionLimit-.1):
                        print('Capture too near joint limit; choose another pose.', flush=True)
                    else:
                        phase, phase_start = 'CAPTURE', tick
                        anchor = target = q.copy()
                        capture_started = tick
                        samples = []
                        saturation_start = None
                        label = 'all 7 joints' if joint is None else f'J{joint+1}'
                        print(f'CAPTURE {label}: unload hand during 3 s settling, then smooth +/-2.86 deg, '
                              'two 8 s cycles. Returns to DRAG: be ready to support/reposition.', flush=True)
                elif command_text in ('r', 'release') and phase in ('HOLD', 'DRAG', 'CAPTURE'):
                    if phase == 'CAPTURE':
                        save_capture('Release requested')
                    phase, phase_start, target = 'RELEASE', tick, q.copy()
                    print('RELEASE: support arm NOW; torque fades off over 3 seconds.', flush=True)
            desired_v = np.zeros(7)
            fraction = 1.
            steady = False
            if phase == 'RAMP':
                fraction = min((tick-phase_start)/2, 1)
                if fraction >= 1:
                    phase = 'HOLD'
                    print('HOLD active. c=capture, d=drag, h=hold, r=release, q!=immediate torque off.', flush=True)
            elif phase == 'RELEASE':
                fraction = max(0, 1-(tick-phase_start)/3)
                if fraction == 0:
                    break
            elif phase == 'CAPTURE':
                target, desired_v, done, steady = trajectory(tick-capture_started, anchor, joint)
                if np.max(abs(q-anchor)) > np.deg2rad(5):
                    save_capture('Motion exceeded 5 degrees from capture pose')
                    phase, target, desired_v = 'HOLD', q.copy(), np.zeros(7)
                    steady = False
                elif done:
                    save_capture()
                    phase, target, desired_v = 'DRAG', q.copy(), np.zeros(7)
                    print('Back to DRAG. Support/reposition; c=capture, h=hold, r=finish.', flush=True)
            command, saturated = bounded_command(gravity, q, v, target, desired_v, phase, fraction, caps)
            if saturated:
                saturation_start = tick if saturation_start is None else saturation_start
                if phase == 'CAPTURE' and tick-saturation_start > .15:
                    requested = .85*gravity+KP*(target-q)+KD*(desired_v-v)
                    exceeded = np.flatnonzero(abs(requested*gc.TORQUE_TO_RAW) > caps)
                    detail = ', '.join(f'J{i+1}: requested {abs(requested[i]*gc.TORQUE_TO_RAW[i]*gc.MA_PER_RAW[i]):.1f} mA '
                                       f'vs cap {caps[i]*gc.MA_PER_RAW[i]:.1f} mA' for i in exceeded)
                    save_capture('Current limit prevented tracking; '+detail+'; capture not fitted')
                    phase, target, desired_v = 'HOLD', q.copy(), np.zeros(7)
                    command, saturated = bounded_command(gravity, q, v, target, desired_v, phase, 1, caps)
                    print('HOLD at present pose after saturation. Support arm if it cannot hold.', flush=True)
            else:
                saturation_start = None
            if phase == 'CAPTURE' and steady:
                samples.append({'q': q.tolist(), 'v': v.tolist(), 'g_raw': gravity.tolist(),
                                'ff': (.85*gravity).tolist(),
                                'resid': (KP*(target-q)+KD*(desired_v-v)).tolist(),
                                'saturated': saturated})
            bus.currents(command)
            writer.writerow([tick-started, phase, saturated, *q, *v, *target, *desired_v,
                             *gravity, *prior_command, *command, *measured])
            prior_command = command.copy()
            if tick-last_status >= .1:
                hardware_check(bus, gc.IDS[status_id], True)
                status_id = (status_id+1) % 7
                last_status = tick
            time.sleep(max(0, 1/CONTROL_HZ-(time.monotonic()-tick)))
    except BaseException as exc:
        fault_state['error'] = str(exc) or type(exc).__name__
        try:
            (report_path.parent/'fault.json').write_text(json.dumps(fault_state, indent=2))
        except OSError:
            pass
        if phase == 'CAPTURE':
            save_capture(str(exc) or type(exc).__name__)
        raise
    finally:
        bus.stop()
        signal.signal(signal.SIGINT, old_sigint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=gc.ROOT/'outputs/diagnostics/hold_capture_01')
    parser.add_argument('--joint', type=int, choices=range(1,8), default=None,
                        help='Optional single-joint capture; default moves all seven joints like B601')
    parser.add_argument('--port', default=DEFAULT_PORT)
    parser.add_argument('--calibration', type=Path, default=gc.ROOT/'calibration.json')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--joint4-max-current-ma', type=float, default=350.,
                        help='J4 software cap in mA (1..900), also bounded by the motor register; other joints stay at 350')
    parser.add_argument('--bias', action='store_true', help='Fit scale plus bias; default is B601 scale-only fit')
    args = parser.parse_args()
    try:
        capture_current_caps(np.full(7, 10000), args.joint4_max_current_ma)
    except ValueError as exc:
        parser.error(str(exc))
    if args.analyze_only:
        captures = [json.loads(line) for line in (args.output/'captures.jsonl').read_text().splitlines()]
        report = fit_captures(captures, args.bias)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    saved, offsets = load_calibration(args.calibration, args.port)
    bus = gc.Bus(args.port, 4000000)
    captures = []
    prepared = False
    try:
        limits = bus.validate()
        latency = Path('/sys/bus/usb-serial/devices')/Path(args.port).resolve().name/'latency_timer'
        if not latency.exists() or latency.read_text().strip() != '1':
            raise RuntimeError('Set USB latency_timer to 1')
        timing = check_bus_timing(bus)
        prepare_output(args.output)
        prepared = True
        (args.output/'metadata.json').write_text(json.dumps({
            'workflow': 'hold-capture-drag', 'method': METHOD, 'calibration': saved, 'joint': args.joint,
            'trajectory': 'quintic_quarters', 'capture_speed_limit_rad_s': CAPTURE_SPEED_LIMIT,
            'amplitude_rad': AMP, 'period_s': PERIOD, 'cycles': CYCLES, 'settle_s': SETTLE, 'fit_bias': args.bias,
            'kp_nm_per_rad': KP.tolist(), 'kd_nm_s_per_rad': KD.tolist(),
            'current_cap_ma': [350.,350.,350.,args.joint4_max_current_ma,350.,350.,350.],
            'bus_preflight': timing, 'control_hz': CONTROL_HZ, 'minimum_control_hz': 200, 'gain': .85,
            'notes': 'No friction compensation or inertia shaping. B601 average PD residual; host current PD, not motor-native PD. Fit is provisional.',
        }, indent=2))
        print('Secure base. Keep a close unloaded catch ready. Faults disable torque. '
              'Support arm for startup and release; do not touch moving links during capture.', flush=True)
        if input('Press Enter to enable HOLD, or type anything to cancel: ').strip():
            return
        bus.validate()
        caps = capture_current_caps(limits, args.joint4_max_current_ma)
        print('Effective current caps J1–J7 (mA):', np.round(caps*gc.MA_PER_RAW, 1), flush=True)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Terminated')))
        with (args.output/'samples.csv').open('w', newline='', buffering=1) as stream, \
             (args.output/'captures.jsonl').open('w', buffering=1) as outcomes:
            writer = csv.writer(stream)
            header = ['time_s', 'phase', 'saturated']
            for name in ('q_rad', 'dq_rad_s', 'target_rad', 'desired_velocity', 'model_gravity_nm',
                         'previous_command_raw', 'command_raw', 'measured_current_ma'):
                header.extend(f'{name}_{i}' for i in gc.IDS)
            writer.writerow(header)
            run_session(bus, offsets, caps, None if args.joint is None else args.joint-1, writer, outcomes, captures, args.output/'report.json', args.bias)
    finally:
        bus.port.closePort()
        if prepared:
            (args.output/'report.json').write_text(json.dumps(fit_captures(captures, args.bias), indent=2))


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(f'Stopped: {exc}', file=sys.stderr)
        sys.exit(1)
