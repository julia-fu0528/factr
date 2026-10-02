"""Offline model audit and read-only, torque-off FACTR measurements.

These measurements check kinematics and predicted gravity. They do NOT identify
physical gravity, friction, or external force without independent effort data.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import dynamixel_sdk as dxl

import gravity_comp as gc

DEFAULT_PORT = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRKBU-if00-port0'
URDF = gc.ROOT / 'models/factr_teleop_franka.urdf'


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_calibration(path, port):
    saved = json.loads(Path(path).read_text())
    if saved['port'] != port or saved['models'] != list(gc.MODELS):
        raise ValueError('Calibration belongs to another arm/port')
    offsets = np.asarray(saved['offsets'], dtype=float)
    if offsets.shape != (7,) or not np.isfinite(offsets).all():
        raise ValueError('Invalid calibration offsets')
    return saved, offsets


def audit_pose(model, q):
    data = model.createData()
    g = np.array(pin.computeGeneralizedGravity(model, data, q))
    eps = 1e-6
    gradient = np.array([
        (pin.computePotentialEnergy(model, data, q + np.eye(7)[i] * eps)
         - pin.computePotentialEnergy(model, data, q - np.eye(7)[i] * eps)) / (2 * eps)
        for i in range(7)])
    M = np.array(pin.crba(model, data, q))
    M = (M + M.T) / 2
    J = np.array(pin.computeFrameJacobian(model, data, q, model.getFrameId('link_7'),
                                         pin.LOCAL_WORLD_ALIGNED))
    # Translation / characteristic length makes the rank metric dimensionless.
    scaled = J.copy()
    scaled[:3] /= .2
    _, sv, vt = np.linalg.svd(scaled, full_matrices=True)
    rank = int(np.sum(sv > 1e-8))
    return {
        'q_rad': q.tolist(), 'gravity_nm': g.tolist(),
        'potential_gradient_error_nm': float(np.max(np.abs(g - gradient))),
        'link_mass_matrix_diagonal_kg_m2': np.diag(M).tolist(),
        'mass_matrix_min_eigenvalue': float(np.linalg.eigvalsh(M).min()),
        'jacobian_rank': rank, 'null_dimension': 7 - rank,
        'scaled_jacobian_singular_values': sv.tolist(),
        'characteristic_length_m': .2,
        'null_velocity_basis_rows': vt[rank:].tolist(),
        'jacobian_null_residual': float(np.linalg.norm(J @ vt[rank:].T)),
    }


def offline_audit(calibration, output):
    saved, offsets = load_calibration(calibration, DEFAULT_PORT)
    model, _ = gc.model()
    reference = gc.angles(np.asarray(saved['raw_angles']), offsets)
    qlist = [gc.HOME, reference]
    rng = np.random.default_rng(42)
    qlist.extend(np.clip(gc.HOME + rng.uniform(-.3, .3, (20, 7)),
                        model.lowerPositionLimit, model.upperPositionLimit))
    poses = [audit_pose(model, q) for q in qlist]
    report = {
        'scope': 'Offline consistency only. Physical gravity accuracy and encoder zero are unverified.',
        'urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(),
        'calibration': saved,
        'saved_reference_deviation_deg': np.degrees(reference - gc.HOME).tolist(),
        'frame': 'link_7 origin, NOT a calibrated physical grip point',
        'rotor_inertia': 'Not identified; link URDF inertia must not be assumed to be total actuator inertia.',
        'max_gradient_error_nm': max(p['potential_gradient_error_nm'] for p in poses),
        'minimum_mass_eigenvalue': min(p['mass_matrix_min_eigenvalue'] for p in poses),
        'poses': poses,
        'next': 'Supported torque-off reference and individual-joint recordings; then supervised powered comparisons.',
    }
    dump_new(output, report)
    print('Saved', output)
    print('Saved reference deviations (deg):', np.round(report['saved_reference_deviation_deg'], 2))
    print('Maximum gravity/potential-gradient discrepancy:', report['max_gradient_error_nm'], 'N m')
    print('HOME Jacobian rank:', poses[0]['jacobian_rank'], '; nullity:', poses[0]['null_dimension'])
    return report


def check_passive(bus):
    """Never disable another controller or alter any motor register."""
    for motor in gc.IDS:
        if bus.read(motor, 64):
            raise RuntimeError(f'ID {motor} is torque-enabled; recorder requires torque OFF')
        if bus.read(motor, 70):
            raise RuntimeError(f'ID {motor} reports a hardware error')


def decode(reader):
    raw, v, current = [], [], []
    for motor in gc.IDS:
        if not reader.isAvailable(motor, 126, 10):
            raise RuntimeError(f'Missing telemetry for ID {motor}')
        current.append(gc.signed(reader.getData(motor, 126, 2), 16))
        v.append(gc.signed(reader.getData(motor, 128, 4)) * .229 * 2 * np.pi / 60)
        raw.append(gc.signed(reader.getData(motor, 132, 4)) * np.pi / 2048)
    return np.array(raw), gc.SIGNS * v, np.array(current) * gc.MA_PER_RAW


def record(args):
    saved, offsets = load_calibration(args.calibration, args.port)
    args.output.mkdir(parents=True, exist_ok=False)
    dump_new(args.output / 'metadata.json', {
        'calibration': saved, 'urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(),
        'mode': 'read-only torque-off', 'hz_requested': args.hz,
        'current_note': 'Signed electrical telemetry; NOT joint torque or hand force.',
        'status': 'started',
    })
    model, data = gc.model()
    bus = gc.Bus(args.port, args.baud)
    try:
        bus.validate()
        reader = dxl.GroupSyncRead(bus.port, bus.packet, 126, 10)
        for motor in gc.IDS:
            if not reader.addParam(motor):
                raise RuntimeError(f'Cannot configure read for ID {motor}')
        stages = [('reference', 5)] + [(f'joint_{i}', args.duration) for i in gc.IDS] if args.guided else [(args.label, args.duration)]
        fields = ['time_s', 'segment', 'read_duration_s']
        for name in ('raw_rad', 'q_rad', 'dq_rad_s', 'present_current_ma', 'model_gravity_nm'):
            fields.extend(f'{name}_{i}' for i in gc.IDS)
        with (args.output / 'samples.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            origin = time.monotonic()
            for label, duration in stages:
                if label == 'reference':
                    prompt = 'Support the arm in docs/calibration_pose.png; hold all joints still.'
                else:
                    prompt = (f'Move only {label.replace("_", " ")} slowly out and back twice; '
                              'support the other links. Stop if a cable tightens.')
                print(prompt if args.guided else f'Recording label: {label}', flush=True)
                input(f'Press Enter when ready for {duration:g} seconds of torque-OFF recording: ')
                check_passive(bus)
                start = last_check = time.monotonic()
                while time.monotonic() - start < duration:
                    tick = time.monotonic()
                    bus.check(reader.txRxPacket())
                    elapsed = time.monotonic() - tick
                    if elapsed > .1:
                        raise RuntimeError('Telemetry took more than 100 ms; recording stopped')
                    raw, velocity, current = decode(reader)
                    q = gc.angles(raw, offsets)
                    g = np.array(pin.computeGeneralizedGravity(model, data, q))
                    writer.writerow([tick-origin, label, elapsed, *raw, *q, *velocity, *current, *g])
                    stream.flush()
                    if tick - last_check >= 1:
                        check_passive(bus)
                        last_check = tick
                    time.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
                print('Segment saved:', label, flush=True)
    finally:
        # This tool has no authority to change torque state, including on exit.
        bus.port.closePort()
    analyze(args.output)


def analyze(directory):
    directory = Path(directory)
    with (directory / 'samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No recorded samples')
    summaries = []
    for label in dict.fromkeys(row['segment'] for row in rows):
        selected = [r for r in rows if r['segment'] == label]
        q = np.array([[float(r[f'q_rad_{i}']) for i in gc.IDS] for r in selected])
        q = np.unwrap(q, axis=0)
        span = np.degrees(np.ptp(q, axis=0))
        result = {'segment': label, 'samples': len(selected), 'range_deg': span.tolist(),
                  'dominant_motor_id': int(np.argmax(span)) + 1}
        if label == 'reference':
            result['mean_deviation_from_HOME_deg'] = np.degrees(q.mean(axis=0) - gc.HOME).tolist()
            result['stationary_under_1_degree'] = bool(span.max() < 1)
            result['note'] = 'Photo/reference matching requires human verification; this is not a zero-offset fit.'
        if label.startswith('joint_') and label[6:].isdigit():
            expected = int(label[6:])
            result['expected_motor_id'] = expected
            result['dominant_matches_label'] = result['dominant_motor_id'] == expected
            result['note'] = 'Index screening only: coupled manual motion and encoder sign still require review.'
        summaries.append(result)
    report = {'scope': 'Kinematic screening only; gravity/friction/force parameters were NOT fitted.',
              'segments': summaries}
    # Do not overwrite an earlier analysis of the same recording.
    output = directory / f'analysis_{time.time_ns()}.json'
    dump_new(output, report)
    print(json.dumps(report, indent=2))
    print('Saved', output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='Offline model checks; never opens the serial port')
    audit.add_argument('--calibration', type=Path, default=gc.ROOT / 'calibration.json')
    audit.add_argument('--output', type=Path, default=gc.ROOT / 'outputs/diagnostics/model_audit.json')
    rec = commands.add_parser('record', help='Read-only recording; refuses torque-enabled motors')
    rec.add_argument('--calibration', type=Path, default=gc.ROOT / 'calibration.json')
    rec.add_argument('--port', default=DEFAULT_PORT)
    rec.add_argument('--baud', type=int, default=4000000)
    rec.add_argument('--output', type=Path, required=True, help='New directory; existing directories refused')
    rec.add_argument('--duration', type=float, default=12)
    rec.add_argument('--hz', type=float, default=50)
    rec.add_argument('--label', default='free_motion')
    rec.add_argument('--guided', action='store_true', help='Reference pose followed by joints 1–7, with Enter prompts')
    analysis = commands.add_parser('analyze')
    analysis.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'audit':
        offline_audit(args.calibration, args.output)
    elif args.command == 'analyze':
        analyze(args.directory)
    else:
        if not 1 <= args.duration <= 300 or not 1 <= args.hz <= 100:
            parser.error('duration must be 1–300 seconds and hz 1–100')
        record(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nRecording interrupted; flushed samples remain on disk. Motor state was not changed.')
