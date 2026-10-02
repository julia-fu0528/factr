"""Standalone current-mode gravity compensation for the original FACTR Franka leader."""
import argparse
import csv
import json
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import dynamixel_sdk as dxl

ROOT = Path(__file__).resolve().parent
IDS = tuple(range(1, 8))
MODELS = (1220, 1030, 1220, 1030, 1220, 1220, 1220)
SIGNS = np.array([1, 1, 1, -1, 1, -1, 1])
HOME = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, 0])
# Upstream FACTR empirical conversion: goal-current register units per N m.
TORQUE_TO_RAW = np.array([1158.73, 1000 / 2.69, 1158.73, 1000 / 2.69,
                          1158.73, 1158.73, 1158.73])
MA_PER_RAW = np.array([1, 2.69, 1, 2.69, 1, 1, 1])
# No fitted gravity correction is currently applied. A trial joint-4 scale
# of 0.30/0.85 caused sagging and was withdrawn; easier lowering did not
# establish the torque needed to support the downstream assembly.
GRAVITY_MODEL_SCALE = np.ones(7)


def signed(value, bits=32):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def angles(raw, offsets):
    # Encoders can reset by whole turns on reboot or mode change.
    q = SIGNS * (raw - offsets)
    return HOME + (q - HOME + np.pi) % (2 * np.pi) - np.pi


def calibration_offsets(samples, velocities):
    samples = np.asarray(samples, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if (samples.ndim != 2 or samples.shape[1] != 7 or len(samples) < 2
            or velocities.shape != samples.shape
            or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(velocities))):
        raise RuntimeError('Invalid calibration samples')
    samples = np.unwrap(samples, axis=0)
    if np.max(np.abs(velocities)) > .05 or np.max(np.ptp(samples, axis=0)) > .02:
        raise RuntimeError('Arm is moving; hold it still during calibration')
    raw = np.mean(samples, axis=0)
    # Original calibration: choose the nearest pi/2-indexed assembly offset.
    offsets = np.round((raw - SIGNS * HOME) / (np.pi / 2)) * (np.pi / 2)
    if np.max(np.abs(angles(raw, offsets) - HOME)) > .35:
        raise RuntimeError('Reference pose differs by >20 degrees; reposition and retry')
    return raw, offsets


def current_command(torque, limits):
    if not np.all(np.isfinite(torque)):
        raise RuntimeError("Nonfinite torque")
    return np.clip(np.rint(torque * SIGNS * TORQUE_TO_RAW), -limits, limits).astype(int)


class Bus:
    def __init__(self, port, baud):
        self.port = dxl.PortHandler(port)
        self.packet = dxl.PacketHandler(2.0)
        if not self.port.openPort():
            raise RuntimeError(f"Cannot open {port}")
        try:
            if not self.port.setBaudRate(baud):
                raise RuntimeError(f"Cannot set baud rate {baud}")
            # Prevent concurrent controllers from opening the same serial device.
            import fcntl
            import termios
            fcntl.ioctl(self.port.ser.fileno(), termios.TIOCEXCL)
        except BaseException:
            self.port.closePort()
            raise
        self.reader = dxl.GroupSyncRead(self.port, self.packet, 126, 10)
        for motor in IDS:
            if not self.reader.addParam(motor):
                raise RuntimeError(f"Cannot add motor {motor}")

    def check(self, result, error=0):
        if result != dxl.COMM_SUCCESS or error:
            raise RuntimeError(f"Dynamixel: {self.packet.getTxRxResult(result)}; "
                               f"{self.packet.getRxPacketError(error)}")

    def read(self, motor, address, size=1):
        value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
            self.port, motor, address)
        try:
            self.check(result, error)
        except RuntimeError as exc:
            raise RuntimeError(f'ID {motor}, read register {address} ({size} bytes): {exc}') from exc
        return value

    def write(self, motor, address, value, size=1):
        result, error = getattr(self.packet, f"write{size}ByteTxRx")(
            self.port, motor, address, value)
        try:
            self.check(result, error)
        except RuntimeError as exc:
            raise RuntimeError(f'ID {motor}, write register {address} ({size} bytes): {exc}') from exc

    def inspect(self):
        found = []
        for motor in range(1, 9):
            model, result, error = self.packet.ping(self.port, motor)
            if result != dxl.COMM_SUCCESS:
                print(f"ID {motor}: no response")
                continue
            self.check(result, error)
            found.append((motor, model))
            print(f"ID {motor}: model={model}, firmware={self.read(motor, 6)}, "
                  f"mode={self.read(motor, 11)}, torque={self.read(motor, 64)}, "
                  f"error={self.read(motor, 70)}, "
                  f"voltage={self.read(motor, 144, 2) / 10:.1f} V, "
                  f"temperature={self.read(motor, 146)} C")
        return found

    def validate(self):
        limits = []
        for motor, expected in zip(IDS, MODELS):
            actual = self.read(motor, 0, 2)
            if actual != expected:
                raise RuntimeError(f"ID {motor}: expected model {expected}, got {actual}; "
                                   "this controller requires the original FACTR arm")
            if self.read(motor, 64):
                raise RuntimeError(f"ID {motor}: already torque-enabled; stop the other controller first")
            if self.read(motor, 70):
                raise RuntimeError(f"ID {motor}: hardware error")
            limits.append(self.read(motor, 38, 2))
        return np.array(limits)

    def state(self):
        result = self.reader.txRxPacket()
        try:
            self.check(result)
        except RuntimeError as exc:
            # The SDK stops at the first failed read. Later buffers can contain
            # old data, so report incomplete buffers rather than claiming a
            # complete list of nonresponding motors. No stale data is returned.
            buffers = getattr(self.reader, 'data_dict', {})
            incomplete = [motor for motor, payload in buffers.items() if len(payload) != 10]
            raise RuntimeError(f'State sync read registers 126–135; incomplete response buffers '
                               f'for IDs {incomplete}: {exc}') from exc
        q, v = [], []
        for motor in IDS:
            if not self.reader.isAvailable(motor, 128, 8):
                raise RuntimeError(f"Missing state for ID {motor}")
            q.append(signed(self.reader.getData(motor, 132, 4)) * np.pi / 2048)
            v.append(signed(self.reader.getData(motor, 128, 4)) * .229 * 2 * np.pi / 60)
        return np.array(q), np.array(v)

    def currents(self, values):
        writer = dxl.GroupSyncWrite(self.port, self.packet, 102, 2)
        for motor, value in zip(IDS, values):
            value = int(value) & 0xffff
            if not writer.addParam(motor, [value & 255, value >> 8]):
                raise RuntimeError(f"Cannot add current for ID {motor}")
        try:
            self.check(writer.txPacket())
        except RuntimeError as exc:
            raise RuntimeError(f'Goal-current sync write: {exc}') from exc

    def present_current_ma(self):
        """Electrical telemetry from the last state read, not measured joint torque."""
        values = []
        for motor, scale in zip(IDS, MA_PER_RAW):
            if not self.reader.isAvailable(motor, 126, 2):
                raise RuntimeError(f'Missing current telemetry for ID {motor}')
            values.append(signed(self.reader.getData(motor, 126, 2), 16) * scale)
        return np.array(values)

    def stop(self):
        # Attempt every motor even if a previous motor is disconnected.
        for motor in IDS:
            for address, value, size in ((64, 0, 1), (98, 0, 1), (102, 0, 2)):
                try:
                    self.write(motor, address, value, size)
                except Exception as exc:
                    print(f"Shutdown ID {motor}, register {address}: {exc}", file=sys.stderr)


def model():
    m = pin.buildModelFromUrdf(str(ROOT / 'models/factr_teleop_franka.urdf'))
    if m.nq != 7 or m.nv != 7:
        raise RuntimeError("Expected seven revolute joints")
    return m, m.createData()


def gravity_gains(args):
    """Absolute joint-4 override; all other joints retain the global gain."""
    gains = np.full(7, args.gain, dtype=float)
    override = getattr(args, 'joint4_gain', None)
    if override is not None:
        gains[3] = override
    if not np.all(np.isfinite(gains)) or np.any((gains < 0) | (gains > 1.2)):
        raise ValueError('Gravity gains must be finite and between 0 and 1.2')
    return gains


def corrected_gravity(raw_gravity):
    """Empirical motor-space correction; does not modify URDF dynamics."""
    return GRAVITY_MODEL_SCALE * np.asarray(raw_gravity)


class StaggeredStatus:
    """One register read per poll instead of a 21-read burst each second."""
    def __init__(self, start):
        self.last = start
        self.index = 0
        self.registers = [(motor, register) for motor in IDS for register in (70, 98, 146)]

    def poll(self, bus, now):
        if now-self.last < .03:
            return
        motor, register = self.registers[self.index]
        value = bus.read(motor, register)
        self.last = now
        self.index = (self.index+1) % len(self.registers)
        if register == 70 and value:
            raise RuntimeError(f'Motor {motor}: hardware fault')
        if register == 98 and value != 10:
            raise RuntimeError(f'Motor {motor}: watchdog fault')
        if register == 146 and value >= 65:
            raise RuntimeError(f'Motor {motor}: temperature >=65 C')


def run(bus, args, offsets, hardware_limits, on_state=None):
    gains = gravity_gains(args)
    m, data = model()
    limits = np.minimum(hardware_limits, np.floor(args.max_current_ma / MA_PER_RAW))
    raw, _ = bus.state()
    if np.max(np.abs(angles(raw, offsets) - HOME)) > .35:
        raise RuntimeError("Place arm back in docs/calibration_pose.png before starting")
    latency = Path('/sys/bus/usb-serial/devices') / Path(args.port).resolve().name / 'latency_timer'
    if not latency.exists() or latency.read_text().strip() != '1':
        raise RuntimeError(f"Set USB latency first: echo 1 | sudo tee {latency}")
    log_stream = None
    # Cleanup covers partial startup, including failure enabling a single motor.
    try:
        log_path = getattr(args, 'log', None)
        writer = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Start a fresh log at the requested path before enabling motors.
            log_stream = log_path.open('w', newline='')
            writer = csv.writer(log_stream)
            columns = ['time_s', 'loop_dt_s', 'read_duration_s', 'gain', 'damping']
            for name in ('q_rad', 'dq_rad_s', 'model_gravity_nm', 'requested_torque_nm',
                         'command_current_raw', 'current_limit_raw', 'present_current_ma',
                         'effective_gravity_gain', 'gravity_model_scale', 'corrected_gravity_nm'):
                columns.extend(f'{name}_{i}' for i in IDS)
            writer.writerow(columns)
            log_stream.flush()
        for motor in IDS:
            if bus.read(motor, 6) < 38:
                raise RuntimeError(f"ID {motor}: firmware >=38 required for watchdog")
            bus.write(motor, 98, 0)
            if bus.read(motor, 11) != 0:
                bus.write(motor, 11, 0)
            if bus.read(motor, 11) != 0:
                raise RuntimeError(f"ID {motor}: current mode verification failed")
            bus.write(motor, 102, 0, 2)
        for motor in IDS:
            bus.write(motor, 64, 1)
            bus.write(motor, 98, 10)  # 200 ms communication watchdog
            if bus.read(motor, 98) != 10:
                raise RuntimeError(f"ID {motor}: watchdog verification failed")
        start = previous = last_status = time.monotonic()
        health = StaggeredStatus(start)
        print("Gravity compensation active. Support arm before Ctrl-C; torque will turn off.")
        print('Gravity gains for joints 1–7:', gains, flush=True)
        print('Gravity model correction:', GRAVITY_MODEL_SCALE, flush=True)
        while time.monotonic() - start < args.duration:
            tick = time.monotonic()
            loop_dt = tick - previous
            if loop_dt > .1:
                raise RuntimeError("Control loop stalled for >100 ms")
            previous = tick
            raw, velocity = bus.state()
            q = angles(raw, offsets)
            read_duration = time.monotonic() - tick
            if read_duration > .1:
                raise RuntimeError("Stale motor state")
            if np.max(np.abs(velocity)) > 4:
                raise RuntimeError("Joint speed exceeds 4 rad/s")
            # Pure gravity plus small viscous damping, with a two-second ramp.
            gravity = np.array(pin.computeGeneralizedGravity(m, data, q))
            corrected = corrected_gravity(gravity)
            torque = min((tick - start) / 2, 1) * (
                gains * corrected
                - args.damping * SIGNS * velocity)
            command = current_command(torque, limits)
            bus.currents(command)
            if writer is not None:
                writer.writerow([tick-start, loop_dt, read_duration, args.gain, args.damping,
                                 *q, *(SIGNS * velocity), *gravity, *torque, *command, *limits,
                                 *bus.present_current_ma(), *gains, *GRAVITY_MODEL_SCALE, *corrected])
                log_stream.flush()
            if on_state is not None:
                on_state(q)
            health.poll(bus, tick)
            if tick - last_status > 1:
                print(f"q={np.round(q, 2)} tau={np.round(torque, 3)}", flush=True)
                last_status = tick
            time.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
    finally:
        bus.stop()
        if log_stream is not None:
            log_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'stop', 'calibrate', 'preview', 'run'])
    parser.add_argument('--port', default='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRKBU-if00-port0')
    parser.add_argument('--baud', type=int, default=4000000)
    parser.add_argument('--calibration', type=Path, default=ROOT / 'calibration.json')
    parser.add_argument('--gain', type=float, default=.85)
    parser.add_argument('--joint4-gain', type=float,
                        help='Joint-4 gain AFTER model correction (0–1.2); normally omit and use --gain')
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--max-current-ma', type=float, default=350)
    parser.add_argument('--hz', type=float, default=200)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--log', type=Path, help='Run only: save diagnostic CSV; overwrite an existing file directly')
    args = parser.parse_args()
    try:
        gains = gravity_gains(args)
    except ValueError as exc:
        parser.error(str(exc))
    for name, low, high in [('gain', 0, 1.2), ('damping', 0, .1),
                            ('max_current_ma', 1, 900), ('hz', 20, 500),
                            ('duration', .1, 86400)]:
        if not low <= getattr(args, name) <= high:
            parser.error(f'{name} must be between {low} and {high}')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    bus = Bus(args.port, args.baud)
    try:
        if args.command == 'inspect':
            found = bus.inspect()
            if not found:
                raise RuntimeError('No motors responded: check external motor power, cable, and baud rate')
            return
        if args.command == 'stop':
            print('Disabling arm torque. The arm must be supported.', flush=True)
            for motor, expected in zip(IDS, MODELS):
                if bus.read(motor, 0, 2) != expected:
                    raise RuntimeError(f'Unexpected model for ID {motor}; no stop commands sent')
            bus.stop()
            states = {motor: bus.read(motor, 64) for motor in IDS}
            print('Torque-enable states:', states)
            if any(states.values()):
                raise RuntimeError('Some motors remain enabled; support arm and disconnect motor power')
            return
        limits = bus.validate()
        if args.command == 'calibrate':
            print('Place arm in docs/calibration_pose.png, with base upright. No torque will be enabled.')
            input('Press Enter when the arm is in the reference pose: ')
            samples, velocities = [], []
            for _ in range(25):
                raw, v = bus.state()
                samples.append(raw)
                velocities.append(v)
                time.sleep(.04)
            raw, offsets = calibration_offsets(samples, velocities)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=args.calibration.parent,
                                                 prefix='.calibration-', delete=False) as output:
                    temporary = Path(output.name)
                    json.dump({'port': args.port, 'models': MODELS,
                               'method': 'indexed',
                               'reference_angles': HOME.tolist(), 'raw_angles': raw.tolist(),
                               'offsets': offsets.tolist()}, output, indent=2)
                    output.write('\n')
                temporary.replace(args.calibration)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            print(f'Saved {args.calibration}; q={angles(raw, offsets)}')
            return
        saved = json.loads(args.calibration.read_text())
        if saved['port'] != args.port or saved['models'] != list(MODELS):
            raise RuntimeError('Calibration belongs to a different arm')
        offsets = np.asarray(saved['offsets'], dtype=float)
        if offsets.shape != (7,) or not np.all(np.isfinite(offsets)):
            raise RuntimeError('Invalid calibration offsets')
        if args.command == 'preview':
            m, data = model()
            q = angles(bus.state()[0], offsets)
            print('Joint angles (rad):', q)
            raw_gravity = np.array(pin.computeGeneralizedGravity(m, data, q))
            print('Raw URDF gravity (N m):', raw_gravity)
            print('Gravity torque after correction and gain (N m):', gains * corrected_gravity(raw_gravity))
        else:
            run(bus, args, offsets, limits)
    finally:
        bus.port.closePort()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
