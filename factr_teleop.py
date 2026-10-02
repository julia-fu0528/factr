"""Read FACTR joint angles and stream to Isaac; optional leader gravity compensation."""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from gravity_comp import Bus, MODELS, ROOT, angles, run, gravity_gains
from teleop_stream import Publisher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRKBU-if00-port0')
    parser.add_argument('--baud', type=int, default=4000000)
    parser.add_argument('--calibration', type=Path, default=ROOT / 'calibration_reference.json')
    parser.add_argument('--udp-port', type=int, default=5557)
    parser.add_argument('--gravity-comp', action='store_true', help='Enable leader motor currents; otherwise read only')
    parser.add_argument('--duration', type=float, default=600)
    parser.add_argument('--hz', type=float, default=200)
    parser.add_argument('--gain', type=float, default=.85)
    parser.add_argument('--joint4-gain', type=float,
                        help='Joint-4 gain AFTER model correction (0–1.2); normally omit and use --gain')
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--max-current-ma', type=float, default=350)
    args = parser.parse_args()
    try:
        gravity_gains(args)
    except ValueError as exc:
        parser.error(str(exc))
    for name, low, high in [('udp_port', 1024, 65535), ('duration', .1, 86400),
                            ('hz', 20, 500), ('gain', 0, 1.2), ('damping', 0, .1),
                            ('max_current_ma', 1, 900)]:
        if not low <= getattr(args, name) <= high:
            parser.error(f'{name} must be in [{low}, {high}]')
    saved = json.loads(args.calibration.read_text())
    if saved['port'] != args.port or saved['models'] != list(MODELS):
        raise RuntimeError('Calibration belongs to a different arm')
    offsets = np.asarray(saved['offsets'], dtype=float)
    if offsets.shape != (7,) or not np.all(np.isfinite(offsets)):
        raise RuntimeError('Invalid offsets')

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    publisher = Publisher(args.udp_port)
    bus = None
    try:
        bus = Bus(args.port, args.baud)
        limits = bus.validate()
        print(f'Streaming seven joints to localhost:{args.udp_port}; calibration={args.calibration}', flush=True)
        if args.gravity_comp:
            run(bus, args, offsets, limits, on_state=publisher.send)
        else:
            print('Read-only mode: support the leader by hand. Gripper remains open in simulation.', flush=True)
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                tick = time.monotonic()
                raw, _ = bus.state()
                if time.monotonic() - tick > .1:
                    raise RuntimeError('Motor read took >100 ms; stream stopped')
                publisher.send(angles(raw, offsets))
                time.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
    finally:
        if bus is not None:
            bus.port.closePort()
        publisher.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nLeader stream stopped.')
