import json
import math
import unittest
from unittest.mock import patch

from teleop_stream import HOME, Receiver, advance_target, decode_packet, franka_target


class StreamTests(unittest.TestCase):
    def test_wrist_mapping_is_applied_once_and_preserves_stale_hold(self):
        original = list(HOME)
        for offset in (45, -45, 0):
            mapped = franka_target(original, offset)
            self.assertEqual(mapped[:6], original[:6])
            self.assertAlmostEqual(mapped[6], original[6] + math.radians(offset))
            self.assertEqual(franka_target(original, offset), mapped)
        self.assertEqual(original, HOME)
        self.assertIsNone(franka_target(None, 45))
        target = franka_target(HOME, 45)
        self.assertEqual(advance_target(target, franka_target(None, 45),
                                        [-3] * 7, [3] * 7, .01, 1.5), target)

    def packet(self, q=HOME, stamp=10):
        return json.dumps({'version': 1, 'time': stamp, 'q': q}).encode()

    def test_bad_and_stale_packets(self):
        for payload in [b'bad', b'[]', b'null', self.packet([0] * 6),
                        self.packet([float('nan')] * 7), self.packet(stamp=9),
                        self.packet(stamp=11)]:
            self.assertIsNone(decode_packet(payload, 10.1, .25))
        self.assertEqual(decode_packet(self.packet(), 10.1, .25), (10., HOME))

    def test_limits_slew_and_disconnect(self):
        target = advance_target([0] * 7, [10] * 7, [-1] * 7, [1] * 7, .1, 2)
        self.assertEqual(target, [.2] * 7)
        self.assertEqual(advance_target(target, None, [-1] * 7, [1] * 7, .1, 2), target)
        self.assertEqual(advance_target([.9] * 7, [10] * 7, [-1] * 7, [1] * 7, .1, 2), [1] * 7)

    def test_receiver_ignores_reordered_packets_and_expires(self):
        # Socket mock exercises draining without requiring network permissions.
        with patch('teleop_stream.socket.socket') as factory:
            receiver = Receiver()
            factory.return_value.recv.side_effect = [self.packet(stamp=10),
                                                     self.packet([1] * 7, stamp=9.95),
                                                     BlockingIOError()]
            with patch('teleop_stream.time.monotonic', return_value=10.1):
                self.assertEqual(receiver.poll(), HOME)
            factory.return_value.recv.side_effect = BlockingIOError()
            with patch('teleop_stream.time.monotonic', return_value=10.3):
                self.assertIsNone(receiver.poll())
            receiver.close()


if __name__ == '__main__':
    unittest.main()
