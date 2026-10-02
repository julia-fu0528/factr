import unittest
from unittest.mock import Mock, patch

import numpy as np

import gravity_comp as gc
import identify_gravity as ident


class IdentificationTests(unittest.TestCase):
    def test_onset_rejects_noise_and_requires_sustained_motion(self):
        detector = ident.Onset(0, 1)
        for _ in range(10):
            self.assertFalse(detector.update(np.deg2rad(.088), .03))
        self.assertFalse(detector.update(np.deg2rad(.3), .04))
        self.assertFalse(detector.update(np.deg2rad(.35), .04))
        self.assertTrue(detector.update(np.deg2rad(.4), .04))
        with self.assertRaisesRegex(RuntimeError, 'opposite'):
            ident.Onset(0, 1).update(-np.deg2rad(.5), -.03)

    @staticmethod
    def result(direction, bracket, pose=1, repeat=1, gravity=.2):
        return {'status': 'onset', 'pose': pose, 'repeat': repeat, 'direction': direction,
                'q_start': gc.HOME.tolist(), 'bracket_model_raw': bracket,
                'model_gravity_nm': gravity}

    def test_midpoint_and_friction_width_in_current_units(self):
        result = ident.pair_thresholds(self.result(1, [120, 122]), self.result(-1, [78, 80]))
        self.assertAlmostEqual(result['gravity_model_signed_ma_estimate'], 100*2.69)
        np.testing.assert_allclose(result['gravity_model_signed_ma_interval'], [99*2.69, 101*2.69])
        self.assertAlmostEqual(result['friction_half_width_ma_estimate'], 21*2.69)

    def test_censored_wrong_pose_and_overlapping_pairs_are_not_fitted(self):
        plus = self.result(1, [120, 122])
        minus = self.result(-1, [78, 80])
        minus['q_start'][5] += .1
        with self.assertRaises(ValueError):
            ident.pair_thresholds(plus, minus)
        with self.assertRaises(ValueError):
            ident.pair_thresholds(plus, self.result(-1, [121, 123]))
        plus['status'] = 'censored'
        report = ident.summarize([plus, self.result(-1, [78, 80])])
        self.assertEqual(report['accepted_pairs'], [])
        self.assertFalse(report['applied_to_controller'])
        self.assertNotIn('candidate_current_fit_NOT_APPLIED', report)

    def test_candidate_fit_recovers_known_synthetic_current_curve(self):
        rows = []
        for pose, g in enumerate([.12, .2, .3], 1):
            midpoint = (1000*g+10)/2.69
            for repeat in (1, 2):
                rows.extend([self.result(1, [midpoint+20, midpoint+22], pose, repeat, g),
                             self.result(-1, [midpoint-22, midpoint-20], pose, repeat, g)])
        report = ident.summarize(rows)
        fit = report['candidate_current_fit_NOT_APPLIED']
        self.assertAlmostEqual(fit['slope_ma_per_model_nm'], 1000)
        self.assertAlmostEqual(fit['bias_ma'], 10)
        self.assertFalse(report['applied_to_controller'])

    def test_motion_bounds_and_saturation(self):
        model, _ = gc.model()
        for joint in (3, 5):
            q = gc.HOME.copy()
            q[joint] += np.deg2rad(3)
            with self.assertRaises(RuntimeError):
                ident.check_motion(q, np.zeros(7), gc.HOME, model, ident.Limits())
        with self.assertRaisesRegex(RuntimeError, 'Speed'):
            ident.check_motion(gc.HOME, np.ones(7), gc.HOME, model, ident.Limits())
        with self.assertRaisesRegex(RuntimeError, 'saturate'):
            ident.support_command(np.ones(7), gc.HOME, np.zeros(7), gc.HOME, np.ones(7))

    def test_partial_enable_failure_stops_all_motors(self):
        bus = Mock()
        with patch.object(ident, 'enable', side_effect=RuntimeError('enable failure')), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'enable failure'):
                ident.trial(bus, np.zeros(7), np.ones(7)*500, gc.HOME, 1, 1, 1, Mock())
        bus.stop.assert_called_once()

    def test_state_failure_after_enable_stops_without_more_commands(self):
        bus = Mock()
        bus.state.side_effect = RuntimeError('read failure')
        with patch.object(ident, 'enable') as enabled, patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'read failure'):
                ident.trial(bus, np.zeros(7), np.ones(7)*500, gc.HOME, 1, 1, 1, Mock())
        enabled.assert_called_once()
        bus.currents.assert_not_called()
        bus.stop.assert_called_once()

    def test_closed_loop_synthetic_onset_and_shutdown(self):
        # Synthetic current threshold plant; no serial access or real sleeping.
        for direction in (1, -1):
            now = [0.]
            q = gc.HOME.copy()
            commanded = np.zeros(7, dtype=int)
            model, data = gc.model()
            baseline = int(gc.current_command(.85*ident.pin.computeGeneralizedGravity(model, data, q),
                                             np.ones(7)*130)[3]*gc.SIGNS[3])
            def clock():
                now[0] += .0001
                return now[0]
            def sleep(dt):
                now[0] += dt
            def state():
                value = gc.SIGNS[3]*commanded[3]
                v = np.zeros(7)
                # Model stiction during the initial support ramp as well.
                if direction == 1 and value >= baseline+6:
                    v[3] = .04
                elif direction == -1 and 0 < value <= baseline-6 and now[0] > 7:
                    v[3] = -.04
                q[:] += v*.02
                return gc.SIGNS*q, gc.SIGNS*v
            bus = Mock()
            bus.state.side_effect = state
            bus.currents.side_effect = lambda command: commanded.__setitem__(slice(None), command)
            bus.present_current_ma.side_effect = lambda: commanded*gc.MA_PER_RAW
            bus.read.side_effect = lambda motor, address: {70: 0, 146: 25, 64: 1, 98: 10}[address]
            writer = Mock()
            with patch.object(ident, 'enable'), patch.object(ident.time, 'monotonic', clock), \
                 patch.object(ident.time, 'sleep', sleep), patch('builtins.print'):
                result = ident.trial(bus, np.zeros(7), np.ones(7)*500, gc.HOME, 1, 1, direction, writer)
            self.assertEqual(result['status'], 'onset')
            self.assertEqual(result['direction'], direction)
            self.assertLess(abs(result['q_onset'][3]-result['q_start'][3]), np.deg2rad(2))
            bus.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
