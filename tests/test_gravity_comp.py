import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import gravity_comp as gc


class GravityTests(unittest.TestCase):
    def test_default_gravity_has_no_hidden_joint4_reduction(self):
        model, data = gc.model()
        for q in [gc.HOME, gc.HOME + np.array([.1, .2, -.1, .3, .1, -.2, .2])]:
            raw = np.array(pin.computeGeneralizedGravity(model, data, q))
            expected = .85 * raw
            np.testing.assert_allclose(.85 * gc.corrected_gravity(raw), expected, atol=1e-12)
            np.testing.assert_allclose(0 * gc.corrected_gravity(raw), np.zeros(7))

    def test_joint4_override_is_absolute_and_only_changes_joint4(self):
        np.testing.assert_allclose(gc.gravity_gains(SimpleNamespace(gain=.85)), [.85]*7)
        np.testing.assert_allclose(
            gc.gravity_gains(SimpleNamespace(gain=.85, joint4_gain=.3)),
            [.85, .85, .85, .3, .85, .85, .85])
        for invalid in (float('nan'), float('inf'), -.1, 1.3):
            with self.assertRaises(ValueError):
                gc.gravity_gains(SimpleNamespace(gain=.85, joint4_gain=invalid))

    def test_stop_recovers_enabled_motors_without_calibration(self):
        bus = Mock()
        bus.read.side_effect = lambda motor, address, *args: gc.MODELS[motor - 1] if address == 0 else 0
        with patch('sys.argv', ['gravity_comp.py', 'stop']), \
             patch.object(gc, 'Bus', return_value=bus), \
             patch('builtins.print'), patch.object(gc.signal, 'signal'):
            gc.main()
        bus.stop.assert_called_once()
        bus.validate.assert_not_called()
        bus.port.closePort.assert_called_once()

    def test_plain_calibrate_uses_original_indexed_offsets(self):
        expected = np.arange(7) * np.pi / 2
        raw = gc.SIGNS * gc.HOME + expected + .1
        bus = Mock()
        bus.state.return_value = (raw, np.zeros(7))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'calibration.json'
            target.write_text('{"previous": true}\n')
            with patch('sys.argv', ['gravity_comp.py', 'calibrate', '--calibration', str(target)]), \
                 patch.object(gc, 'Bus', return_value=bus), \
                 patch('builtins.input', return_value=''), \
                 patch('builtins.print'), patch.object(gc.time, 'sleep'), \
                 patch.object(gc.signal, 'signal'):
                gc.main()
            saved = json.loads(target.read_text())
            self.assertEqual(saved['method'], 'indexed')
            np.testing.assert_allclose(saved['offsets'], expected, atol=1e-12)
        bus.write.assert_not_called()
        bus.currents.assert_not_called()
        bus.port.closePort.assert_called_once()

    def test_failed_recalibration_preserves_existing_file(self):
        bus = Mock()
        bus.state.return_value = (gc.SIGNS * gc.HOME + .6, np.zeros(7))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'calibration.json'
            previous = '{"previous": true}\n'
            target.write_text(previous)
            with patch('sys.argv', ['gravity_comp.py', 'calibrate', '--calibration', str(target)]), \
                 patch.object(gc, 'Bus', return_value=bus), \
                 patch('builtins.input', return_value=''), \
                 patch('builtins.print'), patch.object(gc.time, 'sleep'), \
                 patch.object(gc.signal, 'signal'):
                with self.assertRaisesRegex(RuntimeError, '>20 degrees'):
                    gc.main()
            self.assertEqual(target.read_text(), previous)

    def test_indexed_offsets_preserve_pose_error_and_relative_motion(self):
        offsets = np.arange(7) * np.pi / 2
        raw = offsets + gc.SIGNS * (gc.HOME + .1)
        samples = np.tile(raw, (25, 1))
        _, fitted = gc.calibration_offsets(samples, np.zeros_like(samples))
        delta = np.array([.1, -.2, .05, -.1, .2, .1, -.15])
        np.testing.assert_allclose(gc.angles(raw + gc.SIGNS * delta, fitted),
                                   gc.HOME + .1 + delta, atol=1e-12)

    def test_indexed_calibration_rejects_large_pose_error(self):
        samples = np.tile(gc.SIGNS * gc.HOME + .6, (25, 1))
        with self.assertRaisesRegex(RuntimeError, '>20 degrees'):
            gc.calibration_offsets(samples, np.zeros_like(samples))

    def test_reference_calibration_rejects_motion(self):
        samples = np.tile(gc.SIGNS * gc.HOME, (25, 1))
        samples[:, 3] += np.linspace(0, .04, 25)
        with self.assertRaisesRegex(RuntimeError, 'moving'):
            gc.calibration_offsets(samples, np.zeros_like(samples))

    def test_gravity_matches_potential_energy_gradient(self):
        model, data = gc.model()
        for q in [gc.HOME, gc.HOME + np.array([.1, .2, -.3, .1, .2, .3, -.1])]:
            expected = []
            for joint in range(7):
                delta = np.eye(7)[joint] * 1e-6
                expected.append((pin.computePotentialEnergy(model, data, q + delta)
                                 - pin.computePotentialEnergy(model, data, q - delta)) / 2e-6)
            np.testing.assert_allclose(pin.computeGeneralizedGravity(model, data, q),
                                       expected, atol=1e-8)

    def test_encoder_turn_reset_and_direction(self):
        offsets = np.arange(7) * np.pi / 2
        raw = offsets + gc.SIGNS * gc.HOME
        np.testing.assert_allclose(gc.angles(raw + 2 * np.pi, offsets), gc.HOME, atol=1e-12)
        raw[3] += .1
        self.assertAlmostEqual(gc.angles(raw, offsets)[3], gc.HOME[3] - .1)

    def test_current_bounds_and_nonfinite(self):
        result = gc.current_command(np.ones(7) * 100, np.ones(7) * 50)
        np.testing.assert_equal(result, gc.SIGNS * 50)
        with self.assertRaises(RuntimeError):
            gc.current_command(np.full(7, np.nan), np.ones(7))
        self.assertEqual(gc.signed(0xffffffff), -1)

    def test_partial_enable_failure_stops_all_motors(self):
        bus = Mock()
        bus.state.return_value = (gc.SIGNS * gc.HOME, np.zeros(7))
        bus.read.side_effect = lambda motor, reg: {6: 50, 11: 0, 98: 10}[reg]
        def fail(motor, address, value, *args):
            if motor == 3 and address == 64:
                raise RuntimeError('Simulated enable failure')
        bus.write.side_effect = fail
        args = SimpleNamespace(port='/dev/ttyUSB0', max_current_ma=350, gain=.85)
        with patch.object(gc.Path, 'exists', return_value=True), \
             patch.object(gc.Path, 'read_text', return_value='1'):
            with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                gc.run(bus, args, np.zeros(7), np.ones(7) * 900)
        bus.stop.assert_called_once()

    def test_stream_failure_disables_gravity_compensation(self):
        bus = Mock()
        bus.state.return_value = (gc.SIGNS * gc.HOME, np.zeros(7))
        bus.read.side_effect = lambda motor, reg: {6: 53, 11: 0, 98: 10}[reg]
        args = SimpleNamespace(port='/dev/ttyUSB0', max_current_ma=350,
                               duration=1, gain=.85, damping=.01, hz=200)
        publish = Mock(side_effect=RuntimeError('Stream failure'))
        with patch.object(gc.Path, 'exists', return_value=True), \
             patch.object(gc.Path, 'read_text', return_value='1'), \
             patch.object(gc.time, 'monotonic', side_effect=[0, .01, .02, .03]):
            with self.assertRaisesRegex(RuntimeError, 'Stream failure'):
                gc.run(bus, args, np.zeros(7), np.ones(7) * 900, on_state=publish)
        bus.currents.assert_called_once()
        bus.stop.assert_called_once()

    def test_shutdown_continues_after_disconnected_motor(self):
        bus = object.__new__(gc.Bus)
        bus.write = Mock(side_effect=lambda motor, *a: (_ for _ in ()).throw(
            RuntimeError('Disconnected')) if motor == 1 else None)
        with patch('sys.stderr'):
            bus.stop()
        self.assertEqual(bus.write.call_count, 21)
        bus.write.assert_any_call(7, 64, 0, 1)

    def test_logged_sample_and_callback_failure_still_disable_motors(self):
        import csv
        bus = Mock()
        bus.state.return_value = (gc.SIGNS * gc.HOME, np.zeros(7))
        bus.present_current_ma.return_value = np.arange(7, dtype=float)
        bus.read.side_effect = lambda motor, reg: {6: 53, 11: 0, 98: 10}[reg]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(port='/dev/ttyUSB0', max_current_ma=350,
                                   duration=1, gain=.85, damping=.01, hz=200,
                                   log=Path(directory) / 'run.csv', joint4_gain=.3)
            with patch.object(gc.Path, 'exists', return_value=True), \
                 patch.object(gc.Path, 'read_text', return_value='1'), \
                 patch.object(gc.time, 'monotonic', side_effect=[0, .01, .02, .03]):
                with self.assertRaisesRegex(RuntimeError, 'Callback failure'):
                    gc.run(bus, args, np.zeros(7), np.ones(7)*900,
                           on_state=Mock(side_effect=RuntimeError('Callback failure')))
            with args.log.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['present_current_ma_7']), 6.)
            self.assertAlmostEqual(float(rows[0]['q_rad_4']), gc.HOME[3])
            for i in gc.IDS:
                expected_gain = .3 if i == 4 else .85
                self.assertAlmostEqual(float(rows[0][f'effective_gravity_gain_{i}']), expected_gain)
                self.assertAlmostEqual(float(rows[0][f'requested_torque_nm_{i}']),
                                       .01 * expected_gain * gc.GRAVITY_MODEL_SCALE[i-1]
                                       * float(rows[0][f'model_gravity_nm_{i}']))
                self.assertAlmostEqual(float(rows[0][f'corrected_gravity_nm_{i}']),
                                       gc.GRAVITY_MODEL_SCALE[i-1] * float(rows[0][f'model_gravity_nm_{i}']))
            logged = [int(rows[0][f'command_current_raw_{i}']) for i in gc.IDS]
            np.testing.assert_array_equal(logged, bus.currents.call_args.args[0])
        bus.stop.assert_called_once()

    def test_existing_log_refused_before_enabling(self):
        bus = Mock()
        bus.state.return_value = (gc.SIGNS * gc.HOME, np.zeros(7))
        with tempfile.TemporaryDirectory() as directory:
            logfile = Path(directory) / 'run.csv'
            logfile.write_text('preserve me')
            args = SimpleNamespace(port='/dev/ttyUSB0', max_current_ma=350, log=logfile, gain=.85)
            with patch.object(gc.Path, 'exists', return_value=True), \
                 patch.object(gc.Path, 'read_text', return_value='1'):
                with self.assertRaises(FileExistsError):
                    gc.run(bus, args, np.zeros(7), np.ones(7)*900)
            self.assertEqual(logfile.read_text(), 'preserve me')
        bus.write.assert_not_called()
        bus.currents.assert_not_called()


if __name__ == '__main__':
    unittest.main()
