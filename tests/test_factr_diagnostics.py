import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import factr_diagnostics as diag
import gravity_comp as gc


class DiagnosticTests(unittest.TestCase):
    def test_enabled_motor_is_rejected_without_any_write(self):
        bus = Mock()
        bus.read.side_effect = lambda motor, register: int(motor == 4 and register == 64)
        with self.assertRaisesRegex(RuntimeError, 'ID 4 is torque-enabled'):
            diag.check_passive(bus)
        bus.write.assert_not_called()
        bus.currents.assert_not_called()
        bus.stop.assert_not_called()

    def test_signed_current_and_encoder_velocity_conventions(self):
        reader = Mock()
        reader.isAvailable.return_value = True
        reader.getData.side_effect = lambda motor, address, size: {
            126: 65535, 128: 0xffffffff, 132: 2048}[address]
        raw, v, current = diag.decode(reader)
        np.testing.assert_allclose(raw, np.pi)
        np.testing.assert_allclose(current, -gc.MA_PER_RAW)
        np.testing.assert_allclose(v, -gc.SIGNS * .229 * 2 * np.pi / 60)

    def test_missing_sample_rejected(self):
        reader = Mock()
        reader.isAvailable.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Missing telemetry'):
            diag.decode(reader)

    def test_model_has_regular_task_and_one_internal_motion_at_home(self):
        model, _ = gc.model()
        result = diag.audit_pose(model, gc.HOME)
        self.assertEqual(result['jacobian_rank'], 6)
        self.assertEqual(result['null_dimension'], 1)
        self.assertLess(result['jacobian_null_residual'], 1e-10)
        self.assertGreater(result['mass_matrix_min_eigenvalue'], 0)
        self.assertLess(result['potential_gradient_error_nm'], 1e-8)

    def test_analysis_flags_mismatched_motor_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / 'samples.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['segment'] + [f'q_rad_{i}' for i in gc.IDS])
                writer.writerow(['joint_2', *gc.HOME])
                q = gc.HOME.copy()
                q[3] += .2
                writer.writerow(['joint_2', *q])
            report = diag.analyze(path)
            self.assertFalse(report['segments'][0]['dominant_matches_label'])
            self.assertEqual(report['segments'][0]['dominant_motor_id'], 4)


if __name__ == '__main__':
    unittest.main()
