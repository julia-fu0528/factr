import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch
import gravity_comp as gc
import pinocchio as pin
import numpy as np
from inertia_shadow import MomentumObserver,shaping_matrix,analyze

class ShadowTests(unittest.TestCase):
    def test_constant_external_effort_recovered(self):
        obs=MomentumObserver();p=np.zeros(7);motor=np.arange(7)*.01;external=np.arange(7)*.02
        obs.update(p,motor,0)
        for _ in range(1000):
            p+=(motor+external)*.005
            r,reset=obs.update(p,motor,.005)
        np.testing.assert_allclose(r,external,atol=1e-8)
        self.assertFalse(reset)

    def test_stationary_gravity_mismatch_is_not_zeroed_away(self):
        obs=MomentumObserver()
        for _ in range(1000):r,_=obs.update(np.zeros(7),np.ones(7)*-.1,.005)
        np.testing.assert_allclose(r,.1,atol=1e-8)
        r,reset=obs.update(np.zeros(7),np.ones(7),.2)
        self.assertTrue(reset);np.testing.assert_equal(r,np.zeros(7))

    def test_null_direction_preserved_and_task_assisted(self):
        M=np.diag(np.arange(1,8.));J=np.column_stack([np.eye(6),np.ones(6)*.2])
        K,rank,enabled=shaping_matrix(M,J)
        self.assertTrue(enabled);self.assertEqual(rank,6)
        _,_,vt=np.linalg.svd(J)
        n=vt[-1]
        np.testing.assert_allclose(K@(M@n),0,atol=1e-12)
        for axis in np.eye(6):np.testing.assert_allclose(K@J.T@axis,.25*J.T@axis,atol=1e-12)

    def test_singular_pose_has_no_candidate_assistance(self):
        J=np.zeros((6,7));J[:5,:5]=np.eye(5)
        K,rank,enabled=shaping_matrix(np.eye(7),J)
        self.assertFalse(enabled);self.assertEqual(rank,5);np.testing.assert_equal(K,0)

    def test_replay_gravity_assumption_does_not_change_logged_motor_command(self):
        m,d=gc.model();g=np.array(pin.computeGeneralizedGravity(m,d,gc.HOME))
        command=gc.current_command(.85*g,np.ones(7)*10000)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'input.csv'
            with path.open('w',newline='') as stream:
                w=csv.writer(stream)
                w.writerow(['time_s']+[f'{key}_{i}' for key in ('q_rad','dq_rad_s','command_current_raw') for i in gc.IDS])
                for i in range(400):w.writerow([i*.01,*gc.HOME,*np.zeros(7),*command])
            before=path.read_bytes()
            with patch('builtins.print'):
                full=analyze(path,root/'full')
                scaled=analyze(path,root/'scaled',model_gravity_scale=.85)
            self.assertGreater(full['residual_rms_nm'][3],.02)
            self.assertLess(scaled['residual_rms_nm'][3],.002)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(scaled['hardware_commands_sent'])
