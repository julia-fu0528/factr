import unittest
from unittest.mock import Mock, patch

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from balanced_drag import target_shaping, handle_jacobian, BalancedLimiter
from factr_inertia import LiveController
import factr_inertia as live


class TargetTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(41)
        A=rng.normal(size=(7,7))*.15
        self.M=A@A.T+np.diag(np.linspace(1,2,7))
        self.J=np.column_stack([np.eye(6),np.arange(6)*.04])

    def test_unclipped_target_and_natural_nullspace(self):
        target=np.linspace(.8,1.4,6)
        K,info=target_shaping(self.M,self.J,target,kappa=10,resist=.99)
        self.assertTrue(info['geometry_ok']);self.assertEqual(info['null_dimension'],1)
        self.assertEqual(info['geometry_fade'],1)
        Minv=info['effective_inverse_inertia']
        np.testing.assert_allclose(self.J@Minv@self.J.T,np.diag(1/target),atol=1e-12)
        np.testing.assert_allclose(np.linalg.solve(self.M,np.eye(7)+K),Minv,atol=1e-12)
        n=np.linalg.svd(self.J,full_matrices=True)[2][-1]
        np.testing.assert_allclose(K@(self.M@n),0,atol=1e-12)
        np.testing.assert_allclose(Minv@(self.M@n),n,atol=1e-12)
        np.testing.assert_allclose(info['null_torque_projector']@K,0,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(Minv).min(),0)

    def test_both_assisting_and_resisting_task_modes_with_bounds(self):
        J=np.column_stack([np.eye(6),np.zeros(6)])
        K,info=target_shaping(np.eye(7),J,[.1,10,1,1,1,1],kappa=.5,resist=.25)
        self.assertAlmostEqual(K[0,0],.5)
        self.assertAlmostEqual(K[1,1],-.25)
        np.testing.assert_allclose(K[6],0,atol=1e-12)
        self.assertTrue(np.all(info['clipped_gains']>=-.25-1e-12))
        self.assertTrue(np.all(info['clipped_gains']<=.5+1e-12))

    def test_singular_and_near_singular_fallback(self):
        for small in (0.,1e-6):
            J=self.J.copy();J[-1]=0;J[-1,5]=small
            K,info=target_shaping(self.M,J,np.ones(6))
            self.assertFalse(info['geometry_ok'])
            np.testing.assert_array_equal(K,0)
            np.testing.assert_allclose(info['effective_inverse_inertia'],np.linalg.inv(self.M))

    def test_handle_offset_matches_position_finite_difference(self):
        model,data=gc.model();q=gc.HOME.copy();frame=model.getFrameId('link_7')
        offset=np.array([.02,-.01,.03])
        J=handle_jacobian(pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL),offset)
        pin.framesForwardKinematics(model,data,q)
        R=data.oMf[frame].rotation.copy()
        def position(q):
            pin.framesForwardKinematics(model,data,q)
            return data.oMf[frame].act(offset).copy()
        h=1e-7
        numerical=np.column_stack([(position(q+np.eye(7)[j]*h)-position(q-np.eye(7)[j]*h))/(2*h) for j in range(7)])
        np.testing.assert_allclose(R@J[:3],numerical,atol=1e-8)

    def test_invalid_targets_rejected(self):
        for target in ([0]*6,[float('nan')]*6,[-1]*6):
            with self.assertRaises(ValueError):target_shaping(self.M,self.J,target)


class BalancedLimiterTests(unittest.TestCase):
    def setUp(self):
        self.caps=np.floor(350/gc.MA_PER_RAW)

    def test_current_only_bypasses_extra_limits_but_keeps_total_current_and_shutdown(self):
        c=LiveController(controller='balanced',assist_limits='current-only').limiter
        request=np.full(7,.04);v=np.full(7,.5);base=np.zeros(7)
        c.enabled_for=3.
        effort,info=c.step(request,v,base,self.caps,.004,enabled=True,trace=True)
        self.assertTrue(np.all(effort>c.TORQUE_CAP))
        self.assertGreater(info['positive_power_w'],c.POWER_CAP)
        np.testing.assert_allclose(info['limiter_trace']['power_energy_effort_nm'],request)
        # Headroom still scales the complete vector, including sign-flipped motors.
        base=self.caps-1
        effort,info=c.step(request,v,base,self.caps,.004,enabled=True,trace=True)
        commanded=base+np.rint(effort*gc.SIGNS*gc.TORQUE_TO_RAW)
        self.assertTrue(np.all(abs(commanded)<=self.caps))
        ratios=info['limiter_trace']['headroom_effort_nm']/request
        np.testing.assert_allclose(ratios,ratios[0])
        off,_=c.step(request,v,base,self.caps,.004,enabled=False)
        np.testing.assert_array_equal(off,0)
        self.assertEqual(c.enabled_for,0)
        effort,info=c.step(request,np.full(7,3.1),base,self.caps,.004,enabled=True)
        self.assertEqual(info['reason'],'assist_speed_gate')
        np.testing.assert_array_equal(effort,0)
        _,info=c.step(request,v,base,self.caps,.02,enabled=True)
        self.assertEqual(info['reason'],'timing_fault')
        with self.assertRaises(ValueError):LiveController(controller='uniform',assist_limits='current-only')

    def test_configured_rise_rate_increases_output_without_changing_other_caps(self):
        slow=LiveController(controller='balanced').limiter
        fast=LiveController(controller='balanced',torque_rise_rate=.16).limiter
        # Isolate the rate limiter after the common activation ramp.
        slow.enabled_for=fast.enabled_for=3.
        request=np.zeros(7);request[3]=.02
        v=np.zeros(7);v[3]=.1
        for _ in range(30):
            low,_=slow.step(request,v,np.zeros(7),self.caps,.004,enabled=True)
            high,info=fast.step(request,v,np.zeros(7),self.caps,.004,enabled=True)
            self.assertTrue(np.all(abs(high)<=fast.TORQUE_CAP+1e-12))
            self.assertLessEqual(info['positive_power_w'],fast.POWER_CAP+1e-12)
        self.assertGreater(high[3],low[3])
        np.testing.assert_array_equal(fast.TORQUE_CAP,slow.TORQUE_CAP)
        self.assertEqual(fast.POWER_CAP,slow.POWER_CAP)
        for invalid in (0,-.1,.401,float('nan'),float('inf')):
            with self.assertRaises(ValueError):LiveController(controller='balanced',torque_rise_rate=invalid)

    def test_signed_stationary_coupling_and_reversal_without_motion_dwell(self):
        c=BalancedLimiter();base=np.zeros(7);v=np.zeros(7);v[0]=.1
        request=np.array([-.004,.008,0,0,0,0,0])
        for _ in range(600):effort,info=c.step(request,v,base,self.caps,.004,enabled=True,trace=True)
        self.assertLess(effort[0],0)  # opposing movement is retained
        self.assertGreater(effort[1],0)  # stationary coupled joint is not suppressed
        self.assertLess(info['negative_power_w'],0)
        self.assertEqual(info['limiter_trace']['motion_ramp'][0],1)
        _,after=c.step(request,-v,base,self.caps,.004,enabled=True,trace=True)
        self.assertEqual(after['limiter_trace']['motion_ramp'][0],1)
        off,_=c.step(request,v,base,self.caps,.004,enabled=False)
        np.testing.assert_array_equal(off,0)
        self.assertEqual(c.enabled_for,0)

    def test_limits_preserve_vector_before_rounding_and_work_does_not_stop(self):
        c=BalancedLimiter();request=np.array([.1,-.2,.03,-.1,.04,-.04,.02])
        v=np.sign(request)*.3;base=np.zeros(7)
        for _ in range(2500):
            effort,info=c.step(request,v,base,self.caps,.004,enabled=True,trace=True)
            ratios=info['limiter_trace']['power_energy_effort_nm']/request
            np.testing.assert_allclose(ratios,ratios[0],atol=1e-12)
            self.assertLessEqual(info['positive_power_w'],c.POWER_CAP+1e-12)
            self.assertTrue(np.all(abs(effort)<=c.TORQUE_CAP+1e-12))
            self.assertTrue(np.all(abs(base+np.rint(effort*gc.SIGNS*gc.TORQUE_TO_RAW))<=self.caps))
        self.assertGreater(c.sampled_positive_work_j,.02)
        self.assertIsNone(c.energy_remaining)
        self.assertGreater(np.linalg.norm(effort),0)
        # One joint with no headroom scales the complete vector to zero.
        base[0]=self.caps[0]
        effort,_=c.step(request,v,base,self.caps,.004,enabled=True)
        np.testing.assert_array_equal(effort,0)

    def test_fault_and_speed_gate(self):
        c=BalancedLimiter()
        _,info=c.step(np.ones(7),np.full(7,3.1),np.zeros(7),self.caps,.004,enabled=True)
        self.assertEqual(info['reason'],'assist_speed_gate');self.assertIsNone(c.fault)
        effort,info=c.step(np.ones(7),np.zeros(7),np.zeros(7),self.caps,.02,enabled=True)
        self.assertEqual(info['reason'],'timing_fault');np.testing.assert_array_equal(effort,0)

    def test_live_nullspace_input_does_not_create_shaping_and_off_is_baseline(self):
        c=LiveController(controller='balanced');c.previous_time=3.;c.last_eligible=True
        q=gc.HOME.copy();M=np.array(pin.crba(c.model,c.data,q))
        J=handle_jacobian(pin.computeFrameJacobian(c.model,c.data,q,c.frame,pin.LOCAL),c.handle_offset)
        n=np.linalg.svd(J,full_matrices=True)[2][-1]
        c.operator_command('a',3.)
        with patch.object(c.observer,'update',return_value=(M@n,False)):
            command,info=c.step(q,np.zeros(7),3.004,self.caps)
        np.testing.assert_allclose(info['candidate_assist_nm'],0,atol=1e-12)
        np.testing.assert_array_equal(command,info['baseline'])
        self.assertEqual(info['nullspace_dimension'],1)
        c.operator_command('a',3.005)
        command,info=c.step(q,np.zeros(7),3.008,self.caps)
        np.testing.assert_array_equal(command,info['baseline'])

    def test_balanced_session_logs_sent_internal_component_and_work(self):
        now=[0.];bus=Mock();writer=Mock();issued=set()
        velocity=np.zeros(7);velocity[1]=.15;velocity[3]=.08
        calls=[0]
        def state():
            calls[0]+=1
            return gc.SIGNS*gc.HOME,gc.SIGNS*(velocity if calls[0]>1 else np.zeros(7))
        def commands():
            if now[0]>6:return 'q'
            if now[0]>3.1 and 'a' not in issued:issued.add('a');return 'a'
            return None
        bus.state.side_effect=state;bus.present_current_ma.return_value=np.zeros(7)
        bus.read.side_effect=lambda motor,reg:{70:0,98:10,146:25}[reg]
        c=LiveController(controller='balanced')
        def sleep(dt):now[0]+=max(dt,.000001)
        with patch.object(live,'enable'),patch.object(live.time,'monotonic',side_effect=lambda:now[0]), \
             patch.object(live.time,'sleep',side_effect=sleep),patch('builtins.print'), \
             patch.object(c.observer,'update',return_value=(np.ones(7)*.05,False)):
            report=live.run_session(bus,np.zeros(7),self.caps,c,writer,20,commands)
        self.assertGreater(report['assisting_samples'],0)
        model,data=gc.model();M=pin.crba(model,data,gc.HOME)
        J=pin.computeFrameJacobian(model,data,gc.HOME,model.getFrameId('link_7'),pin.LOCAL)
        _,shape=target_shaping(M,J,c.target,c.kappa,c.resist)
        work=0.
        for call in writer.writerow.call_args_list:
            row=call.args[0]
            self.assertEqual(len(row),len(live.sample_columns()))
            sample=dict(zip(live.sample_columns(),row))
            effort=np.array([sample[f'assist_nm_{j}'] for j in gc.IDS])
            leaked=np.array([sample[f'nullspace_output_nm_{j}'] for j in gc.IDS])
            np.testing.assert_allclose(leaked,shape['null_torque_projector']@effort,atol=1e-12)
            work+=sample['positive_power_w']*sample['loop_dt_s']
            self.assertAlmostEqual(work,sample['sampled_assist_work_j'])
        self.assertAlmostEqual(report['sampled_assist_work_j'],work)
        bus.stop.assert_called_once()


if __name__=='__main__':unittest.main()
