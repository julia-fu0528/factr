import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gravity_comp as gc
import factr_inertia as live


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.caps=np.floor(350/gc.MA_PER_RAW)

    def test_default_is_gravity_only_and_observer_uses_applied_command(self):
        c=live.LiveController();q=gc.HOME.copy();v=np.zeros(7)
        for i in range(901):
            command,info=c.step(q,v,i*.004,self.caps)
            np.testing.assert_array_equal(command,info['baseline'])
            c.previous_command=command.copy()
        self.assertTrue(c.last_eligible)
        np.testing.assert_array_equal(info['extra'],0)
        m,d=gc.model();g=np.array(live.pin.computeGeneralizedGravity(m,d,q))
        applied=np.arange(7,dtype=float)
        c.previous_command=applied
        with patch.object(c.observer,'update',return_value=(np.zeros(7),False)) as observer:
            c.step(q,v,3.604,self.caps)
        np.testing.assert_allclose(observer.call_args.args[1],applied*gc.SIGNS/gc.TORQUE_TO_RAW-.85*g)

    def test_a_toggles_without_resetting_work_and_g_is_unrecognized(self):
        c=live.LiveController();c.last_eligible=True;c.limiter.sampled_positive_work_j=.03
        self.assertIn('ASSISTANCE ON',c.operator_command('a',4))
        self.assertTrue(c.assistance_enabled)
        self.assertIn('ASSISTANCE OFF',c.operator_command('a',5))
        self.assertFalse(c.assistance_enabled)
        self.assertIsNone(c.limiter.energy_remaining)
        self.assertEqual(c.limiter.sampled_positive_work_j,.03)
        c.operator_command('a',5.1)
        self.assertTrue(c.assistance_enabled)
        self.assertIn('Commands:',c.operator_command('g',5.2))
        self.assertTrue(c.assistance_enabled)
        c.operator_command('a',5.3)
        self.assertFalse(c.assistance_enabled)
        c.limiter.fault='timing_fault'
        self.assertIn('locked',c.operator_command('a',6))
        self.assertFalse(c.assistance_enabled)

    def test_toggle_persists_past_three_seconds_until_explicit_off(self):
        c=live.LiveController();q=gc.HOME.copy();v=np.zeros(7);v[3]=.08
        c.previous_time=3.;c.warmup_since=0;c.last_eligible=True
        c.operator_command('a',3.)
        seen_after_three_seconds=False
        with patch.object(c.observer,'update',return_value=(np.ones(7)*.1,False)), \
             patch.object(live,'shaping_matrix',return_value=(np.eye(7)*.1,6,True)):
            for i in range(1,1001):
                command,info=c.step(q,v,3+i*.004,self.caps)
                c.previous_command=command.copy()
                self.assertTrue(np.all(abs(command)<=self.caps))
                self.assertTrue(info['assistance_enabled'])
                if i*.004>3:seen_after_three_seconds |= bool(np.any(info['extra']))
            c.operator_command('a',7.)
            command,info=c.step(q,v,7.004,self.caps)
        self.assertTrue(seen_after_three_seconds)
        self.assertFalse(info['assistance_enabled'])
        np.testing.assert_array_equal(command,info['baseline'])

    def test_assist_speed_switches_off_without_disabling_gravity(self):
        c=live.LiveController();c.previous_time=3.;c.warmup_since=0;c.last_eligible=True
        c.operator_command('a',3.)
        v=np.zeros(7);v[3]=3.1
        with patch.object(c.observer,'update',return_value=(np.zeros(7),False)):
            command,info=c.step(gc.HOME,v,3.004,self.caps)
        self.assertIsNone(c.limiter.fault)
        self.assertFalse(c.assistance_enabled)
        self.assertTrue(info['forced_off'])
        self.assertEqual(info['reason'],'assist_speed_gate')
        np.testing.assert_array_equal(command,info['baseline'])
        command,info=c.step(gc.HOME,np.zeros(7),3.008,self.caps)
        np.testing.assert_array_equal(command,info['baseline'])

    def test_fault_still_disables_assistance(self):
        c=live.LiveController();c.previous_time=3.;c.warmup_since=0;c.last_eligible=True
        c.operator_command('a',3.)
        c.limiter.fault='timing_fault'
        command,info=c.step(gc.HOME,np.zeros(7),3.004,self.caps)
        self.assertFalse(info['assistance_enabled'])
        self.assertTrue(info['forced_off'])
        np.testing.assert_array_equal(command,info['baseline'])

    def test_live_assistance_continues_past_old_energy_budget(self):
        c=live.LiveController();c.previous_time=3.;c.warmup_since=0;c.last_eligible=True
        c.operator_command('a',3.)
        v=np.zeros(7);v[3]=.5
        work=0.
        with patch.object(c.observer,'update',return_value=(np.ones(7)*.1,False)), \
             patch.object(live,'shaping_matrix',return_value=(np.eye(7)*.1,6,True)):
            for i in range(1,1501):
                command,info=c.step(gc.HOME,v,3+i*.004,self.caps)
                c.previous_command=command.copy()
                work+=info['positive_power_w']*info['dt']
                self.assertTrue(info['assistance_enabled'])
                self.assertLessEqual(info['positive_power_w'],c.limiter.POWER_CAP)
                self.assertTrue(np.all(abs(command)<=self.caps))
        self.assertGreater(work,.02)
        self.assertGreater(abs(info['effort'][3]),0)
        self.assertAlmostEqual(work,c.limiter.sampled_positive_work_j)
        self.assertIsNone(c.limiter.energy_remaining)

    def test_session_stop_and_enable_failure_cleanup(self):
        for enable_failure in (False,True):
            bus=Mock();bus.state.return_value=(gc.SIGNS*gc.HOME,np.zeros(7))
            bus.present_current_ma.return_value=np.zeros(7)
            bus.read.side_effect=lambda motor,reg:{70:0,98:10,146:25}[reg]
            now=[0.]
            def clock():return now[0]
            def sleep(dt):now[0]+=max(dt,.000001)
            def commands():return 'q' if now[0]>.04 else None
            c=live.LiveController()
            with patch.object(live,'enable',side_effect=RuntimeError('enable failed') if enable_failure else None), \
                 patch.object(live.time,'monotonic',clock),patch.object(live.time,'sleep',sleep),patch('builtins.print'):
                if enable_failure:
                    with self.assertRaisesRegex(RuntimeError,'enable failed'):
                        live.run_session(bus,np.zeros(7),self.caps,c,Mock(),20,commands)
                else:
                    report=live.run_session(bus,np.zeros(7),self.caps,c,Mock(),20,commands)
                    self.assertEqual(report['exit'],'operator_stop')
                    self.assertGreater(bus.currents.call_count,0)
                    self.assertEqual(report['assisting_samples'],0)
            bus.stop.assert_called_once()

    def test_hard_speed_and_read_fault_cleanup(self):
        bus=Mock();v=np.zeros(7);v[3]=4.1
        bus.state.side_effect=[(gc.SIGNS*gc.HOME,np.zeros(7)),(gc.SIGNS*gc.HOME,gc.SIGNS*v)]
        bus.present_current_ma.return_value=np.zeros(7)
        with patch.object(live,'enable'),patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'J4 speed .*hard cutoff 4'):
                live.run_session(bus,np.zeros(7),self.caps,live.LiveController(),Mock(),20,lambda:None)
        bus.stop.assert_called_once();bus.currents.assert_not_called()

    def test_gravity_only_allows_normal_recorded_speeds(self):
        c=live.LiveController()
        for i,speed in enumerate((.4,1.6,2.6,3.5)):
            v=np.zeros(7);v[3]=speed
            command,info=c.step(gc.HOME,v,3+i*.004,self.caps)
            np.testing.assert_array_equal(command,info['baseline'])
            self.assertIsNone(c.limiter.fault)

    def test_assistance_above_old_cutoff_retains_power_and_current_bounds(self):
        c=live.LiveController();c.previous_time=3.;c.warmup_since=0;c.last_eligible=True
        c.operator_command('a',3.)
        v=np.zeros(7);v[3]=1.6;peak=0.
        with patch.object(c.observer,'update',return_value=(np.ones(7)*.1,False)), \
             patch.object(live,'shaping_matrix',return_value=(np.eye(7)*.1,6,True)):
            for i in range(1,201):
                command,info=c.step(gc.HOME,v,3+i*.004,self.caps)
                peak=max(peak,float(np.max(abs(info['effort']))))
                self.assertLessEqual(float(info['effort']@v),c.limiter.POWER_CAP+1e-12)
                self.assertTrue(np.all(abs(command)<=self.caps))
        self.assertGreater(peak,0)
        self.assertIsNone(c.limiter.fault)

    def test_speed_options_validate_order_and_finite_values(self):
        for kwargs in ({'max_joint_speed':float('nan')}, {'max_joint_speed':5},
                       {'assist_fade_speed':3.1}, {'assist_max_speed':4.1},
                       {'assist_max_speed':float('inf')}):
            with self.assertRaises(ValueError):live.LiveController(**kwargs)
        c=live.LiveController(max_joint_speed=3,assist_fade_speed=1,assist_max_speed=2)
        self.assertEqual(c.hard_speed,3)
        self.assertEqual(c.limiter.speed_off,2)

    def test_unready_enable_does_not_arm(self):
        c=live.LiveController()
        self.assertIn('Not ready',c.operator_command('a',0))
        self.assertFalse(c.assistance_enabled)
        for factor in (float('nan'),0,1.3):
            with self.assertRaises(ValueError):live.LiveController(factor)


    def test_session_toggle_prints_on_off_and_stops_assist_immediately(self):
        now=[0.]; bus=Mock(); writer=Mock(); commands_sent=set()
        def clock():return now[0]
        def sleep(dt):now[0]+=max(dt,.000001)
        def state():
            q=gc.HOME.copy();v=np.zeros(7)
            if now[0]>3.2:
                q[3]+=.08*(now[0]-3.2);v[3]=.08
            return gc.SIGNS*q,gc.SIGNS*v
        def commands():
            if now[0]>7:return 'q'
            if now[0]>=6.6 and 'off' not in commands_sent:
                commands_sent.add('off');return 'a'
            if now[0]>=3.2 and 'a' not in commands_sent:
                commands_sent.add('a');return 'a'
            return None
        bus.state.side_effect=state;bus.present_current_ma.return_value=np.zeros(7)
        bus.read.side_effect=lambda motor,reg:{70:0,98:10,146:25}[reg]
        c=live.LiveController()
        def observer(*_):return np.ones(7)*.1,False
        with patch.object(live,'enable'),patch.object(live.time,'monotonic',clock), \
             patch.object(live.time,'sleep',sleep),patch('builtins.print') as printed, \
             patch.object(c.observer,'update',side_effect=observer), \
             patch.object(live,'shaping_matrix',return_value=(np.eye(7)*.1,6,True)):
            report=live.run_session(bus,np.zeros(7),self.caps,c,writer,20,commands)
        self.assertGreater(report['assisting_samples'],0)
        self.assertIsNone(report['assistance_fault'])
        output='\n'.join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('ASSISTANCE ON',output)
        self.assertIn('ASSISTANCE OFF',output)
        rows=[call.args[0] for call in writer.writerow.call_args_list]
        self.assertTrue(any(6.3<row[0]<6.5 and np.any(row[-7:]) for row in rows))
        logged_work=0.
        for row in rows:
            self.assertEqual(len(row),len(live.sample_columns()))
            sample=dict(zip(live.sample_columns(),row))
            logged_work+=sample['positive_power_w']*sample['loop_dt_s']
            self.assertAlmostEqual(sample['sampled_assist_work_j'],logged_work)
            self.assertAlmostEqual(sample['candidate_assist_nm_4'],.01)
            expected_extra=(sample['command_current_raw_4']-sample['baseline_current_raw_4'])*gc.SIGNS[3]/gc.TORQUE_TO_RAW[3]
            self.assertAlmostEqual(sample['assist_nm_4'],expected_extra)
            if sample['reason']=='operator_disabled':
                self.assertTrue(np.isnan(sample['quantized_effort_nm_4']))
            # Last seven columns always contain actual assistance Nm.
            if row[0]<3.2 or row[0]>6.61:np.testing.assert_array_equal(row[-7:],0)
        self.assertGreater(logged_work,0)
        self.assertAlmostEqual(report['sampled_assist_work_j'],logged_work)
        bus.stop.assert_called_once()


if __name__=='__main__':unittest.main()
