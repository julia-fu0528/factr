import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import gravity_comp as gc
import gravity_capture as cap


class CaptureTests(unittest.TestCase):
    def test_bus_preflight_rejects_return_delay_before_motion(self):
        bus=Mock();bus.read.return_value=250
        with self.assertRaisesRegex(RuntimeError,'Return Delay must be 0'):
            cap.check_bus_timing(bus)
        bus.state.assert_not_called()
        bus.write.assert_not_called()
        bus.currents.assert_not_called()

    def test_bus_preflight_checks_read_time_with_torque_off(self):
        for duration in (.001,.006):
            bus=Mock();bus.read.return_value=0
            times=[t for i in range(100) for t in (i*.01,i*.01+duration)]
            with patch.object(cap.time,'monotonic',side_effect=times),patch('builtins.print'):
                if duration>.004:
                    with self.assertRaisesRegex(RuntimeError,'headroom'):
                        cap.check_bus_timing(bus)
                else:
                    result=cap.check_bus_timing(bus)
                    self.assertAlmostEqual(result['state_read_median_ms'],1.)
            bus.write.assert_not_called()
            bus.currents.assert_not_called()

    def test_current_override_only_changes_j4_and_respects_register(self):
        hardware=np.full(7,1000.)
        caps=cap.capture_current_caps(hardware,450.)
        np.testing.assert_array_equal(np.delete(caps,3),np.delete(np.floor(350/gc.MA_PER_RAW),3))
        self.assertEqual(caps[3],167)
        hardware[3]=130
        self.assertEqual(cap.capture_current_caps(hardware,450.)[3],130)
        for value in (float('nan'),float('inf'),0,901):
            with self.assertRaises(ValueError): cap.capture_current_caps(hardware,value)

    def test_low_actual_loop_rate_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'requires >=200'):
            cap.check_control_rate(50,1.)
        self.assertEqual(cap.check_control_rate(250,1.),250.)
        self.assertIsNone(cap.check_control_rate(2,.01))

    def test_smooth_trajectory_no_steps_and_bounded_speed(self):
        q = gc.HOME.copy()
        end = cap.SETTLE+cap.PERIOD*cap.CYCLES
        ts = np.linspace(0, end, 19001)
        targets = np.array([cap.trajectory(t,q)[0] for t in ts])
        speeds = np.array([cap.trajectory(t,q)[1] for t in ts])
        self.assertLessEqual(np.max(abs(targets-q)), cap.AMP+1e-12)
        self.assertLess(np.max(abs(speeds)), .047)
        self.assertLess(np.max(abs(np.diff(speeds,axis=0)/np.diff(ts)[:,None])), .073)
        np.testing.assert_allclose(np.diff(targets,axis=0)/np.diff(ts)[:,None],
                                   .5*(speeds[1:]+speeds[:-1]), atol=1e-6)
        # Symmetric full periods, center and zero speed at start and finish.
        np.testing.assert_allclose(np.mean(targets-q,axis=0),0,atol=1e-12)
        np.testing.assert_allclose(targets[[0,-1]],np.tile(q,(2,1)))
        np.testing.assert_allclose(speeds[[0,-1]],0)
        self.assertTrue(cap.trajectory(end,q)[2])
        target, _, _, _ = cap.trajectory(cap.SETTLE+cap.PERIOD/4,q,3)
        np.testing.assert_array_equal(np.delete(target,3),np.delete(q,3))

    def test_j3_damping_opposes_overspeed_without_raising_caps(self):
        v=np.zeros(7);v[2]=-.18
        command, saturated=cap.bounded_command(np.zeros(7),gc.HOME,v,gc.HOME,
                                              np.zeros(7),'HOLD',1,np.ones(7)*350)
        self.assertGreater(command[2],0)
        self.assertFalse(saturated)

    @staticmethod
    def samples():
        return [{'q': gc.HOME.tolist(), 'v': [0.]*7, 'g_raw': [.25]*7,
                 'ff': [.2]*7, 'resid': [residual]*7, 'saturated': False}
                for residual in [.06, -.02]*100]

    def test_b601_mean_pd_not_current_or_velocity_filter(self):
        result = cap.summarize_capture(self.samples(), gc.HOME)
        self.assertEqual(result['status'], 'recorded')
        np.testing.assert_allclose(result['resid'], [.02]*7)
        self.assertEqual(len(result['warnings']), 7)
        # Stuck joints are diagnosed but retained like B601, not silently
        # substituted with a current-derived or matched-angle estimate.
        np.testing.assert_allclose(result['zero_velocity_fraction'], np.ones(7))

    def test_saturated_residual_not_fitted_and_legacy_not_mixed(self):
        samples = self.samples()
        samples[0]['saturated'] = True
        result = cap.summarize_capture(samples, gc.HOME)
        self.assertEqual(result['status'], 'rejected')
        report = cap.fit_captures([result, {'status': 'usable', 'paired_bins': []}])
        self.assertEqual(report['candidate_fits'], {})
        self.assertEqual(report['skipped_legacy_captures'], 1)
        self.assertFalse(report['applied'])

    def test_fit_uses_existing_feedforward_plus_pd_scale_and_optional_bias(self):
        captures=[]
        for g in [.1,.2,.3,.4]:
            c=cap.summarize_capture(self.samples(),gc.HOME)
            c.update(g_raw=[g]*7, ff=[.85*g]*7, resid=[.35*g+.03]*7)
            captures.append(c)
        fitted=cap.fit_captures(captures, bias=True)['candidate_fits']['4']
        self.assertAlmostEqual(fitted['scale'],1.2)
        self.assertAlmostEqual(fitted['bias_equivalent_nm'],.03)
        fitted=cap.fit_captures(captures)['candidate_fits']['4']
        self.assertAlmostEqual(fitted['scale'],1.3)
        self.assertEqual(fitted['bias_equivalent_nm'],0)
        self.assertEqual(cap.fit_captures([captures[0]], bias=True)['candidate_fits'],{})

    def test_current_limit_and_zero_release_command(self):
        command, saturated = cap.bounded_command(np.ones(7)*100, gc.HOME, np.zeros(7),
                                                gc.HOME, np.zeros(7), 'HOLD', 1, np.ones(7)*100)
        self.assertTrue(saturated)
        self.assertTrue(np.all(abs(command)<=100))
        command, _ = cap.bounded_command(np.ones(7)*100, gc.HOME, np.zeros(7),
                                         gc.HOME, np.zeros(7), 'RELEASE', 0, np.ones(7)*100)
        self.assertTrue(np.all(command == 0))

    def test_enable_failure_stops(self):
        bus = Mock();bus.state.return_value = (gc.SIGNS*gc.HOME, np.zeros(7))
        with tempfile.TemporaryDirectory() as d, patch.object(cap,'enable',side_effect=RuntimeError('startup')):
            with self.assertRaisesRegex(RuntimeError, 'startup'):
                cap.run_session(bus, np.zeros(7), np.ones(7)*130, 3, Mock(), io.StringIO(), [], Path(d)/'report.json')
        bus.stop.assert_called_once()

    def test_release_keeps_damping_when_gravity_fades(self):
        v=np.zeros(7);v[1]=.1
        command, _ = cap.bounded_command(np.zeros(7), gc.HOME, v, gc.HOME,
                                         np.zeros(7), 'RELEASE', 0, np.ones(7)*350)
        self.assertLess(command[1], 0)

    def test_release_motion_restores_hold_before_hard_speed_abort(self):
        now=[0.];q=gc.HOME.copy();commanded=np.zeros(7)
        def clock(): now[0]+=.00001;return now[0]
        def sleep(dt): now[0]+=dt
        def state():
            v=np.zeros(7)
            if 2.5<now[0]<2.7:
                v[1]=.15;q[1]+=(1/cap.CONTROL_HZ)*v[1]
            return gc.SIGNS*q.copy(),gc.SIGNS*v
        commands=[(2.2,'r\n'),(3.5,'r\n')]
        terminal=Mock();terminal.readline.side_effect=lambda:commands.pop(0)[1]
        def ready(*_):return ([terminal],[],[]) if commands and now[0]>=commands[0][0] else ([],[],[])
        bus=Mock();bus.state.side_effect=state
        bus.currents.side_effect=lambda c:commanded.__setitem__(slice(None),c)
        bus.present_current_ma.side_effect=lambda:commanded*gc.MA_PER_RAW
        bus.read.side_effect=lambda motor,address:{70:0,146:25,64:1,98:10}[address]
        writer=Mock()
        with tempfile.TemporaryDirectory() as directory,patch.object(cap,'enable'), \
             patch.object(cap.time,'monotonic',clock),patch.object(cap.time,'sleep',sleep), \
             patch.object(cap.select,'select',ready),patch.object(cap.sys,'stdin',terminal),patch('builtins.print') as output:
            cap.run_session(bus,np.zeros(7),np.floor(350/gc.MA_PER_RAW),3,writer,io.StringIO(),[],Path(directory)/'report.json')
        rows=[call.args[0] for call in writer.writerow.call_args_list]
        self.assertTrue(any(2.7<r[0]<3.4 and r[1]=='HOLD' for r in rows))
        self.assertTrue(any('RELEASE PAUSED' in str(c) for c in output.call_args_list))
        bus.stop.assert_called_once()

    def test_fault_records_triggering_joint_velocity(self):
        import json
        bus=Mock()
        v=np.zeros(7);v[1]=.6
        bus.state.side_effect=[(gc.SIGNS*gc.HOME,np.zeros(7)),(gc.SIGNS*gc.HOME,gc.SIGNS*v)]
        bus.present_current_ma.return_value=np.zeros(7)
        with tempfile.TemporaryDirectory() as directory,patch.object(cap,'enable'),patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'J2 speed'):
                cap.run_session(bus,np.zeros(7),np.ones(7)*130,3,Mock(),io.StringIO(),[],Path(directory)/'report.json')
            fault=json.loads((Path(directory)/'fault.json').read_text())
            self.assertAlmostEqual(fault['velocity_rad_s'][1],.6)
        bus.stop.assert_called_once()

    def test_capture_overspeed_rejects_and_holds_before_hard_cutoff(self):
        now=[0.]
        def clock(): now[0]+=.00001;return now[0]
        def sleep(dt): now[0]+=dt
        def state():
            v=np.zeros(7)
            if 4 < now[0] < 4.05: v[2]=-.24
            return gc.SIGNS*gc.HOME,gc.SIGNS*v
        commands=[(2.2,'c\n'),(4.5,'q!\n')]
        terminal=Mock();terminal.readline.side_effect=lambda:commands.pop(0)[1]
        def ready(*_): return ([terminal],[],[]) if commands and now[0]>=commands[0][0] else ([],[],[])
        bus=Mock();bus.state.side_effect=state
        bus.present_current_ma.return_value=np.zeros(7)
        bus.read.side_effect=lambda motor,address:{70:0,146:25,64:1,98:10}[address]
        writer=Mock();captures=[]
        with tempfile.TemporaryDirectory() as directory,patch.object(cap,'enable'), \
             patch.object(cap.time,'monotonic',clock),patch.object(cap.time,'sleep',sleep), \
             patch.object(cap.select,'select',ready),patch.object(cap.sys,'stdin',terminal),patch('builtins.print'):
            cap.run_session(bus,np.zeros(7),np.floor(350/gc.MA_PER_RAW),None,writer,
                            io.StringIO(),captures,Path(directory)/'report.json')
            self.assertFalse((Path(directory)/'calib.csv').exists())
        self.assertEqual(len(captures),1)
        self.assertEqual(captures[0]['status'],'rejected')
        self.assertIn('J3 speed',captures[0]['reason'])
        rows=[call.args[0] for call in writer.writerow.call_args_list]
        self.assertTrue(any(row[0]>4.1 and row[1]=='HOLD' for row in rows))
        self.assertEqual(cap.fit_captures(captures)['candidate_fits'],{})
        bus.stop.assert_called_once()

    def test_session_records_b601_capture_and_returns_to_drag(self):
        now = [0.]
        q = gc.HOME.copy(); v = np.zeros(7); commanded = np.zeros(7)
        model, data = gc.model()
        def clock():
            now[0] += .00001
            return now[0]
        def sleep(dt): now[0] += dt
        def state():
            # Deterministic synthetic current-controlled joint. Start exactly
            # supported during ramp; afterwards gravity follows the same model.
            if 2 < now[0] < 23:
                gravity = np.array(cap.pin.computeGeneralizedGravity(model,data,q))
                motor_tau = commanded*gc.SIGNS/gc.TORQUE_TO_RAW
                acc = (motor_tau-.85*gravity-.08*v)/.03
                v[:] += (1/cap.CONTROL_HZ)*acc
                q[:] += (1/cap.CONTROL_HZ)*v
            elif now[0] >= 23:
                # Operator supports the assembly during the requested release.
                v[:] = 0
            return gc.SIGNS*q.copy(), gc.SIGNS*v.copy()
        commands = [(2.2,'c\n'),(22.,'h\n'),(23.,'r\n')]
        terminal = Mock()
        terminal.readline.side_effect = lambda: commands.pop(0)[1]
        def ready(*_):
            return ([terminal],[],[]) if commands and now[0]>=commands[0][0] else ([],[],[])
        bus = Mock();bus.state.side_effect=state
        bus.currents.side_effect=lambda x:commanded.__setitem__(slice(None),x)
        bus.present_current_ma.side_effect=lambda:commanded*gc.MA_PER_RAW
        bus.read.side_effect=lambda motor,address:{70:0,146:25,64:1,98:10}[address]
        writer = Mock();outcomes=io.StringIO();captures=[]
        with tempfile.TemporaryDirectory() as directory, patch.object(cap,'enable'), \
             patch.object(cap.time,'monotonic',clock),patch.object(cap.time,'sleep',sleep), \
             patch.object(cap.select,'select',ready),patch.object(cap.sys,'stdin',terminal),patch('builtins.print'):
            cap.run_session(bus,np.zeros(7),np.floor(350/gc.MA_PER_RAW),None,writer,outcomes,captures,Path(directory)/'report.json')
        phases=[c.args[0][1] for c in writer.writerow.call_args_list]
        self.assertIn('CAPTURE',phases)
        endcap=max(i for i,p in enumerate(phases) if p=='CAPTURE')
        self.assertEqual(phases[endcap+1], 'DRAG')
        self.assertIn('HOLD',phases[endcap+1:])
        self.assertEqual(phases[-1],'RELEASE')
        self.assertEqual(len(captures),1)
        self.assertEqual(captures[0]['status'], 'recorded')
        self.assertEqual(captures[0]['method'], cap.METHOD)
        # Two complete periods at the configured control rate, without the settling samples.
        self.assertGreaterEqual(captures[0]['samples'], int(16*cap.CONTROL_HZ*.98))
        self.assertLessEqual(captures[0]['samples'], int(16*cap.CONTROL_HZ*1.01))
        bus.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
