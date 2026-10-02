import unittest
import numpy as np
import gravity_comp as gc
from inertia_bounded import BoundedAssistance,simulate


class BoundedTests(unittest.TestCase):
    def setUp(self):
        self.c=BoundedAssistance();self.base=np.zeros(7);self.caps=np.floor(350/gc.MA_PER_RAW)

    def step(self, request, v, **kwargs):
        return self.c.step(np.full(7,request),np.full(7,v),self.base,self.caps,.004,enabled=True,**kwargs)

    def test_stiction_with_nonzero_residual_never_pushes(self):
        for _ in range(1000):effort,info=self.step(.5,0.)
        np.testing.assert_array_equal(effort,0)
        self.assertEqual(info['energy_remaining_j'],self.c.ENERGY_BUDGET)

    def test_trace_preserves_outputs_and_exposes_current_quantization(self):
        traced=BoundedAssistance()
        request=np.full(7,.0001)  # Below one motor-current step.
        for _ in range(120):
            args=(request,np.full(7,.1),self.base,self.caps,.004)
            expected,plain=self.c.step(*args,enabled=True)
            actual,info=traced.step(*args,enabled=True,trace=True)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(info['energy_remaining_j'],plain['energy_remaining_j'])
        stages=info['limiter_trace']
        self.assertTrue(np.all(stages['power_energy_effort_nm']>0))
        np.testing.assert_array_equal(stages['quantized_effort_nm'],0)
        np.testing.assert_array_equal(stages['motion_ramp'],1)
        _,blocked=traced.step(*args,enabled=False,trace=True)
        self.assertTrue(all(np.isnan(values).all() for values in blocked['limiter_trace'].values()))

    def test_trace_distinguishes_motion_direction_and_current_headroom(self):
        for _ in range(150):
            effort,info=self.c.step(np.ones(7),np.full(7,.1),self.caps,self.caps,.004,
                                   enabled=True,trace=True)
        stages=info['limiter_trace']
        self.assertTrue(np.all(stages['direction_effort_nm']>0))
        np.testing.assert_array_equal(stages['headroom_effort_nm'][gc.SIGNS>0],0)
        _,info=self.c.step(np.ones(7),np.full(7,-.1),self.base,self.caps,.004,enabled=True,trace=True)
        np.testing.assert_array_equal(info['limiter_trace']['direction_allowed'],0)
        np.testing.assert_array_equal(info['limiter_trace']['direction_effort_nm'],0)

    def test_actual_quantized_assistance_ramps_in_and_stops_on_reversal(self):
        outputs=[]
        for _ in range(200):
            effort,_=self.step(.2,.1);outputs.append(effort)
        outputs=np.array(outputs)
        self.assertGreater(outputs.max(),0)
        self.assertTrue(np.all(abs(outputs)<=self.c.TORQUE_CAP+1e-12))
        # Quantization can add one current step to the continuous rise bound.
        self.assertTrue(np.all(np.diff(outputs,axis=0)<=self.c.RISE_RATE*.004+1/gc.TORQUE_TO_RAW+1e-12))
        np.testing.assert_array_equal(self.step(.2,-.1)[0],0)
        np.testing.assert_array_equal(self.step(.2,0.)[0],0)

    def test_enable_geometry_and_latched_fault(self):
        for _ in range(100):self.step(.2,.1)
        effort,_=self.c.step(np.ones(7),np.ones(7)*.1,self.base,self.caps,.004)
        np.testing.assert_array_equal(effort,0)
        np.testing.assert_array_equal(self.step(.2,.1,eligible=False)[0],0)
        _,info=self.c.step(np.ones(7),np.ones(7)*.1,self.base,self.caps,.02,enabled=True)
        self.assertEqual(info['reason'],'timing_fault')
        for _ in range(100):np.testing.assert_array_equal(self.step(.2,.1)[0],0)

    def test_overspeed_latched_even_after_velocity_falls(self):
        self.step(.2,.36)
        self.assertEqual(self.c.fault,'overspeed')
        np.testing.assert_array_equal(self.step(.2,.1)[0],0)

    def test_current_cap_keeps_baseline_and_energy_never_refills(self):
        self.base=self.caps.copy() # Positive electrical assistance cannot exceed cap.
        used=0.;previous_energy=self.c.ENERGY_BUDGET
        for _ in range(4000):
            effort,info=self.step(.2,.15)
            total=self.base+np.rint(effort*gc.SIGNS*gc.TORQUE_TO_RAW)
            self.assertTrue(np.all(abs(total)<=self.caps))
            power=float(np.sum(np.maximum(effort*.15,0)))
            self.assertLessEqual(power,self.c.POWER_CAP+1e-12)
            used+=power*.004
            self.assertLessEqual(info['energy_remaining_j'],previous_energy+1e-12)
            previous_energy=info['energy_remaining_j']
        self.assertGreater(used,0)
        self.assertLessEqual(used,self.c.ENERGY_BUDGET+1e-12)
        self.assertAlmostEqual(used,self.c.ENERGY_BUDGET-self.c.energy_remaining)
        # Positive model torque requires positive electrical current except on
        # reversed joints; positive electrical headroom was zero.
        np.testing.assert_array_equal(effort[gc.SIGNS>0],0)

    def test_closed_loop_cannot_make_breakaway_earlier(self):
        baseline,_=simulate(assist=False)
        assisted,trace=simulate(assist=True)
        self.assertEqual(baseline['first_motion_s'],assisted['first_motion_s'])
        self.assertEqual(assisted['stationary_assist_max_nm'],0)
        self.assertEqual(assisted['operator_disabled_assist_max_nm'],0)
        self.assertGreater(np.max(abs(np.array(trace)[:,15:22])),0)

    def test_nonfinite_latches_and_invalid_caps_do_not_command(self):
        for value in (float('nan'),float('inf')):
            c=BoundedAssistance()
            effort,info=c.step(np.ones(7)*value,np.zeros(7),self.base,self.caps,.004,enabled=True)
            np.testing.assert_array_equal(effort,0);self.assertEqual(info['reason'],'invalid_state')
        self.caps[0]=0
        np.testing.assert_array_equal(self.step(.2,.1)[0],0)
        self.assertEqual(self.c.fault,'invalid_baseline_or_caps')

    def test_status_poll_is_one_read_and_visits_all_registers(self):
        from unittest.mock import Mock
        bus=Mock();bus.read.side_effect=lambda motor,reg:{70:0,98:10,146:25}[reg]
        health=gc.StaggeredStatus(0.)
        for i in range(21):
            before=bus.read.call_count
            health.poll(bus,(i+1)*.031)
            self.assertEqual(bus.read.call_count,before+1)
        self.assertEqual({c.args for c in bus.read.call_args_list},
                         {(motor,reg) for motor in gc.IDS for reg in (70,98,146)})

    def test_staggered_status_keeps_fault_checks(self):
        from unittest.mock import Mock
        for reg,bad in [(70,1),(98,0),(146,65)]:
            health=gc.StaggeredStatus(0.)
            health.index=health.registers.index((4,reg))
            bus=Mock();bus.read.return_value=bad
            with self.assertRaises(RuntimeError):health.poll(bus,.04)
