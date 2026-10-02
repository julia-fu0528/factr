from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import gravity_comp as gc
import j4_breakaway as jb


class BreakawayTests(unittest.TestCase):
    def setUp(self):
        self.settings=replace(jb.Settings(),ramp_s=.02,unload_s=.01,settle_s=.04,
                              dwell_s=.06,onset_s=.008,brake_s=.02,return_s=.3)
        self.caps=np.floor(350/gc.MA_PER_RAW)
        self.g=np.zeros(7);self.g[3]=.2
        self.base=int(np.rint(.85*self.g[3]*gc.TORQUE_TO_RAW[3]))

    def run_trials(self,directions=(1,-1),threshold=3,settings=None,baseline_drift=False):
        core=jb.Breakaway(gc.HOME,self.caps,self.settings if settings is None else settings)
        q=gc.HOME.copy();v=np.zeros(7);issued=0;history=[]
        with patch.object(jb.pin,'computeGeneralizedGravity',return_value=self.g):
            for i in range(2000):
                t=i*.004
                if core.phase=='HOLD' and issued<len(directions):
                    core.command('+' if directions[issued]>0 else '-',t);issued+=1
                if core.phase in ('BASELINE','SWEEP'):
                    applied=int(gc.SIGNS[3]*core.previous_command[3])
                    velocity=.05*np.sign(applied-self.base) if abs(applied-self.base)>=threshold else 0.
                    if baseline_drift:velocity=.05
                    v[:]=0;v[3]=velocity;q=q+v*.004
                else:
                    new_q=core.target.copy();v=(new_q-q)/.004;q=new_q
                previous=core.previous_command.copy()
                command,info=core.step(q,v,previous*gc.MA_PER_RAW,t)
                core.previous_command=command.copy()
                self.assertTrue(np.all(abs(command)<=core.caps))
                history.append((info,previous.copy(),command.copy(),q.copy()))
                if len(core.results)==len(directions) and core.phase=='HOLD':break
        self.assertEqual(len(core.results),len(directions))
        return core,history

    def test_both_direction_thresholds_and_automatic_small_return(self):
        core,history=self.run_trials()
        for result in core.results:
            self.assertEqual(result['status'],'onset')
            direction=result['direction']
            onset=self.base+3*direction;last=self.base+2*direction
            np.testing.assert_equal(result['bracket_model_raw'],sorted([last,onset]))
            self.assertAlmostEqual(result['onset_model_signed_ma'],onset*2.69)
            self.assertAlmostEqual(result['measured_electrical_current_ma_at_confirmation'],gc.SIGNS[3]*onset*2.69)
        self.assertEqual(core.phase,'HOLD')
        np.testing.assert_allclose(core.target,gc.HOME)
        pair=jb.summarize(core.results)['pairs'][0]
        self.assertEqual(pair['status'],'compared')
        self.assertAlmostEqual(pair['threshold_span_ma'],6*2.69)
        self.assertFalse(jb.summarize(core.results)['calibration_modified'])
        # J4 receives fixed sweep current, not position feedback opposing motion.
        for info,previous,command,q in history:
            if info['phase'] in ('BASELINE','SWEEP'):
                self.assertEqual(command[3],gc.SIGNS[3]*info['sweep_model_raw'])

    def test_censored_is_not_a_threshold(self):
        settings=replace(self.settings,negative_max_offset_ma=5.38)
        core,_=self.run_trials((-1,),threshold=100,settings=settings)
        result=core.results[0]
        self.assertEqual(result['status'],'censored')
        self.assertNotIn('onset_model_signed_ma',result)
        self.assertNotIn('bracket_model_raw',result)

    def test_positive_sweep_can_detect_onset_beyond_old_offset_allowance(self):
        core,_=self.run_trials((1,),threshold=55)
        result=core.results[0]
        self.assertEqual(result['status'],'onset')
        self.assertGreater(result['onset_delta_from_baseline_ma'],120.)

    def test_positive_sweep_stops_at_total_or_lower_hardware_cap(self):
        for cap in (130,100):
            with self.subTest(cap=cap):
                self.caps[3]=cap
                core,_=self.run_trials((1,),threshold=1000)
                result=core.results[0]
                self.assertEqual(result['status'],'censored')
                self.assertIn('total-current ceiling',result['reason'])
                self.assertAlmostEqual(result['last_still_model_signed_ma'],cap*2.69)

    def test_negative_sweep_keeps_original_allowance(self):
        core,_=self.run_trials((-1,),threshold=1000)
        result=core.results[0]
        self.assertEqual(result['status'],'censored')
        self.assertIn('negative sweep offset',result['reason'])
        self.assertAlmostEqual(result['last_still_model_signed_ma'],(self.base-44)*2.69)

    def test_baseline_motion_rejected(self):
        core,_=self.run_trials((1,),baseline_drift=True)
        self.assertEqual(core.results[0]['status'],'rejected')
        self.assertIn('baseline',core.results[0]['reason'])

    def test_pair_requires_same_pose_and_opposite_valid_trials(self):
        core,_=self.run_trials()
        core.results[1]['q_start'][5]+=.05
        self.assertEqual(jb.summarize(core.results)['pairs'][0]['status'],'not_comparable')
        core.results[1]['status']='censored'
        self.assertEqual(jb.summarize(core.results)['pairs'][0]['status'],'not_comparable')

    def test_faults_and_cancel_keep_invalid_trials_out(self):
        core=jb.Breakaway(gc.HOME,self.caps,self.settings)
        with patch.object(jb.pin,'computeGeneralizedGravity',return_value=self.g):
            core.step(gc.HOME,np.zeros(7),np.zeros(7),0)
            with self.assertRaisesRegex(RuntimeError,'gap'):
                core.step(gc.HOME,np.zeros(7),np.zeros(7),.1)
        core=jb.Breakaway(gc.HOME,self.caps,self.settings);core.phase='HOLD'
        core.command('+',0);core.command('x',.001)
        with patch.object(jb.pin,'computeGeneralizedGravity',return_value=self.g):
            core.step(gc.HOME,np.zeros(7),np.zeros(7),.004)
        self.assertEqual(core.results[0]['status'],'rejected')
        self.assertEqual(core.phase,'BRAKE')
        for bad in (np.full(7,np.nan),np.full(7,1.1)):
            with self.assertRaises(RuntimeError):core.step(gc.HOME,bad,np.zeros(7),.008)

    def test_noisy_encoder_motion_does_not_trigger_onset(self):
        core=jb.Breakaway(gc.HOME,self.caps,self.settings)
        core.phase='BASELINE';core.level=core.last_still=self.base
        core.active={'trial':1,'direction':1,'status':'pending','q_start':gc.HOME.tolist(),'baseline_model_raw':self.base}
        core.previous_command[3]=int(gc.SIGNS[3]*self.base)
        with patch.object(jb.pin,'computeGeneralizedGravity',return_value=self.g):
            for i in range(10):
                q=gc.HOME.copy();q[3]+=np.deg2rad(.088)*(i%2)
                v=np.zeros(7);v[3]=.024
                core.step(q,v,np.zeros(7),i*.004)
        self.assertFalse(core.results)

    def test_recovery_preserves_settled_support_error(self):
        core=jb.Breakaway(gc.HOME,self.caps,self.settings)
        settled=gc.HOME.copy();settled[3]-=np.deg2rad(.6)
        onset=settled.copy();onset[3]+=np.deg2rad(.3)
        core.active={'trial':1,'direction':1,'q_start':settled.tolist()}
        core.finish('onset','synthetic',onset,1.)
        np.testing.assert_allclose(core.target-onset,core.anchor-settled,atol=1e-12)
        np.testing.assert_allclose(core.recovery_expected,settled)

    def test_enable_and_read_failures_always_stop(self):
        for failure in ('enable','read'):
            bus=Mock();bus.state.side_effect=[(gc.SIGNS*gc.HOME,np.zeros(7)),RuntimeError('read failed')]
            save=Mock()
            with patch.object(jb,'enable',side_effect=RuntimeError('enable failed') if failure=='enable' else None),patch('builtins.print'):
                with self.assertRaises(RuntimeError):
                    jb.run_session(bus,np.zeros(7),self.caps,Mock(),save,commands=lambda:None)
            bus.stop.assert_called_once();bus.currents.assert_not_called()
            save.assert_called_once()


if __name__=='__main__':unittest.main()
