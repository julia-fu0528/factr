"""Guided J4 current-onset measurements at one held pose; no calibration writes.

Inertia shaping is OFF. Other joints use PD pose support plus 0.85 gravity.
J4 uses fixed open-loop current during each stepped sweep. Measurements require
no hand/catch contact. Thresholds include gravity bias and friction; they are
not direct measurements of hand force, pure stiction, or kinetic friction.
"""
import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from factr_diagnostics import DEFAULT_PORT, load_calibration
from factr_inertia import TerminalCommands
from gravity_capture import KP, KD, check_bus_timing
from identify_gravity import enable

J = 3
HZ = 250.


@dataclass(frozen=True)
class Settings:
    ramp_s: float = 2.
    unload_s: float = 3.
    settle_s: float = .8
    settle_timeout_s: float = 12.
    dwell_s: float = .5
    step_raw: int = 1  # 2.69 mA, one J4 current-register step
    negative_max_offset_ma: float = 120.
    onset_displacement: float = np.deg2rad(.25)
    onset_velocity: float = .025
    onset_s: float = .024
    trial_speed: float = .30
    hard_speed: float = 1.
    travel: float = np.deg2rad(2.)
    hard_travel: float = np.deg2rad(5.)
    brake_s: float = .4
    return_s: float = 2.


class Breakaway:
    """Pure state machine; caller commits previous command after a successful write."""
    def __init__(self, anchor, caps, settings=Settings()):
        self.model, self.data = gc.model()
        self.anchor=np.asarray(anchor,dtype=float).copy()
        self.caps=np.asarray(caps,dtype=float)
        self.s=settings
        self.phase='RAMP';self.phase_start=0.;self.previous_time=None
        self.previous_command=np.zeros(7,dtype=int)
        self.window=deque();self.active=None;self.results=[];self.messages=[]
        self.target=self.anchor.copy();self.recovery_start=self.anchor.copy()
        self.recovery_expected=self.anchor.copy()
        self.level=None;self.last_still=None;self.moving_since=None
        self.level_started=0.

    def command(self, text, now):
        if text in ('+','-'):
            if self.phase!='HOLD':return f'Busy: {self.phase}. Wait for HOLD, or x to cancel the measurement.'
            self.active={'trial':len(self.results)+1,'direction':1 if text=='+' else -1,
                         'status':'pending','q_hold_target':self.anchor.tolist()}
            self.phase='SETTLE';self.phase_start=now;self.window.clear()
            return f'Trial {self.active["trial"]}: model direction {text}. Unload your hand; close catch without contact. Settling, then an automatic current sweep.'
        if text=='x' and self.active is not None:
            self.cancel_requested=True
            return 'Cancelling sweep; returning through the small measured displacement.'
        return 'Commands: + or - then Enter = one sweep; x = cancel sweep; q = immediate torque off (support first).'

    def finish(self, status, reason, q, now, **details):
        start=np.array(self.active.get('q_start',q.tolist()))
        self.recovery_expected=start.copy()
        self.active.update(status=status,reason=reason,q_end=q.tolist(),**details)
        self.results.append(self.active);self.active=None
        self.messages.append(f'Saved {status}: {reason}. Braking, then returning slowly to the held pose. Keep catch close.')
        self.phase='BRAKE';self.phase_start=now
        # Preserve the settled PD position error while braking, so gravity
        # bias support does not disappear when J4's sweep current is removed.
        self.target=self.anchor+(q-start)
        self.recovery_start=self.target.copy()

    def interrupted(self, reason):
        if self.active is not None:
            self.active.update(status='rejected',reason=reason)
            self.results.append(self.active);self.active=None

    def step(self, q, v, measured, now):
        q,v,measured=map(lambda a:np.asarray(a,dtype=float),(q,v,measured))
        if any(a.shape!=(7,) or not np.isfinite(a).all() for a in (q,v,measured)) or not np.isfinite(now):
            raise RuntimeError('Invalid state; torque off')
        dt=0. if self.previous_time is None else now-self.previous_time
        if self.previous_time is not None and not 0<dt<=.01:
            raise RuntimeError('PD update gap exceeded 10 ms; torque off')
        self.previous_time=now
        if np.max(abs(v))>self.s.hard_speed:raise RuntimeError('Joint speed exceeded 1 rad/s; torque off')
        if np.max(abs(q-self.anchor))>self.s.hard_travel:raise RuntimeError('Arm moved more than 5 degrees from held pose; torque off')
        margin=np.minimum(q-self.model.lowerPositionLimit,self.model.upperPositionLimit-q)
        if margin.min()<.04:raise RuntimeError('Near modeled joint limit; torque off')
        elapsed=now-self.phase_start
        if getattr(self,'cancel_requested',False):
            self.cancel_requested=False
            if self.active is not None:self.finish('rejected','operator cancelled/contact suspected',q,now)
        if self.phase in ('BASELINE','SWEEP'):
            start=np.array(self.active['q_start'])
            error=abs(q-start)
            if np.max(abs(v))>self.s.trial_speed or error[J]>self.s.travel or np.max(np.delete(error,J))>self.s.travel:
                self.finish('rejected','trial speed/travel/coupled-motion bound reached',q,now)
        if self.phase=='RAMP' and elapsed>=self.s.ramp_s:
            self.phase='HOLD';self.phase_start=now
            self.messages.append('HOLD ready. + then Enter = positive J4 sweep; - then Enter = negative. Inertia shaping OFF.')
        if self.phase=='BRAKE' and now-self.phase_start>=self.s.brake_s:
            self.phase='RETURN';self.phase_start=now
            self.recovery_start=self.target.copy()
        if self.phase=='RETURN':
            u=np.clip((now-self.phase_start)/self.s.return_s,0,1)
            blend=10*u**3-15*u**4+6*u**5
            self.target=self.recovery_start+blend*(self.anchor-self.recovery_start)
            if u>=1 and np.max(abs(q-self.recovery_expected))<np.deg2rad(1) and np.max(abs(v))<.05:
                self.phase='HOLD';self.phase_start=now;self.target=self.anchor.copy()
                self.messages.append('HOLD ready for the other direction. Keep the same pose; no manual repositioning needed.')
            elif now-self.phase_start>self.s.return_s+4:
                raise RuntimeError('Could not regain the held pose; support arm, torque off')

        g=np.array(pin.computeGeneralizedGravity(self.model,self.data,q))
        hold_torque=.85*gc.corrected_gravity(g)+KP*(self.target-q)-KD*v
        hold_raw=hold_torque*gc.SIGNS*gc.TORQUE_TO_RAW
        saturated=abs(hold_raw)>self.caps

        if self.phase=='SETTLE':
            if now-self.phase_start>self.s.settle_timeout_s:
                self.finish('rejected','could not establish stationary unloaded hold',q,now)
            elif now-self.phase_start>=self.s.unload_s:
                self.window.append((now,q.copy(),v.copy(),self.previous_command.copy(),saturated.copy()))
                while len(self.window)>1 and self.window[1][0]<=now-self.s.settle_s:self.window.popleft()
                if self.window and now-self.window[0][0]>=self.s.settle_s:
                    poses=np.array([x[1] for x in self.window]);speeds=np.array([x[2] for x in self.window])
                    if (np.max(np.ptp(poses,axis=0))<np.deg2rad(.15) and np.max(abs(speeds))<.05
                            and not any(np.any(x[4]) for x in self.window)):
                        baseline=int(np.rint(np.median([gc.SIGNS[J]*x[3][J] for x in self.window])))
                        self.active.update(q_start=q.tolist(),baseline_model_raw=baseline,
                                           baseline_model_signed_ma=baseline*gc.MA_PER_RAW[J],
                                           sweep_bound_model_signed_ma=(self.caps[J]*gc.MA_PER_RAW[J] if self.active['direction']>0 else max(-self.caps[J]*gc.MA_PER_RAW[J],baseline*gc.MA_PER_RAW[J]-self.s.negative_max_offset_ma)),
                                           model_gravity_nm=float(g[J]),time_start_s=now)
                        self.level=self.last_still=baseline;self.level_started=now;self.moving_since=None
                        self.phase='BASELINE';self.phase_start=now
                        self.messages.append(f'J4 baseline {baseline*gc.MA_PER_RAW[J]:.2f} model-signed mA. Stay unloaded; sweep stops at onset.')
                        self.messages.append(f'Sweep bound: {self.active["sweep_bound_model_signed_ma"]:.2f} model-signed mA; total-current cap remains enforced.')

        if self.phase in ('BASELINE','SWEEP'):
            # Only the other six joints retain PD support. J4's feedback terms
            # must not fight the sweep or enter its current threshold.
            if np.any(np.delete(saturated,J)):
                self.finish('rejected','another joint reached its support-current cap',q,now)
            else:
                direction=self.active['direction'];qstart=np.array(self.active['q_start'])
                displacement=direction*(q[J]-qstart[J])
                if displacement < -np.deg2rad(.25):
                    self.finish('rejected','opposite-direction drift; baseline/pose not a valid bracket',q,now)
                else:
                    moving=displacement>=self.s.onset_displacement and direction*v[J]>=self.s.onset_velocity
                    self.moving_since=(now if self.moving_since is None else self.moving_since) if moving else None
                    if self.moving_since is not None and now-self.moving_since>=self.s.onset_s:
                        if self.phase=='BASELINE':
                            self.finish('rejected','frozen baseline itself moved; no stationary bracket',q,now)
                        else:
                            # State/current telemetry belongs to the last successful
                            # write. Never attribute onset to a not-yet-sent step.
                            applied=int(gc.SIGNS[J]*self.previous_command[J])
                            if applied!=self.level:raise RuntimeError('Command/state alignment failure')
                            bracket=sorted([self.last_still,applied])
                            self.finish('onset',f'J4 moved in direction {direction:+d}',q,now,
                                        bracket_model_raw=bracket,
                                        bracket_model_signed_ma=(np.array(bracket)*gc.MA_PER_RAW[J]).tolist(),
                                        onset_model_signed_ma=applied*gc.MA_PER_RAW[J],
                                        onset_delta_from_baseline_ma=(applied-self.active['baseline_model_raw'])*gc.MA_PER_RAW[J],
                                        measured_electrical_current_ma_at_confirmation=float(measured[J]),
                                        time_onset_confirmed_s=now)
                    elif now-self.level_started>=self.s.dwell_s:
                        quiet=(np.max(abs(v))<.025 and abs(q[J]-qstart[J])<np.deg2rad(.2))
                        if not quiet:
                            if now-self.level_started>self.s.dwell_s+1:
                                self.finish('rejected','creep without a clean onset; no stationary threshold bracket',q,now)
                        else:
                            self.last_still=self.level
                            next_level=self.level+direction*self.s.step_raw
                            offset=abs(next_level-self.active['baseline_model_raw'])*gc.MA_PER_RAW[J]
                            offset_limited=direction<0 and offset>self.s.negative_max_offset_ma
                            current_limited=abs(next_level)>self.caps[J]
                            if offset_limited or current_limited:
                                reason='total-current ceiling reached' if current_limited else 'negative sweep offset allowance reached'
                                self.finish('censored',f'no onset: {reason}',q,now,
                                            last_still_model_signed_ma=self.last_still*gc.MA_PER_RAW[J])
                            else:
                                self.level=next_level;self.level_started=now;self.phase='SWEEP'

        # Recompute support after a transition to braking/return, avoiding an
        # extra cycle at the onset current. Recovery targets the detected pose first.
        hold_torque=.85*gc.corrected_gravity(g)+KP*(self.target-q)-KD*v
        command=gc.current_command(hold_torque,self.caps)
        if self.phase=='RAMP':command=np.rint(command*min(now/self.s.ramp_s,1)).astype(int)
        if self.phase in ('BASELINE','SWEEP'):command[J]=int(gc.SIGNS[J]*self.level)
        if np.any(abs(command)>self.caps):raise RuntimeError('Internal current bound failure')
        return command,dict(dt=dt,phase=self.phase,direction=0 if self.active is None else self.active['direction'],
                            trial=0 if self.active is None else self.active['trial'],gravity=g,
                            sweep_model_raw=self.level if self.phase in ('BASELINE','SWEEP') else float('nan'))


def summarize(results):
    pairs=[]
    # Adjacent trials (+ then -, or - then +) make one pair. Do not silently
    # replace rejected trials with successes from a different attempt.
    for i in range(0,len(results)-1,2):
        group=results[i:i+2]
        valid=all(r['status']=='onset' for r in group) and {r['direction'] for r in group}=={-1,1}
        reason='both opposite-direction onset measurements required'
        if valid:
            pose_error=float(np.max(abs(np.array(group[0]['q_start'])-group[1]['q_start'])))
            valid=pose_error<=np.deg2rad(1)
            reason='start poses differ by more than 1 degree'
        if not valid:
            pairs.append(dict(trials=[r['trial'] for r in group],status='not_comparable',reason=reason));continue
        plus=next(r for r in group if r['direction']==1);minus=next(r for r in group if r['direction']==-1)
        gap=plus['onset_model_signed_ma']-minus['onset_model_signed_ma']
        pairs.append(dict(trials=[r['trial'] for r in group],status='compared' if gap>0 else 'inconsistent_threshold_order',
                          start_pose_difference_deg=float(np.rad2deg(pose_error)),
                          positive_threshold_ma=plus['onset_model_signed_ma'],negative_threshold_ma=minus['onset_model_signed_ma'],
                          threshold_span_ma=gap))
    return dict(trials=results,pairs=pairs,calibration_modified=False,
                interpretation='Motor-current onset thresholds with unloaded hand. Includes gravity bias, static friction, possible backlash and onset dynamics. Span is not a pure friction coefficient; no symmetry assumption or gravity fit applied.')


def run_session(bus,offsets,caps,writer,save,duration=180.,commands=None,settings=Settings()):
    commands=TerminalCommands() if commands is None else commands
    core=None;exit_reason='starting';saved_count=0
    try:
        bus.validate();raw,raw_v=bus.state();anchor=gc.angles(raw,offsets)
        if np.max(abs(raw_v))>.025:raise RuntimeError('Support arm stationary before startup; torque remains off')
        core=Breakaway(anchor,caps,settings)
        core.step(anchor,gc.SIGNS*raw_v,np.zeros(7),0.) # validate pose before motor enable
        core.previous_time=None
        enable(bus)
        start=time.monotonic();last_print=start;rate_start=start;count=0;warned=False
        health=gc.StaggeredStatus(start)
        print('RAMP: support for 2 seconds, then HOLD. No inertia shaping. + / - then Enter runs one small automatic J4 sweep.',flush=True)
        while True:
            tick=time.monotonic();now=tick-start
            if now>=duration:exit_reason='duration_complete';break
            text=commands()
            if text in ('q','q!'):exit_reason='operator_stop';break
            if text:print(core.command(text,now),flush=True)
            raw,raw_v=bus.state();measured=bus.present_current_ma()
            if time.monotonic()-tick>.01:raise RuntimeError('State read exceeded 10 ms; torque off')
            q,v=gc.angles(raw,offsets),gc.SIGNS*raw_v
            outgoing,info=core.step(q,v,measured,now)
            if time.monotonic()-tick>.01:raise RuntimeError('PD computation exceeded 10 ms; torque off')
            previous=core.previous_command.copy()
            bus.currents(outgoing);core.previous_command=outgoing.copy()
            writer.writerow([now,info['dt'],info['phase'],info['trial'],info['direction'],info['sweep_model_raw'],
                             *q,*v,*info['gravity'],*previous,*outgoing,*measured])
            if len(core.results)!=saved_count:
                save(core.results,'running');saved_count=len(core.results)
            while core.messages:print(core.messages.pop(0),flush=True)
            health.poll(bus,tick);count+=1
            if tick-rate_start>=1:
                rate=count/(tick-rate_start)
                if rate<200:raise RuntimeError(f'PD loop {rate:.1f} Hz below 200; torque off')
                count=0;rate_start=tick
            if tick-last_print>=1:
                print(f't={now:.0f}s {core.phase}; J4={np.rad2deg(q[J]):.2f} deg; applied model-signed current={gc.SIGNS[J]*previous[J]*gc.MA_PER_RAW[J]:.2f} mA',flush=True)
                last_print=tick
            if duration-now<=10 and not warned:
                print('Session ends in 10 seconds: support arm before torque off.',flush=True);warned=True
            time.sleep(max(0,1/HZ-(time.monotonic()-tick)))
    except BaseException as exc:
        exit_reason=str(exc) or type(exc).__name__
        raise
    finally:
        bus.stop()
        if core is not None:
            core.interrupted(exit_reason)
            save(core.results,exit_reason)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default=DEFAULT_PORT)
    parser.add_argument('--calibration',type=Path,default=gc.ROOT/'calibration.json')
    parser.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/j4_breakaway_01')
    parser.add_argument('--duration',type=float,default=180.)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if args.analyze_only:
        results=json.loads((args.output/'report.json').read_text())['trials']
        print(json.dumps(summarize(results),indent=2));return
    if not np.isfinite(args.duration) or not 15<=args.duration<=600:parser.error('duration must be 15..600 seconds')
    saved,offsets=load_calibration(args.calibration,args.port)
    bus=gc.Bus(args.port,4000000)
    try:
        limits=bus.validate()
        latency=Path('/sys/bus/usb-serial/devices')/Path(args.port).resolve().name/'latency_timer'
        if not latency.exists() or latency.read_text().strip()!='1':raise RuntimeError('USB latency_timer must be 1')
        timing=check_bus_timing(bus)
        caps=np.minimum(limits,np.floor(350/gc.MA_PER_RAW))
        print('Secure base. Start in a comfortable supported pose away from joint limits. This test actively holds the arm, ramps J4 current until onset, then returns at most the small trial displacement. Unload your hand during sweeps; keep a close catch without touching. q/Ctrl-C/fault disables torque: support first.',flush=True)
        if input('Press Enter to enable HOLD, or type anything to cancel: ').strip():return
        args.output.mkdir(parents=True,exist_ok=True)
        meta=dict(calibration=saved,settings=asdict(Settings()),effective_current_caps_ma=(caps*gc.MA_PER_RAW).tolist(),
                  gravity_gain=.85,other_joints='PD support using existing KP/KD',inertia_shaping=False,
                  controller_hz=HZ,timing_preflight=timing,calibration_writes=False,
                  positive_sweep_bound='effective J4 total-current ceiling; no separate positive offset allowance',
                  directions='Model coordinates: + increases calibrated J4 angle, - decreases it. Not labeled lift/lower.',
                  telemetry='Present current belongs to previous_command_raw. Threshold bracket uses last confirmed stationary level and actual previously sent onset level. No hand force is measured.')
        (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        def save(results,exit_reason):
            (args.output/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
            (args.output/'report.json').write_text(json.dumps(dict(summarize(results),exit=exit_reason),indent=2)+'\n')
        save([],'starting')
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        with (args.output/'samples.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['time_s','loop_dt_s','phase','trial','direction','sweep_model_raw']+
                            [f'{key}_{j}' for key in ('q_rad','dq_rad_s','model_gravity_nm','previous_command_raw','command_raw','present_current_ma') for j in gc.IDS])
            run_session(bus,offsets,caps,writer,save,args.duration)
    finally:bus.port.closePort()
    print('Torque OFF. Saved',args.output,flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; torque-off cleanup attempted. Partial results saved.')
    except (RuntimeError,ValueError) as exc:print(f'Stopped: {exc}',file=sys.stderr);sys.exit(1)
