"""Bounded FACTR assistance core, offline replay and synthetic closed-loop checks.

No serial port is opened. This is not a hardware launch command. Bounds are
engineering trial values, not measured physical ratings or a stability proof.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import gravity_comp as gc
from inertia_shadow import MomentumObserver, analyze as shadow_analyze


class BoundedAssistance:
    """Limit *additional* effort; never replace the gravity baseline.

    Requires an explicit operator enable. Each joint must already be moving;
    this deliberately provides no stationary breakaway/friction cancellation.
    Post-limit torque is quantized through the existing current conversion.
    The optional energy budget bounds sampled positive extra work, not
    continuous-time physical energy/passivity. Offline fixtures enable it;
    the live runner disables it and retains instantaneous power limiting.
    """
    TORQUE_CAP = np.array([.01, .02, .01, .02, .005, .005, .003])
    RISE_RATE = .04  # Nm/s, applies to growth; safety reductions are immediate
    POWER_CAP = .01  # W, sum of positive modeled joint power
    ENERGY_BUDGET = .02  # J per controller instance; no automatic refill
    MAX_DT = .01
    SPEED_START = .035
    SPEED_STOP = .02
    SPEED_FADE = .20
    SPEED_OFF = .35
    MOTION_DWELL = .08
    RAMP_TIME = .25
    TRACE_FIELDS = ('moving', 'direction_allowed', 'moving_for_s', 'motion_ramp',
                    'speed_fade', 'capped_magnitude_nm', 'ramped_magnitude_nm',
                    'faded_magnitude_nm', 'slew_magnitude_nm', 'direction_effort_nm',
                    'headroom_effort_nm', 'power_energy_effort_nm', 'quantized_effort_nm')

    def __init__(self, speed_fade=SPEED_FADE, speed_off=SPEED_OFF, latch_overspeed=True,
                 energy_budget_enabled=True):
        if not np.isfinite([speed_fade, speed_off]).all() or not self.SPEED_START < speed_fade < speed_off:
            raise ValueError('Assistance speeds must be finite, with motion threshold < fade < cutoff')
        self.speed_fade = float(speed_fade)
        self.speed_off = float(speed_off)
        self.latch_overspeed = latch_overspeed
        self.previous = np.zeros(7)
        self.direction = np.zeros(7)
        self.moving_for = np.zeros(7)
        self.energy_remaining = self.ENERGY_BUDGET if energy_budget_enabled else None
        self.sampled_positive_work_j = 0.
        self.fault = None

    def _zero(self, reason):
        self.previous[:] = 0
        self.direction[:] = 0
        self.moving_for[:] = 0
        return np.zeros(7), {'reason': reason, 'energy_remaining_j': self.energy_remaining,
                             'positive_power_w': 0.}

    def step(self, candidate, velocity, baseline_raw, caps_raw, dt, *, enabled=False, eligible=True, trace=False):
        # NaN means this stage was bypassed by an earlier global gate.
        stages = {name: np.full(7, np.nan) for name in self.TRACE_FIELDS} if trace else None
        def zero(reason):
            effort, info = self._zero(reason)
            if trace: info['limiter_trace'] = stages
            return effort, info
        arrays = [np.asarray(a, dtype=float) for a in (candidate, velocity, baseline_raw, caps_raw)]
        if any(a.shape != (7,) or not np.isfinite(a).all() for a in arrays):
            self.fault = 'invalid_state'
        elif not np.isfinite(dt) or not 0 < dt <= self.MAX_DT:
            self.fault = 'timing_fault'
        candidate, velocity, baseline_raw, caps_raw = arrays
        if self.fault:
            return zero(self.fault)
        if np.any(caps_raw <= 0) or np.any(abs(baseline_raw) > caps_raw) or np.any(baseline_raw != np.rint(baseline_raw)):
            self.fault = 'invalid_baseline_or_caps'
            return zero(self.fault)
        if not enabled:
            return zero('operator_disabled')
        if not eligible:
            return zero('observer_or_geometry_gate')
        if np.max(abs(velocity)) >= self.speed_off:
            if self.latch_overspeed:
                self.fault = 'overspeed'
                return zero(self.fault)
            return zero('assist_speed_gate')
        if self.energy_remaining is not None and self.energy_remaining <= 0:
            return zero('energy_exhausted')
        speed = abs(velocity)
        sign = np.sign(velocity)
        same_direction = sign == self.direction
        moving = (speed >= self.SPEED_START) | (same_direction & (speed > self.SPEED_STOP))
        self.moving_for = np.where(moving, np.where(same_direction, self.moving_for+dt, dt), 0.)
        self.direction = np.where(moving, sign, 0.)
        ramp = np.clip((self.moving_for-self.MOTION_DWELL)/self.RAMP_TIME, 0, 1)
        fade = np.clip((self.speed_off-speed)/(self.speed_off-self.speed_fade), 0, 1)
        # No retained assist across a reversal; no opposite-direction effort.
        permitted = moving & (candidate*velocity > 0)
        old = np.where(self.previous*velocity > 0, abs(self.previous), 0.)
        magnitude = np.minimum(abs(candidate), self.TORQUE_CAP)*ramp*fade
        magnitude = np.minimum(magnitude, old+self.RISE_RATE*dt)
        effort = np.where(permitted, sign*magnitude, 0.)
        if trace:
            capped = np.minimum(abs(candidate), self.TORQUE_CAP)
            stages.update(moving=moving.astype(int), direction_allowed=permitted.astype(int),
                          moving_for_s=self.moving_for.copy(), motion_ramp=ramp, speed_fade=fade,
                          capped_magnitude_nm=capped, ramped_magnitude_nm=capped*ramp,
                          faded_magnitude_nm=capped*ramp*fade, slew_magnitude_nm=magnitude.copy(),
                          direction_effort_nm=effort.copy())
        # Preserve the baseline exactly, allocating only available extra current.
        electrical = effort*gc.SIGNS*gc.TORQUE_TO_RAW
        electrical = np.clip(electrical, -caps_raw-baseline_raw, caps_raw-baseline_raw)
        effort = electrical*gc.SIGNS/gc.TORQUE_TO_RAW
        if trace: stages['headroom_effort_nm'] = effort.copy()
        power = float(np.sum(np.maximum(effort*velocity, 0.)))
        available_power = self.POWER_CAP
        if self.energy_remaining is not None:
            available_power = min(available_power, self.energy_remaining/dt)
        if power > available_power:
            effort *= available_power/power
        if trace: stages['power_energy_effort_nm'] = effort.copy()
        # Round added current toward zero, so quantization cannot exceed a cap
        # or manufacture positive work after the energy budget was exhausted.
        ramp_state = effort.copy()
        extra_raw = np.trunc(effort*gc.SIGNS*gc.TORQUE_TO_RAW)
        effort = extra_raw*gc.SIGNS/gc.TORQUE_TO_RAW
        if trace: stages['quantized_effort_nm'] = effort.copy()
        power = float(np.sum(np.maximum(effort*velocity, 0.)))
        self.sampled_positive_work_j += power*dt
        if self.energy_remaining is not None:
            self.energy_remaining = max(0., self.energy_remaining-power*dt)
        self.previous = ramp_state
        reason = 'assisting' if np.any(effort) else 'motion_gate_or_below_current_resolution'
        info = {'reason': reason, 'energy_remaining_j': self.energy_remaining,
                'positive_power_w': power}
        if trace: info['limiter_trace'] = stages
        return effort, info


def replay(log, output, model_gravity_scale=.85):
    """Fixed recorded motion, NOT prediction of motion under assistance."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    shadow_analyze(log, output/'unbounded', model_gravity_scale=model_gravity_scale)
    with Path(log).open() as stream: rows = list(csv.DictReader(stream))
    with (output/'unbounded/shadow.csv').open() as stream: proposals = list(csv.DictReader(stream))
    controller = BoundedAssistance(); reasons = {}; peaks = np.zeros(7); out=[]; last=None
    for row, proposal in zip(rows, proposals):
        t=float(row['time_s'])
        def vector(source, key):return np.array([float(source[f'{key}_{i}']) for i in gc.IDS])
        v=vector(row,'dq_rad_s'); base=vector(row,'command_current_raw'); caps=vector(row,'current_limit_raw')
        if last is None:
            effort, info=controller._zero('initializing')
        else:
            effort, info=controller.step(vector(proposal,'candidate_assist_nm'),v,base,caps,t-last,
                                        enabled=True,eligible=bool(int(proposal['proposal_enabled'])))
        last=t; reasons[info['reason']]=reasons.get(info['reason'],0)+1
        peaks=np.maximum(peaks,abs(effort))
        out.append([t,info['reason'],info['energy_remaining_j'],info['positive_power_w'],*effort])
    with (output/'bounded.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['time_s','reason','energy_remaining_j','positive_power_w']+[f'assist_nm_{i}' for i in gc.IDS]);w.writerows(out)
    report={'mode':'fixed-log replay, not a closed-loop performance prediction','hardware_commands_sent':False,
            'operator_enable':'assumed held throughout replay after observer warmup',
            'peak_added_effort_nm':peaks.tolist(),'reason_counts':reasons,'latched_fault':controller.fault,
            'sampled_positive_work_j':controller.ENERGY_BUDGET-controller.energy_remaining,
            'limitations':LIMITATIONS}
    (output/'bounded_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


LIMITATIONS = [
    'Stationary hand effort is unobservable here; assistance cannot initiate breakaway.',
    'Sampled positive-work bounds are not a proof of physical passivity or stability.',
    'Model/torque conversion and geared-motor inertia are unvalidated.',
    'Per-joint gates and current limits alter the ideal task/nullspace shaping law.',
    'Motion does not prove human intent; enabling requires a real operator control before hardware use.',
    'No hardware interface is provided. This software result does not establish safe hardware gains.',
]


def simulate(*, assist=True, model_bias=0., static_friction=.065, kinetic_friction=.055, inertia_scale=1., hand_peak=.075, seed=0):
    """Seven independent joints with static/kinetic friction and known effort.

    Constant inertia toy plant, no robot kinematic/dynamic coupling. Uses full
    closed-loop feedback: observer sees the previous baseline PLUS assistance.
    Different friction/inertia assumptions are a stress test, not a FACTR fit.
    """
    rng=np.random.default_rng(seed)
    M=np.diag(np.array([.05,.08,.04,.08,.025,.02,.02])*inertia_scale)
    invM=1/np.diag(M);q=np.zeros(7);v=np.zeros(7)
    gravity=np.array([0,.1,0,.15,.015,.025,0.])
    caps=np.floor(350/gc.MA_PER_RAW)
    base=gc.current_command(gravity,caps)
    baseline_effort=base*gc.SIGNS/gc.TORQUE_TO_RAW
    controller=BoundedAssistance();obs=MomentumObserver();previous=baseline_effort.copy()
    dt=.004;peak=0.;stationary_assist=0.;off_assist=0.;motion_times=[];trace=[]
    for k in range(2000):
        t=k*dt
        # Two push directions; known sub-threshold ramp followed by breakaway,
        # a reversal, a hands-off interval, and explicit operator disable.
        if 1<t<2: hand_scalar=.8*hand_peak*(t-1)
        elif 2<=t<3.5: hand_scalar=hand_peak
        elif 4<=t<5.5: hand_scalar=-hand_peak
        else:hand_scalar=0.
        hand=np.full(7,hand_scalar)
        noise=rng.normal(0,.0001,7)
        residual,_=obs.update(M@(v+noise),previous-(gravity+model_bias),dt)
        # Uniform 25% lighter target in this joint-space test fixture. The
        # actual replay uses the complete 7-DOF task/nullspace matrix.
        desired=.25*residual
        enabled=assist and 1<t<6.5
        effort,info=controller.step(desired,v,base,caps,dt,enabled=enabled,eligible=t>1)
        if np.max(abs(v))<controller.SPEED_STOP:stationary_assist=max(stationary_assist,float(np.max(abs(effort))))
        if not enabled:off_assist=max(off_assist,float(np.max(abs(effort))))
        previous=baseline_effort+effort
        # Four substeps reduce numerical artifacts at stiction and reversal.
        for _ in range(4):
            applied=previous-gravity+hand-.10*v
            sticking=(abs(v)<1e-6)&(abs(applied)<=static_friction)
            direction=np.where(abs(v)>1e-6,np.sign(v),np.sign(applied))
            acc=(applied-kinetic_friction*direction)*invM
            next_v=v+acc*dt/4
            crossed=(v*next_v<0)&(abs(applied)<=static_friction)
            v=np.where(sticking|crossed,0.,next_v)
            q+=v*dt/4
        if not motion_times and np.max(abs(v))>.001:motion_times.append(t)
        peak=max(peak,float(np.max(abs(v))))
        trace.append([t,*q,*v,*effort,info['energy_remaining_j']])
    return {'assist_enabled':assist,'model_bias_nm':model_bias,'static_friction_nm':static_friction,
            'kinetic_friction_nm':kinetic_friction,'inertia_scale':inertia_scale,'hand_peak_nm':hand_peak,
            'peak_speed_rad_s':peak,'stationary_assist_max_nm':stationary_assist,
            'operator_disabled_assist_max_nm':off_assist,'first_motion_s':motion_times[0] if motion_times else None,
            'latched_fault':controller.fault,'sampled_positive_work_j':controller.ENERGY_BUDGET-controller.energy_remaining},trace


def validate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,(bias,static,kinetic,mass,hand_peak) in enumerate([(0,.065,.055,1.,.075),(.04,.065,.055,1.,.075),(-.04,.065,.055,1.,.075),
                                                  (0,.09,.035,.5,.095),(0,.06,.04,2.,.095),(.04,.1,.05,1.,.095)]):
        for assist in (False,True):
            report,trace=simulate(assist=assist,model_bias=bias,static_friction=static,kinetic_friction=kinetic,inertia_scale=mass,hand_peak=hand_peak,seed=i)
            if report['stationary_assist_max_nm']!=0 or report['operator_disabled_assist_max_nm']!=0 or report['sampled_positive_work_j']>BoundedAssistance.ENERGY_BUDGET+1e-12:
                raise RuntimeError('Synthetic controller invariant failed')
            results.append(report)
            with (output/f'scenario_{i}_{"assisted" if assist else "baseline"}.csv').open('w',newline='') as stream:
                w=csv.writer(stream);w.writerow(['time_s']+[f'{name}_{j}' for name in ('q_rad','dq_rad_s','assist_nm') for j in gc.IDS]+['energy_remaining_j']);w.writerows(trace)
    report={'hardware_commands_sent':False,'scenarios':results,'limitations':LIMITATIONS,
            'interpretation':'Closed-loop toy friction tests exercise bounds and fault behavior, not FACTR performance or identified inertia.'}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {len(results)} closed-loop synthetic runs to {output}; no hardware commands sent.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    test=sub.add_parser('validate');test.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/inertia_bounded_validation')
    rec=sub.add_parser('replay');rec.add_argument('log',type=Path)
    rec.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/inertia_bounded_replay')
    rec.add_argument('--model-gravity-scale',type=float,default=.85)
    args=parser.parse_args()
    if args.command=='validate':validate(args.output)
    else:replay(args.log,args.output,args.model_gravity_scale)


if __name__=='__main__':main()
