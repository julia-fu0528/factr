"""Experimental FACTR gravity + explicitly enabled bounded inertia assistance.

Starts gravity only. 'a' toggles assistance on/off.
'q'/Ctrl-C or hard faults disable motor torque.
Balanced mode targets Cartesian inertia with natural nullspace inertia.
No calibrated friction or breakaway compensation is included.
"""
import argparse
import csv
import json
import select
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from gravity_capture import check_bus_timing
from factr_diagnostics import DEFAULT_PORT, load_calibration
from identify_gravity import enable
from inertia_bounded import BoundedAssistance
from inertia_shadow import MomentumObserver, shaping_matrix
from balanced_drag import BalancedLimiter, target_shaping, handle_jacobian, DEFAULT_MASS, DEFAULT_INERTIA

HZ = 250.
WARMUP_SECONDS = 3.
HARD_SPEED = 4.  # Same hard cutoff as gravity_comp.py; not a target speed.
ASSIST_SPEED_FADE = 2.
ASSIST_SPEED_OFF = 3.
TRACE_SCALARS = ('observer_warm', 'geometry_ok', 'jacobian_rank', 'joint_margin_rad',
                 'nullspace_dimension', 'geometry_ratio', 'geometry_fade', 'negative_power_w')
SHAPING_VECTORS = ('candidate_assist_nm', 'residual_used_nm', 'raw_task_gain', 'clipped_task_gain',
                   'nullspace_input_nm', 'nullspace_output_nm')
TRACE_VECTORS = SHAPING_VECTORS + BoundedAssistance.TRACE_FIELDS


def sample_columns():
    return (['time_s','loop_dt_s','reason','assistance_enabled','eligible','sampled_assist_work_j','positive_power_w','baseline_saturated']+
            [f'current_limit_raw_{i}' for i in gc.IDS]+
            [f'{name}_{i}' for name in ('q_rad','dq_rad_s','model_gravity_nm','residual_nm',
             'baseline_current_raw','command_current_raw','present_current_ma') for i in gc.IDS]+
            list(TRACE_SCALARS)+[f'{name}_{i}' for name in TRACE_VECTORS for i in gc.IDS]+
            [f'assist_nm_{i}' for i in gc.IDS])


class LiveController:
    """Pure computation; the session alone owns hardware access."""
    def __init__(self, factor=1.1, model_gravity_scale=.85, max_joint_speed=HARD_SPEED,
                 assist_fade_speed=ASSIST_SPEED_FADE, assist_max_speed=ASSIST_SPEED_OFF,
                 controller='uniform', target_mass=DEFAULT_MASS, target_inertia=DEFAULT_INERTIA,
                 kappa=.5, resist=.25, handle_offset=(0., 0., 0.), torque_rise_rate=.04,
                 assist_limits='bounded'):
        if not np.isfinite(factor) or not 1 <= factor <= 1.25:
            raise ValueError('factor must be within 1..1.25')
        if not np.isfinite(model_gravity_scale) or not .1 <= model_gravity_scale <= 1.2:
            raise ValueError('model gravity scale must be within 0.1..1.2')
        if (not np.isfinite([max_joint_speed, assist_fade_speed, assist_max_speed]).all()
                or not BoundedAssistance.SPEED_START < assist_fade_speed < assist_max_speed <= max_joint_speed <= 4.):
            raise ValueError('Speeds require 0.035 < assist fade < assist cutoff <= hard cutoff <= 4 rad/s')
        self.model,self.data=gc.model()
        self.frame=self.model.getFrameId('link_7')
        if controller not in ('uniform','balanced'):raise ValueError('Unknown controller')
        self.controller=controller
        if assist_limits not in ('bounded','current-only'):
            raise ValueError('Unknown assistance limit profile')
        if assist_limits=='current-only' and controller!='balanced':
            raise ValueError('current-only limits require balanced controller')
        self.assist_limits=assist_limits
        self.target=np.r_[target_mass,target_inertia].astype(float)
        self.handle_offset=np.asarray(handle_offset,dtype=float)
        if self.target.shape!=(6,) or not np.isfinite(self.target).all() or np.any(self.target<=0):
            raise ValueError('Specify three positive finite target masses and three rotational inertias')
        if self.handle_offset.shape!=(3,) or not np.isfinite(self.handle_offset).all():
            raise ValueError('Handle offset must contain three finite distances in meters')
        if not np.isfinite([kappa,resist]).all() or not 0<=kappa<=1 or not 0<=resist<=.5:
            raise ValueError('Live balanced gains require 0<=kappa<=1 and 0<=resist<=0.5')
        self.kappa=kappa;self.resist=resist
        # One motor-current step: numerical deadband, not a measured friction fit.
        self.residual_deadband=1/gc.TORQUE_TO_RAW
        self.factor=factor;self.gravity_scale=model_gravity_scale
        self.hard_speed=max_joint_speed
        self.observer=MomentumObserver()
        self.limiter=BoundedAssistance(assist_fade_speed,assist_max_speed,latch_overspeed=False,
                                      energy_budget_enabled=False)
        if controller=='balanced':self.limiter=BalancedLimiter(assist_fade_speed,assist_max_speed,
                                                              additional_limits=assist_limits=='bounded')
        if not np.isfinite(torque_rise_rate) or not 0<torque_rise_rate<=.4:
            raise ValueError('Torque rise rate must be finite and within (0, 0.4] Nm/s')
        self.limiter.RISE_RATE=float(torque_rise_rate)
        self.previous_command=np.zeros(7)
        self.previous_time=None;self.warmup_since=0.;self.assistance_enabled=False
        self.last_eligible=False

    def operator_command(self, command, now):
        if command=='a' and self.assistance_enabled:
            self.assistance_enabled=False
            self.limiter._zero('operator_disabled')
            return 'ASSISTANCE OFF | gravity compensation remains ON.'
        if command!='a':return 'Commands: a=toggle assistance ON/OFF, q=torque off.'
        if self.limiter.fault:return f'ASSISTANCE OFF (locked): {self.limiter.fault}. Gravity remains ON; support and q to finish.'
        if not self.last_eligible:return 'ASSISTANCE OFF. Not ready: wait for warmup and a pose away from modeled limits/singularity.'
        self.assistance_enabled=True
        return 'ASSISTANCE ON | no trial timer. Keep holding the handle; a + Enter turns assistance OFF.'

    def step(self, q, v, now, caps):
        if not np.isfinite(np.r_[q,v,now]).all():raise RuntimeError('Nonfinite state; torque off')
        if np.max(abs(v))>self.hard_speed:
            fastest=int(np.argmax(abs(v)))
            raise RuntimeError(f'J{fastest+1} speed {v[fastest]:+.3f} rad/s exceeds hard cutoff '
                               f'{self.hard_speed:g}; torque off')
        margin=np.minimum(q-self.model.lowerPositionLimit,self.model.upperPositionLimit-q)
        if margin.min()<.02:raise RuntimeError('Joint limit margin reached; torque off')
        dt=0. if self.previous_time is None else now-self.previous_time
        if self.previous_time is not None and (dt<=0 or dt>.1):raise RuntimeError('Control timing invalid or stalled >100 ms; torque off')
        data=self.data;model=self.model
        M=np.array(pin.crba(model,data,q));M=(M+M.T)/2
        C=np.array(pin.computeCoriolisMatrix(model,data,q,v))
        gravity=np.array(pin.computeGeneralizedGravity(model,data,q))
        modeled_gravity=self.gravity_scale*gravity
        motor=self.previous_command*gc.SIGNS/gc.TORQUE_TO_RAW
        residual,reset=self.observer.update(M@v,motor-modeled_gravity+C.T@v,dt)
        if reset:self.warmup_since=now
        if self.controller=='balanced':
            J=handle_jacobian(pin.computeFrameJacobian(model,data,q,self.frame,pin.LOCAL),self.handle_offset)
            K,shape=target_shaping(M,J,self.target,self.kappa,self.resist)
            rank,geometry_ok=shape['rank'],shape['geometry_ok']
            # Remove internal effort BEFORE the nonlinear deadband, so a pure
            # nullspace input cannot be converted into task assistance by it.
            task_residual=residual-shape['null_torque_projector']@residual
            used=np.sign(task_residual)*np.maximum(abs(task_residual)-self.residual_deadband,0.)
        else:
            J=np.array(pin.computeFrameJacobian(model,data,q,self.frame,pin.LOCAL_WORLD_ALIGNED))
            K,rank,geometry_ok=shaping_matrix(M,J,self.factor)
            shape=dict(null_dimension=7-rank,geometry_ratio=float('nan'),geometry_fade=float(geometry_ok),
                       raw_gains=np.full(7,np.nan),clipped_gains=np.full(7,np.nan),null_torque_projector=np.zeros((7,7)))
            used=residual
        candidate=K@used
        warm=now-self.warmup_since>=WARMUP_SECONDS
        eligible=warm and geometry_ok and margin.min()>=.15
        self.last_eligible=eligible
        # Existing supported baseline: all joint gains .85, damping .01.
        baseline_requested=min(now/2,1.)*(.85*gc.corrected_gravity(gravity)-.01*v)
        baseline=gc.current_command(baseline_requested,caps)
        saturated=bool(np.any(abs(baseline_requested*gc.TORQUE_TO_RAW)>caps))
        if self.previous_time is None:
            effort,info=self.limiter._zero('initializing')
            info['limiter_trace']={name:np.full(7,np.nan) for name in self.limiter.TRACE_FIELDS}
        else:
            effort,info=self.limiter.step(candidate,v,baseline,caps,dt,
                                         enabled=self.assistance_enabled,eligible=eligible and not saturated,trace=True)
        forced_off = self.assistance_enabled and (
            info['reason']=='assist_speed_gate' or self.limiter.fault is not None)
        if forced_off:
            self.assistance_enabled=False
            self.limiter._zero(info['reason'])
            effort=np.zeros(7)
            info['positive_power_w']=0.

        extra=np.rint(effort*gc.SIGNS*gc.TORQUE_TO_RAW).astype(int)
        command=baseline+extra
        if np.any(abs(command)>caps):raise RuntimeError('Internal current bound failure; torque off')
        self.previous_time=now
        # Commit previous_command only AFTER a successful hardware write.
        return command,dict(info,dt=dt,rank=rank,eligible=eligible,assistance_enabled=self.assistance_enabled,forced_off=forced_off,
                            baseline=baseline,extra=extra,effort=effort,residual=residual,
                            gravity=gravity,baseline_saturated=saturated,
                            candidate_assist_nm=candidate,observer_warm=int(warm),
                            geometry_ok=int(geometry_ok),jacobian_rank=rank,joint_margin_rad=float(margin.min()),
                            nullspace_dimension=shape['null_dimension'],geometry_ratio=shape['geometry_ratio'],
                            geometry_fade=shape['geometry_fade'],residual_used_nm=used,
                            raw_task_gain=shape['raw_gains'],clipped_task_gain=shape['clipped_gains'],
                            nullspace_input_nm=shape['null_torque_projector']@residual,
                            nullspace_output_nm=shape['null_torque_projector']@effort,
                            null_projector=shape['null_torque_projector'])


class TerminalCommands:
    def __call__(self):
        if not select.select([sys.stdin],[],[],0)[0]:return None
        line=sys.stdin.readline()
        if not line:raise RuntimeError('Terminal disconnected; torque off')
        return line.strip().lower()


def run_session(bus,offsets,caps,controller,writer,duration,commands=None):
    commands=TerminalCommands() if commands is None else commands
    previous_fault=None
    statistics={'samples':0,'assisting_samples':0,'peak_assist_nm':np.zeros(7),
                'min_window_hz':None,'exit':'starting','sampled_assist_work_j':0.}
    operation='startup validation'
    try:
        # This final check occurs immediately before enabling, after Enter.
        bus.validate()
        raw,rawv=bus.state();q=gc.angles(raw,offsets)
        if np.max(abs(q-gc.HOME))>.35:raise RuntimeError('Start in docs/calibration_pose.png; torque remains off')
        if np.max(abs(rawv))>.025:raise RuntimeError('Support the arm stationary before startup')
        operation='motor enable'
        enable(bus)
        start=time.monotonic();rate_start=start;count=0;last_print=start
        health=gc.StaggeredStatus(start);ready_printed=False;warned=False
        print(f'Speed settings: assistance fades above {controller.limiter.speed_fade:g}, '
              f'assistance turns off at {controller.limiter.speed_off:g}, '
              f'all torque shuts off above {controller.hard_speed:g} rad/s.',flush=True)
        print('RAMP: support the arm for 2 seconds. Gravity gain .85 on all joints; assistance OFF.',flush=True)
        if controller.assist_limits=='current-only':
            print('CURRENT-ONLY output limits: extra torque, positive-power, and slew caps OFF. Total-current envelope, task-gain limits, activation ramp, speed and fault checks remain ON.',flush=True)
        else:
            print('No cumulative assistance energy cutoff; instantaneous power limit remains active.',flush=True)
            print(f'Extra torque rise-rate setting: {controller.limiter.RISE_RATE:g} Nm/s; torque/current/power caps remain active.',flush=True)
        if controller.controller=='balanced':
            print(f'BALANCED DRAG: target mass xyz={controller.target[:3]} kg; rotational inertia xyz={controller.target[3:]} kg m^2.',flush=True)
            print(f'Handle point: link_7 local offset {controller.handle_offset} m. Internal DOF: natural inertia, zero shaping gain. kappa={controller.kappa:g}, resist={controller.resist:g}.',flush=True)
            print('Signed assist/resist output; 2 s activation ramp. No per-joint motion dwell or friction compensation.',flush=True)
        while True:
            tick=time.monotonic();now=tick-start
            if now>=duration:
                statistics['exit']='duration_complete';break
            command_text=commands()
            if command_text in ('q','q!'):
                statistics['exit']='operator_stop';break
            if command_text:
                print(controller.operator_command(command_text,now),flush=True)
            operation='state sync read'
            raw,rawv=bus.state()
            if time.monotonic()-tick>.01:
                # No stale assist is ever sent. Gravity uses this sample only
                # below the separate hard timeout.
                controller.limiter.fault='stale_state_read'
            if time.monotonic()-tick>.1:raise RuntimeError('State read stalled >100 ms; torque off')
            q,v=gc.angles(raw,offsets),gc.SIGNS*rawv
            operation='decode current telemetry'
            measured=bus.present_current_ma()
            count+=1
            if tick-rate_start>=1:
                hz=count/(tick-rate_start)
                statistics['min_window_hz']=hz if statistics['min_window_hz'] is None else min(hz,statistics['min_window_hz'])
                if hz<200:controller.limiter.fault=f'loop_rate_below_200_Hz ({hz:.1f})'
                rate_start=tick;count=0
            operation='controller computation'
            outgoing,info=controller.step(q,v,now,caps)
            if time.monotonic()-tick>.01:
                controller.limiter.fault='slow_control_computation'
                controller.limiter._zero(controller.limiter.fault)
                was_enabled=controller.assistance_enabled
                controller.assistance_enabled=False
                outgoing=info['baseline'];info.update(reason=controller.limiter.fault,extra=np.zeros(7),effort=np.zeros(7),positive_power_w=0.,negative_power_w=0.,nullspace_output_nm=np.zeros(7),assistance_enabled=False,forced_off=was_enabled)
            operation='goal-current sync write'
            bus.currents(outgoing)
            controller.previous_command=outgoing.copy()
            statistics['sampled_assist_work_j']+=info['positive_power_w']*info['dt']
            trace_values=[info.get(name,0.) for name in TRACE_SCALARS]
            trace_values.extend(value for name in SHAPING_VECTORS for value in info[name])
            trace_values.extend(value for name in controller.limiter.TRACE_FIELDS for value in info['limiter_trace'][name])
            writer.writerow([now,info['dt'],info['reason'],int(info['assistance_enabled']),int(info['eligible']),
                             statistics['sampled_assist_work_j'],info['positive_power_w'],int(info['baseline_saturated']),*caps,*q,*v,*info['gravity'],
                             *info['residual'],*info['baseline'],*outgoing,*measured,*trace_values,*info['effort']])
            statistics['samples']+=1
            statistics['assisting_samples']+=int(np.any(info['effort']))
            statistics['peak_assist_nm']=np.maximum(statistics['peak_assist_nm'],abs(info['effort']))
            operation='hardware status read'
            health.poll(bus,tick)
            if info['reason']=='assist_speed_gate':
                print(f'ASSISTANCE OFF: speed reached {controller.limiter.speed_off:g} rad/s. '
                      'Gravity remains ON. Slow down before enabling again.',flush=True)
            if now>=WARMUP_SECONDS and not ready_printed:
                mode='ON' if controller.assistance_enabled else 'OFF'
                print(f'ASSISTANCE {mode} | gravity ON. a + Enter toggles assistance; support before q.',flush=True)
                ready_printed=True
            if controller.limiter.fault and controller.limiter.fault!=previous_fault:
                print(f'ASSISTANCE OFF (locked): {controller.limiter.fault}. Gravity stays active. Support arm, then q.',flush=True)
                previous_fault=controller.limiter.fault
            if duration-now<=10 and not warned:
                print('Session ends in 10 seconds: support the arm before torque switches off.',flush=True);warned=True
            if tick-last_print>=1:
                mode='ON' if controller.assistance_enabled else 'OFF'
                print(f't={now:.0f}s ASSISTANCE {mode} | {info["reason"]}; extra current (mA)={np.round(info["extra"]*gc.MA_PER_RAW,1)}',flush=True)
                last_print=tick
            time.sleep(max(0.,1/HZ-(time.monotonic()-tick)))
    except BaseException as exc:
        statistics['exit']=str(exc) or type(exc).__name__
        statistics['failed_operation']=operation
        raise
    finally:
        bus.stop()
        statistics['peak_assist_nm']=statistics['peak_assist_nm'].tolist()
        statistics['assistance_fault']=controller.limiter.fault
        # Caller can save this even after an exception.
        controller.session_statistics=statistics
    return statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default=DEFAULT_PORT)
    parser.add_argument('--calibration',type=Path,default=gc.ROOT/'calibration.json')
    parser.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/inertia_live_01')
    parser.add_argument('--duration',type=float,default=120.)
    parser.add_argument('--controller',choices=('balanced','uniform'),default='balanced')
    parser.add_argument('--factor',type=float,help='Uniform mode only: task inertia reduction factor, 1..1.25')
    parser.add_argument('--target-mass',type=float,nargs=3,default=DEFAULT_MASS,metavar=('MX','MY','MZ'),help='Balanced handle-local x/y/z target masses in kg')
    parser.add_argument('--target-inertia',type=float,nargs=3,default=DEFAULT_INERTIA,metavar=('IX','IY','IZ'),help='Balanced handle-local x/y/z rotational inertias in kg m^2')
    parser.add_argument('--kappa',type=float,default=.5,help='Balanced task gain upper bound, 0..1; not a uniform factor')
    parser.add_argument('--resist',type=float,default=.25,help='Balanced resisting task gain magnitude bound, 0..0.5')
    parser.add_argument('--handle-offset',type=float,nargs=3,default=(0.,0.,0.),metavar=('X','Y','Z'),help='Handle point offset from link_7 origin in link_7 axes, meters')
    parser.add_argument('--torque-rise-rate',type=float,default=.04,help='Additional torque magnitude rise-rate bound in Nm/s, >0 and <=0.4; default 0.04')
    parser.add_argument('--assist-limits',choices=('bounded','current-only'),default='bounded',help='Balanced only: current-only bypasses extra torque/power/slew caps; total-current envelope, gain limits, ramp and fault checks remain')
    parser.add_argument('--model-gravity-scale',type=float,default=.85,help='Observer assumption, not a fitted physical parameter')
    parser.add_argument('--max-joint-speed',type=float,default=HARD_SPEED,help='Hard torque-off threshold in rad/s, at most 4 (same default as original gravity controller)')
    parser.add_argument('--assist-fade-speed',type=float,default=ASSIST_SPEED_FADE,help='Begin reducing assistance above this speed in rad/s')
    parser.add_argument('--assist-max-speed',type=float,default=ASSIST_SPEED_OFF,help='Turn assistance off at this speed in rad/s; gravity stays on')
    args=parser.parse_args()
    if args.controller=='balanced' and args.factor is not None:parser.error('--factor applies only with --controller uniform; balanced mode uses --target-mass/--target-inertia')
    if not np.isfinite(args.duration) or not 15<=args.duration<=600:parser.error('duration must be 15..600 seconds')
    try:controller=LiveController(1.1 if args.factor is None else args.factor,args.model_gravity_scale,args.max_joint_speed,
                                  args.assist_fade_speed,args.assist_max_speed,args.controller,
                                  args.target_mass,args.target_inertia,args.kappa,args.resist,args.handle_offset,args.torque_rise_rate,args.assist_limits)
    except ValueError as exc:parser.error(str(exc))
    saved,offsets=load_calibration(args.calibration,args.port)
    bus=gc.Bus(args.port,4000000)
    prepared=False
    try:
        limits=bus.validate()
        latency=Path('/sys/bus/usb-serial/devices')/Path(args.port).resolve().name/'latency_timer'
        if not latency.exists() or latency.read_text().strip()!='1':raise RuntimeError('USB latency_timer must be 1')
        timing=check_bus_timing(bus)
        caps=np.minimum(limits,np.floor(350/gc.MA_PER_RAW))
        print('Experimental live assistance. Secure base; start in reference pose and keep a catch ready. '
              'Support startup/shutdown. Motor torque is OFF until you press Enter. No automatic arm motion is requested.',flush=True)
        if input('Press Enter to enable gravity only, or type anything to cancel: ').strip():return
        args.output.mkdir(parents=True,exist_ok=True)
        metadata={'calibration':saved,'controller':args.controller,'factor':controller.factor if args.controller=='uniform' else None,'model_gravity_scale':args.model_gravity_scale,
                  'target_mass_kg':controller.target[:3].tolist(),'target_rotational_inertia_kg_m2':controller.target[3:].tolist(),
                  'kappa':args.kappa,'resist':args.resist,'handle_frame':'link_7 local axes','handle_offset_m':controller.handle_offset.tolist(),
                  'nullspace_policy':'Natural modeled inertia; zero internal shaping gain. Scalar limits preserve it before current quantization; logged output leakage measures rounding.',
                  'residual_deadband_nm':controller.residual_deadband.tolist() if args.controller=='balanced' else [0.]*7,
                  'friction_compensation':False,
                  'gravity_gain_all_joints':.85,'hz_target':HZ,'duration_s':args.duration,
                  'effective_current_caps_ma':(caps*gc.MA_PER_RAW).tolist(),'timing_preflight':timing,
                  'assist_limits':args.assist_limits,
                  'assist_torque_caps_nm':controller.limiter.TORQUE_CAP.tolist() if args.assist_limits=='bounded' else None,
                  'torque_rise_rate_nm_s':controller.limiter.RISE_RATE if args.assist_limits=='bounded' else None,
                  'assistance_control':'a_enter_toggle','power_cap_w':controller.limiter.POWER_CAP if args.assist_limits=='bounded' else None,
                  'hard_speed_rad_s':controller.hard_speed,
                  'assist_fade_speed_rad_s':controller.limiter.speed_fade,
                  'assist_cutoff_speed_rad_s':controller.limiter.speed_off,
                  'energy_budget_j':None,'energy_budget_enabled':False,
                  'sample_schema_version':4,
                  'limiter_trace_fields':list(BoundedAssistance.TRACE_FIELDS),
                  'trace_note':'NaN means bypassed/not applicable. Balanced moving/direction_allowed/moving_for_s are inapplicable; motion_ramp is global activation ramp. raw/clipped_task_gain entries 1..6 are task eigenmodes (not joints), entry 7 is zero null gain. Magnitudes unsigned; efforts model-signed. assist_nm is final sent assistance, including late overrides. Uniform nullspace diagnostics are not computed.',
                  'note':'Experimental; no continuous-time passivity proof, hand force sensing, or physical inertia validation. Quantities use empirical motor conversion.'}
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');prepared=True
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('Terminated')))
        with (args.output/'samples.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(sample_columns())
            run_session(bus,offsets,caps,controller,writer,args.duration)
    finally:
        bus.port.closePort()
        if prepared:
            report=getattr(controller,'session_statistics',{'exit':'failed before control started'})
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Torque OFF. Saved',args.output,flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nStopped; torque-off cleanup requested.',file=sys.stderr)
    except (RuntimeError,ValueError) as exc:
        print(f'Stopped: {exc}',file=sys.stderr);sys.exit(1)
