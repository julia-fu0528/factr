"""Evaluate balanced drag on recorded motion/residuals; never opens a motor port.

This is a fixed-recording comparison, not a closed-loop motion prediction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

import gravity_comp as gc
from balanced_drag import target_shaping, handle_jacobian, DEFAULT_MASS, DEFAULT_INERTIA
from factr_inertia import LiveController


def replay(source, output, target_mass=DEFAULT_MASS, target_inertia=DEFAULT_INERTIA,
           kappa=.5, resist=.25, handle_offset=(0.,0.,0.), torque_rise_rate=None, assist_limits=None):
    source,output=Path(source),Path(output)
    metadata=json.loads((source/'metadata.json').read_text())
    with (source/'samples.csv').open() as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Empty recording')
    controller=LiveController(controller='balanced',target_mass=target_mass,target_inertia=target_inertia,
                              kappa=kappa,resist=resist,handle_offset=handle_offset,
                              assist_fade_speed=metadata['assist_fade_speed_rad_s'],
                              assist_max_speed=metadata['assist_cutoff_speed_rad_s'],
                              torque_rise_rate=(metadata.get('torque_rise_rate_nm_s') or .04) if torque_rise_rate is None else torque_rise_rate,
                              assist_limits=metadata.get('assist_limits','bounded') if assist_limits is None else assist_limits)
    limiter=controller.limiter
    # Compare shaping/gating only, holding recorded torque and power caps fixed.
    if metadata.get('assist_torque_caps_nm') is not None:limiter.TORQUE_CAP=np.asarray(metadata['assist_torque_caps_nm'])
    if metadata.get('power_cap_w') is not None:limiter.POWER_CAP=metadata['power_cap_w']
    model,data=controller.model,controller.data
    enabled_s=0.;counts=np.zeros(7);old_counts=np.zeros(7);peaks=np.zeros(7)
    candidate_null=0.;continuous_null=0.;quantized_null=0.;compute_times=[];out=[]
    task_gain_min=0.;task_gain_max=0.;disabled_by_speed=False;last_requested=False
    reasons={}
    for i,row in enumerate(rows):
        started=time.perf_counter()
        def vec(name):return np.array([float(row[f'{name}_{j}']) for j in gc.IDS])
        q,v,residual=vec('q_rad'),vec('dq_rad_s'),vec('residual_nm')
        dt=float(row['loop_dt_s']);t=float(row['time_s'])
        requested=bool(int(row['assistance_enabled']))
        if requested and not last_requested:disabled_by_speed=False
        last_requested=requested
        M=np.array(pin.crba(model,data,q));M=(M+M.T)/2
        J=handle_jacobian(pin.computeFrameJacobian(model,data,q,controller.frame,pin.LOCAL),handle_offset)
        K,shape=target_shaping(M,J,controller.target,kappa,resist)
        P=shape['null_torque_projector']
        task_residual=residual-P@residual
        used=np.sign(task_residual)*np.maximum(abs(task_residual)-controller.residual_deadband,0.)
        candidate=K@used
        eligible=bool(int(row['observer_warm'])) and shape['geometry_ok'] and float(row['joint_margin_rad'])>=.15 and not bool(int(row['baseline_saturated']))
        if i==0:
            effort,info=limiter._zero('initializing');pre=np.zeros(7)
        else:
            effort,info=limiter.step(candidate,v,vec('baseline_current_raw'),vec('current_limit_raw'),dt,
                                     enabled=requested and not disabled_by_speed,eligible=eligible,trace=True)
            pre=info['limiter_trace']['power_energy_effort_nm']
            if not np.isfinite(pre).all():pre=np.zeros(7)
            if info['reason']=='assist_speed_gate':disabled_by_speed=True
        if requested:
            enabled_s+=dt;counts+=(effort!=0)*dt;old_counts+=(vec('assist_nm')!=0)*dt
        peaks=np.maximum(peaks,abs(effort))
        candidate_null=max(candidate_null,float(np.linalg.norm(P@candidate)))
        continuous_null=max(continuous_null,float(np.linalg.norm(P@pre)))
        quantized_null=max(quantized_null,float(np.linalg.norm(P@effort)))
        task_gain_min=min(task_gain_min,float(shape['clipped_gains'].min()))
        task_gain_max=max(task_gain_max,float(shape['clipped_gains'].max()))
        reasons[info['reason']]=reasons.get(info['reason'],0)+1
        total=vec('baseline_current_raw')+np.rint(effort*gc.SIGNS*gc.TORQUE_TO_RAW)
        if np.any(abs(total)>vec('current_limit_raw')):raise RuntimeError('Replay current bound failure')
        if limiter.additional_limits and info['positive_power_w']>limiter.POWER_CAP+1e-12:raise RuntimeError('Replay power bound failure')
        out.append([t,int(requested),info['reason'],shape['rank'],shape['null_dimension'],info['positive_power_w'],
                    *candidate,*pre,*effort,*(P@effort),*shape['clipped_gains']])
        compute_times.append(time.perf_counter()-started)
    report=dict(source=str(source),source_sha256=hashlib.sha256((source/'samples.csv').read_bytes()).hexdigest(),
                hardware_commands_sent=False,method='Recorded q, velocity, observer residual, baseline and enable transitions held fixed; hypothetical commands are NOT fed back into observer.',
                target_mass_kg=list(target_mass),target_rotational_inertia_kg_m2=list(target_inertia),
                handle_offset_m=list(handle_offset),kappa=kappa,resist=resist,
                assist_limits=controller.assist_limits,
                torque_rise_rate_nm_s=limiter.RISE_RATE if limiter.additional_limits else None,
                power_cap_w=limiter.POWER_CAP if limiter.additional_limits else None,
                assist_torque_caps_nm=limiter.TORQUE_CAP.tolist() if limiter.additional_limits else None,
                recorded_enabled_s=enabled_s,recorded_assist_percent_by_joint=(100*old_counts/max(enabled_s,1e-12)).tolist(),
                replay_assist_percent_by_joint=(100*counts/max(enabled_s,1e-12)).tolist(),
                peak_extra_torque_nm=peaks.tolist(),peak_extra_current_ma=(peaks*gc.TORQUE_TO_RAW*gc.MA_PER_RAW).tolist(),
                sampled_positive_work_j=limiter.sampled_positive_work_j,limiter_fault=limiter.fault,
                task_gain_range=[task_gain_min,task_gain_max],nullspace_candidate_peak_norm_nm=candidate_null,
                nullspace_prequantization_peak_norm_nm=continuous_null,nullspace_sent_peak_norm_nm=quantized_null,
                computation_ms_p50_p95_p99=(1000*np.percentile(compute_times,[50,95,99])).tolist(),reason_counts=reasons,
                limitations=['Not a closed-loop or physical stability validation.',
                             'Link-only URDF inertia and empirical motor torque conversions; no fitted friction compensation.',
                             'Handle offset defaults to link_7 origin; measured grip-point offset must be supplied if different.',
                             'Quantization perturbs exact task/nullspace separation; active limits alter the target.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'samples.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_s','recorded_enabled','reason','rank','null_dimension','positive_power_w']+
                        [f'{key}_{j}' for key in ('candidate_nm','prequantization_nm','assist_nm','nullspace_output_nm','clipped_task_gain') for j in gc.IDS])
        writer.writerows(out)
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,help='Live recording directory')
    parser.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/balanced_replay_01')
    parser.add_argument('--target-mass',type=float,nargs=3,default=DEFAULT_MASS)
    parser.add_argument('--target-inertia',type=float,nargs=3,default=DEFAULT_INERTIA)
    parser.add_argument('--kappa',type=float,default=.5)
    parser.add_argument('--resist',type=float,default=.25)
    parser.add_argument('--handle-offset',type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument('--torque-rise-rate',type=float,help='Override logged torque rise rate (Nm/s); old logs default to 0.04')
    parser.add_argument('--assist-limits',choices=('bounded','current-only'),help='Override recorded output limit profile')
    args=parser.parse_args()
    replay(args.source,args.output,args.target_mass,args.target_inertia,args.kappa,args.resist,args.handle_offset,args.torque_rise_rate,args.assist_limits)


if __name__=='__main__':main()
