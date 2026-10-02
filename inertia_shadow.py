"""Offline FACTR momentum observer and 7-DOF shaping proposal. No hardware I/O."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import gravity_comp as gc


class MomentumObserver:
    """p_dot = tau_motor - (g - C.T v) + residual.

    Residual includes hand/contact, friction and model error. Link-only M and
    empirical motor conversion make this a diagnostic, not a force sensor.
    """
    def __init__(self, bandwidth_hz=3.):
        self.k = 2*np.pi*bandwidth_hz
        self.integral = None

    def update(self, momentum, motor_minus_beta, dt):
        if self.integral is None or not 0 < dt <= .1:
            self.integral = momentum.copy()
            return np.zeros_like(momentum), True
        # Implicit-Euler residual feedback avoids numerical instability when
        # recorded sample intervals vary. First sample is an initialization.
        residual = self.k*(momentum-self.integral-dt*motor_minus_beta)/(1+self.k*dt)
        self.integral += dt*(motor_minus_beta+residual)
        return residual, False


def shaping_matrix(M, J, factor=1.25):
    """Uniform task inertia reduction, natural inertia in the internal direction.

    Whitened inertia coordinates avoid a singular 7x7 J.T Lambda J target.
    Rows of J are scaled to compare translation/rotation for rank screening.
    This uses the link_7 origin, not a measured handle contact frame.
    """
    eigen, U = np.linalg.eigh(M)
    if eigen.min() <= 0 or not np.isfinite(eigen).all():
        raise ValueError('Inertia matrix must be positive definite')
    root = (U*np.sqrt(eigen))@U.T
    invroot = (U*(1/np.sqrt(eigen)))@U.T
    scaled = J.copy();scaled[:3] /= .2
    _, sv_geom, _ = np.linalg.svd(scaled, full_matrices=False)
    _, sv, vt = np.linalg.svd(scaled@invroot, full_matrices=True)
    rank = int(np.sum(sv > max(sv[0]*1e-10,1e-12)))
    projector = vt[:rank].T@vt[:rank]
    eligible = rank == 6 and sv_geom[-1]/max(sv_geom[0],1e-12) >= .02
    K = root@((factor-1)*projector)@invroot if eligible else np.zeros_like(M)
    return K, rank, eligible


def analyze(path, output, factor=1.25, model_gravity_scale=1.):
    if not 1 <= factor <= 2 or not np.isfinite(factor):
        raise ValueError('Shadow task factor must be finite and between 1 and 2')
    if not np.isfinite(model_gravity_scale) or not 0 < model_gravity_scale <= 1.2:
        raise ValueError('Model gravity scale must be finite and within (0, 1.2]')
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows)<2: raise ValueError('Need at least two recorded samples')
    model, data=gc.model(); frame=model.getFrameId('link_7')
    obs=MomentumObserver(); previous_time=None; previous_command=None
    values=[]; resets=0; gates=0
    def vector(row,name): return np.array([float(row[f'{name}_{i}']) for i in gc.IDS])
    for row in rows:
        t=float(row['time_s']);q=vector(row,'q_rad');v=vector(row,'dq_rad_s')
        command=vector(row,'command_current_raw')
        if not np.isfinite(np.r_[t,q,v,command]).all(): raise ValueError('Nonfinite log sample')
        if previous_time is not None and t<=previous_time: raise ValueError('Log time must increase')
        dt=0 if previous_time is None else t-previous_time
        M=np.array(pin.crba(model,data,q));M=(M+M.T)/2
        C=np.array(pin.computeCoriolisMatrix(model,data,q,v))
        g=np.array(pin.computeGeneralizedGravity(model,data,q))
        # State is read BEFORE this row's command is sent: use previous row's
        # quantized, capped command, not the current row's requested torque.
        motor=np.zeros(7) if previous_command is None else previous_command*gc.SIGNS/gc.TORQUE_TO_RAW
        residual, reset=obs.update(M@v,motor-model_gravity_scale*g+C.T@v,dt)
        J=np.array(pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL_WORLD_ALIGNED))
        K,rank,eligible=shaping_matrix(M,J,factor)
        # A gap restarts the warmup. This is only a proposal; no actuator API.
        if reset: warmup_start=t
        eligible=eligible and t-warmup_start>=2
        candidate=K@residual if eligible else np.zeros(7)
        values.append([t,int(eligible),*residual,*candidate])
        resets+=int(reset);gates+=int(not eligible)
        previous_time=t;previous_command=command
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'shadow.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time_s','proposal_enabled']+[f'{name}_{i}' for name in ('unexplained_torque_nm','candidate_assist_nm') for i in gc.IDS])
        writer.writerows(values)
    a=np.array(values); selection=a[:,1]>0
    report={'input':str(path),'hardware_commands_sent':False,'task_inertia_reduction_factor':factor,
            'nullspace':'Natural modeled inertia retained; no nullspace assistance',
            'model_gravity_scale':model_gravity_scale,
            'gravity_scale_note':'Explicit sensitivity assumption, not fitted or validated physical gravity. Motor commands remain those recorded.',
            'observer_hz':3.,'samples':len(rows),'resets':resets,'gated_samples':gates,
            'residual_rms_nm':np.sqrt(np.mean(a[selection,2:9]**2,axis=0)).tolist() if selection.any() else None,
            'candidate_peak_abs_nm':np.max(abs(a[:,9:16]),axis=0).tolist(),
            'limitations':['URDF link inertia excludes identified geared-motor inertia.',
                           'Residual includes gravity error, friction, contact and unmodeled dynamics; not hand torque.',
                           'Uses previous applied current command and empirical torque conversion, not a torque sensor.',
                           'Frame is link_7 origin; actual handle transform is unmeasured.',
                           'Candidate has no power/current limiter or passivity controller; offline ONLY.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('Saved',output/'shadow.csv')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path)
    p.add_argument('--output',type=Path,default=gc.ROOT/'outputs/diagnostics/inertia_shadow_01')
    p.add_argument('--factor',type=float,default=1.25)
    p.add_argument('--model-gravity-scale',type=float,default=1.,
                   help='Offline model sensitivity only; not a motor gain or automatic bias fit')
    args=p.parse_args();analyze(args.log,args.output,args.factor,args.model_gravity_scale)


if __name__=='__main__':main()
