"""Cartesian inertia targeting with explicit natural-inertia nullspace completion.

Pure computation: no hardware access. This ports B601's target-inertia shaping
idea, not its motor-specific friction parameters or claimed stability bounds.
"""
import numpy as np

import gravity_comp as gc
from inertia_bounded import BoundedAssistance


DEFAULT_MASS = (.15, .15, .15)  # kg, handle-local x/y/z; provisional model targets
DEFAULT_INERTIA = (.0002, .0005, .00002)  # kg m^2, handle-local x/y/z


def handle_jacobian(local_jacobian, offset):
    """Shift link_7's LOCAL Jacobian to an offset point, keeping link_7 axes.

    Pinocchio uses [linear; angular]. v_point = v_origin + omega x offset.
    """
    x, y, z = np.asarray(offset, dtype=float)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    J = np.array(local_jacobian, dtype=float, copy=True)
    J[:3] -= skew @ J[3:]
    return J


def target_shaping(M, J, target, kappa=.5, resist=.25):
    """Return K and diagnostic matrices for tau_extra = K @ residual.

    z = M^(1/2) qdot, A = J M^(-1/2). Q spans range(A.T), N=I-Q Q.T.
    B=A Q. Complete the target inertia in whitened coordinates as
        H_d = Q (B.T Lambda_d B) Q.T + N.
    Thus M_d = M^(1/2) H_d M^(1/2) is full rank, including redundancy.
    Clip eigenvalues of (B.T Lambda_d B)^-1 - I ONLY in task coordinates.
    K = M^(1/2) Q S_task Q.T M^(-1/2); the nullspace shaping gain is zero.
    Without gain clipping/fade, J M_d^-1 J.T = Lambda_d^-1.
    """
    M, J, target = map(lambda x: np.asarray(x, dtype=float), (M, J, target))
    if M.shape != (7, 7) or J.shape != (6, 7) or target.shape != (6,):
        raise ValueError('Expected 7x7 inertia, 6x7 Jacobian, and six target entries')
    if not all(np.isfinite(a).all() for a in (M, J, target)) or np.any(target <= 0):
        raise ValueError('Finite matrices and positive target masses/inertias required')
    if not np.isfinite([kappa, resist]).all() or kappa < 0 or not 0 <= resist < 1:
        raise ValueError('Require kappa >= 0 and 0 <= resist < 1')
    if not np.allclose(M, M.T, atol=1e-12):
        raise ValueError('Inertia must be symmetric')
    eigen, U = np.linalg.eigh(M)
    if eigen.min() <= 0:
        raise ValueError('Inertia must be positive definite')
    root = (U*np.sqrt(eigen)) @ U.T
    invroot = (U/np.sqrt(eigen)) @ U.T
    A = J @ invroot
    _, singular, vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(singular > max(singular[0]*1e-10, 1e-12)))
    Q = vt[:rank].T
    N = np.eye(7)-Q@Q.T
    scaled = J.copy(); scaled[:3] /= .2  # characteristic length for screening only
    geom = np.linalg.svd(scaled, compute_uv=False)
    ratio = float(geom[-1]/max(geom[0], 1e-12))
    fade = float(np.clip((ratio-.02)/.02, 0., 1.))
    info = dict(rank=rank, null_dimension=7-rank, geometry_ok=rank == 6 and fade > 0,
                geometry_ratio=ratio, geometry_fade=fade,
                null_torque_projector=root@N@invroot,
                raw_gains=np.zeros(7), clipped_gains=np.zeros(7),
                effective_inverse_inertia=np.linalg.inv(M))
    if not info['geometry_ok']:
        return np.zeros((7, 7)), info
    B = A @ Q
    Htask = B.T @ np.diag(target) @ B
    Htask_inv = np.linalg.solve(Htask, np.eye(6))
    raw, V = np.linalg.eigh((Htask_inv+Htask_inv.T)/2-np.eye(6))
    clipped = np.clip(raw, -resist, kappa)
    S = Q @ ((V*clipped) @ V.T) @ Q.T * fade
    info.update(raw_gains=np.r_[raw, 0.], clipped_gains=np.r_[clipped*fade, 0.],
                effective_inverse_inertia=invroot@(np.eye(7)+S)@invroot)
    return root@S@invroot, info


class BalancedLimiter(BoundedAssistance):
    """Signed shaping with scalar vector limits, rather than per-joint gates.

    Scalar scaling preserves task/nullspace separation BEFORE current rounding.
    Rounding introduces a small, logged nullspace component. Activation ramp
    restarts on operator off/geometry/speed faults, not on motion reversal.
    No velocity dwell or same-direction requirement; no cumulative work cutoff.
    """
    ACTIVATION_RAMP = 2.

    def __init__(self, speed_fade=2., speed_off=3., additional_limits=True):
        super().__init__(speed_fade, speed_off, latch_overspeed=False, energy_budget_enabled=False)
        self.additional_limits = bool(additional_limits)
        self.enabled_for = 0.

    def _zero(self, reason):
        self.enabled_for = 0.
        return super()._zero(reason)

    def step(self, candidate, velocity, baseline_raw, caps_raw, dt, *, enabled=False, eligible=True, trace=False):
        stages = {name: np.full(7, np.nan) for name in self.TRACE_FIELDS}
        def zero(reason):
            effort, info = self._zero(reason)
            info.update(limiter_trace=stages, negative_power_w=0.)
            return effort, info
        candidate, velocity, baseline_raw, caps_raw = [np.asarray(a, dtype=float) for a in
                                                       (candidate, velocity, baseline_raw, caps_raw)]
        if any(a.shape != (7,) or not np.isfinite(a).all() for a in
               (candidate, velocity, baseline_raw, caps_raw)):
            self.fault = 'invalid_state'
        elif not np.isfinite(dt) or not 0 < dt <= self.MAX_DT:
            self.fault = 'timing_fault'
        if self.fault: return zero(self.fault)
        if np.any(caps_raw <= 0) or np.any(abs(baseline_raw) > caps_raw) or np.any(baseline_raw != np.rint(baseline_raw)):
            self.fault = 'invalid_baseline_or_caps'
            return zero(self.fault)
        if not enabled: return zero('operator_disabled')
        if not eligible: return zero('observer_or_geometry_gate')
        if np.max(abs(velocity)) >= self.speed_off: return zero('assist_speed_gate')
        self.enabled_for += dt
        ramp = min(1., self.enabled_for/self.ACTIVATION_RAMP)
        fade = float(np.clip((self.speed_off-np.max(abs(velocity)))/(self.speed_off-self.speed_fade), 0, 1))

        def scale_to_bounds(values, bounds):
            nonzero = abs(values) > 0
            return min(1., float(np.min(bounds[nonzero]/abs(values[nonzero])))) if nonzero.any() else 1.

        effort = candidate*scale_to_bounds(candidate, self.TORQUE_CAP) if self.additional_limits else candidate.copy()
        stages['capped_magnitude_nm'] = abs(effort)
        effort = effort*ramp
        stages['motion_ramp'] = np.full(7, ramp)
        stages['ramped_magnitude_nm'] = abs(effort)
        effort = effort*fade
        stages['speed_fade'] = np.full(7, fade)
        stages['faded_magnitude_nm'] = abs(effort)
        old = np.where(self.previous*effort > 0, abs(self.previous), 0.)
        if self.additional_limits:
            effort = effort*scale_to_bounds(effort, old+self.RISE_RATE*dt)
        stages['slew_magnitude_nm'] = abs(effort)
        stages['direction_effort_nm'] = effort.copy()  # no direction gate in this mode
        electrical = effort*gc.SIGNS*gc.TORQUE_TO_RAW
        headroom = np.where(electrical >= 0, caps_raw-baseline_raw, caps_raw+baseline_raw)
        effort = effort*scale_to_bounds(electrical, headroom)
        stages['headroom_effort_nm'] = effort.copy()
        positive = float(np.maximum(effort*velocity, 0).sum())
        if self.additional_limits and positive > self.POWER_CAP: effort *= self.POWER_CAP/positive
        stages['power_energy_effort_nm'] = effort.copy()
        self.previous = effort.copy()
        raw = np.trunc(effort*gc.SIGNS*gc.TORQUE_TO_RAW)
        effort = raw*gc.SIGNS/gc.TORQUE_TO_RAW
        stages['quantized_effort_nm'] = effort.copy()
        power = float(np.maximum(effort*velocity, 0).sum())
        negative = float(np.minimum(effort*velocity, 0).sum())
        self.sampled_positive_work_j += power*dt
        return effort, dict(reason='assisting' if np.any(effort) else 'ramp_or_below_current_resolution',
                            energy_remaining_j=None, positive_power_w=power, negative_power_w=negative,
                            limiter_trace=stages)
