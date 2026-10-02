"""Isaac Sim 6.0 Franka follower for the local FACTR stream."""
import argparse
import json
import math
import time
from pathlib import Path

from teleop_stream import HOME, Receiver, advance_target, franka_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--udp-port', type=int, default=5557)
    parser.add_argument('--timeout', type=float, default=.25)
    parser.add_argument('--speed', type=float, default=1.5, help='Maximum joint target speed in rad/s')
    parser.add_argument('--steps', type=int, default=0, help='Stop after N physics steps; 0 means until window closes')
    parser.add_argument('--demo', action='store_true', help='Synthetic motion; no hardware connection')
    parser.add_argument('--report', type=Path, help='Save simulation tracking statistics as JSON')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--wrist-roll-offset-deg', type=float, default=45.,
                        help='Simulation-only joint-7 offset in degrees (default: +45; use 0 to disable)')
    args = parser.parse_args()
    if not -180 <= args.wrist_roll_offset_deg <= 180:
        parser.error('--wrist-roll-offset-deg must be between -180 and 180')
    if (not 1024 <= args.udp_port <= 65535 or not .05 <= args.timeout <= 5
            or not .01 <= args.speed <= 5 or args.steps < 0):
        parser.error('Invalid port, timeout, speed, or step count')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})
    receiver = None
    sim = None
    try:
        import numpy as np
        import omni.kit.app
        # These compatibility extensions ship with the installed 6.0 runtime.
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.core.api', True)
        manager.set_extension_enabled_immediate('isaacsim.robot.manipulators.examples', True)
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.robot.manipulators.examples.franka import Franka
        from pxr import UsdLux
        import omni.usd

        dt = 1 / 120
        sim = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
        set_camera_view(eye=np.array([1.6, 1.6, 1.3]), target=np.array([0, 0, .45]))
        sim.scene.add_default_ground_plane()
        light = UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), '/World/Light')
        light.CreateIntensityAttr(1500)
        robot = sim.scene.add(Franka(prim_path='/World/Franka', name='franka'))
        sim.reset()
        arm_ids = [robot.dof_names.index(f'panda_joint{i}') for i in range(1, 8)]
        finger_ids = [robot.dof_names.index(f'panda_finger_joint{i}') for i in (1, 2)]
        properties = robot.dof_properties
        lower, upper = properties['lower'][arm_ids], properties['upper'][arm_ids]
        controller = robot.get_articulation_controller()
        kp, kd = controller.get_gains()
        kp[arm_ids], kd[arm_ids] = 400., 40.
        controller.set_gains(kps=kp, kds=kd)
        target = np.clip(franka_target(HOME, args.wrist_roll_offset_deg), lower, upper).tolist()
        receiver = None if args.demo else Receiver(args.udp_port, args.timeout)
        initial = robot.get_joint_positions().copy()
        initial[arm_ids] = target
        initial[finger_ids] = .04
        robot.set_joint_positions(initial)
        robot.set_joint_velocities(np.zeros_like(initial))
        steps, samples, error_sum, max_error = 0, 0, 0., 0.
        state_previous = None
        start = time.monotonic()
        deadline = start
        print('READY: Franka simulation. Start factr_teleop.py in another terminal.', flush=True)
        while app.is_running() and (args.steps == 0 or steps < args.steps):
            if not sim.is_playing():
                app.update()
                deadline = time.monotonic()
                continue
            if args.demo:
                desired = [q + .15 * math.sin(steps * dt * 1.2 + i * .3) for i, q in enumerate(HOME)]
            else:
                desired = receiver.poll()
            desired = franka_target(desired, args.wrist_roll_offset_deg)
            state = 'following' if desired is not None else 'waiting / stream stale; holding target'
            if state != state_previous:
                print(state, flush=True)
                state_previous = state
            target = advance_target(target, desired, lower, upper, dt, args.speed)
            command = initial.copy()
            command[arm_ids] = target
            controller.apply_action(ArticulationAction(joint_positions=command))
            sim.step(render=not args.headless)
            actual = robot.get_joint_positions()[arm_ids]
            error = float(np.max(np.abs(actual - target)))
            if not math.isfinite(error):
                raise RuntimeError('Nonfinite simulated joint state')
            error_sum += error
            max_error = max(max_error, error)
            samples += desired is not None
            steps += 1
            deadline += dt
            now = time.monotonic()
            time.sleep(max(0, deadline - now))
            if now - deadline > .1:
                deadline = now  # Avoid bursts of catch-up steps after rendering stalls.
        result = {'steps': steps, 'steps_with_input': samples, 'demo': args.demo,
                  'wrist_roll_offset_deg': args.wrist_roll_offset_deg,
                  'mean_max_joint_error_rad': error_sum / max(steps, 1),
                  'max_joint_error_rad': max_error, 'final_target': target,
                  'final_actual': robot.get_joint_positions()[arm_ids].tolist()}
        print(json.dumps(result), flush=True)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        if receiver is not None:
            receiver.close()
        if sim is not None:
            # Release physics callbacks before closing the USD stage.
            sim.clear_all_callbacks()
            sim.clear_instance()
        # This app has no Replicator capture jobs to wait for.
        app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
