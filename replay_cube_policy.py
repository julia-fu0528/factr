"""Render a recorded policy evaluation in Isaac Sim 6 (replay, not live inference)."""
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recording", type=Path, default=Path("outputs/cube_pick_eval/replay.npz"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--loops", type=int, default=0, help="0 repeats until the window closes")
parser.add_argument("--screenshot", type=Path, help="Capture the midpoint of the first replay")
args = parser.parse_args()
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60)
import numpy as np
record = np.load(args.recording)
states, dt = record["states"], float(record["dt"])
from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
world = None
try:
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.core.api", True)
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import UsdLux
    import omni.usd

    root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac"
    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    print("Loading Franka asset", flush=True)
    add_reference_to_stage(root + "/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd", "/World/Robot")
    robot = world.scene.add(SingleArticulation(prim_path="/World/Robot", name="robot"))
    print("Loading cube asset", flush=True)
    add_reference_to_stage(root + "/Props/Blocks/DexCube/dex_cube_instanceable.usd", "/World/Cube")
    cube = world.scene.add(SingleRigidPrim(prim_path="/World/Cube", name="cube", scale=np.full(3, 0.8)))
    print("Loading table asset", flush=True)
    add_reference_to_stage(root + "/Props/Mounts/SeattleLabTable/table_instanceable.usd", "/World/Table")
    SingleXFormPrim(prim_path="/World/Table", position=np.array([.5, 0, 0]),
                    orientation=np.array([.70710678, 0, 0, .70710678]))
    UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/Light").CreateIntensityAttr(2000)
    set_camera_view(eye=np.array([1.7, 1.4, 1.3]), target=np.array([.35, 0, .4]))
    world.reset()
    faulthandler.cancel_dump_traceback_later()
    names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
    ids = [robot.dof_names.index(n) for n in names]
    print(f"Replaying evaluation: seed={int(record['seed'])}, environment={int(record['environment_index'])}, success={bool(record['success'])}", flush=True)
    loop = 0
    while app.is_running() and (args.loops == 0 or loop < args.loops):
        for frame_index, state in enumerate(states):
            if not app.is_running():
                break
            start = time.monotonic()
            robot.set_joint_positions(state[:9], joint_indices=ids)
            robot.set_joint_velocities(np.zeros(9), joint_indices=ids)
            cube.set_world_pose(position=state[9:12], orientation=state[12:16])
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            # One step synchronizes PhysX poses to the renderer. Every next frame
            # is overwritten from the recording; no policy is run in this viewer.
            world.step(render=True)
            if args.screenshot and loop == 0 and frame_index == len(states) // 2:
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                capture_viewport_to_file(get_active_viewport(), str(args.screenshot.resolve()))
            if not args.headless:
                time.sleep(max(0.0, dt - (time.monotonic() - start)))
        loop += 1
finally:
    if world is not None:
        world.clear_all_callbacks()
        world.clear_instance()
    app.close(wait_for_replicator=False)
