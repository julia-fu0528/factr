"""Train/evaluate a state-based Franka pick policy with Isaac Lab 2.2 and PPO."""
import argparse
import json
from pathlib import Path
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="outputs/cube_pick")
parser.add_argument("--checkpoint", help="Resume training or evaluate this checkpoint")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--eval-rounds", type=int, default=3)
parser.add_argument("--record", action="store_true", help="Save first evaluation batch for visual replay")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner


def main():
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.commands.object_pose.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    for name in ("panda_shoulder", "panda_forearm"):
        cfg.scene.robot.actuators[name].stiffness = 400.0
        cfg.scene.robot.actuators[name].damping = 40.0
    agent = LiftCubePPORunnerCfg()
    agent.seed = args.seed
    agent.device = args.device
    agent.max_iterations = args.iterations
    dump_yaml(str(output / "env.yaml"), cfg)
    dump_yaml(str(output / "agent.yaml"), agent)
    raw = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    robot = raw.scene["robot"]
    metadata = {
        "task": "Franka cube pick / lift", "seed": args.seed,
        "joint_names": robot.joint_names,
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "observations": str(raw.observation_manager),
        "arm_action": "default_joint_pos + 0.5 * action[:7]",
        "gripper_action": "nonnegative opens to 0.04 m, negative closes to 0 m",
        "arm_kp": 400, "arm_kd": 40, "policy_dt": raw.step_dt,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        if not args.eval:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
            runner.save(str(output / "model_final.pt"))
            export_policy_as_jit(runner.alg.policy, runner.obs_normalizer,
                                 str(output / "exported"), filename="policy.pt")
        else:
            if not args.checkpoint:
                raise ValueError("--eval requires --checkpoint")
            policy = runner.get_inference_policy(device=args.device)
            results = []
            # Each batch starts from fresh randomized cube positions, with a new seed.
            for batch in range(args.eval_rounds):
                raw.seed(args.seed + batch)
                obs, _ = env.reset()
                streak = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
                success = torch.zeros_like(streak, dtype=torch.bool)
                active = torch.ones_like(success)
                max_z = torch.zeros(args.num_envs, device=args.device)
                frames = []
                with torch.no_grad():
                    for _ in range(raw.max_episode_length):
                        obs, _, done, _ = env.step(policy(obs))
                        z = raw.scene["object"].data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                        # Require a sustained lift near the fingers, not a tossed cube.
                        valid = active & (~done.bool())
                        cube = raw.scene["object"].data
                        ee = raw.scene["ee_frame"].data.target_pos_w[:, 0, :]
                        held = ((cube.root_pos_w - ee).norm(dim=1) < 0.08) & (cube.root_lin_vel_w.norm(dim=1) < 0.25)
                        streak = torch.where(valid & (z > 0.15) & held, streak + 1, 0)
                        success |= streak >= round(0.5 / raw.step_dt)
                        max_z = torch.where(valid, torch.maximum(max_z, z), max_z)
                        active &= ~done.bool()
                        if args.record and batch == 0:
                            frames.append(torch.cat((robot.data.joint_pos,
                                raw.scene["object"].data.root_pos_w - raw.scene.env_origins,
                                raw.scene["object"].data.root_quat_w), dim=1).cpu().numpy())
                if frames:
                    chosen = int(success.nonzero()[0]) if success.any() else int(max_z.argmax())
                    np.savez_compressed(output / "replay.npz", states=np.stack(frames)[:, chosen],
                                        dt=raw.step_dt, success=bool(success[chosen]),
                                        seed=args.seed + batch, environment_index=chosen)
                results.append({"seed": args.seed + batch, "episodes": args.num_envs,
                                "successes": int(success.sum()), "mean_max_height_m": float(max_z.mean())})
            report = {"checkpoint": str(Path(args.checkpoint).resolve()),
                      "criterion": "for 0.5 continuous seconds: cube center > 0.15 m, within 0.08 m of end effector, speed < 0.25 m/s",
                      "batches": results,
                      "success_rate": sum(x["successes"] for x in results) / (args.num_envs * args.eval_rounds)}
            (output / "evaluation.json").write_text(json.dumps(report, indent=2))
            print("EVALUATION", json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
