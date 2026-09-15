#!/usr/bin/env python3
"""Smoke test of the Isaac Lab port: build the scene, replay the demonstration, time steps.

Runs without a policy. It checks the things the port could silently get
wrong: joint order, the merged palm frame against the URDF kinematics, the
reset placing robot and cuboid on the reference, the position drive tracking
the demonstration under ideal residual actions, terminations and auto-resets,
and finally the raw step throughput at the requested environment count.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from simtoolreal_newton.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_newton.launch import add_env_arguments, make_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=240, help="Replay steps per start index.")
    parser.add_argument("--rsi", type=int, nargs="+", default=[0, 740], help="Reference indices to replay from.")
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--contact", action="store_true", help="Enable the fingertip contact sensor.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    add_env_arguments(parser)
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    import json

    for item in overrides:
        path, raw = item.split("=", 1)
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        setattr(node, parts[-1], value)


def fmt(value):
    return "{:.4g}".format(float(value))


def main():
    args = parse_args()
    cfg = SimToolRealCfg()
    cfg.seed = int(args.seed)
    if args.contact:
        cfg.contact.enabled = True
    apply_overrides(cfg, args.overrides)
    env = make_env(cfg, num_envs=args.num_envs, device=args.sim_device, physics=args.physics, visualizer=args.viz)
    inner = env.unwrapped
    print("joint names (public order):", inner.robot.joint_names)
    print("body names (backend order):", inner.robot.body_names)
    print("num envs {} obs {} actions {} device {}".format(env.num_envs, env.num_obs, env.num_actions, env.device))

    with torch.inference_mode():
        # ---- reset consistency ---------------------------------------------
        obs = env.reset(reference_index=0)
        q = inner.q.clone()
        ref = inner.transform_bank.sample(inner.transform_index, inner.reference_index)
        print("reset |q - q_ref| max:", fmt((q - ref.q).abs().max()))
        palm_pos_sim, palm_quat_sim = inner._palm_pose_world()
        palm_pos_fk, palm_quat_fk = inner.kinematics.palm_pose(q[:, :6])
        palm_pos_fk = palm_pos_fk + inner.robot_base_position
        pos_err = (palm_pos_sim - palm_pos_fk).norm(dim=1).max()
        quat_err = (1.0 - (palm_quat_sim * palm_quat_fk).sum(dim=1).abs()).max()
        print("palm pose: sim vs URDF FK  |dp| max {} m, 1-|q.q'| max {}".format(fmt(pos_err), fmt(quat_err)))
        tips_sim = inner._fingertip_positions_world()
        tips_fk = inner.kinematics.fingertip_positions(q) + inner.robot_base_position
        print("fingertips: sim vs URDF FK |dp| max {} m".format(fmt((tips_sim - tips_fk).norm(dim=-1).max())))
        cube_ref = inner._cube_reference_root_states(ref)
        print("cube reset |dp| max {} m".format(fmt((inner.cube_position - cube_ref[:, :3]).norm(dim=1).max())))
        print("cube z {} table top {}".format(fmt(inner.cube_position[0, 2]), fmt(inner.cfg.table_pos[2] + cfg.table.size_m[2] / 2)))
        print("obs finite:", bool(torch.isfinite(obs).all()), "obs shape", tuple(obs.shape))

        # ---- hold still for a moment: gravity / drive sanity ----------------
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        zero[:, 6:] = inner.positions_to_hand_actions(inner.hand_q)
        for _ in range(30):
            env.step(zero)
        print("after 30 hold steps: |q - target| max {} rad, cube |dp| {} m".format(
            fmt((inner.q - inner.position_targets).abs().max()),
            fmt((inner.cube_position - inner._cube_reference_root_states(inner.transform_bank.sample(inner.transform_index, inner.reference_index))[:, :3]).norm(dim=1).max()),
        ))

        # ---- replay the demonstration with ideal actions --------------------
        for start in args.rsi:
            env.reset(reference_index=int(start))
            stats = {"arm_q": 0.0, "hand_q": 0.0, "palm_kp": 0.0, "tip_kp": 0.0, "cube": 0.0, "reward": 0.0, "early": 0, "done": 0}
            step_times = []
            for step in range(args.steps):
                actions, _ = inner.next_reference_action()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                obs, _, rewards, dones, infos = env.step(actions)
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
                stats["arm_q"] = max(stats["arm_q"], float(infos["max_abs_arm_position_error"].max()))
                stats["hand_q"] = max(stats["hand_q"], float(infos["max_abs_hand_position_error"].max()))
                stats["palm_kp"] = max(stats["palm_kp"], float(infos["palm_keypoint_error_m"].max()))
                stats["tip_kp"] = max(stats["tip_kp"], float(infos["fingertip_keypoint_error_m"].max()))
                stats["cube"] = max(stats["cube"], float(infos["object_position_error_m"].max()))
                stats["reward"] += float(rewards.mean())
                stats["early"] += int(infos["early_termination"].sum())
                stats["done"] += int(dones.sum())
                if not torch.isfinite(obs).all():
                    raise RuntimeError("Non-finite observation at step {}".format(step))
            print(
                "replay from {:4d}: {} steps | max arm q err {} rad | max hand q err {} rad | max palm kp err {} m | "
                "max tip kp err {} m | max cube err {} m | mean reward {} | early {} | dones {} | "
                "{:.2f} ms/step".format(
                    start, args.steps, fmt(stats["arm_q"]), fmt(stats["hand_q"]), fmt(stats["palm_kp"]),
                    fmt(stats["tip_kp"]), fmt(stats["cube"]), fmt(stats["reward"] / args.steps), stats["early"],
                    stats["done"], 1000.0 * sum(step_times[10:]) / max(len(step_times) - 10, 1),
                )
            )
            print("   final reference index", int(inner.reference_index[0]), "cube lift", fmt(infos["object_com_lift_m"].max()),
                  "contact fraction", fmt(infos["fingertip_contact_fraction"].mean()))

        # ---- random actions: terminations and auto-reset -------------------
        env.reset()
        dones_total = 0
        for _ in range(120):
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
            obs, _, rewards, dones, infos = env.step(actions)
            dones_total += int(dones.sum())
        print("random actions: {} episode ends in 120 steps, obs finite {}".format(dones_total, bool(torch.isfinite(obs).all())))

        # ---- throughput ----------------------------------------------------
        env.reset()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for _ in range(10):
            env.step(actions)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.benchmark_steps):
            env.step(actions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(
            "throughput: {} envs x {} steps in {:.2f} s = {:.0f} env-steps/s ({:.2f} ms/step)".format(
                env.num_envs, args.benchmark_steps, elapsed, env.num_envs * args.benchmark_steps / elapsed,
                1000.0 * elapsed / args.benchmark_steps,
            )
        )
    env.close()


if __name__ == "__main__":
    main()
