#!/usr/bin/env python3
"""Smoke test of the per-episode cuboid scale on the live simulator.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/smoke_object_scale.py \
        --checkpoint logs/staged/w6_s7_cont2_it17000/model_17000.pt

Runs three groups of environments in one process with the bar at scale 0.8,
1.0 and 1.2 (same bank pose 122), and reports for each group:

* the solver-side half extents actually bound (shape_scale) and, when the
  MuJoCo-Warp backend exposes them, the per-world geom sizes;
* the bar's centre height right after the reset and after it has settled
  (a 20% bigger bar must rest 5 mm higher, not sink or pop);
* from the grasp window (RSI 790): the object displacement over the first 15
  control steps -- the "fired away" check for fingers snapped onto a bar of
  another size;
* from frame 0 with the given policy (zero-shot): the fraction of roll-outs
  whose maximum object error stays under 7 cm, the median max error and the
  peak lift, per scale.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from evaluate import load_saved_configuration  # noqa: E402
from sweep_pose_success import reset_to_pose  # noqa: E402
from test_headless_env import apply_overrides  # noqa: E402

import simtoolreal_newton.envs.motion_imitation_env as env_module  # noqa: E402
from simtoolreal_newton.launch import make_env  # noqa: E402
from simtoolreal_newton.runners import PPO  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--per-scale", type=int, default=32)
    parser.add_argument("--bank-index", type=int, default=122)
    parser.add_argument("--grasp-rsi", type=int, default=790)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config or checkpoint.parent / "config.json"
    env_cfg, train_cfg = load_saved_configuration(config_path)
    apply_overrides(env_cfg, args.overrides)
    scales = [float(s) for s in args.scales]
    num_envs = len(scales) * int(args.per_scale)
    pattern = torch.tensor(
        [s for s in scales for _ in range(int(args.per_scale))], dtype=torch.float32
    )
    env_cfg.object_randomization.scale_min = min(scales)
    env_cfg.object_randomization.scale_max = max(scales)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.play = True
    env_cfg.viewer.enable_viewer = False
    env_cfg.viewer.reference_ghost = False
    env_cfg.viewer.training_camera_enabled = False
    env_cfg.termination.enabled = False
    env_cfg.object_assist.enabled = False
    train_cfg.runner.record_video = False

    # Deterministic groups instead of a uniform draw: every reset of all envs
    # gets the fixed pattern.
    def fixed_scales(count, randomization_cfg, device, generator=None):
        if count == num_envs:
            return pattern.to(device)
        return torch.ones(count, dtype=torch.float32, device=device)

    env_module.sample_scales = fixed_scales

    env = make_env(env_cfg, num_envs=None, device=args.sim_device, physics=None, visualizer=None)
    try:
        inner = env.unwrapped
        device = env.device
        env.max_episode_length = int(env.reference.last_index)
        env.cfg.env.episode_length = env.max_episode_length
        runner = PPO(env, train_cfg, log_dir=None, device=device)
        runner.load(checkpoint, load_optimizer=False, load_normalizers=True)
        policy = runner.get_inference_policy(device=device)
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        bank = torch.full((num_envs,), int(args.bank_index), device=device, dtype=torch.long)
        groups = {s: torch.nonzero(pattern.to(device) == s).squeeze(1) for s in scales}
        half_z = float(inner.object_half_extents[2])

        def reference_positions():
            sample = inner.transform_bank.sample(inner.transform_index, inner.reference_index)
            return inner._cube_reference_root_states(sample)[:, 0:3]

        print("== bindings")
        for s, ids in groups.items():
            first = int(ids[0])
            bound = inner._object_shape_scale[first].tolist() if inner._object_shape_scale is not None else None
            print("scale {:.2f}: object_scale {:.3f} half extents {} shape_scale binding {}".format(
                s, float(inner.object_scale[first]), [round(v, 4) for v in inner.object_half_extents_per_env[first].tolist()],
                None if bound is None else [round(v, 4) for v in bound]))
        try:
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

            solver = NewtonManager._solver
            geom_size = solver.mjw_model.geom_size.numpy()
            mapping = solver.mjc_geom_to_newton_shape.numpy()
            print("mujoco geom_size array shape {} (worlds x geoms x 3)".format(geom_size.shape))
            # The cube is the shape whose nominal size is the bar's half extents.
            nominal = inner.object_half_extents.cpu().numpy()
            import numpy as np  # noqa: PLC0415

            for g in range(geom_size.shape[1]):
                if np.allclose(geom_size[groups[1.0][0].item() if 1.0 in groups else 0, g], nominal, atol=1e-5):
                    for s, ids in groups.items():
                        print("  world {} (scale {:.2f}) geom {} size {}".format(
                            int(ids[0]), s, g, np.round(geom_size[int(ids[0]), g], 4).tolist()))
                    break
            else:
                print("  (no geom matched the nominal bar size; mapping shape {})".format(mapping.shape))
        except Exception as error:  # noqa: BLE001
            print("  geom_size read-back not available: {!r}".format(error))

        # Resting height: reset at frame 0, read, settle for 30 steps with the policy.
        print("== rest height (frame 0), metres above the reference bar centre of scale 1")
        reference_full = torch.zeros((num_envs,), device=device, dtype=torch.long)
        reset_to_pose(env, env_ids, reference_full, transform_indices=bank)
        z0 = inner.cube_position[:, 2].clone()
        ref_z_nominal = reference_positions()[:, 2] - half_z * (inner.object_scale - 1.0)
        obs = inner.policy_obs
        for _ in range(30):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        z30 = inner.cube_position[:, 2].clone()
        for s, ids in groups.items():
            print("scale {:.2f}: at reset {:+.4f}  after 30 steps {:+.4f}  (expected {:+.4f})".format(
                s, float((z0[ids] - ref_z_nominal[ids]).mean()), float((z30[ids] - ref_z_nominal[ids]).mean()),
                half_z * (s - 1.0)))

        # Grasp-window pop test.
        print("== grasp window (RSI {}): object displacement from its reference over 15 steps".format(args.grasp_rsi))
        reference_grasp = torch.full((num_envs,), int(args.grasp_rsi), device=device, dtype=torch.long)
        obs = reset_to_pose(env, env_ids, reference_grasp, transform_indices=bank)
        start = inner.cube_position.clone()
        max_disp = torch.zeros(num_envs, device=device)
        for _ in range(15):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            err = torch.linalg.vector_norm(inner.cube_position - reference_positions(), dim=1)
            max_disp = torch.maximum(max_disp, err)
        for s, ids in groups.items():
            d = max_disp[ids]
            print("scale {:.2f}: median max error {:.4f} m, > 3 cm in {}/{} envs, > 7 cm in {}/{}".format(
                s, float(d.median()), int((d > 0.03).sum()), len(ids), int((d > 0.07).sum()), len(ids)))

        # Zero-shot from frame 0 with the given policy.
        print("== zero-shot from frame 0 with {}".format(checkpoint.name))
        obs = reset_to_pose(env, env_ids, reference_full, transform_indices=bank)
        steps = int(env.reference.last_index)
        max_error = torch.zeros(num_envs, device=device)
        lift = torch.zeros(num_envs, device=device)
        z_start = inner.cube_position[:, 2].clone()
        for _ in range(steps - 1):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            err = torch.linalg.vector_norm(inner.cube_position - reference_positions(), dim=1)
            max_error = torch.maximum(max_error, err)
            lift = torch.maximum(lift, inner.cube_position[:, 2] - z_start)
        for s, ids in groups.items():
            e = max_error[ids]
            print("scale {:.2f}: pass@7cm {}/{}  pass@10cm {}/{}  median max error {:.4f}  median peak lift {:.3f}".format(
                s, int((e <= 0.07).sum()), len(ids), int((e <= 0.10).sum()), len(ids),
                float(e.median()), float(lift[ids].median())))
    finally:
        env.close()


if __name__ == "__main__":
    main()
