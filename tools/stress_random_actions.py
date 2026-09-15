#!/usr/bin/env python3
"""Drive the environment with random actions and report the first sign of divergence."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from simtoolreal_newton.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_newton.launch import add_env_arguments, make_env  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from test_headless_env import apply_overrides, fmt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_env_arguments(parser)
    args = parser.parse_args()
    cfg = SimToolRealCfg()
    apply_overrides(cfg, args.overrides)
    env = make_env(cfg, num_envs=args.num_envs, device=args.sim_device, physics=args.physics, visualizer=args.viz)
    inner = env.unwrapped
    from isaaclab_newton.physics import NewtonManager as pm
    torch.manual_seed(0)
    with torch.inference_mode():
        env.reset()
        for step in range(args.steps):
            actions = args.action_scale * torch.randn(env.num_envs, env.num_actions, device=env.device)
            obs, _, rewards, dones, infos = env.step(actions)
            q = inner.q
            dq = inner.dq
            cube_v = inner.cube_linear_velocity.norm(dim=1)
            cube_p = inner.cube_position
            finite = torch.isfinite(obs).all(dim=1)
            bad = (~finite) | (dq.abs().amax(dim=1) > 200.0) | (cube_v > 20.0)
            if step % 25 == 0 or bool(bad.any()):
                try:
                    mjd = pm._solver.mjw_data
                    nefc = int(mjd.nefc.numpy().max())
                    nacon = int(mjd.nacon.numpy().reshape(-1)[0])
                except Exception:  # noqa: BLE001
                    nefc = nacon = -1
                print("step {:3d} | max |dq| {} | max cube |v| {} | max |cube z| {} | dones {} | obs finite {} | nefc max {} nacon {}".format(
                    step, fmt(dq.abs().max()), fmt(cube_v.max()), fmt(cube_p[:, 2].abs().max()), int(dones.sum()), int(finite.sum()), nefc, nacon))
            if bool(bad.any()):
                e = int(bad.nonzero()[0])
                print("  first bad env {} at step {}: ref {} q max {} dq max {} cube v {} cube p {}".format(
                    e, step, int(inner.reference_index[e]), fmt(q[e].abs().max()), fmt(dq[e].abs().max()), fmt(cube_v[e]), [round(float(v), 3) for v in cube_p[e]]))
                worst = torch.argsort(dq[e].abs(), descending=True)[:4]
                print("  worst joints:", [(inner.robot.joint_names[int(j)], round(float(dq[e, j]), 1)) for j in worst])
                break
        else:
            print("no divergence in {} steps".format(args.steps))
    env.close()


if __name__ == "__main__":
    main()
