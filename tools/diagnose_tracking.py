#!/usr/bin/env python3
"""Per-joint tracking and contact diagnostics for solver tuning."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from simtoolreal_newton.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_newton.envs.controller import JOINT_NAMES  # noqa: E402
from simtoolreal_newton.launch import add_env_arguments, make_env  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from test_headless_env import apply_overrides, fmt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rsi", type=int, default=740)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--debug-cube-contacts", action="store_true")
    parser.add_argument("--save-frames", type=Path, default=None, help="Directory for PNG frames of env 0.")
    parser.add_argument("--frame-every", type=int, default=10)
    parser.add_argument("--identity-transform", action="store_true",
                        help="Replay with the cuboid at the demonstration's own pose (no bank offset).")
    add_env_arguments(parser)
    args = parser.parse_args()
    cfg = SimToolRealCfg()
    apply_overrides(cfg, args.overrides)
    env = make_env(cfg, num_envs=args.num_envs, device=args.sim_device, physics=args.physics, visualizer=args.viz,
                   debug_cube_contacts=args.debug_cube_contacts, camera=args.save_frames is not None)
    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)
    inner = env.unwrapped
    robot = inner.robot
    print("joint stiffness (env0):", [round(float(v), 2) for v in robot.data.joint_stiffness.torch[0]])
    print("joint damping   (env0):", [round(float(v), 3) for v in robot.data.joint_damping.torch[0]])
    print("joint armature  (env0):", [round(float(v), 4) for v in robot.data.joint_armature.torch[0]])
    print("effort limits   (env0):", [round(float(v), 2) for v in robot.data.joint_effort_limits.torch[0]])
    pm = inner.sim.physics_manager
    model = getattr(pm, "_model", None) or getattr(type(pm), "_model", None)
    if model is not None:
        for attr in ("body_gravcomp", "body_mass"):
            arr = getattr(model, attr, None)
            if arr is not None:
                vals = arr.numpy()
                print("newton model {}[:28]:".format(attr), [round(float(v), 3) for v in vals[:28]])
        print("gravity:", model.gravity if hasattr(model, "gravity") else None)
    from isaaclab_newton.physics import NewtonManager as _pm
    _model = _pm._model
    _labels = list(_model.shape_label)
    for i, lab in enumerate(_labels):
        if "/Cube/" in lab or "/Table/" in lab:
            print("shape", lab.split("/")[-3], "type", int(_model.shape_type.numpy()[i]), "scale", _model.shape_scale.numpy()[i], "world", int(_model.shape_world.numpy()[i]))
    demo_z = inner.reference.cube_pose[:, 2]
    lift = (demo_z > demo_z[0] + 0.01).nonzero()
    print("demo cube lift begins at frame", int(lift[0]) if len(lift) else None)
    with torch.inference_mode():
        env.reset(reference_index=0)
        hold = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        hold[:, 6:] = inner.positions_to_hand_actions(inner.hand_q)
        for _ in range(60):
            env.step(hold)
        err = (inner.q - inner.position_targets)[0]
        print("hold: per-joint |q - target| (rad):")
        for name, e, q, t in zip(JOINT_NAMES, err, inner.q[0], inner.position_targets[0]):
            flag = " <--" if abs(float(e)) > 0.02 else ""
            print("  {:20s} err {:+.4f}  q {:+.4f}  target {:+.4f}{}".format(name, float(e), float(q), float(t), flag))
        print("hold: cube |dp| {} m, |v| {} m/s".format(
            fmt((inner.cube_position - inner._cube_reference_root_states(inner.transform_bank.sample(inner.transform_index, inner.reference_index))[:, :3]).norm(dim=1).max()),
            fmt(inner.cube_linear_velocity.norm(dim=1).max())))
        if args.identity_transform:
            env.reset(reference_index=args.rsi, translation_xy=(0.0, 0.0), yaw_rad=0.0)
        else:
            env.reset(reference_index=args.rsi)
        print("reset: transform idx", inner.transform_index[0].item(), "bank translation", inner.transform_bank.translation[inner.transform_index[0]].tolist(),
              "yaw", float(inner.transform_bank.yaw_rad[inner.transform_index[0]]))
        for step in range(args.steps):
            actions, _ = inner.next_reference_action()
            obs, _, rewards, dones, infos = env.step(actions)
            if args.save_frames is not None and step % args.frame_every == 0:
                from PIL import Image

                Image.fromarray(inner.capture_training_camera_frame(0)).save(
                    args.save_frames / "frame_{:04d}_ref_{:04d}.png".format(step, int(inner.reference_index[0])))
            if step % args.frame_every == 0 or float(infos["object_position_error_m"].max()) > 0.2:
                worst = infos["hand_q_error"] if "hand_q_error" in infos else None
                print(
                    "step {:3d} ref {:4d} | cube err {} m |v| {} | arm q err {} | hand q err {} | tip kp err {} | palm kp {} | early {} | contact frac {} force {}".format(
                        step, int(inner.reference_index[0]), fmt(infos["object_position_error_m"].max()),
                        fmt(inner.cube_linear_velocity.norm(dim=1).max()), fmt(infos["max_abs_arm_position_error"].max()),
                        fmt(infos["max_abs_hand_position_error"].max()), fmt(infos["fingertip_keypoint_error_m"].max()),
                        fmt(infos["palm_keypoint_error_m"].max()), int(infos["early_termination"].sum()),
                        fmt(infos["fingertip_contact_fraction"].mean()), fmt(infos["mean_fingertip_contact_force_n"].mean()),
                    )
                )
                try:
                    import mujoco as _mj
                    mjd = _pm._solver.mjw_data
                    mjm_cpu = _pm._solver.mj_model
                    ncon0 = int(mjd.nacon.numpy().reshape(-1)[0])
                    geoms = mjd.contact.geom.numpy()[:ncon0]
                    dists = mjd.contact.dist.numpy()[:ncon0]
                    worlds = mjd.contact.worldid.numpy()[:ncon0]
                    cube_pairs = []
                    for (g1, g2), d, w in zip(geoms, dists, worlds):
                        if int(w) != 0:
                            continue
                        n1 = str(_mj.mj_id2name(mjm_cpu, _mj.mjtObj.mjOBJ_GEOM, int(g1)))
                        n2 = str(_mj.mj_id2name(mjm_cpu, _mj.mjtObj.mjOBJ_GEOM, int(g2)))
                        if "Cube" in n1 or "Cube" in n2:
                            other = n2 if "Cube" in n1 else n1
                            cube_pairs.append((other.split("/")[-1][-22:], round(float(d), 4)))
                    print("      mujoco contacts with cube (world 0):", cube_pairs)
                    print("      mjwarp nacon {} nefc max {} | cube v env0 {}".format(
                        ncon0, int(mjd.nefc.numpy().max()), [round(float(v), 2) for v in inner.cube_linear_velocity[0]]))
                except Exception as exc:  # noqa: BLE001
                    print("      (no mjwarp stats: {})".format(exc))
                print("      per-finger tip-to-cube surface distance env0 (thumb,index,middle) [m]:",
                      [round(float(v), 4) for v in infos["fingertip_object_distance_per_finger_m"][0]],
                      "| cube net contact force env0 [N]:",
                      [round(float(v), 3) for v in inner.cube_contact_sensor.data.net_forces_w.torch[0, 0]] if inner.cube_contact_sensor is not None else None)
                if inner.cube_contact_sensor is not None:
                    names = list(inner.cube_contact_sensor.filter_object_names or [])
                    for e in range(env.num_envs):
                        matrix = inner.cube_contact_sensor.data.force_matrix_w.torch[e, 0]
                        norms = matrix.norm(dim=-1)
                        top = torch.argsort(norms, descending=True)[:3]
                        touching = [(names[int(i)].split("/")[-1], round(float(norms[int(i)]), 2)) for i in top if float(norms[int(i)]) > 1e-3]
                        print("      env{} cube err {:.4f} |v| {:.3f} tips {} contacts {}".format(
                            e, float(infos["object_position_error_m"][e]), float(inner.cube_linear_velocity[e].norm()),
                            [round(float(v), 3) for v in infos["fingertip_object_distance_per_finger_m"][e]], touching))
                if float(infos["object_position_error_m"].max()) > 0.2:
                    e = (inner.q - inner.position_targets)[0]
                    print("   worst joints:", sorted(zip([abs(float(v)) for v in e], JOINT_NAMES), reverse=True)[:4])
                    break
    env.close()


if __name__ == "__main__":
    main()
