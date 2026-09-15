# SimToolReal on Isaac Lab / Newton

UR5e + Tesollo DG5F demonstration-guided grasping, ported from Isaac Gym to
**Isaac Lab 3.0 with the Newton physics backend (MuJoCo-Warp solver)**.
The task, rewards, reference-state initialisation, transform bank and the
AnimRL PPO runner are the ones from `simtoolreal_animrl`; only the simulator
underneath changed. Training runs *kit-less*: no Isaac Sim / Omniverse is
needed, and the physics step is captured into one CUDA graph. The task side
(control, rewards, observations, resets) runs as eager PyTorch on the GPU.

```
env:     UR5e (6 DoF) + DG5F (20 DoF) articulation, cuboid, table   -> Isaac Lab DirectRLEnv
physics: Newton 1.6 (Warp 1.17) with MuJoCo-Warp; PhysX backends selectable
policy:  AnimRL PPO (checkpoint-compatible with the Isaac Gym repository)
```

## Setup

Requirements: Linux, an NVIDIA GPU with a recent driver (CUDA 12.8 wheels),
[`uv`](https://docs.astral.sh/uv/). Python 3.12 is fetched by `uv`.

```bash
./setup.sh
```

This clones the pinned Isaac Lab `develop` checkout into `deps/IsaacLab`,
syncs its environment (`deps/IsaacLab/.venv`, ~8 GB with torch), installs this
package into it and converts the URDF to USD (`assets/usd/`). Everything below
uses that interpreter:

```bash
PY=deps/IsaacLab/.venv/bin/python
```

Isaac Sim is optional and only needed for the `physx`/`isaacsim_physx`
backends or the Kit viewer (`ISAACLAB_EXTRAS="--extra isaacsim" ./setup.sh`).

## Smoke test

```bash
$PY scripts/test_headless_env.py --num-envs 16
```

Builds the scene, checks joint order, the merged palm frame against the URDF
kinematics, reset placement, replays the demonstration with ideal residual
actions, exercises random actions / auto-resets and reports the step
throughput. `tools/diagnose_tracking.py` prints per-joint tracking and
per-environment cube contacts for solver tuning.

## Training

```bash
$PY scripts/train.py --num-envs 4096 --run-name newton_baseline
```

Same command line as the Isaac Gym repository (`--set path=value` overrides,
`--object-assist`, `--domain-randomization`, `--asymmetric-critic`, `--resume`,
...). New flags: `--physics {newton_mjwarp,physx,ovphysx,isaacsim_physx}` and
`--viz newton` to open the Newton viewer. Runs land in
`logs/simtoolreal/<timestamp>_<run_name>/` with `config.json`,
`metrics.jsonl`, TensorBoard events and `model_<iter>.pt` checkpoints; the
periodic deterministic evaluation (`best_model.pt`, `best_deployment_model.pt`)
runs in a subprocess exactly as before.

```bash
$PY -m tensorboard.main --logdir logs/simtoolreal
```

Evaluate a checkpoint (deterministic mean actions, plots):

```bash
$PY scripts/evaluate.py --checkpoint logs/simtoolreal/<run>/best_model.pt
$PY scripts/evaluate.py --checkpoint ... --viz newton     # watch it
```

The environment is also registered as `SimToolReal-Grasp-Direct` for Isaac
Lab's own entry points (RSL-RL):

```bash
deps/IsaacLab/.venv/bin/isaaclab train --task SimToolReal-Grasp-Direct --num_envs 4096
```

## What changed with respect to the Isaac Gym environment

| Isaac Gym | Here |
| --- | --- |
| `gym.load_asset` URDF, `collapse_fixed_joints` | URDF converted once to USD with fixed joints merged (`scripts/convert_urdf.py`); same surviving bodies (`wrist_3_link` carries mount/base/palm, `rl_dg_<f>_4` carries the tip) |
| PhysX Jacobian tensor of `wrist_3_link` | Palm Jacobian from a PyTorch URDF model (`envs/kinematics.py`), identical on every backend; checked against the transform bank |
| `set_dof_position_target_tensor` | `Articulation.actuators.target_command` with `ImplicitActuatorCfg` PD gains |
| Rigid-shape filter bits | `physics:filteredPairs` on the USD (robot–table off, arm–cube off), imported by Newton |
| Root-state / DOF-state uploads | `write_joint_state_to_sim_index`, `write_root_pose_to_sim_index` |
| `apply_rigid_body_force_tensors` | wrench composers (object assist, impulses) |
| Net contact force tensor | `ContactSensor` on the five distal phalanges |
| `episode_length`, dt 1/60, 2 PhysX substeps | dt 1/60, 4 MJWarp substeps (240 Hz), `implicitfast` |

Configuration lives in `simtoolreal_newton/cfg/simtoolreal_config.py` as
before; the new `sim` section selects the backend and the MJWarp solver
settings (`sim.mjwarp.*`), all overridable with `--set`.

## Physics notes (MuJoCo-Warp vs the PhysX original)

The port was validated by replaying the demonstration with ideal residual
actions in both simulators (`tools/diagnose_tracking.py --identity-transform`
here, the same loop in the Isaac Gym repository). What was found and what the
defaults do about it:

* **Joint tracking.** Two 120 Hz substeps left the light finger joints
  0.1 rad short of their targets; four substeps (240 Hz) with 100 solver
  iterations track to 1e-6 rad. Gravity on the robot is compensated with
  MuJoCo's per-body `gravcomp` (the original disabled gravity on the robot).
* **Collision detection.** MuJoCo-Warp's native mesh-box collision launched the
  resting cuboid at 0.4 m/s from a 5 g fingertip touching it at 5 cm/s.
  Newton's own collision pipeline (`sim.mjwarp.use_mujoco_contacts=false`,
  the default) does not. Multi-point contacts (`enable_multiccd`) made the
  resting box explode and are off.
* **Contact softness.** The MuJoCo default `solref` (0.02 s) also kicked the
  cuboid at first touch; `sim.mjwarp.contact_solref=[0.01, 1.0]` keeps the
  first touch quiet and is the default. Softer values (0.03, 0.05) kick again.
* **Friction combination.** PhysX averaged the two materials of a contact,
  MuJoCo takes the maximum, so `asset.fingertip_friction` is 1.0 here (it was
  1.5) to keep the effective fingertip-cuboid friction at 1.0.
* **Joint velocity limits.** PhysX clamped every joint at the URDF limit
  (3.14 rad/s); MuJoCo-Warp does not, and a residual hand target jumping
  0.4 rad flings a phalanx at 80 rad/s into the cuboid. The environment
  therefore slews the *applied* targets at that limit
  (`control.slew_targets_at_velocity_limit`), which gives the same arrival
  time the clamped joint had; the commanded target in the observation is
  unchanged. Explicit velocity-limited actuators (`DCMotorCfg`) were tried and
  are unstable at 60 Hz with these gains.
* **Depenetration.** PhysX capped the depenetration velocity at 2 m/s; MuJoCo
  has no such cap and a cuboid crushed between rigid finger hulls left at
  tens of m/s, taking the articulation with it. The cuboid's velocity is
  clamped after every step (`sim.mjwarp.object_max_linear_velocity` 4 m/s,
  `object_max_angular_velocity` 40 rad/s). With this, 300 steps of Gaussian
  random actions on 256 environments produce no non-finite state
  (`tools/stress_random_actions.py`).
* **Remaining difference.** Replaying the demonstration at its own cube pose,
  PhysX squeezed the cuboid with ~1 N per fingertip and lifted it 5 cm before
  losing it; MuJoCo-Warp presses ~10x harder and pushes it out of the hand
  around frame 800. The policy learns its own contact strategy in either
  simulator, but expect the grasp phase to differ from the Isaac Gym runs.
  Knobs: `sim.mjwarp.contact_solref`, `contact_condim`, `contact_margin`,
  `contact_gap`, `asset.fingertip_friction`, `control.hand_*_scale`.

Not ported (yet): the reference "ghost" robot and off-screen training video
(both need a renderer; use `--viz newton`), per-environment friction
randomisation (gains, masses, impulses and sensor noise are randomised as
before). The retargeting tools (`scripts/build_transform_bank.py`) still need
`pytorch_kinematics` (`uv pip install pytorch_kinematics`).

## Layout

```
simtoolreal_newton/
  cfg/            AnimRL-style Python configuration (env + train)
  envs/           task modules (pure torch) + motion_imitation_env.py (Isaac Lab)
  runners/        AnimRL PPO, evaluators, plots, deployment score
  tasks/          gym registration for `isaaclab train`
  launch.py       make_env(): backend/visualizer selection, kit-less launch
scripts/          train / evaluate / periodic_evaluate / convert_urdf / test_headless_env
tools/            diagnostics
assets/           URDF + meshes (usd/ generated), demonstrations/, banks/
deps/IsaacLab     pinned Isaac Lab checkout + virtual environment (setup.sh)
```
