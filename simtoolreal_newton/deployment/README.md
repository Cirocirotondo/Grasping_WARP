# Real-robot deployment of SimToolReal-Newton policies

Runs a policy trained in this repository on the physical UR5e + Tesollo DG5F.
Simulation is the default; the arm and the hand are armed separately and
explicitly, so each rung of the ladder below is a flag change, never an edit.

```
simtoolreal_newton/deployment/
  contract.py        the policy contract off the simulator: DeploymentRun (checkpoint,
                     config, demonstration, transform bank, URDF kinematics, network),
                     build_observation(), ActionPipeline (filter -> IK arm / residual
                     hand -> commanded targets -> velocity-limit slew)
  cube_source.py     reference clip / frozen / pose-estimation cuboid pose
  arm_client.py      ZMQ to impedance_controller (UR5e)
  hand_client.py     UDP peer of dg5f_policy_ros_bridge.py (DG5F)
  hand_stiffness.py  the DG5F position PID's p gain, through the ros2 CLI
  safety.py          step limiter, spike monitor, arming prompt
  viewer.py          the MuJoCo window, on sim2sim.MujocoSim with physics off
scripts/run_policy_real.py            the staged controller (this is what you run)
scripts/check_deployment_contract.py  stage 1: the contract against the environment
tests/test_deployment_contract.py     the contract's arithmetic, no simulator needed
```

The contract is **not restated** here. `build_observation` and `ActionPipeline`
call the environment's own modules -- `envs/kinematics.py` for the palm pose,
the fingertips and the palm Jacobian straight from the URDF,
`envs/operational_space.py` for the damped least-squares step,
`envs/rotations.py` and `envs/cuboid_symmetry.py` for the encodings -- on the
CPU, one sample per control step. No MuJoCo, no Isaac Lab in the loop: the
robot integrates the physics, the kinematics module answers "where is the palm
given these joint angles", with the same chain, offsets and limits that produced
the training observation.

## 0. What the policy expects

Read the run's `config.json`; the contract refuses a checkpoint it cannot serve.
The policies to deploy (branch `generalize_size`, finger self-collisions on,
cuboid pose *and* size generalised):

| checkpoint | obs | notes |
| --- | --- | --- |
| `deploy/policies/sc2_anchor_s42_it17300/model_17300.pt` | 113 | more precise, no domain randomisation -- first on the robot |
| `deploy/policies/dr_combo_s7_ladder/model_17500.pt` | 113 | resumed from the one above with cuboid-pose noise/bias, impulses, mass randomisation |
| `deploy/policies/w6_ref_ori_lowlr_it14600_generalize_pose/model_14600.pt` | 112 | earlier, no finger collisions (fingers intersect); kept as the 112-D regression case |

* **Observation, 112 values (+1).** 26 joint positions normalised by the URDF
  limits, the 26 *previously commanded* position targets, 26 joint velocities,
  the phase, then the palm position (3) and 6D rotation (6) in the base frame,
  the five fingertips in the palm frame (15), and the cuboid's 6D rotation (6)
  and centre (3) in the palm frame. The bar's eight-fold symmetry is resolved
  once, against the pose the episode starts at, and held (see
  `envs/cuboid_symmetry.py`). A checkpoint with
  `object_randomization.observe_scale` appends the **bar's scale** as the last
  column, the raw factor `s` (0.8-1.2 in training; a 0.12 x 0.04 x 0.04 m bar
  is 0.8). `--object-scale` supplies it and also lifts the reference clip's
  bar by `half_height * (s - 1)` so a scaled bar rests on the table, as
  `envs/object_scale.py` does in training. The transform bank and the
  demonstration do not change with the scale.
* **Actions, 26 values, through a first-order filter** (`control.action_filter_alpha`,
  0.3 on the deployed checkpoint: `a_f = 0.3 a + 0.7 a_f_prev`). The config is
  explicit that any deployment must run the same filter on the same raw output.
* **Arm: a palm twist, not joint offsets.** `[dx, dy, dz, wx, wy, wz]` in the
  base frame, scaled by `dt` and saturated at `arm_translation_speed_m_per_s` /
  `arm_rotation_speed_rad_per_s` without turning, resolved through the palm
  Jacobian at the *measured* arm configuration by a damped least-squares step
  (`ik_damping`), clamped to `ik_max_joint_delta_rad` per joint per step, and
  **accumulated onto the previously commanded arm target** (not the measured
  state), then clamped to the joint limits.
* **Hand: residuals.** `default_hand + clip(a * scale_hand_joint_target, +-clip_joint_target)`.
* **Applied targets** are slewed toward the commanded ones at the URDF joint
  velocity limit (pi rad/s: 0.052 rad per 60 Hz step). The observation reports
  the commanded target; the drives receive the slewed one.

Why the IK lives here and not in the UR controller: `impedance_controller` does
offer a Cartesian mode (`target_ee_pose` -> `speedL`), but it never materialises
a joint target -- UR's firmware resolves the twist internally, on the
pendant-configured TCP rather than the palm, through its own gains and
clamps -- and 26 of the policy's inputs are exactly the joint targets training's
IK produced. Delegating would feed the network a different signal. So the
deployment runs the same IK and streams `{"target_q": ...}`, as the hand-only
and full controllers before it did.

## 1. Interpreter

The repository's own venv, which already carries torch and this package; the
deployment path imports nothing from Isaac Lab (checked: `isaaclab` and `warp`
are not loaded). It needs `pyzmq` for the arm and the estimator:

```bash
cd /home/duplo/simtoolreal_newton          # branch generalize_size
uv pip install --python deps/IsaacLab/.venv/bin/python pyzmq
PY="env -u PYTHONPATH deps/IsaacLab/.venv/bin/python"
CKPT=deploy/policies/sc2_anchor_s42_it17300/model_17300.pt
SCALE=1.0                                  # the real bar's size / nominal
```

Every `run_policy_real.py` command below takes `--object-scale $SCALE`.

## 2. Hardware bring-up

**Hand.** Bench supply 24 V / 10 A, ethernet, `tesollo` network profile
(169.254.186.0). The driver, then the bridge -- the bridge is reused unmodified
from the SimToolReal deployment tree and runs under ROS 2 Humble's Python:

```bash
cd /home/duplo/git/tesollo_ros2
source /opt/ros/humble/setup.bash && source install_dg5f/setup.bash
ros2 launch dg5f_driver dg5f_right_pid_all_controller.launch.py

cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
source /opt/ros/humble/setup.bash && source /home/duplo/git/tesollo_ros2/install_dg5f/setup.bash
python3 dg5f_policy_ros_bridge.py
```

The bridge owns the last line of hand safety: joint-limit clipping, a 0.12 rad
per-command step clamp, rejection of commands while `/joint_states` is stale,
and a measured-position hold when the policy stops. Hand homing streams the
start target for up to 10 s and requires measured error below 0.18 rad.

**Arm.** Ethernet, `ur5` profile (192.168.1.10), Remote Control on the tablet:

```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
./impedance_controller pc_ur_new.json
```

Note the hand's PD gains in training were scaled to 11.6% of the URDF's
(`control.hand_stiffness_scale`); the real DG5F runs the driver's position PID.
The policies learned to over-command that soft hand: with the hand replayed on
the reference clip, `sc2_anchor`'s commanded hand targets sit a median 0.06 rad
(approach) to 0.18 rad (carry) from the reference pose, p95 0.24-0.55 rad, max
0.77 rad, and 9% of joint-steps ask for a target beyond the URDF limit (up to
0.39 rad beyond); `dr_combo` is the same within a few hundredths. In simulation such targets produce moderate torques; a
stiff hand will actually go there. The bridge clips to the limits and the
runner's `--max-hand-step-rad` bounds the rate, but neither changes the
steady-state target, so expect a harder close than the videos show and start
stage 5 at `--hand-action-scale 0.2`.

**Hand stiffness.** The real hand's stiffness is one ROS parameter. The DG5F
hardware interface takes only an effort (PWM duty) command; `rj_dg_pospid` is a
ROS-side `pid_controller` at 300 Hz with `p` only (`i = d = 0`), so
`gains.<joint>.p` on `/dg5f_right/rj_dg_pospid` is the whole stiffness, and it
applies live. The driver's YAML sets 2.0, which saturates the duty at 0.5 rad
of error -- much stiffer than training's soft hand. Both of these load a gain
on all 20 joints and read it back:

```bash
$PY scripts/set_hand_stiffness.py --show          # what the controller has now
$PY scripts/set_hand_stiffness.py --p 0.5         # any time the driver is up
$PY scripts/run_policy_real.py ... --hand-stiffness 0.5   # at the start of a run
```

Start low and raise it between runs (0.5 -> 0.8 -> 1.2 -> 2.0). A low `p`
sags under gravity and contact because `i` is zero -- as the trained hand did
-- so watch `max|hand-ref|` against the 1.35 rad monitor. Keep `d` at 0 (the
firmware velocity is an int8 rpm). The driver's own current limiter lives in
its closed library and engages around 155 mA; the 170 mA below is only the
bridge's warning, so a warning means limiting has already begun. Lower `p`
means lower stall duty and fewer limiter events.

**Homing needs authority, the rollout wants compliance.** With `i = 0` a soft
hand keeps a steady-state error and never reaches the homing tolerance: at
`p = 0.5` a 0.19 rad error is only 9.5% duty, below the joints' stiction, and
homing aborts (measured on `rj_dg_2_2`). Use `--hand-home-stiffness 2.0`
together with `--hand-stiffness 0.5`: the runner homes at the stiff gain,
drops to the soft one, holds the start pose for
`--hand-home-settle-seconds` (1.0) while the hand sags to its new
equilibrium, prints the resulting error, and only then starts the policy.

**Cuboid pose.** Not needed until stage 7 -- stages 1-6 take the cuboid from
the reference clip.

**Cuboid pose noise (stage 7).** The estimator's position error is bimodal, not
Gaussian. Measured on a static bar at 25 Hz: 95% of samples sit within 3 mm of
one another with a 0.16 mm median step, and the rest snap 16-22 mm away when the
set of recognised faces changes. Those excursions last one or two samples,
never more, and the estimator's own `confidence` nearly separates them -- good
samples 0.46-0.63, bad ones 0.43-0.50.

So the filter that works is a **median**, not a low-pass and not a confidence
gate: `--pose-median-window` defaults to **5**. The excursions last one or two
samples, so a median of five removes them outright -- measured with the arm in
the scene, worst case 1.89 mm against 14.8 mm raw -- for about 80 ms of lag,
which is free while the bar is standing still. Unlike a gate, a median can
never starve the stream: it always has an answer.

`--pose-min-confidence` is **off by default, and is not the filter to use for a
rollout**, although an earlier version of this file said otherwise. Confidence
does separate the modes -- with the arm parked, low-confidence samples deviate
1.91 mm at the median against 0.19 mm for the rest -- but the whole
distribution slides down as the robot occludes the board: a median of 0.625 on a
clear bar becomes 0.577 with the arm parked, and during the approach every
sample falls below a threshold set at 0.55. An absolute gate therefore goes from
rejecting 2% of samples to rejecting all of them exactly when the policy is
reaching for the bar, and the run stalls on `--pose-timeout`. That failure is at
least loud: the `CubeSourceError` names which of the two causes it was.

**When the board cannot be seen at all.** The estimator publishes *nothing*
while it detects no board -- it does not send an empty message -- and the hand
occludes the tags exactly during the approach. So a silent stream usually means
a hidden bar, not a dead process; check its terminal for `Published 1 board
poses` before restarting anything. `--pose-timeout` (2.0 s) is how long the last
pose is held across such a gap, and holding it is sound: the bar does not move
until it is grasped, and what the policy observes is the bar in the **palm**
frame, which keeps evolving correctly from the measured arm while the base-frame
pose stands still.

`--pose-jump-reject-m` (0.05) is a gross-teleport backstop only, and on this
noise it correctly never fires: the bar itself travels up to 16 mm between
samples during transport, so a threshold near the noise scale would fight real
motion. `--pose-filter-alpha` (0, off) is a position-only low-pass, kept for
tuning but not needed -- 40 ms of lag is 16 mm of error at transport speed.
**Orientation is never filtered**: two detections of a symmetric bar can differ
by a symmetry element, and interpolating between them sweeps it through an
orientation it was never in. `canonicalize_cuboid_orientation` resolves that
instead, once per episode, exactly as in training.

## 3. Which clip: the transform bank

This checkpoint generalises over cuboid placements: every episode plays one of
the 1536 clips of `banks/stage1_box.pt`, each a retargeted demonstration for one
planar placement (x +-0.09 m, y 0-0.15 m, yaw -22.5..45 deg about the bar's own
centre). Training placed the bar *exactly* on a bank entry
(`env.rsi_snap_placement_from_index = 0`).

`--bank-index N` plays entry N; `--bank-index nearest` (default) plays the
entry nearest the demonstration's own placement (122 on this bank). With
`--cube-source pose-estimation`, `nearest` reads the live cuboid, converts it
to the bank's (translation, yaw) parameterisation, snaps to the nearest entry
and prints the residual; it aborts above `--max-placement-residual-m` (7 cm) /
`--max-placement-residual-deg` (10) or if the bar is tilted off the table.

Read the residual, do not just clear the limit. Neighbouring entries are **2 mm
apart** over x -0.09..0.09 m, y 0.00..0.15 m, yaw -22.3..44.9 deg, so a residual
of more than a few millimetres does not mean the bank is coarse -- it means the
bar is standing **outside the region the policy was trained on**, and the clip
being played is a demonstration for a materially different pose. The 7 cm
default admits that deliberately; a small residual remains the sign that the
placement is in distribution.

## 4. The commissioning ladder

The runner applies two one-second startup protections by default: the action is
blended from the clip's ideal action (`DeploymentRun.next_reference_action`,
the same one the environment computes) to the learned action over one second of
demonstration frames (`--startup-policy-blend-seconds 1.0`), and the resulting
target is ramped from the verified home pose (`--startup-ramp-seconds 1.0`). The
blend is measured in demonstration time, so a slowed 1 Hz or 5 Hz commissioning
run hands over just as gradually per policy frame.

**Stage 1 -- the contract against the environment.** The real gate. Runs the
policy in the Newton environment and rebuilds, every step, the observation and
the commanded targets from the environment's measured state through the code
above. Observation errors of 1e-5 and target errors of 1e-6 are float32 noise;
anything larger is a frame or convention bug.

```bash
$PY scripts/check_deployment_contract.py --checkpoint $CKPT --rsi-index 0 --object-scale $SCALE
$PY scripts/check_deployment_contract.py --checkpoint $CKPT --rsi-index 760 --bank-index 500 --object-scale 0.8
$PY scripts/check_deployment_contract.py --checkpoint $CKPT --rsi-index 0 --viz newton_gl   # watch it
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m unittest tests.test_deployment_contract tests.test_hand_stiffness
```

Under `--viz newton_gl`, space pauses the simulation: the loop blocks inside
`env.step` and the camera stays live, so the scene can be inspected from any
angle without the policy running on. Newton binds space to its rendering pause
instead -- which freezes the window while the episode continues behind it --
and the viewer's own on-screen help still describes that binding;
`launch.make_env` rebinds the key. The side panel's "Pause Simulation" button
is the same flag, and "Pause Rendering" is Newton's original one. Shift and the
middle mouse button pan the camera; F frames the whole model.

**Stage 2 -- dry run, no hardware.** Confirms the policy loads, the observation
builds and the monitors behave. Nothing is commanded.

```bash
$PY scripts/run_policy_real.py --checkpoint $CKPT --no-realtime
```

**Stage 3 -- read hardware, command nothing.** Needs the low-level controller,
the DG5F driver and the bridge. Verifies both state streams and the kinematics.

```bash
$PY scripts/run_policy_real.py --checkpoint $CKPT \
  --use-real-arm-state --use-real-hand-state --max-steps 200
```

**Stage 4 -- arm alone with ideal context, stepped and slow.** Only the physical
arm state enters the observation and only the arm is commanded; hand `q/dq`
and the cuboid come from the reference clip. Two confirmations: a typed `SEND`,
then Space. `--arm-action-scale` scales the requested palm twist.

```bash
$PY scripts/run_policy_real.py --checkpoint $CKPT \
  --commission-arm-only-ideal-context --debug-step --control-hz 1 \
  --arm-action-scale 0.2 --max-steps 10
```

**Stage 5 -- hand alone with ideal context, stepped and slow.** The mirror
image, at low stiffness rather than a scaled-down action: the policy's hand
targets are what training produced, the hand just follows them softly. While
waiting for Space the runner refreshes the hand target at 20 Hz to keep the
bridge's watchdog satisfied.

```bash
$PY scripts/run_policy_real.py --checkpoint $CKPT --object-scale $SCALE \
  --commission-hand-only-ideal-context --debug-step --control-hz 1 \
  --hand-stiffness 0.5 --hand-home-stiffness 2.0 --max-steps 10
```

**Stage 6 -- both, stepped**, then both free-running slow, then raise the rate,
then the stiffness:

```bash
$PY scripts/run_policy_real.py --checkpoint $CKPT --object-scale $SCALE \
  --send-to-arm --send-to-hand --debug-step --control-hz 1 \
  --arm-action-scale 0.2 --hand-stiffness 0.5 --hand-home-stiffness 2.0 --max-steps 10

$PY scripts/run_policy_real.py --checkpoint $CKPT --object-scale $SCALE \
  --send-to-arm --send-to-hand --control-hz 5 --arm-action-scale 0.3 \
  --hand-stiffness 0.5 --hand-home-stiffness 2.0 --target-smoothing 0.6 --max-steps 100
# then, one change at a time: control-hz 5 -> 10 -> 20 -> 60,
# then arm action scale 0.3 -> 0.5 -> 1.0, then smoothing 0.6 -> 0,
# then hand stiffness 0.5 -> 0.8 -> 1.2 -> 2.0 (scripts/set_hand_stiffness.py between runs).
```

**Stage 7 -- live cuboid.** Verify the estimator frame *before* trusting it.

```bash
# terminal A -- the estimator, in its own environment
cd /home/duplo/git/robohand/src/tag-pose-estimation
/home/duplo/git/robohand-robohand2/.venv/bin/python scripts/run_pose_estimation.py \
  --config config/pose_estimation_configs/parallelepiped_5x5x15cm_robot_frame.json

# terminal B -- check the frame, then a dry run, then hardware
$PY scripts/run_policy_real.py --checkpoint $CKPT --cube-source pose-estimation \
  --check-cube-frame --bank-index nearest --rsi-index 0 --object-scale 1.0

$PY scripts/run_policy_real.py --checkpoint $CKPT --cube-source pose-estimation \
  --bank-index nearest --object-scale 1.0 --no-viewer --max-steps 60

$PY scripts/run_policy_real.py --checkpoint $CKPT --object-scale 1.0 \
  --cube-source pose-estimation --bank-index nearest \
  --send-to-arm --send-to-hand --debug-step --control-hz 1 \
  --arm-action-scale 0.2 --hand-stiffness 0.5 --hand-home-stiffness 2.0 --max-steps 10
```

**Reading `--check-cube-frame`.** Compare *positions*; do not read the raw
quaternions as an error. The bar is symmetric, so the estimator's quaternion and
the clip's routinely differ by one of the 8 symmetries -- a 90 deg roll about the
long axis shows up as an ~87 deg discrepancy that is not a discrepancy at all.
What matters is the residual the placement snap prints after canonicalisation.

**Do not reposition the bar by hand to match a clip.** With `--bank-index
nearest` the runner reads the live bar, converts it to the bank's (translation,
yaw) parameterisation and plays the clip recorded for where the bar actually is;
the bank is dense enough that the residual lands in the low millimetres. Moving
the bar to chase entry 122 is the old workflow and is not needed.

## 4b. The MuJoCo window

On by default, as in the earlier deployments: the robot in its **measured
pose**, the **bar** at the pose the policy is being shown (the estimator's or
the demonstration's, whichever the run uses), and the **green ghost** replaying
the reference clip beside it. `--no-viewer` runs without it, `--no-ghost` drops
the ghost, and closing the window stops the run.

It is `sim2sim.MujocoSim` with the viewer on and **physics never stepped**:
`reset()` writes the measured joints and the observed bar pose and calls
`mj_forward`, so MuJoCo only answers "where is everything, given this state".
Measured cost at 60 Hz: 300 steps in 5.1 s against 5.0 s without it, 2
overruns. `--viewer-hz` caps the redraw rate if a slower machine needs it, and
`--viewer-hold-seconds` (5) keeps the window up afterwards -- after an abort
the last drawn frame is the one that tripped the monitor.

## 5. Safety monitors

| Monitor | Default | What it does |
| --- | --- | --- |
### `--low-safety`: moving the bar during the rollout

To film the policy recovering from a bar moved by hand mid-grasp, the monitors
that would stop it first are the ones asking *is the robot still doing what the
demonstration did* -- and a robustness demo answers that "no" deliberately.
`--low-safety` raises exactly those:

| raised to | |
| --- | --- |
| `--max-palm-reference-error-m` | 1.00 |
| `--max-arm-reference-error-rad` | 3.14 |
| `--max-hand-reference-error-rad` | 3.14 |
| `--max-action-step` / `--max-hand-action-step` | 5.0 |
| `--pose-jump-reject-m` | 1.00 (so a bar you really moved is tracked at once, not after 3 samples) |
| `--pose-timeout` | 5.0 (your own hand occludes the tags while you move it) |

It deliberately leaves the **motion** limits alone: `--max-arm-step-rad`,
`--max-hand-step-rad`, the contract's velocity slew, the IK joint clamp, the
joint limits, the arming prompt and the braking path are all unchanged. A demo
needs the run not to abort; it does not need the arm to move faster. Anything
set explicitly on the command line overrides the preset, and the banner prints
every value it changed.

Keep the e-stop in hand: with these limits the run will follow the bar wherever
you put it, including somewhere the policy was never trained to reach.

| `--spike-mode` | `stop`, or `warn` under `--debug-step` | Aborts on a single-step raw-action jump above `--max-action-step` (arm, 1.8) or `--max-hand-action-step` (fingers, 2.2), after `--spike-grace-steps` (2). The hand gets the looser limit because every spike seen on the robot has been a finger: the policy observes measured hand joints that lag their targets, and the arm has no equivalent. **Stepping by hand downgrades it to a warning**, because the spike is then an artefact of the wait: seconds pass between policy steps while the soft hand keeps creeping toward its target, so the observation advances far more than one control period's worth. Measured on this checkpoint, 0.16 rad of hand sag -- the value `home_hand` reports after softening -- is worth 1.14 of action, while a 30 mm jump in the observed cube is worth only 0.38. For reference the whole ideal rollout peaks at 0.3845, so 1.8 is about five times the nominal worst case; it is the stepping that is unrepresentative, not the threshold. Pass the flag explicitly to override. |
| `--reference-error-mode` | `stop` | Aborts when the robot leaves the reference clip: the palm by `--max-palm-reference-error-m` (0.15 -- a whole bar length, so it catches gross runaway rather than a drifting grasp), hand joints by `termination.hand_position_threshold_rad` (1.35), arm joints by `--max-arm-reference-error-rad` (1.5). The palm limit is an operational number chosen on the robot, **not** `termination.palm_keypoint_threshold_m` (0.08): training terminated on the RMS of four palm keypoints in the *cuboid's* frame, which also carries orientation at 0.1 m/rad, whereas this is the distance between two palm centres in the robot base frame. Borrowing the training number for a different quantity was misleading, so the tie is cut. Measured against the *reference*, not the target. **The palm is the criterion meant to fire**: the policy is a closed-loop grasper, not a trajectory tracker, so it may legitimately leave the clip in joint space while the palm stays put; the arm-joint limit is deliberately loose, a runaway backstop only. The palm monitor watches position only — its orientation error is printed in degrees but never aborts. |
| `--max-arm-step-rad` / `--max-hand-step-rad` | `0.02` / `0.05` | Hard per-tick clamp on the sent target, on top of the contract's own 0.052 rad velocity-limit slew. |
| `--target-smoothing` | `0` | EMA on the sent target. 0.5-0.7 on the first free-running rungs, then back to 0. |
| `--current-warning-ma` | `170` | Reports DG5F motor current above the driver's own limiter threshold. |

Ctrl+C, any abort, and any exception brake the arm (zero-velocity hold, then
`{"stop": true}`) and park the hand at its measured position.

Two flags change what the policy *sees*:

- `--simulated-state-source` (default `reference`) -- state fed back for a
  subsystem not read from hardware. `reference` replays the clip and its
  recorded velocities, reproducing the training observation. `target` assumes
  the subsystem tracks its own command, velocity-clamped; a short-run
  diagnostic, not a substitute.
- `--previous-target-source` (default `commanded`) -- `commanded` is what
  training fed back; `applied` reports what the limiter actually sent. They
  differ whenever the limiter is active, which on the early rungs is most of
  the time.
- `--arm-velocity-source` (default `controller`) -- the controller's state
  message carries `Qd = getTargetQd()`, UR's *target* joint velocity, where
  training observed the measured one; `finite-difference` differentiates the
  measured `Q` across polls instead. The previous deployments ran on `Qd`.

## 6. What has and has not been tested

Verified on this machine (UR5-PC, 2026-09-22, branch `generalize_size`),
without robots:

- `tests/test_deployment_contract.py` + `tests/test_hand_stiffness.py`: 20
  tests pass (112-D and 113-D checkpoints).
- Stage 1 against the `generalize_size` environment: `sc2_anchor` from frame
  0 on entry 122 at scale 1.0 (1107 steps) and from frame 760 on entry 500 at
  scale 0.8 (347 steps); `dr_combo` from frame 0 at scale 1.1 (1107 steps).
  In every case the observation is within 7.2e-7 of the environment's in every
  block (scale column exact), commanded and applied targets within 2.4e-7 rad
  per step, and the free-running pipeline within 8.4e-7 rad over the clip --
  float32 noise. The earlier 112-D checkpoint passed the same check on
  `throughput-and-hygiene`.
- Stage 2 dry run: the whole clip in 6.4 s (about 6 ms per step against the
  16.7 ms budget), no action spike above 0.38.

**Not** tested:
anything against real hardware -- the arm and hand clients are the ones the
earlier deployment exercised against fake endpoints, ported with their
protocols unchanged plus the controller's `stop` message on brake -- and
`--cube-source pose-estimation`, whose frame equivalence with the demonstration
is assumed and must be confirmed with `--check-cube-frame` before use.

