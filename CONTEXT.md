# simtoolreal_newton

Trains a UR5e + DG5F hand policy to grasp and lift a cuboid by imitating one recorded demonstration, in Newton/MuJoCo-Warp, for transfer to the real robot.

## Language

**Demonstration**:
The single recorded 60 Hz trajectory (arm and hand joints plus cuboid pose) the policy imitates and that reference-state initialisation starts episodes from.
_Avoid_: Demo trajectory, reference motion, motion file

**Self-penetration**:
Two links of the robot occupying the same volume, which the simulator allows because hand self-collision is switched off. Distinct from object penetration.
_Avoid_: Penetration (bare), interpenetration, self-collision (that is the simulator feature that prevents it)

**Object penetration**:
A robot link occupying the cuboid's volume; the simulator resists it with contact forces.
_Avoid_: Penetration (bare), crushing

**Self-collision**:
The simulator feature that generates contacts between links of the same robot. Since the SC1 wave it is on for the four long fingers against each other only; palm, thumb and same-finger pairs stay filtered.
_Avoid_: Self-contact, finger contact

**Transform bank entry**:
One planar bar pose (x, y, yaw) for which the demonstration was retargeted; every episode follows one entry. Entry 122 is the demonstration's own pose.
_Avoid_: Pose index (bare), bank pose, RSI pose

**Bar scale**:
The per-episode factor applied to the bar's three sides (mass with the volume), drawn from the training range and, when the policy observes it, appended to the observation.
_Avoid_: Cube size, size factor

**Sim2sim**:
Replaying a checkpoint in native MuJoCo instead of the training simulator, with the same scene, drives and contact model, to measure what survives a change of physics engine. The transfer measure that stands in for the real robot.
_Avoid_: MuJoCo test, cross-sim

**Domain randomization**:
Per-environment variation of physical parameters the policy cannot observe (bar mass, hand drive gains, link masses, contact friction, control delay) so that it cannot fit one simulator's values. Distinct from bar scale and bar pose, which are task variation.
_Avoid_: DR (in prose), noise (that is sensor noise), randomization (bare)

**Sensor noise**:
Per-step Gaussian error added to what the policy observes (joint angles and velocities, the bar's pose in the palm frame), optionally with a constant per-episode bias. Applied to the observation only; rewards and terminations use the true state.
_Avoid_: Observation noise, measurement error, domain randomization

**Impulse**:
A brief random push on the bar or on a robot link, sparse in time, so the policy sees a world that acts on it. Part of the disturbance family, not of domain randomization proper.
_Avoid_: Perturbation, force field, disturbance (bare)

**Family arm**:
A training run in which one family of randomization (or all of them) is switched on, drawn several times so its effect can be told from the run-to-run spread.
_Avoid_: Ablation run, single-lever run
