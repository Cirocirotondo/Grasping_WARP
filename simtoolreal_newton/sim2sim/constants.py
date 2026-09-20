"""Names, frames and offsets the MuJoCo sim2sim shares with the training env.

Everything here duplicates a constant of ``envs/motion_imitation_env.py`` on
purpose: that module imports Isaac Lab at load time and the sim2sim must run
without it. ``tests/test_sim2sim.py`` re-derives each value from the URDF
through :class:`~simtoolreal_newton.envs.kinematics.PalmKinematics`, exactly
as the environment does at startup, so the two cannot drift apart silently.
"""

from __future__ import annotations

import numpy as np

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.envs.controller import (  # noqa: F401  (re-exported)
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
    WRIST_BODY_NAME,
)

ACTION_DIM = len(JOINT_NAMES)
BASE_OBSERVATION_DIM = 112

# The URDF the training kinematics and the USD conversion come from.
ROBOT_URDF = ROOT_DIR / "assets" / "ur5e_right_dg5f.urdf"
ROBOT_BASE_BODY_NAME = "base_link"

# The fixed wrist -> mount -> base -> palm chain is merged by the Newton
# importer; the environment rebuilds the rl_dg_palm frame from wrist_3_link
# with these two constants (xyzw quaternion).
PALM_POSITION_IN_WRIST = np.asarray((0.0, 0.0, 0.0738), dtype=np.float64)
PALM_ORIENTATION_IN_WRIST_XYZW = np.asarray(
    (0.0, 0.0, 0.5, 0.8660254037844386), dtype=np.float64
)
PALM_LINK_NAME = "rl_dg_palm"

# Finger 1 is the thumb. The observation reads the distal phalanx body and
# adds the fixed tip offset, because the tip's own fixed joint is merged away
# in the training articulation.
FINGERTIP_BODY_NAMES = tuple("rl_dg_{}_4".format(finger) for finger in range(1, 6))
FINGERTIP_LINK_NAMES = tuple("rl_dg_{}_tip".format(finger) for finger in range(1, 6))
FINGERTIP_OFFSETS = np.asarray(
    (
        (0.0, 0.0363, 0.0),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0363),
    ),
    dtype=np.float64,
)
FINGER_BODY_NAMES = tuple(
    "rl_dg_{}_{}".format(finger, link) for finger in range(1, 6) for link in range(1, 5)
)

# MuJoCo keeps the fixed links as bodies of their own (``fusestatic`` off, so
# the palm and the tips can be checked against the kinematics). The Newton
# articulation merges them into these parents, and the self-collision filter
# is expressed in Newton's body names, so each Newton body maps to a group of
# MuJoCo bodies.
MUJOCO_BODY_GROUPS = {
    WRIST_BODY_NAME: (
        WRIST_BODY_NAME,
        "flange",
        "tool0",
        "rl_dg_mount",
        "rl_dg_base",
        PALM_LINK_NAME,
    ),
}
for _finger in range(1, 6):
    MUJOCO_BODY_GROUPS["rl_dg_{}_4".format(_finger)] = (
        "rl_dg_{}_4".format(_finger),
        "rl_dg_{}_tip".format(_finger),
    )
# Newton body names of the hand: the merged wrist plus the twenty phalanges.
HAND_BODY_NAMES = (WRIST_BODY_NAME,) + FINGER_BODY_NAMES

# Demonstration (UR base) frame -> env/world frame: x and y mirrored.
WORLD_AXIS_SIGN = np.asarray((-1.0, -1.0, 1.0), dtype=np.float64)

# Collision bitmask groups.
CUBE_BIT = 1
HAND_BIT = 2
TABLE_BIT = 4
