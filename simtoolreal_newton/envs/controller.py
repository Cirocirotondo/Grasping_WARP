"""UR5e + DG5F joint layout and low-level PD gains.

This module carries the *names* and *numbers* that every other module agrees
on: the demonstration's joint order, the fixed drive gains, and the bodies the
observation reads. It is free of any simulator import, unlike the Isaac Gym
original it replaces, so tests and offline tools can use it without a GPU.
"""

from typing import Dict, Tuple

import numpy as np

from simtoolreal_newton.envs.pd_gains import (
    ARM_PD_DAMPING,
    ARM_PD_STIFFNESS,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
    scale_gains,
)

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = tuple(
    "rj_dg_{}_{}".format(finger, joint)
    for finger in range(1, 6)
    for joint in range(1, 5)
)
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES

# The body the (collapsed) palm chain hangs from, and the two hand links whose
# meshes intersect it at rest. Kept for documentation and tests.
WRIST_BODY_NAME = "wrist_3_link"
WRIST_COLLISION_HAND_BODY_NAMES = ("rl_dg_1_2", "rl_dg_4_2")

# Bodies that belong to the arm proper (everything before the hand assembly).
# With fixed joints merged, the DG5F mount/base/palm shapes live on
# ``wrist_3_link``, which therefore counts as part of the hand.
ARM_BODY_NAMES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
)
PALM_LINK_NAME = "rl_dg_palm"


def pd_gain_tables(
    arm_stiffness_scale: float = 1.0,
    arm_damping_scale: float = 1.0,
    hand_stiffness_scale: float = 1.0,
    hand_damping_scale: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-joint ``(stiffness, damping)`` dictionaries in demonstration order.

    Lowering a joint's stiffness lowers its closed-loop bandwidth
    ``omega = sqrt(k / J)``, so the drive filters the policy's step-to-step
    chatter instead of tracking it into the joint, and raises the damping
    ratio ``zeta = d / (2 sqrt(k J))`` at the same time.
    """
    arm_k = scale_gains(ARM_PD_STIFFNESS, arm_stiffness_scale)
    arm_d = scale_gains(ARM_PD_DAMPING, arm_damping_scale)
    hand_k = scale_gains(HAND_PD_STIFFNESS, hand_stiffness_scale)
    hand_d = scale_gains(HAND_PD_DAMPING, hand_damping_scale)
    stiffness = dict(zip(ARM_JOINT_NAMES, arm_k))
    stiffness.update(zip(HAND_JOINT_NAMES, hand_k))
    damping = dict(zip(ARM_JOINT_NAMES, arm_d))
    damping.update(zip(HAND_JOINT_NAMES, hand_d))
    return stiffness, damping


def pd_gain_arrays(**scales) -> Tuple[np.ndarray, np.ndarray]:
    """The same gains as :func:`pd_gain_tables`, as ``(26,)`` arrays."""
    stiffness, damping = pd_gain_tables(**scales)
    return (
        np.asarray([stiffness[name] for name in JOINT_NAMES], dtype=np.float32),
        np.asarray([damping[name] for name in JOINT_NAMES], dtype=np.float32),
    )


def validate_joint_names(names) -> np.ndarray:
    """Return the permutation mapping demonstration order onto ``names``."""
    names = tuple(names)
    missing = sorted(set(JOINT_NAMES) - set(names))
    extra = sorted(set(names) - set(JOINT_NAMES))
    if missing or extra or len(names) != len(JOINT_NAMES):
        raise ValueError(
            "Robot DOFs do not match the demonstration; missing={}, extra={}".format(
                missing, extra
            )
        )
    return np.asarray([names.index(name) for name in JOINT_NAMES], dtype=np.int64)
