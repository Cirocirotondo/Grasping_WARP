"""Batched forward kinematics and geometric Jacobians straight from the URDF.

Isaac Gym handed the environment a Jacobian tensor. Isaac Lab's Newton
backend can too, but the palm the controller steers is a *merged* fixed-joint
frame that no simulator body carries, and the arm's control path should not
depend on which physics backend happens to be running. So the kinematic model
is built here from the URDF once, in plain PyTorch, and used for:

* the palm pose and its ``(6, 6)`` Jacobian over the six arm joints, which the
  in-loop damped least-squares IK consumes every step;
* a cross-check of the simulator's body poses at startup;
* the offline retargeting tools, which no longer need ``pytorch_kinematics``.

Conventions match the rest of the repository: quaternions are ``(x, y, z, w)``,
transforms are ``(B, 4, 4)`` matrices, and the base frame is the URDF root.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from simtoolreal_newton.envs.rotations import matrix_to_quat


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: torch.Tensor  # (4, 4)
    axis: torch.Tensor  # (3,)
    lower: float
    upper: float


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


def _origin_matrix(element: Optional[ET.Element]) -> torch.Tensor:
    matrix = torch.eye(4, dtype=torch.float64)
    if element is None:
        return matrix
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    matrix[:3, :3] = _rpy_to_matrix(*rpy)
    matrix[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return matrix


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """``(B, 3, 3)`` rotation about a unit ``axis`` by ``angle`` ``(B,)``."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    x, y, z = axis[0], axis[1], axis[2]
    rot = torch.empty(angle.shape[0], 3, 3, dtype=angle.dtype, device=angle.device)
    rot[:, 0, 0] = c + x * x * one_c
    rot[:, 0, 1] = x * y * one_c - z * s
    rot[:, 0, 2] = x * z * one_c + y * s
    rot[:, 1, 0] = y * x * one_c + z * s
    rot[:, 1, 1] = c + y * y * one_c
    rot[:, 1, 2] = y * z * one_c - x * s
    rot[:, 2, 0] = z * x * one_c - y * s
    rot[:, 2, 1] = z * y * one_c + x * s
    rot[:, 2, 2] = c + z * z * one_c
    return rot


class UrdfKinematics:
    """Forward kinematics over the whole URDF tree, batched over configurations."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        resolved = Path(urdf_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError("URDF not found: {}".format(resolved))
        self.path = resolved
        self.device = torch.device(device)
        self.dtype = dtype
        root = ET.parse(str(resolved)).getroot()
        self.link_names: List[str] = [link.get("name") for link in root.findall("link")]
        joints: Dict[str, _Joint] = {}
        for element in root.findall("joint"):
            name = element.get("name")
            joint_type = element.get("type")
            parent = element.find("parent").get("link")
            child = element.find("child").get("link")
            axis_element = element.find("axis")
            axis = [1.0, 0.0, 0.0]
            if axis_element is not None:
                axis = [float(v) for v in axis_element.get("xyz").split()]
            axis_tensor = torch.tensor(axis, dtype=torch.float64)
            if joint_type in ("revolute", "continuous", "prismatic"):
                axis_tensor = axis_tensor / axis_tensor.norm()
            limit = element.find("limit")
            lower = float(limit.get("lower", "-inf")) if limit is not None else -math.inf
            upper = float(limit.get("upper", "inf")) if limit is not None else math.inf
            joints[name] = _Joint(
                name,
                joint_type,
                parent,
                child,
                _origin_matrix(element.find("origin")),
                axis_tensor,
                lower,
                upper,
            )
        self.joints = joints
        self.parent_joint: Dict[str, str] = {j.child: j.name for j in joints.values()}
        children = {link: [] for link in self.link_names}
        for joint in joints.values():
            children[joint.parent].append(joint.name)
        self.children = children
        roots = [link for link in self.link_names if link not in self.parent_joint]
        if len(roots) != 1:
            raise ValueError("Expected exactly one root link, found {}".format(roots))
        self.root_link = roots[0]
        self.actuated_joint_names: Tuple[str, ...] = tuple(
            name for name, joint in joints.items() if joint.joint_type != "fixed"
        )
        # Topological (parent-before-child) joint order for the forward pass.
        order: List[str] = []
        stack = [self.root_link]
        while stack:
            link = stack.pop()
            for joint_name in children[link]:
                order.append(joint_name)
                stack.append(joints[joint_name].child)
        self._joint_order = order
        self._origins = {
            name: joint.origin.to(device=self.device, dtype=dtype) for name, joint in joints.items()
        }
        self._axes = {name: joint.axis.to(device=self.device, dtype=dtype) for name, joint in joints.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def chain_to(self, link_name: str) -> List[str]:
        """Joint names from the root down to ``link_name``, in order."""
        if link_name not in self.link_names:
            raise KeyError("Unknown link {!r}".format(link_name))
        chain: List[str] = []
        link = link_name
        while link in self.parent_joint:
            joint_name = self.parent_joint[link]
            chain.append(joint_name)
            link = self.joints[joint_name].parent
        chain.reverse()
        return chain

    def actuated_chain_to(self, link_name: str) -> List[str]:
        return [name for name in self.chain_to(link_name) if self.joints[name].joint_type != "fixed"]

    def joint_limits(self, joint_names: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = torch.tensor([self.joints[n].lower for n in joint_names], dtype=self.dtype, device=self.device)
        upper = torch.tensor([self.joints[n].upper for n in joint_names], dtype=self.dtype, device=self.device)
        return lower, upper

    def fixed_offset(self, parent_link: str, child_link: str) -> torch.Tensor:
        """``(4, 4)`` transform of ``child_link`` in ``parent_link`` through fixed joints only."""
        chain = self.chain_to(child_link)
        parent_chain = self.chain_to(parent_link)
        if chain[: len(parent_chain)] != parent_chain:
            raise ValueError("{} is not an ancestor of {}".format(parent_link, child_link))
        matrix = torch.eye(4, dtype=self.dtype, device=self.device)
        for joint_name in chain[len(parent_chain) :]:
            joint = self.joints[joint_name]
            if joint.joint_type != "fixed":
                raise ValueError("Joint {} between {} and {} is not fixed".format(joint_name, parent_link, child_link))
            matrix = matrix @ self._origins[joint_name]
        return matrix

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def _joint_values(
        self, q: torch.Tensor, joint_names: Sequence[str]
    ) -> Dict[str, torch.Tensor]:
        if q.ndim != 2 or q.shape[1] != len(joint_names):
            raise ValueError("q must have shape (B, {}), got {}".format(len(joint_names), tuple(q.shape)))
        q = q.to(device=self.device, dtype=self.dtype)
        return {name: q[:, index] for index, name in enumerate(joint_names)}

    def forward(
        self,
        q: torch.Tensor,
        joint_names: Sequence[str],
        link_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Poses ``(B, 4, 4)`` of the requested links in the root frame.

        Actuated joints absent from ``joint_names`` are held at zero.
        """
        values = self._joint_values(q, joint_names)
        batch = q.shape[0]
        eye = torch.eye(4, dtype=self.dtype, device=self.device).expand(batch, 4, 4)
        poses: Dict[str, torch.Tensor] = {self.root_link: eye}
        wanted = set(self.link_names if link_names is None else link_names)
        for joint_name in self._joint_order:
            joint = self.joints[joint_name]
            parent_pose = poses[joint.parent]
            local = self._origins[joint_name].expand(batch, 4, 4)
            if joint.joint_type == "fixed":
                child_pose = parent_pose @ local
            else:
                angle = values.get(joint_name)
                if angle is None:
                    angle = torch.zeros(batch, dtype=self.dtype, device=self.device)
                motion = eye.clone()
                if joint.joint_type == "prismatic":
                    motion[:, :3, 3] = self._axes[joint_name].unsqueeze(0) * angle.unsqueeze(1)
                else:
                    motion[:, :3, :3] = _axis_angle_matrix(self._axes[joint_name], angle)
                child_pose = parent_pose @ local @ motion
            poses[joint.child] = child_pose
        return {name: pose for name, pose in poses.items() if name in wanted}

    def pose_and_jacobian(
        self, q: torch.Tensor, joint_names: Sequence[str], link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pose ``(B, 4, 4)`` of ``link_name`` and its geometric Jacobian.

        The Jacobian is ``(B, 6, len(joint_names))``, ``[linear; angular]``,
        expressed in the root frame and referenced at the link origin. Columns
        follow ``joint_names``; joints not on the chain to the link get zero
        columns.
        """
        values = self._joint_values(q, joint_names)
        batch = q.shape[0]
        eye = torch.eye(4, dtype=self.dtype, device=self.device).expand(batch, 4, 4)
        pose = eye
        column_of = {name: index for index, name in enumerate(joint_names)}
        axes_world: List[Tuple[int, torch.Tensor, torch.Tensor, str]] = []
        for joint_name in self.chain_to(link_name):
            joint = self.joints[joint_name]
            frame = pose @ self._origins[joint_name].expand(batch, 4, 4)
            if joint.joint_type == "fixed":
                pose = frame
                continue
            angle = values.get(joint_name)
            if angle is None:
                angle = torch.zeros(batch, dtype=self.dtype, device=self.device)
            axis_world = frame[:, :3, :3] @ self._axes[joint_name]
            origin_world = frame[:, :3, 3]
            motion = eye.clone()
            if joint.joint_type == "prismatic":
                motion[:, :3, 3] = self._axes[joint_name].unsqueeze(0) * angle.unsqueeze(1)
            else:
                motion[:, :3, :3] = _axis_angle_matrix(self._axes[joint_name], angle)
            pose = frame @ motion
            if joint_name in column_of:
                axes_world.append((column_of[joint_name], axis_world, origin_world, joint.joint_type))
        jacobian = torch.zeros(batch, 6, len(joint_names), dtype=self.dtype, device=self.device)
        target = pose[:, :3, 3]
        for column, axis_world, origin_world, joint_type in axes_world:
            if joint_type == "prismatic":
                jacobian[:, :3, column] = axis_world
            else:
                jacobian[:, :3, column] = torch.cross(axis_world, target - origin_world, dim=1)
                jacobian[:, 3:, column] = axis_world
        return pose, jacobian


class PalmKinematics:
    """The UR5e arm -> DG5F palm chain, as the environment and the tools use it.

    ``arm_q`` is ``(B, 6)`` in :data:`controller.ARM_JOINT_NAMES` order; every
    output is in the robot base frame, which the environment asserts is the
    world frame up to a translation.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        palm_link: str = "rl_dg_palm",
        fingertip_links: Sequence[str] = ("rl_dg_1_tip", "rl_dg_2_tip", "rl_dg_3_tip", "rl_dg_4_tip", "rl_dg_5_tip"),
    ) -> None:
        from simtoolreal_newton.envs.controller import ARM_JOINT_NAMES, HAND_JOINT_NAMES

        self.kinematics = UrdfKinematics(urdf_path, device=device, dtype=dtype)
        self.device = self.kinematics.device
        self.dtype = dtype
        self.palm_link = palm_link
        self.fingertip_links = tuple(fingertip_links)
        self.arm_joint_names = tuple(ARM_JOINT_NAMES)
        self.hand_joint_names = tuple(HAND_JOINT_NAMES)
        self.joint_names = self.arm_joint_names + self.hand_joint_names
        chain = self.kinematics.actuated_chain_to(palm_link)
        if tuple(chain) != self.arm_joint_names:
            raise ValueError(
                "The actuated chain to {} is {}, expected the six arm joints".format(palm_link, chain)
            )
        lower, upper = self.kinematics.joint_limits(self.joint_names)
        self.lower_limits = lower
        self.upper_limits = upper

    def palm_matrices(self, arm_q: torch.Tensor) -> torch.Tensor:
        """``(B, 4, 4)`` palm poses in the base frame."""
        pose, _ = self.kinematics.pose_and_jacobian(arm_q, self.arm_joint_names, self.palm_link)
        return pose

    def palm_pose(self, arm_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3)`` position and ``(B, 4)`` xyzw quaternion of the palm."""
        pose = self.palm_matrices(arm_q)
        return pose[:, :3, 3], matrix_to_quat(pose[:, :3, :3])

    def jacobian(self, arm_q: torch.Tensor) -> torch.Tensor:
        """``(B, 6, 6)`` geometric Jacobian at the palm origin, ``[linear; angular]``."""
        _, jacobian = self.kinematics.pose_and_jacobian(arm_q, self.arm_joint_names, self.palm_link)
        return jacobian

    def palm_pose_and_jacobian(self, arm_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose, jacobian = self.kinematics.pose_and_jacobian(arm_q, self.arm_joint_names, self.palm_link)
        return pose[:, :3, 3], matrix_to_quat(pose[:, :3, :3]), jacobian

    def link_poses(self, joint_positions: torch.Tensor, link_names: Iterable[str]) -> Dict[str, torch.Tensor]:
        """Poses of arbitrary links from all 26 joints (demonstration order)."""
        return self.kinematics.forward(joint_positions, self.joint_names, link_names)

    def fingertip_positions(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """``(B, 5, 3)`` fingertip positions in the base frame from all 26 joints."""
        poses = self.link_poses(joint_positions, self.fingertip_links)
        return torch.stack([poses[name][:, :3, 3] for name in self.fingertip_links], dim=1)

    def hand_keypoints(self, joint_positions: torch.Tensor, lever_arm_m: float) -> torch.Tensor:
        """``(B, 9, 3)`` hand keypoints in the base frame, from all 26 joints."""
        from simtoolreal_newton.envs.keypoints import hand_keypoints

        poses = self.link_poses(joint_positions, (self.palm_link,) + self.fingertip_links)
        palm = poses[self.palm_link]
        fingertips = torch.stack([poses[name][:, :3, 3] for name in self.fingertip_links], dim=1)
        return hand_keypoints(palm[:, :3, 3], matrix_to_quat(palm[:, :3, :3]), fingertips, lever_arm_m)

    def elbow_height_margin(self, arm_q: torch.Tensor) -> torch.Tensor:
        """Elbow (forearm origin) height minus wrist-1 height, per configuration."""
        poses = self.kinematics.forward(arm_q, self.arm_joint_names, ("forearm_link", "wrist_1_link"))
        return poses["forearm_link"][:, 2, 3] - poses["wrist_1_link"][:, 2, 3]

    def palm_offset_in_wrist(self) -> torch.Tensor:
        """``(4, 4)`` pose of the palm in ``wrist_3_link`` (the merged fixed chain)."""
        return self.kinematics.fixed_offset("wrist_3_link", self.palm_link)

    def fingertip_offsets(self) -> torch.Tensor:
        """``(5, 3)`` fixed tip origins in their ``rl_dg_<finger>_4`` parent frames."""
        offsets = []
        for link in self.fingertip_links:
            parent = self.kinematics.joints[self.kinematics.parent_joint[link]].parent
            offsets.append(self.kinematics.fixed_offset(parent, link)[:3, 3])
        return torch.stack(offsets)
