"""The MuJoCo sim2sim against the contracts of the training environment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.cfg import SimToolRealCfg, update_config_from_dict
from simtoolreal_newton.envs.kinematics import PalmKinematics
from simtoolreal_newton.envs.rotations import matrix_to_quat, quat_to_rotation_6d
from simtoolreal_newton.sim2sim import constants
from simtoolreal_newton.sim2sim.controller import ActionPipeline
from simtoolreal_newton.sim2sim.mujoco_sim import MujocoSceneConfig, MujocoSim
from simtoolreal_newton.sim2sim.observation import build_observation
from simtoolreal_newton.sim2sim.policy import apply_overrides
from simtoolreal_newton.sim2sim.reference import ReferenceTrack

STAGED_CONFIG = ROOT_DIR / "logs" / "staged" / "sc2_anchor_s42_it17300" / "config.json"
BANK = ROOT_DIR / "banks" / "stage1_box.pt"


def _env_cfg(self_collision: bool = True) -> SimToolRealCfg:
    cfg = SimToolRealCfg()
    if STAGED_CONFIG.is_file():
        with STAGED_CONFIG.open() as stream:
            update_config_from_dict(cfg, json.load(stream)["env_cfg"], strict=False)
    cfg.asset.self_collision = self_collision
    cfg.object_randomization.bank_path = "banks/stage1_box.pt"
    return cfg


class ConstantsTest(unittest.TestCase):
    def test_constants_match_the_urdf_kinematics(self):
        kinematics = PalmKinematics(constants.ROBOT_URDF)
        offset = kinematics.palm_offset_in_wrist()
        np.testing.assert_allclose(offset[:3, 3].numpy(), constants.PALM_POSITION_IN_WRIST, atol=1e-6)
        np.testing.assert_allclose(
            np.abs(matrix_to_quat(offset[:3, :3]).numpy()),
            np.abs(constants.PALM_ORIENTATION_IN_WRIST_XYZW),
            atol=1e-6,
        )
        np.testing.assert_allclose(kinematics.fingertip_offsets().numpy(), constants.FINGERTIP_OFFSETS, atol=1e-6)
        self.assertEqual(constants.ACTION_DIM, 26)
        self.assertEqual(constants.BASE_OBSERVATION_DIM, 112)


class ObservationTest(unittest.TestCase):
    def _state(self):
        rng = np.random.default_rng(0)

        def quat():
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            return q if q[3] >= 0 else -q

        return {
            "joint_positions": rng.uniform(-1.0, 1.0, 26),
            "joint_velocities": rng.normal(size=26),
            "robot_position_world": np.asarray((0.0, 0.6, 0.55)),
            "robot_orientation_world_xyzw": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "wrist_position_world": rng.normal(size=3),
            "wrist_orientation_world_xyzw": quat(),
            "fingertip_body_positions_world": rng.normal(size=(5, 3)),
            "fingertip_body_orientations_world_xyzw": np.stack([quat() for _ in range(5)]),
            "cube_position_world": rng.normal(size=3),
            "cube_orientation_world_xyzw": quat(),
        }

    def test_layout_is_112_plus_optional_scale(self):
        cfg = _env_cfg()
        track_symmetries = ReferenceTrack.load(cfg).symmetries if BANK.is_file() else torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        state = self._state()
        lower, upper = np.full(26, -2.0), np.full(26, 2.0)
        targets = np.linspace(-0.5, 0.5, 26)
        base = build_observation(state, targets, 0.25, lower, upper, track_symmetries, 0)
        self.assertEqual(base.shape, (112,))
        self.assertEqual(base.dtype, np.float32)
        np.testing.assert_allclose(base[0:26], np.clip(2 * (state["joint_positions"] + 2) / 4 - 1, -1, 1), atol=1e-6)
        np.testing.assert_allclose(base[26:52], targets, atol=1e-6)
        np.testing.assert_allclose(base[52:78], state["joint_velocities"], atol=1e-6)
        self.assertAlmostEqual(float(base[78]), 0.25)
        # Rotation blocks are orthonormal 6D pairs.
        for start in (82, 103):
            first, second = base[start : start + 3], base[start + 3 : start + 6]
            self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)
            self.assertAlmostEqual(float(np.dot(first, second)), 0.0, places=5)
        scaled = build_observation(state, targets, 0.25, lower, upper, track_symmetries, 0, scale=1.2)
        self.assertEqual(scaled.shape, (113,))
        np.testing.assert_allclose(scaled[:112], base)
        self.assertAlmostEqual(float(scaled[112]), 1.2, places=6)
        told = build_observation(state, targets, 0.25, lower, upper, track_symmetries, 0, scale=1.2, observed_scale_override=0.8)
        self.assertAlmostEqual(float(told[112]), 0.8, places=6)

    def test_palm_block_uses_the_robot_frame(self):
        state = self._state()
        state["wrist_orientation_world_xyzw"] = np.asarray((0.0, 0.0, 0.0, 1.0))
        state["wrist_position_world"] = np.asarray((0.1, 0.8, 0.9))
        obs = build_observation(state, np.zeros(26), 0.0, np.full(26, -2.0), np.full(26, 2.0), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), 0)
        np.testing.assert_allclose(obs[79:82], np.asarray((0.1, 0.2, 0.35)) + constants.PALM_POSITION_IN_WRIST, atol=1e-6)
        expected = quat_to_rotation_6d(torch.as_tensor(constants.PALM_ORIENTATION_IN_WRIST_XYZW, dtype=torch.float32))
        np.testing.assert_allclose(obs[82:88], expected.numpy(), atol=1e-6)


class ActionPipelineTest(unittest.TestCase):
    def _pipeline(self, alpha=0.3):
        cfg = _env_cfg()
        cfg.control.action_filter_alpha = alpha
        kinematics = PalmKinematics(constants.ROBOT_URDF)
        lower, upper = kinematics.lower_limits.numpy(), kinematics.upper_limits.numpy()
        return cfg, ActionPipeline(cfg, lower, upper, np.full(26, np.pi), kinematics)

    def test_zero_action_holds_the_arm_and_returns_the_hand_to_default(self):
        cfg, pipeline = self._pipeline(alpha=1.0)
        q = np.asarray(list(cfg.init_state.default_arm_joint_angles) + list(cfg.init_state.default_hand_joint_angles))
        pipeline.reset(q)
        applied = pipeline.command(np.zeros(26), q[:6])
        np.testing.assert_allclose(applied, q, atol=1e-6)
        np.testing.assert_allclose(pipeline.previous_targets, q, atol=1e-6)

    def test_hand_residual_scale_clip_and_slew(self):
        cfg, pipeline = self._pipeline(alpha=1.0)
        q = np.asarray(list(cfg.init_state.default_arm_joint_angles) + list(cfg.init_state.default_hand_joint_angles))
        pipeline.reset(q)
        action = np.zeros(26)
        action[6] = 100.0  # far beyond the clip
        applied = pipeline.command(action, q[:6])
        commanded = pipeline.previous_targets
        self.assertAlmostEqual(commanded[6] - q[6], min(100.0 * cfg.control.scale_hand_joint_target, cfg.control.clip_joint_target), places=6)
        # The drive receives the target slewed at the URDF velocity limit.
        self.assertAlmostEqual(applied[6] - q[6], np.pi * pipeline.dt, places=6)

    def test_arm_twist_moves_the_palm_along_the_requested_axis(self):
        cfg, pipeline = self._pipeline(alpha=1.0)
        q = np.asarray(list(cfg.init_state.default_arm_joint_angles) + list(cfg.init_state.default_hand_joint_angles))
        pipeline.reset(q)
        kinematics = pipeline.kinematics
        before = kinematics.palm_pose(torch.as_tensor(q[:6], dtype=torch.float32).unsqueeze(0))[0][0].numpy()
        action = np.zeros(26)
        action[2] = 1.0  # +z at full speed
        pipeline.command(action, q[:6])
        after_q = pipeline.previous_targets[:6]
        after = kinematics.palm_pose(torch.as_tensor(after_q, dtype=torch.float32).unsqueeze(0))[0][0].numpy()
        step = after - before
        expected = cfg.control.arm_translation_speed_m_per_s * pipeline.dt
        self.assertGreater(step[2], 0.8 * expected)
        self.assertLess(np.linalg.norm(step[:2]), 0.2 * expected)

    def test_low_pass_filter_is_applied_before_the_targets(self):
        cfg, pipeline = self._pipeline(alpha=0.3)
        q = np.asarray(list(cfg.init_state.default_arm_joint_angles) + list(cfg.init_state.default_hand_joint_angles))
        pipeline.reset(q)
        action = np.zeros(26)
        action[7] = 1.0
        pipeline.command(action, q[:6])
        self.assertAlmostEqual(float(pipeline.filtered_actions[0, 7]), 0.3, places=6)
        self.assertAlmostEqual(pipeline.previous_targets[7] - q[7], 0.3 * cfg.control.scale_hand_joint_target, places=6)


@unittest.skipUnless(BANK.is_file(), "transform bank not built")
class MujocoBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _env_cfg(self_collision=True)
        cls.track = ReferenceTrack.load(cls.cfg)
        cls.sim = MujocoSim(MujocoSceneConfig.from_env_cfg(cls.cfg))

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_collision_graph_matches_training(self):
        sim = self.sim
        # Selective finger self-collision: fingers 2-5 across each other only.
        self.assertTrue(sim.bodies_can_collide("rl_dg_2_2", "rl_dg_3_2"))
        self.assertTrue(sim.bodies_can_collide("rl_dg_2_tip", "rl_dg_5_tip"))
        self.assertFalse(sim.bodies_can_collide("rl_dg_1_2", "rl_dg_2_2"))  # thumb
        self.assertFalse(sim.bodies_can_collide("rl_dg_palm", "rl_dg_2_3"))  # palm
        self.assertFalse(sim.bodies_can_collide("rl_dg_base", "rl_dg_4_3"))  # merged palm group
        self.assertFalse(sim.bodies_can_collide("rl_dg_2_1", "rl_dg_2_3"))  # same finger
        # External graph: hand-bar and bar-table on, arm-bar and robot-table off.
        self.assertTrue(sim.bodies_can_collide("rl_dg_2_tip", "cube"))
        self.assertTrue(sim.bodies_can_collide("wrist_3_link", "cube"))
        self.assertTrue(sim.bodies_can_collide("cube", "table"))
        self.assertFalse(sim.bodies_can_collide("wrist_2_link", "cube"))
        self.assertFalse(sim.bodies_can_collide("rl_dg_2_tip", "table"))
        self.assertFalse(sim.bodies_can_collide("forearm_link", "table"))

    def test_self_collision_off_filters_every_hand_pair(self):
        sim = MujocoSim(MujocoSceneConfig.from_env_cfg(_env_cfg(self_collision=False)))
        try:
            self.assertFalse(sim.bodies_can_collide("rl_dg_2_2", "rl_dg_3_2"))
            self.assertTrue(sim.bodies_can_collide("rl_dg_2_tip", "cube"))
        finally:
            sim.close()

    def test_training_gains_effort_limits_and_contact_model_are_installed(self):
        m = self.sim.model
        from simtoolreal_newton.envs.controller import pd_gain_arrays

        stiffness, damping = pd_gain_arrays(hand_stiffness_scale=float(self.cfg.control.hand_stiffness_scale))
        for index, name in enumerate(constants.JOINT_NAMES):
            actuator = m.actuator("{}_pos".format(name))
            self.assertAlmostEqual(float(actuator.gainprm[0]), float(stiffness[index]), places=5)
            self.assertAlmostEqual(float(actuator.biasprm[1]), -float(stiffness[index]), places=5)
            self.assertAlmostEqual(float(actuator.biasprm[2]), -float(damping[index]), places=5)
            self.assertEqual(int(actuator.forcelimited[0]), 1)
        self.assertAlmostEqual(float(m.actuator("rj_dg_2_1_pos").forcerange[1]), 7.5)
        cube = m.geom("cube_geom")
        np.testing.assert_allclose(cube.solref, self.cfg.sim.mjwarp.contact_solref)
        np.testing.assert_allclose(cube.solimp, self.cfg.sim.mjwarp.contact_solimp)
        self.assertEqual(int(cube.condim[0]), int(self.cfg.sim.mjwarp.contact_condim))
        self.assertEqual(int(m.opt.cone), 1)  # elliptic
        self.assertAlmostEqual(float(m.opt.impratio), float(self.cfg.sim.mjwarp.impratio))
        self.assertAlmostEqual(float(m.opt.timestep) * self.sim.config.substeps, float(self.cfg.sim.dt))
        self.assertGreater(int(m.ngravcomp), 0)
        tip = m.geom(int(np.flatnonzero(m.geom_bodyid == m.body("rl_dg_2_tip").id)[0]))
        self.assertAlmostEqual(float(tip.friction[0]), float(self.cfg.asset.fingertip_friction))
        self.assertAlmostEqual(float(tip.friction[1]), float(self.cfg.asset.fingertip_torsional_friction))

    def test_mujoco_poses_agree_with_the_urdf_kinematics(self):
        kinematics = PalmKinematics(constants.ROBOT_URDF)
        base = np.asarray(self.cfg.init_state.pos)
        from simtoolreal_newton.sim2sim.observation import fingertip_positions_from_bodies, palm_pose_from_wrist

        for frame in (0, 400, 798):
            sample = self.track.sample(122, frame)
            self.sim.reset(sample.q[0].numpy(), np.zeros(26), self.track.cube_root_state(122, frame))
            state = self.sim.get_state()
            palm_position, palm_orientation = palm_pose_from_wrist(
                state["wrist_position_world"], state["wrist_orientation_world_xyzw"]
            )
            expected_position, expected_orientation = kinematics.palm_pose(sample.q[:, :6])
            np.testing.assert_allclose(palm_position[0].numpy(), expected_position[0].numpy() + base, atol=1e-5)
            np.testing.assert_allclose(
                np.abs(palm_orientation[0].numpy()), np.abs(expected_orientation[0].numpy()), atol=1e-5
            )
            np.testing.assert_allclose(state["palm_link_position_world"], palm_position[0].numpy(), atol=1e-5)
            tips = fingertip_positions_from_bodies(
                state["fingertip_body_positions_world"], state["fingertip_body_orientations_world_xyzw"]
            )
            np.testing.assert_allclose(tips[0].numpy(), kinematics.fingertip_positions(sample.q)[0].numpy() + base, atol=1e-5)
            np.testing.assert_allclose(tips[0].numpy(), state["fingertip_link_positions_world"], atol=1e-5)

    def test_reset_holds_the_pose_and_the_bar_rests_on_the_table(self):
        sample = self.track.sample(122, 0)
        root = self.track.cube_root_state(122, 0)
        self.sim.reset(sample.q[0].numpy(), np.zeros(26), root)
        for _ in range(120):
            self.sim.step_control()
        state = self.sim.get_state()
        np.testing.assert_allclose(state["joint_positions"], sample.q[0].numpy(), atol=1e-3)
        m = self.sim.model
        rest = float(m.body("table").pos[2] + m.geom("table_geom").size[2] + m.geom("cube_geom").size[2])
        self.assertAlmostEqual(float(state["cube_position_world"][2]), rest, places=3)
        self.assertLess(float(np.linalg.norm(state["cube_position_world"][:2] - root[:2])), 2e-3)

    def test_scale_changes_size_mass_and_rest_height(self):
        scaled = MujocoSim(MujocoSceneConfig.from_env_cfg(self.cfg, object_scale=1.2))
        try:
            m = scaled.model
            np.testing.assert_allclose(m.geom("cube_geom").size * 2.0, np.asarray(self.cfg.object.size_m) * 1.2)
            self.assertAlmostEqual(float(m.body("cube").mass[0]), self.cfg.object.mass_kg * 1.2 ** 3, places=6)
            root = self.track.cube_root_state(122, 0, scale=1.2)
            nominal = self.track.cube_root_state(122, 0)
            self.assertAlmostEqual(root[2] - nominal[2], 0.5 * self.cfg.object.size_m[2] * 0.2, places=6)
        finally:
            scaled.close()

    def test_continuous_placement_serves_the_nearest_bank_entry(self):
        translation = self.track.bank.translation[122].numpy()
        yaw = float(self.track.bank.yaw_rad[122])
        self.assertEqual(self.track.nearest_transform(translation[:2], yaw), 122)
        moved = self.track.cube_root_state(122, 0, np.asarray((translation[0] + 0.01, translation[1], 0.0)), yaw)
        base = self.track.cube_root_state(122, 0)
        np.testing.assert_allclose(moved[:3] - base[:3], (0.01, 0.0, 0.0), atol=1e-6)


class OverridesTest(unittest.TestCase):
    def test_set_overrides_parse_json_values(self):
        cfg = _env_cfg()
        apply_overrides(cfg, ["sim.mjwarp.contact_solref=[0.02,1.0]", "object_randomization.scale_min=0.8"])
        self.assertEqual(list(cfg.sim.mjwarp.contact_solref), [0.02, 1.0])
        self.assertEqual(cfg.object_randomization.scale_min, 0.8)
        with self.assertRaises(KeyError):
            apply_overrides(cfg, ["sim.mjwarp.no_such_field=1"])


if __name__ == "__main__":
    unittest.main()
