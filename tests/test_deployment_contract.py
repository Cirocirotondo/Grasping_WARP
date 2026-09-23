"""The deployment contract, checked without a simulator.

``scripts/check_deployment_contract.py`` compares the contract against the
running environment; these tests pin the parts that need no physics: the
observation's width and the reset action as a fixed point, the filter, the IK
step's geometry, the velocity-limit slew, and the placement arithmetic that
picks a bank clip for a measured cuboid.
"""

import math
import unittest

import numpy as np
import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.deployment import (
    ACTION_DIM,
    ARM_DOF,
    OBSERVATION_DIM,
    ActionPipeline,
    DeploymentRun,
    ObservationInputs,
    ReferenceCube,
    build_observation,
)
from simtoolreal_newton.envs.rotations import quat_multiply, quat_rotate

CHECKPOINT = ROOT_DIR / "deploy/policies/w6_ref_ori_lowlr_it14600_generalize_pose/model_14600.pt"
SCALE_CHECKPOINT = ROOT_DIR / "deploy/policies/sc2_anchor_s42_it17300/model_17300.pt"


@unittest.skipUnless(SCALE_CHECKPOINT.is_file(), "the scale-observing checkpoint is not in the repository")
class ScaleObservingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = DeploymentRun(SCALE_CHECKPOINT)
        cls.transform_index = cls.deployment.default_transform_index()

    def inputs(self, object_scale):
        run = self.deployment
        sample = run.reference_sample(self.transform_index, 0)
        q = sample.q[0].numpy().astype(np.float64)
        return ObservationInputs(
            joint_positions=q,
            joint_velocities=sample.dq[0].numpy().astype(np.float64),
            previous_targets=q,
            reference_index=0,
            cube_pose_base=run.cube_pose_base(run.reference_cube_pose(self.transform_index, 0, object_scale)),
            symmetry_index=0,
            object_scale=object_scale,
        )

    def test_the_scale_is_the_last_column(self):
        run = self.deployment
        self.assertTrue(run.observes_scale)
        self.assertEqual(run.observation_dim, OBSERVATION_DIM + 1)
        observation = build_observation(run, self.inputs(0.9))
        self.assertEqual(observation.shape, (OBSERVATION_DIM + 1,))
        self.assertAlmostEqual(float(observation[-1]), 0.9, places=6)
        self.assertEqual(run.act(observation).shape, (ACTION_DIM,))

    def test_a_missing_scale_is_refused(self):
        with self.assertRaises(ValueError):
            build_observation(self.deployment, self.inputs(1.0)._replace(object_scale=None))

    def test_the_reference_bar_is_lifted_for_a_scaled_bar(self):
        run = self.deployment
        nominal = run.reference_cube_pose(self.transform_index, 0, 1.0)
        scaled = run.reference_cube_pose(self.transform_index, 0, 1.2)
        self.assertAlmostEqual(float(scaled[2] - nominal[2]), 0.5 * 0.05 * 0.2, places=7)
        np.testing.assert_allclose(scaled[[0, 1, 3, 4, 5, 6]], nominal[[0, 1, 3, 4, 5, 6]])
        replayed, _, _ = ReferenceCube(run, self.transform_index, 1.2).cube_state(0)
        np.testing.assert_allclose(replayed, scaled)

    def test_the_scale_only_touches_its_own_column(self):
        run = self.deployment
        base = self.inputs(1.0)
        observation = build_observation(run, base)
        other = build_observation(run, base._replace(object_scale=0.8))
        np.testing.assert_allclose(other[:-1], observation[:-1])
        self.assertAlmostEqual(float(other[-1]), 0.8, places=6)


@unittest.skipUnless(CHECKPOINT.is_file(), "the deployed checkpoint is not in the repository")
class DeploymentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = DeploymentRun(CHECKPOINT)
        cls.transform_index = cls.deployment.default_transform_index()

    def reference_inputs(self, reference_index):
        run = self.deployment
        sample = run.reference_sample(self.transform_index, reference_index)
        q = sample.q[0].numpy().astype(np.float64)
        return ObservationInputs(
            joint_positions=q,
            joint_velocities=sample.dq[0].numpy().astype(np.float64),
            previous_targets=q,
            reference_index=reference_index,
            cube_pose_base=run.cube_pose_base(sample.cube_pose[0].numpy()),
            symmetry_index=0,
        )

    def test_the_observation_has_the_checkpoints_width(self):
        observation = build_observation(self.deployment, self.reference_inputs(0))
        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertEqual(self.deployment.policy.policy_latent_net[0].weight.shape[1], OBSERVATION_DIM)
        self.assertEqual(self.deployment.act(observation).shape, (ACTION_DIM,))

    def test_the_reset_action_is_a_fixed_point(self):
        run = self.deployment
        q = run.reference_q(self.transform_index, 0)
        pipeline = ActionPipeline(run)
        pipeline.reset(q)
        reset_action = pipeline.filtered_actions.numpy().copy()
        self.assertTrue(np.all(reset_action[:ARM_DOF] == 0.0))
        targets = pipeline.command(reset_action, q[:ARM_DOF])
        np.testing.assert_allclose(targets, q, atol=1e-6)
        np.testing.assert_allclose(pipeline.apply(), q, atol=1e-6)

    def test_the_action_filter_is_first_order(self):
        run = self.deployment
        q = run.reference_q(self.transform_index, 0)
        pipeline = ActionPipeline(run)
        pipeline.reset(q)
        before = pipeline.filtered_actions.numpy().copy()
        raw = np.full(ACTION_DIM, 0.5)
        pipeline.command(raw, q[:ARM_DOF])
        expected = before + run.action_filter_alpha * (raw - before)
        np.testing.assert_allclose(pipeline.filtered_actions.numpy(), expected, atol=1e-6)

    def test_a_translation_twist_moves_the_palm_where_asked(self):
        run = self.deployment
        original_alpha = run.action_filter_alpha
        run.action_filter_alpha = 1.0
        try:
            q = run.reference_q(self.transform_index, 0)
            pipeline = ActionPipeline(run)
            pipeline.reset(q)
            palm_before, _ = run.palm_pose(q)
            action = np.zeros(ACTION_DIM)
            action[0] = 1.0
            targets = pipeline.command(action, q[:ARM_DOF])
            palm_after, _ = run.palm_pose(targets)
            requested = pipeline.last_info["requested_twist"]
            self.assertAlmostEqual(float(np.linalg.norm(requested[:3])), run.arm_translation_speed * run.dt, places=6)
            moved = palm_after - palm_before
            np.testing.assert_allclose(moved, requested[:3], atol=2e-4)
            self.assertLess(float(np.max(np.abs(targets[:ARM_DOF] - q[:ARM_DOF]))), run.ik_max_joint_delta + 1e-6)
            np.testing.assert_allclose(targets[ARM_DOF:], q[ARM_DOF:], atol=1e-6)
        finally:
            run.action_filter_alpha = original_alpha

    def test_the_twist_request_saturates_without_turning(self):
        run = self.deployment
        q = run.reference_q(self.transform_index, 0)
        pipeline = ActionPipeline(run)
        pipeline.reset(q)
        action = np.zeros(ACTION_DIM)
        action[0:3] = (30.0, 40.0, 0.0)
        pipeline.command(action, q[:ARM_DOF])
        requested = pipeline.last_info["requested_twist"][:3]
        self.assertAlmostEqual(float(np.linalg.norm(requested)), run.arm_translation_speed * run.dt, places=6)
        np.testing.assert_allclose(requested / np.linalg.norm(requested), (0.6, 0.8, 0.0), atol=1e-6)

    def test_applied_targets_slew_at_the_velocity_limit(self):
        run = self.deployment
        q = run.reference_q(self.transform_index, 0)
        pipeline = ActionPipeline(run)
        pipeline.reset(q)
        pipeline.position_targets = pipeline.position_targets + 1.0
        applied = pipeline.apply()
        np.testing.assert_allclose(applied - q, run.target_slew_per_step.numpy(), atol=1e-6)
        self.assertAlmostEqual(float(run.target_slew_per_step[0]), math.pi / 60.0, places=6)

    def test_next_reference_action_carries_the_target_onto_the_next_frame(self):
        run = self.deployment
        original_alpha = run.action_filter_alpha
        run.action_filter_alpha = 1.0
        try:
            index = 300
            q = run.reference_q(self.transform_index, index)
            next_q = run.reference_q(self.transform_index, index + 1)
            pipeline = ActionPipeline(run)
            pipeline.reset(q)
            action = run.next_reference_action(pipeline.previous_arm_targets, self.transform_index, index)
            targets = pipeline.command(action, q[:ARM_DOF])
            np.testing.assert_allclose(targets[ARM_DOF:], next_q[ARM_DOF:], atol=1e-5)
            palm_target, _ = run.palm_pose(targets)
            palm_next, _ = run.palm_pose(next_q)
            self.assertLess(float(np.linalg.norm(palm_target - palm_next)), 2e-3)
        finally:
            run.action_filter_alpha = original_alpha

    def test_a_bank_entrys_own_cuboid_maps_back_to_that_entry(self):
        run = self.deployment
        for index in (0, self.transform_index, run.bank.transform_count - 1):
            pose = run.bank.cube_pose[index, 0].numpy()
            placement = run.placement_from_cube_pose(pose)
            nearest, residual_xy, residual_yaw = run.nearest_transform_index(placement)
            self.assertEqual(nearest, index)
            self.assertLess(residual_xy, 1e-4)
            self.assertLess(abs(residual_yaw), 1e-4)
            self.assertLess(placement.tilt_rad, 1e-3)
            np.testing.assert_allclose(placement.translation.numpy(), run.bank.translation[index].numpy(), atol=1e-4)

    def test_the_reference_cuboid_resolves_to_the_identity_symmetry(self):
        run = self.deployment
        cube = ReferenceCube(run, self.transform_index)
        pose, _, _ = cube.cube_state(0)
        chosen = run.choose_symmetry_index(run.cube_pose_base(pose), run.reference_sample(self.transform_index, 0).cube_pose[0])
        self.assertAlmostEqual(abs(float(run.cuboid_symmetries[chosen, 3])), 1.0, places=6)

    def test_a_relabelled_bar_produces_the_same_observation(self):
        run = self.deployment
        inputs = self.reference_inputs(0)
        base = inputs.cube_pose_base
        # Rotate the bar half a turn about its own long axis: a different
        # quaternion for a shape that has not moved.
        flip = torch.tensor([1.0, 0.0, 0.0, 0.0])
        flipped = torch.cat((base[:3], quat_multiply(base[3:7], flip)))
        reference = run.reference_sample(self.transform_index, 0).cube_pose[0]
        index_flipped = run.choose_symmetry_index(flipped, reference)
        observation = build_observation(run, inputs)
        observation_flipped = build_observation(run, inputs._replace(cube_pose_base=flipped, symmetry_index=index_flipped))
        np.testing.assert_allclose(observation_flipped, observation, atol=1e-5)

    def test_the_cuboid_frame_mapping_is_the_environments(self):
        run = self.deployment
        pose = run.bank.cube_pose[self.transform_index, 0]
        base = run.cube_pose_base(pose.numpy())
        np.testing.assert_allclose(base[:3].numpy(), (pose[:3] * torch.tensor([-1.0, -1.0, 1.0])).numpy(), atol=1e-6)
        x, y, z, w = pose[3:7]
        expected = torch.tensor([-y, x, w, -z])
        expected = expected / torch.linalg.vector_norm(expected)
        np.testing.assert_allclose(base[3:7].numpy(), expected.numpy(), atol=1e-6)
        # The relabelled quaternion still rotates vectors the way a pi turn about z would.
        vector = torch.tensor([[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(
            quat_rotate(base[3:7].unsqueeze(0), vector).numpy(),
            (quat_rotate(pose[3:7].unsqueeze(0), vector) * torch.tensor([-1.0, -1.0, 1.0])).numpy(),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
