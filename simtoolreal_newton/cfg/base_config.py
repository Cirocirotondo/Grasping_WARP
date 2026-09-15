"""Python configuration classes following cmm-25-a3-animrl."""

import inspect


class ABCConfig:
    """Recursively instantiate nested configuration classes."""

    def __init__(self) -> None:
        self._init_member_classes(self)

    @classmethod
    def _init_member_classes(cls, obj) -> None:
        for key in dir(obj):
            if key == "__class__":
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                instance = value()
                setattr(obj, key, instance)
                cls._init_member_classes(instance)


class BaseEnvCfg(ABCConfig):
    """AnimRL-compatible environment configuration surface."""

    seed = 42

    class sim:
        # Robot-specific exception to AnimRL's generic 5 ms / decimation 4:
        # one control step must match one sample of the 60 Hz demonstration.
        dt = 1.0 / 60.0
        # Solver substeps per control step (240 Hz). Two substeps, the rate the
        # Isaac Gym configuration ran PhysX at, leave the light finger joints
        # 0.1 rad short of their targets in MuJoCo-Warp; four converge.
        substeps = 4
        gravity = [0.0, 0.0, -9.81]
        device = "cuda:0"
        # Physics backend selected through Isaac Lab: "newton_mjwarp" (Newton
        # with the MuJoCo-Warp solver; kit-less, no Isaac Sim needed),
        # "physx" (auto: OvPhysX kit-less, or Isaac Sim PhysX when Kit runs),
        # "ovphysx" or "isaacsim_physx".
        physics = "newton_mjwarp"
        # Whether the whole step is captured into one CUDA graph. Off only
        # for debugging: it costs most of the throughput.
        use_cuda_graph = True

        class mjwarp:
            # Constraint and contact capacity per environment. The hand can
            # touch the cuboid with up to twenty links, and every contact is a
            # pyramidal cone of four rows plus the joint limits, so the budget
            # is generous rather than tight. Increase if Newton warns.
            njmax = 400
            nconmax = 96
            iterations = 100
            ls_iterations = 50
            solver = "newton"
            integrator = "implicitfast"
            cone = "pyramidal"
            impratio = 1.0
            # Convex collision iterations; the multi-finger hand needs more
            # than MuJoCo's default of 35 to converge.
            ccd_iterations = 60
            tolerance = 1.0e-6
            # Caps applied to the cuboid's velocity after each control step.
            # PhysX limited the depenetration velocity to 2 m/s; MuJoCo has no
            # such limit and a cuboid crushed between rigid finger hulls can
            # leave at tens of m/s, which then breaks the articulation too.
            object_max_linear_velocity = 4.0
            object_max_angular_velocity = 40.0
            # Contact forces act once two shapes are closer than
            # contact_margin (like PhysX rest_offset); contacts are detected
            # a further contact_gap away (like contact_offset - rest_offset).
            contact_margin = 0.0
            contact_gap = 0.005
            # MuJoCo's own collision detection (True) or Newton's collision
            # pipeline (False; GJK/MPR with contact reduction). MuJoCo-Warp's
            # native mesh-box collision launched the resting cuboid at
            # 0.4 m/s from a light fingertip touch; Newton's pipeline does not.
            use_mujoco_contacts = False
            # Up to four contact points per colliding pair instead of one. A
            # single point per finger-cuboid pair cannot pin a box and the
            # grasp kicks the object away.
            enable_multiccd = False
            # Debugging only: run the reference CPU MuJoCo instead of MJWarp.
            use_mujoco_cpu = False
            # MuJoCo contact constraint softness (time constant [s], damping
            # ratio) applied to every collider. None keeps MuJoCo's (0.02, 1),
            # which let a light fingertip touch launch the resting cuboid at
            # 0.3 m/s; 0.01 (the shortest the 240 Hz substep allows with
            # margin) keeps the first touch quiet. Softer values (0.03, 0.05)
            # kick again.
            contact_solref = [0.01, 1.0]
            # Contact friction dimensionality: 3 = sliding only, 4 adds
            # torsional friction, 6 adds rolling friction.
            contact_condim = 3

        class physx:
            # Only used with the PhysX backends. Values mirror the original
            # Isaac Gym configuration.
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.002
            rest_offset = 0.0
            bounce_threshold_velocity = 0.2
            max_depenetration_velocity = 2.0
            max_gpu_contact_pairs = 8 * 1024 * 1024
            default_buffer_size_multiplier = 25.0

    class env:
        # Values from AnimRL WalkCfg/CartwheelCfg.
        num_envs = 4096
        episode_length = 360
        env_spacing = 2.0

        num_observations = None
        num_privileged_obs = None
        num_actions = None
        reference_state_initialization = True
        play = False
        debug = False

    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class viewer:
        enable_viewer = False
        camera_position = [-1.8,-2.0, 1.5] # [1.8, 2.0, 1.5]
        camera_lookat = [0.0, 0.6, 0.75]
        training_camera_enabled = False
        training_camera_env_index = 0
        training_camera_width = 640
        training_camera_height = 480
        # Horizontal field of view of the off-screen camera. Isaac Gym's own
        # default is 90 degrees, which is what every recorded training video so
        # far used; a narrower angle tightens the shot around the robot.
        training_camera_fov_deg = 90.0
        # Second, collision-free actor per environment that replays the
        # demonstration kinematically as a side-by-side visual benchmark.
        # It doubles the simulated bodies, so it stays off for training.
        reference_ghost = False
        reference_ghost_offset = [0.8, 0.0, 0.0]
        reference_ghost_color = [0.15, 0.85, 0.25]


class BaseTrainCfg(ABCConfig):
    """AnimRL runner configuration retained for the future PPO milestone."""

    algorithm_name = "PPO"

    class policy:
        log_std_init = 0.0
        # Keep exploration bounded. The learned parameter is log(sigma), while
        # this setting is expressed directly in action-standard-deviation units.
        max_action_std = 3.0
        actor_hidden_dims = [512, 256]
        critic_hidden_dims = [512, 256]
        activation = "elu"

    class algorithm:
        value_loss_coef = 0.5
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        surrogate_coef = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 0.5e-4
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        bootstrap = True

    class runner:
        num_steps_per_env = 24
        max_iterations = 3000
        normalize_observation = True
        save_interval = 100
        # Off by default: the kit-less Warp camera used for video costs
        # startup time and GPU memory. `train.py --record-video` turns it on.
        record_video = False
        record_video_interval = 500
        record_video_duration_s = 10.0
        record_video_fps = 60
        # Retained only so config.json files written before MP4 recording was
        # introduced still deserialize; these legacy fields are ignored.
        record_gif = False
        record_gif_interval = 100
        record_iters = 10
        experiment_name = "simtoolreal"
        run_name = "llcfix_animrl"
        tensorboard = True
        tensorboard_flush_secs = 10
        evaluation_enabled = True
        evaluation_interval = 500
        evaluation_num_envs = 64
        evaluation_seed = 123
        evaluation_fixed_phases = [0.0, 0.25, 0.5, 0.75]
        # Divergence guard. The entropy bonus is a constant upward force on the
        # unbounded log_std parameter, so a run whose surrogate loss stops
        # opposing it grows the action std without limit and never recovers.
        # These thresholds are deliberately far above anything a healthy run
        # reaches (the 2026-08-26 runs peak at std 2.45 and never clip a single
        # action target) so the guard only fires on a policy that is already
        # unrecoverable, rather than on transient early exploration.
        abort_on_divergence = True
        abort_action_std = 15.0
        abort_action_target_clipped_fraction = 0.8
        # Consecutive iterations above a threshold before aborting. The metrics
        # average ~98k samples per iteration and are very smooth, so this is
        # cheap insurance rather than a necessity.
        abort_patience = 3
        # W&B remains a future optional backend; this milestone logs locally.
        wandb = False
        wandb_group = "default"
