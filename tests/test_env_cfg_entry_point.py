"""The registered Isaac Lab entry point instantiates the cfg class bare."""

import unittest


class EnvCfgEntryPointTest(unittest.TestCase):
    def test_a_bare_cfg_carries_the_animrl_fields(self):
        try:
            from simtoolreal_newton.envs.motion_imitation_env_cfg import MotionImitationEnvCfg
        except ImportError as error:  # pragma: no cover - needs the Isaac Lab venv
            self.skipTest("Isaac Lab is not importable here: {}".format(error))
        cfg = MotionImitationEnvCfg()
        # ``isaaclab train --task SimToolReal-Grasp-Direct`` builds this object
        # and the environment reads both fields from it before building the
        # AnimRL configuration; the fields once sat inside make_camera_cfg().
        self.assertTrue(hasattr(cfg, "animrl_cfg"))
        self.assertTrue(hasattr(cfg, "animrl_overrides"))
        self.assertIsNone(cfg.animrl_cfg)
        self.assertEqual(cfg.animrl_overrides, {})


if __name__ == "__main__":
    unittest.main()
