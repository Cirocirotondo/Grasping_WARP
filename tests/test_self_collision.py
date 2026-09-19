import unittest

from simtoolreal_newton.cfg import SimToolRealCfg
from simtoolreal_newton.envs.controller import ARM_BODY_NAMES
from simtoolreal_newton.envs.self_collision import (
    PALM_BODY_NAME,
    allowed_body_pairs,
    filtered_body_pairs,
    finger_body_names,
)

ROBOT_BODY_NAMES = tuple(ARM_BODY_NAMES) + (PALM_BODY_NAME,) + finger_body_names()


def asset_cfg(**overrides):
    asset = SimToolRealCfg().asset
    asset.self_collision = True
    for key, value in overrides.items():
        setattr(asset, key, value)
    return asset


class SelfCollisionPairsTest(unittest.TestCase):
    def test_body_name_inventory(self):
        self.assertEqual(len(finger_body_names()), 20)
        self.assertIn("rl_dg_1_1", finger_body_names())
        self.assertIn("rl_dg_5_4", finger_body_names())
        # 6 arm bodies + the wrist carrying the palm + 20 finger links.
        self.assertEqual(len(ROBOT_BODY_NAMES), 27)

    def test_master_switch_off_allows_nothing(self):
        asset = asset_cfg()
        asset.self_collision = False
        self.assertEqual(allowed_body_pairs(asset, ROBOT_BODY_NAMES), set())
        self.assertEqual(len(filtered_body_pairs(asset, ROBOT_BODY_NAMES)), 27 * 26 // 2)

    def test_default_is_four_fingers_all_cross(self):
        allowed = allowed_body_pairs(asset_cfg(), ROBOT_BODY_NAMES)
        # 6 finger pairs among {2,3,4,5} x 4 links x 4 links.
        self.assertEqual(len(allowed), 96)
        self.assertIn(frozenset(("rl_dg_2_1", "rl_dg_5_4")), allowed)
        # The thumb and the palm stay out, and so does a finger with itself.
        for pair in allowed:
            self.assertFalse(any(name.startswith("rl_dg_1_") for name in pair))
            self.assertNotIn(PALM_BODY_NAME, pair)
            self.assertNotEqual(*(name.rsplit("_", 1)[0] for name in pair))

    def test_adjacent_fingers_only(self):
        allowed = allowed_body_pairs(asset_cfg(self_collision_adjacent_fingers_only=True), ROBOT_BODY_NAMES)
        # (2,3), (3,4), (4,5) x 16 link pairs.
        self.assertEqual(len(allowed), 48)
        self.assertIn(frozenset(("rl_dg_2_4", "rl_dg_3_4")), allowed)
        self.assertNotIn(frozenset(("rl_dg_2_4", "rl_dg_4_4")), allowed)

    def test_palm_adds_sixteen(self):
        allowed = allowed_body_pairs(asset_cfg(self_collision_with_palm=True), ROBOT_BODY_NAMES)
        self.assertEqual(len(allowed), 96 + 16)
        self.assertIn(frozenset((PALM_BODY_NAME, "rl_dg_3_2")), allowed)
        self.assertNotIn(frozenset((PALM_BODY_NAME, "rl_dg_1_2")), allowed)

    def test_same_finger_adds_non_adjacent_links(self):
        allowed = allowed_body_pairs(asset_cfg(self_collision_same_finger=True), ROBOT_BODY_NAMES)
        # Per finger: (1,3), (1,4), (2,4); four fingers.
        self.assertEqual(len(allowed), 96 + 12)
        self.assertIn(frozenset(("rl_dg_4_1", "rl_dg_4_3")), allowed)
        self.assertNotIn(frozenset(("rl_dg_4_1", "rl_dg_4_2")), allowed)

    def test_finger_selection_and_absent_bodies(self):
        allowed = allowed_body_pairs(asset_cfg(self_collision_fingers=[1, 2]), ROBOT_BODY_NAMES)
        self.assertEqual(len(allowed), 16)
        # Bodies the model does not carry are never named in a pair.
        few = ("rl_dg_2_1", "rl_dg_3_1", "rl_dg_4_1")
        self.assertEqual(allowed_body_pairs(asset_cfg(), few), {frozenset(p) for p in (
            ("rl_dg_2_1", "rl_dg_3_1"), ("rl_dg_2_1", "rl_dg_4_1"), ("rl_dg_3_1", "rl_dg_4_1"))})

    def test_filtered_is_the_exact_complement(self):
        asset = asset_cfg()
        allowed = allowed_body_pairs(asset, ROBOT_BODY_NAMES)
        filtered = filtered_body_pairs(asset, ROBOT_BODY_NAMES)
        self.assertEqual(len(allowed) + len(filtered), 27 * 26 // 2)
        self.assertEqual(allowed & filtered, set())
        self.assertIn(frozenset(("base_link", "rl_dg_3_2")), filtered)
        self.assertIn(frozenset((PALM_BODY_NAME, "rl_dg_2_1")), filtered)


if __name__ == "__main__":
    unittest.main()
