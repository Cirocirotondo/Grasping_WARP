"""The viewer's space bar pauses the simulation, not just the rendering.

Newton's own handler toggles ``_paused``, which only stops the viewer from
drawing while Isaac Lab keeps stepping the environment. The rebinding in
:mod:`simtoolreal_newton.launch` moves the key onto ``_paused_training``, the
flag ``SimulationContext.update_visualizers`` blocks on.

The viewer is stubbed: the real one needs a GPU and a window, and what is
under test is which attribute the key toggles.
"""

import sys
import types
import unittest
from unittest import mock


class _Ui:
    def __init__(self, capturing=False):
        self._capturing = capturing

    def is_capturing(self):
        return self._capturing


SPACE = 32
OTHER = 65


def _stub_viewer_module():
    """Stand in for pyglet and the Isaac Lab Newton viewer class."""
    pyglet = types.ModuleType("pyglet")
    pyglet.window = types.ModuleType("pyglet.window")
    pyglet.window.key = types.SimpleNamespace(SPACE=SPACE)

    inherited_calls = []

    class NewtonViewerGL:
        def __init__(self):
            self.ui = _Ui()
            self._paused = False
            self._paused_training = False

        def on_key_press(self, symbol, modifiers):
            inherited_calls.append(symbol)
            if symbol == SPACE:
                self._paused = not self._paused

    module = types.ModuleType("isaaclab_visualizers.newton.newton_visualizer")
    module.NewtonViewerGL = NewtonViewerGL
    package = types.ModuleType("isaaclab_visualizers")
    newton_pkg = types.ModuleType("isaaclab_visualizers.newton")

    modules = {
        "pyglet": pyglet,
        "pyglet.window": pyglet.window,
        "isaaclab_visualizers": package,
        "isaaclab_visualizers.newton": newton_pkg,
        "isaaclab_visualizers.newton.newton_visualizer": module,
    }
    return modules, NewtonViewerGL, inherited_calls


class SpacePauseBindingTest(unittest.TestCase):
    def setUp(self):
        from simtoolreal_newton import launch

        self.launch = launch
        self.modules, self.viewer_cls, self.inherited_calls = _stub_viewer_module()
        patcher = mock.patch.dict(sys.modules, self.modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.launch._bind_space_to_simulation_pause()

    def test_space_toggles_the_simulation_pause_and_leaves_rendering_alone(self):
        viewer = self.viewer_cls()

        viewer.on_key_press(SPACE, 0)
        self.assertTrue(viewer._paused_training)
        self.assertFalse(viewer._paused, "space must not touch the rendering pause")

        viewer.on_key_press(SPACE, 0)
        self.assertFalse(viewer._paused_training)
        self.assertFalse(viewer._paused)

    def test_space_is_swallowed_so_newton_never_sees_it(self):
        viewer = self.viewer_cls()
        viewer.on_key_press(SPACE, 0)
        self.assertEqual(self.inherited_calls, [])

    def test_every_other_key_still_reaches_newton(self):
        viewer = self.viewer_cls()
        viewer.on_key_press(OTHER, 0)
        self.assertEqual(self.inherited_calls, [OTHER])

    def test_a_space_typed_into_a_text_field_pauses_nothing(self):
        viewer = self.viewer_cls()
        viewer.ui = _Ui(capturing=True)
        viewer.on_key_press(SPACE, 0)
        self.assertFalse(viewer._paused_training)
        self.assertFalse(viewer._paused)
        self.assertEqual(self.inherited_calls, [])

    def test_patching_twice_does_not_stack_handlers(self):
        self.launch._bind_space_to_simulation_pause()
        viewer = self.viewer_cls()
        viewer.on_key_press(SPACE, 0)
        self.assertTrue(viewer._paused_training)

    def test_a_missing_viewer_package_is_not_an_error(self):
        with mock.patch.dict(sys.modules, {"isaaclab_visualizers.newton.newton_visualizer": None}):
            self.launch._bind_space_to_simulation_pause()


if __name__ == "__main__":
    unittest.main()
