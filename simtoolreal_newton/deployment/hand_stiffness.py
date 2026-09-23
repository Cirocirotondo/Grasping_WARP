"""The real hand's stiffness is one ROS parameter; set and read it from here.

The DG5F's ros2_control hardware interface exposes only an effort (PWM duty)
command, and ``rj_dg_pospid`` -- a ``pid_controller/PidController`` running at
300 Hz on the ROS side -- closes the position loop on top of it with ``p`` only
(``i = d = 0``). Nothing is written to the motor firmware. So the hand's whole
stiffness is ``gains.<joint>.p`` on ``/dg5f_right/rj_dg_pospid``: 2.0 in the
driver's YAML today, applied live by ``control_toolbox::PidROS`` whenever the
parameter changes.

Training's soft hand (``control.hand_stiffness_scale`` 0.116, about 5 N m/rad)
has no calibrated mapping to duty per radian, so the value that matches it is
found on the bench, low to high. With ``p = 2.0`` a 0.5 rad error already
saturates the duty; ``p`` well below 1 turns the hand into a weak spring that
sags under contact and gravity (``i`` is zero), which is what the policy saw.

Both entry points shell out to the ``ros2`` CLI under the ROS environment, so
the policy process itself never imports ROS.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from simtoolreal_newton.envs.controller import HAND_JOINT_NAMES

DEFAULT_CONTROLLER_NODE = "/dg5f_right/rj_dg_pospid"
DEFAULT_ROS_SETUP = (
    "/opt/ros/humble/setup.bash",
    "/home/duplo/git/tesollo_ros2/install_dg5f/setup.bash",
)
DEFAULT_TIMEOUT_S = 30.0


class HandStiffnessError(RuntimeError):
    """The controller could not be reached, or did not take the gains."""


def _ros2(arguments: Sequence[str], ros_setup: Sequence[str], timeout: float) -> str:
    """Run ``ros2 <arguments>`` in a shell that sourced the ROS environment."""
    sources = " && ".join("source {}".format(_quote(path)) for path in ros_setup if path)
    command = "{}ros2 {}".format(sources + " && " if sources else "", " ".join(_quote(a) for a in arguments))
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=float(timeout),
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        )
    except FileNotFoundError as error:
        raise HandStiffnessError("bash is not available: {}".format(error))
    except subprocess.TimeoutExpired:
        raise HandStiffnessError(
            "ros2 {} did not answer within {:.0f} s. Is the DG5F driver running?".format(arguments[0], timeout)
        )
    if completed.returncode != 0:
        raise HandStiffnessError(
            "ros2 {} failed (exit {}): {}".format(
                " ".join(arguments), completed.returncode, (completed.stderr or completed.stdout).strip()
            )
        )
    return completed.stdout


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def gains_yaml(p: float, joints: Iterable[str] = HAND_JOINT_NAMES) -> str:
    """The ``ros2 param load`` document setting ``p`` on every listed joint."""
    p = float(p)
    if not p > 0.0 or p != p:
        raise ValueError("The stiffness gain must be a positive number")
    lines = ["/**:", "  ros__parameters:", "    gains:"]
    for joint in joints:
        lines.append("      {}:".format(joint))
        # A bare integer would be typed as int and rejected by a double parameter.
        lines.append("        p: {!r}".format(p))
    return "\n".join(lines) + "\n"


def read_hand_gains(
    node: str = DEFAULT_CONTROLLER_NODE,
    ros_setup: Sequence[str] = DEFAULT_ROS_SETUP,
    timeout: float = DEFAULT_TIMEOUT_S,
    joints: Iterable[str] = HAND_JOINT_NAMES,
) -> Dict[str, float]:
    """``p`` per joint, from one ``ros2 param dump`` of the controller."""
    dump = _ros2(["param", "dump", node], ros_setup, timeout)
    lines = dump.splitlines()
    gains: Dict[str, float] = {}
    for joint in joints:
        gains[joint] = _p_gain_in_dump(lines, joint, node)
    return gains


def _p_gain_in_dump(lines: List[str], joint: str, node: str) -> float:
    """``p`` under the ``<joint>:`` mapping, found by indentation."""
    for index, line in enumerate(lines):
        if line.strip() != "{}:".format(joint):
            continue
        indent = len(line) - len(line.lstrip())
        for inner in lines[index + 1 :]:
            if inner.strip() and len(inner) - len(inner.lstrip()) <= indent:
                break
            match = re.match(r"^\s*p:\s*([-+0-9.eE]+)\s*$", inner)
            if match:
                return float(match.group(1))
        raise HandStiffnessError("{} lists no p gain for {}".format(node, joint))
    raise HandStiffnessError("{} does not list gains for {}".format(node, joint))


def set_hand_stiffness(
    p: float,
    node: str = DEFAULT_CONTROLLER_NODE,
    ros_setup: Sequence[str] = DEFAULT_ROS_SETUP,
    timeout: float = DEFAULT_TIMEOUT_S,
    joints: Sequence[str] = tuple(HAND_JOINT_NAMES),
    verify: bool = True,
) -> Dict[str, float]:
    """Load ``p`` onto every hand joint of the running controller and read it back."""
    document = gains_yaml(p, joints)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="dg5f_gains_", delete=False) as handle:
        handle.write(document)
        path = Path(handle.name)
    try:
        _ros2(["param", "load", node, str(path)], ros_setup, timeout)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    if not verify:
        return {joint: float(p) for joint in joints}
    gains = read_hand_gains(node, ros_setup, timeout, joints)
    wrong = {joint: value for joint, value in gains.items() if abs(value - float(p)) > 1e-9}
    if wrong:
        raise HandStiffnessError(
            "The controller reports different gains after the load: {}".format(
                ", ".join("{}={:g}".format(joint, value) for joint, value in wrong.items())
            )
        )
    return gains


def format_gains(gains: Dict[str, float]) -> str:
    values = sorted(set(round(v, 6) for v in gains.values()))
    if len(values) == 1:
        return "p = {:g} on all {} joints".format(values[0], len(gains))
    return ", ".join("{}={:g}".format(joint, value) for joint, value in gains.items())
