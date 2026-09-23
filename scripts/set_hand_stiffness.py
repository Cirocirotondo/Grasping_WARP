#!/usr/bin/env python3
"""Set or show the real DG5F's stiffness: the ``p`` gain of its position PID.

The hand's ros2_control loop is a ROS-side ``pid_controller`` on top of a raw
PWM-duty command, so ``gains.<joint>.p`` on ``/dg5f_right/rj_dg_pospid`` is the
whole stiffness; it applies live. Run this at any point while the driver is up,
including between runs of ``scripts/run_policy_real.py``, to raise the
stiffness gradually:

    deps/IsaacLab/.venv/bin/python scripts/set_hand_stiffness.py --show
    deps/IsaacLab/.venv/bin/python scripts/set_hand_stiffness.py --p 0.5

``--p`` with the runner's ``--hand-stiffness`` flag does the same thing at the
start of a run. Neither needs ROS in this interpreter: both shell out to the
``ros2`` CLI under the ROS environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_newton.deployment.hand_stiffness import (  # noqa: E402
    DEFAULT_CONTROLLER_NODE,
    DEFAULT_ROS_SETUP,
    HandStiffnessError,
    format_gains,
    gains_yaml,
    read_hand_gains,
    set_hand_stiffness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=None, help="Position gain to load on every hand joint.")
    parser.add_argument("--show", action="store_true", help="Print the controller's current gains.")
    parser.add_argument("--node", default=DEFAULT_CONTROLLER_NODE)
    parser.add_argument("--ros-setup", nargs="*", default=list(DEFAULT_ROS_SETUP), help="setup.bash files to source.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the parameter document instead of loading it.")
    args = parser.parse_args()
    if args.p is None and not args.show:
        parser.error("give --p VALUE and/or --show")
    try:
        if args.p is not None:
            if args.dry_run:
                print(gains_yaml(args.p), end="")
            else:
                gains = set_hand_stiffness(args.p, node=args.node, ros_setup=args.ros_setup, timeout=args.timeout)
                print("DG5F stiffness set: {}".format(format_gains(gains)))
        if args.show and not args.dry_run:
            print("DG5F stiffness now: {}".format(format_gains(read_hand_gains(args.node, args.ros_setup, args.timeout))))
    except (HandStiffnessError, ValueError) as error:
        print("ERROR: {}".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
