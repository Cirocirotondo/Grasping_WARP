#!/usr/bin/env python3
"""Convert the UR5e + DG5F URDF into the multi-physics USD the environment spawns.

Run once after cloning (``setup.sh`` does it). The conversion merges the fixed
joints exactly like Isaac Gym's ``collapse_fixed_joints`` did, so the surviving
bodies are ``base_link .. wrist_3_link`` (with mount, base and palm merged) and
``rl_dg_<finger>_<1..4>`` (with the tips merged into the distal phalanges).
Drive gains are written at runtime by the environment's actuator configs, so
the values authored here only matter for viewing the asset on its own.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_newton.envs.controller import pd_gain_tables  # noqa: E402
from simtoolreal_newton.envs.motion_imitation_env_cfg import (  # noqa: E402
    ROBOT_URDF,
    ROBOT_USD,
    SOURCE_URDF,
    USD_DIR,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-run the conversion even if the USD exists.")
    args = parser.parse_args()

    # The URDF references its meshes relative to the ``assets`` directory, so
    # the converter reads a copy that lives there.
    ROBOT_URDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_URDF, ROBOT_URDF)

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    stiffness, damping = pd_gain_tables()
    cfg = UrdfConverterCfg(
        asset_path=str(ROBOT_URDF),
        usd_dir=str(USD_DIR),
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=False,
        force_usd_conversion=bool(args.force),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
        ),
    )
    converter = UrdfConverter(cfg)
    print("USD:", converter.usd_path)
    if Path(converter.usd_path).resolve() != ROBOT_USD.resolve():
        raise RuntimeError("The converter wrote {} but the environment expects {}".format(converter.usd_path, ROBOT_USD))

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(ROBOT_USD))
    bodies = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p.GetName() for p in stage.Traverse() if "Joint" in p.GetTypeName() and p.GetTypeName() != "PhysicsFixedJoint"]
    print("rigid bodies ({}): {}".format(len(bodies), bodies))
    print("joints ({}): {}".format(len(joints), joints))


if __name__ == "__main__":
    main()
