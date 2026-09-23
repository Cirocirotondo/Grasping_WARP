"""Real-robot deployment: the policy contract off the simulator, plus hardware clients.

Everything here is plain PyTorch and numpy. ``contract`` rebuilds the
environment's observation and action pipeline from the environment's own
building blocks; the clients speak to the UR5e low-level controller (ZMQ), the
DG5F ROS bridge (UDP) and the tag pose estimator (ZMQ). The staged controller
that ties them together is ``scripts/run_policy_real.py``.
"""

from .arm_client import ArmClient, ArmClientError
from .contract import (
    ACTION_DIM,
    ARM_DOF,
    HAND_DOF,
    OBSERVATION_DIM,
    ActionPipeline,
    DeploymentRun,
    ObservationInputs,
    Placement,
    build_observation,
)
from .cube_source import (
    CubeSource,
    CubeSourceError,
    FrozenCube,
    PoseEstimationCube,
    ReferenceCube,
)
from .hand_client import HandClient, HandClientError
from .safety import SafetyAbort, SpikeMonitor, TargetLimiter, confirm_send, wait_for_key
from .viewer import DeploymentViewer, ViewerUnavailable

__all__ = [
    "ACTION_DIM",
    "ARM_DOF",
    "HAND_DOF",
    "OBSERVATION_DIM",
    "ActionPipeline",
    "ArmClient",
    "ArmClientError",
    "CubeSource",
    "CubeSourceError",
    "DeploymentRun",
    "DeploymentViewer",
    "FrozenCube",
    "HandClient",
    "HandClientError",
    "ObservationInputs",
    "Placement",
    "PoseEstimationCube",
    "ReferenceCube",
    "SafetyAbort",
    "SpikeMonitor",
    "TargetLimiter",
    "ViewerUnavailable",
    "build_observation",
    "confirm_send",
    "wait_for_key",
]
