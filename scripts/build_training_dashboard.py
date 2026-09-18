#!/usr/bin/env python3
"""Build the phone-friendly training dashboard from the fleet's run logs.

The dashboard is a single self-contained HTML file: metric series, run
configuration diffs, the verdict sweeps and the recorded videos are all
inlined, so it can be published as an Artifact and read from a phone with
nothing else running.

    python3 scripts/build_training_dashboard.py
    python3 scripts/build_training_dashboard.py --featured 4 --videos-per-run 6

It reads every run directory under ``logs/simtoolreal``: the local ones, and
the light copies pulled from the fleet by ``scripts/fleet_pull_light.sh``,
whose directory name carries the host as a ``TARS_`` / ``CASE_`` / ``UR5_``
prefix. Each run is labelled with its host and with the wave question from
``scripts/build_control_panel.py``.

The companion template is scripts/dashboard_template.html.
"""

import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from simtoolreal_newton.cfg import (  # noqa: E402  (needs the path above)
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
)
from build_control_panel import WAVE  # noqa: E402  (the campaign's wave sheet)

TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
DATA_MARKER = "__DASHBOARD_DATA__"

# Directory-name prefix -> the host the run was pulled from. No prefix means
# the run trained on the desktop this script runs on.
HOST_PREFIXES = {"TARS_": "tars", "CASE_": "case", "UR5_": "ur5"}

# One series per line chart. Every entry is (key, label, unit, "up" or "down"
# for the direction that means progress). The keys are the ones train.py
# writes into metrics.jsonl; GPU_AGENT_BRIEF.md section 1 says which of them
# actually decide whether a policy is good.
SERIES = [
    ("mean_reward", "Mean step reward", "", "up"),
    ("evaluation_score", "Evaluation score", "", "up"),
    ("episode_return", "Episode return", "", "up"),
    ("episode_length", "Episode length", "steps", "up"),
    ("episode_mean_object_position_error_m", "Object position error", "m", "down"),
    (
        "episode_mean_object_orientation_error_rad",
        "Object orientation error",
        "rad",
        "down",
    ),
    ("episode_max_peak_object_com_lift_m", "Best lift", "m", "up"),
    ("episode_mean_fingertip_object_distance_m", "Fingertip-to-object", "m", "down"),
    # How much of the 1108-frame clip the episodes reach. It moves before the
    # lift does, so it is the early sign that the carry is being learned.
    ("episode_reference_end_fraction", "Clip reached", "frac", "up"),
    # Fingertip contact is what tells a grasp from a shove.
    ("mean_fingertip_contact_fraction", "Fingertip contact", "frac", "up"),
    ("episode_early_termination_fraction", "Early termination", "frac", "down"),
    ("mean_action_std", "Action std", "", "flat"),
    ("object_assist_scale", "Object-assist scale", "", "flat"),
    # Both evaluation cohorts, because they measure different things.
    # "uniform" follows the training RSI distribution; "fixed" replays four
    # phases of the demonstration (frames 0, 277, 554, 831), three of which
    # start outside the pregrasp window most recipes train.
    (
        "evaluation_uniform_early_termination_fraction",
        "Early termination (trained starts)",
        "frac",
        "down",
    ),
    (
        "evaluation_fixed_early_termination_fraction",
        "Early termination (fixed starts)",
        "frac",
        "down",
    ),
    (
        "evaluation_uniform_mean_peak_object_com_lift_m",
        "Object lift (trained starts)",
        "m",
        "up",
    ),
    (
        "evaluation_fixed_mean_peak_object_com_lift_m",
        "Object lift (fixed starts)",
        "m",
        "up",
    ),
    (
        "evaluation_fixed_mean_object_orientation_error_rad",
        "Object orientation error (fixed starts)",
        "rad",
        "down",
    ),
    # Scale-free, so it is the one smoothness number comparable across runs
    # with different reward sigmas.
    (
        "evaluation_fixed_mean_rms_ee_action_rate",
        "EE action rate (command vibration)",
        "",
        "down",
    ),
    (
        "evaluation_fixed_mean_rms_arm_joint_rate",
        "Arm joint rate (IK output vibration)",
        "rad",
        "down",
    ),
    (
        "evaluation_fixed_mean_rms_hand_position_error",
        "Hand joint error vs the demo",
        "rad",
        "down",
    ),
    # How often the IK step hit its per-joint clamp. Persistently high means
    # the policy asks for more than the arm can deliver in one control step.
    (
        "evaluation_fixed_mean_arm_joint_delta_clipped",
        "IK clamp saturation",
        "",
        "down",
    ),
]
SERIES_KEYS = [key for key, _, _, _ in SERIES]

# Evaluation figures inlined per featured run, as (file stem, label, caption).
EVAL_FIGURES = [
    ("object_pose_tracking", "Object pose vs the demonstration",
     "position and orientation of the bar against the recorded clip"),
    ("arm_action_per_joint", "Arm end-effector action",
     "commanded twist and a_t - a_(t-1), one panel per twist component"),
    ("hand_action_per_joint", "Hand action per joint",
     "action vs ideal (demo) vs a_t - a_(t-1), one panel per hand joint"),
]

# The sweep summary fields scripts/sweep_pose_success.py writes, in the order
# the dashboard shows them, as (key, label, digits, scale).
SWEEP_FIELDS = [
    ("fail_fraction", "fail", 2, 1.0),
    ("median_peak_lift_m", "lift mm", 0, 1000.0),
    ("median_final_orientation_error_rad", "ori rad", 2, 1.0),
    ("mean_contact_fraction", "contact", 2, 1.0),
    ("mean_rms_ee_action_rate", "rms ee", 3, 1.0),
    ("median_fail_reference_index", "fail idx", 0, 1.0),
    ("blown_count", "blown", 0, 1.0),
]

# Fields whose value is per-run noise rather than an experiment choice.
IGNORED_CONFIG_PATHS = {
    "env.num_envs",
    "env.play",
    "env.debug",
    "seed",
    "sim.device",
    "viewer.enable_viewer",
    "viewer.camera_position",
    "viewer.camera_lookat",
    "viewer.training_camera_enabled",
    "viewer.reference_ghost",
    "train.runner.run_name",
    "train.runner.experiment_name",
    "train.runner.record_video",
    "train.runner.record_gif",
    "train.runner.max_iterations",
    "train.runner.wandb",
    "train.runner.wandb_group",
    "train.runner.tensorboard",
    "train.runner.tensorboard_flush_secs",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "logs" / "simtoolreal",
        help="Directory holding the run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "dashboard" / "training_dashboard.html",
        help="HTML file to write.",
    )
    parser.add_argument(
        "--featured",
        type=int,
        default=5,
        help="Most recent runs that get full charts, videos and config diff.",
    )
    parser.add_argument(
        "--videos-per-run",
        type=int,
        default=5,
        help="Recorded clips embedded per featured run (newest first).",
    )
    parser.add_argument(
        "--eval-videos-per-run",
        type=int,
        default=2,
        help="Evaluation replays embedded per featured run (newest first).",
    )
    parser.add_argument(
        "--eval-video-width",
        type=int,
        default=560,
        help="Width the embedded evaluation replays are re-encoded to.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=240,
        help="Points per series after downsampling.",
    )
    parser.add_argument(
        "--live-minutes",
        type=float,
        default=6.0,
        help="A run whose metrics changed this recently counts as live.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=480,
        help="Width the embedded clips are re-encoded to.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=28,
        help="x264 quality for the embedded clips (higher is smaller).",
    )
    parser.add_argument(
        "--max-video-mb",
        type=float,
        default=7.0,
        help="Total budget for embedded video, before base64 expansion.",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=1100,
        help="Width the embedded evaluation plots are re-encoded to.",
    )
    parser.add_argument(
        "--max-figure-mb",
        type=float,
        default=2.5,
        help="Total budget for embedded evaluation plots, before base64.",
    )
    parser.add_argument(
        "--no-videos",
        dest="videos",
        action="store_false",
        help="Skip video embedding entirely (much faster).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include runs started on or after this YYYY-MM-DD date.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Discovery


def discover_runs(log_root, since=None):
    """Return every directory that holds both a config and a metrics file."""
    runs = []
    for config_path in sorted(log_root.glob("**/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            continue
        started = run_started_at(run_dir)
        if since is not None and started is not None and started < since:
            continue
        runs.append(run_dir)
    return runs


def run_started_at(run_dir):
    """Parse the leading YYYY-MM-DD[_HHMMSS] stamp the run directories carry."""
    for part in (run_dir.name, run_dir.parent.name):
        match = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?", part)
        if match:
            return match.group(1)
    return None


def started_timestamp(run_dir):
    """Sort key: the directory stamp, falling back to the config's mtime."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", str(run_dir))
    if match:
        try:
            return datetime.strptime(
                match.group(1) + match.group(2), "%Y-%m-%d%H%M%S"
            ).timestamp()
        except ValueError:
            pass
    return (run_dir / "config.json").stat().st_mtime


def host_of(run_dir):
    """The machine a run trained on, from the pull prefix on its directory."""
    for prefix, host in HOST_PREFIXES.items():
        if run_dir.name.startswith(prefix):
            return host
    return "desktop"


def run_key(run_dir):
    """The bare recipe name: no host prefix, no timestamp.

    This is the key scripts/build_control_panel.py files the wave sheet under,
    and the name the supervisor and the GPU agents use for a run.
    """
    name = run_dir.name
    for prefix in HOST_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{6}_", "", name)


def wave_entry(key):
    """What this run is asking, from the campaign sheet in the control panel."""
    entry = (WAVE.get("runs") or {}).get(key)
    if not entry:
        return None
    return {"change": entry.get("change"), "question": entry.get("question")}


# --------------------------------------------------------------------------
# Metrics


def count_lines(path):
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def load_metrics(path, max_points):
    """Downsample metrics.jsonl into the series the dashboard draws.

    Evaluation rows are sparse and always kept: they are the only
    deterministic measurement of the policy and would otherwise be sampled
    away.
    """
    total = count_lines(path)
    stride = max(1, total // max(1, max_points))
    series = {key: [] for key in SERIES_KEYS}
    last = None
    first = None
    recent_iteration_times = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            is_eval = '"evaluation_score"' in line
            if not (is_eval or index % stride == 0 or index == total - 1):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            iteration = row.get("iteration")
            if iteration is None:
                continue
            if first is None:
                first = row
            if not is_eval or last is None:
                last = row
            for key in SERIES_KEYS:
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    series[key].append([int(iteration), round(float(value), 6)])
            duration = row.get("iteration_time_s")
            if isinstance(duration, (int, float)):
                recent_iteration_times.append(float(duration))
    series = {key: values for key, values in series.items() if values}
    return {
        "series": series,
        "first": first or {},
        "last": last or {},
        "rows": total,
        "seconds_per_iteration": (
            sum(recent_iteration_times[-40:]) / len(recent_iteration_times[-40:])
            if recent_iteration_times
            else None
        ),
    }


def trend(points, window=0.25):
    """Signed change of a series' tail, as (delta, relative_delta)."""
    if len(points) < 6:
        return None, None
    tail = max(3, int(len(points) * window))
    early = [value for _, value in points[-2 * tail:-tail]] or [points[0][1]]
    late = [value for _, value in points[-tail:]]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    delta = late_mean - early_mean
    scale = max(abs(early_mean), 1e-9)
    return delta, delta / scale


# --------------------------------------------------------------------------
# Configuration


def flatten(values, prefix=""):
    flat = {}
    for key, value in values.items():
        path = "{}.{}".format(prefix, key) if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def default_flat_config():
    env = flatten(config_to_dict(SimToolRealCfg()))
    train = flatten(config_to_dict(SimToolRealTrainCfg()), "train")
    env.update(train)
    return env


def config_difference(saved, defaults):
    """Everything this run set differently from the repository defaults."""
    flat = flatten(saved.get("env_cfg", {}))
    flat.update(flatten(saved.get("train_cfg", {}), "train"))
    rows = []
    for path in sorted(set(flat) | set(defaults)):
        if path in IGNORED_CONFIG_PATHS:
            continue
        mine = flat.get(path, "<absent>")
        theirs = defaults.get(path, "<absent>")
        if mine == theirs:
            continue
        if isinstance(mine, float) and isinstance(theirs, float):
            if math.isclose(mine, theirs, rel_tol=1e-9, abs_tol=1e-12):
                continue
        rows.append({"path": path, "value": mine, "default": theirs})
    return rows


def reconstruct_command(runtime):
    """Rebuild the train.py invocation from the runtime block config.json saved."""
    parts = ["python scripts/train.py"]
    simple = [
        ("iterations", "--iterations"),
        ("num_envs", "--num-envs"),
        ("seed", "--seed"),
        ("run_name", "--run-name"),
        ("log_dir", "--log-dir"),
        ("resume", "--resume"),
        ("start_iteration", "--start-iteration"),
        ("save_interval", "--save-interval"),
        ("eval_interval", "--eval-interval"),
        ("eval_num_envs", "--eval-num-envs"),
        ("eval_seed", "--eval-seed"),
        ("object_assist_start_iteration", "--object-assist-start-iteration"),
        ("object_assist_end_iteration", "--object-assist-end-iteration"),
        ("final_eval_rsi_index", "--final-eval-rsi-index"),
        ("physics", "--physics"),
        ("sim_device", "--sim-device"),
        ("viz", "--viz"),
    ]
    for key, flag in simple:
        value = runtime.get(key)
        if value not in (None, ""):
            parts.append("{} {}".format(flag, value))
    # The mutually exclusive pairs: None means "whatever the config says".
    for key, on, off in [
        ("record_video", "--record-video", "--no-record-video"),
        ("object_assist", "--object-assist", "--no-object-assist"),
        ("contact_observations", "--contact-observations",
         "--no-contact-observations"),
    ]:
        value = runtime.get(key)
        if value is True:
            parts.append(on)
        elif value is False:
            parts.append(off)
    for key, flag in [
        ("no_periodic_eval", "--no-periodic-eval"),
        ("domain_randomization", "--domain-randomization"),
        ("asymmetric_critic", "--asymmetric-critic"),
    ]:
        if runtime.get(key):
            parts.append(flag)
    if runtime.get("final_eval") is False:
        parts.append("--no-final-eval")
    for override in runtime.get("overrides") or []:
        parts.append("--set {}".format(override))
    return " ".join(parts)


# --------------------------------------------------------------------------
# Verdict sweeps


def sweep_checkpoint(path, payload):
    """(sort key, label) for one sweep file, newest checkpoint first.

    A sweep of ``best_model.pt`` has no iteration of its own, so it sorts
    after every numbered checkpoint and is labelled ``best``.
    """
    checkpoint = str(payload.get("checkpoint") or "")
    match = re.search(r"model_(\d+)\.pt", checkpoint)
    if match:
        return int(match.group(1)), "it {}".format(int(match.group(1)))
    if "best" in checkpoint or path.stem.endswith("best"):
        return -1, "best"
    match = re.search(r"_(\d+)$", path.stem)
    if match:
        return int(match.group(1)), "it {}".format(int(match.group(1)))
    return -1, path.stem


def sweep_kind(name):
    if name.startswith("sweep_one_rsi0"):
        return "one pose, RSI 0"
    if name.startswith("sweep_one"):
        return "one pose"
    if name.startswith("sweep_grid"):
        return "grid"
    return name.replace("sweep_", "").replace("_", " ")


def load_sweeps(run_dir):
    """The pose-sweep verdicts sitting next to a run, newest checkpoint first.

    These are what scripts/sweep_pose_success.py writes: the fraction of
    placements whose object leaves the 7 cm criterion, and the physical
    numbers behind it. Episode means hide the grasp; this does not.
    """
    sweeps = []
    for path in sorted(run_dir.glob("sweep_*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        order, label = sweep_checkpoint(path, payload)
        values = {}
        for key, _, _, _ in SWEEP_FIELDS:
            value = summary.get(key)
            values[key] = (
                float(value)
                if isinstance(value, (int, float)) and math.isfinite(value)
                else None
            )
        sweeps.append(
            {
                "file": path.name,
                "kind": sweep_kind(path.name),
                "checkpoint": label,
                "order": order,
                "rsi_index": payload.get("rsi_index"),
                "threshold_m": payload.get("threshold_m"),
                "bank_index": payload.get("bank_index"),
                "episodes": len(payload.get("rows") or []),
                "summary": values,
            }
        )
    sweeps.sort(key=lambda item: (-item["order"], item["file"]))
    return sweeps


# --------------------------------------------------------------------------
# Videos and figures


def resolve_ffmpeg():
    """The ffmpeg to re-encode with: the system one, else imageio-ffmpeg's.

    The training venv ships imageio-ffmpeg (train.py records video with it),
    so a box without a system ffmpeg still gets its clips inlined.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not this interpreter's venv, try it directly
        pass
    venv_python = REPO_ROOT / "deps" / "IsaacLab" / ".venv" / "bin" / "python"
    if venv_python.is_file():
        try:
            out = subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env={"PATH": "/usr/bin:/bin"},
            ).stdout.strip().splitlines()
            if out and Path(out[-1]).is_file():
                return out[-1]
        except (OSError, subprocess.SubprocessError):
            return None
    return None


FFMPEG = None


def encode_videos(run_dir, count, width, crf, budget_bytes, kind="training"):
    """Re-encode the newest clips small enough to inline as data URIs.

    ``training`` clips are the 10 s rollouts the trainer records every 500
    updates, named by iteration. ``evaluation`` clips are the deterministic
    replays the evaluator writes, and are ordered by mtime instead.
    """
    video_dir = run_dir / ("videos" if kind == "training" else "eval_videos")
    if not video_dir.is_dir() or count <= 0 or not FFMPEG:
        return [], 0
    order = (
        iteration_of
        if kind == "training"
        else (lambda path: path.stat().st_mtime)
    )
    clips = sorted(video_dir.glob("*.mp4"), key=order, reverse=True)[:count]
    encoded = []
    used = 0
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "clip.mp4"
        for clip in sorted(clips, key=order):
            if used >= budget_bytes:
                break
            command = [
                FFMPEG, "-v", "error", "-y", "-i", str(clip),
                "-vf", "scale={}:-2".format(int(width)),
                "-r", "24",
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                "-pix_fmt", "yuv420p", "-crf", str(int(crf)),
                "-movflags", "+faststart", "-an", str(target),
            ]
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError):
                continue
            payload = target.read_bytes()
            used += len(payload)
            encoded.append(
                {
                    "iteration": iteration_of(clip),
                    "name": clip.stem,
                    "bytes": len(payload),
                    "src": "data:video/mp4;base64,"
                    + base64.b64encode(payload).decode("ascii"),
                }
            )
    return encoded, used


def iteration_of(path):
    match = re.search(r"iteration_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def encode_eval_figure(run_dir, stem, width):
    """Inline one evaluation plot by file stem, scaled and re-encoded as JPEG.

    These plots are one tall panel per joint -- the hand one is 20 -- so the
    source PNG runs to megabytes and has to be recompressed to survive
    inlining.
    """
    candidates = sorted(run_dir.glob("eval_plots/**/{}.png".format(stem)))
    if not candidates or not FFMPEG:
        return None
    source = candidates[-1]
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "{}.jpg".format(stem)
        command = [
            FFMPEG, "-v", "error", "-y", "-i", str(source),
            "-vf", "scale={}:-2".format(int(width)),
            "-q:v", "7", str(target),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        payload = target.read_bytes()
    return {
        "name": str(source.relative_to(run_dir)),
        "bytes": len(payload),
        "src": "data:image/jpeg;base64,"
        + base64.b64encode(payload).decode("ascii"),
    }


def encode_eval_figures(run_dir, width, budget_bytes):
    """Inline the evaluation plots for one run, inside a byte budget."""
    figures = []
    used = 0
    for stem, label, caption in EVAL_FIGURES:
        if used >= budget_bytes:
            break
        figure = encode_eval_figure(run_dir, stem, width)
        if figure is None:
            continue
        if used + figure["bytes"] > budget_bytes and figures:
            break
        figure["label"] = label
        figure["caption"] = caption
        figures.append(figure)
        used += figure["bytes"]
    return figures, used


def encode_overview_figure(run_dir, width=880):
    """Inline the final evaluation overview plot, if the run produced one."""
    candidates = sorted(run_dir.glob("eval_plots/**/overview.png"))
    if not candidates or not FFMPEG:
        return None
    source = candidates[-1]
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "overview.jpg"
        command = [
            FFMPEG, "-v", "error", "-y", "-i", str(source),
            "-vf", "scale={}:-2".format(int(width)),
            "-q:v", "6", str(target),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        payload = target.read_bytes()
    return {
        "name": str(source.relative_to(run_dir)),
        "src": "data:image/jpeg;base64,"
        + base64.b64encode(payload).decode("ascii"),
    }


# --------------------------------------------------------------------------
# Status and findings


def running_training_commands():
    """The train.py processes alive on this machine right now."""
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,etimes,args"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    processes = []
    for line in output.splitlines()[1:]:
        if "scripts/train.py" not in line or "ps -eo" in line:
            continue
        pid, elapsed, args = line.strip().split(None, 2)
        processes.append(
            {"pid": int(pid), "elapsed_s": int(elapsed), "command": args}
        )
    return processes


def gpu_status():
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    fields = [item.strip() for item in output.splitlines()[0].split(",")]
    if len(fields) != 5:
        return None
    return {
        "name": fields[0],
        "utilization": fields[1],
        "memory_used": fields[2],
        "memory_total": fields[3],
        "temperature_c": fields[4],
    }


def supervisor_done(run_dir):
    """The supervisor's verdict line, when the run has finished under it."""
    path = run_dir / "SUPERVISOR_DONE"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def classify(run, live_seconds):
    """live / finished / failed / diverged / stopped, the way the fleet reads it.

    A pulled run is only as live as its last rsync, so a remote run that the
    fleet pull has not touched recently reads as stopped rather than live.
    That is the honest statement: the page knows nothing newer.
    """
    last = run["last"]
    age = time.time() - run["updated_at"]
    if last.get("divergence_abort"):
        return "diverged"
    done = run.get("supervisor_done")
    if done:
        return "finished" if "status=done" in done else "failed"
    if age <= live_seconds:
        return "live"
    if run["planned_iterations"] and run["iteration"] + 1 >= run[
        "planned_iterations"
    ]:
        return "finished"
    return "stopped"


def findings(run):
    """Short plain-language readings of how the run is behaving."""
    notes = []
    series = run["series"]
    if run["status"] == "diverged":
        notes.append(
            {
                "level": "critical",
                "text": "Aborted by the divergence guard (action std above 15 "
                "or most targets clipped). The policy is not recoverable.",
            }
        )
    if run["status"] == "failed":
        notes.append(
            {
                "level": "critical",
                "text": "The supervisor gave up: {}".format(
                    run.get("supervisor_done") or "no progress"
                ),
            }
        )
    reward = series.get("mean_reward", [])
    delta, relative = trend(reward)
    if delta is not None:
        if relative > 0.02:
            notes.append(
                {
                    "level": "good",
                    "text": "Reward still climbing: {:+.3f} over the last "
                    "quarter of the run.".format(delta),
                }
            )
        elif relative < -0.02:
            notes.append(
                {
                    "level": "warning",
                    "text": "Reward is falling: {:+.3f} over the last quarter "
                    "of the run.".format(delta),
                }
            )
        else:
            notes.append(
                {
                    "level": "neutral",
                    "text": "Reward has plateaued ({:+.3f} over the last "
                    "quarter).".format(delta),
                }
            )

    # The verdict that decides the wave, when the run has been swept.
    sweep = next(
        (item for item in run.get("sweeps", []) if item["kind"] == "one pose"),
        None,
    ) or (run.get("sweeps") or [None])[0]
    if sweep:
        fail = sweep["summary"].get("fail_fraction")
        lift = sweep["summary"].get("median_peak_lift_m")
        if fail is not None and lift is not None:
            ok = fail < 0.5 and lift >= 0.15
            notes.append(
                {
                    "level": "good" if ok else "warning",
                    "text": "Pose sweep ({} {}): {:.0f}% of placements leave "
                    "the 7 cm criterion, median lift {:.0f} mm. Wave 1 wants "
                    "under 50% and at least 150 mm.".format(
                        sweep["kind"], sweep["checkpoint"], fail * 100.0,
                        lift * 1000.0,
                    ),
                }
            )
        index = sweep["summary"].get("median_fail_reference_index")
        if fail and index is not None:
            phase = (
                "before the grasp" if index < 740
                else "in the grasp" if index < 830
                else "in the carry"
            )
            notes.append(
                {
                    "level": "neutral",
                    "text": "Typical failure at reference frame {:.0f}, {}.".format(
                        index, phase
                    ),
                }
            )

    lift_series = (
        series.get("evaluation_fixed_mean_peak_object_com_lift_m")
        or series.get("evaluation_uniform_mean_peak_object_com_lift_m")
        or []
    )
    if lift_series:
        latest = lift_series[-1][1]
        if latest < 0.05:
            notes.append(
                {
                    "level": "warning",
                    "text": "Deterministic lift is {:.0f} mm; the demo lifts "
                    "250 mm and wave 1 wants at least 50 mm.".format(
                        latest * 1000.0
                    ),
                }
            )
        else:
            notes.append(
                {
                    "level": "good",
                    "text": "Deterministic lift {:.0f} mm (the demo lifts "
                    "250 mm).".format(latest * 1000.0),
                }
            )
    contact = series.get("mean_fingertip_contact_fraction", [])
    if contact:
        latest = contact[-1][1]
        notes.append(
            {
                "level": "good" if latest > 0.5 else "neutral",
                "text": "Fingertips touch the bar {:.0f}% of the time in "
                "training.".format(latest * 100.0),
            }
        )
    reached = series.get("episode_reference_end_fraction", [])
    if reached:
        latest = reached[-1][1]
        notes.append(
            {
                "level": "good" if latest > 0.3 else "neutral",
                "text": "{:.0f}% of episodes reach the end of the "
                "clip.".format(latest * 100.0),
            }
        )
    early = series.get("episode_early_termination_fraction", [])
    if early:
        latest = early[-1][1]
        if latest > 0.9:
            notes.append(
                {
                    "level": "warning",
                    "text": "{:.0f}% of episodes still end on a tracking "
                    "failure rather than reaching the horizon.".format(
                        latest * 100.0
                    ),
                }
            )
    std = series.get("mean_action_std", [])
    if std:
        delta, _ = trend(std)
        if std[-1][1] > 0.6 and (delta or 0) > 0:
            notes.append(
                {
                    "level": "warning",
                    "text": "Action std is {:.2f} and growing; the recipes cap "
                    "it at 0.5 and the divergence guard fires at 15.".format(
                        std[-1][1]
                    ),
                }
            )
    assist = series.get("object_assist_scale", [])
    if assist and assist[-1][1] > 0:
        notes.append(
            {
                "level": "neutral",
                "text": "Object assist is at scale {:.2f}; the policy carries "
                "the rest of the load.".format(assist[-1][1]),
            }
        )
    return notes


# --------------------------------------------------------------------------
# Assembly


def build_run(run_dir, defaults, args, featured, budget_bytes, figure_budget_bytes=0):
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    metrics_path = run_dir / "metrics.jsonl"
    metrics = load_metrics(
        metrics_path, args.max_points if featured else 60
    )
    last = metrics["last"]
    runtime = saved.get("runtime", {})
    runner = saved.get("train_cfg", {}).get("runner", {})
    planned = runtime.get("iterations") or runner.get("max_iterations")
    iteration = int(last.get("iteration", 0))
    checkpoints = sorted(
        (path.name for path in run_dir.glob("model_*.pt")),
        key=lambda name: int(re.findall(r"\d+", name)[0]),
    )
    key = run_key(run_dir)
    env_cfg = saved.get("env_cfg", {})
    run = {
        "id": str(run_dir.relative_to(args.log_root)),
        "name": run_dir.name,
        "key": key,
        "host": host_of(run_dir),
        "wave": wave_entry(key),
        "group": (
            str(run_dir.parent.relative_to(args.log_root))
            if run_dir.parent != args.log_root
            else None
        ),
        "path": str(run_dir),
        "started": run_started_at(run_dir),
        "started_at": started_timestamp(run_dir),
        "updated_at": metrics_path.stat().st_mtime,
        "iteration": iteration,
        "planned_iterations": int(planned) if planned else None,
        "series": metrics["series"],
        "last": last,
        "seconds_per_iteration": metrics["seconds_per_iteration"],
        "seed": env_cfg.get("seed"),
        "num_envs": env_cfg.get("env", {}).get("num_envs"),
        "episode_length": env_cfg.get("env", {}).get("episode_length"),
        "rsi": {
            "distribution": env_cfg.get("env", {}).get(
                "reference_init_distribution"
            ),
            "pregrasp_start": env_cfg.get("env", {}).get(
                "rsi_pregrasp_start_index"
            ),
            "max_start": env_cfg.get("env", {}).get("rsi_max_start_index"),
            "early_probability": env_cfg.get("env", {}).get(
                "rsi_early_probability"
            ),
        },
        "demonstration": Path(
            str(env_cfg.get("motion", {}).get("file", ""))
        ).name,
        "lineage": saved.get("lineage"),
        "command": reconstruct_command(runtime),
        "overrides": runtime.get("overrides") or [],
        "differences": config_difference(saved, defaults),
        "checkpoints": len(checkpoints),
        "best_checkpoint": (run_dir / "best_model.pt").is_file(),
        "supervisor_done": supervisor_done(run_dir),
        "sweeps": load_sweeps(run_dir),
        "featured": featured,
        "videos": [],
        "video_bytes": 0,
        "eval_videos": [],
        "eval_figures": [],
        "figure_bytes": 0,
        "overview": None,
        "total_time_s": last.get("total_time_s"),
        "total_timesteps": last.get("total_timesteps"),
    }
    run["status"] = classify(run, args.live_minutes * 60.0)
    run["findings"] = findings(run) if featured else []
    if featured and args.videos:
        videos, used = encode_videos(
            run_dir,
            args.videos_per_run,
            args.video_width,
            args.video_crf,
            budget_bytes,
        )
        run["videos"] = videos
        run["video_bytes"] = used
        # The evaluation replay is the deterministic one, so it is worth its
        # bytes even when the training clips already fit.
        replays, replay_bytes = encode_videos(
            run_dir,
            args.eval_videos_per_run,
            args.eval_video_width,
            args.video_crf,
            max(0, budget_bytes - used),
            kind="evaluation",
        )
        run["eval_videos"] = replays
        run["video_bytes"] += replay_bytes
        if not videos and not replays:
            run["overview"] = encode_overview_figure(run_dir)
    if featured and figure_budget_bytes > 0:
        figures, figure_used = encode_eval_figures(
            run_dir, args.figure_width, figure_budget_bytes
        )
        run["eval_figures"] = figures
        run["figure_bytes"] = figure_used
    return run


def main():
    global FFMPEG
    args = parse_args()
    log_root = args.log_root.expanduser().resolve()
    args.log_root = log_root
    if not log_root.is_dir():
        raise SystemExit("No such log root: {}".format(log_root))
    if args.videos:
        FFMPEG = resolve_ffmpeg()
        if FFMPEG is None:
            print("no ffmpeg (system or imageio-ffmpeg); embedding no videos")
            args.videos = False
        else:
            print("re-encoding with {}".format(FFMPEG))

    defaults = default_flat_config()
    run_dirs = discover_runs(log_root, args.since)
    if not run_dirs:
        raise SystemExit("No runs with metrics found under {}".format(log_root))
    run_dirs.sort(key=started_timestamp, reverse=True)
    print("Found {} runs under {}".format(len(run_dirs), log_root))

    # A run that is still writing metrics is always featured, wherever it
    # started: the point of the page is to watch what is training now, and the
    # fleet's newest run is not always the one that is alive.
    live_seconds = args.live_minutes * 60.0
    now = time.time()
    live_dirs = {
        run_dir
        for run_dir in run_dirs
        if not (run_dir / "SUPERVISOR_DONE").is_file()
        and now - (run_dir / "metrics.jsonl").stat().st_mtime <= live_seconds
    }

    budget = args.max_video_mb * 1024 * 1024
    figure_budget = args.max_figure_mb * 1024 * 1024
    runs = []
    for index, run_dir in enumerate(run_dirs):
        # A swept run has a verdict to show, so it gets a card too.
        featured = (
            index < args.featured
            or run_dir in live_dirs
            or any(run_dir.glob("sweep_*.json"))
        )
        try:
            run = build_run(run_dir, defaults, args, featured, budget, figure_budget)
        except Exception as error:  # a broken run must not sink the dashboard
            print("  skipped {}: {}".format(run_dir.name, error))
            continue
        budget -= run["video_bytes"]
        figure_budget -= run["figure_bytes"]
        runs.append(run)
        print(
            "  {:<52} {:<8} it {:>6} {:<9} {} clips {} plots {} sweeps".format(
                run["id"][:52],
                run["host"],
                run["iteration"],
                run["status"],
                len(run["videos"]) + len(run["eval_videos"]),
                len(run["eval_figures"]),
                len(run["sweeps"]),
            )
        )

    # A live run always leads, then the most recently touched.
    runs.sort(
        key=lambda run: (run["status"] != "live", -run["updated_at"])
    )
    payload = {
        "generated_at": time.time(),
        "generated_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": subprocess.run(
            ["hostname"], capture_output=True, text=True
        ).stdout.strip(),
        "log_root": str(log_root),
        "repo": str(REPO_ROOT),
        "gpu": gpu_status(),
        "processes": running_training_commands(),
        "wave": {
            "title": WAVE.get("title"),
            "baseline_label": WAVE.get("baseline_label"),
            "baseline": WAVE.get("baseline"),
            "sweep_baseline": WAVE.get("sweep_baseline"),
        },
        "series_meta": [
            {"key": key, "label": label, "unit": unit, "better": better}
            for key, label, unit, better in SERIES
        ],
        "sweep_fields": [
            {"key": key, "label": label, "digits": digits, "scale": scale}
            for key, label, digits, scale in SWEEP_FIELDS
        ],
        "runs": runs,
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_MARKER not in template:
        raise SystemExit("The template lost its {} marker".format(DATA_MARKER))
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(DATA_MARKER, blob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(
        "\nWrote {} ({:.1f} MB, {} runs, {} embedded clips, {} sweeps)".format(
            args.output,
            args.output.stat().st_size / 1024 / 1024,
            len(runs),
            sum(len(run["videos"]) + len(run["eval_videos"]) for run in runs),
            sum(len(run["sweeps"]) for run in runs),
        )
    )


if __name__ == "__main__":
    main()
