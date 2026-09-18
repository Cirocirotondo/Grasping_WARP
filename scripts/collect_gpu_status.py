#!/usr/bin/env python3
"""Report what every GPU on this machine is training, as JSON on stdout.

    python3 scripts/collect_gpu_status.py --host tars

Runs unchanged on the desktop and on tars/case (standard library only, the
servers' system python has no torch): the control panel builder calls it
locally and over ssh and merges the results. A run is found through the
process or container that drives it (train.py's --run-name locally, the
strn_gpu<N>_<run> container name on a server), through the supervisor's
`logs/queue/<run>.rundir` marker, or -- for anything else still writing --
through a metrics.jsonl touched in the last 15 minutes. Recently finished
runs (SUPERVISOR_DONE younger than two days) are reported too, so a GPU that
just went idle still shows what it produced.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

EVAL_KEYS = {
    "early_term": "evaluation_fixed_early_termination_fraction",
    "lift": "evaluation_fixed_mean_peak_object_com_lift_m",
    "rms_ee": "evaluation_fixed_mean_rms_ee_action_rate",
    "ori_err": "evaluation_fixed_mean_object_orientation_error_rad",
    "rms_hand": "evaluation_fixed_mean_rms_hand_position_error",
    "rms_pos": "evaluation_fixed_mean_rms_position_error",
}
TRAIN_KEYS = {
    "reward": "mean_reward",
    "action_std": "mean_action_std",
    "ref_end": "episode_reference_end_fraction",
    "early_term": "episode_early_termination_fraction",
    "lift": "episode_mean_peak_object_com_lift_m",
    "contact": "mean_fingertip_contact_fraction",
    "fps": "fps",
}
LIVE_WINDOW_S = 15 * 60
DONE_WINDOW_S = 48 * 3600


def sh(command):
    try:
        return subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=20
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def gpu_status():
    out = sh(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu"
        " --format=csv,noheader,nounits"
    )
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            gpus.append(
                {
                    "index": int(parts[0]),
                    "mem_used_mib": int(float(parts[1])),
                    "mem_total_mib": int(float(parts[2])),
                    "util": int(float(parts[3])),
                }
            )
    return gpus


def drivers_local():
    """{run_name: (gpu, handle)} from live train.py processes."""
    found = {}
    out = sh("pgrep -af 'python[0-9.]* scripts/train\\.py'")
    for line in out.splitlines():
        match = re.search(r"--run-name[= ]([^ ]+)", line) or re.search(
            r"train\.runner\.run_name=([^ ]+)", line
        )
        if not match:
            continue
        device = re.search(r"--sim-device[= ]cuda:(\d+)", line)
        found[match.group(1)] = (int(device.group(1)) if device else 0, "pid " + line.split()[0])
    return found


def drivers_docker():
    """{run_name: (gpu, handle)} from strn_gpu<N>_<run> containers, running or not."""
    found = {}
    out = sh(
        ". ~/.docker_env.sh 2>/dev/null; docker ps -a --filter name=strn_gpu"
        " --format '{{.Names}}\t{{.Status}}'"
    )
    for line in out.splitlines():
        name, _, status = line.partition("\t")
        match = re.match(r"strn_gpu(\d+)_(.+)$", name)
        if not match or match.group(2).startswith("exec_"):
            continue
        found[match.group(2)] = (int(match.group(1)), name + " (" + status + ")")
    return found


def run_dir_for(run_name, log_root):
    marker = os.path.join("logs", "queue", run_name + ".rundir")
    if os.path.isfile(marker):
        path = open(marker).read().strip()
        if os.path.isdir(path):
            return path
    matches = sorted(glob.glob(os.path.join(log_root, "*_" + run_name)))
    return matches[-1] if matches else None


def tail_lines(path, n=400):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1500000))
            data = handle.read().decode("utf-8", "replace")
        return data.splitlines()[-n:]
    except OSError:
        return []


def read_rows(path, keep=800):
    rows = []
    for line in tail_lines(path, keep):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def all_eval_rows(path):
    """Every evaluation row: the file is scanned once, cheap enough at 10k rows."""
    evals = []
    try:
        with open(path) as handle:
            for line in handle:
                if '"evaluation_score"' not in line:
                    continue
                try:
                    evals.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return evals


def supervisor_state(run_name, run_dir):
    log = os.path.join("logs", "queue", run_name + ".log")
    state = {"attempts": 0, "target": None, "last_sup": None, "log": log if os.path.isfile(log) else None}
    if not os.path.isfile(log):
        return state
    lines = tail_lines(log, 3000)
    for line in lines:
        if line.startswith("[sup]"):
            state["last_sup"] = line
            if "tentativo" in line or "attempt" in line:
                if "uscito" not in line and "exited" not in line:
                    state["attempts"] += 1
            match = re.search(r"target (\d+)", line)
            if match:
                state["target"] = int(match.group(1))
        match = re.match(r"Iteration (\d+)/(\d+)", line)
        if match:
            state["target"] = int(match.group(2))
            state["log_iteration"] = int(match.group(1))
    return state


def describe_run(run_name, run_dir, gpu, handle, log_root):
    metrics = os.path.join(run_dir, "metrics.jsonl")
    now = time.time()
    try:
        age = now - os.path.getmtime(metrics)
    except OSError:
        age = None
    rows = read_rows(metrics)
    train_rows = [r for r in rows if "iteration" in r and "evaluation_score" not in r]
    last = train_rows[-1] if train_rows else {}
    sup = supervisor_state(run_name, run_dir)
    done_marker = os.path.join(run_dir, "SUPERVISOR_DONE")
    done = open(done_marker).read().strip() if os.path.isfile(done_marker) else None
    iteration = int(last.get("iteration", sup.get("log_iteration") or 0))
    target = sup.get("target")

    # Wall-clock pace from the run's own clock: total_time_s is inherited on
    # resume, but its differences between recent rows are real seconds.
    rate = None
    if len(train_rows) >= 20:
        a, b = train_rows[-min(300, len(train_rows))], train_rows[-1]
        dt = b.get("total_time_s", 0) - a.get("total_time_s", 0)
        di = b.get("iteration", 0) - a.get("iteration", 0)
        if dt > 0 and di > 0:
            rate = 60.0 * di / dt
    eta_min = (target - iteration) / rate if (rate and target and target > iteration) else None

    if done:
        state = "done" if "status=done" in done else "failed"
    elif age is not None and age < LIVE_WINDOW_S and handle:
        state = "running"
    elif handle and "(Up" in handle:
        state = "starting" if not train_rows else "running"
    else:
        state = "stalled" if handle and "Up" in handle else "stopped"

    evals = []
    for row in all_eval_rows(metrics):
        point = {"it": int(row.get("iteration", 0))}
        for short, key in EVAL_KEYS.items():
            if key in row:
                point[short] = round(float(row[key]), 4)
        evals.append(point)

    window = train_rows[-300:]
    train = {}
    for short, key in TRAIN_KEYS.items():
        values = [r[key] for r in window if key in r]
        if values:
            train[short] = round(sum(values) / len(values), 4)

    videos = sorted(glob.glob(os.path.join(run_dir, "videos", "*.mp4")))
    eval_videos = sorted(glob.glob(os.path.join(run_dir, "eval_videos", "*.mp4")))
    try:
        started = datetime.fromtimestamp(os.path.getctime(os.path.join(run_dir, "config.json")))
        started_iso = started.isoformat(timespec="minutes")
    except OSError:
        started_iso = None
    sweeps = {}
    for stem in ("sweep_best", "sweep_last", "sweep_one_last", "sweep_one_rsi0_last"):
        path = os.path.join(run_dir, stem + ".json")
        if os.path.isfile(path):
            try:
                sweeps[stem] = json.load(open(path))
            except ValueError:
                pass

    return {
        "run_name": run_name,
        "run_dir": os.path.basename(run_dir),
        "gpu": gpu,
        "handle": handle,
        "state": state,
        "iteration": iteration,
        "target": target,
        "it_per_min": round(rate, 2) if rate else None,
        "eta_min": round(eta_min) if eta_min is not None else None,
        "started_at": started_iso,
        "last_row_age_s": round(age) if age is not None else None,
        "attempts": sup["attempts"],
        "last_sup": sup["last_sup"],
        "done": done,
        "evals": evals,
        "train": train,
        "videos": len(videos),
        "eval_videos": len(eval_videos),
        "last_video": os.path.basename(videos[-1]) if videos else None,
        "sweeps": sweeps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="label: local, tars or case")
    parser.add_argument("--log-root", default="logs/simtoolreal")
    args = parser.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    drivers = drivers_local() if args.host in ("local", "ur5") else drivers_docker()
    runs = {}
    for run_name, (gpu, handle) in drivers.items():
        run_dir = run_dir_for(run_name, args.log_root)
        if run_dir:
            runs[run_name] = (run_dir, gpu, handle)
    now = time.time()
    for run_dir in glob.glob(os.path.join(args.log_root, "*")):
        metrics = os.path.join(run_dir, "metrics.jsonl")
        done = os.path.join(run_dir, "SUPERVISOR_DONE")
        live = os.path.isfile(metrics) and now - os.path.getmtime(metrics) < LIVE_WINDOW_S
        recent_done = os.path.isfile(done) and now - os.path.getmtime(done) < DONE_WINDOW_S
        if not (live or recent_done):
            continue
        run_name = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{6}_", "", os.path.basename(run_dir))
        if run_name not in runs:
            runs[run_name] = (run_dir, 0 if args.host in ("local", "ur5") else None, None)

    report = {
        "host": args.host,
        "hostname": os.uname().nodename,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpu_status(),
        "runs": [
            describe_run(name, run_dir, gpu, handle, args.log_root)
            for name, (run_dir, gpu, handle) in sorted(runs.items())
        ],
    }
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
