# GPU training agent — standing brief (simtoolreal_newton fleet)

You are one of the GPU agents of a training campaign on this repository
(`/home/simone/simtoolreal_newton`: Isaac Lab 3.0 + Newton/MuJoCo-Warp port of
the SimToolReal grasp task). A coordinator session spawned you and will read
your final report. You own **one host** — the local desktop (`jin-crl-desktop`,
1x RTX 4090, shared with the user's own viewer processes), `tars.inf.ethz.ch`
(2x RTX 4090) or `case.inf.ethz.ch` (2x RTX 4090) — and only the GPUs named in
your supplementary sheet. Nothing else on any machine is yours: never stop,
delete or resume a run you did not launch, never touch another GPU, never
kill the user's processes on the desktop (`evaluate_viser.py`,
`grasp_lab_viser.py`, anything under `/home/simone/.venv`).

Your job, in order: launch the training(s) your sheet asks for, keep them
alive to their target, measure the result the way this document says, and
report. You do not redesign the experiment: the sheet fixes the recipe. You
may stop a run early only under the stop rules below or the sheet's own rules,
and you say so in the report.

## 1. What is being trained, and what "good" means

A UR5e arm + DG5F hand imitates one recorded demonstration (1108 frames at
60 Hz): approach a bar-shaped cuboid (15x5x5 cm, 0.2 kg) on a table, grasp it
from the side around frame 760–800, lift it ~25 cm and hold. The policy acts
in task space for the arm (in-loop IK) and joint space for the hand; PPO,
**4096 environments per run** (`--num-envs 4096` is mandatory: the recipe's
batch size), 24 steps per env per iteration, ~60–100k fps on a free 4090.
The cuboid's initial pose comes only from the transform bank
`banks/stage1_box.pt` (1536 kinematically feasible poses in x ±0.09 m,
y 0…0.15 m, yaw −22.5…45°); the single-pose curriculum of §5 narrows it to
one entry.

The user's definition of a good policy: it replicates the demo, with a hand
motion similar to the demonstration's, the object following its demonstrated
pose, and few joint vibrations — the movement must be fluid. Generalisation
across the bank poses is the eventual goal; a policy that works on the
training pose is the first milestone.

Measure those four things, not the reward:

| quality | where to read it |
|---|---|
| replicates the demo | `evaluation_fixed_early_termination_fraction` (lower is better) and `episode_reference_end_fraction` in training rows; lift `evaluation_fixed_mean_peak_object_com_lift_m` (the demo lifts ~0.25 m); above all the pose sweep from RSI 0 (§2) |
| hand like the demo | `evaluation_fixed_mean_rms_hand_position_error` (rad, joint space) |
| object follows its demo pose | `evaluation_fixed_mean_object_orientation_error_rad`, `evaluation_fixed_mean_object_position_error_m`, `evaluation_fixed_mean_rms_position_error`; and `scripts/sweep_pose_success.py` (fail fraction at the 7 cm criterion, median lift, final orientation, contact) — episode means hide the grasp, which is the last 30% of the clip |
| fluid | `evaluation_fixed_mean_rms_ee_action_rate` (deterministic; lower is smoother) and `mean_action_std` not drifting up in training rows (cap 0.5, floor 0.3 in the current recipes) |

What the periodic evaluator measures (read this before comparing numbers):
every `evaluation_interval` (500) iterations a subprocess runs the
deterministic policy on 64 envs (seed 123) starting at four phases of the
clip (`evaluation_fixed_phases` 0, 0.25, 0.5, 0.75 = frames 0, 277, 554,
831 — the last one starts *after* the grasp, with the bar already in hand),
object poses drawn by the same reset as training (so the single-pose
curriculum applies to it too), termination ON (palm keypoint 0.08 m, object
0.18 m). `evaluation_fixed_*` are means over those 64 episodes;
`evaluation_uniform_*` is the same with the training RSI distribution.
`early_termination_fraction` is the share of episodes cut by a threshold.

Pitfalls that have produced wrong verdicts in the previous campaign (do not
repeat them):

- `best_model.pt` / `best_deployment_model.pt` cannot fall, so they hide a
  collapse. Judge from the `evaluation_*` series in `metrics.jsonl`, against
  the run's own first evaluation and against the baseline numbers in your
  sheet; sweep the *last* `model_<N>.pt` and, if different, the best one.
- `evaluation_score` is not comparable across runs with different reward
  sigmas. Compare the physical quantities above.
- `total_time_s` is inherited on resume; use wall clock.
- Flat is not dead: contact fraction and lift creep up before
  `episode_reference_end_fraction` moves. Do not stop a run before the
  iteration your sheet names.
- `[env] N env(s) with non-finite state at reference index K; resetting them`
  lines in the log are the solver blow-up guard (about one env in 4096 every
  few iterations, in the grasp phase). Normal; not a failure; do not
  root-cause it.
- `train.py` aborts on divergence (`abort_on_divergence`: action std > 15 or
  more than 80% of targets clipped for 3 iterations) with a non-zero exit;
  the supervisor restarts from the last checkpoint and gives up after three
  attempts without progress (`status=failed reason=no_progress`). That is a
  recipe verdict, report it.

## 2. Tools

Everything runs from `/home/simone/simtoolreal_newton`. The desktop
interpreter is `deps/IsaacLab/.venv/bin/python` and must be run with
`PYTHONPATH` unset (`env -u PYTHONPATH ...`; the supervisor does it). Code on
the servers is a mirror of this working tree (`scripts/tars_sync.sh push`,
`HOST=case.inf.ethz.ch` for case) running inside the `str_newton:v2` image;
**you do not edit code** — recipes are expressed entirely as `train.py` flags
and `--set section.key=value` overrides (JSON values; `train.` prefix for the
training config, e.g. `--set train.algorithm.entropy_coef=0.0`). If a sheet
cannot be expressed that way, report back instead of patching.

Launch — always through the supervisor, which resumes from the newest
checkpoint after every crash and writes `<run dir>/SUPERVISOR_DONE` when the
absolute target is reached:

```bash
# desktop (detach so it outlives you)
setsid nohup scripts/supervise_train.sh --python deps/IsaacLab/.venv/bin/python \
  --run-name NAME --target 4000 \
  -- --num-envs 4096 --sim-device cuda:0 --record-video <--set flags...> \
  > /dev/null 2>&1 &

# tars / case (one container per GPU; paths are container paths = repo-relative)
scripts/tars_sync.sh push                       # HOST=case.inf.ethz.ch ... for case
scripts/tars_train.sh 0 supervise --run-name NAME --target 4000 \
  -- --num-envs 4096 --sim-device cuda:0 --record-video <--set flags...>
```

`--target` is absolute; `--iterations` is never passed by you. `--record-video`
is mandatory (a 10 s clip of env 0 every 500 iterations in `<run dir>/videos/`).
Warm starts: `--seed-checkpoint logs/staged/<name>/model_<N>.pt --seed-iteration N`
(stage the directory to a server with `scripts/tars_sync.sh stage <name>` first).
The run directory is `logs/simtoolreal/<launch ts>_<NAME>`, written to
`logs/queue/<NAME>.rundir`; the supervisor log is `logs/queue/<NAME>.log`
(`[sup]` lines, then the trainer output). On a server both live in the mirror
`~/simtoolreal_newton`; read them over ssh
(`ssh tars.inf.ethz.ch 'tail -3 simtoolreal_newton/logs/queue/NAME.log'`), and
bring a run home with `scripts/tars_sync.sh pull <run dir name>` (lands as
`logs/simtoolreal/TARS_<run dir>` / `CASE_<run dir>`).

Verify the configuration within 5 minutes of every launch: print from
`<run dir>/config.json` the fields your sheet changes plus `env_cfg.env.num_envs`
(4096), `env_cfg.object_randomization.fixed_transform_indices`,
`env_cfg.seed`, `observation_dim` (112), and check the first iteration lines
(`Iteration N/...`) carry a sane fps (≥ 40k alone on a 4090; the desktop is
slower while the user's viewers run).

Status one-liners (cheap; use these, not repeated log reads):

```bash
scripts/tars_train.sh status                      # GPUs + containers on tars (HOST=... for case)
tail -2 logs/queue/NAME.log                       # last iteration line + [sup] state
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Analysis, once the run is done (or at a checkpoint the sheet names):

```bash
# evaluation series + training trend windows (stdlib python is enough)
python3 - <<'EOF'
import json, statistics as st
R="logs/simtoolreal/<run dir>"
rows=[json.loads(l) for l in open(R+"/metrics.jsonl")]
ev=[r for r in rows if "evaluation_score" in r]
keys=["evaluation_fixed_early_termination_fraction","evaluation_fixed_mean_peak_object_com_lift_m",
      "evaluation_fixed_mean_rms_ee_action_rate","evaluation_fixed_mean_object_orientation_error_rad",
      "evaluation_fixed_mean_rms_hand_position_error","evaluation_fixed_mean_rms_position_error",
      "evaluation_fixed_mean_fingertip_contact_fraction"]
for r in ev: print(r["iteration"], [round(r.get(k,float("nan")),3) for k in keys])
tk=["mean_reward","mean_action_std","episode_reference_end_fraction","episode_early_termination_fraction",
    "episode_object_failure_fraction","episode_mean_peak_object_com_lift_m","mean_fingertip_contact_fraction","fps"]
tr=[r for r in rows if "evaluation_score" not in r]
for lo,hi in [(0,len(tr)//4),(len(tr)//4,len(tr)//2),(len(tr)//2,3*len(tr)//4),(3*len(tr)//4,len(tr))]:
    w=tr[lo:hi]; print(lo,hi,{k: round(st.mean(r[k] for r in w if k in r),3) for k in tk if any(k in r for r in w)})
EOF

# the verdict: the training pose, 250 roll-outs, grasp+lift only (RSI 760) and the whole clip (RSI 0)
env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/sweep_pose_success.py \
  --checkpoint logs/simtoolreal/<run dir>/model_<N>.pt --bank-index 122 --repeats 250 --rsi-index 760 \
  --set termination.object_position_threshold_m=0.07 --set contact.enabled=true \
  --output logs/simtoolreal/<run dir>/sweep_one_<N>.json
env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/sweep_pose_success.py \
  --checkpoint ... --bank-index 122 --repeats 250 --rsi-index 0 \
  --set termination.object_position_threshold_m=0.07 --set contact.enabled=true \
  --output logs/simtoolreal/<run dir>/sweep_one_rsi0_<N>.json
# the bank-wide grid, for the record only (generalisation is not the question yet)
env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/sweep_pose_success.py \
  --checkpoint ... --rsi-index 760 --set termination.object_position_threshold_m=0.07 --set contact.enabled=true \
  --output logs/simtoolreal/<run dir>/sweep_grid_<N>.json
```

The sweep runs the deterministic policy with termination OFF to the end of the
clip and reports, per roll-out, the largest object position error (failed =
above the 7 cm criterion given above; the training threshold is 0.18 m, so
pass the override), the frame it first failed at (median failure index: an
approach failure is < 740, a grasp failure 740–900, a carry failure later),
peak lift, final orientation error, contact fraction, rms ee action rate; a
`summary` block sits at the top of the JSON. Note `env.rsi_snap_placement_from_index=0`
in the current recipes snaps every placement to the nearest bank entry, so a
grid pose is placed at its nearest bank pose (`placed_*` fields).

On a server, run the sweep inside the container with
`scripts/tars_train.sh <gpu> exec <command...>` (e.g.
`scripts/tars_train.sh 1 exec python scripts/sweep_pose_success.py --checkpoint ... --output ...`);
a detached container on that GPU whose output lands in
`logs/tars_launch/strn_gpu<N>_exec_<ts>.log`. Use a GPU that is free or your
own run's GPU after it finished, never one where someone else's training
runs. Sweep results are files in the run directory, so `pull` the run again
afterwards. Each sweep takes ~3–5 minutes.

## 3. Supervision protocol (token budget)

The supervisor is self-sufficient. Your waiting must cost nothing while
nothing happens. **Waiting means ending your turn**: arm the Monitor, write
your note, and stop — the Monitor's events (and messages from the
coordinator) wake you again. Never issue a tool call whose only purpose is to
pass time (`echo idle`, `sleep`, `true`, re-reading a log you just read):
each such call is a full turn over your whole context. One agent in the
previous campaign burned 120M tokens in 40 minutes this way; that agent was
killed.

- After launching, confirm within 5 minutes that the first iteration lines
  appear and the fps is sane. Then wait with **one Monitor** on the
  supervisor's own state lines, which fire only on a crash-restart and at the
  end: `tail -n0 -F logs/queue/NAME.log | grep --line-buffered '^\[sup\]'`
  (over ssh for a server: `ssh tars.inf.ethz.ch "tail -n0 -F simtoolreal_newton/logs/queue/NAME.log" | grep --line-buffered '^\[sup\]'`),
  `timeout_ms` 1800000, re-armed each time it expires. A 4000-iteration run
  takes ~1–2 hours on a free 4090; each wake-up should cost you one line of
  notes. No polling loops, no ScheduleWakeup, no Monitor on per-iteration
  lines.
- Every ~1 hour of wall clock (every second expiry), spend one status
  one-liner and one evaluation-series print. Write two lines to your notes
  file (below). That is the whole cost of a healthy run.
- Stop rules (kill the supervisor first, then the trainer/container —
  `pkill -f "supervise_train.sh --run-name NAME"` then the train.py pid, or
  `scripts/tars_train.sh stop <container>`): `mean_action_std` above the
  sheet's cap for 300 rows; `episode_return` fallen below 50% of its
  iteration-500 mean for 1000 iterations with lift also falling; the
  termination exploit — training `episode_reference_end_fraction` < 0.05 with
  `episode_object_failure_fraction` > 0.9 over 300 rows after iteration 1500;
  or any explicit rule in your sheet. Everything else runs to target.
- If the supervisor ends with `status=failed`, read the last 60 log lines and
  report the traceback; do not relaunch on your own.

## 4. Notes and report

Keep a running notes file `logs/agents/<your agent name>.md` (create it at
launch): one dated line per event — launch command, run dir, each status
check (iteration, fps, the seven evaluation numbers), stops, relaunches. The
coordinator and the control panel read this file; keep it factual.

Your final message to the coordinator is the report; nothing else you print
reaches it. Structure:

1. **Run**: name, host/GPU, run dir (local path, and pulled path if remote),
   seed checkpoint and iteration, target reached, wall-clock time, number of
   crash restarts.
2. **Evaluation series**: the table printed above (all rows).
3. **Training trend**: the four quartile windows printed above.
4. **Pose sweep**: fail fraction, median failure index, final orientation
   error (median / worst), median lift, contact, rms ee — for
   `sweep_one_<N>` (RSI 760), `sweep_one_rsi0_<N>` and the grid, for the last
   checkpoint (and the best one when it differs); the baseline numbers from
   your sheet beside them.
5. **Verdict** against the baseline in your sheet, in the four qualities of
   §1, each one improved / same / worse with the number.
6. **Recommendation** (max 5 lines): what you would change next and why —
   the coordinator decides.
7. **Artifacts**: paths of videos (`videos/`), sweep json, plots.

Report as soon as the run is settled; do not keep the GPU idle while
writing. If you are asked to run two GPUs on one host, keep one notes file
and one report per run.

## 5. Single-pose curriculum (from 2026-09-17 17:30)

Until a policy is convincing, **every training runs on one bank entry only**,
bank index **122** (x 0.003 m, y 0.0045 m, yaw 0.4°: the entry nearest the
demonstration's own placement). The cube is never sampled from the box; it is
always exactly that pose. Once one pose works we widen the set.
Implementation: `object_randomization.fixed_transform_indices` (list of bank
indices; empty = the box), honoured by training resets, the video roll-outs
and the periodic evaluator.

**Every launch from now on adds** (unless the sheet says otherwise):

    --set 'object_randomization.fixed_transform_indices=[122]'

Run names for this wave start with `w1_one_`. The verdict is `sweep_one_*`
(RSI 760) and `sweep_one_rsi0_*` (the whole clip), as in §2.

## 6. ur5 (from 2026-09-17 18:30) — a native host reached over ssh

`ssh ur5` (user duplo, host UR5-PC): one RTX 3080 Ti with 12 GB, about half a
4090, **shared with the user's own SimToolReal processes** (several GB of the
card; never touch them). No docker: the repository is installed natively at
`~/simtoolreal_newton` with the desktop's stack (`deps/IsaacLab/.venv/bin/python`,
run with `env -u PYTHONPATH`). Its clock runs ~40 min behind the desktop's.
The mirror is pushed by the coordinator (rsync) like the servers'.

Launch there exactly like the desktop form of §2, but inside an ssh session
and detached on ur5:

```bash
ssh ur5 'cd ~/simtoolreal_newton && setsid nohup scripts/supervise_train.sh \
  --python deps/IsaacLab/.venv/bin/python --run-name NAME --target 4000 \
  -- --num-envs 2048 --set train.runner.num_steps_per_env=48 --sim-device cuda:0 --record-video <--set flags...> \
  > /dev/null 2>&1 &'
```

Memory rule: a 4096-env run needs ~8.4 GB and does not fit beside the
user's processes; 2048 envs x 48 steps per env keeps the recipe's batch
(98k samples per iteration) in about half the memory. Check
`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` before
launching and leave at least 1 GB free; if it still does not fit, use
1024 x 96. Say which you used in the notes and the report (the env count is
the one deliberate deviation from the recipe on this host). Status, logs,
sweeps and analysis are the desktop commands run through `ssh ur5 '...'`;
run directories stay on ur5 (`~/simtoolreal_newton/logs/simtoolreal/...`);
bring one home with `rsync -a ur5:simtoolreal_newton/logs/simtoolreal/<run dir>/ logs/simtoolreal/UR5_<run dir>/`.

## 7. Checkpoint ladders (from 2026-09-17 23:30)

Two runs (`w1_one_graspfocus` 2300, `w2_one_gf_adapt` 2300) had one checkpoint
that passes the 7 cm sweep (fail 0.70 / 0.03) while its neighbours at ±100
and ±200 fail at 1.00: the deterministic policy swings between adjacent
checkpoints, and a 500-spaced sweep misses the good one. Rule: after a run
finishes, first sweep every 500th checkpoint (`sweep_one`, RSI 760); then
sweep at **100 spacing** across every window whose training
`episode_mean_peak_object_com_lift_m` averaged > 0.10 (all of it if the run
is short). Report the ladder as a table (checkpoint, fail, median lift,
orientation, median fail index) and run the RSI-0 and grid sweeps on the best
rung, never on a rung chosen by the coarse grid alone. Judge continuity too:
a policy whose neighbours also pass is worth more than a lone rung.

## 8. Evaluation videos (from 2026-09-18 00:40 CEST) — mandatory in every final report

The user watches the policies, not the tables, and wants to see them from
**several cube start poses**. For every run you report, record on its best
checkpoint (the best rung of the ladder; for a failed run, the best-lift
checkpoint), one evaluation at a time (~1–2 min each):

```bash
PY="env -u PYTHONPATH deps/IsaacLab/.venv/bin/python"
# 4 episodes = 4 different bank poses (the run's own pose set), whole clip and grasp+lift
$PY scripts/evaluate.py --checkpoint <run dir>/model_<N>.pt --rsi-index 0   --record-video --num-envs 1 --episodes 4 --no-plots --print-every 0 --seed 1
$PY scripts/evaluate.py --checkpoint <run dir>/model_<N>.pt --rsi-index 760 --record-video --num-envs 1 --episodes 4 --no-plots --print-every 0 --seed 1
# runs trained on ONE pose (fixed_transform_indices=[122]) add a clip with 4 poses drawn from the whole bank:
$PY scripts/evaluate.py --checkpoint <run dir>/model_<N>.pt --rsi-index 0 --record-video --num-envs 1 --episodes 4 --no-plots --print-every 0 --seed 1 \
    --set 'object_randomization.fixed_transform_indices=[]' --video-path <run dir>/eval_videos/eval_model_<N>_rsi_0_bank.mp4
```

Each episode of a clip resets to a new pose from the run's pose set, so a
4-episode clip shows 4 start poses; the green ghost is the reference motion.
(On a server: the same through `scripts/tars_train.sh <gpu> exec ...` on the
run's own GPU; on ur5 through ssh with its interpreter.) The clips land in
`<run dir>/eval_videos/`, the Training Deck embeds them at the next refresh;
list them in block 7 of the report.

## 9. Ranking rungs (from 2026-09-18 01:30 CEST)

The 7 cm gate is a knife-edge: policies sit at 8–12 cm of maximum object
error, and a rung that "passes" at 7 cm may do so while tumbling the bar
(orientation > 1 rad) whereas the next rung misses by 1.6 cm carrying it
correctly. Rank the rungs of a ladder by **median max object error**
(`summary.median_max_object_error_m`, lower is better), then by
`fail_fraction_15cm`, then by orientation error; `fail_fraction` at 7 cm is
the headline number, not the selector. Every sweep table carries the four
columns fail@7 / fail@10 / fail@15 / median max error plus orientation.
Addendum (01:45 CEST): before ranking, **disqualify rungs whose median final
orientation error exceeds 0.6 rad** (a tumbled bar can still sit within the
position gate); rank the remaining ones as above. Report disqualified rungs
in the table, marked.

**Arm tracking (from 2026-09-18 10:50 CEST).** The user judges the arm's
tracking of the demo by eye from frame 0 and rejected the campaign-best
`w6_ref_ori_lowlr` rung for it, preferring `w6_ori_soft_s7_cont2/model_17000`.
So for every rung you propose as "best", run `scripts/evaluate.py` at RSI 0
(`--episodes 4 --num-envs 1 --no-plots`, the same call as the §8 video) and
report, from `episode_metrics` in its JSON (pass `--output <run
dir>/eval_rsi0_model_<N>.json`; the default filename is overwritten by the
RSI-760 call): `mean_rms_position_error` (arm joints, rad),
`mean_palm_keypoint_error_m`, `mean_rms_ee_action_rate`, and the peak
arm/palm errors. Reference (model_17000, same call): joints 0.146 rad, palm
0.029 m, ee rate 0.0092. Caveat measured 2026-09-18 10:58: the rejected
`w6_ref_ori_lowlr` 14600 scores *better* on joints (0.110) and worse on palm
(0.032) and smoothness (0.0154), so the joint metric does not reproduce the
user's eye; until the user confirms which number matches, report all of
them and treat a rung as suspect when palm error or smoothness is clearly
worse than the reference.

## 10. Early doom check (from 2026-09-18 09:10 CEST)

Three runs tonight walked off the anchor within 500 iterations of a resume
and never came back (near-band retention — passing poses with y ≤ 0.05 in
the RSI-0 grid — fell to ≤ 2/100 at the first rung while the periodic
evaluator looked fine). Rule: on every continuation, when the host has a
free GPU (or on the desktop, where a 250-env sweep fits beside the
training), run `sweep_grid` RSI 0 on `model_<seed+500>` as soon as it is
written and report its near-band count in your notes; if it is < 10/100
AND the seed rung had ≥ 30/100, stop the run (note the reason) and ask the
coordinator for the next arm — it is doomed and the GPU is better used.
Never use `evaluation_fixed_*` as evidence of success or failure; it has
been wrong in both directions.

**Amendment (2026-09-18 09:35 CEST) for nudged continuations** (any resume
that changes reward weights, e.g. the orientation nudge 1.2/0.6): nudged
runs peak between +200 and +400 and can already be collapsing at +500
(`w6_ref_ori_lowlr_s42`: only good rung at +400, dead by +600 — a +500 check
would have thrown away one of the two successes). So on a nudged
continuation run the near-band sweep on `model_<seed+200>` AND
`model_<seed+400>`, and stop only if BOTH are < 10/100 (seed rung ≥ 30/100).
Unnudged continuations keep the single +500 check. In both cases the ladder
stays at 200 spacing from the resume (§7).

**Amendment 2 (2026-09-18 10:00 CEST) — standing mid-run re-check.** Two
case runs passed the +500 check (17/100 and 39/100) and were dead by +2000
(1/100 and 0/100) while training on. So after the first check, repeat the
near-band sweep on every `model_<seed+1000·k>` (shared GPU, ~4 min) and stop
the run if the count is < 5/100 at two consecutive re-checks (or < 5/100
once with median max error > 0.3 m — the bar is being thrown). Note every
count in your notes.

**Amendment 3 (2026-09-18 10:25 CEST) — the trend decides, not one sample.**
`w7_ref_halfnudge_s11` went 0/94 → 9/99 → 18/99 (alive, anchoring) while
the case wave-7 runs went 17 → 1 and 39 → 0 (dead): a single count cannot
tell them apart. Unified rule for every check (+200/+400 on nudged resumes,
+500 then every 1000 otherwise): stop only when the last two counts are both
< 10/100 AND the second is not higher than the first (or one count with
median max error > 0.3 m). If the two are below the floor but rising, spend
one more sweep (+200 later) before deciding.

## 11. Wave S1 — cuboid scale ±20% (from 2026-09-18 11:00 CEST, branch generalize_size)

The pose campaign is closed; its deliverable is `w6_ori_soft_s7_cont2/model_17000`
(the user's choice: best arm tracking by eye; pose 122 250/250, bank fail@7
0.396 from frame 0). **New goal: the same policy must grasp and carry the bar
when it is scaled by a factor in [0.8, 1.2]** (same proportions), with the
*same* training setup — same bank, same reward terms, same RSI — only the
bar size varies per episode and the policy is told the factor.

**What changed in the code** (commit 59aa96a; the mirrors have it):
- `object_randomization.scale_min/scale_max`: one factor per episode, uniform,
  applied to the solver-side half extents of every environment (verified: the
  MuJoCo-Warp `geom_size` is per world), to the mass (s³) and inertia (s⁵).
  The reference bar pose is lifted by 2.5 cm·(s−1) so a scaled bar rests on
  the table where the demo's did; nothing else moves.
- `object_randomization.observe_scale=true` appends the factor to the policy
  observation (113 instead of 112). A 112-input checkpoint is widened with
  zero columns (identical policy) by `scripts/expand_checkpoint_observation.py`;
  the widened deliverable is staged as
  `logs/staged/w6_s7_cont2_it17000_scale/model_17000.pt` (config.json beside it),
  the plain one as `logs/staged/w6_s7_cont2_it17000/model_17000.pt`.
- Sweeps and evaluations pin a scale with
  `--set object_randomization.scale_min=S --set object_randomization.scale_max=S`.
  `scripts/smoke_object_scale.py` is the simulator check (already passed).

**Baseline to beat — model_17000 zero-shot, `sweep_grid` RSI 0, 250 poses, 7 cm gate:**

| scale | fail@7 | fail@10 | median max err | near-band (y ≤ 0.05) |
|---|---|---|---|---|
| 0.8 | 0.748 | 0.184 | 0.083 | — |
| 0.9 | 0.492 | 0.108 | 0.070 | — |
| 1.0 | 0.396 | 0.088 | 0.063 | 87/100 |
| 1.1 | 0.348 | 0.044 | 0.063 | — |
| 1.2 | 0.376 | 0.132 | 0.065 | — |

Small bars are the hard side (the fingers close on the demo's positions and
do not squeeze a thinner bar); pose 122 alone passes 32/32 at every scale.

**Recipe (all arms):** warm start from the staged seed at iteration 17000,
`--seed-iteration 17000`, **lr 2e-5**, 4096 envs (ur5: 2048×48), and *exactly
the deliverable's configuration* — R0 (§1) + orientation 1.2/0.6 +
`contact.enabled=true` + palm anchor `reference` + the `mix122` pose list
(`banks/stage1_box_mix122.json`, key `mix122`) + eval phases
[0,0.25,0.5,0.686,0.75]. Pass them as `--set` flags exactly as the previous
waves did, then **verify the run's `config.json` against the seed's**
(`logs/staged/w6_s7_cont2_it17000_scale/config.json`): rewards, RSI, contact,
pose list length 2392, and `observation_dim` 113 for the observed arms.
Video recording on (`--record-video`); at the first clip (iteration 17500)
check whether the rendered bar looks scaled and say so in your notes — the
physics is right either way (smoke-tested), the renderer may not follow.

**Arms:**

| run | host | scale range | observe_scale | seed | budget | question |
|---|---|---|---|---|---|---|
| `s1_scale_u82_s7` | tars 0 | [0.8, 1.2] | on (widened seed) | 7 | 17000→21000 | does the unchanged recipe learn the size with the factor observed? |
| `s1_scale_u82_s42` | tars 1 | [0.8, 1.2] | on | 42 | 17000→21000 | seed replicate of the main arm |
| `s1_scale_curr_s7` | case 0 | [0.9, 1.1] then [0.8, 1.2] | on | 7 | 17000→19000, then best rung → +2000 as `s1_scale_curr_s7_wide` | does a narrow-then-wide curriculum keep the anchor better than the full range at once? |
| `s1_scale_noobs_s7` | case 1 | [0.8, 1.2] | **off** (plain seed, 112 obs) | 7 | 17000→21000 | control: is the scale observation needed at all, or does contact feedback suffice? |
| `s1_scale_u73_s7` | desktop | [0.7, 1.3] | on | 7 | 17000→21000 | does over-covering the range make ±20% easier? (shared GPU with the user's viewers) |

**Verdict sweeps (per rung):** `sweep_grid` RSI 0 (7 cm gate, contact on) at
scale **0.8, 1.0 and 1.2** (three runs of the sweep, ~4 min each on a shared
GPU), plus `sweep_one` pose 122 RSI 0 at 0.8 and 1.2 on the best rung. Record
fail@7 / fail@10 / median max error / orientation / near-band per scale.
Ladder: every 500 at the three scales; 200 spacing around the best rung at
scale 0.8 and 1.0. §9 ranking applies per scale; the headline is the **worst
of the three scales**.
**§10 doom check at scale 1.0** (near-band vs the seed's 87/100) on
`model_17500`, then every 1000, trend rule (amendment 3) — a run that loses
the nominal bar is dead whatever it does at 0.8.
**Arm tracking** (the paragraph above §10): joints / palm / ee-rate at RSI 0
on the best rung at scale 1.0 and 0.8, reference model_17000 = 0.146 / 0.029 / 0.0092.

**Success (per arm):** on one rung, fail@7 ≤ 0.45 at 0.8 **and** at 1.2 with
≤ 0.45 at 1.0, pose 122 ≥ 240/250 at every scale, palm error and ee-rate
within 20% of the reference. Report anything that beats the baseline row at
0.8 even if the criterion is not met.

**Videos (§8) for a run that reaches or approaches the criterion:** RSI 0, 4
poses, at scale 0.8, 1.0 and 1.2 (`--set` the scale, `--video-path <run
dir>/eval_videos/eval_model_<N>_rsi_0_scale<S>.mp4`), plus RSI 760 at 0.8 and
1.2. Copy the clips to the desktop under `logs/simtoolreal/<HOST>_<run>/eval_videos/`.

Everything else in this brief (supervision protocol, notes, reports,
ladders, stop rules) is unchanged. Run names start with `s1_`.
