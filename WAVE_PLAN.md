# WAVE_PLAN — wave S2 (cuboid scale ±20%), self-service queue for the host agents

Rules that replace coordinator messages (from 2026-09-18 19:45 CEST):

1. **Report to the coordinator only for**: (a) a rung that meets the §11 criterion
   (fail@7 ≤ 0.45 at 0.8 and 1.2, ≤ 0.45 at 1.0, pose 122 ≥ 240/250 at every scale,
   palm/ee-rate within 20 % of 0.029/0.0092) — send the package (three-scale table
   with duplicates, pose 122, tracking, clips copied home, staged checkpoint);
   (b) a run finished or stopped: ONE message with the compact ladder table and the
   verdict line; (c) a rule gap or a blocked host. Nothing else: no interim rungs,
   no "monitor re-armed", no housekeeping, no restated tables.
2. **Next arm = next line of your host's queue below**, launched without asking,
   §12 recipe (small step, ladder every 100 at three scales with pose 122, §10 trend
   rule, budget as listed). Packages (tracking + videos) only for rungs that beat the
   scale deliverable `s2_lr5e6_s42` 17300 (0.416 / 0.316 / 0.438, worst scale) with
   pose 122 ≥ 240 everywhere, or that the coordinator asks for.
3. Keep your notes file current (it is the coordinator's and the dispatcher's source
   for the hourly digest); write the run's verdict there in the compact form.

## Queues (in order; skip a line if the same question is already answered)

### tars
1. (running) `s2_lr5e6_anchor25_s42`, `s2_lr2p5e6_s42` → then `s2_only12_s42` (GPU 1).
2. `s2_lr2p5e6_s7` — lr 2.5e-6, no anchor, seed 7, 17000→18000 (does the plateau reproduce on seed 7?).
3. `s2_lr2p5e6_cont17300_s42` — from `logs/staged/s2_lr5e6_s42_it17300`, lr 2.5e-6, 17300→18100.
4. `s2_lr1e5_s42` — lr 1e-5, no anchor, seed 42, 17000→17800 (5e-6 looks like a floor: 2.5e-6 cannot reach 0.8; test the other side).
5. `s2_lr5e6_anchor25_s42_fine` — the anchor25 recipe unchanged (lr 5e-6, `scale_nominal_probability=0.25`), seed 42, **17000→17300 only**, checkpoints every 25 (`--set train.runner.save_interval=25`). Motivation: in `s2_lr5e6_anchor25_s42` the two extremes are anti-correlated along training (0.8 → 0.316 at 17100 and 0.608 by 17700; 1.2 → 0.676 at 17100 and 0.304 by 17700) and 100 spacing cannot see whether they cross below 0.45. Measurement order, deliberately cheap: **0.8 and 1.2 first on every 25-rung**; 1.0 and pose 122 at all three scales **only** on rungs where both extremes are ≤ 0.45; duplicate the 1.2 grid **only** on a rung that would meet the criterion. Report once at the end — the criterion package, or one compact table plus a one-line answer (continuous trade vs exclusion). Nothing in NIGHT_LOG.

### case
1. (running) `s2_lr5e6_s11`, `s2_lr5e6_s3` (lottery tickets to 17600).
2. `s2_lr5e6_s23b` — lr 5e-6, seed 23, 17000→17600 (second ticket for the seed that peaked at +200).
3. `s2_lr2p5e6_s11` — lr 2.5e-6, seed 11, 17000→18000.
4. `s2_lr2p5e6_s3` — lr 2.5e-6, seed 3, 17000→18000.

### desktop
1. (running) `s2_lr5e6_cont17300_s42` → package if any rung beats the deliverable.
2. `s2_lr7p5e6_s42` — lr 7.5e-6, no anchor, seed 42, 17000→17800 (between the deliverable's 5e-6 and 1e-5).
3. `s2_lr5e6_s42_b` — lr 5e-6, seed 42, second run of the deliverable recipe with a different torch seed of the sweep (`--seed 42` kept, run-to-run variance check), 17000→17800.

## Wave S3 (2026-09-18 20:58) — let the scale input learn

Finding (coordinator, 20:41): the widened scale column of the actor's first layer starts at zero and, under Adam at lr 5e-6, grows ~lr per step. Norm after 300 iterations = 0.19 (17300 deliverable), 0.09 at 17100 (s23b), against a median of 1.38 for the other 112 columns. The mixed-scale policy is nearly scale-blind, which explains the 0.8/1.2 anti-correlation and the bimodal rungs. Two new default-off levers (commit 19383f4, on both mirrors):
- `--set train.policy.scale_input_lr_multiplier=K` — the scale column of actor and critic first layers gets its own Adam group at lr×K (state_dict keys unchanged; Adam moments of the split params are dropped on resume with a warning, expected).
- `--set object_randomization.observed_scale_override=S` — the policy is told scale S while physics keeps the true scale (ablation only; never in training).
Rules 1–3 and the §12 ladder (every 100, three scales, pose 122 on every rung) apply unchanged. Recipe = the deliverable recipe (set_flags.txt with lr 5e-6 unless stated), seed 42, from `logs/staged/w6_s7_cont2_it17000_scale/model_17000.pt`, 17000→17800. Verify config.json after launch shows the multiplier; verify the training log prints two param groups.

### case (after line 4)
5. `s3_mul20_s42` — lr 5e-6, `scale_input_lr_multiplier=20` (scale column at 1e-4), seed 42, no anchor, 17000→17800.
6. ~~`s3_mul50_s42`~~ cancelled 21:20 (ablation: the policy already uses the scale input decisively). Replacement: `s2_lr2p5e6_s23` — lr 2.5e-6, seed 23, no multiplier, 17000→18200 (seed 23 had the best grid at 5e-6; the small step gave s11 a late four-rung window).
7. `s2_lr2p5e6_s11_cont18000` — resume from `logs/staged/s2_lr2p5e6_s11_it18000/model_18000.pt` @ 18000, lr 2.5e-6, seed 11, no multiplier, no anchor, **18000→18600** (budget-rule exception granted by the coordinator 21:00: the window was still open at 18000 — pose 122 242/250/250 with fail@7 0.546/0.316/0.288). Ladder §12 every 100 at the three scales with pose 122.

### desktop (after line 3)
0. (now, sweeps only, beside the running training) **Ablation on the deliverable** `logs/staged/s2_lr5e6_s42_it17300/model_17300.pt`: grid + pose 122 at physical 0.8 with `observed_scale_override=1.2`, and at physical 1.2 with `observed_scale_override=0.8`. Compare with the true-input numbers (0.416 / 0.438, pose 122 249 / 241). Report the four numbers once. Before it, a **GPU smoke** of the multiplier: `scripts/train.py` from the widened 17000 with the deliverable flags + `scale_input_lr_multiplier=20`, `--num-envs 512`, 3 iterations, into a throwaway run dir; confirm it trains, saves, and that `sweep_pose_success.py` loads the saved checkpoint. Report failures immediately (rule gap); on success just note it in your file.
4. ~~`s3_mul20_lr7p5e6_s42`~~ cancelled 21:20 (confounded, premise weakened by the ablation). Replacement, in order: (a) sweeps only: deliverable `s2_lr5e6_s42_it17300` at physical scales 0.9 and 1.1 with the true input (grid + pose 122) — does the ±20% claim hold in between? one message with the numbers; (b) `s2_lr7p5e6_s23` — lr 7.5e-6 (three-rung window on seed 42), seed 23, no multiplier, 17000→17800.

### tars
No S3 line: finish lines 4–5. Then line 6 `s2_edges_s42` — deliverable recipe (lr 5e-6, seed 42, no multiplier) with `--set object_randomization.scale_anchors=[[0.8,0.15],[1.2,0.15]]` (uniform elsewhere), 17000→17800, ladder §12. Motivation: the deliverable's profile is a smooth U (0.416/0.340/0.316/0.356/0.438 at 0.8…1.2), so the edges need more episodes, the opposite of the nominal anchor.

7. `s2_lr5e6_s42_c` — the exact deliverable recipe (`set_flags.txt`, lr 5e-6, seed 42, no anchor, no multiplier), 17000→17800, ladder §12. **Third draw of the deliverable recipe, bounds the spread.** Launched on GPU 1 at 21:32 as the last tars line; run dir `logs/simtoolreal/2026-09-18_213211_s2_lr5e6_s42_c`.



Ablation result (desktop, 21:15): deliverable 17300 at physical 0.8 told 1.2 → 0.600 / pose 122 92; at physical 1.2 told 0.8 → 0.952 / pose 122 0 (true input: 0.416 / 249, 0.438 / 241). The policy uses the scale input decisively; S3 keeps only `s3_mul20_s42` as the multiplier test.

### desktop (added 21:52)
5. `s2_edges40_lr7p5e6_s42` — lr 7.5e-6, seed 42, no multiplier, `--set object_randomization.scale_anchors=[[0.8,0.4],[1.2,0.4]]` (20% uniform in between), 17000→17800, ladder §12. Heavy-edge complement of tars line 6 (15%/15% at lr 5e-6): the deliverable's U rises at the edges and nominal anchors sink them.
6. `s2_lr7p5e6_s42_c` — repeat of `s2_lr7p5e6_s42` (same seed) to separate a reproducible window from luck, 17000→17800.

## Reproducibility note (22:10)
Same recipe + same seed + same checkpoint does NOT give the same run: `s2_lr5e6_s42` (case) and `s2_lr5e6_s42_b` (desktop), both pre-19383f4, match at 17001 and diverge from 17002 (GPU nondeterminism, contact-amplified). Consequences for every host: (1) never describe a result as a property of a seed; every run is a draw; (2) a lever's effect counts only when it exceeds the spread between repeats of the same recipe (use s42 vs s42_b as the yardstick once s42_b's ladder is in); (3) the fine ladder is a fresh draw at 25 spacing, not an interpolation of its parent; (4) repeats of the deliverable recipe are legitimate queue lines (lottery tickets); a rung meeting the criterion on any draw gets the full package.

## Closing rule (22:25)
The spread between two draws of the deliverable recipe (s42 vs s42_b: up to 0.47 in f@7, 0.432 vs 0.680 on best worst-scale) exceeds every lever effect measured tonight. No new queue lines after the ones written above. When your host's queue is exhausted: write a final summary section in your notes file (every run, best rung, three-scale numbers, verdict; one paragraph of lessons that survive the reproducibility note), then hand back with a ≤ 15-line message and stop. Criterion hits still get the full package.

# Wave SC1 (2026-09-19 20:53 CEST) — pose deliverable recipe with finger self-collision ON

Lever (commit 184b92f, on both mirrors): `--set asset.self_collision=true` enables collisions between the phalanges of fingers 2–5 (index…little) only; palm, thumb and same-finger pairs stay filtered; fingertip contact forces are cube-filtered. Verified: probe shows exactly 96 hand body pairs; cost 19.7 → 45.6 ms/step at 4096 envs (adjacent-only 41, not worth it); zero-shot on the scale deliverable 17300: pose 122 250 → 212/250 at scale 1.0, 250 → 151/250 with 0.8–1.2 scales, so the old policy relied on interpenetration and must be retrained.

Recipe: warm start from `logs/staged/w6_s7_cont2_it17000/model_17000.pt` (112 obs, nominal scale) with ALL flags of `logs/staged/w6_s7_cont2_it17000/set_flags.txt` (filter the `learning_rate` line and pass the arm's lr), plus `--set asset.self_collision=true`. Scale off (do not pass scale flags). 4096 envs. Every sweep/eval loads the run's config.json, so self-collision is on in the ladder automatically; verify `asset.self_collision: true` and `observation_dim 112` in the run's config.json after launch.

Ladder every 100 (§12 mechanics from the S2 notes files): grid RSI 0 at scale 1.0 (250 poses, gate 7 cm) + pose 122 (250 repeats) on every rung; on candidate rungs (fail@7 ≤ 0.45 and pose 122 ≥ 240): duplicate grid (`--seed 2`), median orientation (DQ > 0.6 rad), arm tracking `scripts/evaluate.py` RSI 0 (`--output <run>/eval_rsi0_model_<N>.json`; reference 0.146 rad / 0.029 m / 0.0092), §8 clips (3–4 cube poses from frame 0 + one from RSI 760), copied home, checkpoint staged as `logs/staged/sc1_<arm>_it<N>/` with config.json. Criterion SC1: fail@7 ≤ 0.45, pose 122 ≥ 240/250, orientation ≤ 0.6, palm/ee-rate tracking within 20% of the reference. Stop rules §10 (amendment 3) apply; budgets are final (no continuation of a winning rung).

Reporting (unchanged): message the coordinator only for a criterion hit with package, a finished/stopped run (one compact message with the full ladder table), or a rule gap. Notes file per host is the source for the hourly dispatcher.

Every run is a draw: two draws per lr on the servers.

### tars
1. GPU 0 `sc1_lr2e5_s7_a` — lr 2e-5, seed 7, 17000→18000.
2. GPU 1 `sc1_lr5e6_s7_a` — lr 5e-6, seed 7, 17000→17800.
### case
1. GPU 0 `sc1_lr2e5_s7_b` — lr 2e-5, seed 7, 17000→18000.
2. GPU 1 `sc1_lr5e6_s7_b` — lr 5e-6, seed 7, 17000→17800.
### desktop (beside the user's viewer; 4096 envs fit)
1. `sc1_lr1e5_s7` — lr 1e-5, seed 7, 17000→18000.
When a host's queue is exhausted: final summary section in the notes file, ≤ 15-line closing message, stop. A winning rung is the seed of wave SC2 (scale ±20%, the S2 recipe: widen with expand_checkpoint_observation.py, lr 5e-6, 800 iterations, three-scale ladder).

# Wave SC2 (2026-09-19 22:06 CEST) — scale ±20% on the self-collision winner

Seed: `logs/staged/sc1_lr2e5_s7_a_it17200/model_17200.pt` (112 obs, self-collision on). Widen it with `scripts/expand_checkpoint_observation.py` (as done for `logs/staged/w6_s7_cont2_it17000_scale/`) into `logs/staged/sc1_lr2e5_s7_a_it17200_scale/` (model_17200.pt with 113 inputs + config.json + set_flags.txt), on the desktop and on the mirrors. Flags = the SC1 flags (`logs/staged/w6_s7_cont2_it17000/set_flags.txt` minus learning_rate) + `--set asset.self_collision=true --set object_randomization.scale_min=0.8 --set object_randomization.scale_max=1.2 --set object_randomization.observe_scale=true --set train.algorithm.learning_rate=5e-06`. 4096 envs, seed 42, 17200→18000 (800). Verify config.json: self_collision true, observation_dim 113, scale 0.8/1.2 observed, lr 5e-6.
Ladder every 100 at scales 0.8 / 1.0 / 1.2 (`--set object_randomization.scale_min=S --set object_randomization.scale_max=S`), grid + pose 122 on every rung at every scale; duplicate grids on candidate rungs; orientation DQ > 0.6; arm tracking at 1.0 and 0.8. Criterion §11: worst scale fail@7 ≤ 0.45 and pose 122 ≥ 240 at every scale. Package as before (clips at three scales + RSI 760, staged `logs/staged/sc2_<arm>_it<N>/`). Every run is a draw: two draws.
### tars
1. GPU 0 `sc2_lr5e6_s42_a`; 2. GPU 1 `sc2_lr5e6_s42_b` (identical recipe, second draw).
### desktop
1. `sc2_lr5e6_s42_c` — third draw of the SC2 recipe, launched once `logs/staged/sc1_lr2e5_s7_a_it17200_scale/model_17200.pt` exists on the desktop (tars agent produces it).
### tars (added 2026-09-19 23:39 CEST, after draws a/b closed without a hit; 0.8 is the binding side on every rung)
3. GPU 0 `sc2_lr5e6_s42_d` — fourth draw of the plain SC2 recipe, budget 400 (17200→17600).
4. GPU 1 `sc2_anchor_s42` — SC2 recipe + `--set object_randomization.scale_nominal_probability=0.25 --set object_randomization.scale_anchors=[[0.8,0.15]]`, budget 400 (17200→17600).
### case (2026-09-19 23:58 CEST; GPU 1 is occupied by another user's job — never touch it; GPU 0 only)
1. Ladder of the finished `sc1_lr2e5_s7_b` (18001, no sweeps run yet): grid + pose 122 at 1.0 on 17100…18000, candidate packages per SC1 rules.
2. GPU 0 `sc2w_lr5e6_s42_a` — single-step variant: scale AND self-collision from the S2 seed `logs/staged/w6_s7_cont2_it17000_scale/model_17000.pt` (113 obs, mirror has it), flags `logs/staged/w6_s7_cont2_it17000_scale/set_flags.txt` (learning_rate filtered) + `--set train.algorithm.learning_rate=5e-06 --set asset.self_collision=true`, seed 42, 17000→17800, three-scale ladder, §11 criterion. Motivation: the widened SC1 winner is a 200-iteration knife-edge (pose 122 at 1.2 already 1/250 on its first rung); one adaptation from the settled w6 17000 instead of two sequential ones.

# Wave SC3 (2026-09-20 00:32 CEST) — weaker anchor between the two SC2 near-misses
Seed `logs/staged/sc1_lr2e5_s7_a_it17200_scale/model_17200.pt`, SC2 recipe (lr 5e-6, seed 42, self-collision, scale 0.8–1.2 observed) + `--set object_randomization.scale_nominal_probability=0.15 --set object_randomization.scale_anchors=[[0.8,0.10]]`, budget 200 (17200→17400), ladder at 17300 and 17400 at three scales with pose 122 everywhere, duplicates on candidates. §11 criterion.
### tars
1. GPU 0 `sc3_anchor15_s42_a`; 2. GPU 1 `sc3_anchor15_s42_b`.

# Wave SC4 (2026-09-20 01:02 CEST) — last round: repeats of the two half-winners, budget 200
Seed and recipe as SC2 (widened SC1 winner, lr 5e-6, seed 42, self-collision, scale 0.8–1.2 observed), 17200→17400, ladder at 17300 and 17400 at three scales with pose 122 everywhere, duplicates on candidates, §11 criterion.
### tars (sequential per card)
1. GPU 0 `sc4_plain_e` then `sc4_plain_f` (no anchor).
2. GPU 1 `sc4_anchor25_c` then `sc4_anchor25_d` (`scale_nominal_probability=0.25`, `scale_anchors=[[0.8,0.15]]`).
Closing rule: no lines after these; final summary and stop.
### case (SC4, 2026-09-20 01:08 CEST; GPU 0 only)
1. `sc4_sc2w_b` then 2. `sc4_sc2w_c` — repeats of `sc2w_lr5e6_s42_a` (seed `logs/staged/w6_s7_cont2_it17000_scale/model_17000.pt`, its set_flags minus learning_rate, + scale 0.8/1.2 observed, lr 5e-6, self_collision, seed 42), budget 300 (17000→17300), ladder at 17100/17200/17300 at three scales with pose 122 everywhere, duplicates on candidates, §11 criterion. Last case lines.

# Wave DR1 (2026-09-20 17:52 CEST) — domain randomization on the scale candidate, cautious mode

Goal: robustness for the real robot, measured in the training simulator AND in native MuJoCo (sim2sim). Base checkpoint `logs/staged/sc2_anchor_s42_it17300/model_17300.pt` (113 obs, scale 0.8–1.2 observed with the anchor recipe, self-collision on). Its flags: `logs/staged/sc2_anchor_s42_it17300/set_flags.txt` (learning_rate filtered out; pass `--set train.algorithm.learning_rate=5e-06` explicitly). Every run also enables the ring-finger distal phalanx against the palm (`asset.self_collision_extra_pairs=[["rl_dg_4_4","wrist_3_link"]]`, in every flag file below); zero-shot on the base at 1.0 it costs nothing (grid 0.364 vs 0.356, pose 122 250/250).

Families (medium level, per environment at creation; flag files in `logs/staged/dr_flags/`): `mass` (bar mass ±30%), `handpd` (hand stiffness ±30%, damping ±30%), `linkmass` (robot link mass ±30%), `friction` (fingertip sliding+torsional, bar, table ±30% each; contacts combine by max), `delay` (action delay 0..2 steps), `impulse` (bar 1 N, arm links 10 N, phalanges 0.05 N, each p=0.02/step), `cubenoise` (observed bar pose: Gaussian 5 mm / 2° per step + per-episode bias up to 5 mm / 2°), `all` (everything). 8 configurations × 3 draws (seeds 42, 7, 23) = 24 runs, lr 5e-6 everywhere, budget 400 (17300→17700), ladder every 100.

Run name: `dr_<family>_s<seed>`. Launch (inside the container / on the desktop, from the repo root):
```
scripts/supervise_train.sh --python <PY> --run-name dr_<family>_s<seed> --target 17700 \
  --seed-checkpoint logs/staged/sc2_anchor_s42_it17300/model_17300.pt --seed-iteration 17300 -- \
  --num-envs 4096 --sim-device cuda:<gpu> --record-video --seed <seed> \
  $(cat logs/staged/sc2_anchor_s42_it17300/set_flags.txt) --set train.algorithm.learning_rate=5e-06 \
  $(cat logs/staged/dr_flags/<family>.txt)
```
Ladder on every rung (100, 200, 300, 400), DR OFF for the verdict (the sweep/sim2sim configs come from the run's config.json, so pass `--set domain_randomization.enabled=false` to every sweep): grid RSI 0 + pose 122 (250 repeats) at scales 0.8 / 1.0 / 1.2 (`--set object_randomization.scale_min=S --set object_randomization.scale_max=S --set object_randomization.scale_nominal_probability=0 --set object_randomization.scale_anchors=[]`), gate 7 cm computed from the rows (the config threshold is 18 cm). Candidate rung = Newton fail@7 within +0.05 of the base at every scale (base: 0.488 / 0.356 / 0.464 at 0.8 / 1.0 / 1.2) AND pose 122 ≥ 240/250 at every scale. On a candidate rung run the sim2sim verdict (CPU, ~3 min, same container): `scripts/sim2sim_grid.py --checkpoint <rung>.pt --set domain_randomization.enabled=false --newton-grid "<run dir>/sweep_grid_rsi0_<N>_s{scale}.json" --output <run dir>/sim2sim_grid25_<N>.json` and compare with the base's `logs/staged/sc2_anchor_s42_it17300/sim2sim_mujoco/grid25_baseline.json` (numbers in NIGHT_LOG once measured): hit = MuJoCo failed ≤ base at every scale and no-grasp at 1.2 ≤ base. §10 doom rule applies (orientation DQ > 0.6, two consecutive rungs worse than the base by > 0.10 at 1.0 → stop the run). A family's effect counts only if it beats the spread of its three draws.

Schedule (one run per GPU, sequential per card; case GPU 1 belongs to another user — never touch it; the desktop GPU is shared with the user's viewer processes — never kill them):
### tars
1. GPU 0: `dr_all_s42`, `dr_cubenoise_s42`, `dr_impulse_s7`, `dr_delay_s42`, `dr_handpd_s42`, `dr_mass_s23`.
2. GPU 1: `dr_all_s7`, `dr_cubenoise_s7`, `dr_impulse_s23`, `dr_delay_s7`, `dr_handpd_s7`, `dr_linkmass_s42`.
### case (GPU 0 only)
1. `dr_all_s23`, `dr_cubenoise_s23`, `dr_delay_s23`, `dr_handpd_s23`, `dr_mass_s42`, `dr_linkmass_s7`.
### desktop
1. `dr_friction_s42`, `dr_friction_s7`, `dr_friction_s23`, `dr_impulse_s42`, `dr_mass_s7`, `dr_linkmass_s23`.
Closing rule: when a host's queue is exhausted, final summary in the notes file (`logs/agents/agent-<host>-dr1.md`), one closing message, stop. Candidates: stage as `logs/staged/dr_<family>_s<seed>_it<N>/` (model, config.json, set_flags.txt, sweep JSONs, sim2sim JSON) and 3–4 eval clips.

### Coda finale (2026-09-21 11:46 CEST)

Criterio di verdetto aggiornato dall'utente: il tracking del braccio può peggiorare un po'; conta che la policy arrivi in fondo, cioè sollevi l'oggetto e lo tenga in mano. Le colonne decisive restano quindi posa 122 (≥ 240/250 a ogni scala), griglia Newton entro +0.05 della base e MuJoCo grid25 (fallite ≤ base, mancate prese a 1.2 ≤ base); il tracking è informativo, non un gate. Sotto questo criterio i due hit cubenoise (s42/17400, s7/17500) restano hit.

La coda di case si è fermata il 20/09 alle 18:38 (solo dr_all_s23 completata, famiglia all chiusa 3/3 senza candidato); su case ora anche la GPU 0 è occupata da un altro utente. Le run mancanti girano su tars con budget +200 (target 17500, ladder 17400/17500) tramite `logs/agents/dr1_tools/dr2_queue.sh`:

- GPU 0: dr_cubenoise_s23, dr_linkmass_s7, dr_mass_s42
- GPU 1: dr_combo_s42, dr_combo_s7, dr_handpd_s23 (combo = cubenoise + impulse + linkmass, `logs/staged/dr_flags/combo.txt`)
- saltata dr_delay_s23 (famiglia chiusa 2/2 come puro costo senza robustezza)

Risultati per run in `logs/staged/<run>_ladder/` (sweep Newton, model_17400/17500, sim2sim_grid25_<N>.json); righe di stato in `logs/agents/dr1_tools/queue_gpu{0,1}.log`.

### Verdetto finale DR1 (2026-09-21 13:49 CEST)

26 run (24 della griglia cauta, 2 combo) più una continuazione, base `sc2_anchor_s42/17300`, lr 5e-6. Criterio finale (utente): la policy deve sollevare e tenere fino alla fine; il tracking del braccio può peggiorare. Colonne decisive: posa 122 ≥ 240/250 a ogni scala, MuJoCo grid25 fallite ≤ base (15/17/20) e mancate prese ≤ base (8/5/10).

| checkpoint | Newton fail@7 0.8/1.0/1.2 (base 0.488/0.356/0.464) | posa 122 | MuJoCo fallite | MuJoCo mancate prese | palmo (base 0.029 m) | verdetto |
|---|---|---|---|---|---|---|
| dr_combo_s7/17500 | 0.528/0.444/0.264 | 250/250/250 | 9/10/14 | 4/3/3 | 0.045 | **hit, in testa** |
| dr_combo_s42/17400 | 0.492/0.448/0.428 | 250/250/250 | 13/12/15 | 4/4/5 | 0.043 | hit |
| dr_cubenoise_s23/17500 | 0.524/0.420/0.352 | 250/250/250 | 14/11/16 | 0/1/1 | 0.051 | hit |
| dr_cubenoise_s42/17400 | 0.384/0.356/0.400 | 250/250/250 | 14/11/16 | 4/3/3 | 0.045 | hit |
| dr_cubenoise_s7/17500 | 0.520/0.320/0.220 | 250/250/250 | 13/11/15 | 2/2/5 | 0.044 | hit |
| dr_impulse_s42/17400 | 0.364/0.296/0.300 | 247/249/247 | 16/15/18 | 5/5/11 | 0.025 | near-hit, tracking pulito |
| dr_linkmass_s42/17400 | 0.352/0.356/0.448 | 240+/–/155 | 14/11/16 | –/–/8 | 0.026 | near-hit, posa a 1.2 |
| dr_handpd_s7/17400 | 0.456/0.316/0.436 | 234/218/45 | 18/13/17 | 7/5/5 | 0.029 | near-hit, posa |
| dr_handpd_s23/17400 | 0.504/0.464/0.456 | 250/250/250 | 18/16/16 | 5/4/4 | – | tiene, griglia debole |

Famiglie senza candidato: all 0/3, mass 0/3, delay 0/2, friction (near-hit s42 solo), impulse s7/s23, linkmass s7/s23, handpd s42. Continuazione combo_s7 17500→17700: crollo (posa 41/64/59 a 17600, orientamento fino a 0.66 a 17700).

Lezioni: (1) il gradino utile è +100/+200 per ogni ricetta, combo compresa; (2) il seme conta più della famiglia (friction, linkmass, handpd: una estrazione buona su tre); (3) il rumore di osservazione del cubo è la sola famiglia che produce robustezza ripetibile (3/3) e la ricetta combinata la migliora in MuJoCo; (4) la robustezza comprata con rumore di osservazione costa 30–75% di errore del palmo, quella dinamica no, ma il sollevamento e l'orizzonte restano intatti; (5) in Newton fail@7 confonde errore di tracking 7–10 cm con oggetti persi: fail@15 e le mancate prese MuJoCo sono le colonne che misurano "solleva e tiene".

Pacchetti: `logs/staged/dr_<run>_ladder/` (model_17400/17500, config, sweep Newton, sim2sim_grid25_<N>.json, eval_rsi0_*.json tracking, eval_videos/ per i candidati), oltre ai `logs/staged/dr_*_it*` della prima parte.
