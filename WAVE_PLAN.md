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
