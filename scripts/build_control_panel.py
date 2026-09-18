#!/usr/bin/env python3
"""Build the control panel page: every GPU on every host, what it trains, how far.

    python3 scripts/build_control_panel.py            # -> logs/control_panel/index.html
    python3 scripts/build_control_panel.py --hosts local tars

Collects with scripts/collect_gpu_status.py (locally, and over ssh on the
servers where the same script runs from the code mirror), appends the agents'
notes and the campaign's wave sheet, and renders scripts/control_panel_template.html
with the data inlined. Standard library only; the page is then published as
an Artifact by the coordinator.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HOSTS = {
    "local": {"label": "desktop", "hostname": "jin-crl-desktop", "gpus": 1},
    "tars": {"label": "tars", "hostname": "tars.inf.ethz.ch", "gpus": 2},
    "case": {"label": "case", "hostname": "case.inf.ethz.ch", "gpus": 2},
    "ur5": {"label": "ur5", "hostname": "ur5", "gpus": 1},
}
REMOTE_ROOT = "simtoolreal_newton"

# The campaign sheet: what each run is asking, and what it must beat. Kept
# here rather than derived from configs so the panel says *why* a run exists.
WAVE = {
    "title": "WAVE S1 (branch generalize_size, from 2026-09-18 11:00): cuboid scale in [0.8, 1.2] with the deliverable recipe, warm start from w6_ori_soft_s7_cont2 @17000 (widened to observe the scale). Baseline zero-shot bank fail@7 from frame 0: 0.8 -> 0.748, 0.9 -> 0.492, 1.0 -> 0.396, 1.1 -> 0.348, 1.2 -> 0.376. Earlier: pose campaign closed 10:35 — deliverable w6_ori_soft_s7_cont2 @17000 (user's choice, best arm tracking by eye)",
    "baseline_label": "s2s_bankbox (desktop, all bank poses) eval @1500",
    "baseline": {
        "early_term": 0.719, "lift": 0.006, "rms_ee": 0.065,
        "ori_err": 0.086, "rms_hand": 0.233, "rms_pos": 0.215,
    },
    "sweep_baseline": "DELIVERABLE chosen by the user (2026-09-18 10:45, best arm tracking from frame 0 by eye): w6_ori_soft_s7_cont2 @17000 (desktop): pose 122 250/250 err 0.025, RSI-0 bank fail@7 0.40 / fail@10 0.09 / err 0.063; w6_ref_ori_lowlr @14600 (reference rung + orientation 1.2/0.6 + lr 2e-5): RSI-0 grid fail 0.172, 98% within 15 cm, err 0.060; @15200 pose 122 250/250 err 0.031, RSI-760 fail 0.168 none beyond 15 cm. Runner-up: w5_ref_lowlr @18000 — RSI-760 bank: all 250 within 15 cm, err 0.076, carries to frame 1072; pose 122 all within 10 cm; RSI-0 bank fail 0.69 / err 0.080 (monotone ladder, continuing). SHARPEST on the anchor: w5_mix_ori_soft @12500 solves pose 122 from frame 0 (250/250, err 0.066) with bank RSI-0 fail 0.69 / err 0.079. REFERENCE on the bank: w4_bank_mix_s7 @14000 (12.5% anchoring) RSI-0 grid fail 0.552 (112/250), med err 0.075, RSI-760 fail 0.304 with none beyond 15 cm, pose 122 250/250. Seed 42 twin: w4_bank_mix @12000 (12.5% anchoring) RSI-0 grid fail 0.594, med max err 0.075, pose 122 kept 250/250. From the pregrasp: w3_bank_holdrew @11000 grid RSI 760 fail 0.43 (57% pass), lift 0.26, orientation 0.35; from RSI 0: w3_bank_holdrew @11000 fail 0.61 / w3_bank_from_one @10000 fail 0.68. Earlier: w3_carry_holdrew @5500 sweep_grid RSI 760 fail 0.60 (100/250), lift 0.265, orientation 0.41, contact 0.92 (then decayed). SOLVED POSE 122: w2_one_cont @8000 — 0% fail at 7 cm from RSI 0 over the whole clip (250 roll-outs), lift 0.27 m, orientation 0.18 rad, rms_ee 0.013; at RSI 760 0% fail, orientation 0.375. Earlier best on all poses: w2_carry_objpose @5400 (ALL bank poses, RSI 760 grid): 84% fail at 7 cm (39/250 pass, first passes of the campaign), median max object error 0.084 m, lift 0.26 m, orientation 0.45 rad, contact 0.90, rms_ee 0.015; decays after 5500. Earlier: w1_one_graspfocus @2300 (pose 122, RSI 760): 70% fail at 7 cm but 30% complete the clip, median lift 0.23 m, orientation 0.72 rad, contact 0.64, rms_ee 0.023; collapses after 2400. w1_one_base @4000 (pose 122): sweep_one RSI 760 100% fail at 7 cm but only at index 1020, contact 0.93, median lift 0.34 m, orientation 0.54 rad; from RSI 0 it nudges the bar at 740. s2s_bankbox @4000 (all poses): sweep_one RSI 760 100% fail at 7 cm, contact 0.86, median lift 0.44 m (demo 0.25), orientation 0.77 rad, median fail index 975 (carry) — the grasp is there, the carry is not. Success for wave 1: eval lift >= 0.05 m and sweep_one (RSI 760, 7 cm) fail < 50% with median lift >= 0.15 m",
    "runs": {
        "s1_scale_u82_s23": {"change": "WAVE S1 (tars 0, after the ladders): main recipe [0.8, 1.2] observed, seed 23, dense 100-spaced early ladder", "question": "resume success rate over seeds (7 dived and recovered, 42 kept the anchor)"},
        "s1_scale_u82_s11": {"change": "WAVE S1 (tars 1, after the ladders): main recipe [0.8, 1.2] observed, seed 11", "question": "fourth seed of the main arm"},
        "s1_scale_u82_s3": {"change": "WAVE S1 (desktop): main recipe [0.8, 1.2] observed, seed 3", "question": "fifth seed of the main arm"},
        "s1_scale_curr_s7_wide": {"change": "WAVE S1 (case 0): stage 2 of the curriculum — curr_s7 model_18000 continued 2000 at [0.8, 1.2]", "question": "does widening from the narrow-range best rung hold the anchor better than the full range from 17000?"},
        "s1_scale_curr_s42_wide": {"change": "WAVE S1 (case 1): stage 2 of the curriculum from curr_s42's best rung, 2000 at [0.8, 1.2]", "question": "same on the seed that keeps the anchor"},
        "s1_scale_u82_s7": {"change": "WAVE S1 (tars 0): deliverable recipe + bar scale uniform [0.8, 1.2] with the factor observed, lr 2e-5, seed 7, 17000->21000", "question": "does the unchanged recipe learn the size when the factor is observed?"},
        "s1_scale_u82_s42": {"change": "WAVE S1 (tars 1): same as s1_scale_u82_s7 with seed 42", "question": "seed replicate of the main arm"},
        "s1_scale_curr_s7": {"change": "WAVE S1 (case 0): scale [0.9, 1.1] first (17000->19000), then best rung -> +2000 at [0.8, 1.2] (s1_scale_curr_s7_wide)", "question": "does narrow-then-wide keep the anchor better than the full range at once?"},
        "s1_scale_noobs_s7": {"change": "WAVE S1 (case 1): scale [0.8, 1.2] WITHOUT the scale observation (plain 112-input seed)", "question": "STOPPED by §10 at 18600 (near-band 12 -> 3 -> 0 -> 0): NO — without the scale observation the policy holds the bar but never re-anchors"},
        "s1_scale_curr_s42": {"change": "WAVE S1 (case 1): curriculum [0.9, 1.1] then [0.8, 1.2] with seed 42, dense 100-spaced early ladder", "question": "the curriculum on the seed that keeps the anchor at the resume"},
        "s1_scale_u73_s7": {"change": "WAVE S1 (desktop): scale [0.7, 1.3] with the factor observed", "question": "STOPPED by §10 at 18000 (near-band 87 -> 4 -> 0/100, bar thrown); no rung beat the baseline"},
        "s1_scale_u73_s42": {"change": "WAVE S1 (desktop): scale [0.7, 1.3] with the factor observed, seed 42", "question": "same seed as the tars arm that beats the baseline at 0.8: does the wider range help or hurt?"},
        "smoke_scale_resume": {"change": "3-iteration smoke of the warm start from the widened checkpoint (113 observations)", "question": "resume check only"},
        "s2s_bankbox": {"change": "baseline recipe on all 1536 bank poses (RSI mixture 740-798, 20% from frame 0)", "question": "reference trajectory of the recipe before the single-pose curriculum"},
        "w1_one_base": {"change": "baseline recipe, bank pose 122 only", "question": "does the recipe grasp when the pose never changes?"},
        "w1_one_hand48_objsig10": {"change": "pose 122 + position_hand_weight 0.48 + object_position_std_m 0.10", "question": "do the Isaac Gym campaign's two reward findings transfer to this simulator?"},
        "w1_one_graspfocus": {"change": "pose 122 + rsi_early_probability 0.0 (every episode starts in the pregrasp window 740-798)", "question": "all samples on the grasp: does a lift appear?"},
        "w1_one_hand48": {"change": "pose 122 + position_hand_weight 0.48 only (ur5, 2048 envs x 48 steps)", "question": "is the hand weight alone what helps, or is the object sigma needed too?"},
        "w2_carry_objpose": {"change": "WAVE 2 (desktop): s2s_bankbox continued from model_4000, all bank poses, object_position sigma 0.10 / weight 1.5, orientation sigma 0.4 / weight 1.5, object termination 0.12, contact on, eval phase at frame 760", "question": "does pricing the object pose turn the reliable grasp (100% contact, +20 cm over-lift, 45 deg twist) into a demo-like carry?"},
        "w2_one_cont_ori": {"change": "WAVE 2 (tars 0): w1_one_base continued 4000->8000, orientation weight 1.5 / sigma 0.4, contact measured", "question": "STOPPED ~5100: orientation priced alone makes the untouched bar the cheapest solution (grasp lost)"},
        "w2_one_cont_early50": {"change": "WAVE 2 (tars 0): w1_one_base continued 4000->8000 with rsi_early_probability 0.5 (half the episodes from frame 0)", "question": "NO: 100% fail, bar tumbled (1.6 rad) — halving the pregrasp starts costs the grasp; the control solved RSI 0 anyway"},
        "w3_carry_std01": {"change": "WAVE 3 (desktop): w2_carry_objpose continued from its peak model_5400 with min_action_std 0.3 -> 0.1, all bank poses", "question": "NO: collapsed again at ~7100 with std 0.289 — noise floor refuted"},
        "w3_bank_from_one": {"change": "WAVE 3 (tars 0): w2_one_cont model_8000 (solves pose 122) continued on the WHOLE bank, 8000->12000", "question": "PARTLY: RSI-0 grid fail 0.956 -> 0.684 at model_10000 (79/250 pass), then regressed to 0.867 at 12000; pose 122 forgotten — the competent region moved (y 0.03-0.15) instead of growing"},
        "w3_one_holdrew": {"change": "WAVE 3 (tars 1): w2_one_cont continued from its best sweep checkpoint with the fingertip contact reward on (0.15/finger)", "question": "NO: from 0% fail / contact 0.79 to 100% fail / contact 0.13 — the hold bonus destroyed the skill"},
        "w3_carry_measured": {"change": "WAVE 3 (ur5): w2_carry_objpose continued from model_5400 with the palm keypoint reward anchored to the measured bar (all poses, 2048x48)", "question": "best rung 6800: grid RSI 760 fail 0.896, then collapse 7100-7700 — no better than the desktop lineage; retired"},
        "w3_band_y05": {"change": "WAVE 3 (case 1): w2_one_cont model_8000 continued on the 556 bank poses with y <= 0.05 (translation curriculum), 8000->12000", "question": "A/B vs w3_bank_from_one: does a translation curriculum widen faster than the whole bank at once?"},
        "w3_bank_holdrew": {"change": "WAVE 3 (case 0): w2_one_cont model_8000 continued on the whole bank WITH the fingertip hold reward (0.15/finger), 8000->12000", "question": "grid RSI 0 fail 0.606 at model_11000 (best 11000-11200), pose 122 partly kept (lift 0.20, err 0.091) — better than uniform alone, but the region still drifts"},
        "w3_carry_holdrew": {"change": "WAVE 3 (desktop): w2_carry_objpose model_5400 continued on all poses WITH the hold reward (0.15/finger)", "question": "NO: decayed even sooner (contact 0.69 -> 0.03 by 6100); objpose lineage retired"},
        "w3_bank_measured": {"change": "WAVE 3 (desktop): w2_one_cont model_8000 continued on the whole bank with the palm reward anchored to the measured bar", "question": "no decay, but RSI-0 grid fail@7 1.0 at every rung; best 11500 median max err 0.143 (uniform seeds: 0.083 / 0.165)"},
        "w3_bank_from_one_s7": {"change": "WAVE 3 (tars 1): w3_bank_from_one repeated with seed 7", "question": "noise floor of the widening: how much of the variants' difference is seed?"},
        "w4_bank_cont": {"change": "WAVE 4 (tars 0, killed at start): continuation from model_12000", "question": "superseded: the run had peaked at 10000"},
        "w4_bank_mix": {"change": "WAVE 4 (tars 0): w3_bank_from_one model_10000 continued 10000->14000 on an ANCHORED pose list (all poses + near band twice + pose 122 x300)", "question": "YES: model_12000 RSI-0 grid fail 0.594 (101/249, best of the campaign), med max err 0.075; pose 122 250/250; region grew (y 0.00-0.15) instead of migrating; oscillation and orientation 0.6 rad remain"},
        "w4_band_y10": {"change": "WAVE 4 (case 1): w3_band_y05 continued 12000->16000 on the 1035 poses with y <= 0.10 (curriculum stage 2)", "question": "killed at start: its seed (band_y05 @12000) had forgotten pose 122; replaced by the anchoring A/B"},
        "w4_bank_mix25": {"change": "WAVE 4 (case 1): w3_bank_from_one model_10000 continued 10000->14000 on the STRONGLY anchored list (pose 122 = 25% of resets)", "question": "best rung 10500 (500 past the seed): RSI-0 grid f@7 0.75, med max err 0.100, orientation 0.31; no gain over the next 3500 (oscillates); same level as the lucky uniform seed (0.083) without the seed lottery"},
        "w4_bank_mix_from8000": {"change": "WAVE 4 (ur5): the origin policy w2_one_cont model_8000 continued on the anchored list (12.5%), 8000->12000, 2048x48", "question": "is anchoring from the origin better than anchoring after the migration (tars 0 / case 1 from model_10000)?"},
        "w4_bank_mix25_s7": {"change": "WAVE 4 (case 0): w4_bank_mix25 repeated with seed 7", "question": "REPLICATED: same best rung (10500), med max err 0.109 vs 0.100 — the anchored widening is reproducible (uniform seeds differed 3x)"},
        "w4_bank_mix25_hold": {"change": "WAVE 4 (desktop): model_10000 on the strongly anchored list (25%) plus the hold reward (0.15/finger), 10000->14000", "question": "NO from RSI 0 (med err 0.159 vs 0.100 anchor-only); from the pregrasp the best desktop sweep (all poses within 15 cm, med err 0.084) at 10500; orientation erodes after 11000; pose 122 from frame 0 still lost"},
        "w4_bank_mix_s7": {"change": "WAVE 4 (tars 1): w4_bank_mix (12.5% anchoring) repeated with seed 7", "question": "REFERENCE POLICY: model_14000 RSI-0 grid fail 0.552 (112/250), med err 0.075; RSI-760 grid fail 0.304, none beyond 15 cm; pose 122 250/250 — anchoring reproduces across seeds"},
        "w5_mix25_objpose": {"change": "WAVE 5 (case 1): w4_bank_mix25 model_10500 continued with the object-pose set (position 1.5/0.10, orientation 1.5/0.4, threshold 0.12) on the 25% anchored list", "question": "NO: med err 0.100 -> 0.087 at model_13000, fail@7 worse (0.86); the 10 cm wall does not yield to reweighting"},
        "w5_mix_lowlr": {"change": "WAVE 5 (tars 0): w4_bank_mix (12.5%) continued from its best rung with learning rate 5e-5 -> 2e-5", "question": "plateau then drift: no catastrophic rungs 12500-14000, monotone decay after; best 12500 f@7 0.548 / err 0.073 = within the seed spread"},
        "w5_mix_ori_soft": {"change": "WAVE 5 (case 0): w4_bank_mix model_12000 (best whole-clip checkpoint) continued with a moderate orientation nudge (1.2/0.6), position term unchanged", "question": "YES, partly: model_14000 med err 0.082, 93% within 15 cm, orientation 0.46, carries to frame 1032; degenerates into the stillness exploit by 16000"},
        "w5_mix_ori_soft_s7": {"change": "WAVE 5 (desktop): w5_mix_ori_soft repeated with seed 7", "question": "model_13000: RSI-0 grid f@7 0.78, 93% within 15 cm, err 0.083, orientation 0.39 — reproduces the nudge (orientation gain) on a second seed"},
        "w5_mix_ori_lowlr": {"change": "WAVE 5 (tars 1): w4_bank_mix model_12000 continued with the orientation nudge (1.2/0.6) AND lr 2e-5", "question": "NO: bank generalisation halved (13/248 vs 112/250), orientation worse; one knife-edge rung (13000) solves pose 122 (236/250)"},
        "w5_ref_lowlr": {"change": "WAVE 5 (case 1): the reference policy w4_bank_mix_s7 model_14000 continued with lr 2e-5, 14000->18000", "question": "monotone improvement to the last rung (err 0.149 -> 0.080), 5 mm short of the reference; continued 4000 more"},
        "w4_bank_mix_s11": {"change": "WAVE 5 (tars 0): w4_bank_mix (12.5% anchoring) third seed (11)", "question": "COLLAPSED at ~11500 (grasp lost): third distinct outcome of the same recipe — run-level collapse is still a seed lottery"},
        "w6_ori_soft_cont_lowlr": {"change": "WAVE 6 (case 0): w5_mix_ori_soft model_12500 (solves pose 122 + bank 0.69) continued 2000 iterations at lr 2e-5", "question": "NO on the bank: training metrics consolidated (early_term 0.031) but bank sweep worse than the parent (0.80/0.085 vs 0.69/0.079); orientation line closed on case"},
        "w6_ref_ori_lowlr": {"change": "WAVE 6 (tars 0): the reference w4_bank_mix_s7 model_14000 continued 2000 iterations with the orientation nudge (1.2/0.6) at lr 2e-5", "question": "CAMPAIGN BEST: model_14600 RSI-0 grid fail 0.172 (207/250), 98% within 15 cm; 15200 pose 122 250/250 err 0.031; RSI-760 fail 0.168, none beyond 15 cm — success criterion met"},
        "w6_ori_soft_s7_cont_lowlr": {"change": "WAVE 6 (desktop): w5_mix_ori_soft_s7 model_13000 continued 2000 iterations at lr 2e-5", "question": "best rung = last (15000): err 0.089, orientation 0.323 (best of the campaign), still improving; continued"},
        "w6_ori_lowlr_cont": {"change": "WAVE 6 (tars 1): w5_mix_ori_lowlr best rung continued unchanged 2000 iterations at lr 2e-5", "question": "NO: ragged ladder, pose-122 win evaporated (max 70/250), bank 0.72 — the weak lineage stays weak; near-band retention of the start rung predicts receptivity"},
        "w6_ori_soft_cont2": {"change": "WAVE 6 (case 0, branch A): the consolidated rung continued 2000 more at lr 2e-5", "question": "does the plateau keep improving?"},
        "w4_bank_mix_s23": {"change": "WAVE 6 (case 0, branch B): fourth anchored seed (23)", "question": "10500: fail@7 0.684 / err 0.093 — four-seed anchored baseline 0.594 / 0.552 / collapse / 0.684; best rung always 500 past the seed at lr 5e-5"},
        "w6_ref_lowlr_cont": {"change": "WAVE 6 (case 1): w5_ref_lowlr model_18000 continued unchanged 4000 more at lr 2e-5 (18000->22000)", "question": "NO: best rung = first (18500, err 0.069); near-band collapsed 42 -> 0/100 between 21000 and 21100 (mid-run collapse, evaluator blind)"},
        "w6_ref_lowlr_s42": {"change": "WAVE 6 (tars 0): the reference w4_bank_mix_s7 model_14000 continued at lr 2e-5 with seed 42 (case runs seed 7)", "question": "SEED: this seed degrades monotonically (0.63 -> 1.00) while its evaluator series looked perfect; near-band retention collapsed at the first rung"},
        "w6_ref_ori_lowlr_s42": {"change": "WAVE 6 (tars 1): the campaign-best recipe (reference rung + orientation 1.2/0.6 + lr 2e-5) repeated with seed 42", "question": "PEAK REPRODUCES (14400: fail@7 0.275, pose 122 236/250), stability does not (2 usable rungs vs 6)"},
        "w6_ori_soft_s7_cont2": {"change": "WAVE 6 (desktop): model_15000 continued 2000 more at lr 2e-5", "question": "YES: model_17000 (last rung, plateau) RSI-0 grid fail@7 0.396, fail@10 0.088, err 0.063 — second best of the campaign; continued 4000 in one budget"},
        "w7_ori_soft_s7_long": {"change": "WAVE 7 (desktop, branch A): the best rung continued 4000 iterations in one go at lr 2e-5", "question": "NO: stopped by §10 at 18500 (near-band 87 -> 41 -> 0/100); the deliverable stays model_17000"},
        "w6_ref_ori_lowlr_s11": {"change": "WAVE 6 (desktop, branch B): the campaign-best recipe with seed 11", "question": "third seed for the best recipe"},
        "w7_ref_ori_long": {"change": "WAVE 7 (tars 1): the campaign-best rung (w6_ref_ori_lowlr 15200) continued 4000 in one budget at lr 2e-5, with the adaptive-width fix", "question": "STOPPED by the §10 check at +500: near-band 82 -> 0/98 (continuations from a peak rung are knife-edges)"},
        "w7_ref_lowlr_long": {"change": "WAVE 7 (case 1): w6_ref_lowlr_cont 18500 + orientation nudge 1.2/0.6 at lr 2e-5, seed 42 (w7_reflowlr_ori_s42)", "question": "STOPPED by §10 mid-run (near-band 42 -> 39/100 at +500 -> 0/100 at +2000, med max err 0.654): the nudge on a drifted substrate throws the bar"},
        "w7_reflowlr_ori": {"change": "WAVE 7 (case 0): the monotone lineage (w5_ref_lowlr 18000 or better) + orientation nudge 1.2/0.6 at lr 2e-5, 4000 iterations", "question": "STOPPED by §10 mid-run (near-band 42 -> 17/100 at +500 -> 1/100 at +2000): no — the nudge only works on a fresh, well-anchored substrate"},
        "w7_lottery_s3": {"change": "WAVE 7 (case 0): seed lottery of the campaign-best recipe — w4_bank_mix_s7 14000 + orientation nudge 1.2/0.6, lr 2e-5, seeds 3, 5, 13, 17 in turn, 200-iteration tickets (stop the ticket if near-band < 30/100 at +200), the first survivor gets the full 200-spaced ladder", "question": "CLOSED after two tickets (s3 2/98, s5 9/100 at +200, both killed): full nudge = 2 good seeds of 6 (s7, s42) — a third; superseded by the half-nudge question"},
        "w7_ref_ori_anchor25_s11": {"change": "WAVE 7 (case 1): w4_bank_mix_s7 14000 + orientation nudge 1.2/0.6 at lr 2e-5, seed 11 (a failed seed), with the 25% anchored list (mix122_strong), §10 at +200/+400", "question": "YES: near-band 0 -> 1 -> 40/100 at +200/+400/+600, fail@15 0.000, err 0.083 — the 25% anchor rescues the seed that died twice with mix122; running to 16000"},
        "w7_reflowlr_ori_s42": {"change": "WAVE 7 (tars 0): the seed-42 monotone lineage's best rung + orientation nudge 1.2/0.6 at lr 2e-5, 4000 iterations", "question": "does the nudge on a strong small-step rung reproduce the campaign-best jump on the other lineage?"},
        "w6_ref_ori_lowlr_s11": {"change": "WAVE 6 (tars 1): the campaign-best recipe with seed 11", "question": "STOPPED by §10 at +500 (near-band 64 -> 0/93): seed 11 dies on both recipes it was tried on"},
        "w8_hand_nudge": {"change": "WAVE 8 (desktop): the deliverable w6_cont2 17000 continued 2000 at lr 2e-5 with position_hand_weight 0.3 -> 0.45", "question": "STOPPED by §10 at +500: rms_hand 0.25 -> 0.152 (first time it ever moved) but near-band 87 -> 0/100 — 0.45 is too strong even on a solved policy"},
        "w8_best_anchor25_lr1e5": {"change": "WAVE 8 (desktop): campaign-best rung w6_ref_ori_lowlr 14600 continued 2000 at lr 1e-5 with the 25% anchored list (mix122_strong), seed 7, no reward change", "question": "NO — STOPPED by §10 at +500 (near-band 74 -> 0/100, 1/250, med max err 0.21, 7 blown): with no reward change at all, continuation itself walks off the peak; lr and anchor strength are not levers"},
        "w8_best_early40": {"change": "WAVE 8 (desktop, if the RSI-760 diagnostic shows the grasp survives at 15100): w6_ref_ori_lowlr 14600 continued at lr 2e-5 with rsi_early_probability 0.2 -> 0.4 (twice the frame-0 starts)", "question": "is the collapse forgetting of the approach phase (RSI-0 dies while RSI-760 holds)? then more frame-0 starts should protect it"},
        "w8_best_std04": {"change": "WAVE 8 (desktop, if the grasp collapses too): w6_ref_ori_lowlr 14600 continued at lr 2e-5 with min_action_std 0.3 -> 0.4", "question": "is the peak a brittle narrow optimum that a wider exploration floor keeps the policy from leaving?"},
        "w6_ref_ori_lowlr_s23": {"change": "WAVE 6 (tars 1): the campaign-best recipe with seed 23", "question": "STOPPED by §10 at +500 (near-band 0/100, 8/250): the nudge recipe works on 2 seeds of 4 — not a seed-11 pathology"},
        "w7_ref_ori_lr1e5_s11": {"change": "WAVE 7 (tars 1): campaign-best recipe (w4_bank_mix_s7 14000 + orientation nudge 1.2/0.6) at lr 1e-5 instead of 2e-5, seed 11 (a failed seed), 2000 iterations, §10 at +200/+400", "question": "STOPPED by §10 at +400 (near-band 0/91, 1/100): no — the lr is not the lever, seed 11 dies the same way at 1e-5"},
        "w7_ref_halfnudge_s11": {"change": "WAVE 7 (tars 1): w4_bank_mix_s7 14000 + a half orientation nudge (1.0/0.7 instead of 1.2/0.6) at lr 2e-5, seed 11, §10 at +200/+400", "question": "YES so far: seed 11 (dead with the full nudge at both lrs) anchors with the half nudge — near-band 0 -> 9 -> 18/99 at +200/+400/+600, orientation 0.69 -> 0.45; running"},
        "w7_ref_halfnudge_s7": {"change": "WAVE 7 (case 0, after the current lottery ticket): w4_bank_mix_s7 14000 + half orientation nudge 1.0/0.7 at lr 2e-5, seed 7 (the seed that reached 0.172 with the full nudge)", "question": "does the gentler nudge reach the same peak as the full one on a good seed, or does it trade reliability for ceiling?"},
        "w2_one_cont": {"change": "WAVE 2 (tars 1): w1_one_base continued 4000->8000 unchanged (control)", "question": "how much comes from more iterations alone?"},
        "w2_one_gf_ori": {"change": "WAVE 2 (case 0): graspfocus (rsi_early 0) from scratch + orientation weight 1.5 / sigma 0.4, eval phases 760/831, contact measured", "question": "STOPPED 2009: holds the bar perfectly still (contact 0.80, lift 0.03) — orientation alone pays stillness"},
        "w2_one_gf_objpose": {"change": "WAVE 2 (case 0): graspfocus from scratch + full object-pose set (position 1.5/0.10, orientation 1.5/0.4, threshold 0.12), eval phases 760/831", "question": "peaked ~2500 (contact 0.84) then decayed like every pregrasp-only run; stopped ~3100, ladder pending"},
        "w2_one_gf_adapt": {"change": "WAVE 2 (case 1): graspfocus from scratch + KL-adaptive learning rate (schedule=adaptive), eval phases 760/831", "question": "is the collapse after the 2300 peak a step-size instability?"},
        "w1_one_contact": {"change": "pose 122 + fingertip contact bonus (contact.enabled, reward_enabled, 0.05 per finger x thumb/index/middle)", "question": "does paying for fingertip contact produce the grasp?"},
    },
}


def collect_local():
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "collect_gpu_status.py"), "--host", "local"],
        capture_output=True, text=True, timeout=60,
    )
    return json.loads(out.stdout)


def collect_remote(host):
    command = (
        "cd {} && python3 scripts/collect_gpu_status.py --host {}".format(REMOTE_ROOT, host)
    )
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOSTS[host]["hostname"], command],
        capture_output=True, text=True, timeout=90,
    )
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(out.stderr.strip()[-300:] or "no output")


def agent_notes():
    notes = {}
    directory = os.path.join(REPO, "logs", "agents")
    if not os.path.isdir(directory):
        return notes
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = [l.rstrip() for l in handle.read().splitlines() if l.strip()]
        notes[name[:-3]] = {
            "updated": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="minutes"),
            "lines": lines[-14:],
        }
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=list(HOSTS))
    parser.add_argument("--output", default=os.path.join(REPO, "logs", "control_panel", "index.html"))
    args = parser.parse_args()

    hosts = []
    for host in args.hosts:
        entry = {"key": host, "label": HOSTS[host]["label"], "hostname": HOSTS[host]["hostname"], "expected_gpus": HOSTS[host]["gpus"]}
        try:
            report = collect_local() if host == "local" else collect_remote(host)
            entry.update({"ok": True, "collected_at": report["collected_at"], "gpus": report["gpus"], "runs": report["runs"]})
        except Exception as error:  # noqa: BLE001 - the panel must render with a host down
            entry.update({"ok": False, "error": str(error)[:200], "gpus": [], "runs": []})
        hosts.append(entry)

    data = {
        "built_at": datetime.now().isoformat(timespec="minutes"),
        "hosts": hosts,
        "wave": WAVE,
        "agents": agent_notes(),
    }
    template_path = os.path.join(REPO, "scripts", "control_panel_template.html")
    with open(template_path, encoding="utf-8") as handle:
        page = handle.read()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("__PANEL_DATA__", blob)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(page)
    with open(os.path.join(os.path.dirname(args.output), "status.json"), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    running = sum(1 for h in hosts for r in h["runs"] if r["state"] == "running")
    print("{} — {} hosts, {} running runs -> {}".format(data["built_at"], sum(h["ok"] for h in hosts), running, args.output))


if __name__ == "__main__":
    main()
