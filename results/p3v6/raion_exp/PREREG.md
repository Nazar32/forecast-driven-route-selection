# Raion-level forecast experiments: pre-declared protocol (written 2026-10-03, before any val result was read)

## Windows
- val: train < 2025-12-01, isotonic Dec 2025, evaluation departures 2026-01-01 .. 2026-02-28 (raion regime).
- val2 (robustness of the top candidates): train < 2025-11-01, isotonic Nov 2025, evaluation Dec 2025.
- final: train < 2026-02-01, isotonic Feb 2026, evaluation 2026-03-01 .. 2026-09-06; used once.
- Disclosure: val and final lie inside the 2026 block on which the earlier (untuned) raion model was evaluated
  (aggregate results +0.7 % / +1.5 % were seen). Nothing was tuned on 2026 data before this protocol.

## Fixed settings (not tuned)
- Per-unit recalibration: daily, previous 42 days of matured targets, intercept by Newton (exact MLE of a logit
  offset), shrinkage d/(d+14) with d = number of days with matured targets, |offset| <= 1.5.
- Switching margin: relative, switch away from the static route only if
  E[static] - E[best] > eps * E[static], eps in {0, 0.02, 0.05, 0.1, 0.2}; the same eps applies to every strategy.
- Baselines: fixed static (fewest pre-eval km, as in the paper), rolling static (28 days, re-picked hourly from
  ended trips), Markov (full history, as in the paper), rolling Markov (daily refit, 42 days),
  rolling semi-Markov (daily refit, 60 days; pooled duration hazard + unit factor), reactive, oracle.
- Paired CIs: moving 3-day block bootstrap of differences of pooled reductions (same resampled blocks).

## Candidates (30): feats {base, inst, full} x calibration {raw, recal} x eps (5 values), raion truth.
No candidates are added after reading val results; any new idea is a new, separately declared round.

## Primary metric
Pooled reduction of km under alert vs the fixed static route, OSM primary, 78 OD pairs, val window.

## Selection rule
1. Guard: drop candidates whose share of ODs worse than -2 % exceeds that of rolling Markov (eps = 0) + 0.05.
2. c* = argmax primary among the rest.
3. One-SE simplicity: among candidates within the half-width of the paired 3-day CI of (c* - candidate), choose
   the simplest: base < inst < full, then raw < recal, then eps closest to the val-optimal eps of the reference
   baseline.
4. Tie-break: higher share of ODs better than static, then lower mean extra time.
5. Robustness: the selected candidate and c* are re-run on val2; if the selected candidate's sign vs the reference
   baseline differs between val and val2, fall back to the next simpler candidate that agrees on both.

## Reference baseline for the final claim
The one of {rolling semi-Markov, rolling Markov} with the higher val primary at the selected eps.

## Final success criterion
The paired 3-day-block 95 % CI of (selected - reference baseline) of the pooled OSM reduction on final excludes 0.
Also reported: vs fixed static, rolling static, oracle; corridors (report only); the oblast_any control
(gap pred - baseline under raion vs oblast_any truth, same features). A null or negative result is reported as such.

# Round 4 (declared 2026-10-03 after the round-2 val results of base/inst and the 2024 control were seen)

Finding that triggered it: on the paper's own 2024 test block (control, the paper's forecasts), a 42-day rolling
Markov gives 9.35 % and a 28-day rolling static 9.11 % vs 6.21 % for the paper's ML (paired CI of ML - rolling
Markov [-3.96, -2.33]). The critic's neighbour-pair diagnostic on 2024 (blend and hybrid) was seen before this
declaration; val was not looked at for blends.

ML source: ctrl2024 = paper forecasts (p3v5); val = the round-2 selected Q (rule above, once `full` is done).
Fixed: recalibration as in round 2 (started at T_CAL + 7 d where earlier P_recal is needed), markov_roll 42 d,
primary eps = 0 (eps = 0.05 also reported).

Candidates:
- C1 blend50: P = 0.5 P_recal + 0.5 P_markov_roll at every lead (lead 0 then max(act0, .)).
- C2 stack: per lead j, daily refit on departures t with t + j <= D - 1 h in the previous 42 days (not before
  T_CAL + 7 d): logistic y ~ a + b logit(P_recal) + c logit(P_markov_roll), pooled over units, ridge 1 toward
  (b, c) = (0.5, 0.5), logits clipped to +-9.
- C3 anchor: retrained ML without the categorical region; adds per unit (hours <= r) the rolling 42-day Markov
  pi, p11, p01, the 28- and 42-day rates, the 28-day rate at the target clock hour (per k) and the logit of the
  rolling-Markov forecast for model k; plus recalibration. Trained on val (raion) and on c2024 (oblast_any truth,
  train < 2024-06-01, calib Jun-Jul 2024, eval 2024-07-31 15:00 .. 12-31; training rows from one raion per
  oblast).
- C4 anchor+blend50.
- C5 c2024_ref (control only, not selectable): harness ML on c2024 with base+inst, no region, no anchors.

Selection: Delta_p(c) = pooled OSM reduction(c) - pooled(markov_roll), eps 0, p in {ctrl2024, val}, with
3-day-block paired CIs. Guard: drop c if its share of ODs worse than -2 % exceeds markov_roll's + 0.05 in either
period. Score S(c) = min(Delta_2024, Delta_val); c* = argmax S; one-SE simplicity with the 2024 paired-CI half-width,
order C1 < C2 < C4 < C3. Go to final only if Delta_2024 CI lower bound > 0, Delta_val > 0 and same sign on val2;
otherwise final is run for pred_recal vs markov_roll and reported as null.
Final: paired CI of (selected - markov_roll) on 2026-03..09, eps 0, must exclude 0. Reference fixed: markov_roll.
2024H2 is a development set from now on; no 2024 number is called held-out.

## Round 4 clarifications (written before anchor_val results were read)
- C3 configuration, identical in val, val2 and final: feats base+inst+anchor+no_region, all units
  (one_per_oblast was used only for the c2024 run, to save memory).
- hw = half-width of the 3-day-block paired 95 % CI of (c* - candidate) on ctrl2024 (from pooled.npz).
- Guard uses worse2 at eps 0 in both periods (ceilings: markov_roll share + 0.05).
- Val ML source for C1/C2: the round-2 selection with the one-SE step evaluated from cross-run paired CIs
  (pooled.npz); only its feature set is carried over (round 4 uses recal and eps 0).
- val2: only the selected candidate's sign vs markov_roll is checked (supersedes the round-2 "c* too").
- Known mismatch: on 2024 the C1/C2 ML source is the paper model, on val the harness model (reported as such).
- Each run records cfg and the code hash of p3_raion_exp.py in eval.json.

## Selection outcome (recorded before val2 and final were run)
- Round 2 (val, cross-run paired CIs): c* = full pred_recal@0.05 (1.45 %); one-SE simplest = inst pred@0.1 (0.85 %)
  -> val ML source for C1/C2 = inst.
- Round 4 (Delta vs markov_roll, 3-day paired CI):
  C1 blend50: 2024 +0.19 [-0.04, 0.43], val +0.73 [0.21, 1.28]
  C2 stack:   2024 +0.25 [0.03, 0.48],  val +0.77 [0.25, 1.34]
  C3 anchor:  2024 +0.73 [0.29, 1.20],  val -0.00 [-0.75, 0.80], guard failed on val (worse2 0.18 > 0.13)
  C4 anchor+blend50: 2024 +1.05 [0.75, 1.35], val +0.75 [0.18, 1.33]
  c* = C4 (S = 0.75); C1, C2 outside the one-SE band (paired CI of C4 - C1 [0.59, 1.11], C4 - C2 [0.56, 1.04]).
  SELECTED: C4. Go/no-go on 2024 and val: pass. Next: val2 sign check, then final.
- val2 (Dec 2025, anchor retrained train < 2025-11-01): C4 blend50 1.27 % vs markov_roll 1.11 % -> Delta +0.16,
  same sign as on val and 2024 -> go (paired CI [-0.61, 0.75], not significant on its own). Final run started with the frozen configuration
  (feats base+inst+anchor+no_region, all units, blend50 = 0.5 P_recal + 0.5 P_markov_roll, eps 0).

## FINAL (confirmatory, run once; code md5 34fd2d8a1dafa4e2bf38897258abe131)
Departures 2026-03-01 .. 2026-09-06 (4542 h), raion truth, OSM primary, 78 OD pairs, eps 0.
- C4 (anchor + blend50): 2.29 % [1.47, 3.41] vs fixed static; 85 % of ODs better, 0 % worse than -2 %, +17 min.
- markov_roll 1.41 % [0.72, 2.24]; markov (paper) 0.62 %; static_roll 0.65 %; oracle 29.1 %.
- Primary test: C4 - markov_roll = +0.88, paired 3-day-block CI [0.57, 1.27] -> excludes 0: SUCCESS.
- Also: anchor pred 2.27 (vs markov_roll [0.46, 1.39]), pred_recal 2.22, stack 2.24.
- For reference, the earlier untuned raion model on the same block (raion_w2026): 1.5 % [0.7, 2.3], 63 % ODs better.
- Corridors (report only): L-K +2.3, O-K -0.5, Kh-L +3.1, D-K -0.1.
- Hash note: anchor_val was trained before code hashes were recorded (the only later edits before its training
  ended were the hiA assert and the hash/pooled saving); re-evaluated from its saved probs with the frozen code
  (md5 34fd2d8a...): identical numbers (blend50 1.497, markov_roll 0.749). ctrl2024 uses p3_raion_ctrl2024.py.
- Robustness of the final CI (critic, other seeds): 7-day blocks [0.56, 1.26], 14-day blocks [0.52, 1.17];
  monthly C4 - markov_roll Mar..Sep: +0.17 +0.68 +0.73 +1.95 +0.61 +1.27 +0.66.

# Round 5: oblast-level confirmatory test (declared 2026-10-03, before any 2025 run)
- Author's request: confirm C4 at oblast level and rebuild the paper around C4.
- Implementation for the paper: "units": "oblast" (the paper's 25 regions, presence = any oblast- or raion-level
  alert in the oblast, identical to the paper's truth in 2024: 0 differing cells; oblast-binned routes and the
  oblast road graph as in the paper). C4 configuration frozen: feats base+inst+anchor+no_region, recal as before,
  blend50 with markov_roll (42 d), eps 0.
- Dev sanity (2024 block, development data, no tuning): oblast-unit C4 on window c2024.
- CONFIRMATORY (run once): window c2025 = train < 2024-12-01, isotonic Dec 2024, departures 2025-01-01 .. 2025-04-30,
  truth oblast_any, OSM primary 78 ODs. Primary: oblast-unit C4 - markov_roll, paired 3-day-block 95 % CI excludes 0.
  Secondary (reported, not decisive): the raion-unit configuration used for anchor_c2024 (one_per_oblast training).
- Disclosure: Jan-Apr 2025 was seen in aggregate for older methods in the abandoned p3v4 attempt (v3 work);
  nothing in C4 was tuned on it.

## Round 5 result: oblast confirmatory test (run once; code md5 65ebe801..., additive changes only: oblast units, windows)
Departures 2025-01-01 .. 2025-04-30 (2862 h), oblast units, truth oblast_any, OSM primary 78 ODs, eps 0.
- Dev sanity before it (2024 block, oblast units): C4 10.35 %, markov_roll 9.34 % (raion-unit version: 10.40 / 9.35).
- C4 3.02 % [2.20, 3.82] vs fixed static, 90 % ODs better, 0 % worse than -2 %, +17 min.
- markov_roll 1.70 %, markov (full history) 0.40 %, static_roll 1.35 %, oracle 21.4 %.
- PRIMARY: C4 - markov_roll = +1.32, paired 3-day-block CI [1.09, 1.58] -> excludes 0: SUCCESS.
- Also: anchored ML alone 3.23 (vs markov_roll [1.10, 2.03]), recal 3.40, stack 3.29.
- Corridors (report only): L-K +2.1, O-K +4.5, Kh-L +3.7, D-K -0.0.
