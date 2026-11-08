# TAMU Datathon 2026 — Short Deck PLO5 Bot

Bot for the TAMU Datathon poker tournament: heads-up Short Deck (36-card) 5-card Pot-Limit Omaha. Repo: `Stuartdhas-tx/tamu-datathon` (private). `README.md` is the organizers' participant guide and the source of truth for rules and limits.

## Environment

- Use the project venv: `.venv/` (Python **3.12.15**, created with `uv`). It matches the tournament image (`python:3.12-slim` in `docker/Dockerfile.base`), so don't use newer-Python-only syntax. The machine-wide default `python3` is 3.14.8 — don't use it for this project.
- Install/update deps: `uv pip install --python .venv/bin/python -r requirements-dev.txt`
- Run things with `.venv/bin/python ...` (or `source .venv/bin/activate` first).
- Docker (via colima) is available for `tools/check_build.sh`.

## Layout

- `agent.py` — **the bot** (FastAPI server, port 5008 / `$PORT`). Only edit between `YOUR CODE HERE` and `END OF YOUR CODE`; must stay at repo root.
- `brain.py` — table-driven hand features + strategy lookup; loads `tab.bin`, `thresholds.bin`, `pf_bucket.bin`, `pf_feat.bin`, `ctx.npy`; `agent.py` loads `strategy.npy`, `counter.npy`, `counter_aggressive.npy`, `counters/br_<model>.npy` (11 per-model best responses) and `models.npz` (opponent models for showdown evidence; built by `models.py`).
- `opponent.py` — opponent stats tracking (`OppStats`) and `opp_decisions` (replays a shown-down hand from their seat).
- Tests: `test_agent.py` (organizers'), `test_bot.py` (failure paths, counter gates, rules/bluff/steal), `test_models.py`, `test_opponent.py`.
- `train.c` — offline solver that produces the tables (`clang -O3 -march=native -pthread train.c -o train -lm`). `gen_tables.py`, `buckets.py`, `check.py`, `export.py`, `arena.py`, `zoo.py` are offline tooling, excluded from the Docker image via `.dockerignore`.
- `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, `tools/` — organizer-provided; edits have no effect on the real tournament.

## Commands

- Tests: `.venv/bin/python -m pytest -q`
- Local match (3 terminals): `.venv/bin/python agent.py` · `.venv/bin/python baseline-bots/baseline_random.py` · `.venv/bin/python judge_engine.py --deals 200`
- Pre-submit build check: `./tools/check_build.sh` — needs Docker **buildx** (`brew install docker-buildx`, registered via `cliPluginsExtraDirs` in `~/.docker/config.json`); without it the legacy builder fails on `RUN --mount` in `Dockerfile.agent`. Last run: ALL CHECKS PASSED (adds 2 MB over base).

## Competition (from the official Participant Guide PDF)

- Official starter repo: `tamu-datathon-org/heads-up-bundle` (public). As of 2026-11-07 our organizer files (`game.py`, `evaluator.py`, `judge_engine.py`, `docker/`, `tools/`, `test_agent.py`, baselines) are identical to it.
- Settings (stacks, blinds, time bank, ...) **may be tuned before the tournament**; the starter repo's `game.py` is the source of truth. The solver tables (`train.c`, `brain.context_key`) assume 80-chip stacks, 1/2 blinds, 4-raise cap — if upstream `game.py` changes, contexts stop matching and the bot silently degrades to `_fallback()` pot odds. Re-diff against upstream before submitting.
- Round robin; standings by total points, **ties broken by average bb/deal across all matches**.
- Team registration is mandatory (one person on two teams invalidates the whole team). Submission = team name + **public** repo URL + full 40-char commit SHA; latest successful build plays, a failed build keeps the previous one. Build time limit 30 min.

## Second bot: `poker-bot/` ("ShortDeckShark", untracked)

A separate, heuristic bot added alongside the solver: `poker-bot/agent.py` (template plumbing) → `strategy.py` (Monte Carlo equity with the evaluator table, pot odds + stack-off price, c-bets / nut-flush-draw semi-bluffs / blocker bluffs, opponent model with showdown-calibrated range narrowing; `LEVEL` 1–4, default 4; self-budgets to `TIME_TARGET = 4.0` s per half, ≤ 4 ms per decision). Data: `preflop_eq.bin` (built by `build_preflop_table.py`). Own tools: `sim.py`, `scripts/fuzz.py`, `scripts/sweep.py`. It imports `evaluator`/`game` from the repo root, so to submit it its files must move to the root (replacing the solver's `agent.py`). In the practice toolkit, refer to it as `poker-bot/agent.py`.

## Practice toolkit (`practice/`, excluded from the Docker image)

Bot names everywhere: `agent`, a file in `practice/bots/` (`exploit`, `smallball`, ...), `zoo:<name>` (zoo.py), or a path. `practice/common.py` plays in-process matches with the judge's rules (time bank + 1 ms grace, 1 s limit, 5-error forfeit, `judge_engine.decide_verdict`), loading a fresh bot copy per half.

- `practice/check_robustness.py [bot]` — chaos opponent, JSON round trip, rare-spot coverage; also calls the bot's `_decide()` directly to catch errors hidden by its own try/except.
- `practice/check_timing.py [bot] --slowdown 3` — cold startup, per-street latency, time bank on this machine and on an emulated slower one (patches `time.perf_counter` to run k× fast, so deadline-based bots adapt as they really would).
- `practice/check_rules.py [--upstream]` — solver's assumed constants, context/table coverage, organizer files vs official repo.
- `practice/compare.py [--bots ...] [--field ...] [--seeds N] [--round-robin]` — candidates vs field on identical cards; leaderboard by points then bb/deal + matchup matrix.
- `practice/serve.py <bot> --port 5010` — serve any practice bot over HTTP for `judge_engine.py`.
- `practice/solver.py exploitability [table.npy] [--export br.npy]` — best response (train.c `BR_VS`) on a 1.44M-deal eval pool → bb/deal a perfect opponent wins inside the abstraction. `solver.py refine table.npy --out new.npy` — warm-start CFR (patched copy of train.c in `practice/work/`, gitignored). Only compare numbers measured on the same pool size; tiny pools overfit and inflate them. A table can also be played directly as `strat=<path.npy>`.
- `practice/tables/`: `br_agent.npy` (best response to `strategy.npy` as agent plays it, i.e. purified at `agent.PURIFY`; played by the `exploiter` opponent — regenerate with `solver.py exploitability --export practice/tables/br_agent.npy` whenever `strategy.npy` or `PURIFY` changes), `fortress.npy` (slot for the `fortress` bot, ROLE experimental; currently = `strategy.npy`), `counter_passive.npy` / `counter_folder.npy` (used by the failed `chameleon` experiment).
- `compare.py --seed-start N` uses fresh card seeds; confirm any promising result on fresh seeds before acting on it.
- Opponent-adaptive model selection: `practice/opponents_gen.py` (generated training opponents `thr:V,H,M,B,S` / `noisy:NAME:Q`, never the named test bots), `practice/bots/_features.py` (10-number behaviour profile: fold/raise/bet rates, showdown rate, showdown looseness, preflop folds, risk = std of our result per hand), `practice/train_selector.py` (profiles + every portfolio model's points/bb vs each training opponent → `practice/tables/selector.json`), `practice/bots/selector.py` (plays agent for `profile_hands`=150, then picks a portfolio model from the 7 nearest training opponents; risk-aware: a model must score ≥ agent's points against every neighbour). Portfolio: agent, greedy, smallball, poker-bot Shark.

## Minimum-loss findings (exploitability = bb/deal a best response wins, 1.44M-deal eval pool)

- Shipped `strategy.npy` as played (purify 4%): **0.27** — the lowest of everything tried. A real-game best response beat `agent` in only 3 of 18 matches (rest chops).
- Worse: fresh 1B-iteration equilibrium 0.30; warm-start refine (2B iters, no discount) 0.41; purify 15% 0.31; `greedy` 0.93; `smallball` rule **2.14** (its best response won 6/6 in the real game). Rules that win more points vs fixed bots do it by becoming predictable, which adaptive opponents punish.
- The solver's pot-odds fallback never fires in real play (0 of 47k decisions).
- Statistical floor: vs an exactly equal opponent ~2.5% of matches are "losses" from card luck alone; only a real edge removes that.
- Final gauntlet (21 opponents incl. `exploiter`, 10 seeds, 240 matches each): `agent` 183W/55C/**2L** (1 vs exploiter, 1 vs `greedy` by chance); `greedy` 0L and `smallball` 1L but both far more exploitable by a targeted best response; Shark 62L. Across all runs `agent` lost 7 of 564 matches — every loss vs a solver-family bot or the exploiter, **none vs any heuristic bot**.
- Since the submitted repo must be public (and contains `train.c`), anyone could compute a best response to `strategy.npy`: make it public as late as allowed.

## More wins vs fewer losses: `agent.PURIFY` (changed 0.04 → 0.25 on 2026-11-07)

- Chops come from real but small edges (passive tight bots: eq_shade/eq_tight/trapper/adaptive ≈ +0.5 bb/deal ± 0.5). Less random mixing narrows the interval: purify 0.25 vs 0.04, pooled over 525 field matches on two seed sets (0–9 and fresh 100–114): **481W/42C/2L vs 461W/59C/5L**, +0.022±0.013 points/match. 0.15/0.20/0.30/0.35 and greedy gained less or nothing.
- Cost: exploitability 0.27 → 0.59 bb/deal (0.31 at 0.15, 0.81 at 0.35, 0.93 greedy). A best response to the purified strategy (`exploiter`, retrained) beats `agent` in 8 of 12 matches (−0.80 bb/deal); against the 0.04 version its counterpart won ~1 in 7–10. Only matters if a competitor targets this exact strategy. To go back to minimum-loss play, set `PURIFY = 0.04` and regenerate the exploiter table.
- Didn't work: counter-strategy tables per opponent type (`chameleon`, best responses to a distorted copy of our strategy) — 61.5 vs 71.5 points, brittle vs the real bots; train.c's restricted-response mode changes almost nothing (multiplicative distortion of a pure best response is a no-op).
- `exploit` / `greedy` / `smallball` / `combo` call agent's `_strategy`, so they now inherit PURIFY 0.25.
- Opponent-adaptive `selector` (2026-11-07; 48 generated training opponents, evaluated on the 22 named bots, 15 fresh seeds each): v1 (5-NN average, bb tiebreak) −0.006±0.016 points/match vs agent, +0.62±0.18 bb/deal (picked the Shark vs `trapper` → 7 chops); v2 risk-aware (7-NN, worst-neighbour rule) **+0.006±0.016 points/match (294W/33C/3L vs 290W/37C/3L), +0.32±0.14 bb/deal**. So: no significant reduction in chops, a real bb/deal (tiebreak) gain. Limits: portfolio models differ by tiny amounts in points vs any one bot, and a 150-hand profile is noisy (picks can differ between the two halves). Upper bound measured earlier: even perfect per-opponent selection ≈ +0.01–0.03 points/match. Not in `agent.py`; packaging it would mean shipping the Shark's `strategy.py` + `preflop_eq.bin`, greedy/smallball logic, `_features.py` and `selector.json` at the repo root.
- After the change: pytest 13/13, robustness + timing PASSED, `tools/check_build.sh` ALL CHECKS PASSED (needs the buildx config noted above).
- Candidates (`ROLE = "candidate"`): `exploit` (solver + OppStats re-weighting, priors measured solver-vs-solver), `greedy` (argmax), `smallball` (pot raises -> half-pot unless strength >= 0.8), `combo` (exploit + smallball). Opponents: `shover`, `mc_equity` (official evaluator MC; ~6.3 s bank at 3x slowdown), `adaptive` (exploits us via OppStats).
- Findings (2026-11-07, 3 runs, 1,483 matches): every bot 0 errors. Solver variants chop each other. Pooled paired vs `agent` (points/match): `smallball` +0.030±0.023 (only significant gain), `greedy` +0.014±0.021, `combo` +0.013±0.029, `exploit` +0.004±0.023. `poker-bot` Shark: −0.300±0.071 points/match vs `agent` (70.5/125 pts vs ~110) — highest bb/deal (+1.71; +11.8 vs a station) but loses to tight equity bots (eq_tight/eq_shade/eq_shade2 LLLLL) and to every solver variant (≈ −1 bb/deal). Solver stays the submission (later tuned: PURIFY 0.25, see below).

## Teammates' branches on GitHub (fetched 2026-11-07; local checkout is still at Initial Commit `1734116`)

- `main` → `9b65a8a` "CHECK1" (Ajay Pal): agent.py only — solver at PURIFY 0.04 plus a rule layer (pot odds + value pot-raises) once an opponent is clearly passive-and-sticky or wild (≥ 40 opportunities). Conflicts with our uncommitted agent.py edit.
- `feature/sai` → `6ca28af` (sai0215; `poker-bot` branch = its first commit): solver plus two counter tables (`counter.npy` honest, `counter_aggressive.npy`), unlocked only when showdown evidence (`models.npz`, `models.py`) fits that family, then Thompson sampling between arms on chips won. Adds 10 held-out opponents to `zoo.py`. Own tests 23/23, robustness + timing PASSED (0.73 s of 8 s at 3× slowdown).
- Head-to-head in a scratch copy of the sai tree (30 opponents, fresh seeds 300–304 and 400–409): `feature/sai` vs ours (PURIFY 0.25) **+0.017±0.011 points/match pooled (n=450), +2.3 bb/deal**; 287W/29C/4L vs 275W/40C/5L on the 10-seed pass. `main` ≈ original on points (−0.007±0.019), +0.73 bb/deal. Our PURIFY 0.25 gain did **not** replicate on this field (−0.003±0.020 vs the original), and PURIFY 0.25 on sai's bot changes nothing (+0.002±0.008). Nearly all remaining losses of every version are vs `exploiter` (which targets our PURIFY-0.25 table, not sai's).
- **Current `agent.py` (2026-11-07) = sai's bot + main's extreme-opponent rules.** Working tree has sai's `agent.py`, `opponent.py` (adds `opp_decisions`), `zoo.py`, `models.py`, `models.npz`, `counter.npy`, `counter_aggressive.npy`, `test_bot.py`, `test_models.py`, `test_opponent.py` (via `git restore --source=origin/feature/sai --worktree`); the old PURIFY-0.25 agent is in the session scratchpad backup only. Added `RULES` (`"override"` default / `"arm"` / `"off"`), `_extreme()`, `_rule_action()`: once `OppStats` shows passive-and-sticky or wild play (≥ `RULE_MIN`=40 chances), pot odds + pot raises replace the arm (those hands aren't credited to the bandit arms; showdown evidence still is). `_strategy(..., arm=0)` keeps the practice bots working. `models.npz`'s 11 opponent models are exact copies of zoo bots (eq_* honest, cbet/maniac/alwayspot/alwayshalf/random aggressive), so measured gains vs those are optimistic.
- Rules vs sai (31 opponents incl. Shark, fresh seeds 500–509, paired): override **+0.35±0.15 bb/deal**, points +0.002±0.007; arm +0.18±0.11 bb/deal; off = sai (sanity). Gains only vs station/alwayshalf/alwayspot (+3 to +5 bb/deal); the rules never fire vs mc_equity, eq_*, exploiter (spy-counted), so differences there are card noise. pytest 23/23, robustness + timing (0.73 s at 3×) + check_rules + `tools/check_build.sh` ALL PASSED. `practice/tables/br_agent.npy` (exploiter) regenerated 2026-11-08 against the current PURIFY-0.04 arm 0: 0.275 bb/deal.

## Exploitation layers added on top of sai's bot (2026-11-07, all in `agent.py`)

- **Ceilings** (best response trained against each `models.npz` model with `practice/solver.py exploitability <model.npy> --purify 0`, bb/deal inside the abstraction): basic 8.5, loose 10.3, tight 3.9, aggro 7.6, shade 3.5, shade2 3.4, cbet 27.4, maniac 13.5, pot 9.2, half 15.6, rand 20.5. Against tight/balanced bots even a perfect counter wins ~3–4, so a field average of 10 bb/deal is not reachable; the room is vs loose/aggressive bots.
- **Per-model portfolio** (Bard et al. 2013 implicit modelling): `counters/br_<model>.npy` = those 11 best responses (arms 3–13, `MODEL_ARM0`). A model's counter unlocks when its showdown evidence beats balanced by the usual gate AND the next most likely model by `MODEL_UNLOCK`=5; `MODEL_PRIOR`=(0.5, 20) is deliberately sceptical (an optimistic 3.0 kept wrong counters for hundreds of hands, e.g. `br_rand` vs streetbluff, `br_loose` vs mc_equity). Measured vs portfolio off: +0.35±0.10 bb/deal pooled (two runs, n=688), sceptical version +0.33±0.14 (other zoo+bots +0.19±0.12, held-out generated +0.14±0.25 n.s.); points tie. Big: maniac +5.6, passive +2.3, eq_loose +1.8; small costs cbet −0.8, exploiter −0.27, shover −0.3. Aggressive-only (`MODEL_FIRST`=6) gained ~nothing. Gains vs the exact model copies (+0.8–1.0) are optimistic.
- **`_adjust()` postflop exploits** (deviation-based, need `EXPLOIT_MIN`=30 chances): `BLUFF`=(0.40, 0.55) — half-pot bluff when checked to with strength < 0.40 vs opponents folding > 55% to our bets: +0.23±0.12 bb/deal, held-out +0.57±0.34, 0 losses. Tested and left OFF: `VALUE_RAISE` (turn calls into pot raises with strong hands, the "be more proactive" idea) −0.11±0.08 (cbet −1.7, eq_shade −0.6: their bets are strong or their bluffs fold to raises), with `VALUE_RAISE_LOOSE` gate −0.10; `THIN_VALUE` ±0.
- Before the changes, strong hands were already bet 88–95% when checked to; only when facing a bet do we mostly call (0.8–0.9 strength: raise 59% flop / 46% turn / 23% river) — and calling there measured better than raising.
- `STEAL` (preflop, first in: pot-raise every hand once they fold > x to our first-in pot raises, counted in `memory["steal"]`) tested OFF: −0.09 to −0.15 bb/deal at 0.5/0.65/0.8. Bigger raises do make tight bots fold (nit 4% → 100%, blinder 0% → 73% from half-pot to pot) but nit gains nothing (we already won those pots postflop) and eq_shade/shade2 lose 0.4–0.6 (they keep their good hands). Per-opponent "significant" swings at 8 seeds include pure noise: the steal never fired vs mc_equity/maniac/passive yet showed −0.8 to −1.6 there — trust pooled numbers, spy-count whether a feature fires before believing a per-opponent effect.
- Official baselines only (`baseline-bots/*.py` loaded by path in compare.py, 10 seeds): final agent **+11.7 bb/deal** (random 13.7, always_half 13.0, calling_station 12.4, always_pot 7.8), sai +8.5, old PURIFY-0.25 agent +3.5. "+8" claims are most likely vs these.
- Held-out test field: `opponents_gen.training_set(12, seed=777)` with `/` separators (`thr:V/H/M/B/S`; `make()` accepts both so names survive comma-separated `--field` lists).
- `test_bot.py` adapted: fixed table slots (None if missing), `test_model_gate` added; 24/24 pass. Real HTTP match via `judge_engine.py` (`OMP_NUM_THREADS=1`): **0.00 s of 8 s** charged on an idle machine (~0.4 ms per decision, inside the 1 ms grace); 2.5–3.1 s only under heavy CPU contention from parallel jobs. `tools/check_build.sh` ALL PASSED; inside the image (1 CPU, no network) all 14 tables load.
- Don't stop test servers with `pkill -f agent.py`: it also matches `poker-bot/agent.py` in running compare.py arguments. Kill by PID.

## Hard constraints (from README)

- 8 s thinking per 1000-hand half (~4–7 ms/decision), max 1 s per decision; server must answer within 15 s of start.
- 5 errors (crash/illegal/invalid action) in a half = forfeit = match loss. Crashes in `on_hand_end()` count too.
- Use the `memory` dict for state (wiped each half); no network at match time; only `/tmp` writable.
- Keep the `PORT` env handling and server routes/reply format intact. FastAPI/uvicorn versions are fixed.
- Image may add ≤ 3 GB over base (base has numpy + CPU PyTorch); files < 100 MB, no Git LFS.
