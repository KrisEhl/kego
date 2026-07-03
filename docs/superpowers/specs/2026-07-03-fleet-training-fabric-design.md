# Fleet Training Fabric — Design Spec

- **Date:** 2026-07-03
- **Status:** Approved — open decisions resolved (see §9)
- **Owner:** Kristian
- **Scope:** A reusable, competition-agnostic system to run kego training across many
  machines (SSH over Tailscale), keep omarchyd as the hub, and tie every trained model
  and machine into one central view.

## 1. Context & motivation

Training currently happens either locally or as a single `kego train-agent --executor ray`
job on the omarchyd Ray cluster. We have several idle machines and recurring compute
bottlenecks across competitions (self-play for pokemon, GBDT/neural fan-out for tabular,
etc.). We want to use the whole fleet, with results and checkpoints aggregated centrally.

Much of the plumbing already exists and is reused rather than rebuilt:

- **Central MLflow** on omarchyd (`http://192.168.178.32:5000`) with an artifact store.
- **Offline tracking** — `MLFLOW_TRACKING_URI=sqlite:///…` + `kego sync` (idempotent, keeps
  full metric history).
- **Ship-local-code** — the Ray path already rsyncs/uploads the working tree so a stale
  remote checkout can't be run.
- **`kego ls` / `kego status` / `kego logs`** — run/queue/log views over MLflow + Ray.
- Pokemon-specific **gauntlet eval** (per-opponent + rule-agent-average win rates),
  **best-checkpointing**, and the **tournament** (`run_tournament.py`, agents load a
  checkpoint via `MCTS_MODEL_PATH`).

## 2. Goals / non-goals

**Goals**
- Run training on any fleet machine via a single command, from this Mac.
- One central store (MLflow on omarchyd) of every run, metric, and best checkpoint.
- A leaderboard of all trained models across machines (any competition).
- A cross-machine status view (which boxes are alive, their load/GPU, what's running).
- Offline tolerance: a disconnected box trains to a local store and reconciles on reconnect.
- Competition-agnostic: primitives live in `kego/` core; competitions plug in scoring.

**Non-goals (for now)**
- An always-on scheduler / autoscaler (manual/`--target` dispatch is enough at this scale).
- Turning every box into a Ray worker (see §4 — standalone-per-machine is primary).
- A web UI beyond MLflow's own UI + CLI tables.
- Secure multi-tenant auth (single-user, trusted Tailscale network).

## 3. The fleet

All machines are reachable by SSH over Tailscale.

| Machine | Role | Notes |
|---|---|---|
| **omarchyd** | `hub` + compute | MLflow server, git origin, artifact store, Ray head; RTX 3090 + 2080 Ti |
| **M5 MacBook Pro (48 GB)** | `cpu` | Strong CPU self-play box (many fast cores) |
| **this Mac** | `cpu` + dev/submit | Where the operator drives the fleet |
| **WSL + RTX 2080 Ti** | `gpu` | Real GPU; bigger models / training-heavy |
| **Linux laptop + small GPU** | `gpu` | Modest GPU; small models / training step |

**Insight that shapes assignment:** pokemon self-play is CPU-bound (measured — ~90% of a
move is the batch-1 NN forward on CPU workers). So CPU-rich Macs are first-class self-play
nodes; GPU boxes favor bigger models and the batched training step. Role is advisory, not
enforced.

## 4. Architecture

Hub-and-spoke with **MLflow as the single spine**. Every run — Ray, SSH-launched, GPU, or
CPU — logs to the same central MLflow, tagged with its machine and code SHA.

```
                       omarchyd (HUB)
   git origin · MLflow (runs, metrics, MODEL REGISTRY, artifacts) · Ray head · fleet.toml
        ▲            ▲                     ▲                 ▲
   log/artifacts     │ ssh-launch + log    │ ray            │ ssh-launch + log
   ┌────┴────┐  ┌────┴─────┐        ┌───────┴──────┐  ┌──────┴──────┐
   │ M5 Mac  │  │ this Mac │        │ Ray worker   │  │ WSL 2080 Ti │  ...
   │ (cpu)   │  │ (dev/cpu)│        │ (co-located) │  │ (gpu)       │
   └─────────┘  └──────────┘        └──────────────┘  └─────────────┘
```

**Primary compute model: standalone-per-machine over SSH.** Each box runs its own
`kego train-agent`/`kego run`, launched (detached) by a dispatcher from the operator's
machine, logging to central MLflow. Rationale:

- Self-play (and most sweeps) are **embarrassingly independent** → N boxes = N agents/models
  exploring different seeds/configs → a *population*, which is a real lever (esp. for the
  pokemon specialists).
- **Robust to flaky links** — the Mac hangs as a Ray worker; WSL needs mirrored-networking
  + firewall rules for Ray. Plain SSH over Tailscale + ship-local-code avoids all of it.
- The **Ray cluster stays as one `--target`** for the co-located 3090/worker where a single
  fan-out job is preferable.

## 5. Components

### 5.1 `fleet.toml` (machine registry) — core

Location: repo root `fleet.toml` (checked in, no secrets) with optional `~/.kego/fleet.toml`
overrides. Schema:

```toml
[hub]
name   = "omarchyd"
mlflow = "http://192.168.178.32:5000"

[[machine]]
name = "omarchyd"
ssh  = "kristian@omarchyd"          # Tailscale-resolvable
role = "hub"                         # hub | gpu | cpu
repo = "/home/kristian/projects/kego"
data = "/home/kristian/projects/kego/data"
gpus = ["rtx3090", "rtx2080ti"]

[[machine]]
name = "m5"
ssh  = "kristian@m5"
role = "cpu"
repo = "/Users/kristian/projects/kego"
data = "/Users/kristian/projects/kego/data"
gpus = []
```

Parser + dataclasses in `kego/fleet.py`. No secrets in the file (SSH keys/Tailscale handle
auth).

### 5.2 Tracking contract — core (`kego/tracking.py`)

- **URI resolution:** use `KEGO_MLFLOW` env → else `[hub].mlflow` from fleet.toml → ping it;
  if unreachable, fall back to `sqlite:///~/.kego/offline.db`. Print which was chosen.
- **Run lifecycle:** the dispatcher/runner owns it (mirrors existing kego convention —
  training code never calls `mlflow.start_run()`; it reads `KEGO_MLFLOW_RUN_ID`). If no run
  id is injected (e.g. a bare local run), the training entry creates one.
- **Standard tags** on every run: `machine`, `git_sha`, `task` (competition slug), `role`,
  `config` (JSON of the training config).
- **Standard metrics** (logged live): `progress_pct`, and competition-provided scalars. For
  pokemon: `gauntlet_avg`, `wr_<opponent>` (random/zacian/…/self), `loss_value`,
  `loss_policy`, `best_gauntlet_avg`.
- A thin helper: `track = Tracker.resolve()`; `track.log_metric(name, val, step)`;
  `track.set_tags({...})`. Safe no-op if MLflow import/connection fails (never crash a run
  over telemetry).

### 5.3 Model registry — core

- On each **new best checkpoint**, log the `.pth` as an MLflow artifact and register a
  version in the MLflow Model Registry under model name = `task` slug, with tags
  `{machine, git_sha, gauntlet_avg, wr_*, config, run_id, model_args}`.
- **Two-tier metric (important — fixed-opponent scores saturate):**
  - `gauntlet_avg` (vs the fixed rule agents) is the **in-training signal**: it gates
    best-checkpointing and serves as a fixed-scale **anchor**. It stops discriminating once
    agents beat the rule agents (~90%), so it is *not* the fleet ranking metric.
  - **League Elo** (from §5.6, anchored on the rule agents) is the **relevance-preserving
    ranking metric**: agent-vs-agent play never saturates because opponents scale with the
    population. Registry tags carry the latest `elo` + `elo_rd` (uncertainty), updated by
    league runs.
- `kego models --task <slug>` → leaderboard **ranked by league Elo** (± uncertainty), with
  `gauntlet_avg` shown alongside as the training-time signal; columns: rank, elo±rd,
  gauntlet_avg, machine, git_sha, key config, artifact path.
- Any machine can `pull` a registered checkpoint by version (artifact download) — this is
  how the cross-fleet league (§5.6) gets its contenders.

### 5.4 Dispatcher — core (`kego <cmd> --target <machine|cluster|auto>`)

Extends the existing CLI. For `--target <machine>` (non-Ray):

1. **Ship code:** rsync the working tree to `<machine>:<repo>` (excludes `.git`, `.venv`,
   `data`, `outputs`, `model_data`, other competitions — same list as the Ray upload). No
   commit required (matches ship-local-code).
2. **Env:** assume `uv sync` done (see `kego fleet setup`); optionally run it.
3. **Create MLflow run** (central-or-offline), obtain `run_id`, set standard tags.
4. **Launch detached:** `ssh <ssh> 'bash -lc "cd <repo> && KEGO_MLFLOW_RUN_ID=<id> nohup uv
   run kego <cmd> … >~/.kego/logs/<id>.log 2>&1 &"'` (tmux optional).
5. Print the `run_id`; progress is followed via `kego ls` / `kego fleet status` / `kego logs`.

`--target cluster` keeps the existing Ray path. `--target auto` (Phase 4) picks by role +
live load from `fleet status`.

`kego fleet setup <machine>` (one-time per box): clone/rsync repo, `uv sync`, download the
competition data (`make download-competition-data`) so `cg`/data is present.

### 5.5 `kego fleet status` — core

SSH-poll each machine in `fleet.toml`: `uptime`/load, `nvidia-smi` util+mem (if `gpus`),
running `kego`/`train-agent` processes, free disk. Merge with MLflow run states. Output one
table: machine | alive | role | gpu util/mem | running runs | last log line. Pure SSH-poll
(no daemon).

### 5.6 League / cross-fleet ranking — competition plug-in

Produces the relevance-preserving ranking metric (§5.3). `run_tournament.py` gains
`--from-registry [--task <slug>] [--top N]`, running an **anchored Elo rating ladder** rather
than a full O(N²) round-robin:

- **Anchors:** the rule agents are always included with **pinned Elo** — they calibrate every
  round to one stable scale, so agents registered at different times/machines are comparable.
- **Ladder, not full round-robin:** each new/updated agent plays a bounded number of games
  vs the anchors + a sample of the current top; periodically re-play top agents to keep
  ratings fresh. Cost stays bounded as the registry grows.
- **Output:** updated `elo` + `elo_rd` (uncertainty from games played) written back to each
  registry version's tags → `kego models` ranks by it. Promote/submit the top-Elo agent.

Competitions provide the anchor set + how to instantiate a checkpoint as a player (pokemon:
the rule agents + `MCTS_MODEL_PATH`).

**Maintenance & storage (no separate DB — MLflow is the spine):**

- **State lives on the hub:**
  - *Checkpoints* → MLflow artifact store, attached to each registered version (keep all for
    now; monitor store size and add pruning later if needed — see §9).
  - *Ratings* (`elo`, `elo_rd`, `games`, `rating_status`, `active`) → **Model Registry tags**
    on each version — authoritative, read directly by `kego models`.
  - *Match history* → artifacts (CSV) in an MLflow experiment `league-<task>`, one run per
    ladder round — auditable/reproducible.
- **A single league process on the hub owns updates** (one serialized writer → no races when
  several boxes register at once). Triggers: manual `kego league run --task <slug>`; on new
  registration (version tagged `rating_status=pending`); a hub **cron** every N hours to
  place pending agents + refresh leaders.
- **Ladder update:** new agent enters at provisional Elo + high RD; plays K placement games
  vs the pinned anchors + a sample of the top → Glicko update; each round re-plays leaders vs
  recent entrants for freshness. To play a match, pull both versions' `.pth` from the artifact
  store (cached locally), instantiate via the competition's player hook.
- **Bounded cost:** dedup identical checkpoints (git_sha + config + weight hash); retire
  clearly-dominated agents to `active=false` (kept in the registry, out of the active ladder).
- **Offline agents** register locally and enter the `pending` queue on `kego fleet sync`.

### 5.7 Offline reconciliation — core (extends `kego sync`)

- Disconnected box logs to `sqlite:///~/.kego/offline.db` and saves checkpoints locally.
- `kego fleet sync`: for each machine, rsync its offline db + artifacts to the hub and
  `kego sync --from sqlite:///…` into central MLflow (idempotent; tagged `kego_synced_from`).
- Optional: auto-sync on reconnect (a `kego train-agent` that started offline pushes on exit
  if the hub is now reachable).

## 6. A run's lifecycle (data flow)

1. Operator: `kego train-agent --target m5 --epochs 200` (pokemon config from `kego.toml`).
2. Dispatcher rsyncs working tree → m5, creates MLflow run (central or offline), tags it.
3. m5 runs self-play + gauntlet; logs `gauntlet_avg`/`wr_*`/loss/progress live to MLflow.
4. On each new best, m5 registers the checkpoint (artifact + registry version + score tags).
5. Operator watches `kego ls` / `kego fleet status`; ranks via `kego models`.
6. If m5 was offline, `kego fleet sync` reconciles when omarchyd returns.
7. `run_tournament --from-registry top-8` ranks the fleet's best; winner is promoted.

## 7. Core vs competition-specific

- **Core (`kego/`):** `fleet.toml` parser, `tracking.py` (URI resolution + Tracker),
  registry helpers, dispatcher (`--target`), `kego fleet status/setup/sync`, `kego models`.
- **Competition:** (a) which scalars to log, (b) the in-training **gate** metric (pokemon →
  `gauntlet_avg`), (c) the `train`/eval entry (`run_training_loop`, gauntlet), and (d) the
  league anchor set + how to instantiate a checkpoint as a player. The fleet **ranking** is
  always league Elo (§5.6), independent of the competition.

## 8. Phased delivery

Each phase is independently useful and shippable.

- **Phase 0 — Foundation (highest value / lowest risk).** `fleet.toml` + `kego/tracking.py`
  (central-or-offline) + wire pokemon `run_training_loop` to log metrics + register best
  checkpoint + `kego models` leaderboard (ranked by `gauntlet_avg` until the league lands in
  Phase 3). *Outcome:* every box's runs — even manually launched — appear centrally with scores.
- **Phase 1 — Dispatcher.** `kego train-agent --target <machine>` (rsync + SSH-launch +
  tracked) and `kego fleet setup`. *Outcome:* one-command multi-box training.
- **Phase 2 — Views.** `kego fleet status` + polish `kego models`.
- **Phase 3 — League + auto-sync.** `run_tournament --from-registry`, `kego fleet sync`.
- **Phase 4 — (optional).** Ray for co-located nodes; `--target auto` load-based scheduling.

## 9. Risks & open decisions

- **Heterogeneous devices:** checkpoints are portable (state_dict), but a run started with
  one `MODEL_ARGS` must be scored/loaded with the same. Store `model_args` in run tags +
  with the checkpoint so the registry/tournament always builds the matching architecture.
  *(We already made `MODEL_ARGS` a single source of truth.)*
- **Data presence:** the pokemon `cg` engine must exist on each box (via competition data).
  `kego fleet setup` handles it; `fleet status` should flag `data_missing`.
- **MLflow artifact size:** 26 MB+ checkpoints × many runs. **Decided:** keep all for now,
  just monitor store size; revisit a prune policy (e.g. top-K per task + latest per machine)
  once it's actually a problem.
- **Clock/`git_sha` skew across machines:** always tag `git_sha`; `kego models` shows it so
  we never compare across divergent code unknowingly (same lesson as the Ray stale-code fix).
- **Offline artifact sync:** local-only checkpoints must rsync to the hub, not just the
  metrics. `kego fleet sync` copies both.
- **Ladder constants (defer):** Glicko update params, placement `K`, top-sample size, and the
  re-play cadence are Phase-3 knobs — **tune later against actual match data**, not upfront.
- **Code shipping — decided: rsync** the working tree (no commit, works with a dirty tree),
  consistent with the ship-local-code we already validated. (Not git-pull.)

## 10. Testing strategy

- `kego/fleet.py`: parse `fleet.toml` (valid, missing fields, overrides) — pure unit tests.
- `kego/tracking.py`: URI resolution (central reachable → central; unreachable → sqlite
  fallback), Tracker no-op when MLflow absent, tag/metric round-trip against a temp
  `sqlite:///` store — unit tests, no network.
- Registry helpers: register + query a leaderboard against a temp sqlite MLflow.
- Dispatcher: unit-test the rsync/ssh command construction (no live SSH); an opt-in
  integration test against `localhost`/one fleet host, skipped by default.
- `kego fleet status`: parse `nvidia-smi`/`uptime`/`ps` output fixtures — unit tests.
- Follow existing conventions (`tests/`, skip integration when hosts/`cg` unavailable).

## Appendix A — Example league round (illustrative)

Made-up but internally-consistent numbers across the fleet. A new checkpoint `v45` from the
**M5** has just registered as `rating_status=pending`; the ladder runs a placement (v45 vs
the anchors + the current #1) plus a leader-refresh.

### A.1 The MLflow experiment `league-pokemon-tcg-ai-battle` (one run per round)

```
Run: ladder-2026-07-03T14:30Z          run_id = d5e9a1…
  params:
    round_type        = placement+refresh
    games_per_pairing = 20
    anchors           = random,zacian,lucario,dragapult,abomasnow   # pinned Elo
    rated_versions    = 45                # the pending agent
    refreshed         = 41,37             # leaders re-played for freshness
    league_git_sha    = 4d6b8af
  metrics:
    n_matches = 8   n_games = 160   top_elo = 1748
  tags:  task = pokemon-tcg-ai-battle
  artifacts:  matches.csv   standings.csv
```

### A.2 `matches.csv` (match-history artifact)

```
round,             player_a, player_b,       deck_a,    deck_b,    a_wins,b_wins,draws,games
2026-07-03T14:30, mcts@v45, rule:random,    abomasnow, random,     20, 0, 0, 20
2026-07-03T14:30, mcts@v45, rule:zacian,    abomasnow, zacian,     18, 2, 0, 20
2026-07-03T14:30, mcts@v45, rule:lucario,   abomasnow, lucario,    16, 4, 0, 20
2026-07-03T14:30, mcts@v45, rule:dragapult, abomasnow, dragapult,  15, 5, 0, 20
2026-07-03T14:30, mcts@v45, rule:abomasnow, abomasnow, abomasnow,  12, 8, 0, 20   # beats the specialist
2026-07-03T14:30, mcts@v45, mcts@v41,       abomasnow, abomasnow,  11, 9, 0, 20   # beats the current #1
2026-07-03T14:30, mcts@v41, mcts@v37,       abomasnow, abomasnow,  12, 8, 0, 20
2026-07-03T14:30, mcts@v37, rule:abomasnow, abomasnow, abomasnow,   9,11, 0, 20
```

### A.3 `standings.csv` (Elo after the round → mirrored to registry tags)

```
rank, agent,           elo,  rd,  games, active, kind
   1, mcts@v45,       1748,  41,   140,  true,   mcts
   2, mcts@v41,       1712,  48,   120,  true,   mcts
   3, mcts@v37,       1666,  55,    80,  true,   mcts
   -, rule:abomasnow, 1650,   0,   inf,  true,   anchor
   4, mcts@v30,       1588,  70,    40,  true,   mcts
   -, rule:dragapult, 1520,   0,   inf,  true,   anchor
   5, mcts@v22,       1502,  95,    20,  true,   mcts
   -, rule:lucario,   1450,   0,   inf,  true,   anchor
   -, rule:zacian,    1350,   0,   inf,  true,   anchor
   -, rule:random,    1200,   0,   inf,  true,   anchor
```

### A.4 Registry tags on the model version (authoritative rating + provenance)

```
name = pokemon-tcg-ai-battle   version = 45   source = mlflow-artifacts:/…/mcts.pth
tags:
  elo=1748  elo_rd=41  games=140  rating_status=rated  active=true
  machine=m5  git_sha=4d6b8af  model_args="(256, 4, 512, 2, 2)"  search_count=25
  gauntlet_avg=69.5  wr_abomasnow=58 wr_zacian=88 wr_dragapult=72 wr_lucario=80 wr_random=100
  run_id=d5e9…   config={"self_play_games":48,"batched":true,"train_steps":200,…}
```

### A.5 `kego models` leaderboard (reads tags, ranks by Elo)

```
pokemon-tcg-ai-battle — 5 agents (+5 anchors) · league round 2026-07-03T14:30

RANK  ELO ±RD     GAUNTLET  MACHINE       GIT      SC  MODEL_ARGS       VER
  1   1748 ±41    69.5%     m5            4d6b8af  25  (256,4,512,2,2)  v45  *
  2   1712 ±48    71.2%     wsl-2080ti    8aac235  25  (256,4,512,2,2)  v41
  3   1666 ±55    66.0%     this-mac      e7bd6e8  25  (256,4,512,2,2)  v37
  ·   1650 (anchor)  —      rule:abomasnow
  4   1588 ±70    61.5%     linux-laptop  4d6b8af  10  (128,2,256,1,1)  v30
  ·   1520 (anchor)  —      rule:dragapult · 1450 lucario · 1350 zacian · 1200 random
  5   1502 ±95    58.0%     m5            2164a1f  25  (256,4,512,2,2)  v22
```

### A.6 Why this design exists (the entry that proves it)

`v41` has the **higher `gauntlet_avg` (71.2 vs 69.5)** so the *fixed-opponent* metric ranks
it first — but head-to-head `v45` **beat `v41` 11–9**, so its **Elo is higher (1748)** and it
is correctly #1. Once agents are strong, `gauntlet_avg` disagrees with real strength and the
anchored league Elo resolves it. `v45` also sits **above the `abomasnow` anchor (1650)**,
making "we beat the specialist" a legible, permanent fact on the leaderboard.
