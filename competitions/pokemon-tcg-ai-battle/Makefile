# Makefile for the pokemon-tcg-ai-battle competition.
# Run from this directory (`make train-agent`) or with `make -C <path> train-agent`.
# The task slug is auto-detected from kego.toml, but passed explicitly for clarity.

TASK    ?= pokemon-tcg-ai-battle
EPOCHS  ?= 1
OUTPUT  ?= outputs/mcts.pth
INIT    ?=

INIT_FLAG =
ifneq ($(INIT),)
  INIT_FLAG = --init-checkpoint $(INIT)
endif

# Train the agent locally (serial executor).
#   make train-agent EPOCHS=10 OUTPUT=outputs/mcts.pth
train-agent:
	uv run kego train-agent --task $(TASK) --epochs $(EPOCHS) --output $(OUTPUT) $(INIT_FLAG)

# Train on the Ray cluster (submits via the Ray executor, tails logs, scp's weights back).
#   make train-agent-cluster EPOCHS=50
train-agent-cluster:
	uv run kego train-agent --executor ray --task $(TASK) --epochs $(EPOCHS) --output $(OUTPUT) $(INIT_FLAG)

# Dispatch training to a standalone fleet machine (rsync working tree + SSH-launch detached).
#   make train-agent-fleet MACHINE=omarchyl EPOCHS=200
# Follow with `kego status` or monitor in MLflow. Requires the machine in fleet.toml.
train-agent-fleet:
	if [ -z "$(MACHINE)" ]; then echo "MACHINE is unset (e.g. MACHINE=omarchyl)" && exit 1; fi
	uv run kego train-agent --task $(TASK) --target $(MACHINE) --epochs $(EPOCHS) --output $(OUTPUT) $(INIT_FLAG)

# --- Ray cluster (the head runs on omarchyd; required before train-agent-cluster) ---
# The head must run ON the head node, so these SSH in and reuse the canonical
# competitions/playground targets. First run syncs the playground venv (has ray).
RAY_HOST       ?= kristian@omarchyd
RAY_PLAYGROUND ?= /home/kristian/projects/kego/competitions/playground

# Start the Ray head on the cluster (run after a cluster restart).
ray-head:
	ssh $(RAY_HOST) 'bash -lc "cd $(RAY_PLAYGROUND) && make start-head"'

# Show the Ray cluster status (nodes, resources).
ray-status:
	ssh $(RAY_HOST) 'bash -lc "cd $(RAY_PLAYGROUND) && uv run ray status"'

# Stop Ray on the cluster head.
ray-stop:
	ssh $(RAY_HOST) 'bash -lc "cd $(RAY_PLAYGROUND) && make stop"'

# Query the MLflow model registry for registered task models.
models:
	uv run kego models --task $(TASK)

# Run a round-robin round, update each registered agent's league Elo, and write it back
# to the registry (so `make models` reflects the new games). More rounds = sharper ratings.
#   make leaderboard GAMES=4 SEARCH_COUNT=10 WORKERS=8 DEBUG=1
GAMES        ?= 200
SEARCH_COUNT ?= 10
WORKERS      ?= $(shell n=$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); if [ "$$n" -gt 2 ]; then echo $$((n - 2)); else echo 1; fi)
DEBUG        ?= 0
WRITE_RATINGS ?= 1
FLEET_WRITE_RATINGS ?= 0
FLEET_WORKERS ?=
CACHE_DIR    ?=
PARTIAL_SAVE_EVERY ?= 5000

# Toggle verbose logging based on DEBUG
DEBUG_FLAG =
ifeq ($(DEBUG),1)
  DEBUG_FLAG = --debug
endif

RATING_FLAG =
ifeq ($(WRITE_RATINGS),1)
  RATING_FLAG = --write-ratings
else
  RATING_FLAG = --no-write-ratings
endif

FLEET_RATING_FLAG =
ifeq ($(FLEET_WRITE_RATINGS),1)
  FLEET_RATING_FLAG = --write-ratings
else
  FLEET_RATING_FLAG = --no-write-ratings
endif

CACHE_FLAG =
ifneq ($(CACHE_DIR),)
  CACHE_FLAG = --cache-dir $(CACHE_DIR)
endif

FLEET_WORKERS_FLAG =
ifneq ($(FLEET_WORKERS),)
  FLEET_WORKERS_FLAG = --workers $(FLEET_WORKERS)
endif

leaderboard:
	uv run kego leaderboard --games $(GAMES) --task $(TASK) --search-count $(SEARCH_COUNT) --workers $(WORKERS) $(DEBUG_FLAG) $(RATING_FLAG) $(CACHE_FLAG) --partial-save-every $(PARTIAL_SAVE_EVERY)

# Dispatch a detached leaderboard run to a standalone fleet machine. Defaults to read-only
# registry mode so multiple machines can evaluate in parallel without racing Elo tag writes.
#   make leaderboard-fleet MACHINE=mn-exjk9p93n75h GAMES=20
leaderboard-fleet:
	if [ -z "$(MACHINE)" ]; then echo "MACHINE is unset (e.g. MACHINE=omarchyl)" && exit 1; fi
	uv run kego leaderboard --target $(MACHINE) --games $(GAMES) --task $(TASK) --search-count $(SEARCH_COUNT) $(FLEET_WORKERS_FLAG) $(DEBUG_FLAG) $(FLEET_RATING_FLAG) $(CACHE_FLAG) --partial-save-every $(PARTIAL_SAVE_EVERY)

# Show the latest saved league matrix without running games. Pass RUN_ID=<mlflow-run>
# to inspect a specific run.
leaderboard-show:
	uv run kego leaderboard-show --task $(TASK) $(if $(RUN_ID),--run-id $(RUN_ID),)

# Merge saved read-only leaderboard runs into one matrix. Set WRITE_RATINGS=1 to
# perform the single authoritative Elo write from the merged games.
leaderboard-merge:
	uv run kego leaderboard-merge --task $(TASK) $(if $(RUN_IDS),--run-ids $(RUN_IDS),) $(if $(LATEST),--latest $(LATEST),) $(RATING_FLAG)

# Automatically identify, download, configure, and submit the current leaderboard leader model.
submit-leader:
	uv run python submit_leader.py

.PHONY: train-agent train-agent-cluster train-agent-fleet ray-head ray-status ray-stop models leaderboard leaderboard-fleet leaderboard-show leaderboard-merge submit-leader
