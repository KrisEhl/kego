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
GAMES        ?= 4
SEARCH_COUNT ?= 10
WORKERS      ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DEBUG        ?= 0

# Toggle verbose logging based on DEBUG
DEBUG_FLAG =
ifeq ($(DEBUG),1)
  DEBUG_FLAG = --debug
endif

leaderboard:
	uv run python run_league.py --games $(GAMES) --task $(TASK) --search-count $(SEARCH_COUNT) --workers $(WORKERS) $(DEBUG_FLAG)

# Automatically identify, download, configure, and submit the current leaderboard leader model.
submit-leader:
	uv run python submit_leader.py

.PHONY: train-agent train-agent-cluster train-agent-fleet ray-head ray-status ray-stop models leaderboard submit-leader
