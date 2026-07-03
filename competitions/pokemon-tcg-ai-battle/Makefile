# Makefile for the pokemon-tcg-ai-battle competition.
# Run from this directory (`make train-agent`) or with `make -C <path> train-agent`.
# The task slug is auto-detected from kego.toml, but passed explicitly for clarity.

TASK    ?= pokemon-tcg-ai-battle
EPOCHS  ?= 1
OUTPUT  ?= outputs/mcts.pth

# Train the agent locally (serial executor).
#   make train-agent EPOCHS=10 OUTPUT=outputs/mcts.pth
train-agent:
	uv run kego train-agent --task $(TASK) --epochs $(EPOCHS) --output $(OUTPUT)

# Train on the Ray cluster (submits via the Ray executor, tails logs, scp's weights back).
#   make train-agent-cluster EPOCHS=50
train-agent-cluster:
	uv run kego train-agent --executor ray --task $(TASK) --epochs $(EPOCHS) --output $(OUTPUT)

# Dispatch training to a standalone fleet machine (rsync working tree + SSH-launch detached).
#   make train-agent-fleet MACHINE=m5 EPOCHS=200
# Follow with `kego ls` / `kego logs <run>`. Requires the machine in fleet.toml + the hub up.
train-agent-fleet:
	if [ -z "$(MACHINE)" ]; then echo "MACHINE is unset (e.g. MACHINE=m5)" && exit 1; fi
	uv run kego train-agent --task $(TASK) --target $(MACHINE) --epochs $(EPOCHS) --output $(OUTPUT)

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

.PHONY: train-agent train-agent-cluster train-agent-fleet ray-head ray-status ray-stop
