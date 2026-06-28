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

.PHONY: train-agent train-agent-cluster
