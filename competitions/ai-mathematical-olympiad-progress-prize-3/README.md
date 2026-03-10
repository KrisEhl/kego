# AI Mathematical Olympiad Progress Prize 3

**Task**: 110 competition math problems, 5-digit integer answers (0–99999).
**Evaluation**: Accuracy (exact match) on hidden test set.
**Hardware**: 2x H100 (80GB each), ~9-hour time limit.

## Approach

Tool-Integrated Reasoning (TIR): interleave LLM reasoning with Python code execution.
Self-consistency / majority voting over N samples.
Drop-in model via `LLM_BASE_URL` env var — same code, different endpoint.

## Local Setup

**Option A: MLX (Apple Silicon, recommended — 20-40% faster than llama.cpp)**
```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/QwQ-32B-4bit --port 8080
# In another terminal:
export LLM_BASE_URL=http://localhost:8080/v1
```

**Option B: Ollama**
```bash
ollama pull qwq:32b   # ~19GB
ollama serve          # starts server on :11434
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwq:32b
```

## Usage

```bash
# Evaluate on 10 reference problems (n=4 voting samples)
uv run python solve.py --mode eval

# Evaluate on first 3 problems with TIR (single deterministic trace)
uv run python solve.py --mode eval --limit 3 --tir --verbose

# Solve a single problem
uv run python solve.py --mode solve --problem "What is 2+2?"

# More voting samples for better accuracy
N_SAMPLES=16 uv run python solve.py --mode eval --limit 5
```

## Config (env vars)

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API endpoint |
| `LLM_MODEL` | `mlx-community/QwQ-32B-4bit` | Model name / path |
| `N_SAMPLES` | `4` | Voting samples per problem |
| `MAX_TIR_STEPS` | `4` | Max code execution iterations |
| `CODE_TIMEOUT` | `15` | Python execution timeout (seconds) |

## Kaggle Models

| Model | Size | VRAM | Notes |
|---|---|---|---|
| OpenReasoning-Nemotron-32B | 32B | ~70GB FP16, ~35GB AWQ | Strong baseline |
| OpenMath-Nemotron-14B-Kaggle | 14B | ~28GB FP16 | AIMO2 winner fine-tune |
| GPT-OSS-120B | 120B | fits 2x H100 via MXFP4 | Strongest known open model |
| QwQ-32B | 32B | ~19GB Q4 locally | Good for local dev |

## Results

| Submit | Model | N | Accuracy (ref) | LB |
|---|---|---|---|---|
| — | — | — | — | — |
