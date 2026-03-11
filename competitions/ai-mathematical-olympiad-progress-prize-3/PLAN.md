# Plan: AIMO Progress Prize 3 — Maximize LB Score

Competition deadline: **2026-04-15**. Submission limit: **1 per day**, 1 final submission.
Prize pool: $2.2M. Problems: national olympiad → IMO level, integer answers 0–999,999.

---

## Status

### Current best

**v16 submitted to competition grader — COMPLETE, score pending (AIMO3 doesn't show public scores via CLI).**

- Model: QwQ-32B AWQ via vLLM
- N=16 samples, majority vote
- TIR loop: up to 6 steps, but MAX_TOKENS was too high → effectively 1-2 steps (fixed in v17)
- 2× H100 on Kaggle

**v17 regular run complete — ready for tomorrow's submission.**

- Same as v16 but: MAX_TOKENS=(16384-2000)//6=2397 for proper 6-step TIR
- Also: vLLM wait loop exits immediately on process crash (no 10-min wait on P100)

### Local baseline (QwQ-32B AWQ, N=4, maj@4, AIME 2025)

**8/30 = 26.7%** — reference point for all future local comparisons.

Correct: aime25_0 (70), aime25_2 (16), aime25_3 (117), aime25_5 (504), aime25_7 (77), aime25_15 (468), aime25_16 (49), aime25_18 (106).

Pattern on misses: many problems where all 4 samples confidently agree on the wrong answer — not a voting problem, the model is genuinely wrong. More samples (N=16) would help marginally but not fix fundamental model capability gaps.

### Submissions

| Version | LB Score | Local maj@4 | Notes |
|---|---|---|---|
| v11 | failed | — | Bug: `KAGGLE_IS_COMPETITION_RERUN` never set → ran mock server, submitted all-42 |
| v12 | ERROR | — | Bug: TENSOR_PARALLEL=2 hardcoded, Kaggle regular kernel only has 1 GPU |
| v13/v14 | ERROR | — | P100 sm_60 incompatibility: vLLM fails → serve() never runs → regular run ERROR |
| v15 | ERROR | — | Same P100 issue — serve() called but placeholder parquet missing |
| v16 | COMPLETE (no score shown) | — | Fix: graceful vLLM fallback (VLLM_STARTED flag), placeholder parquet written first |
| v17 | ready to submit | — | Fix: MAX_TOKENS=2397/step + early vLLM crash detection |

---

## Local Eval Setup

Hardware: 2× RTX 3090 (48GB VRAM total), Linux.

### Inference backend

**LMDeploy TurboMind** — ~1.8× faster than vLLM for quantized models. Used by AIMO-2 2nd place.

```bash
pip install lmdeploy

lmdeploy serve api_server nvidia/OpenReasoning-Nemotron-32B \
  --tp 2 --quant-policy 4 --kv-quant-policy 8 --server-port 8000
```

### Eval datasets

| Dataset | Problems | Use |
|---|---|---|
| `MathArena/aime_2025` | 30 | **Primary fast eval** — best proxy for AIMO PP3 difficulty |
| `HuggingFaceH4/aime_2024` | 30 | Secondary eval, combined with 2025 for 60-problem suite |
| AMC12 integer subset (2022+2023) | ~83 | Lower-difficulty signal, used by AIMO-1 winners |
| `KbsdJames/Omni-MATH` | 4428 | Large-scale signal when 60 problems is too noisy |
| Competition reference set | 10 | Calibration to actual LB |

### Eval protocol

```
Fast (exploratory):    N=4,  30 problems (AIME 2025)    → ~5 min
Medium (comparison):   N=8,  60 problems (AIME 24+25)   → ~20 min
Pre-submission:        N=16, 90+ problems (AIME+AMC)    → ~50 min
```

Only submit when **pre-submission eval shows consistent +2–3 problems on 90+ problems with N=16**.

### Calibration

After each Kaggle submission: compute `kaggle_score / local_maj@N`. Track ratio over time. If local eval becomes too easy (high ratio), switch to harder problems.

---

## Next Steps (ordered by expected impact)

### Step 1: Switch to OpenReasoning-Nemotron-32B ✓ IN PROGRESS

Downloaded: `abhishekchohan/OpenReasoning-Nemotron-32B-W4A16` (19.2GB, compressed-tensors W4A16)
Path: `/home/kristian/projects/kego/models/OpenReasoning-Nemotron-32B-W4A16`

**vLLM settings that work (2× RTX 3090):**
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve .../OpenReasoning-Nemotron-32B-W4A16 \
  --port 8080 --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 --trust-remote-code --disable-log-requests
```
- W4A16 takes ~21GB on single GPU → must use TP=2 (2× 3090, 48GB total)
- `--max-num-seqs 32` needed to avoid OOM in CUDA graph warmup (default=256 OOM)
- `--enforce-eager` works but is ~5× slower — avoid it
- With `max_model_len=16384`: use `MAX_TOKENS=2500` for 6-step TIR (context fills after 2 steps with MAX_TOKENS=7500)

**Final result** (2026-03-11, N=4, AIME 2025): **3/30 = 10.0%** — significantly worse than QwQ-32B (8/30 = 26.7%).
- W4A16 quantization severely degrades capability at this level.
- **Decision: abandon Nemotron W4A16, keep QwQ-32B as base model.**

### Step 1: STATUS — CLOSED (Nemotron W4A16 not viable)

### Step 2: Set up local eval harness

Build `eval.py` that:
- Loads AIME 2024/2025 from HuggingFace datasets
- Calls local inference server (LMDeploy / vLLM)
- Runs TIR loop (same logic as notebook `solve_with_voting`)
- Reports pass@1 and maj@N per problem + aggregate accuracy
- Saves results to CSV for comparison

### Step 3: GenSelect aggregation (replace majority vote)

**Highest-leverage improvement from AIMO-2 research.** Instead of majority vote, prompt the model to read N candidate solutions and select the best one. The AIMO-2 1st place team (NVIDIA) used GenSelect and it was their key differentiator.

- Paper: [arXiv:2602.02143](https://arxiv.org/abs/2602.02143)
- Implementation: prompt the same model with all N solutions, ask it to output the index of the most reliable one
- Test on AIME 2025 (N=8): compare GenSelect vs majority vote accuracy
- If +2 problems locally → worth a submission

### Step 4: Mixed CoT + TIR sampling

Instead of all N samples using TIR, use a mix:
- 50% pure chain-of-thought (no code execution)
- 50% TIR (code execution enabled)

AIMO-2 2nd place team used 7 CoT + 8 code-based per problem. Rationale: different solution styles catch different problem types; diversity > homogeneity.

### Step 5: Increase N and context length on Kaggle

The H100s are much faster than local 3090s. Can afford:
- N=32 instead of N=16 (majority vote improves)
- `max_model_len=16384` or `32768` (hard IMO problems need long reasoning chains)
- `max_tokens=8192` per sample (currently 4096)

Test locally with N=8 vs N=16 on AIME 2025 to measure the gain before burning a submission.

### Step 6: Early stopping

Stop sampling a problem when consensus is clear:
- If 5/8 samples agree → accept answer, move to next problem (saves tokens for hard problems)
- Question-level: skip problems already solved with high confidence, focus budget on uncertain ones

AIMO-2 2nd place used this to stay within the 9-hour time limit at high N.

### Step 7: OpenMath-Nemotron-32B as alternative

`nvidia/OpenMath-Nemotron-32B` — 540K training problems, 1.7M TIR solutions. AIME 2024 maj@64 = 93.3%. TIR native, CC-BY-4.0.

Compare to OpenReasoning-Nemotron-32B on local eval. They may be complementary (ensemble both on Kaggle).

### Step 8: Verification-and-refinement pipeline

Instead of just selecting the best answer, use the model to:
1. Generate N solutions
2. For each solution, verify by re-reading and checking the logic
3. If a solution fails verification, generate a revised solution
4. Aggregate over verified solutions

This approach raised baseline accuracy from ~35% to 85.7% on IMO 2025 problems (arXiv:2507.15855). High effort but potentially very high reward.

---

## Already Tried / Dead Ends

- **`run_local_gateway` in dev mode**: `*data_batch` unpacks polars DataFrame to column names → predict called with 2 string args instead of 1 DataFrame. Fixed by bypassing it entirely (direct loop over `iter_slices`).
- **`TENSOR_PARALLEL=2` hardcoded**: Regular Kaggle kernel environment can give 1 or 2 GPUs. v12 errored because only 1 GPU was available (T4/P100) but TP=2 was hardcoded. Fix: detect GPU count with `nvidia-smi`. Note: Kaggle code competitions need the kernel to have completed a run (produced the output file) BEFORE you can submit it to the competition grader.
- **Submission 400 "Notebook is still running"**: You can't submit a kernel version to the competition grader until it has finished its regular Kaggle run. Wait for `KernelWorkerStatus.COMPLETE` or `ERROR`.
- **Regular Kaggle kernel has Tesla P100 (sm_60)**: PyTorch >= 2.0 requires sm_70+. vLLM fails to start on P100. If Cell 3 raises RuntimeError, serve() never runs → regular run ERROR → Kaggle refuses competition submission. Fix: wrap vLLM startup in try-except, set `VLLM_STARTED=False`, have predict() return 42, call serve() — it uses local test data in non-competition mode.
- **OpenReasoning-Nemotron-32B-W4A16 is worse than QwQ-32B**: Full eval (N=4, all 30 AIME 2025 problems) = **3/30 = 10.0%** vs QwQ **8/30 = 26.7%**. W4A16 quantization severely degrades capability. Stick with QwQ-32B on Kaggle. The full BF16 model would require >48GB VRAM (out of scope).
- **MAX_TOKENS=12288 on 2× H100 fills context after 1 step**: With MAX_MODEL_LEN=16384, setting MAX_TOKENS=12288 uses 12288+300=12588 tokens after step 1. Step 2 gets only 16384-12588-200=3596 tokens. Step 3 context full → effectively 2-step TIR instead of 6. Fixed: MAX_TOKENS=(MAX_MODEL_LEN-2000)//MAX_TIR_STEPS=2397 for 6 proper steps.
- **Submissions table**: v11-v15 all failed (various bugs). v16 COMPLETE on grader (score not shown by CLI — AIMO3 may not expose public scores). v17 ready for tomorrow.

---

## Key Insights from AIMO-2 Winners

1. **TIR is non-negotiable**: every top solution uses code execution (not pure CoT)
2. **Aggregation matters more than N**: GenSelect >> majority vote at equal compute
3. **Mixed trajectories**: CoT + TIR diversity > all-TIR
4. **Model quality dominates**: OpenReasoning-Nemotron-32B at N=8 likely beats QwQ-32B at N=32
5. **AIME 2025 as proxy**: best correlation with AIMO PP3 private test set difficulty
6. **Longer context**: hard problems need 16K+ token reasoning chains; 4096 is too short for IMO level

## Reference

- [AIMO-2 1st place writeup (NVIDIA/NemoSkills)](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills) — arXiv:2504.16891
- [AIMO-2 2nd place code](https://github.com/imagination-research/aimo2)
- [AIMO-1 1st place (Project Numina)](https://github.com/project-numina/aimo-progress-prize)
- [GenSelect paper](https://arxiv.org/abs/2602.02143)
- [OpenReasoning-Nemotron blog](https://huggingface.co/blog/nvidia/openreasoning-nemotron)
- [Verification-and-refinement for IMO 2025](https://arxiv.org/abs/2507.15855)
