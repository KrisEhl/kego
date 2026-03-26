# Quantum Rimay Workflow - Usage Guide

## Overview

This script provides a professional workflow for quantum feature extraction using the Kipu Quantum Rimay API, with manual approval checkpoints and comprehensive model evaluation.

## Key Features

✅ **Two Manual Approval Checkpoints**
- Approve input datapool (with your data)
- Approve output datapool + job configuration

✅ **Comprehensive Model Evaluation** (4 experiments)
1. Raw features only (all 13)
2. Ablation-pruned classical features (13 raw + 11 engineered)
3. Quantum features only
4. Combined ablation-pruned + quantum

✅ **Flexible Execution Modes**
- Full workflow with approvals
- Skip quantum (classical only)
- Resume from existing metadata
- Manual datapool evaluation

## Usage Patterns

### 1. Test Classical Features (Fastest, ~10 seconds)

Skip quantum computation and only evaluate classical features:

```bash
uv run python quantum_rimay_workflow.py --skip-quantum
```

**Flow:**
- Load data
- Show input datapool (press ENTER)
- Train models on classical features
- Show comparison: Raw (13 feats) vs Ablation (21 feats)

**When to use:** Before submitting quantum jobs, verify your data is correct

---

### 2. Full Workflow with Manual Approvals (Recommended)

Complete workflow with two approval checkpoints:

```bash
uv run python quantum_rimay_workflow.py
```

**Flow:**
1. ✅ Checkpoint 1: Approve INPUT datapool (your data)
   - Shows: Datapool ID, file size
   - Press ENTER to continue

2. ✅ Checkpoint 2: Approve BOTH datapools + config
   - Shows: Input ID, Output ID, num_shots, num_runs
   - Press ENTER to submit quantum job

3. ⏳ Auto-waits for quantum computation (~30-60s)
4. 📊 Evaluates all 4 model configurations
5. 💾 Saves: metadata.json, dataset.json, quantum features

**When to use:** Production runs, want full control

---

### 3. Download Existing Results (No Manual Approval)

From a previous run with saved metadata:

```bash
uv run python quantum_rimay_workflow.py --skip-submit
```

Downloads quantum features from output datapool and evaluates.

**When to use:** Job already submitted and completed

---

### 4. Evaluate Results from Specific Datapool

If you have a datapool ID from elsewhere (e.g., from notebook):

```bash
uv run python quantum_rimay_workflow.py --evaluate-only \
  --output-datapool-id d6c98f7a-ba0e-49d2-94ee-732150dfa9e5
```

**When to use:** Manually specifying datapool (from run.ipynb, etc.)

---

### 5. Skip to Download (Quick Resume)

Already submitted job and now want to download:

```bash
uv run python quantum_rimay_workflow.py --evaluate-only
```

Loads datapool ID from saved metadata.json.

**When to use:** Job completed, just download results

---

## Command-Line Options

```
--skip-quantum              Skip quantum computation, test classical only
--evaluate-only             Only download and evaluate (no upload/submit)
--skip-upload               Skip data upload, reuse existing metadata
--skip-submit               Skip job submission, download from existing
--output-datapool-id ID     Specify output datapool manually
--num-shots N               Measurement shots (default: 100)
--num-runs N                Independent runs (default: 1)
```

## Setup Requirements

Set these environment variables:

```bash
export PLANQK_ACCESS_TOKEN="your_token"
export PLANQK_ORGANIZATION_ID="your_org_id"
export PLANQK_CONSUMER_KEY="your_key"
export PLANQK_CONSUMER_SECRET="your_secret"
export PLANQK_SERVICE_ENDPOINT="https://..."
```

Or add to `.env` file.

## Output Files

```
results/
├── dataset.json              # Your data (13 raw features)
├── rimay_metadata.json      # Execution tracking (IDs, timestamps)
├── rimay_execution_logs.txt # Error details (if job fails)
└── quantum_output/          # Quantum features (if successful)
    ├── 1_Xq_train_0.npy
    ├── 1_Xq_validation_0.npy
    ├── 1_yq_train_0.npy
    └── 1_yq_validation_0.npy
```

## Example Session

```bash
# Test classical features first
$ uv run python quantum_rimay_workflow.py --skip-quantum
[...]
Exp 1: Raw Features Only (all 13)
Test AUC: 0.95742

Exp 2: Ablation-Pruned Features (21 feats)
Test AUC: 0.94813

Delta: -0.00929  (raw is better!)

# Now submit quantum job
$ uv run python quantum_rimay_workflow.py
[...]
Input Datapool:  405f9bb9-a205-41c0-bcc3-aeeff57f9457
Press ENTER to continue...

Output Datapool: 8712f345-b9d3-4e6a-a8f2-8b7c3d4e5f6a
Press ENTER to submit...

Waiting for completion...
✓ Execution completed in 42.3s

Downloading quantum features...
Run 0: train=(700, 5), test=(300, 5)

Exp 3: Quantum Features Only
Test AUC: 0.84521

Exp 4: Ablation-Pruned + Quantum
Test AUC: 0.96145  ← BEST!

Delta vs baseline: +0.00403
```

## Troubleshooting

### "No metadata found"
You need to run the full workflow first, or provide `--output-datapool-id`:
```bash
uv run python quantum_rimay_workflow.py --evaluate-only --output-datapool-id <ID>
```

### Job fails with no logs
Check `rimay_execution_logs.txt` in results/ folder.
Most common: Data format mismatch or quota exceeded

### Want to retry with different params
```bash
# Use existing data, different shots
uv run python quantum_rimay_workflow.py --skip-upload --num-shots 500
```

## Key Differences from test_quantum_features.py

| Feature | quantum_rimay_workflow.py | test_quantum_features.py |
|---------|---------------------------|-------------------------|
| Quantum Backend | Rimay API (production) | Local numpy simulator |
| Approvals | 2 checkpoints (data + job) | None (automatic) |
| Resumability | Via saved metadata | Manual script modification |
| Evaluation | 4 configurations | 5 configs + ablation |
| Data encoding | Automatic str→int | Manual handling |
| Error logging | Saves result object | Limited |

## Next Steps

1. **First run:** Test with `--skip-quantum` to verify data
2. **Submit job:** Run full workflow `quantum_rimay_workflow.py`
3. **Monitor:** Check console output or use Rimay dashboard with execution ID
4. **Evaluate:** Results auto-download and compare at end

---

**Questions?** Check the inline comments in the script or test with `--skip-quantum` first.
