# Plan: BirdCLEF+ 2026

Competition: multilabel bird species identification from passive acoustic monitoring (PAM)
recordings in the Pantanal wetlands, South America.

- **Metric**: cmAP (class-mean average precision)
- **Prize**: $50,000
- **Deadline**: May 27, 2026
- **Data**: ~15GB audio (.ogg). 35,549 labeled training clips.
- **Task**: Given 5s soundscape chunk, predict which of 234 species are present (multilabel)

### Data facts (confirmed from EDA)

| Property | Value |
|---|---|
| Training CSV | `train.csv` |
| Training recordings | 35,549 |
| Target species | **234** (from taxonomy.csv) |
| Species with training data | **206** (28 zero-shot, mostly sonotype splits `47158son*`) |
| Primary label format | Mixed: numeric iNat taxon ID OR eBird code (e.g. `ashgre1`) |
| Filename format | Includes subdirectory: `{taxon_id}/{filename}.ogg` |
| Class breakdown | Aves 93%, Amphibia 5%, Insecta 1%, Mammalia <1%, Reptilia <1% |
| Recordings/species | min=1, median=125, max=499 |
| Secondary labels | 4,372 recordings (12%) have secondary species |
| Test row IDs | `{soundscape_stem}_{end_second}` at 5, 10, 15, ... (5s non-overlapping chunks) |

---

## Status

### Current best: LB 0.783

Gap to public SED baseline (0.862) = ~0.08. Investigating root cause — see "Public baseline analysis" below.

### Results

| Version | LB | Notes |
|---|---|---|
| EfficientNet-B0 plain, 30 epochs | 0.758 | 5-fold, 128-mel |
| EfficientNet-B3 naive SED, 50 epochs | 0.750 | too few temporal frames |
| EfficientNet-B3 plain, 50 epochs | 0.776 | 5-fold, 128-mel |
| **B0 NoisyStudent + GEMFreqPool + AttentionSED, 50ep** | **0.782** | 224-mel, minmax norm, time_mask=100 |
| B0 baseline, early stopping patience=15 | 0.783 | same arch, time_mask=100, no dual loss |
| B0 baseline + dual loss + time_mask=30 (all 5 folds) | 0.765 | **REGRESSION** — see dead ends |
| B0 baseline + dual loss + hard secondary labels | training | GPU0: fold0→2→4, GPU1: fold1→3 |
| EfficientNet-B1 BirdSet XCL pretrained (5-fold) | 0.782 | 256-mel, 1-channel, patience=10 — no gain over B0 NoisyStudent |

### Local validation findings

**Neither clip OOF nor soundscape cmAP reliably predicts LB ranking.** B3-SED has highest local scores but lowest LB (0.750). Root cause: the 66 train soundscapes only cover 75/234 species — not representative of test set. **Use LB submissions as ground truth; local metrics only useful as coarse sanity checks.**

### Public baseline analysis (LB 0.862)

Pulled `aidensong123/birdclef-2026-sed-baseline-lb-0-862` to inspect the inference notebook. Key findings:

- **Architecture is identical** to ours: tf_efficientnet_b0.ns_jft_in1k + GEMFreqPool + AttentionSED
- **Only 1 fold checkpoint** — 0.862 comes from a single-fold model, not a 5-fold ensemble
- **Checkpoint named `best_perch_fold0.pt`** from dataset `aidensong123/perch-fold` — strongly suggests **Perch-based pretraining or pseudo-labeling** (Google's AudioSet bird classifier)
- Checkpoint has a `stage` field (checked via `ckpt.get("stage", "supervised")`), indicating a multi-stage training pipeline
- **The gap (0.862 vs 0.783) is almost certainly from Perch pretraining, not architecture**

| Factor | Us | Public baseline (0.862) |
|---|---|---|
| Backbone | tf_efficientnet_b0.ns_jft_in1k | same |
| Head | GEMFreqPool + AttentionSED | same |
| Dual loss | ✅ | likely ✅ |
| SpecAugment | freq=30×2, time=30×2 | likely same |
| Folds submitted | 5-fold ensemble | **1 fold only** |
| **Pretraining** | ImageNet only | **Perch / AudioSet pseudo-labels** |

### Kaggle setup

- **Notebook**: `aldisued/birdclef-2026-baseline-inference` (v1)
- **Dataset**: `aldisued/birdclef2026-baseline` (5× fold checkpoints, ~125MB)
- **CRITICAL**: `enable_gpu: false` — competition GPU limit is 0 min; GPU requests cause silent failure
- Kaggle mounts data at `/kaggle/input/competitions/birdclef-2026/` and datasets at `/kaggle/input/datasets/{owner}/{slug}/`; notebook auto-detects both via glob

### Submit command
```bash
cd competitions/birdclef-2026
kaggle kernels push
kaggle competitions submit -c birdclef-2026 \
  -k aldisued/birdclef-2026-baseline-inference \
  -f submission.csv -v <VERSION> -m "<description>"
```

---

## What's done

- [x] EDA + data setup (`analyze_data.py`, plots in `plots/`)
- [x] Training pipeline (`train.py`): EfficientNet-B0/B3, 5-fold, early stopping, mixup, label smoothing
- [x] Precomputed spec cache: 128-mel `specs_cache/` (~10.5GB) + 224-mel `specs_cache_224/`
- [x] OOF evaluation (`eval_oof.py`): cmAP + ROC-AUC per fold
- [x] Kaggle inference notebook with CPU-only sliding-window inference, auto-detects model type from checkpoint
- [x] `BirdModelBaseline`: GEMFreqPool + AttentionSED head, NoisyStudent backbone — LB **0.782**
- [x] Early stopping (`--patience`, default 10) — models converge around epoch 25–50
- [x] Dead end: naive SED head (0.750) — too few temporal frames; replaced by proper GEMFreqPool+AttentionSED
- [x] Dual frame+clip loss: `0.5 * clip_BCE + 0.5 * frame_BCE`
- [x] SpecAugment fix: time_mask=30 frames (was 100)
- [x] Retrain all 5 folds with dual loss + time_mask=30 → LB **0.765** (regression vs 0.783)
- [x] Analyzed public baseline — gap explained by Perch pretraining, not architecture
- [~] Hard secondary labels — incomplete (folds 0–1 only), never submitted. Dead end.
- [x] **BirdModelBirdSet**: EfficientNet-B1 pretrained on BirdSet XCL (9,736 species via HuggingFace). All 5 folds trained: best val losses 0.0316–0.0324, stopped ep20–55. Checkpoints at `aldisued/birdclef2026-birdset`. LB **0.782** (no gain over B0 NoisyStudent).
- [x] CosineAnnealingLR fix: T_max was `epochs=200` (effectively disabled). Changed to `T_max=patience*3` + `eta_min=1e-5` for future runs.
- [x] Patience default lowered to 10 (was 15)
- [x] `pseudo_label_perch.py`, `pseudo_label_birdnet.py`, `pseudo_label_self.py` written. Pseudo-label + background noise wired into `train.py`.
- [ ] **Perch pseudo-label generation** — RUNNING (Mar 21, ~18:00). Processing 10,658 train soundscapes in 5s windows on GPU (~25 min). Output: `perch_pseudo_labels.csv` + `perch_pseudo_labels_soft.npz`.
  - **TF GPU requires `LD_LIBRARY_PATH`** — the bundled nvidia-* CUDA wheels (from `tensorflow[and-cuda]`) are not auto-discovered. Must set manually before running:
    ```bash
    VENV=~/projects/kego/.venv
    NV=$VENV/lib/python3.13/site-packages/nvidia
    export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/nccl/lib
    KEGO_PATH_DATA=~/projects/kego/data .venv/bin/python competitions/birdclef-2026/pseudo_label_perch.py
    ```
  - `uv run python` does not work for TF scripts — use `.venv/bin/python` directly.

---

## Next steps (ordered by expected impact)

> Full literature review in `research.md`. The 0.079 gap to public baseline is almost entirely from Perch pseudo-labeling. Transformer models (BEATs, AudioMAE, AST) are infeasible for CPU-only inference.

### ✅ DONE

- **Dual loss + time_mask=30** → LB 0.765 (regression)
- **BirdModelBirdSet** (EfficientNet-B1, BirdSet XCL) → LB 0.782 (no gain)
- **Hard secondary labels** — dead end (never fully trained or submitted)

### 🔄 IN PROGRESS

- **Perch pseudo-label generation** (Step 5 prerequisite) — running on `omarchyd` since Mar 21 ~14:00. ETA 6–14h on CPU. Check: `ssh kristian@omarchyd "tail -5 ~/projects/kego/competitions/birdclef-2026/perch_pseudo_label.log"`

---

### Step 4 — P1: Per-species threshold calibration (free gain, do now)

Expected: **+0.005–0.015 LB**. Grid search sigmoid threshold per class on OOF predictions.
Zero retraining cost. Should be done before and after every model improvement.

---

### Step 5 — P0: Perch Pseudo-Labeling Pipeline (highest impact)

Expected: **+0.05–0.08 LB**. This is what the public baseline (0.862) does.

**Pipeline** (BirdCLEF 2024 1st place, `arpoyda`):
1. Run `google/bird-vocalization-classifier` v4 on all 35,549 training clips → soft labels (n_clips × 234)
2. Map Perch's 10,000+ species to our 234 via eBird codes / taxonomy
3. Pretraining stage: EfficientNet-B0 on Perch soft labels, 5–10 epochs
4. Fine-tuning stage: ground-truth hard labels, 20–30 epochs
5. All 5 folds → submit

**Resources**:
- Kaggle Models: `google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4`
- GitHub: https://github.com/google-research/perch
- Reference: https://github.com/arpoyda/BirdCLEF_2024
- Perch 2.0 (arXiv:2512.03219): 14,597 species incl. Pantanal birds + 28 zero-shot species

**CPU inference**: Perch uses EfficientNet backbone → TFLite gives ~10x speedup; runs ~16 min on Kaggle CPU (confirmed in BirdCLEF 2025 paper arXiv:2507.08236).

---

### Step 6 — P1: Multi-year BirdCLEF data pretraining

Expected: **+0.01–0.025 LB**. BirdCLEF 2021/2022/2023/2024 datasets on Kaggle (all public).
Pretrain on all available bird audio → fine-tune on BirdCLEF 2026. Used by multiple top-4 solutions.

---

### Step 7 — P1: PANNs CNN14 (AudioSet pretrained)

Expected: **+0.015–0.03 LB**. Easier alternative/complement to Perch.
- Weights: `Cnn14_mAP=0.431.pth` from Zenodo record 3960586
- GitHub: https://github.com/qiuqiangkong/audioset_tagging_cnn
- CNN-based → fast CPU inference. Replace output layer (527→234), fine-tune.

---

### Step 8 — P2: Background noise augmentation

Expected: **+0.005–0.015 LB**. Mix "no-call" segments from competition soundscapes (p=0.3–0.5).
Helps domain adaptation from XC clips to passive soundscapes. Used by BirdCLEF 2023 1st place.

---

### Step 9 — P2: BirdNET for zero-shot species + ensemble

- 28 zero-shot species currently output 0 → drag on cmAP
- BirdNET covers 6,512 species (`pip install birdnet-analyzer`); likely covers all 28
- Ensemble: Perch-pretrained B0 + original baseline B0 + CNN14 for diversity

---

### Step 10 — P3: Temporal smoothing + SoftAUCLoss (if plateauing ≥0.84)

- Adjacent segment smoothing: `0.85 * current + 0.15 * avg(neighbors)` (BirdCLEF 2025 4th place)
- SoftAUCLoss: directly optimizes cmAP gradient (reference: `dylanliu2` BirdCLEF 2025)

---

## Hardware & Training Convention

Training on **2× RTX 3090** at `kristian@omarchyd` (Tailscale) / `kristian@omarchyd.fritz.box` (LAN).
Spec cache at `data/birdclef/birdclef-2026/specs_cache/` (~10.5GB, precomputed).

### Checkpoint naming — always use `--tag`

Checkpoints are saved to `competitions/birdclef-2026/outputs/{tag}_fold{N}.pt`.
**Always pass `--tag <experiment-name>` to prevent overwriting previous runs.**

```bash
ssh kristian@omarchyd
cd ~/projects/kego
git pull

# Example: BirdSet v2 run
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python competitions/birdclef-2026/train.py \
  --birdset --fold 0 --tag birdset-v2 &
CUDA_VISIBLE_DEVICES=1 ~/.local/bin/uv run python competitions/birdclef-2026/train.py \
  --birdset --fold 1 --tag birdset-v2 &
```

The checkpoint stores all hyperparams: `epoch`, `val_loss`, `backbone`, `n_mels`, `n_fft`, `hop_length`,
`hard_labels`, `dual_loss`, `freq_mask`, `time_mask`, `patience`, `lr`, `seed`, `fold`, `tag`.

### Inspect a checkpoint

```bash
cd ~/projects/kego
~/.local/bin/uv run python3 -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu')
for k, v in ck.items():
    if k != 'model': print(f'  {k}: {v}')
" competitions/birdclef-2026/outputs/mytag_fold0.pt
```

### Experiment log

| Tag | Args | Folds | LB | Notes |
|---|---|---|---|---|
| *(legacy, no tag)* | `--baseline`, patience=15, time_mask=100 | 0–4 | **0.783** | **Current best** |
| *(legacy, no tag)* | `--baseline --hard-labels`, patience=10 | 0–1 only | — | Mixed/incomplete; never submitted |
| `birdset-v1` | `--birdset`, patience=10 | 0–4 | 0.782 | BirdSet XCL pretrained; no gain |

---

## Key references

- [SED Baseline LB 0.862](https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862)
- [PyTorch Baseline Inference](https://www.kaggle.com/code/antoinemasq/birdclef-2026-pytorch-baseline-inference)
- [BirdCLEF 2025 1st place](https://www.kaggle.com/competitions/birdclef-2025/discussion) — ConvNeXt + AudioSet pretraining
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer)

---

## Dead ends / lessons learned

- `enable_gpu: true` in kernel-metadata.json → silent scoring failure (competition GPU limit = 0 min)
- OOF cmAP (clip-level) ≠ LB cmAP (soundscape-level) — large gap expected; don't optimize OOF
- **Soundscape cmAP also unreliable as LB proxy**: 66 train soundscapes cover only 75/234 species → B3-SED scores highest locally (0.2994) but lowest LB (0.750). Use LB submissions as ground truth.
- Plain classifier head underperforms SED head significantly for soundscape-level scoring
- Naive SED head (mean freq pool + sigmoid attention): 0.750 — worse than plain B3 (0.776) due to only ~10 temporal frames after 32× backbone downsampling
- Longer training beyond ~50 epochs doesn't help — val loss plateau is flat; early stopping with patience=15 is appropriate
- **Dual loss + time_mask=30 regressed LB 0.783 → 0.765**: two changes happened at once (added dual loss, reduced time_mask 100→30). Unclear which caused it. Possible: time_mask=100 provided better regularization despite being "wrong"; or dual loss destabilizes training with patience=15. Not worth isolating — the architecture gap vs public baseline is from Perch pretraining, not these details.
- **Public baseline gap explained**: the 0.862 model uses Perch pseudo-labels or AudioSet pretraining, NOT just architecture/augmentation changes. Checkpoint named `best_perch_fold0.pt` from `aidensong123/perch-fold`; single fold gets 0.862 vs our 5-fold ensemble at 0.783. Architecture is identical.
- **BirdSet XCL pretraining** (9,736 Xeno-Canto species, EfficientNet-B1): LB 0.782 — no gain over B0 NoisyStudent (0.783). Bird-domain pretraining from a generic Xeno-Canto dataset does NOT help. The gap to 0.862 requires competition-specific pseudo-labeling (Perch) or AudioSet pretraining.
