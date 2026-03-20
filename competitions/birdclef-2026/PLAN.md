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

### Current best: LB 0.782

Gap to public SED baseline (0.862) = ~0.08. Most likely cause: augmentation (SpecAugment freq/time masking).

### Results

| Version | LB | Notes |
|---|---|---|
| EfficientNet-B0 plain, 30 epochs | 0.758 | 5-fold, precomputed spec cache |
| EfficientNet-B3 plain, 50 epochs | 0.776 | 5-fold, val_loss=0.031 |
| EfficientNet-B3 naive SED, 50 epochs | 0.750 | worse — too few temporal frames (~10) |
| **B0 NoisyStudent + GEMFreqPool + AttentionSED, 50ep** | **0.782** | n_mels=224, specs_cache_224/, minmax norm |

### Architecture comparison vs public baseline

| Factor | Us | Public baseline (0.862) |
|---|---|---|
| Backbone | tf_efficientnet_b0.ns_jft_in1k | same |
| Head | GEMFreqPool + AttentionSED | same |
| **SpecAugment** | ❌ none | ✅ likely freq+time masking |
| Mixup | ✅ | ✅ |
| Training epochs | ~30–50 (early stopping) | unknown |

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

---

## Next steps (ordered by expected impact)

### Step 1: SpecAugment — expected +0.03–0.05

Most likely cause of gap to public baseline (0.862). Add frequency masking + time masking to spectrogram during training:
- `FreqMask`: zero out F consecutive mel bins (e.g. F=27, 2 masks)
- `TimeMask`: zero out T consecutive time frames (e.g. T=100, 2 masks)
- Apply after mel spec computation, before normalization
- Standard in all top audio classification solutions

### Step 2: Secondary label handling

Currently secondary labels are soft (0.5). Try hard positives at 1.0 (simpler, often better).

### Step 3: Pretrained audio backbone

Use weights pretrained on bird/environmental audio rather than ImageNet. Options investigated:

**Option A — PANNs CNN14 (recommended first try)**
- Pretrained on Google AudioSet (2M clips, 527 classes incl. many bird/nature sounds)
- CNN architecture → fast CPU inference (critical for Kaggle scoring)
- Same mel spectrogram interface as current pipeline; spec cache needs recompute (different mel params)
- Used by BirdCLEF 2025 winner (ConvNeXt + AudioSet pretraining)
- Install: `pip install panns-inference` or load via timm

**Option B — DBD-research-group/AST-BirdSet-XCM (HuggingFace)**
- AST (Audio Spectrogram Transformer) fine-tuned on BirdSet (large Xeno-Canto subset — same source as our data)
- 86M params, HuggingFace Transformers compatible
- **Risk**: Transformer CPU inference is 5–10× slower than CNN → may blow 90-min Kaggle scoring limit
- Worth benchmarking inference speed before committing

**Option C — BirdNET-Analyzer**
- Cornell Lab classifier trained on ~6000 bird species
- Harder to integrate (different SR/mel params, proprietary training data)
- Lower priority

**Decision rule**: Test inference speed on one soundscape file before choosing. If AST processes <2s/soundscape on CPU, it's viable. Otherwise, PANNs CNN14.

### Step 5: Ensemble

Once multiple backbones/configs trained:
- Average sigmoid outputs across folds and backbones
- Greedy ensemble selection on OOF cmAP

---

## Hardware

Training on **2× RTX 3090** at `kristian@omarchyd` (Tailscale) / `kristian@omarchyd.fritz.box` (LAN).
Spec cache at `data/birdclef/birdclef-2026/specs_cache/` (~10.5GB, precomputed).

```bash
ssh kristian@omarchyd
cd ~/projects/kego
git pull
CUDA_VISIBLE_DEVICES=0 uv run python competitions/birdclef-2026/train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 uv run python competitions/birdclef-2026/train.py --fold 1 &
```

---

## Key references

- [SED Baseline LB 0.862](https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862)
- [PyTorch Baseline Inference](https://www.kaggle.com/code/antoinemasq/birdclef-2026-pytorch-baseline-inference)
- [BirdCLEF 2025 1st place](https://www.kaggle.com/competitions/birdclef-2025/discussion) — ConvNeXt + AudioSet pretraining
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer)

---

## Dead ends / lessons learned

- `enable_gpu: true` in kernel-metadata.json → silent scoring failure (competition GPU limit = 0 min)
- OOF cmAP (clip-level) ≠ LB cmAP (soundscape-level) — expect a large gap; don't over-optimize OOF
- Plain classifier head underperforms SED head significantly for soundscape-level scoring
- Naive SED head (mean freq pool + sigmoid attention): 0.750 — worse than plain B3 (0.776) due to only ~10 temporal frames after 32× backbone downsampling
- Longer training beyond ~50 epochs doesn't help — val loss plateau is flat; early stopping with patience=15 is appropriate
- Our architecture matches public baseline exactly (same backbone, GEMFreqPool, AttentionSED) — gap (0.782 vs 0.862) is likely SpecAugment
