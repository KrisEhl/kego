# Plan: BirdCLEF+ 2026

Competition: multilabel bird species identification from passive acoustic monitoring (PAM)
recordings in the Pantanal wetlands, South America.

- **Metric**: Custom multilabel metric (likely padded-cMAP or ROC-AUC per species, then averaged)
- **Prize**: $50,000
- **Deadline**: May 27, 2026
- **Data**: ~16GB audio (.ogg). 35,549 labeled training clips.
- **Task**: Given 5s soundscape chunk, predict which of 234 species are present (multilabel)

### Data facts (confirmed from EDA)

| Property | Value |
|---|---|
| Training CSV | `train.csv` |
| Training recordings | 35,549 |
| Target species | **234** (from taxonomy.csv) |
| Species with training data | **206** (28 zero-shot, mostly sonotype splits `47158son*`) |
| Primary label format | Mixed: numeric iNat taxon ID OR eBird code (e.g. `ashgre1`) |
| Filename format | Includes subdirectory: `{taxon_id}/{filename}.ogg` → `train_audio/{filename}` |
| Class breakdown | Aves 93%, Amphibia 5%, Insecta 1%, Mammalia <1%, Reptilia <1% |
| Recordings/species | min=1, median=125, max=499 |
| Secondary labels | 4,372 recordings (12%) have secondary species |
| Test row IDs | `{soundscape_stem}_{end_second}` at 5, 10, 15, ... (5s non-overlapping chunks) |

---

## Status

### Current best

None yet. Baseline public SED notebook: **LB 0.862**.

### Local baseline

Not started.

### Submissions

| Version | LB Score | Notes |
|---|---|---|
| — | — | Not yet submitted |

---

## Architecture: Standard BirdCLEF Approach

Every top BirdCLEF solution uses the same core pipeline:

```
Audio clip (5s)
  → Mel-spectrogram (128 mel bins, hop_length=512 @ 32kHz)
  → CNN backbone (EfficientNet-B3/B4 or ConvNeXt-Small)
  → Sigmoid output (one logit per species)
  → BCE loss with label smoothing
  → Test: sliding window over soundscape, max-pool per species
```

---

## Steps (ordered by expected impact)

### Step 1: EDA + data setup

- [ ] Inspect `train_metadata.csv`: species counts, recording durations, geographic spread
- [ ] Check class distribution — long tail expected (common species >> rare)
- [ ] Listen to 5-10 samples per species type; understand audio quality
- [ ] Understand test format: soundscape files → need sliding window inference
- [ ] Check if unlabeled soundscape data is provided (common in BirdCLEF for self-supervised pretraining)

### Step 2: First training baseline (EfficientNet-B0, fast iteration)

Goal: reproduce ~0.85 LB with minimal code.

**Data pipeline:**
- Load training clips; resample to 32kHz
- Crop/pad to 5 seconds
- Compute log-mel spectrogram: n_mels=128, fmin=20, fmax=16000
- Normalize per-image (mean/std from ImageNet)

**Model:**
- `timm.create_model('efficientnet_b0', pretrained=True, num_classes=N_SPECIES)`
- Replace first conv if needed for 1-channel input (or stack spectrogram 3×)

**Training:**
- BCE loss + label smoothing 0.05
- AdamW, cosine LR schedule, 30 epochs
- Stratified 5-fold CV by species (rare species need to appear in val)
- Secondary label handling: treat as soft labels (0.5) not hard positives

**Augmentation (audio-domain):**
- `audiomentations`: AddGaussianNoise, TimeStretch, PitchShift, Shift
- Mixup on spectrograms (works well for BirdCLEF)
- Random power (volume) augmentation

**Inference:**
- Sliding window: 5s chunks with 2.5s overlap over soundscape
- Max-pool predictions across all chunks per soundscape
- Threshold at 0.5 (tune on OOF)

### Step 3: Scale up backbone + training tricks

Once baseline is running:

- **EfficientNet-B3/B4 or ConvNeXt-Small** — typical sweet spot for BirdCLEF
- **Pretrained on AudioSet** via `hear-eval-kit` or `BirdNET` features as additional input
- **Focal loss** — handles class imbalance better than BCE
- **CutMix / SpecAugment** — key augmentation for spectrograms
- **Secondary labels** — use competition-provided secondary labels as soft positives (0.3–0.5)
- **Longer training** (50–100 epochs) with warmup

### Step 4: Self-supervised pretraining on unlabeled soundscapes *(optional fallback)*

Only pursue if supervised baseline plateaus and there's time budget (~1-2 days of compute).

If unlabeled soundscape data is available (BirdCLEF 2024/2025 included it):
- Pretrain with BYOL / MoCo on unlabeled Pantanal soundscapes
- Fine-tune on labeled data
- Expected gain: +0.02–0.05 LB
- **Skip if** Step 3 ensemble already beats public SED baseline by a comfortable margin

### Step 5: Ensemble + post-processing

- **Ensemble**: 5-fold × 3 seeds × 2 backbones → 30 models, weighted average
- **Pseudo-labeling**: generate soft labels on test soundscapes, retrain
- **Species-specific thresholds**: optimize per-species threshold on OOF (vs global 0.5)
- **Calibration**: Platt scaling on OOF predictions

---

## Hardware Plan

Training on **2× RTX 3090** at `kristian@omarchyd.fritz.box`:

```bash
# Single-GPU training (DDP not needed for EfficientNet-B0/B3)
ssh kristian@omarchyd.fritz.box
cd ~/projects/kego/competitions/birdclef-2026
python train.py --fold 0 --backbone efficientnet_b0 --gpu 0
```

For full 5-fold training: run 5 parallel jobs (1 GPU each × 2 GPUs → 2 folds at a time).

Data needs to be downloaded/synced to Linux machine:
```bash
rsync -avz --rsync-path=/usr/bin/rsync ~/projects/kego/data/birdclef/ \
  kristian@omarchyd.fritz.box:~/projects/kego/data/birdclef/
```

---

## Key References

- [BirdCLEF 2025 1st place](https://www.kaggle.com/competitions/birdclef-2025/discussion) — ConvNeXt + AudioSet pretraining
- [BirdCLEF 2024 solutions](https://www.kaggle.com/competitions/birdclef-2024/discussion)
- [SED Baseline LB 0.862](https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-lb-0-862) — starting point
- [PyTorch Baseline Inference](https://www.kaggle.com/code/antoinemasq/birdclef-2026-pytorch-baseline-inference)
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) — pretrained bird sound model

---

## Dead Ends / Notes

- Raw waveform models (wav2vec2, etc.) generally underperform mel-spectrogram CNNs for BirdCLEF
- Per-species normalization of spectrograms doesn't help (ImageNet normalization works fine)
