# BirdCLEF+ 2026

Multilabel bird species identification from passive acoustic monitoring (PAM) recordings in the Pantanal wetlands.

- **Metric**: cmAP (class-mean average precision)
- **Deadline**: May 27, 2026
- **Prize**: $50,000
- **Data**: ~15GB .ogg at `data/birdclef/birdclef-2026/`. Precomputed 224-mel HTK specs at `specs_cache_224_htk/`.
- **Species**: 234 target species (206 with training data, 28 zero-shot)

## Results

| Kernel | LB | Notes |
|---|---|---|
| v14 (EfficientNet-B0 plain) | 0.758 | 5-fold, 128-mel |
| v16 (EfficientNet-B3 plain) | 0.776 | 5-fold, 128-mel |
| v13 (soundscape-v3 + temporal smooth) | 0.864 | HTK mel, warm restarts, gain aug |
| v15 (soundscape-v5 + inference tricks) | 0.876 | 50% stride, class-type smooth, file-max prior |
| **v21 (soundscape-v7 + class-cond pooling)** | **0.882** | **Current best** |
| v23 (soundscape-v8 HGNetV2 wrong mel: 224-mel, hop=512, HTK) | 0.858 | Regression — HGNetV2 trained with EfficientNet mel config |
| v31 (2 folds + TTA) | 0.857 | Regression — fold diversity > TTA gain |
| v32 (sharpened smooth + T=1.1) | pending | Post-processing swap experiment |
| pending (soundscape-v8-hgnetv2: HGNetV2-B0 correct mel 256-mel, hop=625) | pending | Correct mel config for HGNetV2 |

## Strategy

See `PLAN.md` for full strategy, experiment log, and next steps.

---

## Scripts and Notebooks

### `training/`

**`training/train_cnn.py`** — Main training script. EfficientNet-B0 NoisyStudent + GEMFreqPool + AttentionSEDHead, trained on XC clips + labeled soundscapes.

```bash
# Example: train soundscape-v7 (current best config)
ssh kristian@omarchyd "(cd /home/kristian/projects/kego && CUDA_VISIBLE_DEVICES=0 KEGO_PATH_DATA=... nohup uv run python competitions/birdclef-2026/training/train_cnn.py --baseline --soundscape-labels --htk --warm-restarts --gain-aug --fold 0 --n_folds 4 --tag soundscape-v7 >> /tmp/soundscape-v7_fold0.log 2>&1) &"
```

Key flags: `--baseline` (EfficientNet-B0), `--soundscape-labels` (include 66 labeled soundscapes), `--htk` (HTK mel scale), `--warm-restarts` (CosineAnnealingWarmRestarts), `--gain-aug` (±12dB gain augmentation), `--mixup-alpha 1.0`.

**`training/train_perch_probes_v2.py`** — Trains per-class LogisticRegression probes on Perch v4 embeddings from all 35,549 training clips (181 species with ≥5 clips). Supersedes the soundscape-only v1. Output: `perch_probes_v2.pkl` (uploaded as Kaggle dataset).

### `inference/`

**`inference/kaggle_inference.ipynb`** + **`inference/kernel-metadata.json`** — Active submission notebook.

- Loads soundscape-v7 `.pt` checkpoints from `aldisued/birdclef2026-soundscape-v7`
- 50% overlap sliding windows, class-conditional pooling (max=events, mean=texture)
- Post-processing: sharpened temporal smooth + temperature scaling + file-max prior
- Submits to `aldisued/birdclef-2026-baseline-inference`

```bash
kaggle kernels push -p competitions/birdclef-2026/inference/
```

**`inference/kaggle_perch_inference.ipynb`** + **`inference/kernel-perch-metadata.json`** — Perch frozen feature extractor inference. Loads pre-computed embeddings from cache dataset + trained LogReg probes. Currently blocked: Perch runs too slow on Kaggle CPU to generate the initial cache.

**`inference/kaggle_perch_cache_test.ipynb`** + **`inference/kernel-perch-cache-metadata.json`** — Cache kernel: runs Perch on test soundscapes, saves `test_perch_cache.npz` to output. Once a scoring environment runs this and we download the output, we can enable fast Perch inference.

**`inference/convert_to_onnx.py`** — Converts soundscape-v7 `.pt` checkpoints to ONNX format. Max diff PyTorch vs ONNX ~1e-7. Note: ONNX Runtime is slower than PyTorch on Kaggle CPU — dead end for the submission kernel.

**`inference/convert_to_openvino.py`** — Converts to OpenVINO IR format. Blocked: `openvino` pip install requires internet, which Kaggle disables.

### `data/`

**`data/precompute_specs.py`** — Precomputes log-mel spectrograms for all training clips as float16 `.npy` files. Run once on the training server. Gives 14× speedup vs on-the-fly librosa decoding.

```bash
KEGO_PATH_DATA=... uv run python competitions/birdclef-2026/data/precompute_specs.py
```

**`data/perch_cache_soundscapes.py`** — Runs Perch v4 on the 66 labeled train soundscapes, saves embeddings + logits as `.npz`. Used for probe training.

**`data/perch_cache_train_clips.py`** — Runs Perch v4 on all 35,549 XC training clips, saves embeddings for probe training.

### `eval/`

**`eval/eval_oof.py`** — Evaluates OOF predictions from train.py checkpoints. Computes per-fold and averaged cmAP on held-out XC clips and labeled soundscapes.

**`eval/benchmark_inference.py`** — Times inference per soundscape file for different configs (folds, TTA, overlap stride). Used to verify Kaggle budget headroom.

**`eval/test_pooling_logic.py`** — Unit tests for the class-conditional pooling and overlap aggregation logic used in inference.

### `eda/`

**`eda/analyze_data.py`** — EDA scripts: species distribution, soundscape label coverage, class breakdown.

**`eda/eda.ipynb`** — Interactive EDA notebook.

### `research/`

**`research/research.md`** — Full research notes: public notebook analysis, approach breakdown, what 0.90+ teams are doing.

**`research/research-lit.md`** — Literature review: ecoacoustics papers, PCEN, calibration, co-occurrence priors.

---

## Archive

`archive/` contains superseded or dead-end scripts:
- `predict.py` — early inference script, superseded by `kaggle_inference.ipynb`
- `pseudo_label_birdnet.py/perch.py/self.py` — pseudo-labeling experiments, all dead ends
- `train_perch_probes.py` — soundscape-only probe training (53 species), superseded by v2

---

## Hardware

Training: 2× RTX 3090 at `kristian@omarchyd` (Tailscale). `KEGO_PATH_DATA=/home/kristian/projects/kego/data`.

Spec caches:
- `specs_cache/` — 128-mel, linear
- `specs_cache_224/` — 224-mel, linear
- `specs_cache_224_htk/` — 224-mel, HTK (used by soundscape-v3+)
- `specs_cache_soundscape_224_htk/` — 224-mel HTK for all 66 soundscape files (complete, 16.7GB)
- `specs_cache_hgnetv2/` — 256-mel, n_fft=2048, hop=625, fmin=20, slaney norm (for HGNetV2 BirdModel)
- `specs_cache_soundscape_hgnetv2/` — same config for train_soundscapes/
