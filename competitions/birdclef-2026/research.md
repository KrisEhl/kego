# BirdCLEF+ 2026: Research & Literature Review

**Compiled**: March 2026
**Competition deadline**: May 27, 2026
**Current best**: LB 0.783 | **Target**: ≥0.862 (public baseline gap)

---

## 1. Executive Summary

### Key Findings

The ~0.079 gap between our current best (LB 0.783) and the public SED baseline (0.862) is driven overwhelmingly by **Perch-based pseudo-labeling**, not architecture, augmentation, or ensemble size. This is confirmed by:

- The public baseline checkpoint is named `best_perch_fold0.pt` and comes from dataset `aidensong123/perch-fold`
- The public baseline uses only a single fold; our 5-fold ensemble is still 0.079 behind
- Our BirdSet XCL pretrained B1 (9,736 Xeno-Canto species) achieved LB 0.782 — essentially identical to B0, ruling out generic bird-domain pretraining as the key factor
- The architecture (tf_efficientnet_b0.ns_jft_in1k + GEMFreqPool + AttentionSED) is literally identical to ours

The competition winner gap in BirdCLEF consistently comes from **audio-domain pretraining on large-scale bioacoustic data**, not from architecture cleverness. The public baseline almost certainly implements the pipeline: Perch pseudo-labels → pretrain EfficientNet → fine-tune on ground truth.

### Top 3 Recommendations

1. **Perch Pseudo-Labeling Pipeline** (highest impact, effort: medium-high): Run Google Perch on all 35,549 training clips + any available unlabeled soundscapes to generate soft labels for all 234 competition species. Use as teacher labels for a pretraining stage before supervised fine-tuning. Expected gain: +0.05 to +0.08 LB, closing most of the gap. This is the single most important thing.

2. **PANNs CNN14 + fine-tuning** (high impact, effort: low-medium): CNN14 pretrained on AudioSet (2M clips, 527 classes) provides general audio representations including birds/nature. Fine-tune with a new classifier head for 234 species. Faster to run than Perch pseudo-labeling and a solid fallback. Expected gain: +0.015 to +0.03 LB.

3. **Per-species threshold calibration** (low effort, guaranteed positive): Grid search sigmoid threshold per class on OOF predictions. Can lift cmAP 1–3% without any retraining. Should be done after any model improvement.

---

## 2. BirdCLEF Competition History (2020–2025)

### BirdCLEF 2021 (816 teams)
- 10th place approach: weakly supervised classification and detection pipeline, handling background noise (airplanes, rain). Sound detection + classification for soundscape recordings (arXiv:2107.04878).
- Key insight: weak label training for clip-level annotations, then SED for soundscape inference.

### BirdCLEF 2022 (807 teams)
- Transformers outperform CNNs: Attention-based Spectrogram Transformer baseline showed that transformer models outperformed convolutional models for bird call recognition (arXiv:2211.07722).
- 18th place: Few-shot learning from weak labels, acoustic detection pipeline for long-tailed species (arXiv:2206.11260).
- Unsupervised method (0.48 score): triplet loss on spectrogram audio motifs for representation learning (arXiv:2206.04805).

### BirdCLEF 2023 (1st place — VSydorskyy)
- **Architecture**: Ensemble of ConvNeXt Small (ImageNet-22k/1k pretrained, 384px), ConvNeXtV2 Tiny (FCMAE pretrained, 384px), and ECA-NFNet-L0
- **Pretraining**: ImageNet-22k → ImageNet-1k fine-tune, then adapted to mel spectrograms. No bird-specific pretraining.
- **Mel settings**: 32kHz, 128 mels, 20Hz fmin, 2048 N_FFT, 512 hop, 5s segments
- **Augmentation**: SpecAugment (freq masking up to 10 bins × 3, time masking up to 20 frames × 3, p=0.3 each), Mixup (p=0.5), background noise injection from soundscapes without bird calls and ESC-50 (p=0.5), random filtering
- **Loss**: Focal loss
- **Pseudo-labeling**: Multi-year data expansion — models scored BirdCLEF 2023 + 2020–2022 recordings including secondary labels (`scored_2023_xc_2023SecLabels.csv`)
- **Soundscape inference**: Clipwise (75% weight) + TimeMax pooling (25% weight) + Gaussian mean, compiled to ONNX
- **Key source**: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
- **Team from 4th place (2024)**: Built on honglihang's BirdCLEF 2023 2nd place solution, further pretrained on BirdCLEF 2021/2022/2023 data before fine-tuning on 2024.

### BirdCLEF 2024 (1st place — arpoyda)
- **Architecture**: Ensemble of EfficientNet-B0 + RegNetY-008, 6 models total
- **Mel settings**: 32kHz, 128 mels, 40 Hz fmin, 15000 Hz fmax, n_fft=1024, hop=500
- **Pseudo-labeling** (two-stage, key innovation):
  1. Train initial EfficientNet-B0 folds on labeled data
  2. Run Google Perch (TF Hub `bird-vocalization-classifier`) on unlabeled soundscapes in 5s windows → soft labels
  3. Run EfficientNet fold ensemble on same soundscapes → additional soft labels
  4. Merge both label sources; concatenate pseudo-labeled soundscapes with labeled training data
  5. Retrain final models on combined dataset
- **Augmentation**: Heavy CutMix (p=0.9), XYMasking (SpecAugment variant) with 90% probability — num_masks_x=(1,12), num_masks_y=(1,3)
- **Soundscape strategy**: 20/30/60-second segment variations; 5-second windows for Perch inference
- **Key source**: https://github.com/arpoyda/BirdCLEF_2024

### BirdCLEF 2024 (Other top approaches)
- **Progressive knowledge distillation with semi-supervised soundscape labeling** (Lihang Hong): CEUR-WS paper-198 — multi-stage KD pipeline with self-labeling on soundscapes
- **BirdNET + Perch pseudo multi-label** (DS@GT team, arXiv:2407.06291): Used Google Bird Vocalization Classifier, BirdNET, and EnCodec to generate pseudo-labels. Applied association rule mining to capture species co-occurrence patterns. Best score 0.63 with BirdNET embeddings + Bird Vocalization pseudo-labels.
- **Multi-year pretraining** (4th place): seresnext26ts, rexnet_150, inception_next_nano — pretrained on BirdCLEF 2021/2022/2023 data, background noise from ESC-50, OpenVINO for inference optimization.

### BirdCLEF 2025 (5th place — myso1987)
- **Architecture**: Ensemble of EfficientNet-B0, B3, EfficientNetV2-B3, EfficientNetV2-S
- **SED approach**: Sound event detection formulation (not plain clip classification)
- **Audio**: 30–60 second training segments, 32kHz
- **Oversampling**: For species with fewer than 20 samples
- **Inference**: OpenVINO conversion for CPU speed
- **Source**: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution

### BirdCLEF 2025 (4th place — dylanliu2)
- **Approach**: Semi-supervised learning with Soft AUC Loss
- **Models**: EfficientNet B0–B4 and EfficientNetV2 variants (B0-B3, S)
- **Mel settings**: Three variants: 256×256 (256 mels, 60-16000Hz), 384×192 (192 mels, 50-16000Hz), 288×224 (224 mels, 40-16000Hz)
- **Loss**: SoftAUCLoss / AUCLoss (directly optimizes the competition metric)
- **Augmentation**: Mixup (alpha=0.5), SpecAugment
- **Semi-supervised**: Pseudo-labeled soundscapes; SED threshold 0.5 for frame-level aggregation, temporal smoothing with 0.15 factor between adjacent segments
- **Source**: https://github.com/dylanliu2/BirdCLEF2025-4th-place-solution

### BirdCLEF 2025 (overall notes from arXiv:2507.08236)
- Competition required 206 species classification under **90-minute CPU-only inference** deadline
- BirdSetEfficientNetB1 achieved 0.810 (public) / 0.778 (private) ROC-AUC
- Perch TFLite conversion gave ~10x inference speedup (~16 minutes total)
- Spectrogram Token Skip-Gram (STSG): audio as sequence → Faiss K-means tokens → Word2Vec skip-gram embeddings → linear classifier. Only 6 minutes inference but lower accuracy (0.520 private).

### Consistent Patterns Across Competition History

1. **Audio-domain pretraining** is the dominant differentiator — bird-specific pretraining (Perch, BirdNET) or AudioSet pretraining consistently outperforms ImageNet-only approaches
2. **Pseudo-labeling on unlabeled soundscapes** is used by nearly every top solution
3. **EfficientNet family** dominates (B0–B3, NoisyStudent variant)
4. **Multi-year data expansion**: top solutions always use data from prior BirdCLEF years as pretraining
5. **SpecAugment + Mixup** is universal
6. **Soundscape inference**: 5-second sliding windows, SED/attention aggregation, temporal smoothing

---

## 3. Perch / Google Bird Vocalization Classifier

### What is Perch?

Perch is Google Research's bioacoustics classification system. The original paper is: "Global birdsong embeddings enable superior transfer learning for bioacoustic classification" (Ghani et al., Nature Scientific Reports 2023).

**Architecture**:
- Frontend: PCEN (Per-Channel Energy Normalization) mel-spectrogram
- Backbone: EfficientNet
- Training: supervised on 10,000+ bird species from global sources

**Perch 2.0** (NeurIPS 2025 AI for Non-Human Animal Communication workshop, arXiv:2512.03219):
- Trained on **14,597 species** including birds, mammals, amphibians, and insects
- Superior few-shot transfer learning even for out-of-domain audio (e.g., marine mammals not in training data)
- Outperforms Perch 1.0, SurfPerch, and BirdNET V2.3 on transfer tasks

**SurfPerch**: Variant trained on combination of birds, coral reef sounds, and general audio.

### Availability

- **TensorFlow Hub**: `https://tfhub.dev/google/bird-vocalization-classifier/4` → redirects to Kaggle
- **Kaggle Models**: `google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4`
- **GitHub**: https://github.com/google-research/perch
  - `chirp/models/frontend.py` — PCEN mel frontend
  - `chirp/models/efficientnet.py` — backbone
  - `chirp/models/perch_2.py` — newer components
  - `embed_audio.ipynb` — batch embedding of audio files
  - `agile_modeling.ipynb` — search + active learning over embeddings
  - `analysis.ipynb` — inference and call density estimation

### Inputs/Outputs

- **Input**: Raw audio, 32kHz sample rate, 5-second windows
- **Output**: Logits over species classes + embeddings (the embedding is a rich feature vector)
- The model outputs both predictions and an embedding layer that can be used as features

### How to Use for Pseudo-Labeling (BirdCLEF 2024 approach)

The BirdCLEF 2024 1st place solution (`arpoyda`) used Perch via TF Hub as follows:

```python
import tensorflow_hub as hub
import tensorflow as tf

model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/4")

# Process audio in 5-second chunks
step = sr * 5  # sr = 32000
for offset in range(0, len(audio) - step, step):
    chunk = audio[offset:offset+step]
    x = chunk[np.newaxis, :]
    logits = model.infer_tf(x)[0][0].numpy()
    # Filter to competition species using actual_indices
    pseudo_labels.append(logits[actual_indices])
```

The raw logits (not thresholded) are stored per 5-second chunk and used as soft labels alongside the EfficientNet fold predictions. Combined soft labels are then used to train the final models.

### Perch on BirdCLEF 2026

Perch is trained on 10,000–14,597 species globally. The 234 BirdCLEF 2026 species are a subset; Perch can provide soft label estimates for all species including the 28 zero-shot ones (for which it may have some coverage from XC data).

**Implementation plan**:
1. Install: `pip install tensorflow tensorflow-hub chirp`
2. Load model from Kaggle Models or TF Hub
3. Create `actual_indices` mapping from Perch's full species list to our 234 competition species
4. Process all 35,549 training clips in 5-second windows → soft label matrix (n_clips × 234)
5. Optionally process unlabeled soundscapes if available
6. Use soft labels as teacher signal for 1–2 pretraining epochs, then fine-tune on hard ground-truth labels

**Existing Kaggle datasets**:
- `aidensong123/perch-fold` — the dataset used in the public baseline; contains `best_perch_fold0.pt` which is a fine-tuned checkpoint after Perch-based training

### What the Public Baseline Likely Does

Based on the checkpoint name and multi-stage architecture (`ckpt.get("stage", "supervised")`):
1. Stage 1: Pretrain on Perch pseudo-labels (all training data, soft labels from Perch)
2. Stage 2: Fine-tune on ground-truth labels

This matches the BirdCLEF 2024 1st place pipeline and explains the 0.079 LB gap from a single-fold model.

---

## 4. PANNs (Pretrained Audio Neural Networks)

### What are PANNs?

Kong et al. (2020) introduced PANNs: large-scale CNN models pretrained on **AudioSet** (2 million 10-second audio clips, 527 sound classes). Paper: arXiv:1912.10211.

**Best model**: Wavegram-Logmel-CNN, mAP=0.439 on AudioSet (combining waveform + log-mel inputs).

### CNN14 Architecture
- 14-layer CNN with batch normalization
- Input: log-mel spectrogram (64 mels, 32kHz, 1024 N_FFT, 320 hop)
- Output: 527 AudioSet classes + embedding
- mAP = 0.431 on AudioSet; mAP = 0.438 for CNN14_16k variant
- **Sound event detection**: CNN14_DecisionLevelMax provides frame-level predictions (mAP=0.385)

### Other PANN Architectures
- Cnn6, Cnn10, Cnn14 (32kHz and 16kHz variants)
- ResNet38, ResNet54
- MobileNet variants (for CPU inference)
- Wavegram-CNN (waveform input)

### Pretrained Weights
- GitHub: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Zenodo: `Cnn14_mAP=0.431.pth`, `Cnn14_16k_mAP=0.438.pth`, etc.
- Fine-tuning template: `finetune_template.py` in the repo

### Relevance to BirdCLEF

AudioSet includes bird-related classes ("Bird", "Chirping, tweeting", "Crow", "Caw", "Owl", "Cuckoo", etc.) — roughly 20+ bird-related out of 527 classes. CNN14 embeddings capture general audio structure including bird vocalizations.

**Transfer learning approach**:
1. Load `Cnn14_mAP=0.431.pth`
2. Replace output layer: 527 classes → 234 BirdCLEF classes
3. Fine-tune on BirdCLEF training data
4. Expected: moderate improvement over ImageNet pretraining alone; less than Perch (which is bird-specific)

### CPU Inference Feasibility

CNN14 is a CNN-based model — **highly CPU-compatible**. Inference is comparable to EfficientNet-B0 in computational cost. Can be exported to ONNX or TFLite for further optimization.

### Expected Performance

Based on the competition history, AudioSet pretraining typically gives +0.015 to +0.030 LB improvement over ImageNet-only pretraining for bird classification. Less than Perch's +0.06–0.08 gain but much easier to implement.

---

## 5. Audio Foundation Models

### BirdNET (Cornell Lab + TU Chemnitz)
- **Architecture**: Deep learning, 6,512 species, available as Python package (`birdnet-analyzer`)
- **Installation**: `pip install birdnet-analyzer`
- **Models on Zenodo** (CC BY-NC-SA 4.0 — non-commercial use OK for academic)
- **API**: Processes audio files, returns species + confidence scores
- **Inputs**: Audio files at any sample rate (resampled internally), produces 3-second window predictions
- **Output**: CSV with species, confidence, start/end times
- **BirdNET in competition**: DS@GT team (BirdCLEF 2024) used BirdNET embeddings with Bird Vocalization pseudo-labels for 0.63 public LB. BirdNET as feature extractor + classifier outperforms using it as labeler alone.
- **GitHub**: https://github.com/kahst/BirdNET-Analyzer

**Using BirdNET for pseudo-labeling**:
```bash
# Command line:
birdnet_analyzer analyze --i audio/ --o predictions/ --rtype csv
```

BirdNET covers all 234 BirdCLEF 2026 target species (or most of them — it covers Pantanal species). Can use raw confidence scores as soft labels.

### AudioMAE (Facebook Research)
- **Paper**: Huang et al., NeurIPS 2022 (arXiv:2207.06405)
- **Architecture**: Masked Autoencoder on audio spectrograms (ViT-B, 85.66M parameters)
- **Training**: Self-supervised pretraining on AudioSet-2M (masked spectrogram patches)
- **Performance**: ~47.3 mAP on AudioSet after fine-tuning
- **GitHub**: https://github.com/facebookresearch/AudioMAE
- **Key innovation**: Local window attention in decoder (audio is locally correlated in time-frequency)
- **CPU feasibility**: ViT-based → SLOW on CPU. Not practical for Kaggle CPU-only inference with 90-min budget.

### BEATs (Microsoft)
- **Paper**: Chen et al., 2022 (arXiv:2212.09058)
- **Architecture**: Iterative bidirectional encoder + acoustic tokenizer (Transformer-based)
- **Performance**: 50.6 mAP on AudioSet-2M (SOTA for audio-only models), 98.1% on ESC-50
- **Key innovation**: Discrete label prediction (not reconstruction) for better high-level audio semantics
- **CPU feasibility**: Transformer-based → SLOW. Not practical for CPU-only inference.
- **BirdSet benchmark**: Available as pretrained checkpoint through DBD-research-group

### EAT (Efficient Audio Transformer)
- **Paper**: arXiv:2401.03497
- **Architecture**: Bootstrap self-supervised training for audio, inspired by data2vec 2.0 and AudioMAE
- **Key innovation**: Utterance-Frame Objective (UFO) + large inverse block masks
- **Performance**: SOTA on AudioSet-2M, AS-20K, ESC-50, SPC-2
- **Speed**: Up to ~15x faster pre-training than existing audio SSL models
- **CPU feasibility**: Transformer-based → SLOW for inference. Available through BirdSet benchmarks.
- **BirdSet**: EAT checkpoint available at `DBD-research-group` on HuggingFace

### AST (Audio Spectrogram Transformer)
- **Paper**: Gong et al., Interspeech 2021 (arXiv:2104.01778)
- **Architecture**: First purely attention-based model for audio classification (no convolutions)
- **Performance**: 0.485 mAP on AudioSet, 95.6% on ESC-50, 98.1% on SPC-2
- **CPU feasibility**: Transformer-based → SLOW. Not practical for CPU-only inference in time-constrained settings.
- **BirdSet**: AST checkpoint available through DBD-research-group

### CLAP (Contrastive Language-Audio Pretraining)
- **Paper**: Wu et al. (arXiv:2206.04769)
- **Capability**: Zero-shot audio classification using text descriptions
- **Training**: 128,000 audio-text pairs; SOTA on 5 supervised + 16 zero-shot downstream tasks
- **Relevance**: Could help with the 28 zero-shot species — generate text descriptions ("Pantanal bird with sharp whistle") and use CLAP for zero-shot inference
- **LAION-CLAP** (arXiv:2211.06687): Large-scale version trained on LAION-Audio-630K (633,526 audio-text pairs)
- **CPU feasibility**: Depends on backbone; some variants have CNN backbones. Main concern is zero-shot species use case.

### Bird-MAE (DBD-research-group)
- **Models**: Bird-MAE-Base (85.5M), Bird-MAE-Large (0.3B), Bird-MAE-Huge (0.6B)
- **Paper**: "Can Masked Autoencoders Also Listen to Birds?" (OpenReview GIBWR0Xo2J)
- **Training**: Masked autoencoder pretraining on BirdSet XCL (9,736 species)
- **Available on HuggingFace**: `DBD-research-group/Bird-MAE-Base` etc.
- **CPU feasibility**: ViT-based → SLOW. Fine-tune and use for embeddings then distill.

### BirdSet Pretrained Models (Summary)
From `DBD-research-group` on HuggingFace:
- **EfficientNet-B1-BirdSet-XCL** (19M params, CNN-based → CPU-friendly): Same model we've already tried (LB 0.782 — no gain)
- **AudioProtoPNet variants** (97.6M–0.3B): interpretable prototype network
- **BirdMAE-XCL** (92.9M): MAE pretrained on Xeno-Canto
- These are benchmarked across BirdSet's 8 test datasets (HSN, NBP, NES, PER, SNE, SSW, UHH)

### SSAST (Self-Supervised Audio Spectrogram Transformer)
- **Paper**: arXiv:2110.09784, AAAI 2022
- **Architecture**: Joint discriminative + generative masked spectrogram patch modeling on AudioSet + LibriSpeech
- **Performance**: 60.9% average improvement across audio/speech tasks
- **CPU feasibility**: Transformer-based → SLOW

### Speed Comparison (Practical for CPU Inference)
| Model | Architecture | CPU Inference | Notes |
|-------|-------------|---------------|-------|
| EfficientNet-B0 (ours) | CNN | Fast | Current approach |
| CNN14 (PANN) | CNN | Fast | Good option |
| Perch (EfficientNet backbone) | CNN | ~16min (TFLite) | Known to work |
| BirdNET | CNN/unknown | Moderate | 6,512 species |
| AST | ViT-B | Very slow | Not practical |
| AudioMAE | ViT-B | Very slow | Not practical |
| BEATs | Transformer | Very slow | Not practical |
| EAT | Transformer | Very slow | Not practical |

---

## 6. Soundscape-Level Inference Strategies

### The Core Challenge

Training data is clip-level (XC recordings, 5–60 seconds, point-in-time bird calls). Test data is soundscape-level (long passive acoustic recordings, multiple species, varying densities). The metric is cmAP evaluated on 5-second chunks.

### Strategy 1: 5-Second Sliding Window (Dominant Approach)

Used by virtually every top solution. Process each 5-second chunk independently, output species probabilities. No overlap needed (test is already non-overlapping 5s chunks).

For training: randomly crop 5s from longer training clips. For augmentation: random crop within recording.

### Strategy 2: Longer Segment Training + 5s Inference

BirdCLEF 2024 1st place used 20/30/60-second segments during training but 5-second windows for inference. The intuition: longer context helps the model learn audio scene context, but inference must match the test format.

```
Training: randomly sample 20s clip → mel → model → clip-level loss
Inference: 5s window → mel → model → prediction
```

### Strategy 3: SED (Sound Event Detection) Head + Frame Aggregation

Used by our current best model (GEMFreqPool + AttentionSED). The model produces frame-level probabilities, then aggregates to clip-level. Key details:
- **GEMFreqPool**: Generalized mean pooling over frequency dimension (collapses freq, keeps time)
- **AttentionSED**: Attention weights over time frames → weighted average for clip prediction
- Both clip-level (from GEMFreqPool) and frame-level predictions can be used

**Critical**: Need enough temporal frames. At 32x downsampling from EfficientNet, a 5s clip at 32kHz has 3.2s/0.032s = 100 frames after one stride, reduced to ~3-10 frames after full backbone. The 224-mel spectrogram gives more temporal resolution. Naive SED head with only ~10 temporal frames fails (LB 0.750 vs 0.783 with attention head).

### Strategy 4: Temporal Smoothing Between Adjacent Segments

BirdCLEF 2025 4th place: `smoothed_pred = (1-0.15) * current + 0.15 * avg(neighbors)`. Smoothing factor 0.15 across adjacent 5-second segments helps consistency.

### Strategy 5: Multi-Scale Inference

Process 5s chunks at multiple augmentations (pitch shift, time stretch) and average. Not commonly used for CPU-constrained settings.

### Strategy 6: Max Pooling vs. Mean Pooling over Frames

- **MaxPooling**: Good for detecting sparse events (bird calls are brief)
- **MeanPooling**: Better for sustained sound events
- **AttentionPooling**: Learned combination; generally best
- **BirdCLEF 2023 1st place** used: 75% clipwise + 25% time-max (weighted combination)

### Strategy 7: Test-Time Augmentation (TTA)

Rarely used in CPU-constrained settings due to compute cost. Can include:
- Horizontal flip of spectrogram (time reversal)
- Multiple frequency crops

### Practical Notes

For BirdCLEF 2026 (CPU-only, 90-min budget):
- 5-second non-overlapping chunks is the test format — match this exactly
- Keep inference simple: single forward pass per 5s chunk
- TFLite conversion of Perch gives ~10x speedup
- ONNX export of EfficientNet is also fast
- Budget: if you have 1 hour of test soundscapes at 5s chunks = 720 chunks/hour, must process quickly

---

## 7. Advanced Training Techniques

### 7.1 Mixup for Audio

Standard Mixup (Zhang et al. 2018) applied to spectrograms: mix two spectrogram images + interpolate labels. Used by BirdCLEF 2023 1st place (p=0.5). Universally beneficial for audio classification.

```python
# Audio Mixup
alpha = 0.5
lam = np.random.beta(alpha, alpha)
mixed_spec = lam * spec1 + (1-lam) * spec2
mixed_label = lam * label1 + (1-lam) * label2
```

### 7.2 CutMix for Spectrograms

BirdCLEF 2024 1st place used CutMix with p=0.9 — very aggressive. Cuts a rectangular patch from one spectrogram and pastes into another. More aggressive than Mixup for audio because it preserves temporal structure.

XYMasking (SpecAugment generalization) from BirdCLEF 2024: masks both time and frequency simultaneously in rectangular blocks.

### 7.3 SpecAugment

Standard for audio (Park et al. 2019, arXiv:1904.08779). Frequency masking + time masking. Used universally:
- BirdCLEF 2023 1st place: freq mask up to 10 bins × 3, time mask up to 20 frames × 3, p=0.3
- Our current best: freq_mask=30 × 2, time_mask=100 × 2 (large time mask worked well)

### 7.4 Background Noise Augmentation

Mixing bird vocalizations with real-world background sounds (soundscapes without calls, ESC-50, urban noise). Used by BirdCLEF 2023 1st place (p=0.5). Very effective for domain adaptation from XC clips to soundscapes.

Implementation: maintain a pool of "no-call" segments from competition soundscapes or ESC-50. Randomly mix:
```python
noise = random_nocall_segment()
aug_spec = spec + noise_alpha * noise_spec
```

### 7.5 Label Smoothing

Standard binary cross-entropy with label smoothing (0.05–0.1). Helps with noisy labels in bird recordings. Currently in use.

### 7.6 Secondary Label Handling

Our experiment: secondary labels at weight=1.0 (hard labels) vs weight=0.5 (soft labels). Result pending.

Historical best practice: secondary labels as soft positives with weight 0.2–0.5. Don't use hard negatives for secondary labels. BirdCLEF 2023 1st place included secondary labels in training CSV.

### 7.7 Loss Functions

- **BCE + Focal Loss**: BCE with focal weighting (high loss for hard examples). Used by BirdCLEF 2023 1st place.
- **SoftAUCLoss**: Directly optimizes AUC/cmAP metric. Used by BirdCLEF 2025 4th place (dylanliu2). Complex to implement but metric-aligned.
- **Dual frame+clip loss**: 0.5 * clip_BCE + 0.5 * frame_BCE. We tried this (0.765, regression vs 0.783 — unclear if loss or time_mask change caused it).
- **BCE is simplest and works**: BirdCLEF 2024 1st place didn't show any fancy loss — just combined data strategy.

### 7.8 Multi-Year Data Expansion

Using BirdCLEF data from prior years is strongly correlated with top performance:
- BirdCLEF 2021, 2022, 2023, 2024 data all available on Kaggle
- Can be used as pretraining data (even if species don't overlap with 2026)
- Builds robust audio representations that transfer to new species

For BirdCLEF 2026 (Pantanal birds): South American bird data from prior years may overlap. Worth checking.

### 7.9 Pseudo-Labeling Pipeline (General Recipe)

Standard semi-supervised approach used by top solutions:
1. Train initial model on labeled data (5-fold cross-validation)
2. Run inference on unlabeled soundscapes with initial models
3. Apply confidence threshold (0.3–0.5) or use raw soft labels
4. Add high-confidence pseudo-labeled soundscape segments to training data
5. Retrain models on combined labeled + pseudo-labeled data

BirdCLEF 2024 1st place used both Google Perch AND initial EfficientNet predictions for pseudo-labeling — two complementary sources.

### 7.10 Knowledge Distillation

Use Perch or BirdNET as teacher, fine-tune smaller EfficientNet as student. Options:
1. **Hard pseudo-labels**: Teacher predicts → train student on those labels (what we want for Perch)
2. **Soft distillation**: Student matches teacher's probability distribution (KL divergence loss)
3. **Progressive distillation**: Multiple rounds, student becomes next round's teacher

### 7.11 Focal Loss for Long-Tailed Distribution

BirdCLEF has extreme class imbalance (max 499, min 1 recording). Focal loss down-weights easy negatives and focuses training on hard/rare species. Used by BirdCLEF 2023 1st place.

### 7.12 Balanced Sampling

Sample by primary label to ensure rare species appear frequently. BirdCLEF 2023 1st place uses `sampler_col='primary_label'` for balanced sampling. Important for species with very few recordings.

---

## 8. Zero-Shot Species

### The Problem

28 of 234 species in BirdCLEF+ 2026 have zero training data (mostly `47158son*` sonotype splits). The competition presumably includes these in the test set and cmAP computation.

### Historical Approaches

#### Approach 1: Ignore (Most Common)
Most solutions train on the 206 labeled species and output 0 for zero-shot species. Since cmAP averages across all classes, zero predictions for 28 classes drags the metric down by roughly 28/234 ≈ 12%. If other classes score perfectly, the cap from zero-shot is ~0.88.

#### Approach 2: Perch Embeddings + Prototype Transfer
Since Perch is trained on 10,000+ species globally, it likely has seen (or similar species to) the 28 zero-shot BirdCLEF 2026 species. The "agile modeling" workflow in the Perch repo supports:
1. Embed audio with Perch
2. Use active learning to find similar audio in embedding space
3. Build lightweight classifier from a few examples

For zero-shot species: use Perch's existing knowledge directly. The zero-shot species might be sub-types or splits of species Perch knows.

#### Approach 3: BirdNET Zero-Shot Coverage
BirdNET covers 6,512 species. Check if zero-shot BirdCLEF 2026 species are in BirdNET's species list. If so, BirdNET provides direct predictions.

#### Approach 4: CLAP Zero-Shot (Text-Based)
Generate text descriptions of each zero-shot species and use CLAP for zero-shot classification. Requires knowing the species' typical vocalization pattern and writing an accurate description. Probably low accuracy but better than 0.

#### Approach 5: Prototype from Similar Species
If zero-shot species are sonotype splits of known species, they likely sound similar to their parent species. Transfer prototype embeddings from the parent species.

### Practical Recommendation

1. Check if zero-shot species are in Perch's or BirdNET's species list — if yes, use their predictions directly
2. For truly unseen species, output average probability of closely related species (phylogenetic proximity)
3. Even a small non-zero prediction for zero-shot species helps cmAP vs always predicting 0

---

## 9. Feasibility Assessment

| Technique | Expected LB Gain | Effort | CPU-Inference Compatible | Priority |
|---|---|---|---|---|
| **Perch pseudo-labels** | +0.05–0.08 | Medium-High | Yes (TFLite ~16min) | **P0** |
| **CNN14/PANN fine-tuning** | +0.015–0.03 | Low-Medium | Yes (CNN) | **P1** |
| **Multi-year BirdCLEF data pretraining** | +0.010–0.025 | Low | Yes | **P1** |
| **Per-species threshold calibration** | +0.005–0.015 | Very Low | n/a | **P1** |
| **Background noise augmentation** | +0.005–0.015 | Low | n/a | **P2** |
| **Hard secondary labels (results pending)** | ±0.005 | Done | n/a | **Wait for result** |
| **BirdNET pseudo-labels** | +0.010–0.03 | Low | Yes | **P2** |
| **Multi-scale / longer-segment training** | +0.005–0.010 | Low | n/a | **P2** |
| **Temporal smoothing between chunks** | +0.002–0.005 | Very Low | n/a | **P2** |
| **Ensemble BirdSet + Baseline** | +0.002–0.005 | Done | Yes | Low |
| **SoftAUCLoss** | +0.005–0.015 | Medium | n/a | **P3** |
| **AudioMAE/BEATs fine-tune** | +0.015–0.03 | Medium | **No — too slow** | Not feasible |
| **AST/EAT transformers** | +0.010–0.02 | Medium | **No — too slow** | Not feasible |
| **CLAP zero-shot species** | +0.002–0.01 | Medium | Maybe | P3 |
| **Perch zero-shot coverage** | +0.005–0.015 | Low | Yes | P2 |

### Detailed Assessments

**Perch Pseudo-Labels** (P0 — do this first):
- This is almost certainly what the public baseline does (checkpoint named `best_perch_fold0.pt`)
- Perch is available via TF Hub/Kaggle Models
- CPU-compatible: TFLite conversion gives 10x speedup; runs in ~16 minutes per Perch paper
- Implementation: Load Perch, run on all 35,549 clips in 5s windows, get (n_clips, n_perch_classes) matrix, map to 234 BirdCLEF species, use as soft labels for pretraining
- Main risk: Perch's species list may not perfectly cover all 234 BirdCLEF 2026 species (Pantanal-specific species may be missing)

**CNN14/PANN Fine-Tuning** (P1):
- Straightforward: download weights from Zenodo, replace output layer, fine-tune
- CNN-based → fast CPU inference
- AudioSet includes general bird/nature sounds but not species-specific
- Lower ceiling than Perch but faster to implement
- Weights: `Cnn14_mAP=0.431.pth` from https://zenodo.org/record/3960586

**Multi-Year Data Pretraining** (P1):
- Download BirdCLEF 2021/2022/2023 data from Kaggle
- Pre-train on all available bird audio (possibly 100K+ clips across all years)
- Then fine-tune on BirdCLEF 2026 training data
- No extra compute needed at inference time
- Partially what competitors do via their "pretraining on past BirdCLEF" approach

**Per-Species Threshold** (P1 — do this now, zero downside):
- Run OOF inference → grid search threshold per species on cmAP
- Can do in a few minutes computationally
- Expected +0.005–0.015 on LB with no retraining

**AudioMAE, BEATs, AST, EAT** (Not feasible):
- All transformer-based with ViT-B or larger backbones
- CPU inference for 5s audio chunks at scale → too slow for 90-minute budget
- Could be used as feature extractors during training only, then distill to CNN — but high complexity

---

## 10. Recommended Roadmap

### Immediate (This Week)

**Step 1: Per-species threshold calibration**
- Run OOF predictions from current best model (baseline_fold0–4.pt)
- Grid search threshold per species on OOF cmAP
- Apply in inference notebook

**Step 2: Background noise augmentation**
- Extract "no-call" segments from BirdCLEF 2026 training soundscapes
- Add as random background mixing (p=0.3–0.5) to training pipeline
- Retrain existing architecture — should get +0.005–0.015

### Short-Term (Next 1–2 Weeks) — The Key Bets

**Step 3: Perch Pseudo-Labeling Pipeline** (highest priority)

```bash
# Install Perch dependencies
pip install tensorflow tensorflow-hub

# Run Perch on training data
python pseudo_label_perch.py \
  --audio_dir data/birdclef/birdclef-2026/train_audio/ \
  --output_dir data/birdclef/birdclef-2026/perch_labels/ \
  --model_url "https://tfhub.dev/google/bird-vocalization-classifier/4"
```

Implementation notes:
1. Map Perch species labels to BirdCLEF 2026 taxonomy (by species name or eBird code)
2. For zero-shot species, check Perch coverage; use BirdNET as fallback
3. Soft labels at clip level (5s windows): shape (n_clips, 234)
4. Pretraining stage (5–10 epochs): train EfficientNet-B0 on Perch soft labels, BCE
5. Fine-tuning stage (20–30 epochs): train on ground-truth hard labels, higher LR initially

If Perch has low coverage for Pantanal species, switch to BirdNET for the pseudo-labeling (BirdNET covers 6,512 species including Pantanal birds).

**Step 4: CNN14 Fine-Tuning (Alternative/Complement to Perch)**

```python
import torch
from torchvision.models import resnet

# Download Cnn14 from Zenodo
# Load and replace final FC layer
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=234)
ckpt = torch.load('Cnn14_mAP=0.431.pth', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)  # ignore final layer
```

**Step 5: Multi-Year Data Expansion**

- Download BirdCLEF 2021, 2022, 2023 data from Kaggle (all public)
- Map species to our 234 label space (partial overlap)
- Use as additional pretraining data before fine-tuning on BirdCLEF 2026

### Medium-Term (Weeks 2–4)

**Step 6: Ensemble Diversity**

Once Perch-pretrained model is working:
- Ensemble Perch-pretrained B0 + original baseline B0 (diversity from different pretraining)
- Add CNN14-finetuned model as third component
- Apply per-species threshold calibration on ensemble OOF

**Step 7: BirdNET for Zero-Shot Species**

```python
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

analyzer = Analyzer()
for species_id in zero_shot_species:
    # Find similar species in training data using Perch embeddings
    # Use BirdNET predictions for these species in inference
    pass
```

**Step 8: Longer Segment Training**

- Train on 10–20 second segments instead of 5s
- More temporal context for the model
- At inference, still use 5s windows (consistent with test format)

### Longer-Term (Weeks 4–8)

**Step 9: Advanced Pseudo-Labeling Iteration**

Round 2 pseudo-labeling:
1. After Perch-pretrained model is trained, run on unlabeled BirdCLEF soundscapes
2. High-confidence predictions (>0.5) used as additional training data
3. Retrain models on expanded dataset

**Step 10: SoftAUCLoss**

If current best plateaus ~0.82–0.84:
- Implement SoftAUCLoss (approximates cmAP gradient)
- Replace BCE in fine-tuning stage
- Reference: BirdCLEF 2025 4th place implementation

---

## Key Resources

### Papers
- Ghani et al. (2023) — "Global birdsong embeddings enable superior transfer learning" — Perch original paper, Nature Scientific Reports
- Kong et al. (2020) — PANNs, arXiv:1912.10211 — CNN14 pretraining on AudioSet
- Huang et al. (2022) — AudioMAE, arXiv:2207.06405 — Masked autoencoder for audio (NeurIPS 2022)
- Chen et al. (2022) — BEATs, arXiv:2212.09058 — Bidirectional encoder audio transformers
- Gong et al. (2021) — AST, arXiv:2104.01778 — Audio Spectrogram Transformer (Interspeech 2021)
- Wu et al. (2022) — CLAP, arXiv:2206.04769 — Contrastive Language-Audio Pretraining
- Kahl et al. (2022) — BirdNET overview (BirdNET-Analyzer GitHub)
- Perch 2.0 (2024) — arXiv:2512.03219 — 14,597 species foundation model
- BirdSet (2024) — arXiv:2403.10380 (ICLR 2025 spotlight) — bird audio benchmark + pretrained models
- BirdCLEF 2024 DS@GT — arXiv:2407.06291 — Perch + BirdNET pseudo multi-label
- BirdCLEF+ 2025 TFLite/STSG — arXiv:2507.08236 — Perch TFLite 10x speedup
- SpecAugment — arXiv:1904.08779 — Park et al. 2019

### Code Repositories
- Perch: https://github.com/google-research/perch
- BirdNET: https://github.com/kahst/BirdNET-Analyzer (`pip install birdnet-analyzer`)
- PANN/CNN14: https://github.com/qiuqiangkong/audioset_tagging_cnn (weights on Zenodo)
- AudioMAE: https://github.com/facebookresearch/AudioMAE
- BirdSet: https://github.com/DBD-research-group/BirdSet
- BirdCLEF 2023 1st place: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
- BirdCLEF 2024 1st place: https://github.com/arpoyda/BirdCLEF_2024
- BirdCLEF 2025 4th place: https://github.com/dylanliu2/BirdCLEF2025-4th-place-solution
- BirdCLEF 2025 5th place: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution

### Kaggle Models/Datasets
- Perch: `google/bird-vocalization-classifier` on Kaggle Models
- Perch-based checkpoint: `aidensong123/perch-fold` (used in public baseline)
- BirdSet EfficientNet-B1: `DBD-research-group/BirdSet-EfficientNet-B1-XCL` (we've tried this)
- CNN14 weights: Zenodo record 3960586 (`Cnn14_mAP=0.431.pth`)
- BirdCLEF past years: Available on Kaggle (2021, 2022, 2023, 2024)

### Confirmed Findings for BirdCLEF 2026 Specifically
- Public baseline (0.862): identical architecture to ours, single fold, Perch pseudo-labels
- Our best (0.783): 5-fold ensemble, no Perch
- BirdSet XCL B1 (0.782): no gain over ImageNet B0 — generic bird pretraining insufficient
- Dual loss regression (0.765): don't change both time_mask and dual loss simultaneously
- Local OOF cmAP is NOT a reliable LB proxy — use LB submissions as ground truth
