# Alternative Backbone Research

Gap to leader: **0.915 → 0.940**. Analysis suggests this requires a fundamentally different
base model, not incremental tuning of the current Perch+ProtoSSM stack. All ideas here follow
the **offline precompute artifact pattern** already established for `jaejohn/perch-meta`:
precompute embeddings on cluster → upload as Kaggle dataset → load in scoring kernel with
near-zero runtime cost.

-----

## 1. AVES — Animal Vocalization Encoder

HuBERT-style self-supervised model pretrained on a large corpus of animal sounds.
Bioacoustics-specific, so embeddings may be complementary to Perch's (different training
objective and data mixture → different residual errors).

**Why it matters**: Perch is trained for 10,932 bird species with supervised labels. AVES is
self-supervised on a broader animal sound corpus. For the non-Aves classes (Amphibia, Insecta,
Mammalia, Reptilia) where Perch relies on genus-level proxies, AVES embeddings may be more
directly discriminative.

**TODOs**:

- [ ] Download AVES checkpoint (`biodiversityml/aves-base-all` on HuggingFace)
- [ ] Write `data/precompute_aves_embeddings.py` — mirror structure of `precompute_probe_scores.py`; output `full_emb_aves.npy` (N×768) for the 59 labeled soundscapes
- [ ] Benchmark embedding quality: fit LogReg probes on AVES embeddings, compare OOF cmAP vs Perch baseline (0.926)
- [ ] If OOF cmAP ≥ 0.85: upload as Kaggle dataset `birdclef2026-aves-meta`
- [ ] Extend ProtoSSM Stage 1 to accept `concat(perch_emb 1536, aves_emb 768)` = 2304-dim input; retrain
- [ ] Submit and compare LB

**Risk**: AVES 768-dim vs Perch 1536-dim — verify input shape compatibility with ProtoSSM.

-----

## 2. BEATs — Self-Supervised Audio Tokenizer (Microsoft)

Iterative masked audio modelling, SOTA on several environmental sound benchmarks. General
audio (not bioacoustics-specific), but strong transfer to novel sound classes.

**Why it matters**: Their gap analysis explicitly states reaching 0.940 requires "a
fundamentally better base model." BEATs is the strongest general-purpose frozen audio extractor
not yet tried. Its training objective (discrete token prediction) is orthogonal to Perch's
(species classification), so embeddings should have low correlation with Perch's errors.

**TODOs**:

- [ ] Download BEATs checkpoint (`microsoft/BEATs` on HuggingFace, `BEATs_iter3_plus_AS2M.pt`)
- [ ] Write `data/precompute_beats_embeddings.py`; resample soundscape windows to 16 kHz (BEATs default), output `full_emb_beats.npy` (N×768)
- [ ] Run OOF cmAP probe benchmark (same as AVES step above)
- [ ] If promising: train ProtoSSM with `concat(perch_emb, beats_emb)` input
- [ ] If budget allows: test BEATs-only ProtoSSM as a diversity ensemble partner

**Note**: BEATs expects 16 kHz input; soundscapes are 32 kHz — add resampling step.

-----

## 3. CLAP — Zero-Shot for the 28 Zero-Shot Species

LAION Contrastive Language-Audio Pretraining. Supports text queries against audio embeddings,
enabling zero-shot classification without any audio examples.

**Why it matters**: 28 target species have no XC/iNat training audio (mostly insect sonotypes
`47158son*`). Current pipeline scores these at near-chance. Even a weak CLAP prior would beat
random for these classes and costs no additional inference time if precomputed.

**TODOs**:

- [ ] Install `msclap` (`pip install msclap`) and download CLAP 2023 checkpoint
- [ ] Write `data/precompute_clap_zero_shot.py`:
  - For each of the 28 zero-shot species, craft text queries (e.g. `"insect stridulation in tropical wetland"`, `"frog call in Pantanal"`)
  - Compute CLAP audio embeddings for all 59 soundscape windows
  - Output per-window cosine similarity scores for the 28 species → `clap_zeroshot_scores.npy`
- [ ] Evaluate on labeled soundscapes: does CLAP rank positive windows above background for these sonotypes?
- [ ] If any signal: add as a fixed prior in the inference notebook for zero-shot species only (no model change needed)
- [ ] Iterate text query wording to maximise per-class AP on the 14 held-out soundscapes

**Risk**: Sonotypes are not linguistically well-defined; text query quality will determine
whether this works. Low cost to try — no retraining required.

-----

## 4. MixIT — Unsupervised Source Separation (Preprocessing)

Google's Mix of Mixtures source separation. Separates a multi-species soundscape into
individual source tracks without labels.

**Why it matters**: Perch embeddings are computed on 5s soundscape windows containing multiple
overlapping species. Separating sources before embedding could sharpen species-specific
signal, especially for rare or quiet callers masked by louder dominant species. This is an
upstream preprocessing change — compatible with the existing pipeline.

**TODOs**:

- [ ] Clone `google-research/sound-separation` and download MixIT YFCC checkpoint
- [ ] Write `data/precompute_mixit_separation.py`:
  - For each soundscape window, run MixIT to produce K=4 separated source tracks
  - Compute Perch embeddings on each source track
  - Output `full_emb_mixit.npy` shape (N, K, 1536); aggregate by max-pool across sources
- [ ] Compare OOF cmAP of max-pooled MixIT embeddings vs direct Perch embeddings
- [ ] If OOF ≥ 0.930: upload separated embeddings as Kaggle dataset, retrain ProtoSSM
- [ ] **Also test**: use MixIT as diversity signal — add per-source Perch logit variance as an
  additional probe feature (cheap, no architecture change)

**Risk**: MixIT may not separate cleanly on highly overlapping tropical soundscapes. Runtime on
cluster should be benchmarked before committing — separation can be slow.
