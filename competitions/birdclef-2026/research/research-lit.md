I read your plan, and the strongest signals in it already line up with what the broader literature and competition practice tend to reward for multilabel ecoacoustic soundscapes: weakly supervised clip labeling plus temporal aggregation, domain-specific pretrained embeddings, and post-processing that explicitly respects the difference between transient bird calls and more stationary amphibian/insect choruses. Your own findings on BCE over CE, temporal smoothing, overlapping windows, and the usefulness of soundscape labels are all consistent with that direction  ￼  ￼

My bottom-line recommendation is this:
	1.	Keep your current fine-tuned CNN track, but treat it as only one half of the solution.
	2.	Push hard on a frozen-embedding or lightly-adapted bioacoustics foundation model track.
	3.	Put more effort into inference-time modeling of time, habitat/site, and taxonomic structure than into ever-larger backbones.
	4.	For the non-bird minority classes, use class-family-specific logic rather than forcing one global post-processing rule.

What follows is a field-oriented synthesis focused on multiclass or multilabel classification in soundscapes, not just clip classification.

What the field has learned

The animal sound classification literature has converged on a few robust ideas.

First, soundscape recognition is usually not “single-label classification.” It is closer to weakly supervised sound event detection or multilabel tagging. In practice, the best systems do not simply classify a whole 5 s clip. They produce framewise or short-window evidence, then pool, smooth, calibrate, and add priors. This is why your temporal smoothing and overlap tricks help so much  ￼

Second, pretrained audio embeddings matter disproportionately in low-label or long-tail ecological settings. Generic audio pretraining helps, but models trained on bird or wildlife data usually help more when the label space overlaps or can be mapped taxonomically. That makes your Perch direction unusually well grounded.

Third, post-processing is not a hack here. In ecoacoustics, temporal continuity, diel patterns, site occupancy, and taxonomy are genuine structure in the data. The literature often under-emphasizes this because papers frame it as “classification,” but the best applied systems quietly use priors, calibration, smoothing, and thresholding.

Fourth, augmentation is useful but easy to overdo. Simple waveform gain, time masking, SpecAugment, and mixup often help. Arbitrary background mixing can hurt if labels become contaminated by unlabeled species, which matches your observation that background noise augmentation was neutral  ￼

The approaches most relevant to your Kaggle setup

1) Weakly supervised SED / MIL is usually better than plain clip classification

For soundscapes with multiple overlapping taxa, the most transferable design pattern is:

audio → time-frequency representation or embedding sequence → framewise scores → attention/MIL pooling → clipwise multilabel probabilities

Why it matters for you:
	•	Bird calls are sparse and short.
	•	Frog and insect choruses are often sustained textures.
	•	Plain global pooling can dilute sparse events.
	•	Naive SED heads can fail if temporal resolution is too low, which you already saw with heavy downsampling  ￼

What tends to work:
	•	CRNN or CNN+attention pooling
	•	Generalized mean pooling or attention pooling
	•	Multi-instance learning with max/softmax/attention pooling
	•	Higher temporal resolution backbones or less aggressive downsampling
	•	Multi-scale heads so one branch handles transient calls and another handles textures

Practical implication:
Your “naive SED head” likely failed because architecture, not idea, was wrong. I would not abandon weakly supervised temporal modeling; I would reintroduce it with a higher-frame-rate frontend, dilated temporal convolutions, or a transformer/Conformer head on top of CNN features.

2) Domain pretrained bioacoustic models are a major lever

The strongest recent shift in the field is away from training from scratch on competition data and toward using large pretrained models as frozen or semi-frozen feature extractors.

Most relevant families:
	•	Perch / Google bird vocalization foundation models
Especially good when you can map target taxa to a much larger training vocabulary, or use embeddings plus lightweight downstream probes. This matches the strongest public track in your plan  ￼
	•	BirdNET
Strong bird-specialized embeddings/classifier family. Even if not directly usable end-to-end for your label space, BirdNET-style embeddings can be an excellent side-channel or teacher model for birds.
	•	PANNs / AudioSet-pretrained CNNs
Often strong general-purpose audio starting points, especially for species with less direct overlap to bird-specialized models.
	•	AST / PASST / HTS-AT / BEATs-style transformers
These can work well, but in CPU-constrained Kaggle settings the cost-benefit may be worse than efficient CNNs plus strong inference logic.
	•	BirdSet-style pretraining
Conceptually appealing, but your current result says generic bird-domain pretraining alone did not transfer strongly in this exact setup  ￼ That is not surprising: pretraining helps most when label granularity, geography, and recording conditions align well, or when the downstream adaptation is strong.

Practical implication:
I would prioritize a 2-branch ensemble:
	•	Branch A: Perch embeddings/logits + simple downstream probes + taxonomic mapping
	•	Branch B: Your fine-tuned soundscape CNN
Then calibrate and blend.

3) Class-conditional post-processing is underused and very valuable

This is one of the clearest insights from both your plan and the literature.

Birds often behave like discrete acoustic events.
Amphibians and many insects behave like persistent or quasi-stationary textures.

That means the optimal temporal prior differs by class family:
	•	birds: local max propagation, sparse-event hysteresis, short-context smoothing
	•	frogs/insects: neighborhood averaging, persistence constraints, slower temporal smoothing
	•	mammals: species dependent, often sparse but lower-frequency

This is a very defensible design, not competition overfitting. Your current class-type-aware smoothing is exactly the kind of biologically informed post-processing that the field tends to reward  ￼

I would go further:
	•	learn per-class smoothing parameters from validation
	•	use separate threshold calibration by taxonomic group
	•	model onset-like versus texture-like classes with different pooling rules
	•	add persistence penalties for chorusing taxa and burstiness priors for birds

4) Context priors often matter as much as model architecture

Ecoacoustic data are not i.i.d. Species presence depends on:
	•	site
	•	habitat
	•	season
	•	time of day
	•	weather
	•	local species co-occurrence

The literature contains many examples where occupancy-style or metadata-informed models improve detection. In competitions, this often appears as:
	•	site×hour prior
	•	file-level prior
	•	co-occurrence prior
	•	taxonomy-aware smoothing
	•	threshold calibration per location or recorder

Your file-max prior and planned site×hour prior fit this pattern very well  ￼

I would add two more structured priors:
	•	co-occurrence graph smoothing: increase probability of species that strongly co-occur with already high-probability species, but only within habitat/time constraints
	•	taxonomic backoff: if species evidence is weak but genus-level evidence is strong, distribute confidence carefully across candidate species

That second point is especially relevant for your unmapped amphibian/insect classes.

Pre-processing methods that are genuinely useful

Time-frequency representation choices

For bird and mixed soundscape work, mel spectrograms remain the dominant practical representation. The field does not show a universal win for more exotic frontends over a strong mel pipeline, especially in competition settings.

What does repeatedly matter:
	•	enough frequency coverage for insects and some frogs
	•	enough temporal resolution for short bird notes
	•	log or dB compression
	•	consistent normalization
	•	sometimes PCEN instead of log-mel in variable background conditions

For your case, the literature would support trying:
	•	log-mel and PCEN as parallel channels
	•	2-channel input: log-mel + PCEN
	•	multi-resolution mel: one branch with short window/hop, one with longer context
	•	optional harmonic-percussive decomposition, mainly if you see many anthropogenic or rain-like confounders

Your high-scoring mel setup is very plausible already; I would not spend much time on frontend novelty beyond adding a second channel or multiresolution branch  ￼

Denoising and source separation

Usually helpful only when used conservatively.

Potentially useful:
	•	high-pass filtering to remove low-frequency rumble
	•	stationary noise reduction for recorder hum/wind
	•	rain/noise detectors to gate or downweight poor segments
	•	band-limited enhancement for taxa with known frequency ranges

Often harmful:
	•	aggressive denoising that distorts harmonics
	•	hard VAD that removes quiet calls
	•	blind source separation unless tuned very carefully

For your challenge, I would treat denoising as a data-quality branch, not a universal preprocessing step. One practical approach is to compute a “quality/confidence” score and use it to modulate thresholds rather than trying to fully clean every recording.

Segmentation

The literature supports overlapped windows and event localization more than non-overlapping chunks. Your 50% stride gain is exactly what I would expect in soundscapes with boundary effects  ￼

You can push this further with:
	•	multi-offset inference, not just one circular shift
	•	short-window auxiliary detector for sparse calls
	•	boundary-aware aggregation where windows near chunk edges contribute to both adjacent chunks

Augmentation methods: what usually works and what often fails

Usually useful
	•	random gain / amplitude scaling
	•	time masking and frequency masking
	•	SpecAugment-style masking
	•	mixup, especially for multilabel tagging
	•	cutmix on spectrograms, cautiously
	•	time shifting / circular shift
	•	small pitch shifts only if biologically plausible
	•	additive noise from matched domain backgrounds, but only with label caution

Often mixed or risky
	•	arbitrary background mixing from unlabeled soundscapes
	•	large pitch shifts that move species out of plausible range
	•	strong time-stretching that changes call cadence unnaturally
	•	pseudo-labeling on overwhelmingly negative windows

Your own result that pseudo-labeling unlabeled soundscapes with near-zero signal did not help is completely consistent with the weakly supervised bioacoustics literature: pseudo-labeling helps when the teacher has confident positives, not when nearly all windows are negative or ambiguous  ￼

One augmentation family I think you should test more deliberately is hard-negative mining rather than more positive augmentation:
	•	rain
	•	insects when target is bird
	•	human voice
	•	mechanical noise
	•	geophony
This often improves precision on multilabel soundscapes more than another round of generic masking.

Alternative modeling strategies beyond standard deep CNNs

You asked not to limit to deep learning, so here is the realistic landscape.

Classical pipeline on learned embeddings

A very strong non-end-to-end baseline is:
	•	pretrained embedding extractor
	•	PCA or supervised dimensionality reduction
	•	one-vs-rest logistic regression / linear SVM / gradient boosting
	•	per-class threshold calibration

This is especially attractive for:
	•	CPU-bound inference
	•	small fully labeled soundscape subsets
	•	zero-shot or taxonomic backoff cases

In fact, your planned Perch + PCA + logistic regression probes is not a “cheap approximation”; it is a method family with strong support in bioacoustics and audio tagging practice  ￼

HMM / CRF / occupancy-style temporal models

Older but still useful as post-processing:
	•	HMMs for temporal persistence
	•	CRFs for smoothing adjacent windows
	•	dynamic occupancy models with site/time priors

These are rarely fashionable in competition writeups, but they remain quite sensible when the acoustic process is structured. For frogs/insects, a persistence model can be very effective.

Template matching / spectrographic correlation

Not competitive as a full solution for 234 classes in dense soundscapes, but still useful for:
	•	a small set of distinctive species
	•	weak-label verification
	•	pseudo-label filtering
	•	error analysis

Hierarchical classification

Underused in Kaggle, but biologically reasonable:
	1.	detect broad taxonomic/acoustic type
	2.	specialize within branch
	3.	backoff to genus/family when species evidence is weak

This could be valuable for your mixed Aves/Amphibia/Insecta label space and especially for the zero-shot or weak-shot tails.

Concrete ideas I would add to your roadmap

Highest expected value

A. Perch ensemble, not replacement
Do not treat Perch as an alternative to your CNN. Treat it as an orthogonal expert. The strongest likely endgame is:
	•	Perch logits mapped to target species
	•	Perch embeddings with per-class probes
	•	your soundscape CNN
	•	calibrated blend

B. Learn per-class thresholds and temperatures
Global thresholding is rarely optimal in long-tail multilabel ecology. Learn:
	•	per-class temperature or Platt scaling
	•	per-class threshold
	•	separate calibration by class family

C. Per-class or per-family temporal kernels
You already have two families. Extend to 3–5 acoustic archetypes:
	•	transient tonal birds
	•	repetitive bird song
	•	chorusing frogs
	•	continuous insects
	•	rare low-frequency mammals/reptiles

D. Use a 2-channel frontend
log-mel + PCEN is one of the most plausible “small engineering, real gain” changes.

E. Multi-resolution inference
Short windows for bird notes, longer windows for choruses. Even without retraining, this can be done on the embedding branch.

Medium expected value

F. Co-occurrence graph prior
Build a sparse prior from training soundscapes:
	•	species-species PMI or conditional probability
	•	optionally conditioned on site/hour
Then apply a small correction only to borderline predictions.

G. Hierarchical taxonomic backoff
For weak or unseen non-bird species:
	•	map to genus/family proxies
	•	infer species score from taxonomy-aware redistribution
This is more principled than a flat max proxy.

H. Hard-negative curriculum
Mine false positives and fine-tune specifically against them.

Lower expected value

I. Larger CNNs
Worth trying only if they fit the inference budget. Your own results already suggest diminishing returns here  ￼

J. Aggressive pseudo-labeling of unlabeled soundscapes
I would deprioritize this unless confidence filtering becomes much stronger.

What I would do next in your exact situation

Given your current LB and plan, I would reorder priorities slightly.
	1.	Finish the Perch pipeline fully.
	2.	Add probe training on embeddings, but also try direct one-vs-rest linear classifiers on raw embeddings before PCA, because sometimes PCA hurts rare classes.
	3.	Learn per-class calibration and thresholds on the soundscape-labeled subset.
	4.	Blend Perch branch with your CNN branch.
	5.	Add a lightweight context layer: site×hour prior plus family-specific smoothing.
	6.	Only then revisit backbone or extra pretraining.

I would also explicitly test these ablations:
	•	Perch logits only
	•	Perch embeddings only
	•	Perch logits + probes
	•		•	genus/family backoff
	•		•	per-class thresholds
	•		•	class-family smoothing
	•		•	site×hour prior
	•		•	blend with CNN

That sequence will tell you whether the gain is coming from recognition, calibration, or ecological context.

Papers and lines of work most worth reading

I am grouping these by why they matter to you rather than by date.

Core animal/bird sound recognition and ecoacoustics

Stowell, D., Wood, M., Stylianou, Y., & Glotin, H. (2019). Automatic acoustic detection of birds through deep learning: the first Bird Audio Detection challenge. Methods in Ecology and Evolution.
Why read it: foundational for weak labels, noisy field audio, and evaluation mindset in bird acoustics.

Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics.
Why read it: one of the most practically important bird classification systems; relevant for pretrained bird-specialized inference and deployment ideas.

Lostanlen, V., Salamon, J., Farnsworth, A., et al. (2019). Per-channel energy normalization: Why and how? IEEE Signal Processing Letters / related ecoacoustic adoption literature.
Why read it: PCEN is highly relevant for noisy, nonstationary field recordings.

Salamon, J., & Bello, J. P. (2017). Deep convolutional neural networks and data augmentation for environmental sound classification. IEEE Signal Processing Letters.
Why read it: classic augmentation paper; many lessons transfer directly to wildlife audio.

Weakly supervised audio tagging / sound event detection

Cakir, E., Parascandolo, G., Heittola, T., Huttunen, H., & Virtanen, T. (2017). Convolutional recurrent neural networks for polyphonic sound event detection. IEEE/ACM Transactions on Audio, Speech, and Language Processing.
Why read it: still one of the clearest templates for overlapping-label audio.

Kong, Q., Xu, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM TASLP.
Why read it: practical pretraining recipe and strong embeddings for audio tagging.

Hershey, S., Chaudhuri, S., Ellis, D. P. W., et al. (2017). CNN architectures for large-scale audio classification. ICASSP.
Why read it: strong grounding for AudioSet-pretrained audio CNNs that often transfer to ecoacoustics.

Gemmeke, J. F., Ellis, D. P. W., Freedman, D., et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. ICASSP.
Why read it: not wildlife-specific, but central to the pretraining ecosystem.

Recent foundation / transfer learning for birds and ecoacoustics

Kahl, S., Pardo, B., et al. (2024 or near-recent BirdSet work). BirdSet: A large-scale benchmark for avian bioacoustics.
Why read it: benchmark framing, transfer learning expectations, and why some bird-domain pretraining helps less than hoped when label mismatch is large.

Google bioacoustics / Perch line of work (recent foundation-model papers and technical reports around Perch / bird vocalization classifier).
Why read it: closest match to your current strongest next step; especially useful for zero-shot mapping and embedding-based downstream classification.

Classical or non-deep-learning strategies still worth borrowing from

Briggs, F., Raich, R., & Fern, X. Z. (2012). Acoustic classification of multiple simultaneous bird species: A multi-instance multi-label approach. Journal of the Acoustical Society of America / related conference literature.
Why read it: directly relevant to multilabel bird mixtures before the deep-learning wave.

Stowell, D., & Plumbley, M. D. (2014). Automatic large-scale classification of bird sounds is strongly improved by unsupervised feature learning. PeerJ.
Why read it: useful reminder that representation learning plus simple classifiers can go very far.

Ecoacoustic context, occupancy, and metadata

Aide, T. M., Corrada-Bravo, C., Campos-Cerqueira, M., Milan, C., Vega, G., & Alvarez, R. (2013). Real-time bioacoustics monitoring and automated species identification. PeerJ.
Why read it: older, but valuable for the practical view that metadata and monitoring context matter.

Ruff, Z. J., Lesmeister, D. B., Duchac, L. S., et al. (2020). Automated identification of avian vocalizations with deep convolutional neural networks. Remote Sensing in Ecology and Conservation.
Why read it: practical deployment perspective and field-noise issues.

How these papers map onto your plan

Your current plan already captures several field-backed ideas:
	•	BCE instead of CE for multilabel soundscapes  ￼
	•	overlap inference and temporal smoothing  ￼
	•	leveraging soundscape labels rather than only clip labels  ￼
	•	moving toward a large wildlife-specific frozen model plus lightweight downstream heads  ￼

Where the literature suggests expanding the plan:
	•	reintroduce temporal heads, but with better time resolution
	•	add PCEN or multichannel frontends
	•	learn class-specific calibration and smoothing
	•	treat taxonomy and metadata as first-class modeling inputs
	•	use embedding-based linear models as serious ensemble members, not just baselines

My strongest single recommendation

The biggest likely jump is not another CNN variant. It is a calibrated ensemble of:
	•	Perch logits
	•	Perch embedding probes
	•	your fine-tuned CNN
	•	family-aware temporal smoothing
	•	site/hour/file priors
	•	per-class thresholds

That combination best matches both the literature and the empirical pattern in your own plan.

You’ve experienced ScholarGPT — now meet what’s next.
Scholar Deep Research Agent elevates your research game with:
🔍 350M+ trusted papers from top academic publishers, updated hourly.
🧠 Advanced multiple AI models dig through millions of sources for pinpoint insights, fast.
📝 Auto-generated highlights, smart notes, and visual reports
📁 All saved directly to your AI-powered knowledge base
ScholarGPT helped you search. Now, transform how you think.
Explore Scholar Deep Research￼
