# Local Evaluation Strategy — BirdCLEF 2026

## The Core Problem

Only a small subset of `train_soundscapes` has expert annotations. Three data sources exist with very different trust levels:

|Source                       |Label Quality                  |Format        |Test Proximity           |
|-----------------------------|-------------------------------|--------------|-------------------------|
|`train_audio` (XC/iNat)      |Clean, single-species          |Short clips   |Low                      |
|Labeled `train_soundscapes`  |Expert-annotated, multi-species|5-sec segments|**High**                 |
|Unlabeled `train_soundscapes`|None                           |5-sec segments|High (but unusable as GT)|

**Critical:** Segments absent from `train_soundscapes_labels.csv` are **not confirmed negatives** — they are simply unannotated.

-----

## Strategy 1: Site-Stratified Hold-Out

Split by **recording site** (extracted from filenames like `_S05_`), not by segment or file. Sites overlap between train and test soundscapes, so a site-based split gives a realistic out-of-distribution estimate.

```python
labels["site"] = labels["filename"].str.extract(r'_(S\d+)_')
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(gss.split(labels, groups=labels["site"]))
```

Random segment splitting lets the model see every site's acoustic conditions during training — inflating local CV scores.

-----

## Strategy 2: 5-Fold CV Grouped by Soundscape File

To maximise use of the scarce labeled data, run 5-fold CV. Group by **soundscape file**, not by segment — segments within the same 1-minute recording are temporally correlated and must stay together.

```python
labels["soundscape_id"] = labels["filename"].str.replace(".ogg", "")
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(labels, groups=labels["soundscape_id"]))
```

-----

## Strategy 3: Implement Padded cMAP

The competition metric is **padded class-wise mean Average Precision**. Unpadded AP is optimistic; implement it exactly:

```python
def padded_cmap(y_true, y_pred, padding_factor=5):
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue  # Skip species absent from val set
        col_true = np.concatenate([y_true[:, c], np.zeros(padding_factor)])
        col_pred = np.concatenate([y_pred[:, c], np.zeros(padding_factor)])
        aps.append(average_precision_score(col_true, col_pred))
    return np.mean(aps) if aps else 0.0
```

Only evaluate species that have at least one positive label in the local val set, mirroring the hidden test scorer.

-----

## Strategy 4: Treat Unlabeled Segments Correctly

When building the ground-truth matrix, treat unannotated segments as all-zeros — but track this assumption as a known limitation. The eval will slightly underestimate true performance for species calling quietly in unannotated windows.

```python
for end_t in range(5, 65, 5):          # All 12 segments per minute
    seg = file_labels[file_labels["end"] == end_t]
    for sp in species_list:
        row[sp] = 1 if (len(seg) > 0 and sp in seg_species) else 0
```

-----

## Strategy 5: Track Soundscape-Only Species Separately

The competition warns that some species appear **only** in labeled soundscapes, not in `train_audio`. These are the hardest to model and most likely to be your weakest point.

```python
audio_species      = set(train_df["primary_label"].unique())
soundscape_species = set(";".join(labels["primary_label"]).split(";"))
soundscape_only    = soundscape_species - audio_species

# Always report three numbers:
# 1. Overall cMAP
# 2. cMAP on audio-present species
# 3. cMAP on soundscape-only species  ← primary weakness signal
```

If soundscape-only AP is much lower, focus on better utilisation of labeled soundscapes (fine-tuning, higher sample weight, etc.).

-----

## Strategy 6: Per-Class AP Breakdown

Add a per-class report to diagnose systematic failures:

```python
def per_class_ap_report(y_true, y_pred, species_names):
    aps = {sp: average_precision_score(y_true[:, i], y_pred[:, i])
           for i, sp in enumerate(species_names) if y_true[:, i].sum() > 0}
    report = pd.Series(aps).sort_values()
    print(report.head(20))   # Worst species
    print(report.tail(20))   # Best species
    return report
```

Look for patterns: failures concentrated in a taxonomic class (e.g., all insects low), or in data-sparse species.

-----

## Summary of Key Principles

|Principle                                                         |Reason                                                      |
|------------------------------------------------------------------|------------------------------------------------------------|
|Split by **soundscape file**, not segment                         |Prevents temporal leakage within a 1-min recording          |
|Split by **site** for hold-out                                    |Realistic OOD estimate matching test conditions             |
|Use **padded cMAP** with correct padding factor                   |Matches competition scorer; unpadded AP is too optimistic   |
|Track **soundscape-only species** separately                      |Pinpoints your biggest vulnerability                        |
|Never treat **unlabeled segments as hard negatives** in evaluation|Avoids penalising correct detections in unannotated windows |
|Report **per-class AP breakdown**                                 |Guides which species/taxonomic class to focus improvement on|
