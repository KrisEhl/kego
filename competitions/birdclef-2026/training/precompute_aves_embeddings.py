"""Precompute WavLM embeddings for the 59 labeled soundscapes.

Uses microsoft/wavlm-base-plus — a self-supervised audio model trained on speech (WavLM,
Chen et al. 2022) that achieves SOTA on the SUPERB benchmark. Unlike Perch which is
supervised on bird species, WavLM has a fundamentally different training objective (masked
prediction with denoising), producing embeddings that are potentially complementary to Perch's
for novel or out-of-distribution species.

Original plan was biodiversityml/aves-base-all (AVES, Earth Species Project) but that model
is gated on HuggingFace. WavLM-base-plus is publicly accessible and similarly sized (768-dim).

Expected output:
    data/perch-meta/full_emb_wavlm.npy  — (N, 768) float32, N = 708 windows
    (same row order as full_perch_arrays.npz)

Run on cluster (needs GPU for reasonable speed, ~5 min):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    uv run python competitions/birdclef-2026/training/precompute_aves_embeddings.py

After running, probe benchmark is built in to the script — check printed OOF cmAP.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
SOUNDSCAPE_DIR = DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes"
PERCH_META_PARQUET = DATA_ROOT / "perch-meta/full_perch_meta.parquet"
PERCH_NPZ = DATA_ROOT / "perch-meta/full_perch_arrays.npz"
TAXONOMY_CSV = DATA_ROOT / "birdclef/birdclef-2026/taxonomy.csv"
LABELS_CSV = DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes_labels.csv"
TRAIN_CSV = DATA_ROOT / "birdclef/birdclef-2026/train.csv"
OUT_EMB = DATA_ROOT / "perch-meta/full_emb_wavlm.npy"

SAMPLE_RATE = 16000  # WavLM expects 16kHz
WINDOW_SEC = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
WAVLM_MODEL = "microsoft/wavlm-base-plus"


def load_window_16k(path: Path, start_sec: float) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, offset=start_sec, duration=WINDOW_SEC)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    return y[:WINDOW_SAMPLES].astype(np.float32)


def get_aves_embeddings(model, feature_extractor, audio_batch: list[np.ndarray], device: str) -> np.ndarray:
    """Run AVES on a list of 5-second waveforms, return mean-pooled (N, 768) embeddings."""
    inputs = feature_extractor(
        audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=False)
    # last_hidden_state: (batch, frames, 768) — mean-pool over frames
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb.astype(np.float32)


def build_ground_truth(meta, labels_csv, taxonomy):
    primary_labels = taxonomy["primary_label"].tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(primary_labels)}

    def parse_end(s):
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    ldf = labels_csv.copy()
    ldf["end_sec"] = ldf["end"].apply(parse_end)
    ldf["stem"] = ldf["filename"].str.replace(".ogg", "", regex=False)
    ldf["row_id"] = ldf["stem"] + "_" + ldf["end_sec"].astype(str)
    rid_to_species: dict[str, set] = {}
    for _, row in ldf.iterrows():
        rid = row["row_id"]
        rid_to_species.setdefault(rid, set())
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                rid_to_species[rid].add(lbl)

    n = len(meta)
    Y_true = np.zeros((n, len(primary_labels)), dtype=np.float32)
    for i, row_id in enumerate(meta["row_id"]):
        for sp in rid_to_species.get(row_id, set()):
            Y_true[i, label_to_idx[sp]] = 1.0
    return Y_true, primary_labels


def padded_cmap(y_true, y_pred, padding_factor=5):
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        col_true = np.concatenate([y_true[:, c], np.zeros(padding_factor)])
        col_pred = np.concatenate([y_pred[:, c], np.zeros(padding_factor)])
        aps.append(average_precision_score(col_true, col_pred))
    return float(np.mean(aps)) if aps else 0.0


def oof_probe_cmap(emb: np.ndarray, Y_true: np.ndarray, meta: pd.DataFrame) -> float:
    """Leave-one-file-out LogReg probe benchmark on AVES embeddings."""
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)

    meta = meta.reset_index(drop=True)
    meta["stem"] = meta["row_id"].str.rsplit("_", n=1).str[0]
    unique_files = meta["stem"].unique()
    n_cls = Y_true.shape[1]
    oof_preds = np.zeros_like(Y_true, dtype=np.float32)

    active_cls = np.where(Y_true.sum(axis=0) >= 2)[0]
    print(f"  OOF probe: {len(unique_files)} files, {len(active_cls)} active classes (≥2 pos)")

    for fold_i, held_file in enumerate(unique_files):
        if fold_i % 10 == 0:
            print(f"  Fold {fold_i}/{len(unique_files)}")
        val_mask = meta["stem"].values == held_file
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        for c in active_cls:
            if Y_true[train_mask, c].sum() < 1:
                continue
            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
            clf.fit(emb_scaled[train_mask], Y_true[train_mask, c])
            oof_preds[val_mask, c] = clf.predict_proba(emb_scaled[val_mask])[:, 1]

    return padded_cmap(Y_true, oof_preds)


def main():
    from transformers import AutoFeatureExtractor, WavLMModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load meta and build row order (must match full_perch_arrays.npz)
    meta = pd.read_parquet(PERCH_META_PARQUET)
    npz = np.load(PERCH_NPZ)
    print(f"Total windows: {len(meta)}")

    # Load WavLM
    print(f"\nLoading WavLM model: {WAVLM_MODEL}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL)
    model = WavLMModel.from_pretrained(WAVLM_MODEL).to(device).eval()
    print(f"WavLM loaded. Param count: {sum(p.numel() for p in model.parameters()):,}")

    # Extract embeddings in file order matching meta
    meta["stem"] = meta["row_id"].str.rsplit("_", n=1).str[0]
    meta["end_sec"] = meta["row_id"].str.rsplit("_", n=1).str[1].astype(int)
    meta = meta.sort_values(["stem", "end_sec"]).reset_index(drop=True)
    unique_stems = meta["stem"].unique()

    all_emb = np.zeros((len(meta), 768), dtype=np.float32)
    BATCH = 8

    for fi, stem in enumerate(unique_stems):
        if fi % 10 == 0:
            print(f"  File {fi}/{len(unique_stems)}: {stem}", flush=True)

        ogg_path = SOUNDSCAPE_DIR / f"{stem}.ogg"
        if not ogg_path.exists():
            print(f"  WARNING: {ogg_path} not found")
            continue

        file_rows = meta[meta["stem"] == stem].sort_values("end_sec")
        row_indices = file_rows.index.tolist()
        end_secs = file_rows["end_sec"].tolist()

        # Batch windows
        audios = [load_window_16k(ogg_path, float(e - WINDOW_SEC)) for e in end_secs]

        for i in range(0, len(audios), BATCH):
            batch = audios[i : i + BATCH]
            batch_idx = row_indices[i : i + BATCH]
            emb_batch = get_aves_embeddings(model, feature_extractor, batch, device)
            all_emb[batch_idx] = emb_batch

    np.save(OUT_EMB, all_emb)
    print(f"\nSaved WavLM embeddings: {all_emb.shape} → {OUT_EMB}")
    print(
        f"Emb range: [{all_emb.min():.4f}, {all_emb.max():.4f}]  "
        f"norm mean: {np.linalg.norm(all_emb, axis=1).mean():.2f}"
    )

    # ---- Probe benchmark ----
    print("\n--- OOF LogReg probe benchmark ---")
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    labels_csv = pd.read_csv(LABELS_CSV)
    Y_true, primary_labels = build_ground_truth(meta, labels_csv, taxonomy)

    # WavLM-only probes
    wavlm_cmap = oof_probe_cmap(all_emb, Y_true, meta)
    print(f"\nWavLM-only OOF cmAP : {wavlm_cmap:.4f}")

    # Perch-only baseline for comparison
    perch_logits = npz["logits"] if "logits" in npz else None
    if perch_logits is not None:
        perch_cmap = padded_cmap(Y_true, 1 / (1 + np.exp(-perch_logits)))
        print(f"Perch logits cmAP   : {perch_cmap:.4f}  (in-sample, reference)")

    # Concat probes
    print("\nFitting concat(Perch emb 1536, WavLM emb 768) probes...")
    perch_emb = npz["embeddings"] if "embeddings" in npz else npz.get("emb")
    if perch_emb is not None:
        concat_emb = np.concatenate([perch_emb, all_emb], axis=1)
        concat_cmap = oof_probe_cmap(concat_emb, Y_true, meta)
        print(f"Concat OOF cmAP     : {concat_cmap:.4f}")
        print(f"Delta vs WavLM-only : {concat_cmap - wavlm_cmap:+.4f}")
        print(f"Delta vs Perch-only : {concat_cmap - perch_cmap:+.4f}")

    print("\nDone. Summary:")
    print(f"  WavLM cmAP  : {wavlm_cmap:.4f}")
    if perch_logits is not None:
        print(f"  Perch cmAP  : {perch_cmap:.4f}")
    if perch_emb is not None:
        print(f"  Concat cmAP : {concat_cmap:.4f}")


if __name__ == "__main__":
    main()
