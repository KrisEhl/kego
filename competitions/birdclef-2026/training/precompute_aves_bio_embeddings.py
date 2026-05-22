"""Precompute AVES embeddings for the 59 labeled soundscapes.

Uses publicly downloadable checkpoints from ESP (Earth Species Project) on GCS:
  - aves-base-bio.torchaudio.pt   (768-dim, 12-layer, trained on animal sounds)
  - birdaves-biox-large.torchaudio.pt (1024-dim, trained specifically on bird audio)

Both use the wav2vec2 architecture via torchaudio — no HuggingFace auth needed.

Download (run once on cluster):
    cd data/perch-meta
    curl -L -o aves-base-bio.pt \
      'https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt'
    curl -L -o birdaves-biox-large.pt \
      'https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-large.torchaudio.pt'

Run:
    KEGO_PATH_DATA=/home/kristian/projects/kego/data AVES_MODEL=base \\
    uv run python competitions/birdclef-2026/training/precompute_aves_bio_embeddings.py

    KEGO_PATH_DATA=/home/kristian/projects/kego/data AVES_MODEL=large \\
    uv run python competitions/birdclef-2026/training/precompute_aves_bio_embeddings.py

Outputs:
    data/perch-meta/full_emb_aves_base.npy   — (N, 768) for base
    data/perch-meta/full_emb_aves_large.npy  — (N, 1024) for large
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
SOUNDSCAPE_DIR = DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes"
PERCH_META_PARQUET = DATA_ROOT / "perch-meta/full_perch_meta.parquet"
PERCH_NPZ = DATA_ROOT / "perch-meta/full_perch_arrays.npz"
TAXONOMY_CSV = DATA_ROOT / "birdclef/birdclef-2026/taxonomy.csv"
LABELS_CSV = DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes_labels.csv"

# Select model via env var: AVES_MODEL=base or large
AVES_MODEL = os.environ.get("AVES_MODEL", "large")

if AVES_MODEL == "large":
    CKPT_PATH = DATA_ROOT / "perch-meta/birdaves-biox-large.pt"
    EMB_DIM = 1024
    OUT_EMB = DATA_ROOT / "perch-meta/full_emb_aves_large.npy"
    TORCHAUDIO_BUNDLE = torchaudio.models.wav2vec2_large
else:
    CKPT_PATH = DATA_ROOT / "perch-meta/aves-base-bio.pt"
    EMB_DIM = 768
    OUT_EMB = DATA_ROOT / "perch-meta/full_emb_aves_base.npy"
    TORCHAUDIO_BUNDLE = torchaudio.models.wav2vec2_base

SAMPLE_RATE = 16000
WINDOW_SEC = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
BATCH = 8


def load_window_16k(path: Path, start_sec: float) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        str(path), sr=SAMPLE_RATE, offset=start_sec, duration=WINDOW_SEC
    )
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    return y[:WINDOW_SAMPLES].astype(np.float32)


def get_aves_embeddings(
    model, audio_batch: list[np.ndarray], device: str
) -> np.ndarray:
    """Extract mean-pooled last hidden state from AVES for a batch of waveforms."""
    # wav2vec2 expects (batch, time) float32
    max_len = max(len(a) for a in audio_batch)
    padded = np.zeros((len(audio_batch), max_len), dtype=np.float32)
    for i, a in enumerate(audio_batch):
        padded[i, : len(a)] = a
    x = torch.from_numpy(padded).to(device)
    with torch.no_grad():
        features, _ = model.extract_features(x)
        # features[-1]: (batch, frames, dim) — mean-pool over frames
        emb = features[-1].mean(dim=1).cpu().numpy()
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


def padded_cmap(y_true, y_pred, pad=5):
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        ct = np.concatenate([y_true[:, c], np.zeros(pad)])
        cp = np.concatenate([y_pred[:, c], np.zeros(pad)])
        aps.append(average_precision_score(ct, cp))
    return float(np.mean(aps)) if aps else 0.0


def oof_probe(emb, Y_true, meta, tag):
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)
    meta2 = meta.copy().reset_index(drop=True)
    meta2["stem"] = meta2["row_id"].str.rsplit("_", n=1).str[0]
    unique_files = meta2["stem"].unique()
    oof = np.zeros_like(Y_true)
    active = np.where(Y_true.sum(axis=0) >= 2)[0]
    print(f"  {tag}: {len(unique_files)} files, {len(active)} active", flush=True)
    for stem in unique_files:
        vm = meta2["stem"].values == stem
        tm = ~vm
        for c in active:
            if Y_true[tm, c].sum() < 1:
                continue
            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
            clf.fit(emb_s[tm], Y_true[tm, c])
            oof[vm, c] = clf.predict_proba(emb_s[vm])[:, 1]
    cm = padded_cmap(Y_true, oof)
    print(f"  {tag} OOF cmAP: {cm:.4f}", flush=True)
    return cm, oof


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: AVES-{AVES_MODEL} ({EMB_DIM}-dim) from {CKPT_PATH.name}")

    # Load meta (original order — NPZ is in this order)
    meta = pd.read_parquet(PERCH_META_PARQUET)
    npz = np.load(PERCH_NPZ)
    perch_emb = npz["emb_full"]  # (792, 1536) in original meta order
    perch_logits = npz["scores_full_raw"]
    print(f"Total windows: {len(meta)}")

    # Load AVES model
    print("\nLoading AVES model...", flush=True)
    model = TORCHAUDIO_BUNDLE()
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"AVES loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # Extract embeddings in original meta order
    all_emb = np.zeros((len(meta), EMB_DIM), dtype=np.float32)
    meta["stem"] = meta["row_id"].str.rsplit("_", n=1).str[0]
    meta["end_sec_num"] = meta["row_id"].str.rsplit("_", n=1).str[1].astype(int)

    for fi, (stem, group) in enumerate(meta.groupby("stem", sort=False)):
        if fi % 10 == 0:
            print(f"  File {fi}/{meta['stem'].nunique()}: {stem}", flush=True)

        ogg_path = SOUNDSCAPE_DIR / f"{stem}.ogg"
        if not ogg_path.exists():
            print(f"  WARNING: {ogg_path} not found")
            continue

        row_indices = group.index.tolist()
        end_secs = group["end_sec_num"].tolist()
        audios = [load_window_16k(ogg_path, float(e - WINDOW_SEC)) for e in end_secs]

        for i in range(0, len(audios), BATCH):
            batch = audios[i : i + BATCH]
            batch_idx = row_indices[i : i + BATCH]
            emb_batch = get_aves_embeddings(model, batch, device)
            all_emb[batch_idx] = emb_batch

    np.save(OUT_EMB, all_emb)
    print(f"\nSaved: {all_emb.shape} → {OUT_EMB}")
    print(
        f"Range: [{all_emb.min():.4f}, {all_emb.max():.4f}]  "
        f"norm mean: {np.linalg.norm(all_emb, axis=1).mean():.2f}"
    )

    # ---- Probe benchmark ----
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    labels_csv_df = pd.read_csv(LABELS_CSV)
    Y_true, _ = build_ground_truth(meta, labels_csv_df, taxonomy)

    print("\n--- OOF probe benchmark ---")
    perch_is = padded_cmap(Y_true, 1 / (1 + np.exp(-perch_logits)))
    print(f"Perch logits in-sample cmAP: {perch_is:.4f}")

    aves_cm, aves_oof = oof_probe(all_emb, Y_true, meta, f"AVES-{AVES_MODEL}")
    perch_cm, _ = oof_probe(perch_emb, Y_true, meta, "Perch emb")

    # Concat probe
    concat_emb = np.concatenate([perch_emb, all_emb], axis=1)
    concat_cm, concat_oof = oof_probe(
        concat_emb, Y_true, meta, f"Concat(Perch+AVES-{AVES_MODEL})"
    )

    # Also save concat OOF predictions for Stage2 training feasibility check
    np.save(OUT_EMB.parent / f"oof_concat_aves_{AVES_MODEL}.npy", concat_oof)

    print(f"\n=== FINAL RESULTS (AVES-{AVES_MODEL}) ===")
    print(f"  Perch in-sample  : {perch_is:.4f}")
    print(f"  Perch emb OOF    : {perch_cm:.4f}")
    print(
        f"  AVES emb OOF     : {aves_cm:.4f}  (delta vs Perch: {aves_cm - perch_cm:+.4f})"
    )
    print(
        f"  Concat OOF       : {concat_cm:.4f}  (delta vs Perch: {concat_cm - perch_cm:+.4f})"
    )

    if concat_cm > perch_cm + 0.01:
        print(f"\n✓ AVES-{AVES_MODEL} adds value (concat +{concat_cm - perch_cm:.4f})")
        print("  → Upload to Kaggle dataset, retrain ProtoSSM with concat input")
    else:
        print(
            f"\n✗ AVES-{AVES_MODEL} dead end (concat {concat_cm - perch_cm:+.4f} vs threshold +0.01)"
        )


if __name__ == "__main__":
    main()
