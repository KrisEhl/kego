"""Precompute CLAP zero-shot similarity scores for the 28 soundscape-only sonotype species.

These are the 47158son* insect sonotypes with zero training audio. Perch logit delta
is exactly 0.000 for all of them — Perch is completely blind to their presence.

CLAP (Contrastive Language-Audio Pretraining, LAION 2023) can assign text-to-audio
cosine similarities based on text descriptions of the target sound. This provides a
weak but non-zero signal where the current pipeline has nothing.

Setup (run once on cluster):
    pip install msclap
    # Model auto-downloads from HuggingFace on first run

Run:
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    uv run python competitions/birdclef-2026/training/precompute_clap_zeroshot.py

Output:
    data/perch-meta/clap_zeroshot_scores.npy  — shape (n_windows, 28), float32
    data/perch-meta/clap_zeroshot_labels.json — ordered list of 28 species labels
    data/perch-meta/clap_zeroshot_queries.json — text queries used per species

Uncertainty note:
    CLAP text queries distinguish sound characteristics, not temporal patterns.
    Sonotypes in the same family may share similar query descriptions.
    Effectiveness is uncertain — evaluate OOF cmAP vs Perch-only baseline
    before using in the submission kernel.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch


# Monkey-patch torchaudio.load to use soundfile instead of torchcodec.
# torchaudio 2.11+ requires torchcodec which needs libnppicc.so.13 (CUDA NPP).
# soundfile handles WAV files without any CUDA dependency.
def _torchaudio_load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (frames, channels); torchaudio returns (channels, frames)
    tensor = torch.from_numpy(data.T)
    return tensor, sr


try:
    import torchaudio

    torchaudio.load = _torchaudio_load_via_soundfile
except ImportError:
    pass

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
SOUNDSCAPE_DIR = DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes"
TAXONOMY_CSV = DATA_ROOT / "birdclef/birdclef-2026/taxonomy.csv"
PERCH_META_PARQUET = DATA_ROOT / "perch-meta/full_perch_meta.parquet"

OUT_SCORES = DATA_ROOT / "perch-meta/clap_zeroshot_scores.npy"
OUT_LABELS = DATA_ROOT / "perch-meta/clap_zeroshot_labels.json"
OUT_QUERIES = DATA_ROOT / "perch-meta/clap_zeroshot_queries.json"

SAMPLE_RATE_CLAP = 44100  # CLAP default
WINDOW_SEC = 5  # Competition window size
WINDOW_SAMPLES = SAMPLE_RATE_CLAP * WINDOW_SEC


# ---------------------------------------------------------------------------
# Text query library for insect sonotypes
# These are approximate descriptions — CLAP effectiveness depends on how well
# the audio matches the text. Queries are intentionally varied to maximise
# the chance of at least one matching the acoustic character.
# ---------------------------------------------------------------------------
SONOTYPE_QUERIES: dict[str, list[str]] = {
    # Dawn / early morning stridulators (3-4 AM UTC based on positive windows)
    "47158son10": [
        "insect stridulation in tropical dawn chorus",
        "cricket chirping at night in tropical forest",
        "rhythmic insect buzzing before sunrise",
    ],
    # Morning stridulators (7 AM UTC)
    "47158son15": [
        "insect calling in tropical morning soundscape",
        "cicada sound in wetland morning",
        "tropical insect stridulation at dawn",
    ],
    "47158son16": [
        "insect noise in tropical morning chorus",
        "grasshopper stridulation in wetland",
        "high-pitched insect buzzing in savanna",
    ],
    "47158son18": [
        "insect stridulation in tropical morning",
        "cicada buzzing in humid tropical habitat",
        "continuous insect sound in morning",
    ],
    # Evening stridulators (19 UTC = ~3 PM local)
    "47158son09": [
        "insect calling in tropical afternoon",
        "cricket or cicada in tropical wetland afternoon",
        "insect sound in subtropical afternoon",
    ],
    "47158son12": [
        "insect stridulation in tropical afternoon chorus",
        "grasshopper or katydid calling in afternoon",
        "loud insect buzzing in wetland afternoon",
    ],
    # Generic fallback for species without clear temporal pattern
    "default": [
        "insect stridulation in South American wetland",
        "tropical insect calling Pantanal",
        "cicada or cricket sound in tropical savanna",
    ],
}


def get_queries(label: str) -> list[str]:
    return SONOTYPE_QUERIES.get(label, SONOTYPE_QUERIES["default"])


def load_window(path: Path, start_sec: float, sr_target: int = SAMPLE_RATE_CLAP) -> np.ndarray:
    """Load a 5-second window from a .ogg file, resampled to CLAP's expected rate."""
    import librosa  # lazy import

    y, sr = librosa.load(str(path), sr=sr_target, offset=start_sec, duration=WINDOW_SEC)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    return y[:WINDOW_SAMPLES]


def main() -> None:
    from msclap import CLAP  # type: ignore

    taxonomy = pd.read_csv(TAXONOMY_CSV)
    meta = pd.read_parquet(PERCH_META_PARQUET)
    train_audio = pd.read_csv(DATA_ROOT / "birdclef/birdclef-2026/train.csv")

    audio_species = set(train_audio["primary_label"].astype(str).unique())
    all_labels = taxonomy["primary_label"].tolist()

    # Identify the 28 soundscape-only species
    sc_only = [lbl for lbl in all_labels if lbl not in audio_species and "son" in lbl.lower()]
    print(f"Soundscape-only sonotype species: {len(sc_only)}")
    for sp in sorted(sc_only):
        print(f"  {sp}")

    # Build text query list and save
    queries_per_species = {sp: get_queries(sp) for sp in sc_only}
    with open(OUT_QUERIES, "w") as f:
        json.dump(queries_per_species, f, indent=2)
    print(f"\nQueries written to {OUT_QUERIES}")

    # Load CLAP model (auto-downloads ~1GB on first run)
    print("\nLoading CLAP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = CLAP(version="2023", use_cuda=(device == "cuda"))

    # Pre-encode all text queries
    all_queries_flat = []
    query_idx: dict[str, list[int]] = {}
    for sp in sc_only:
        q_list = get_queries(sp)
        query_idx[sp] = list(range(len(all_queries_flat), len(all_queries_flat) + len(q_list)))
        all_queries_flat.extend(q_list)

    print(f"Encoding {len(all_queries_flat)} text queries...")
    text_embeddings = clap_model.get_text_embeddings(all_queries_flat)  # (n_q, d)
    if hasattr(text_embeddings, "cpu"):
        text_embeddings = text_embeddings.cpu().numpy()
    text_embeddings = np.array(text_embeddings, dtype=np.float32)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Build per-species mean text embedding
    sp_text_emb = np.zeros((len(sc_only), text_embeddings.shape[1]), dtype=np.float32)
    for i, sp in enumerate(sc_only):
        idxs = query_idx[sp]
        sp_text_emb[i] = text_embeddings[idxs].mean(axis=0)
    sp_text_emb = sp_text_emb / np.linalg.norm(sp_text_emb, axis=1, keepdims=True)
    print(f"Per-species text embeddings: {sp_text_emb.shape}")

    # Unique soundscape files in meta
    meta["stem"] = meta["row_id"].str.rsplit("_", n=1).str[0]
    unique_stems = meta["stem"].unique()
    print(f"\nProcessing {len(unique_stems)} soundscape files...")

    scores = np.zeros((len(meta), len(sc_only)), dtype=np.float32)

    for fi, stem in enumerate(unique_stems):
        if fi % 10 == 0:
            print(f"  {fi}/{len(unique_stems)}  ({stem})")

        # Reconstruct file path
        ogg_path = SOUNDSCAPE_DIR / f"{stem}.ogg"
        if not ogg_path.exists():
            print(f"    WARNING: {ogg_path} not found — skipping")
            continue

        # Get all windows for this file (sorted by end_sec)
        file_rows = (
            meta[meta["stem"] == stem].sort_values("end_sec")
            if "end_sec" in meta.columns
            else meta[meta["stem"] == stem]
        )
        row_indices = file_rows.index.tolist()
        row_ids = file_rows["row_id"].tolist()

        # Load each 5-second window and save to temp WAV files
        # (msclap.get_audio_embeddings expects file paths, not numpy arrays)
        window_audios = []
        for rid in row_ids:
            end_sec = int(rid.rsplit("_", 1)[-1])
            start_sec = end_sec - WINDOW_SEC
            audio = load_window(ogg_path, float(start_sec))
            window_audios.append(audio)

        # Write to temp WAV files at 44100 Hz (CLAP's expected rate)
        tmp_wav_paths = []
        tmp_dir = tempfile.mkdtemp()
        for j, wav in enumerate(window_audios):
            tmp_path = os.path.join(tmp_dir, f"w{j}.wav")
            sf.write(tmp_path, wav, SAMPLE_RATE_CLAP)
            tmp_wav_paths.append(tmp_path)

        # Batch encode audio via file paths
        audio_embeddings = clap_model.get_audio_embeddings(tmp_wav_paths, resample=False)  # (n_windows, d)

        # Clean up temp files
        for p in tmp_wav_paths:
            os.unlink(p)
        os.rmdir(tmp_dir)
        if hasattr(audio_embeddings, "cpu"):
            audio_embeddings = audio_embeddings.cpu().numpy()
        audio_embeddings = np.array(audio_embeddings, dtype=np.float32)
        audio_embeddings = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)

        # Cosine similarity with per-species text embedding
        sims = audio_embeddings @ sp_text_emb.T  # (n_windows, n_species)
        for j, ridx in enumerate(row_indices):
            scores[ridx] = sims[j]

    # Save
    np.save(OUT_SCORES, scores)
    with open(OUT_LABELS, "w") as f:
        json.dump(sc_only, f, indent=2)

    print("\nDone!")
    print(f"Scores shape : {scores.shape}")
    print(f"Score range  : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Saved to     : {OUT_SCORES}")

    # Quick sanity check: do positives score higher than negatives?
    from sklearn.metrics import average_precision_score

    try:
        labels_csv = pd.read_csv(DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes_labels.csv")
        all_labels_list = taxonomy["primary_label"].tolist()
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels_list)}

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

        n_rows = len(meta)
        Y_check = np.zeros((n_rows, len(sc_only)), dtype=np.float32)
        sc_to_col = {sp: i for i, sp in enumerate(sc_only)}
        for i, row_id in enumerate(meta["row_id"]):
            for sp in rid_to_species.get(row_id, set()):
                if sp in sc_to_col:
                    Y_check[i, sc_to_col[sp]] = 1.0

        active = [c for c in range(len(sc_only)) if Y_check[:, c].sum() > 0]
        aps = [average_precision_score(Y_check[:, c], scores[:, c]) for c in active]
        print(f"\nSanity check AP on {len(active)} active sonotypes:")
        print(f"  Mean AP  = {np.mean(aps):.4f}")
        print(f"  Best     = {max(aps):.4f} ({sc_only[active[np.argmax(aps)]]})")
        print(f"  Worst    = {min(aps):.4f} ({sc_only[active[np.argmin(aps)]]})")
        print("\nPer-species AP:")
        for c in active:
            print(f"  {sc_only[c]:<16} n_pos={int(Y_check[:, c].sum()):3d}  AP={aps[active.index(c)]:.4f}")
    except Exception as e:
        print(f"Sanity check failed: {e}")


if __name__ == "__main__":
    main()
