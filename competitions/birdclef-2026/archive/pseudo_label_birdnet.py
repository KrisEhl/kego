"""
BirdNET pseudo-labeler for BirdCLEF 2026 train soundscapes.

Runs BirdNET v2.4 (6,522 species) on all train soundscapes to produce
pseudo-labels for bird species. Complements self-training pseudo-labels
(pseudo_label_self.py) which cover all 234 species including non-birds.

BirdNET is purpose-built for field recordings — significantly better than
our 0.783 LB model at detecting birds in passive soundscapes.

Coverage: BirdNET covers ~160/234 competition bird species (by scientific
name match). Non-bird species (Amphibia, Insecta, Mammalia, Reptilia) are
not covered — use self-training pseudo-labels for those.

Usage (on GPU server — BirdNET runs on CPU only):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        uv run python competitions/birdclef-2026/pseudo_label_birdnet.py

    # Parallel (split by index):
    ... --start-idx 0    --end-idx 5329 &
    ... --start-idx 5329 --end-idx 10658 &

Output:
    data/birdclef/birdclef-2026/birdnet_pseudo_labels[_start_end].csv
        Columns: soundscape_filename, start_sec, end_sec, primary_label, max_conf, n_species
        One row per 5s window with ≥1 confident BirdNET detection.
        primary_label: semicolon-separated competition species codes.
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_DIR = DATA

# Pantanal centre — BirdNET uses lat/lon for species filtering
LAT, LON = -17.0, -57.0

MIN_CONF = 0.1  # BirdNET confidence threshold (0–1). Keep low; filter later.


# ── build scientific-name → competition label mapping ─────────────────────────
def _load_taxonomy() -> dict[str, str]:
    """Map scientific_name (lowercase) → primary_label for competition species."""
    mapping: dict[str, str] = {}
    with open(TAXONOMY_CSV) as f:
        for row in csv.DictReader(f):
            sci = row["scientific_name"].strip().lower()
            mapping[sci] = row["primary_label"]
    return mapping


def _build_birdnet_mapping(
    analyzer_labels: list[str], taxonomy: dict[str, str]
) -> dict[str, str]:
    """Map BirdNET label strings to competition primary_labels.

    BirdNET labels: "ScientificName_CommonName"
    Returns dict: birdnet_label → primary_label (only for mapped species).
    """
    mapping: dict[str, str] = {}
    for label in analyzer_labels:
        sci = label.split("_")[0].strip().lower()
        if sci in taxonomy:
            mapping[label] = taxonomy[sci]
    return mapping


# ── per-file worker ────────────────────────────────────────────────────────────
def process_file(args_tuple) -> list[dict]:
    """Process a single soundscape file. Returns list of pseudo-label rows."""
    sc_path, birdnet_to_comp, min_conf = args_tuple

    # Import here so each worker process initialises BirdNET independently
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()

    try:
        rec = Recording(
            analyzer,
            str(sc_path),
            lat=LAT,
            lon=LON,
            date=None,  # no date filter
            min_conf=min_conf,
        )
        rec.analyze()
        detections = rec.detections  # list of dicts
    except Exception as e:
        print(f"  WARN: {sc_path.name}: {e}", flush=True)
        return []

    # Aggregate detections into 5-second windows
    # BirdNET uses 3s sliding windows; we bucket by floor(start/5)*5
    windows: dict[int, dict[str, float]] = {}  # start_sec → {species: max_conf}
    for det in detections:
        label = det.get("label", "")
        comp_label = birdnet_to_comp.get(label)
        if comp_label is None:
            continue
        conf = float(det.get("confidence", 0.0))
        start_sec = int(det.get("start_time", 0))
        window_start = (start_sec // 5) * 5
        if window_start not in windows:
            windows[window_start] = {}
        if (
            comp_label not in windows[window_start]
            or conf > windows[window_start][comp_label]
        ):
            windows[window_start][comp_label] = conf

    rows = []
    for start_sec in sorted(windows):
        species_conf = windows[start_sec]
        species_list = sorted(species_conf, key=lambda s: -species_conf[s])
        rows.append(
            {
                "soundscape_filename": sc_path.name,
                "start_sec": start_sec,
                "end_sec": start_sec + 5,
                "primary_label": ";".join(species_list),
                "max_conf": round(max(species_conf.values()), 4),
                "n_species": len(species_list),
            }
        )
    return rows


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.1,
        help="BirdNET confidence threshold (default 0.1)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel worker processes (default 4)"
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    # Load taxonomy and build BirdNET→competition mapping
    taxonomy = _load_taxonomy()

    # Load BirdNET labels once (in main process)
    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()
    birdnet_to_comp = _build_birdnet_mapping(analyzer.labels, taxonomy)
    n_mapped = len(birdnet_to_comp)
    n_comp_covered = len(set(birdnet_to_comp.values()))
    print(
        f"BirdNET→competition mapping: {n_mapped} BirdNET labels → {n_comp_covered} competition species"
    )
    del analyzer  # free memory before forking

    # Soundscape files
    sc_files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
    end_idx = args.end_idx or len(sc_files)
    sc_files = sc_files[args.start_idx : end_idx]
    print(f"Soundscapes to process: {len(sc_files)}")

    suffix = f"_{args.start_idx}_{end_idx}" if args.start_idx or args.end_idx else ""
    out_csv = OUT_DIR / f"birdnet_pseudo_labels{suffix}.csv"

    t_start = time.time()
    n_rows = 0

    worker_args = [(p, birdnet_to_comp, args.min_conf) for p in sc_files]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "soundscape_filename",
                "start_sec",
                "end_sec",
                "primary_label",
                "max_conf",
                "n_species",
            ],
        )
        writer.writeheader()

        with mp.Pool(processes=args.workers) as pool:
            for i, rows in enumerate(pool.imap(process_file, worker_args, chunksize=4)):
                writer.writerows(rows)
                n_rows += len(rows)

                if (i + 1) % 200 == 0 or i == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(sc_files) - i - 1) / rate
                    print(
                        f"  [{i + 1}/{len(sc_files)}]  {rate:.1f} files/s  "
                        f"pseudo-rows: {n_rows}  ETA {eta / 60:.1f} min",
                        flush=True,
                    )

    total_windows = len(sc_files) * 12
    print(
        f"\nPseudo-labeled rows: {n_rows} / {total_windows} windows "
        f"({100 * n_rows / total_windows:.1f}%)"
    )
    print(f"Saved → {out_csv}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
