"""BirdCLEF+ 2026 — Data analysis script.

Prints a comprehensive overview of the training data:
- Label distribution (primary + secondary)
- Class breakdown (birds vs amphibians vs insects vs mammals)
- Zero-shot species
- Audio duration distribution
- Geographic distribution
- Rating distribution

Usage:
    python analyze_data.py
    python analyze_data.py --audio-sample 200  # sample N files for duration stats
"""

import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-sample",
        type=int,
        default=100,
        help="Number of audio files to sample for duration stats (0 = skip)",
    )
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sub_cols = pd.read_csv(DATA / "sample_submission.csv", nrows=0).columns[1:].tolist()

    train["primary_label"] = train["primary_label"].astype(str)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

    # -----------------------------------------------------------------------
    section("OVERVIEW")
    # -----------------------------------------------------------------------
    print(f"Training recordings : {len(train):,}")
    print(f"Taxonomy species    : {len(taxonomy)}")
    print(f"Submission targets  : {len(sub_cols)}")
    train_labels = set(train["primary_label"])
    tax_labels = set(taxonomy["primary_label"])
    zero_shot = tax_labels - train_labels
    print(f"Zero-shot species   : {len(zero_shot)} (in taxonomy, no training data)")

    # -----------------------------------------------------------------------
    section("CLASS BREAKDOWN")
    # -----------------------------------------------------------------------
    print("\nTraining recordings by class:")
    print(train["class_name"].value_counts().to_string())
    print("\nTaxonomy species by class:")
    print(taxonomy["class_name"].value_counts().to_string())

    # -----------------------------------------------------------------------
    section("LABEL DISTRIBUTION (primary_label)")
    # -----------------------------------------------------------------------
    vc = train["primary_label"].value_counts()
    print(f"Unique species with training data : {len(vc)}")
    print(
        f"Recordings/species — min: {vc.min()}, median: {int(vc.median())}, "
        f"mean: {vc.mean():.0f}, max: {vc.max()}"
    )
    print(f"\nSpecies with only 1 recording  : {(vc == 1).sum()}")
    print(f"Species with < 5 recordings    : {(vc < 5).sum()}")
    print(f"Species with < 10 recordings   : {(vc < 10).sum()}")
    print(f"Species with >= 100 recordings : {(vc >= 100).sum()}")
    print(f"Species with >= 499 recordings : {(vc >= 499).sum()} (capped)")

    print("\nTop 20 species by recording count:")
    top20 = vc.head(20).reset_index()
    top20.columns = ["primary_label", "count"]
    top20 = top20.merge(
        taxonomy[["primary_label", "common_name", "class_name"]],
        on="primary_label",
        how="left",
    )
    print(top20.to_string(index=False))

    print("\nBottom 20 species (fewest recordings):")
    bot20 = vc.tail(20).reset_index()
    bot20.columns = ["primary_label", "count"]
    bot20 = bot20.merge(
        taxonomy[["primary_label", "common_name", "class_name"]],
        on="primary_label",
        how="left",
    )
    print(bot20.to_string(index=False))

    # -----------------------------------------------------------------------
    section("SECONDARY LABELS")
    # -----------------------------------------------------------------------
    has_secondary = ~train["secondary_labels"].astype(str).str.strip().isin(
        ["[]", "nan", ""]
    )
    print(
        f"Recordings with secondary labels : {has_secondary.sum()} ({has_secondary.mean():.1%})"
    )

    all_secondary: list[str] = []
    for raw in train.loc[has_secondary, "secondary_labels"]:
        try:
            labels = ast.literal_eval(str(raw))
            all_secondary.extend(str(l) for l in labels)
        except (ValueError, SyntaxError):
            pass

    sec_vc = pd.Series(all_secondary).value_counts()
    print(f"Total secondary label occurrences   : {len(all_secondary)}")
    print(f"Unique species as secondary         : {len(sec_vc)}")
    print("\nTop 15 most common secondary labels:")
    top_sec = sec_vc.head(15).reset_index()
    top_sec.columns = ["primary_label", "count"]
    top_sec = top_sec.merge(
        taxonomy[["primary_label", "common_name"]], on="primary_label", how="left"
    )
    print(top_sec.to_string(index=False))

    # -----------------------------------------------------------------------
    section("ZERO-SHOT SPECIES")
    # -----------------------------------------------------------------------
    zs_df = taxonomy[taxonomy["primary_label"].isin(zero_shot)].copy()
    print(f"\n{len(zs_df)} species with no training data:")
    print(
        zs_df[
            ["primary_label", "scientific_name", "common_name", "class_name"]
        ].to_string(index=False)
    )

    # -----------------------------------------------------------------------
    section("GEOGRAPHIC DISTRIBUTION")
    # -----------------------------------------------------------------------
    print(f"Recordings with lat/lon : {train['latitude'].notna().sum():,}")
    print(
        f"Latitude  — min: {train['latitude'].min():.1f}, max: {train['latitude'].max():.1f}"
    )
    print(
        f"Longitude — min: {train['longitude'].min():.1f}, max: {train['longitude'].max():.1f}"
    )
    print("\nData source breakdown (collection):")
    if "collection" in train.columns:
        print(train["collection"].value_counts().to_string())

    # -----------------------------------------------------------------------
    section("QUALITY RATING")
    # -----------------------------------------------------------------------
    if "rating" in train.columns:
        print(f"Mean rating : {train['rating'].mean():.2f}")
        print(f"Recordings with rating=0 (unrated) : {(train['rating'] == 0).sum()}")
        bins = [0, 1, 2, 3, 4, 5]
        labels_r = ["0", "1", "2", "3", "4-5"]
        binned = pd.cut(train["rating"], bins=[-0.1, 0, 1, 2, 3, 5], labels=labels_r)
        print("\nRating distribution:")
        print(binned.value_counts().sort_index().to_string())

    # -----------------------------------------------------------------------
    section("AUDIO DURATION STATS")
    # -----------------------------------------------------------------------
    if args.audio_sample > 0:
        audio_dir = DATA / "train_audio"
        all_files = list(audio_dir.rglob("*.ogg"))
        sample_n = min(args.audio_sample, len(all_files))
        rng = np.random.default_rng(42)
        sample_files = rng.choice(all_files, size=sample_n, replace=False)

        durations = []
        samplerates = set()
        errors = 0
        for f in sample_files:
            try:
                info = sf.info(f)
                durations.append(info.duration)
                samplerates.add(info.samplerate)
            except Exception:
                errors += 1

        print(f"Sampled {sample_n} files ({errors} errors)")
        print(f"Sample rates : {sorted(samplerates)}")
        d = np.array(durations)
        print(
            f"Duration — min: {d.min():.1f}s, p25: {np.percentile(d, 25):.1f}s, "
            f"median: {np.median(d):.1f}s, p75: {np.percentile(d, 75):.1f}s, "
            f"max: {d.max():.1f}s"
        )
        print(f"Clips < 5s   : {(d < 5).sum()} ({(d < 5).mean():.1%})")
        print(f"Clips >= 60s : {(d >= 60).sum()} ({(d >= 60).mean():.1%})")
    else:
        print("Skipped (--audio-sample 0)")

    # -----------------------------------------------------------------------
    section("TEST SOUNDSCAPES")
    # -----------------------------------------------------------------------
    test_dir = DATA / "test_soundscapes"
    test_files = [
        f for f in test_dir.iterdir() if f.suffix in (".ogg", ".wav", ".flac")
    ]
    print(f"Public test soundscapes : {len(test_files)}")
    if test_files:
        info = sf.info(test_files[0])
        print(f"Sample: {info.samplerate}Hz, {info.duration:.1f}s, {info.channels}ch")
        sub = pd.read_csv(DATA / "sample_submission.csv", nrows=0)
        n_rows = pd.read_csv(DATA / "sample_submission.csv").shape[0]
        print(f"Submission rows (public test) : {n_rows}")
        print(
            f"  → {n_rows / len(test_files):.0f} chunks/soundscape avg "
            f"(at 5s/chunk = ~{n_rows * 5 / len(test_files) / 60:.0f} min/soundscape)"
        )


if __name__ == "__main__":
    main()
