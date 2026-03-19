"""BirdCLEF+ 2026 — Data analysis script.

Prints a comprehensive overview of the training data and saves plots to
analyze_plots.png:
- Label distribution (primary + secondary)
- Class breakdown (birds vs amphibians vs insects vs mammals)
- Zero-shot species
- Audio duration distribution
- Geographic distribution
- Rating distribution

Usage:
    python analyze_data.py
    python analyze_data.py --audio-sample 200  # sample N files for duration stats
    python analyze_data.py --no-plots          # text only
"""

import argparse
import ast
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

DATA = (
    Path(os.getenv("KEGO_PATH_DATA", Path(__file__).parent.parent.parent / "data"))
    / "birdclef"
    / "birdclef-2026"
)


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
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "plots" / "analyze_plots.png"),
        help="Output path for plots",
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
    durations: np.ndarray | None = None
    if args.audio_sample > 0:
        audio_dir = DATA / "train_audio"
        all_files = list(audio_dir.rglob("*.ogg"))
        sample_n = min(args.audio_sample, len(all_files))
        rng = np.random.default_rng(42)
        sample_files = rng.choice(all_files, size=sample_n, replace=False)

        dur_list = []
        samplerates = set()
        errors = 0
        for f in sample_files:
            try:
                info = sf.info(f)
                dur_list.append(info.duration)
                samplerates.add(info.samplerate)
            except Exception:
                errors += 1

        durations = np.array(dur_list)
        print(f"Sampled {sample_n} files ({errors} errors)")
        print(f"Sample rates : {sorted(samplerates)}")
        print(
            f"Duration — min: {durations.min():.1f}s, p25: {np.percentile(durations, 25):.1f}s, "
            f"median: {np.median(durations):.1f}s, p75: {np.percentile(durations, 75):.1f}s, "
            f"max: {durations.max():.1f}s"
        )
        print(f"Clips < 5s   : {(durations < 5).sum()} ({(durations < 5).mean():.1%})")
        print(
            f"Clips >= 60s : {(durations >= 60).sum()} ({(durations >= 60).mean():.1%})"
        )
    else:
        print("Skipped (--audio-sample 0)")

    # -----------------------------------------------------------------------
    section("RECORDING TYPES (XC metadata)")
    # -----------------------------------------------------------------------
    if "type" in train.columns:
        xc = train[train["collection"] == "XC"].copy()
        type_counts: dict[str, int] = {}
        for raw in xc["type"]:
            try:
                types = ast.literal_eval(str(raw))
                for t in types:
                    t = t.strip()
                    if t:
                        type_counts[t] = type_counts.get(t, 0) + 1
            except (ValueError, SyntaxError):
                pass
        type_series = pd.Series(type_counts).sort_values(ascending=False)
        print(f"XC recordings: {len(xc):,} (iNat has no type metadata)")
        print("Top recording types:")
        print(type_series.head(15).to_string())
    else:
        type_series = pd.Series(dtype=int)

    # -----------------------------------------------------------------------
    section("TRAIN SOUNDSCAPES")
    # -----------------------------------------------------------------------
    sc_path = DATA / "train_soundscapes_labels.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        print(f"Labeled soundscape rows : {len(sc):,}")
        print(f"Unique soundscapes      : {sc['filename'].nunique()}")
        n_chunks_per_sc = sc.groupby("filename").size()
        print(
            f"Chunks per soundscape   : min={n_chunks_per_sc.min()}, max={n_chunks_per_sc.max()}, median={n_chunks_per_sc.median():.0f}"
        )
        print(
            f"Typical duration        : {n_chunks_per_sc.median() * 5:.0f}s = {n_chunks_per_sc.median() * 5 / 60:.1f} min"
        )
        species_per_chunk = sc["primary_label"].str.split(";").str.len()
        print(
            f"Species per 5s chunk    : min={species_per_chunk.min()}, max={species_per_chunk.max()}, median={species_per_chunk.median():.1f}, mean={species_per_chunk.mean():.1f}"
        )
        print(
            "\nNote: soundscape labels have multiple co-occurring species per chunk (';'-separated)"
        )
        print(
            "      This reflects real test conditions — very different from isolated training clips."
        )
    else:
        sc = None
        species_per_chunk = None
        print("train_soundscapes_labels.csv not found")

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
        n_rows = pd.read_csv(DATA / "sample_submission.csv").shape[0]
        print(f"Submission rows (public test) : {n_rows}")
        print(
            f"  → {n_rows / len(test_files):.0f} chunks/soundscape avg "
            f"(at 5s/chunk = ~{n_rows * 5 / len(test_files) / 60:.0f} min/soundscape)"
        )

    # -----------------------------------------------------------------------
    section("PLOTS")
    # -----------------------------------------------------------------------
    if args.no_plots:
        print("Skipped (--no-plots)")
        return

    fig, axes = plt.subplots(5, 3, figsize=(18, 22))
    fig.suptitle(
        "BirdCLEF+ 2026 — Training Data Overview", fontsize=14, fontweight="bold"
    )

    # 1. Recordings per species (log-scale histogram)
    ax = axes[0, 0]
    ax.hist(vc.values, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Recordings per species")
    ax.set_ylabel("Number of species")
    ax.set_title("Recording count distribution")
    ax.set_yscale("log")
    ax.axvline(
        vc.median(), color="orange", linestyle="--", label=f"median={int(vc.median())}"
    )
    ax.legend(fontsize=8)

    # 2. Top 25 species by recording count
    ax = axes[0, 1]
    top25 = vc.head(25)
    ax.barh(range(len(top25)), top25.values[::-1], color="steelblue")
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(top25.index[::-1], fontsize=6)
    ax.set_xlabel("Recordings")
    ax.set_title("Top 25 species")

    # 3. Class breakdown — recordings per class (log scale, counts in tick labels)
    ax = axes[0, 2]
    class_recs = train["class_name"].value_counts()
    class_spp = taxonomy["class_name"].value_counts()
    classes = class_recs.index.tolist()
    x = np.arange(len(classes))
    ax.bar(x, [class_recs.get(c, 0) for c in classes], color="#4e79a7")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{c}\n{class_recs.get(c, 0):,} rec\n{class_spp.get(c, 0)} spp"
            for c in classes
        ],
        fontsize=7,
    )
    ax.set_ylabel("Recordings (log scale)")
    ax.set_title("Recordings by class (birds dominate)")

    # 13. Species count per class (new)
    ax = axes[4, 0]
    classes_tax = class_spp.index.tolist()
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    bars = ax.bar(
        classes_tax,
        [class_spp[c] for c in classes_tax],
        color=colors[: len(classes_tax)],
    )
    ax.set_ylabel("Number of species")
    ax.set_title("Species count by class (taxonomy)")
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 14. 2D histogram: recording count distribution per class
    ax = axes[4, 1]
    spp_df = (
        vc.reset_index()
        .rename(columns={"primary_label": "primary_label", "count": "n_recordings"})
        .merge(
            taxonomy[["primary_label", "class_name"]], on="primary_label", how="left"
        )
    )
    log_bins = np.logspace(0, np.log10(vc.max() + 1), 20)
    class_order = class_spp.index.tolist()  # sorted by species count desc
    heatmap = np.zeros((len(class_order), len(log_bins) - 1), dtype=int)
    for i, cls in enumerate(class_order):
        vals = spp_df.loc[spp_df["class_name"] == cls, "n_recordings"].values
        heatmap[i], _ = np.histogram(vals, bins=log_bins)
    im = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="YlOrRd",
        extent=[0, len(log_bins) - 1, len(class_order) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(class_order, fontsize=8)
    bin_centers = (log_bins[:-1] + log_bins[1:]) / 2
    tick_pos = np.linspace(0, len(log_bins) - 2, 6)
    tick_vals = [int(bin_centers[min(int(p), len(bin_centers) - 1)]) for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_vals, fontsize=7)
    ax.set_xlabel("Recordings per species")
    ax.set_title("Species distribution: count vs recordings (2D histogram)")
    plt.colorbar(im, ax=ax, label="# species", shrink=0.8)

    axes[4, 2].set_visible(False)

    # 4. Rating distribution
    ax = axes[1, 0]
    if "rating" in train.columns:
        rating_vc = train["rating"].value_counts().sort_index()
        ax.bar(rating_vc.index.astype(str), rating_vc.values, color="steelblue")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Quality rating distribution")
    else:
        ax.set_visible(False)

    # 5. Audio duration histogram (sampled)
    ax = axes[1, 1]
    if durations is not None:
        ax.hist(durations, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(
            np.median(durations),
            color="orange",
            linestyle="--",
            label=f"median={np.median(durations):.0f}s",
        )
        ax.axvline(5, color="red", linestyle=":", label="5s clip window")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"Audio duration (n={len(durations)} sampled)")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No duration data\n(use --audio-sample N)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_title("Audio duration")

    # 6. Geographic scatter (lat/lon)
    ax = axes[1, 2]
    geo = train.dropna(subset=["latitude", "longitude"])
    class_list = geo["class_name"].unique()
    cmap = {
        "Aves": "#4e79a7",
        "Amphibia": "#f28e2b",
        "Insecta": "#e15759",
        "Mammalia": "#76b7b2",
    }
    for cls in class_list:
        sub_geo = geo[geo["class_name"] == cls]
        ax.scatter(
            sub_geo["longitude"],
            sub_geo["latitude"],
            s=1,
            alpha=0.3,
            label=cls,
            color=cmap.get(cls, "gray"),
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Geographic distribution (n={len(geo):,})")
    ax.legend(fontsize=7, markerscale=5)

    # 7. Secondary label frequency (top 20)
    ax = axes[2, 0]
    if len(sec_vc) > 0:
        top_sec_plot = sec_vc.head(20)
        ax.barh(range(len(top_sec_plot)), top_sec_plot.values[::-1], color="coral")
        ax.set_yticks(range(len(top_sec_plot)))
        ax.set_yticklabels(top_sec_plot.index[::-1], fontsize=6)
        ax.set_xlabel("Occurrences as secondary label")
        ax.set_title("Top 20 secondary labels")
    else:
        ax.set_visible(False)

    # 8. Recordings per species cumulative (how much of data top-N species cover)
    ax = axes[2, 1]
    sorted_counts = np.sort(vc.values)[::-1]
    cumsum = np.cumsum(sorted_counts)
    ax.plot(range(1, len(cumsum) + 1), cumsum / cumsum[-1] * 100, color="steelblue")
    ax.axhline(80, color="orange", linestyle="--", label="80%")
    ax.axhline(50, color="red", linestyle=":", label="50%")
    idx_80 = np.searchsorted(cumsum / cumsum[-1], 0.80)
    ax.axvline(idx_80, color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Top-N species (sorted by count)")
    ax.set_ylabel("Cumulative % of recordings")
    ax.set_title("Recording concentration by species")
    ax.legend(fontsize=8)

    # 9. Zero-shot species by class
    ax = axes[2, 2]
    zs_class = zs_df["class_name"].value_counts()
    ax.bar(zs_class.index, zs_class.values, color="salmon")
    ax.set_ylabel("Count")
    ax.set_title(f"Zero-shot species by class (n={len(zs_df)})")
    for i, (cls, cnt) in enumerate(zs_class.items()):
        ax.text(i, cnt + 0.2, str(cnt), ha="center", fontsize=9)

    # 10. Recording type breakdown (XC only)
    ax = axes[3, 0]
    if len(type_series) > 0:
        top_types = type_series.head(12)
        ax.barh(range(len(top_types)), top_types.values[::-1], color="steelblue")
        ax.set_yticks(range(len(top_types)))
        ax.set_yticklabels(top_types.index[::-1], fontsize=7)
        ax.set_xlabel("Recordings (XC only)")
        ax.set_title("Recording types (XC metadata)")
    else:
        ax.set_visible(False)

    # 11. Species co-occurrence per chunk in soundscapes
    ax = axes[3, 1]
    if species_per_chunk is not None:
        ax.hist(
            species_per_chunk,
            bins=range(1, species_per_chunk.max() + 2),
            color="coral",
            edgecolor="white",
            linewidth=0.5,
            align="left",
        )
        ax.axvline(
            species_per_chunk.mean(),
            color="navy",
            linestyle="--",
            label=f"mean={species_per_chunk.mean():.1f}",
        )
        ax.set_xlabel("Species per 5s chunk")
        ax.set_ylabel("Chunk count")
        ax.set_title("Co-occurrence in train soundscapes")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    # 12. Chunks per soundscape distribution
    ax = axes[3, 2]
    if sc is not None:
        ax.hist(
            n_chunks_per_sc.values,
            bins=20,
            color="mediumseagreen",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Chunks per soundscape (5s each)")
        ax.set_ylabel("Soundscapes")
        ax.set_title(f"Soundscape length (n={len(n_chunks_per_sc)} soundscapes)")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
