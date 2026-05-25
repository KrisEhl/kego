"""Explore Rogii wellbore geology prediction data and visualize 5 sites."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = (
    Path(
        os.environ.get(
            "KEGO_PATH_DATA",
            f"{os.environ['HOME']}/projects/kego/data",
        )
    )
    / "rogii"
    / "rogii-wellbore-geology-prediction"
)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

GEOLOGY_LAYERS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
LAYER_COLORS = {
    "ANCC": "tab:red",
    "ASTNU": "tab:orange",
    "ASTNL": "tab:olive",
    "EGFDU": "tab:green",
    "EGFDL": "tab:cyan",
    "BUDA": "tab:blue",
}
# Stratigraphic zone between surface[i] (top) and surface[i+1] (bottom).
# Color zone by lower-bounding surface, matching typewell Geology label.
ZONE_FILLS = [
    ("ANCC", "ASTNU"),
    ("ASTNU", "ASTNL"),
    ("ASTNL", "EGFDU"),
    ("EGFDU", "EGFDL"),
    ("EGFDL", "BUDA"),
]


def list_well_ids(directory: Path) -> list[str]:
    ids = set()
    for path in directory.iterdir():
        match = re.match(r"^([0-9a-f]+)__horizontal_well\.csv$", path.name)
        if match:
            ids.add(match.group(1))
    return sorted(ids)


def load_well(well_id: str, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(directory / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(directory / f"{well_id}__typewell.csv")
    return horizontal, typewell


def summarize(well_ids: list[str], directory: Path, label: str) -> pd.DataFrame:
    rows = []
    for well_id in well_ids:
        horizontal, typewell = load_well(well_id, directory)
        target_col = "TVT" if "TVT" in horizontal.columns else None
        rows.append(
            {
                "set": label,
                "well_id": well_id,
                "h_rows": len(horizontal),
                "h_cols": ",".join(horizontal.columns),
                "t_rows": len(typewell),
                "md_min": horizontal["MD"].min(),
                "md_max": horizontal["MD"].max(),
                "tvt_input_nonnull": horizontal["TVT_input"].notna().sum(),
                "target_nonnull": (horizontal[target_col].notna().sum() if target_col else 0),
                "tvt_input_pct": 100 * horizontal["TVT_input"].notna().mean(),
            }
        )
    return pd.DataFrame(rows)


def wellbore_zone(horizontal: pd.DataFrame) -> pd.Series:
    """Label each MD with the stratigraphic zone containing well Z."""
    if not all(layer in horizontal.columns for layer in GEOLOGY_LAYERS):
        return pd.Series([None] * len(horizontal), index=horizontal.index)
    z = horizontal["Z"]
    labels = pd.Series(["above_ANCC"] * len(horizontal), index=horizontal.index)
    # Surfaces ordered top -> bottom (shallow -> deep). Z is negative; deeper = more negative.
    # Surface depth values are negative; deeper layer has smaller (more negative) value.
    # Determine ordering empirically from medians.
    surf_order = sorted(
        GEOLOGY_LAYERS,
        key=lambda l: horizontal[l].median(),
        reverse=True,  # shallowest (largest Z) first
    )
    for i, top_layer in enumerate(surf_order):
        top_z = horizontal[top_layer]
        if i + 1 < len(surf_order):
            bot_z = horizontal[surf_order[i + 1]]
            mask = (z <= top_z) & (z > bot_z)
        else:
            mask = z <= top_z
        labels.loc[mask] = top_layer
    return labels


def plot_well(well_id: str, horizontal: pd.DataFrame, typewell: pd.DataFrame, ax_row):
    ax_path, ax_gr, ax_tvt = ax_row

    has_surfaces = all(layer in horizontal.columns for layer in GEOLOGY_LAYERS)

    # 1) Well path (Z vs MD) + filled stratigraphic zones + surfaces
    md = horizontal["MD"].to_numpy()
    if has_surfaces:
        # Determine top->bottom ordering by median depth (shallowest first = largest Z value)
        surf_order = sorted(
            GEOLOGY_LAYERS,
            key=lambda l: horizontal[l].median(),
            reverse=True,
        )
        for i in range(len(surf_order) - 1):
            top = horizontal[surf_order[i]].to_numpy()
            bot = horizontal[surf_order[i + 1]].to_numpy()
            ax_path.fill_between(
                md,
                top,
                bot,
                color=LAYER_COLORS[surf_order[i]],
                alpha=0.18,
                linewidth=0,
                label=f"zone {surf_order[i]}",
            )
        for layer in GEOLOGY_LAYERS:
            ax_path.plot(
                md,
                horizontal[layer],
                color=LAYER_COLORS[layer],
                lw=0.9,
            )
    ax_path.plot(md, horizontal["Z"], color="black", lw=1.4, label="Well Z")
    ax_path.set_title(f"{well_id}: well path + stratigraphy")
    ax_path.set_xlabel("MD (ft)")
    ax_path.set_ylabel("Depth Z (ft)")
    ax_path.legend(fontsize=5, ncol=3, loc="lower left")
    ax_path.grid(alpha=0.3)

    # 2) GR(MD) with wellbore-zone background strip
    zones = wellbore_zone(horizontal) if has_surfaces else None
    if zones is not None:
        # Find contiguous runs of same zone
        change = zones.ne(zones.shift()).cumsum()
        for _, group in zones.groupby(change):
            label = group.iloc[0]
            color = LAYER_COLORS.get(label, "lightgray")
            md_segment = md[group.index]
            ax_gr.axvspan(
                md_segment.min(),
                md_segment.max(),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
    ax_gr.plot(md, horizontal["GR"], color="black", lw=0.7)
    ax_gr.set_title(f"{well_id}: horizontal GR (bg=zone)")
    ax_gr.set_xlabel("MD (ft)")
    ax_gr.set_ylabel("GR")
    ax_gr.grid(alpha=0.3)

    # 3) Typewell GR vs TVT with horizontal Geology bands (topology) + scatter
    if "Geology" in typewell.columns:
        geo = typewell.dropna(subset=["Geology"]).sort_values("TVT")
        # Find contiguous runs of same Geology label in TVT order -> draw horizontal bands
        geo_change = geo["Geology"].ne(geo["Geology"].shift()).cumsum()
        for _, group in geo.groupby(geo_change):
            layer = group["Geology"].iloc[0]
            ax_tvt.axhspan(
                group["TVT"].min(),
                group["TVT"].max(),
                color=LAYER_COLORS.get(layer, "gray"),
                alpha=0.18,
                linewidth=0,
            )
            ax_tvt.text(
                0.02,
                (group["TVT"].min() + group["TVT"].max()) / 2,
                layer,
                transform=ax_tvt.get_yaxis_transform(),
                fontsize=6,
                va="center",
                color=LAYER_COLORS.get(layer, "gray"),
            )
    ax_tvt.plot(typewell["GR"], typewell["TVT"], color="black", lw=0.6, label="typewell GR")
    target_col = "TVT" if "TVT" in horizontal.columns else None
    if target_col:
        ax_tvt.scatter(
            horizontal["GR"],
            horizontal[target_col],
            s=2,
            color="red",
            alpha=0.3,
            label="hwell GR vs TVT",
        )
    ax_tvt.invert_yaxis()
    ax_tvt.set_title(f"{well_id}: typewell GR vs TVT (bands=Geology)")
    ax_tvt.set_xlabel("GR")
    ax_tvt.set_ylabel("TVT (ft)")
    ax_tvt.legend(fontsize=6, loc="best")
    ax_tvt.grid(alpha=0.3)


def main():
    train_ids = list_well_ids(TRAIN_DIR)
    test_ids = list_well_ids(TEST_DIR)
    print(f"Train wells: {len(train_ids)}")
    print(f"Test wells:  {len(test_ids)} -> {test_ids}")

    # Pick 5 representative training wells (spread by hash for variety)
    rng = np.random.default_rng(seed=42)
    chosen_train = list(rng.choice(train_ids, size=5, replace=False))
    print(f"Chosen train wells: {chosen_train}")

    # Summaries
    train_summary = summarize(chosen_train, TRAIN_DIR, "train")
    test_summary = summarize(test_ids, TEST_DIR, "test")
    summary = pd.concat([train_summary, test_summary], ignore_index=True)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "wells_summary.csv", index=False)

    # Visualize 5 train wells
    fig, axes = plt.subplots(5, 3, figsize=(18, 22))
    for ax_row, well_id in zip(axes, chosen_train):
        horizontal, typewell = load_well(well_id, TRAIN_DIR)
        plot_well(well_id, horizontal, typewell, ax_row)
    fig.suptitle("Rogii wellbore geology — 5 train sites", fontsize=14, y=1.0)
    fig.tight_layout()
    out_path = OUT_DIR / "five_train_wells.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Bonus: visualize test wells (no TVT target, but geology surfaces also absent)
    fig2, axes2 = plt.subplots(len(test_ids), 3, figsize=(18, 4.5 * len(test_ids)))
    if len(test_ids) == 1:
        axes2 = np.array([axes2])
    for ax_row, well_id in zip(axes2, test_ids):
        horizontal, typewell = load_well(well_id, TEST_DIR)
        plot_well(well_id, horizontal, typewell, ax_row)
    fig2.suptitle("Rogii — test sites (TVT missing)", fontsize=14, y=1.0)
    fig2.tight_layout()
    out_path2 = OUT_DIR / "test_wells.png"
    fig2.savefig(out_path2, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
