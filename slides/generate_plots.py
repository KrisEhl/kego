"""Generate all plot assets used in the slides.

Usage:
    cd slides && python generate_plots.py
    # or via Makefile:
    cd slides && make plots
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLIDES_DIR = Path(__file__).parent
DATA_DIR = Path.home() / "projects/kego/data/playground/playground-series-s6e2"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
HEALTHY_COLOR = "#4fc3f7"
SICK_COLOR = "#ef5350"
GRID_COLOR = "#333355"


# ---------------------------------------------------------------------------
# Plot 1: Patient radar chart (healthy vs heart disease)
# ---------------------------------------------------------------------------
def plot_patient_radar() -> None:
    df = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")
    df["target"] = (df["Heart Disease"] == "Presence").astype(int)

    raw_features = [
        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium",
    ]
    labels = [
        "Age",
        "Sex",
        "Chest\npain type",
        "Blood\npressure",
        "Cholesterol",
        "Fasting\nblood sugar",
        "EKG\nresults",
        "Max\nheart rate",
        "Exercise\nangina",
        "ST\ndepression",
        "Slope\nof ST",
        "Vessel\ncount",
        "Thallium",
    ]

    mins = df[raw_features].min()
    maxs = df[raw_features].max()
    df_norm = (df[raw_features] - mins) / (maxs - mins)
    df_norm["target"] = df["target"]

    healthy = df_norm[df_norm["target"] == 0][raw_features].mean().values
    sick = df_norm[df_norm["target"] == 1][raw_features].mean().values

    n = len(raw_features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    healthy_closed = np.append(healthy, healthy[0])
    sick_closed = np.append(sick, sick[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=0)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--")
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    ax.spines["polar"].set_color(GRID_COLOR)

    ax.fill(angles_closed, healthy_closed, alpha=0.20, color=HEALTHY_COLOR)
    ax.plot(angles_closed, healthy_closed, color=HEALTHY_COLOR, linewidth=2.5)
    ax.fill(angles_closed, sick_closed, alpha=0.20, color=SICK_COLOR)
    ax.plot(angles_closed, sick_closed, color=SICK_COLOR, linewidth=2.5)

    ax.scatter(angles, healthy, s=60, color=HEALTHY_COLOR, zorder=5)
    ax.scatter(angles, sick, s=60, color=SICK_COLOR, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=12, color="#ccccdd")
    ax.tick_params(axis="x", pad=18)

    legend_elements = [
        mpatches.Patch(facecolor=HEALTHY_COLOR, alpha=0.7, label="No heart disease"),
        mpatches.Patch(facecolor=SICK_COLOR, alpha=0.7, label="Heart disease"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=13,
        framealpha=0,
        labelcolor="white",
    )

    fig.tight_layout()
    out = SLIDES_DIR / "patient_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Feature scatter — 6-panel distribution by target
# ---------------------------------------------------------------------------
def plot_feature_scatter() -> None:
    df = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")
    df["target"] = (df["Heart Disease"] == "Presence").astype(int)

    thallium_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    chest_pain_map = {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    }
    df["Thallium_label"] = df["Thallium"].map(thallium_map)
    df["Chest pain label"] = df["Chest pain type"].map(chest_pain_map)

    panels = [
        ("Age", "Age", False),
        ("Max HR", "Max Heart Rate", False),
        ("ST depression", "ST Depression", False),
        ("Chest pain label", "Chest Pain Type", True),
        ("Number of vessels fluro", "Vessel Count (Fluoroscopy)", False),
        ("Thallium_label", "Thallium Stress Test", True),
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)

    healthy = df[df["target"] == 0]
    sick = df[df["target"] == 1]

    for ax, (col, title, is_cat) in zip(axes.flat, panels):
        ax.set_facecolor(BG)
        if is_cat:
            categories = df[col].dropna().unique()
            x = np.arange(len(categories))
            h_counts = healthy[col].value_counts().reindex(categories, fill_value=0)
            s_counts = sick[col].value_counts().reindex(categories, fill_value=0)
            h_pct = h_counts / h_counts.sum()
            s_pct = s_counts / s_counts.sum()
            w = 0.35
            ax.bar(
                x - w / 2, h_pct, w, color=HEALTHY_COLOR, alpha=0.8, label="No disease"
            )
            ax.bar(
                x + w / 2, s_pct, w, color=SICK_COLOR, alpha=0.8, label="Heart disease"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=8)
        else:
            ax.hist(
                healthy[col].dropna(),
                bins=20,
                color=HEALTHY_COLOR,
                alpha=0.6,
                density=True,
                label="No disease",
            )
            ax.hist(
                sick[col].dropna(),
                bins=20,
                color=SICK_COLOR,
                alpha=0.6,
                density=True,
                label="Heart disease",
            )

        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.yaxis.set_visible(False)

    handles = [
        mpatches.Patch(color=HEALTHY_COLOR, alpha=0.8, label="No heart disease"),
        mpatches.Patch(color=SICK_COLOR, alpha=0.8, label="Heart disease"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=11,
        framealpha=0,
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = SLIDES_DIR / "feature_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating slide plots...")
    plot_patient_radar()
    plot_feature_scatter()
    print("Done.")
