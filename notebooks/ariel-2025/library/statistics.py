import os
from collections import defaultdict
from glob import glob

import pandas as pd


def check_files(train, train_info, wavelengths, path_data_train):
    # --- 1. Basic Train Set Stats ---
    print("🪐 Number of training planets:", train.shape[0])
    print("📈 Number of target labels (wavelengths):", train.shape[1] - 1)
    print("🔬 Length of wavelength grid:", wavelengths.shape[0])

    # --- 2. Target Stats (per flux column) ---
    target_cols = [col for col in train.columns if col != "planet_id"]
    flux_summary = train[target_cols].describe()
    print("\n📊 Flux value summary (first 5 rows):")
    print(flux_summary)

    # --- 3. Unique Stars ---
    if "planet_id" in train_info.columns:
        num_stars = train_info.drop("planet_id").unique().shape[0]
    else:
        num_stars = train_info.unique().shape[0]
    print("\n🌟 Number of unique stars in training:", num_stars)

    # --- 4. Planets with Multiple Observations ---
    obs_counts = defaultdict(int)
    train_planets = os.listdir(path_data_train)

    for pid in train_planets:
        air_obs = glob(f"train/{pid}/AIRS-CH0_signal_*.parquet")
        obs_counts[pid] = len(air_obs)

    multi_obs = {pid: count for pid, count in obs_counts.items() if count > 1}
    print("\n🔁 Planets with multiple observations:", len(multi_obs))

    # --- 5. Check Calibration File Coverage ---
    missing_calibs = []
    expected = {"dark", "dead", "flat", "linear_corr", "read"}

    for pid in train_planets:
        for band in ["AIRS-CH0", "FGS1"]:
            calib_path = f"train/{pid}/{band}_calibration"
            calib_files = (
                {os.path.splitext(f)[0] for f in os.listdir(calib_path)}
                if os.path.exists(calib_path)
                else set()
            )
            missing = expected - calib_files
            if missing:
                missing_calibs.append((pid, band, missing))

    print("\n🧪 Planets missing calibration files:", len(missing_calibs))
    if missing_calibs:
        print("   Example:", missing_calibs[0])

    # --- 6. Optional: Distribution of Observations Per Planet ---
    obs_distribution = pd.Series(list(obs_counts.values())).value_counts().sort_index()
    print("\n🗂 Observation count distribution per planet (AIR-CH0):")
    print(obs_distribution)

    # --- 7. Planet-Star Uniqueness Check ---
    merged = train[["planet_id"]].join(train_info, on="planet_id", how="left")
    unique_links = merged[
        ["planet_id"] + [col for col in train_info.columns if col != "planet_id"]
    ].unique()
    print("\n🔗 Unique planet-star mappings:", unique_links.shape[0])
