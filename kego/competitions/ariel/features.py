import multiprocessing
import os

import dotenv
import pandas as pd
import tqdm


def get_phase(i, planet_id, n_steps, dataset, adc_info):
    f_signal = pd.read_parquet(
        os.environ["FOLDER_COMPETITION"] + f"{dataset}/{planet_id}/FGS1_signal.parquet"
    )
    mean_signal = f_signal.values.mean(axis=1)  # mean over the 32*32 pixels
    net_signal = mean_signal[1::2] - mean_signal[0::2]
    gain = adc_info.FGS1_adc_gain.values[i]
    return [
        net_signal[i * n_steps : (i + 1) * n_steps].mean() * gain
        for i in range(len(net_signal) // n_steps + 1)
    ]


if __name__ == "__main__":
    dotenv.load_dotenv(override=True, verbose=True)
    dataset = "train"
    train_adc_info = pd.read_csv(
        os.environ["FOLDER_COMPETITION"] + "train_adc_info.csv", index_col="planet_id"
    )
    adc_info = train_adc_info
    for n_steps in [100, 400, 1000, 2000, 8000][::-1]:
        print(f"Starting {n_steps=} ...")
        planet_ids = adc_info.index

        indices = range(len(planet_ids))
        print(planet_ids)
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        #     result = pool.apply_async(
        phases = []
        for i, planet_id in tqdm.tqdm(zip(indices, planet_ids), total=len(indices)):
            phase = get_phase(i, planet_id, n_steps, dataset, adc_info)
        phases.append(phase)
        # phases = result.get()
        df = pd.DataFrame(
            phases, columns=[f"phase_{i}" for i in range(len(phases[0]))], index=indices
        )
        filepath = f"phases_step{n_steps}_nids{len(indices)}.csv"
        print(f"... saving {filepath}")
        df.to_csv(filepath)
