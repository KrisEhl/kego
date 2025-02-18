from typing import Sequence

import numpy as np
import pandas as pd

import kego.lists


class Normalizer:
    def __init__(self, *dfs: pd.DataFrame):
        self.dfs = list(dfs)

    def fix_types(
        self,
        columns_to_ignore: list[str] | None = [],
        features: list[str] | None = None,
    ):
        if columns_to_ignore is None:
            columns_to_ignore = []
        self.columns_to_ignore = columns_to_ignore

        self.features = features
        if self.features is None:
            self.features = list(
                set(kego.lists.flatten_list([df.columns for df in self.dfs]))
            )

        self.features = [c for c in self.features if not c in columns_to_ignore]

        print(f"There are {len(self.features)} FEATURES: {self.features}")

        FEATURES_CATEGORICAL = []
        for feature in self.features:
            if any([df[feature].dtype == "object" for df in self.dfs]):
                FEATURES_CATEGORICAL.append(feature)
                self.dfs = [
                    df.assign(**{feature: df[feature].fillna("NAN")}) for df in self.dfs
                ]
        print(
            f"In these features, there are {len(FEATURES_CATEGORICAL)} CATEGORICAL FEATURES: {FEATURES_CATEGORICAL}"
        )
        self.features_categorical = FEATURES_CATEGORICAL
        combined = pd.concat(self.dfs, axis=0, ignore_index=True)

        print("Combined data shape:", combined.shape)

        # LABEL ENCODE CATEGORICAL FEATURES
        print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")

        for feature in self.features:

            # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
            if feature in FEATURES_CATEGORICAL:
                print(f"{feature}, ", end="")
                combined[feature], _ = combined[feature].factorize()
                combined[feature] -= combined[feature].min()
                combined[feature] = combined[feature].astype("int32")
                combined[feature] = combined[feature].astype("category")

            # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
            else:
                if combined[feature].dtype == "float64":
                    combined[feature] = combined[feature].astype("float32")
                if combined[feature].dtype == "int64":
                    combined[feature] = combined[feature].astype("int32")

        n_dfs = [0] + np.cumsum([len(df) for df in self.dfs]).tolist()
        self.dfs = [
            combined.loc[n_dfs[i] : n_dfs[i + 1] - 1] for i in range(len(self.dfs))
        ]
        return self.dfs
