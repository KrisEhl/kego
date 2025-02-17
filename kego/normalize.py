import pandas as pd


class Normalizer:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def fix_types(
        self,
        columns_to_ignore: list[str] | None = [],
        features: list[str] | None = None,
    ):
        if columns_to_ignore is None:
            columns_to_ignore = []

        self.features = features
        if self.features is None:
            self.features = list(self.train.columns)

        self.features = [c for c in self.train.columns if not c in columns_to_ignore]

        print(f"There are {len(self.features)} FEATURES: {self.features}")

        FEATURES_CATEGORICAL = []
        for c in self.features:
            if self.train[c].dtype == "object":
                FEATURES_CATEGORICAL.append(c)
                self.train[c] = self.train[c].fillna("NAN")
                self.test[c] = self.test[c].fillna("NAN")
        print(
            f"In these features, there are {len(FEATURES_CATEGORICAL)} CATEGORICAL FEATURES: {FEATURES_CATEGORICAL}"
        )
        self.features_categorical = FEATURES_CATEGORICAL

        if self.test is not None:
            combined = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        else:
            combined = self.train

        print("Combined data shape:", combined.shape)

        # LABEL ENCODE CATEGORICAL FEATURES
        print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")

        for c in self.features:

            # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
            if c in FEATURES_CATEGORICAL:
                print(f"{c}, ", end="")
                combined[c], _ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")
                combined[c] = combined[c].astype("category")

            # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
            else:
                if combined[c].dtype == "float64":
                    combined[c] = combined[c].astype("float32")
                if combined[c].dtype == "int64":
                    combined[c] = combined[c].astype("int32")

        if self.test is not None:
            self.train = combined.iloc[: len(self.train)].copy()
            self.test = combined.iloc[len(self.train) :].reset_index(drop=True).copy()
        else:
            combined = self.train
        return self.train, self.test
