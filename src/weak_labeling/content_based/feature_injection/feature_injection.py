import pandas as pd

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv
from src.weak_labeling.content_based.feature_injection.stylistic_features import (
    append_stylistic_features,
)


def inject_features(df_to_inject: pd.DataFrame) -> pd.DataFrame:
    df_to_inject = append_stylistic_features(df_to_inject)
    return df_to_inject


if __name__ == "__main__":
    df = read_raw_data(Datasets.NELA_COVID_DATASET_SAMPLED)
    injected_df = inject_features(df)
    print(injected_df.head())
    save_to_csv(injected_df, Datasets.INJECTED_NELA_COVID_DATASET_SAMPLED)
