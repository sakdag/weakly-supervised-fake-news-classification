import pandas as pd

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv, balanced_undersample
from src.weak_labeling.content_based.feature_injection.stylistic_features import (
    append_stylistic_features,
)


def inject_features(df_to_inject: pd.DataFrame) -> pd.DataFrame:

    df_to_inject = append_stylistic_features(df_to_inject)
    return df_to_inject

def run_feature_injection(dataset: Datasets):
    if dataset == Datasets.NELA_ELECTIONS_2020_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_ELECTIONS_DATASET_2020_SAMPLED
    elif dataset == Datasets.NELA_CAPITOL_2021_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_CAPITOL_DATASET_2021_SAMPLED
    elif dataset == Datasets.NELA_COVID_2020_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_COVID_2020_DATASET_SAMPLED
    elif dataset == Datasets.NELA_COVID_2021_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_COVID_2021_DATASET_SAMPLED
    else:
        print(f"Cant inject to dataset: {dataset}")
        return

    df = read_raw_data(dataset)
    df = balanced_undersample(df=df, n=10000)
    print(df.head())

    injected_df = inject_features(df)
    print(injected_df.head())

    save_to_csv(injected_df, injected_dataset)


if __name__ == "__main__":
    run_feature_injection(dataset=Datasets.NELA_ELECTIONS_2020_DATASET_SAMPLED)
