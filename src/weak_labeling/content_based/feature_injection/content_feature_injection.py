import pandas as pd

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import (
    read_raw_data,
    save_to_csv,
    balanced_undersample,
)
from src.weak_labeling.content_based.feature_injection.stylistic_features import (
    append_stylistic_features,
)
from src.weak_labeling.utils import get_injected_dataset_name


def inject_content_features(df_to_inject: pd.DataFrame) -> pd.DataFrame:
    df_to_inject = append_stylistic_features(df_to_inject)
    return df_to_inject


def run_content_feature_injection(dataset: Datasets):
    injected_dataset = get_injected_dataset_name(dataset)
    if not injected_dataset:
        return

    df = read_raw_data(dataset)
    if injected_dataset != dataset:
        df = balanced_undersample(df=df, n=10000)
    print(df.head())

    injected_df = inject_content_features(df)
    print(injected_df.head())

    save_to_csv(injected_df, injected_dataset)


if __name__ == "__main__":
    # run_content_feature_injection(dataset=Datasets.NELA_COVID_2021_DATASET_SAMPLED)
    run_content_feature_injection(dataset=Datasets.NELA_COVID_2020_DATASET_SAMPLED)
    run_content_feature_injection(dataset=Datasets.NELA_CAPITOL_2021_DATASET_SAMPLED)
    run_content_feature_injection(dataset=Datasets.NELA_ELECTIONS_2020_DATASET_SAMPLED)
