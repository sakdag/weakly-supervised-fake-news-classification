import pandas as pd
from sklearn import preprocessing

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    save_to_csv,
    read_csv,
    balanced_undersample,
)
from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.content_based.feature_injection.readability_features import (
    append_readability_features,
)
from src.weak_labeling.content_based.feature_injection.stylistic_features import (
    append_stylistic_features,
)


def normalize_content_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    scaler = preprocessing.MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def run_content_feature_injection(
    dataset: Dataset,
    zero: bool = False,
    parallelize: bool = False,
    stylistic_features: bool = True,
    readability_features: bool = True,
    pos_tagging_features: bool = True,
    calculated_features: bool = True,
):
    """
    Injects content features into given dataset
    :param dataset: Input dataset containing content and title columns
    :param zero: If True, eliminates all previously injected features and starts from scratch.
                 By default, this is false meaning old features in dataset will not be eliminated
    :param parallelize: If provided, feature injection utilizes Dask to parallelize the df apply
    :param stylistic_features: flag to enable/disable injecting stylistic content features
    :param readability_features: flag to enable/disable injecting readability content features
    :param pos_tagging_features: flag to enable/disable injecting pos_tagging content features
    :param calculated_features: flag to enable/disable injecting calculated content features
    :return: Saves and returns injected dataset
    """
    if zero:
        injected_df = read_csv(path=dataset.sampled_path)
    else:
        injected_df = read_csv(path=dataset.injected_path)

    # injected_df = injected_df.sample(n=100)

    # if zero:
    #     injected_df = balanced_undersample(df=injected_df, n=10000)
    print(injected_df.head())

    stylistic_feature_list = []
    if stylistic_features:
        stylistic_feature_list.extend(ContentFeatures.get_stylistic_count_features())
        stylistic_feature_list.extend(ContentFeatures.get_stylistic_ratio_features())
    if pos_tagging_features:
        stylistic_feature_list.extend(ContentFeatures.get_pos_tagging_features())
        stylistic_feature_list.extend(ContentFeatures.get_punctuation_features())

    all_features = []
    if stylistic_feature_list:
        all_features.extend(stylistic_feature_list)
        injected_df = append_stylistic_features(
            injected_df,
            stylistic_feature_list,
        )
    if readability_features:
        all_features.extend(ContentFeatures.get_readability_features())
        injected_df = append_readability_features(
            injected_df, ContentFeatures.get_readability_features()
        )
    if calculated_features:
        all_features.extend(ContentFeatures.get_calculated_features())
        pass

    print(injected_df.head())

    # injected_df = normalize_content_features(injected_df, all_features)
    # print(injected_df.head())

    save_to_csv(injected_df, dataset.injected_path)


if __name__ == "__main__":
    datasets = [
        Datasets.NELA_COVID_2021,
        Datasets.NELA_ELECTIONS_2020,
        # Datasets.NELA_COVID_2020,
        # Datasets.NELA_CAPITOL_2021,
    ]
    for dataset in datasets:
        run_content_feature_injection(
            dataset=dataset,
            zero=True,
            parallelize=False,
            stylistic_features=True,
            pos_tagging_features=True,
            readability_features=True,
            calculated_features=False,
        )
