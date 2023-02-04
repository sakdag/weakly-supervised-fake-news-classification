from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    save_to_csv,
    read_csv,
)
from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.content_based.feature_injection.readability_features import (
    append_readability_features,
)
from src.weak_labeling.content_based.feature_injection.stylistic_features import (
    append_stylistic_features,
)


def run_content_feature_injection(
    dataset: Dataset,
    zero: bool = False,
    parallelize: bool = False,
    stylistic_features: bool = True,
    readability_features: bool = True,
):
    """
    Injects content features into given dataset
    :param dataset: Input dataset containing content and title columns
    :param zero: If True, eliminates all previously injected features and starts from scratch.
                 By default, this is false meaning old features in dataset will not be eliminated
    :param parallelize: If provided, feature injection utilizes Dask to parallelize the df apply
    :param stylistic_features: flag to enable/disable injecting stylistic content features
    :param readability_features: flag to enable/disable injecting readability content features
    :return: Saves and returns injected dataset
    """
    if zero:
        injected_df = read_csv(path=dataset.sampled_path)
    else:
        injected_df = read_csv(path=dataset.injected_path)

    # if zero:
    #     injected_df = balanced_undersample(df=injected_df, n=1000)
    print(injected_df.head())

    if stylistic_features:
        injected_df = append_stylistic_features(
            injected_df,
             ContentFeatures.get_stylistic_count_features() + ContentFeatures.get_stylistic_ratio_features()
        )
    if readability_features:
        injected_df = append_readability_features(injected_df, ContentFeatures.get_readability_features())

    print(injected_df.head())

    save_to_csv(injected_df, dataset.injected_path)


if __name__ == "__main__":
    # run_content_feature_injection(dataset=Datasets.NELA_ELECTIONS_2020, zero=False)
    # run_content_feature_injection(dataset=Datasets.NELA_COVID_2020, zero=False)
    # run_content_feature_injection(dataset=Datasets.NELA_CAPITOL_2021, zero=False)
    run_content_feature_injection(
        dataset=Datasets.NELA_ELECTIONS_2020,
        zero=False,
        parallelize=False,
        stylistic_features=False,
        readability_features=True,
    )
