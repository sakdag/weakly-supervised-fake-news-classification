import pandas as pd

from src.preprocessing.cleanup import separate_real_and_fake
from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv
from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.content_based.threshold_selection.consts import (
    FEATURE_PERCENTILES,
)


def select_thresholds_for_content_features(injected_df: pd.DataFrame):
    real_df, fake_df = separate_real_and_fake(injected_df)

    real_desc = real_df.describe(percentiles=FEATURE_PERCENTILES)
    fake_desc = fake_df.describe(percentiles=FEATURE_PERCENTILES)

    for feature in ContentFeatures.get_all():
        wps_desc_df = pd.concat(
            [real_desc[feature], fake_desc[feature]], axis=1, keys=["real", "fake"]
        )
        wps_desc_df["abs_diff"] = wps_desc_df["real"] - wps_desc_df["fake"]
        print(feature)
        print(wps_desc_df)
        print("\n\n")

        # TODO: Calculate the thresholds and update json with them

    return real_desc, fake_desc


if __name__ == "__main__":
    df = read_raw_data(Datasets.INJECTED_NELA_COVID_DATASET_SAMPLED)
    select_thresholds_for_content_features(df)
    # save_to_csv(injected_df, Datasets.INJECTED_NELA_COVID_DATASET_SAMPLED)
