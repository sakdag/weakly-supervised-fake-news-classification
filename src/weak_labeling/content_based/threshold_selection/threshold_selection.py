from typing import Optional, Tuple

import pandas as pd

from src.preprocessing.cleanup import separate_real_and_fake
from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv
from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.content_based.threshold_selection.consts import (
    FEATURE_PERCENTILES,
    Direction,
    UPPER_PERCENTILES,
    LOWER_PERCENTILES,
)
from src.weak_labeling.content_based.threshold_selection.utils import (
    get_thresholds_json,
    update_and_save_thresholds_json,
)


def find_threshold_for_feature(
    direction: Direction, feature: str, percentiles_dict: dict
) -> Optional[dict]:
    print(f"Finding threshold for feature: {feature} and direction: {direction}")

    threshold, side, selected_percentile = None, None, None
    percentiles_to_process = (
        UPPER_PERCENTILES if direction == Direction.UPPER else LOWER_PERCENTILES
    )
    real_max_diff = abs(
        percentiles_dict[FEATURE_PERCENTILES[0]]["real"]
        - percentiles_dict[FEATURE_PERCENTILES[-1]]["real"]
    )
    fake_max_diff = abs(
        percentiles_dict[FEATURE_PERCENTILES[0]]["fake"]
        - percentiles_dict[FEATURE_PERCENTILES[-1]]["fake"]
    )
    total_max_diff = min(real_max_diff, fake_max_diff)

    print(total_max_diff)
    print(real_max_diff)
    print(fake_max_diff)

    print(percentiles_dict)

    for percentile in percentiles_to_process:
        if abs(percentiles_dict[percentile]["diff"]) > (total_max_diff / 40):
            if percentiles_dict[percentile]["diff"] > 0:
                side = "real" if direction == Direction.UPPER else "fake"
                threshold = percentiles_dict[percentile]["real"]
            else:
                side = "fake" if direction == Direction.UPPER else "real"
                threshold = percentiles_dict[percentile]["fake"]
            selected_percentile = percentile
            break

    if threshold and side:
        print(
            f"Adding labeling function, feature: {feature}, direction: {Direction.UPPER}, side: {side}, threshold: {threshold}"
        )
        return {
            "side": side,
            "threshold": threshold,
            "percentile": selected_percentile,
        }

    return None


def select_thresholds_for_content_features(injected_df: pd.DataFrame):
    real_df, fake_df = separate_real_and_fake(injected_df)

    real_desc = real_df.describe(percentiles=FEATURE_PERCENTILES)
    fake_desc = fake_df.describe(percentiles=FEATURE_PERCENTILES)

    thresholds = {}

    for feature in ContentFeatures.get_all():
        wps_desc_df = pd.concat(
            [real_desc[feature], fake_desc[feature]], axis=1, keys=["real", "fake"]
        )
        wps_desc_df["abs_diff"] = wps_desc_df["real"] - wps_desc_df["fake"]

        print(feature)
        print(wps_desc_df)

        percentiles_dict = {}
        for idx, row in wps_desc_df.iterrows():
            if "%" in row.name:
                percentiles_dict[int(str(row.name).split("%")[0]) / 100] = {
                    "real": row.real,
                    "fake": row.fake,
                    "diff": row.real - row.fake,
                }

        feature_thresholds = {}
        threshold = find_threshold_for_feature(
            direction=Direction.UPPER,
            feature=feature,
            percentiles_dict=percentiles_dict,
        )
        if threshold:
            feature_thresholds[Direction.UPPER.value] = threshold

        threshold = find_threshold_for_feature(
            direction=Direction.LOWER,
            feature=feature,
            percentiles_dict=percentiles_dict,
        )
        if threshold:
            feature_thresholds[Direction.LOWER.value] = threshold

        print(feature_thresholds)

        thresholds[feature] = feature_thresholds

    update_and_save_thresholds_json(updated_thresholds=thresholds)


if __name__ == "__main__":
    df = read_raw_data(Datasets.INJECTED_NELA_COVID_2021_DATASET_SAMPLED)
    select_thresholds_for_content_features(df)
