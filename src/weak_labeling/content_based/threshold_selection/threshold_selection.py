from typing import Optional, Tuple

import pandas as pd

from src.preprocessing.cleanup import separate_real_and_fake
from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_csv, read_model_feature_injected_csv
from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.content_based.feature_injection.content_feature_injection import (
    normalize_content_features,
)
from src.weak_labeling.content_based.threshold_selection.consts import (
    FEATURE_PERCENTILES,
    Direction,
    UPPER_PERCENTILES,
    LOWER_PERCENTILES,
)
from src.weak_labeling.content_based.threshold_selection.utils import (
    update_and_save_thresholds_json,
)


def find_threshold_for_feature(
    feature: str,
    percentiles_dict: dict,
    direction: Direction = Direction.UPPER,
    count_feature: bool = False,
) -> Optional[dict]:
    print(f"Finding threshold for feature: {feature} and direction: {direction}")

    threshold, side, selected_percentile = None, None, None
    if count_feature:
        percentiles_to_process = FEATURE_PERCENTILES
    else:
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
        if abs(percentiles_dict[percentile]["diff"]) > (
            (total_max_diff / 18)
        ):  # 1 / 200
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


def get_thresholds_for_content_features(injected_df: pd.DataFrame):
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
        if (
            feature in ContentFeatures.get_stylistic_count_features_content()
            or feature in ContentFeatures.get_stylistic_count_features_title()
        ):
            threshold = find_threshold_for_feature(
                feature=feature,
                percentiles_dict=percentiles_dict,
                count_feature=True,
            )
            if threshold:
                feature_thresholds[Direction.UPPER.value] = threshold
        else:
            threshold = find_threshold_for_feature(
                feature=feature,
                percentiles_dict=percentiles_dict,
                direction=Direction.UPPER,
            )
            if threshold:
                feature_thresholds[Direction.UPPER.value] = threshold

            threshold = find_threshold_for_feature(
                feature=feature,
                percentiles_dict=percentiles_dict,
                direction=Direction.LOWER,
            )
            if threshold:
                feature_thresholds[Direction.LOWER.value] = threshold

        print(feature_thresholds)

        thresholds[feature] = feature_thresholds

    return thresholds


if __name__ == "__main__":
    run_configs = [
        {
            "dataset": Datasets.NELA_ELECTIONS_2020,
            "train_dataset": Datasets.NELA_COVID_COMBINED,
        },
        {
            "dataset": Datasets.NELA_COVID_COMBINED,
            "train_dataset": Datasets.NELA_ELECTIONS_2020,
        },
        {
            "dataset": Datasets.NELA_ELECTIONS_SUBSET_2020,
        },
        {
            "dataset": Datasets.NELA_COVID_SUBSET,
        },
    ]
    for config in run_configs:
        if "train_dataset" in config.keys():
            df = read_model_feature_injected_csv(config["dataset"].injected_path, config["train_dataset"].name)
        else:
            df = read_csv(config["dataset"].injected_path)
        # df = normalize_content_features(df, features=ContentFeatures.get_all())
        thresholds = get_thresholds_for_content_features(df)

        update_and_save_thresholds_json(updated_thresholds=thresholds, dataset=config["dataset"])
