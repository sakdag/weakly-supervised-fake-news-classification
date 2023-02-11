from typing import List

import pandas as pd
from tqdm import tqdm
from readability import Readability

from src.weak_labeling.consts.content_features import ContentFeatures


def append_readability_features(df: pd.DataFrame, features=List[str]) -> pd.DataFrame:
    if not features:
        return df

    # Initialize empty columns to fill while iterating
    tqdm.pandas()

    res = df.progress_apply(
        append_readability_features_helper,
        axis=1,
        result_type="expand",
        features_to_apply=features,
    )
    res.columns = features

    full = df.merge(res, left_index=True, right_index=True)
    print(full)

    return full


def append_readability_features_helper(row, **kwargs):
    features = kwargs["features_to_apply"]

    readability = Readability(row.content)

    features_to_return = []
    if ContentFeatures.CONTENT_GUNNING_FOG_INDEX in features:
        features_to_return.append(readability.gunning_fog().score)
    if ContentFeatures.CONTENT_AUTOMATED_READABILITY_INDEX in features:
        features_to_return.append(readability.ari().score)
    if ContentFeatures.CONTENT_FLESCH_KINCAID_INDEX in features:
        features_to_return.append(readability.flesch_kincaid().score)

    return tuple(features_to_return)
