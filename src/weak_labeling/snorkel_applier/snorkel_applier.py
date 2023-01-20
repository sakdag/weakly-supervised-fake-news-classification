from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv
from src.weak_labeling.content_based.labeling_functions.lf_creation import (
    create_content_lfs,
)
from src.weak_labeling.model_based.model_based_lfs import get_model_based_lfs
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel


def generate_labels(
    df: pd.DataFrame, lfs: List[LabelingFunction], sample_n: Optional[int] = None
):
    train, test = train_test_split(df, train_size=0.8)
    print(f"{len(train)} - {len(test)}")

    if sample_n:
        train = train.sample(n=sample_n).reset_index(drop=True)

    applier = PandasLFApplier(lfs)
    l_train = applier.apply(train)

    LFAnalysis(L=l_train, lfs=lfs).lf_summary(Y=train.label.values)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=train, n_epochs=500, log_freq=100, seed=1234)

    predictions = label_model.predict(train)

    matches = np.count_nonzero(train["label"] == predictions)
    mismatches = predictions.size - matches
    print(mismatches / predictions.size * 100)

    df["predicted_labels"] = predictions
    return df


if __name__ == "__main__":
    content_lfs = create_content_lfs()
    model_based_lfs = get_model_based_lfs()
    labeling_functions = content_lfs + model_based_lfs

    a_df = read_raw_data(Datasets.INJECTED_NELA_ELECTIONS_DATASET_SAMPLED)

    df_with_predictions = generate_labels(a_df, labeling_functions)

    save_to_csv(df_with_predictions, Datasets.LABELED_NELA_ELECTIONS_DATASET_SAMPLED)
