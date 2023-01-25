from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv
from src.weak_labeling.content_based.labeling_functions.lf_creation import (
    create_content_lfs,
)
from src.weak_labeling.model_based.labeling_functions.model_based_lfs import (
    get_model_based_lfs,
)
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

pd.set_option("display.max_columns", None)


def generate_labels(
    df: pd.DataFrame, lfs: List[LabelingFunction]
):
    train, test = train_test_split(df, train_size=0.8)
    print(f"{len(train)} - {len(test)}")

    applier = PandasLFApplier(lfs)
    l_train = applier.apply(df=train)
    l_test = applier.apply(df=test)

    analysis = LFAnalysis(L=l_train, lfs=lfs).lf_summary(Y=train.label.values)

    print(analysis)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=l_train, n_epochs=500, log_freq=100, seed=1234)

    # majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    #     "accuracy"
    # ]
    # print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

    label_model_acc = label_model.score(L=l_test, Y=test.label.values, tie_break_policy="random")[
        "accuracy"
    ]
    print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

    # df["predicted_labels"] = label_model.predict(l_train + l_test)
    return df


if __name__ == "__main__":
    content_lfs = create_content_lfs()
    model_based_lfs = get_model_based_lfs()
    labeling_functions = content_lfs + model_based_lfs

    print(labeling_functions)

    a_df = read_raw_data(Datasets.INJECTED_NELA_ELECTIONS_DATASET_2020_SAMPLED)

    df_with_predictions = generate_labels(a_df, labeling_functions)

    save_to_csv(
        df_with_predictions, Datasets.LABELED_NELA_ELECTIONS_DATASET_2020_SAMPLED
    )
