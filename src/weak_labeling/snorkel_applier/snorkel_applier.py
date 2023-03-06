from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data, save_to_csv, read_csv
from src.weak_labeling.content_based.labeling_functions.lf_creation import (
    create_content_lfs,
)
from src.weak_labeling.model_based.labeling_functions.model_based_lfs import (
    get_model_based_lfs,
)
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

pd.set_option("display.max_columns", None)


def generate_labels(df: pd.DataFrame, lfs: List[LabelingFunction], top_n: int = -1):
    df = df.sample(frac=1)
    train, test = train_test_split(df, train_size=0.8, shuffle=False)
    print(f"{len(train)} - {len(test)}")

    applier = PandasLFApplier(lfs)
    l_train = applier.apply(df=train)
    l_test = applier.apply(df=test)

    analysis = LFAnalysis(L=l_train, lfs=lfs).lf_summary(Y=train.label.values)
    print(tabulate(analysis, headers="keys", tablefmt="psql"))

    if top_n != -1:
        sorted_analysis = analysis.sort_values("Emp. Acc.", ascending=False)
        best_lf_names = sorted_analysis.iloc[:top_n, 0]
        best_lfs = [x for x in lfs if x.name in best_lf_names]

        applier = PandasLFApplier(best_lfs)
        l_train = applier.apply(df=train)
        l_test = applier.apply(df=test)

        analysis = LFAnalysis(L=l_train, lfs=best_lfs).lf_summary(Y=train.label.values)
        print(tabulate(analysis, headers="keys", tablefmt="psql"))

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=l_train, n_epochs=500, log_freq=100, seed=1234)

    majority_model = MajorityLabelVoter()
    majority_model_score = majority_model.score(
        L=l_test, Y=test.label.values, tie_break_policy="abstain"
    )
    print(
        f"{'Majority Vote Accuracy:':<25} {majority_model_score['accuracy'] * 100:.1f}%"
    )
    print(majority_model_score)

    label_model_score = label_model.score(
        L=l_test, Y=test.label.values, tie_break_policy="abstain"
    )
    print(f"{'Label Model Accuracy:':<25} {label_model_score['accuracy'] * 100:.1f}%")
    print(label_model_score)

    # probabilistic_labels = label_model.predict_proba(np.concatenate([l_train, l_test]))
    # print(probabilistic_labels)

    # for idx, row in df.iterrows():
    #     print(row[0])
    #     print(row[1])
    #     print(type(row[0]))
    #     break

    df["majority_vote_label"] = label_model.predict(np.concatenate([l_train, l_test]))
    df["label_model_label"] = label_model.predict(np.concatenate([l_train, l_test]))
    return df


if __name__ == "__main__":
    content_lfs = create_content_lfs(dataset=Datasets.NELA_COVID_2021)
    # model_based_lfs = get_model_based_lfs()
    labeling_functions = content_lfs  # + model_based_lfs

    print(labeling_functions)

    a_df = read_csv(Datasets.NELA_ELECTIONS_2020.injected_path)
    # a_df = normalize_content_features(a_df, features=ContentFeatures.get_all())

    df_with_predictions = generate_labels(a_df, labeling_functions, 6)

    save_to_csv(df_with_predictions, Datasets.NELA_ELECTIONS_2020.weak_labeled_path)
