from typing import List, Optional

import numpy as np
import pandas as pd
from hyperlm import HyperLabelModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    read_raw_data,
    save_to_csv,
    read_csv,
    read_model_feature_injected_csv,
    save_weak_labeled_csv,
)
from src.weak_labeling.content_based.labeling_functions.lf_creation import (
    create_content_lfs,
)
from src.weak_labeling.model_based.labeling_functions.model_based_lfs import (
    get_model_based_lfs,
)
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

from src.weak_labeling.weak_labeler.weak_label_aggregators import WeakLabelAggregators

pd.set_option("display.max_columns", None)


def generate_labels(
    df: pd.DataFrame,
    lfs: List[LabelingFunction],
    weak_label_aggregators: list,
    top_n: int = 0,
    mid_range_to_eliminate: int = 0,
):
    """Applies provided labeling functions to the dataframe, then aggregates generated weak labels"""

    df = df.sample(frac=1)
    applier = PandasLFApplier(lfs)
    weak_labels = applier.apply(df=df)

    analysis = LFAnalysis(L=weak_labels, lfs=lfs).lf_summary(Y=df.label.values)
    print(tabulate(analysis, headers="keys", tablefmt="psql"))

    if top_n > 0:
        sorted_analysis = analysis.sort_values("Emp. Acc.", ascending=False)
        best_lf_names = sorted_analysis.iloc[:top_n, 0]
        best_lfs = [x for x in lfs if x.name in best_lf_names]

        applier = PandasLFApplier(best_lfs)
        weak_labels = applier.apply(df=df)

        analysis = LFAnalysis(L=weak_labels, lfs=best_lfs).lf_summary(Y=df.label.values)
        print(tabulate(analysis, headers="keys", tablefmt="psql"))

    weak_labeled_df = df[["title", "content", "label"]].copy()
    weak_labeled_df = weak_labeled_df.rename(columns={"label": "actual_label"})
    for aggregator in weak_label_aggregators:
        if aggregator == WeakLabelAggregators.SNORKEL_MAJORITY_VOTE:
            agg_labels = aggregate_with_snorkel_majority_vote(
                weak_labels=weak_labels, actual_labels=df.label.values
            )
            assert len(agg_labels) == len(df)
            weak_labeled_df[f"{aggregator.lower()}_label"] = agg_labels
        elif aggregator == WeakLabelAggregators.SNORKEL_LABEL_MODEL:
            prob_agg_labels, agg_labels = aggregate_with_snorkel_label_model(
                weak_labels=weak_labels,
                actual_labels=df.label.values,
                mid_range_to_eliminate=mid_range_to_eliminate,
            )
            assert len(prob_agg_labels) == len(weak_labels)
            assert len(agg_labels) == len(weak_labels)
            weak_labeled_df[f"{aggregator.lower()}_label"] = agg_labels
            fake_probs = [probs[1] for probs in prob_agg_labels]
            assert len(fake_probs) == len(weak_labels)
            weak_labeled_df[f"{aggregator.lower()}_fake_prob"] = fake_probs
        if aggregator == WeakLabelAggregators.HYPER_LABEL_MODEL:
            agg_labels = aggregate_with_hyper_label_model(
                weak_labels=weak_labels, actual_labels=df.label.values
            )
            assert len(agg_labels) == len(df)
            weak_labeled_df[f"{aggregator.lower()}_label"] = agg_labels

    return weak_labeled_df


def aggregate_with_snorkel_majority_vote(weak_labels, actual_labels):
    majority_model = MajorityLabelVoter()
    majority_model_score = majority_model.score(
        L=weak_labels,
        Y=actual_labels,
        tie_break_policy="abstain",
        metrics=["accuracy", "precision", "recall", "f1", "coverage"],
    )
    print("Majority vote score:")
    print(majority_model_score)
    return majority_model.predict(weak_labels)


def aggregate_with_snorkel_label_model(
    weak_labels, actual_labels, mid_range_to_eliminate: int = 0
):
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=weak_labels, n_epochs=500, log_freq=100, seed=1234)

    label_model_score = label_model.score(
        L=weak_labels,
        Y=actual_labels,
        tie_break_policy="abstain",
        metrics=["accuracy", "precision", "recall", "f1", "coverage"],
    )
    print("Label model score:")
    print(label_model_score)

    # Used only for testing mid-range elimination
    probabilistic_labels = label_model.predict_proba(weak_labels)
    if mid_range_to_eliminate > 0:
        low_cutoff = round(50 - (mid_range_to_eliminate / 2)) / 100
        abstain_count = 0
        mid_range_count = 0
        new_labels = []
        new_actual_labels = []
        print(f"low_cutoff: {low_cutoff}")
        for prob_label, actual_label in zip(probabilistic_labels, actual_labels):
            real_prob = prob_label[0]
            fake_prob = prob_label[1]
            if real_prob == 0.5:
                abstain_count += 1
            elif (real_prob > fake_prob > low_cutoff) or (
                fake_prob > real_prob > low_cutoff
            ):
                mid_range_count += 1
            else:
                new_actual_labels.append(actual_label)
                new_labels.append(0 if prob_label[0] > 0.5 else 1)

        print(f"Number of elements to abstain {abstain_count}")
        print(f"Number of elements in mid range {mid_range_count}")
        print(f"New classification report after mid range elimination:")
        cr = classification_report(new_actual_labels, new_labels)
        print(cr)

    return probabilistic_labels, label_model.predict(weak_labels)


def aggregate_with_hyper_label_model(weak_labels, actual_labels):
    hlm = HyperLabelModel()
    predictions = hlm.infer(weak_labels)
    print("Hyper label model score:")
    cr = classification_report(actual_labels, predictions, digits=5)
    print(cr)
    return predictions


def run_weak_labeler(
    dataset_to_weak_label: Dataset, train_dataset: Dataset, weak_label_aggregators: list
):
    content_lfs = create_content_lfs(dataset=train_dataset)
    model_based_lfs = get_model_based_lfs()
    labeling_functions = content_lfs + model_based_lfs

    print(labeling_functions)

    a_df = read_model_feature_injected_csv(
        dataset_to_weak_label.injected_path, train_dataset.name
    )

    df_with_predictions = generate_labels(
        a_df,
        labeling_functions,
        weak_label_aggregators=weak_label_aggregators,
        top_n=10,
        mid_range_to_eliminate=0,
    )

    save_weak_labeled_csv(
        df_with_predictions,
        Datasets.NELA_ELECTIONS_2020.weak_labeled_path,
        train_dataset.name,
    )


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
            "dataset": Datasets.NELA_ELECTIONS_2020,
            "train_dataset": Datasets.NELA_ELECTIONS_SUBSET_2020,
        },
        {
            "dataset": Datasets.NELA_COVID_COMBINED,
            "train_dataset": Datasets.NELA_COVID_SUBSET,
        },
    ]
    for config in run_configs:
        dataset = config["dataset"]
        train_dataset = config["train_dataset"]
        print(
            f"Running weak labeler for dataset: {dataset} and train_dataset: {train_dataset}"
        )
        run_weak_labeler(
            dataset,
            train_dataset,
            WeakLabelAggregators.get_active_aggregators(),
        )
