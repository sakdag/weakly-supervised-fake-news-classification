"""RoBERTa end-model training utilities (simpletransformers backend)."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.preprocessing.consts import Dataset, Datasets
from src.preprocessing.read_raw_data import balanced_undersample, read_weak_labeled_csv


def prepare_weak_labeled_frame(
    dataset: Dataset,
    train_dataset: Dataset,
    top_n: int = 5,
    weak_label_column: str = "snorkel_majority_vote_label",
    max_rows: Optional[int] = 1000,
) -> pd.DataFrame:
    """Load weak labels, drop abstains, and optionally undersample."""
    df = read_weak_labeled_csv(
        dataset.weak_labeled_path,
        train_df_name=train_dataset.name,
        top_n=top_n,
    )
    df = df[["title", "content", "actual_label", weak_label_column]].copy()
    df = df.drop(df[df[weak_label_column] == -1].index)
    df = df.rename(columns={weak_label_column: "weak_label", "content": "text"})

    print(f"Rows after dropping abstains: {len(df)}")
    print(
        "Weak-label accuracy vs actual:",
        accuracy_score(df["actual_label"], df["weak_label"]),
    )

    under_sampler = RandomUnderSampler(random_state=42)
    undersampled_df, _ = under_sampler.fit_resample(df, df["weak_label"])
    print(undersampled_df["weak_label"].value_counts())

    if max_rows is not None:
        undersampled_df = balanced_undersample(
            undersampled_df, max_rows, "weak_label"
        )

    undersampled_df = undersampled_df.copy()
    undersampled_df["text"] = (
        undersampled_df["title"] + ". " + undersampled_df["text"]
    )
    undersampled_df.drop(columns=["title"], inplace=True)
    return undersampled_df


def split_label_views(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/eval/test frames keyed by weak and actual labels."""
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["weak_label"],
        random_state=42,
    )
    train_df, test_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["weak_label"], random_state=42
    )

    def with_labels(frame: pd.DataFrame, label_col: str, drop_col: str) -> pd.DataFrame:
        out = frame.copy()
        out.rename(columns={label_col: "labels"}, inplace=True)
        out.drop(columns=[drop_col], inplace=True)
        return out

    return (
        with_labels(train_df, "actual_label", "weak_label"),
        with_labels(train_df, "weak_label", "actual_label"),
        with_labels(eval_df, "actual_label", "weak_label"),
        with_labels(eval_df, "weak_label", "actual_label"),
        with_labels(test_df, "actual_label", "weak_label"),
        with_labels(test_df, "weak_label", "actual_label"),
    )


def build_roberta_model(use_cuda: bool = True) -> ClassificationModel:
    model_args = ClassificationArgs(
        num_train_epochs=3,
        overwrite_output_dir=True,
        manual_seed=42,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
    )
    return ClassificationModel(
        model_type="roberta",
        model_name="roberta-base",
        use_cuda=use_cuda,
        num_labels=2,
        args=model_args,
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df_weak: pd.DataFrame,
    test_df_actual: pd.DataFrame,
    use_cuda: bool = True,
) -> ClassificationModel:
    model = build_roberta_model(use_cuda=use_cuda)
    model.train_model(train_df, eval_df=eval_df)

    weak_predictions, _ = model.predict(list(test_df_weak["text"]))
    print("Evaluation against weak labels:")
    print(classification_report(test_df_weak["labels"], weak_predictions, digits=5))

    actual_predictions, _ = model.predict(list(test_df_actual["text"]))
    print("Evaluation against actual labels:")
    print(
        classification_report(test_df_actual["labels"], actual_predictions, digits=5)
    )
    return model


def run_paper_end_model(
    dataset: Dataset = Datasets.NELA_COVID_COMBINED,
    train_dataset: Dataset = Datasets.NELA_ELECTIONS_2020,
    top_n: int = 5,
    use_cuda: bool = True,
    train_on_actual_labels: bool = True,
):
    """Reproduce the RoBERTa end-model experiment used in the paper pipeline."""
    df = prepare_weak_labeled_frame(dataset, train_dataset, top_n=top_n)
    (
        train_actual,
        train_weak,
        eval_actual,
        eval_weak,
        test_actual,
        test_weak,
    ) = split_label_views(df)

    print(len(train_actual), len(eval_actual), len(test_actual))

    if train_on_actual_labels:
        return train_and_evaluate(
            train_actual, eval_actual, test_weak, test_actual, use_cuda=use_cuda
        )
    return train_and_evaluate(
        train_weak, eval_weak, test_weak, test_actual, use_cuda=use_cuda
    )


if __name__ == "__main__":
    run_paper_end_model()
