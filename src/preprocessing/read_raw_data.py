import os

import pandas
from imblearn.under_sampling import RandomUnderSampler

from src.preprocessing.cleanup import cleanup_df
from src.preprocessing.consts import Datasets, DATA_FOLDER_PREFIX
import sqlite3
import pandas as pd

from src.utils.pandas_utils import is_string_series
from src.utils.path_utils import get_project_root


def read_raw_data(
    dataset: Datasets = Datasets.NELA_ELECTIONS_DATASET,
) -> pandas.DataFrame:
    dataset_path = os.path.join(get_project_root(), DATA_FOLDER_PREFIX, dataset.value)
    print(f"Reading dataset from path: {dataset_path}")
    if dataset.value.endswith("csv"):
        df = pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    else:
        conn = sqlite3.connect(dataset_path)
        query = "SELECT * FROM newsdata"
        df = pd.read_sql_query(query, conn)
    print(f"Dataframe size: {len(df)}")
    return df


def save_to_csv(df: pd.DataFrame, dataset: Datasets):
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        dataset.value,
    )
    print(f"Saving dataframe to: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


def sample_and_save(dataset: Datasets = Datasets.NELA_ELECTIONS_DATASET):
    df = read_raw_data(dataset)
    if dataset == Datasets.NELA_ELECTIONS_DATASET:
        sampled_dataset = Datasets.NELA_ELECTIONS_DATASET_SAMPLED
    elif dataset == Datasets.NELA_COVID_DATASET:
        sampled_dataset = Datasets.NELA_COVID_DATASET_SAMPLED
    else:
        print(f"Cant sample dataset: {dataset}")
        return

    dataset_path = os.path.join(
        get_project_root(), DATA_FOLDER_PREFIX, sampled_dataset.value
    )
    print(dataset_path)

    labels_df = read_raw_data(Datasets.NELA_SOURCE_LABELS)
    # print(labels_df)
    labels_df = labels_df[labels_df.label != 2]
    # print(labels_df)
    sources = labels_df.source.unique()
    # print(sources)

    print(len(df))
    # print(df)
    df = df[df.source.isin(sources)]
    # print(selected_df)
    df.reindex()
    # print(selected_df)
    print(len(df))

    # sampled_df = selected_df.sample(n=50000)
    # print(sampled_df)

    df = pd.merge(df, labels_df, on="source", how="outer")
    print(df.label.value_counts())

    df = cleanup_df(df)
    print(df.label.value_counts())

    print(is_string_series(df.title))
    print(is_string_series(df.content))
    # Type changes
    print(df.dtypes)

    print(f"Saving dataset to path: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)

    # Fixing string bug, read and write again
    df = read_raw_data(dataset=sampled_dataset)
    df.dropna(inplace=True)
    print(len(df))

    # undersample = RandomUnderSampler(sampling_strategy="majority")
    undersample = RandomUnderSampler(sampling_strategy={0: 500, 1: 500})
    df, labels = undersample.fit_resample(df, df.label)
    print(len(df))
    print(df.label.value_counts())

    # Make max elements in each label type 25k
    # n_max = 25000
    # undersampled_df = merged_df.groupby("label").apply(
    #     lambda x: x.sample(n=min(n_max, len(x)))
    # )
    # print(undersampled_df["label"].value_counts())

    print(f"Saving dataset to path: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


if __name__ == "__main__":
    # df = read_raw_data(dataset=Datasets.NELA_COVID_DATASET_SAMPLED)
    # print(len(df))
    # print(is_string_series(df.title))
    # print(is_string_series(df.content))
    # print(df.dtypes)

    sample_and_save(dataset=Datasets.NELA_COVID_DATASET)
