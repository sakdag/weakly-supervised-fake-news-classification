import os

import pandas

from src.read_data.consts import RawDatasets, RAW_DATA_FOLDER_PREFIX
import argparse
import sqlite3
import pandas as pd

from src.utils.path_utils import get_project_root


def read_raw_data(
    dataset: RawDatasets = RawDatasets.ELECTIONS_DATASET,
) -> pandas.DataFrame:
    dataset_path = os.path.join(
        get_project_root(), RAW_DATA_FOLDER_PREFIX, dataset.value
    )
    print(f"Reading dataset from path: {dataset_path}")
    if dataset.value.endswith("csv"):
        return pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    else:
        conn = sqlite3.connect(dataset_path)
        query = "SELECT * FROM newsdata"
        return pd.read_sql_query(query, conn)


def sample_and_save(dataset: RawDatasets = RawDatasets.ELECTIONS_DATASET):
    df = read_raw_data(dataset)
    if dataset == RawDatasets.ELECTIONS_DATASET:
        dataset_file_name = RawDatasets.ELECTIONS_DATASET_SAMPLED.value
    elif dataset == RawDatasets.COVID_DATASET:
        dataset_file_name = RawDatasets.COVID_DATASET_SAMPLED.value
    else:
        print(f"Cant sample dataset: {dataset}")
        return

    dataset_path = os.path.join(
        get_project_root(), RAW_DATA_FOLDER_PREFIX, dataset_file_name
    )
    print(dataset_path)

    labels_df = read_raw_data(RawDatasets.SOURCE_LABELS)
    print(labels_df)
    labels_df = labels_df[labels_df.label != 2]
    print(labels_df)
    sources = labels_df.source.unique()
    print(sources)

    print(df)
    selected_df = df[df.source.isin(sources)]
    print(selected_df)
    selected_df.reindex()
    print(selected_df)

    for idx, row in selected_df.iterrows():
        print(row)
        break

    # sampled_df = selected_df.sample(n=50000)
    # print(sampled_df)

    merged_df = pd.merge(selected_df, labels_df, on="source", how="outer")
    print(merged_df.label.value_counts())

    # Make max elements in each label type 25k
    n_max = 25000
    undersampled_df = merged_df.groupby("label").apply(
        lambda x: x.sample(n=min(n_max, len(x)))
    )
    print(undersampled_df["label"].value_counts())

    undersampled_df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


if __name__ == "__main__":
    # df = read_raw_data(dataset=RawDatasets.ELECTIONS_DATASET)
    # print(df)
    sample_and_save(dataset=RawDatasets.COVID_DATASET)
