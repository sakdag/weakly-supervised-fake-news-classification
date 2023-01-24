import os
import re

import pandas
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from src.preprocessing.cleanup import cleanup_df
from src.preprocessing.consts import Datasets, DATA_FOLDER_PREFIX
import sqlite3
import pandas as pd

from src.utils.pandas_utils import is_string_series
from src.utils.path_utils import get_project_root


def read_raw_data(
    dataset: Datasets = Datasets.NELA_ELECTIONS_2020_DATASET,
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


def sample_and_save(dataset: Datasets = Datasets.NELA_ELECTIONS_2020_DATASET):
    df = read_raw_data(dataset)
    if dataset == Datasets.NELA_ELECTIONS_2020_DATASET:
        sampled_dataset = Datasets.NELA_ELECTIONS_2020_DATASET_SAMPLED
    elif dataset == Datasets.NELA_CAPITOL_2021_DATASET:
        sampled_dataset = Datasets.NELA_CAPITOL_2021_DATASET_SAMPLED
    elif dataset == Datasets.NELA_COVID_2020_DATASET:
        sampled_dataset = Datasets.NELA_COVID_2020_DATASET_SAMPLED
    elif dataset == Datasets.NELA_COVID_2021_DATASET:
        sampled_dataset = Datasets.NELA_COVID_2021_DATASET_SAMPLED
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
    print(df.dtypes)

    undersample = RandomUnderSampler(sampling_strategy={0: 10000, 1: 10000})
    df, labels = undersample.fit_resample(df, df.label)
    print(len(df))
    print(df.label.value_counts())

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        val = row["content"].encode("ascii", errors="ignore").decode()
        val = val.replace("@ @ @ @ @ @ @", "")
        val = val.replace("@ @ @ @ @", "")
        val = re.sub(' +', ' ', val)
        df.at[i, 'content'] = val
        df.at[i, 'title'] = row["title"].encode("ascii", errors="ignore").decode()

    print(f"Saving dataset to path: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)

    # Fixing string bug, read and write again
    df = read_raw_data(dataset=sampled_dataset)
    df.dropna(inplace=True)
    print(len(df))

    # undersample = RandomUnderSampler(sampling_strategy="majority")
    # undersample = RandomUnderSampler(sampling_strategy={0: 5000, 1: 5000})
    # df, labels = undersample.fit_resample(df, df.label)
    # print(len(df))
    # print(df.label.value_counts())

    # Make max elements in each label type 25k
    # n_max = 25000
    # undersampled_df = merged_df.groupby("label").apply(
    #     lambda x: x.sample(n=min(n_max, len(x)))
    # )
    # print(undersampled_df["label"].value_counts())

    print(f"Saving dataset to path: {dataset_path}")
    df = df.sample(frac=1)
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


def balanced_undersample(df: pd.DataFrame, n: int):
    undersample = RandomUnderSampler(sampling_strategy={0: int(n/2), 1: int(n/2)})
    undersampled_df, labels = undersample.fit_resample(df, df.label)
    print(f"Dataframe size: {len(undersampled_df)}")
    print(undersampled_df.label.value_counts())
    return undersampled_df


def drop_na_and_save(dataset: Datasets):
    df = read_raw_data(dataset)

    dataset_path = os.path.join(
        get_project_root(), DATA_FOLDER_PREFIX, dataset.value
    )

    df.dropna(inplace=True)
    print(len(df))

    print(f"Saving dataset to path: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


if __name__ == "__main__":
    # df = read_raw_data(dataset=Datasets.NELA_COVID_DATASET_SAMPLED)
    # print(len(df))
    # print(is_string_series(df.title))
    # print(is_string_series(df.content))
    # print(df.dtypes)

    # drop_na_and_save(dataset=Datasets.NELA_COVID_2021_DATASET_SAMPLED)

    sample_and_save(dataset=Datasets.NELA_COVID_2021_DATASET)
    sample_and_save(dataset=Datasets.NELA_COVID_2020_DATASET)
    sample_and_save(dataset=Datasets.NELA_ELECTIONS_2020_DATASET)
    sample_and_save(dataset=Datasets.NELA_CAPITOL_2021_DATASET)
