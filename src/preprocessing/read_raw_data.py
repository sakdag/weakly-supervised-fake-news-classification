import os
import re
import time

import pandas
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from src.preprocessing.cleanup import cleanup_df
from src.preprocessing.consts import (
    Datasets,
    DATA_FOLDER_PREFIX,
    Dataset,
    NELA_SOURCE_LABELS,
)
import sqlite3
import pandas as pd

from src.utils.pandas_utils import is_string_series
from src.utils.path_utils import get_project_root


def read_raw_data(
    dataset: Dataset = Datasets.NELA_ELECTIONS_2020,
) -> pandas.DataFrame:
    dataset_path = os.path.join(
        get_project_root(), DATA_FOLDER_PREFIX, dataset.raw_path
    )
    print(f"Reading raw dataset from path: {dataset_path}")
    if dataset_path.endswith("csv"):
        df = pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    else:
        conn = sqlite3.connect(dataset_path)
        query = "SELECT * FROM newsdata"
        df = pd.read_sql_query(query, conn)
    print(f"Read dataset, dataframe size: {len(df)}")
    return df


def read_csv(
    path: str,
) -> pandas.DataFrame:
    dataset_path = os.path.join(get_project_root(), DATA_FOLDER_PREFIX, f"{path}.csv")
    print(f"Reading dataset from path: {dataset_path}")
    df = pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    print(f"Read dataset, dataframe size: {len(df)}")
    return df


def save_to_csv(df: pd.DataFrame, path: str):
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        f"{path}.csv",
    )
    print(f"Saving dataframe to: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


def read_model_feature_injected_csv(
    path: str,
    train_df_name: str,
) -> pandas.DataFrame:
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        f"{path}-injected-with-{train_df_name}.csv",
    )
    print(f"Reading dataset from path: {dataset_path}")
    df = pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    print(f"Read dataset, dataframe size: {len(df)}")
    return df


def save_model_feature_injected_csv(df: pd.DataFrame, path: str, train_df_name: str):
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        f"{path}-injected-with-{train_df_name}.csv",
    )
    print(f"Saving dataframe to: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


def read_weak_labeled_csv(
    path: str,
    train_df_name: str,
) -> pandas.DataFrame:
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        f"{path}-labeled-with-{train_df_name}.csv",
    )
    print(f"Reading dataset from path: {dataset_path}")
    df = pd.read_csv(dataset_path, delimiter=",", encoding="utf-8")
    print(f"Read dataset, dataframe size: {len(df)}")
    return df


def save_weak_labeled_csv(df: pd.DataFrame, path: str, train_df_name: str):
    dataset_path = os.path.join(
        get_project_root(),
        DATA_FOLDER_PREFIX,
        f"{path}-labeled-with-{train_df_name}.csv",
    )
    print(f"Saving dataframe to: {dataset_path}")
    df.to_csv(dataset_path, sep=",", encoding="utf-8", index=False)


def sample_and_save(dataset: Dataset = Datasets.NELA_ELECTIONS_2020):
    df = read_raw_data(dataset)

    dataset_path = os.path.join(
        get_project_root(), DATA_FOLDER_PREFIX, dataset.sampled_path
    )
    print(dataset_path)

    nela_path = os.path.join(get_project_root(), DATA_FOLDER_PREFIX, NELA_SOURCE_LABELS)
    labels_df = read_csv(nela_path)
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

    # undersample = RandomUnderSampler(sampling_strategy={0: 10000, 1: 10000})
    # df, labels = undersample.fit_resample(df, df.label)
    # print(len(df))
    print(df.label.value_counts())

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        val = row["content"].encode("ascii", errors="ignore").decode()
        val = val.replace("@ @ @ @ @ @ @", "")
        val = val.replace("@ @ @ @ @", "")
        val = re.sub(" +", " ", val)
        df.at[i, "content"] = val
        df.at[i, "title"] = row["title"].encode("ascii", errors="ignore").decode()

    print(f"Saving dataset to path: {dataset_path}")
    save_to_csv(df, dataset_path)

    # Fixing string bug, read and write again
    df = read_csv(dataset_path)
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

    df = df[df["content"].str.split().str.len().gt(150)]
    print(len(df))

    print(f"Saving dataset to path: {dataset_path}")
    df = df.sample(frac=1)
    save_to_csv(df, dataset_path)


def balanced_undersample(df: pd.DataFrame, n: int, label_column: str = "label"):
    df = df.sample(frac=1).reset_index(drop=True)
    min_label_count = int(n / 2)
    label_value_counts = df[label_column].value_counts()
    if label_value_counts[0] < min_label_count:
        min_label_count = label_value_counts[0]
    if label_value_counts[1] < min_label_count:
        min_label_count = label_value_counts[1]
    undersample = RandomUnderSampler(
        sampling_strategy={0: min_label_count, 1: min_label_count}
    )
    undersampled_df, labels = undersample.fit_resample(df, df[label_column])
    print(f"Dataframe size: {len(undersampled_df)}")
    print(undersampled_df[label_column].value_counts())
    return undersampled_df


def drop_na_and_save(path: str):
    df = read_csv(path)
    df.dropna(inplace=True)
    print(len(df))
    save_to_csv(df, path)


def drop_less_than_100_words_and_save(path: str):
    df = read_csv(path)
    df = df[df["content"].str.split().str.len().gt(150)]
    print(len(df))
    save_to_csv(df, path)


def save_subset_of_dataset(
    source_dataset: Dataset, dest_dataset: Dataset, n: int = 2000
):
    source_df = read_csv(source_dataset.injected_path)
    subset_df = balanced_undersample(source_df, n)
    save_to_csv(subset_df, dest_dataset.injected_path)
    subset_df_new = subset_df[["id", "content", "title", "label"]].copy()
    subset_df_new.sample(frac=1).reset_index(drop=True)
    save_to_csv(subset_df_new, dest_dataset.sampled_path)


def combine_and_save_datasets(
    dataset_1: Dataset, dataset_2: Dataset, dest_dataset: Dataset
):
    df_1 = read_csv(dataset_1.injected_path)
    df_2 = read_csv(dataset_2.injected_path)
    combined_df = pd.concat([df_1, df_2], axis=0)
    save_to_csv(combined_df, dest_dataset.injected_path)

    combined_df_new = combined_df[["id", "content", "title", "label"]].copy()
    combined_df_new.sample(frac=1).reset_index(drop=True)
    save_to_csv(combined_df_new, dest_dataset.sampled_path)


if __name__ == "__main__":
    # df = read_csv(path=Datasets.NELA_COVID_2020.sampled_path)
    # print(len(df))
    # print(is_string_series(df.title))
    # print(is_string_series(df.content))
    # print(df.dtypes)

    # drop_na_and_save(path=Datasets.NELA_COVID_2021.sampled_path)

    # sample_and_save(dataset=Datasets.NELA_CAPITOL_2021)
    # sample_and_save(dataset=Datasets.NELA_COVID_2021)
    # sample_and_save(dataset=Datasets.NELA_COVID_2020)
    # sample_and_save(dataset=Datasets.NELA_ELECTIONS_2020)

    # save_subset_of_dataset(
    #     source_dataset=Datasets.NELA_ELECTIONS_2020,
    #     dest_dataset=Datasets.NELA_ELECTIONS_SUBSET_2020,
    #     n=2000,
    # )

    combine_and_save_datasets(
        dataset_1=Datasets.NELA_COVID_2020,
        dataset_2=Datasets.NELA_COVID_2021,
        dest_dataset=Datasets.NELA_COVID_COMBINED,
    )

    # drop_less_than_100_words_and_save(path=Datasets.NELA_ELECTIONS_2020.injected_path)
