from statistics import mean

import pandas as pd
from tqdm import tqdm
from snorkel.labeling import labeling_function

from src.consts.labels import Labels
from src.old_trials.preprocessors import spacy_content, NLP
from src.preprocessing.consts import RawDatasets
from src.preprocessing.read_raw_data import read_raw_data
from src.preprocessing.preprocessing_utils import cleanup_df, separate_real_and_fake

WORD_PER_SENTENCE_THRESHOLD = 24.1


@labeling_function(pre=[spacy_content])
def lf_word_per_sentence(x):
    wps = len(x.content.split()) / sum(1 for dummy in x.spacy_content.sents)
    return Labels.FAKE if wps > WORD_PER_SENTENCE_THRESHOLD else Labels.REAL


def get_average_word_per_sentence(df: pd.DataFrame):
    real_df, fake_df = separate_real_and_fake(df)

    real_average_wps = get_average_wps_helper(real_df)
    print(f"Real average wps: {real_average_wps}")

    fake_average_wps = get_average_wps_helper(fake_df)
    print(f"Fake average wps: {fake_average_wps}")


def get_average_wps_helper(df: pd.DataFrame):
    wps_list = []
    for idx, row in tqdm(df.iterrows()):
        wps_list.append(
            len(row.content.split()) / sum(1 for dummy in NLP(row.content).sents)
        )
    return mean(wps_list)


if __name__ == "__main__":
    covid_df: pd.DataFrame = read_raw_data(RawDatasets.COVID_DATASET_SAMPLED)
    covid_df_cleaned = cleanup_df(covid_df)
    get_average_word_per_sentence(covid_df_cleaned)
