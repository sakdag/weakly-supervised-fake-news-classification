from statistics import mean

import pandas as pd
from tqdm import tqdm
from snorkel.labeling import labeling_function

from src.consts.labels import Labels
from src.old_trials.preprocessors import spacy_content, NLP
from src.preprocessing.consts import RawDatasets
from src.preprocessing.read_raw_data import read_raw_data
from src.preprocessing.preprocessing_utils import cleanup_df, separate_real_and_fake

ADVERB_THRESHOLD = 0.75


@labeling_function(pre=[spacy_content])
def lf_adverb_frequency(x):
    count = 0
    for token in x.spacy_content:
        if token.pos_ == "ADV":
            count += 1
    adverb_per_sentence = count / sum(1 for dummy in x.spacy_content.sents)
    return Labels.FAKE if adverb_per_sentence > ADVERB_THRESHOLD else Labels.REAL


def get_average_adverb_frequency(df: pd.DataFrame):
    real_df, fake_df = separate_real_and_fake(df)

    real_adv_freq = get_average_adverb_frequency_helper(real_df)
    print(f"Real adverb frequency: {real_adv_freq}")

    fake_adv_freq = get_average_adverb_frequency_helper(fake_df)
    print(f"Fake adverb frequency: {fake_adv_freq}")


def get_average_adverb_frequency_helper(df: pd.DataFrame):
    adverb_per_sentence_list = []
    for idx, row in tqdm(df.iterrows()):
        count = 0
        doc = NLP(row.content)
        for token in doc:
            if token.pos_ == "ADV":
                count += 1
        adverb_per_sentence_list.append((count / sum(1 for dummy in doc.sents)))
    return mean(adverb_per_sentence_list)


if __name__ == "__main__":
    covid_df: pd.DataFrame = read_raw_data(RawDatasets.COVID_DATASET_SAMPLED)
    covid_df_cleaned = cleanup_df(covid_df)
    get_average_adverb_frequency(covid_df_cleaned)
