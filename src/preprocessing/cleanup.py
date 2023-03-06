import pandas as pd
from gensim import utils
import gensim.parsing.preprocessing as gsp

GENSIM_FILTERS = [
    gsp.strip_tags,
    gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.remove_stopwords,
    gsp.strip_short,
    gsp.stem_text,
]


def cleanup_df(df: pd.DataFrame):
    copied_df = df.copy(deep=True)
    copied_df = copied_df[["id", "title", "content", "label"]]
    copied_df.dropna(inplace=True)
    # copied_df = copied_df[copied_df["content"].apply(len) < 5000]
    return copied_df.sample(frac=1).reset_index(drop=True)


def separate_real_and_fake(df: pd.DataFrame):
    mask = df["label"] == 0
    real_df = df[mask]
    fake_df = df[~mask]
    print(f"len(real_df): {len(real_df)}, len(fake_df): {len(fake_df)}")
    return real_df, fake_df


def clean_text(s: str) -> str:
    s = s.lower()
    s = utils.to_unicode(s)
    for f in GENSIM_FILTERS:
        s = f(s)
    return s
