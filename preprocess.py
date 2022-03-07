# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

import ast
import sys
from glob import glob
from os.path import exists

import pandas as pd


def evaluate(x):
    if pd.isna(x):
        return x
    return ast.literal_eval(x)


def aggregate_data(save=False):
    data_files = glob("data/raw/*.csv")
    data_files.sort()

    df_list = []

    for file in data_files:
        df = pd.read_csv(file)
        df_list.append(df)

    ted_talks = pd.concat(df_list, ignore_index=True)

    ted_talks.drop_duplicates("talk_id", inplace=True, ignore_index=True)
    ted_talks.sort_values("talk_id", inplace=True, ignore_index=True)

    ted_talks = ted_talks[ted_talks.native_lang == "en"].reset_index(drop=True)
    ted_talks.drop("native_lang", axis=1, inplace=True)

    ted_talks.drop("url", axis=1, inplace=True)
    ted_talks.drop("all_speakers", axis=1, inplace=True)
    ted_talks.drop("available_lang", axis=1, inplace=True)

    if save:
        ted_talks.to_csv("data/ted_talks_agg.csv", index=False)

    return ted_talks


def preprocess_data():
    ted_talks = aggregate_data()

    ted_talks.occupations = ted_talks.occupations.apply(lambda x: evaluate(x)[0] if pd.notnull(x) else ["unknown"])
    ted_talks.about_speakers = ted_talks.about_speakers.apply(lambda x: evaluate(x)[0] if pd.notnull(x) else "unknown")
    ted_talks.recorded_date = ted_talks.apply(lambda x: x.recorded_date if pd.notnull(x.recorded_date) else x.published_date, axis=1)

    rate = (ted_talks.comments / ted_talks.views).median()
    ted_talks.comments = ted_talks.apply(lambda x: int(x.comments) if pd.notnull(x.comments) else int(rate * x.views), axis=1)

    ted_talks.related_talks = ted_talks.related_talks.apply(lambda x: list(evaluate(x).keys()))

    ted_talks.rename(columns={"speaker_1": "speaker", "about_speakers": "about_speaker"}, inplace=True)

    # Save preprocessed dataset
    ted_talks.to_csv("data/ted_talks_prepro.csv", index=False)


def load_data():
    if not exists("data/ted_talks_prepro.csv"):
        preprocess_data()

    df = pd.read_csv("data/ted_talks_prepro.csv")

    df.occupations = df.occupations.apply(evaluate)
    df.topics = df.topics.apply(evaluate)
    df.related_talks = df.related_talks.apply(evaluate)
    df.recorded_date = pd.to_datetime(df.recorded_date)
    df.published_date = pd.to_datetime(df.published_date)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate_data(save=True)
    elif len(sys.argv) == 0:
        preprocess_data()
