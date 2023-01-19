# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

import ast
import sys
from os.path import exists

import pandas as pd


def evaluate(x):
    if pd.isna(x):
        return x
    return ast.literal_eval(x)


def select_data(save=False):
    ted_talks = pd.read_csv("data/ted_talks_en.csv")

    ted_talks.sort_values("talk_id", inplace=True, ignore_index=True)

    ted_talks = ted_talks[ted_talks.native_lang == "en"].reset_index(drop=True)

    ted_talks[[
        "title",
        "speaker_1",
        "published_date",
        "available_lang",
        "duration",
        "url"
    ]].to_csv("data/ted_talks_infos.csv", index=False)

    ted_talks.drop("native_lang", axis=1, inplace=True)
    ted_talks.drop("url", axis=1, inplace=True)
    ted_talks.drop("all_speakers", axis=1, inplace=True)
    ted_talks.drop("available_lang", axis=1, inplace=True)

    if save:
        ted_talks.to_csv("data/ted_talks_selected.csv", index=False)

    return ted_talks


def preprocess_data():
    ted_talks = select_data()

    ted_talks.occupations = ted_talks.occupations.apply(lambda x: evaluate(x)[0] if pd.notnull(x) else ["unknown"])
    ted_talks.about_speakers = ted_talks.about_speakers.apply(lambda x: evaluate(x)[0] if pd.notnull(x) else "unknown")
    ted_talks.recorded_date.fillna(ted_talks.published_date, inplace=True)

    rate = (ted_talks.comments / ted_talks.views).median()
    ted_talks.comments = ted_talks.comments.fillna(rate * ted_talks.views).astype(int)

    ted_talks.related_talks = ted_talks.related_talks.apply(lambda x: list(evaluate(x).keys()))

    ted_talks.rename(columns={"speaker_1": "speaker", "about_speakers": "about_speaker"}, inplace=True)

    # Save preprocessed dataset
    ted_talks.to_csv("data/ted_talks_preprocessed.csv", index=False)


def load_data():
    if not exists("data/ted_talks_preprocessed.csv"):
        preprocess_data()

    df = pd.read_csv("data/ted_talks_preprocessed.csv")

    df.occupations = df.occupations.apply(evaluate)
    df.available_lang = df.available_lang.apply(evaluate)
    df.topics = df.topics.apply(evaluate)
    df.related_talks = df.related_talks.apply(evaluate)

    df.recorded_date = pd.to_datetime(df.recorded_date)
    df.published_date = pd.to_datetime(df.published_date)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        select_data(save=True)
    elif len(sys.argv) == 1:
        preprocess_data()
