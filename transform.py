# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

import numpy as np
import pandas as pd


def transform_data():
    ted_talks = pd.read_csv("data/ted_talks_preprocessed.csv")
    ted_talks.recorded_date = pd.to_datetime(ted_talks.recorded_date)
    ted_talks.published_date = pd.to_datetime(ted_talks.published_date)

    # Discussion rate
    ted_talks["discussion_rate"] = ted_talks.comments / ted_talks.views

    # Dates
    ted_talks["recorded_date"] = ted_talks.recorded_date.apply(lambda date: date.timestamp())
    # Dataset created on May 1st, 2020
    ted_talks["views_by_day"] = ted_talks.apply(lambda elt: elt.views / (pd.to_datetime("2020-05-01") - elt.published_date).days, axis=1)

    # Sentiments
    ted_talks["sentiment"] = np.load("data/embeddings/sentiments.npy")

    # Countries
    event_country_mapping = pd.read_csv("data/event_country_mapping.csv")
    ted_talks = pd.merge(ted_talks, event_country_mapping, on="event")

    # Drop unwanted attributes and sort
    ted_talks.drop(["about_speaker", "published_date", "event", "available_lang", "description", "transcript"], axis=1, inplace=True)
    ted_talks.sort_values(by="talk_id", inplace=True)

    # Save transformed dataset
    ted_talks.to_csv("data/ted_talks_transformed.csv", index=False)


if __name__ == "__main__":
    transform_data()
