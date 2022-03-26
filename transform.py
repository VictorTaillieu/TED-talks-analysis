# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

import pandas as pd


def transform_data():
    ted_talks = pd.read_csv("data/ted_talks_preprocessed.csv")

    ted_talks["discussion_rate"] = ted_talks.comments / ted_talks.views

    event_country_mapping = pd.read_csv("data/event_country_mapping.csv")

    ted_talks = pd.merge(ted_talks, event_country_mapping, on="event")
    ted_talks.drop("event", axis=1, inplace=True)

    # Save transformed dataset
    ted_talks.to_csv("data/ted_talks_transformed.csv", index=False)


if __name__ == "__main__":
    transform_data()
