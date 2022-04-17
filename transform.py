# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

from ast import literal_eval

import numpy as np
import pandas as pd
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.spatial import distance
# Using https://huggingface.co/sentence-transformers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def compute_embeddings(df):
    bert = SentenceTransformer('bert-base-nli-mean-tokens')

    desc_embeds = bert.encode(df.description)
    speak_embeds = bert.encode(df.about_speaker)

    np.save("../data/desc_embeddings.npy", desc_embeds)
    np.save("../data/distances/speak_embeddings.npy", speak_embeds)


def compute_sentiments(df):
    analyser = SentimentIntensityAnalyzer()

    compound_mean = lambda transcript: np.mean([
        analyser.polarity_scores(sentence)['compound'] for sentence in sent_tokenize(transcript)
    ])

    tqdm.pandas()
    sentiments = df.transcript.progress_apply(compound_mean)

    np.save("../data/distances/sentiments.npy", sentiments)


def boolean_df(item_lists, unique_items):
    bool_dict = {}

    for item in unique_items:
        bool_dict[item] = item_lists.apply(lambda x: item in x)

    return pd.DataFrame(bool_dict)


def compute_topic_distance(df):
    top = df.topics.apply(lambda x: literal_eval(x))
    one_hot_topics = boolean_df(top, top.explode().unique()).astype(int)

    topics_dist = distance.squareform(distance.pdist(one_hot_topics, "jaccard"))

    np.save("../data/distances/topics_dist.npy", topics_dist)


def compute_distances(df):
    compute_embeddings(df)
    compute_sentiments(df)
    compute_topic_distance(df)


def transform_data():
    ted_talks = pd.read_csv("data/ted_talks_preprocessed.csv")
    ted_talks.recorded_date = pd.to_datetime(ted_talks.recorded_date)
    ted_talks.published_date = pd.to_datetime(ted_talks.published_date)

    # Discussion rate
    ted_talks["discussion_rate"] = (ted_talks.comments / ted_talks.views).fillna(0)

    # Dates
    ted_talks["recorded_date"] = ted_talks.recorded_date.apply(lambda date: date.timestamp())
    # Dataset created on May 1st, 2020
    ted_talks["views_by_day"] = ted_talks.apply(lambda elt: elt.views / (pd.to_datetime("2020-05-01") - elt.published_date).days, axis=1)

    # Sentiments
    ted_talks["sentiment"] = np.load("data/distances/sentiments.npy")

    # Countries
    event_country_mapping = pd.read_csv("data/event_country_mapping.csv")
    ted_talks = pd.merge(ted_talks, event_country_mapping, on="event")

    # Drop unwanted attributes and sort
    ted_talks.sort_values(by="talk_id", inplace=True)
    ted_talks.drop(["talk_id", "about_speaker", "published_date", "event", "available_lang", "description", "transcript"], axis=1, inplace=True)

    # Save transformed dataset
    ted_talks.to_csv("data/ted_talks_transformed.csv", index=False)


if __name__ == "__main__":
    transform_data()
