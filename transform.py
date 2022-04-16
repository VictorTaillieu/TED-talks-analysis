# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)

import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Using https://huggingface.co/sentence-transformers
from sentence_transformers import SentenceTransformer


def compute_embeddings():
    bert = SentenceTransformer('bert-base-nli-mean-tokens')
    
    desc_embeds = bert.encode(df_prepro.description)
    speak_embeds = bert.encode(df_prepro.about_speaker)
    
    np.save("../data/desc_embeddings.npy", desc_embeds)
    np.save("../data/embeddings/speak_embeddings.npy", speak_embeds)


def compute_sentiments():
    analyser = SentimentIntensityAnalyzer()

    compound_mean = lambda transcript: np.mean([
        analyser.polarity_scores(sentence)['compound'] for sentence in sent_tokenize(transcript)
    ])
    
    tqdm.pandas()
    sentiments = df_prepro.transcript.progress_apply(compound_mean)
    
    np.save("../data/embeddings/sentiments.npy", sentiments)


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
    ted_talks["sentiment"] = np.load("data/embeddings/sentiments.npy")

    # Countries
    event_country_mapping = pd.read_csv("data/event_country_mapping.csv")
    ted_talks = pd.merge(ted_talks, event_country_mapping, on="event")

    # Drop unwanted attributes and sort
    ted_talks.sort_values(by="talk_id", inplace=True)
    ted_talks.drop(["talk_id", "about_speaker", "published_date", "event", "available_lang", "description", "transcript"], axis=1, inplace=True)

    # Save transformed dataset
    ted_talks.to_csv("data/ted_talks_transformed.csv", index=False)


if __name__ == "__main__":
    # compute_embeddings()
    # compute_sentiments()
    transform_data()
