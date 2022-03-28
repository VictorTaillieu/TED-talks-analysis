import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


def cum_dist(dist_matrix, history):
    return np.mean([dist_matrix[i] for i in history], axis=0)


def scale(data):
    scaler = MinMaxScaler()
    reshaped_values = data.reshape(-1, 1)

    return scaler.fit_transform(reshaped_values).T[0]


def predict(history, k):
    df = pd.read_csv("data/ted_talks_transformed.csv")

    # Descriptions
    desc_embeddings = np.load("data/embeddings/desc_embeddings.npy")

    desc_dist = pairwise_distances(desc_embeddings, metric="cosine")
    np.fill_diagonal(desc_dist, 1)
    desc_cum_dist = cum_dist(desc_dist, history)

    # About speaker
    speak_embeddings = np.load("data/embeddings/speak_embeddings.npy")

    speak_dist = pairwise_distances(speak_embeddings, metric="cosine")
    np.fill_diagonal(speak_dist, 1)
    speak_cum_dist = cum_dist(speak_dist, history)

    # General infos
    gen_infos = np.array([
        scale(df.views.values),
        scale(df.comments.values),
        scale(df.duration.values),
        scale(df.discussion_rate.values)
    ]).T

    gen_dist = pairwise_distances(gen_infos, metric="euclidean")
    # Which value to put between a talk and itself?
    np.fill_diagonal(gen_dist, np.max(gen_dist))
    gen_cum_dist = cum_dist(gen_dist, history)

    # Topic distance
    topic_dist = np.load("data/embeddings/related_distance.npy")
    np.fill_diagonal(topic_dist, np.max(topic_dist))
    topic_cum_dist = cum_dist(topic_dist, history)

    # Sentiments
    sent_cum_dist = np.mean([abs(df.sentiment[i] - df.sentiment) for i in history], axis=0)

    # Dates
    date_cum_dist = np.mean([abs(df.recorded_date[i] - df.recorded_date) for i in history], axis=0)

    # Combine distances
    combined_dist_matrix = np.array([
        scale(dist_matrix) for dist_matrix in [
            desc_cum_dist,
            speak_cum_dist,
            gen_cum_dist,
            topic_cum_dist,
            sent_cum_dist,
            date_cum_dist
        ]
    ])

    weights = np.array([
        1,  # description
        1,  # about_speaker
        1,  # general_infos
        1,  # topics
        1,  # sentiment
        1   # dates
    ]).reshape(-1, 1)

    global_distance = np.mean(weights * combined_dist_matrix, axis=0)

    indices = np.argsort(global_distance)[:k]
    return df.title[indices]


print(predict([1888, 3925], 5))
