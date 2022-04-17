import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval

from transform import compute_distances


class TEDflix:
    def __init__(self, weights=None):
        self.trained = False

        self.df = pd.read_csv("data/ted_talks_transformed.csv")

        if weights is None:
            self.weights = np.array([
                1,  # description
                1,  # about_speaker
                1,  # general_infos
                1,  # related dist
                1,  # sentiment
                1,  # dates
                1   # topics dist
            ]).reshape(-1, 1)
        else:
            self.weights = weights

    @staticmethod
    def cum_dist(dist_matrix, history):
        return np.mean([dist_matrix[i] for i in history], axis=0)

    @staticmethod
    def scale(data):
        scaler = MinMaxScaler()
        reshaped_values = data.reshape(-1, 1)

        return scaler.fit_transform(reshaped_values).T[0]

    def train(self, recompute=False):
        if recompute:
            compute_distances(self.df)
        
        # Descriptions
        desc_embeddings = np.load("data/distances/desc_embeddings.npy")
        self.desc_dist = pairwise_distances(desc_embeddings, metric="cosine")
        np.fill_diagonal(self.desc_dist, 1)

        # About speaker
        speak_embeddings = np.load("data/distances/speak_embeddings.npy")
        self.speak_dist = pairwise_distances(speak_embeddings, metric="cosine")
        np.fill_diagonal(self.speak_dist, 1)

        # General infos
        gen_infos = np.array([TEDflix.scale(self.df[attr].values) for attr in [
            "views", "comments", "duration", "discussion_rate"
        ]]).T
        self.gen_dist = pairwise_distances(gen_infos, metric="euclidean")
        np.fill_diagonal(self.gen_dist, np.max(self.gen_dist))  # Which value to put between a talk and itself?

        # Related distance
        self.related_dist = np.load("data/distances/related_distance.npy")
        np.fill_diagonal(self.related_dist, np.max(self.related_dist))

        # Topics distance
        self.topics_dist = np.load("data/distances/topics_dist.npy")
        np.fill_diagonal(self.topics_dist, np.max(self.topics_dist))

        # Sentiment distance
        self.sent_dist = pairwise_distances(np.array(self.df.sentiment).reshape(-1, 1), metric="l1")
        np.fill_diagonal(self.sent_dist, np.max(self.sent_dist))

        # Date distance
        self.date_dist = pairwise_distances(np.array(self.df.recorded_date).reshape(-1, 1), metric="l1")
        np.fill_diagonal(self.date_dist, np.max(self.date_dist))

        self.trained = True

    def predict(self, history, k=5):
        if not self.trained:
            print("Warning: the model has not been trained.")
            return None

        # Combine distances
        combined_dist_matrix = np.array([
            TEDflix.scale(TEDflix.cum_dist(dist_matrix, history)) for dist_matrix in [
                self.desc_dist,
                self.speak_dist,
                self.gen_dist,
                self.related_dist,
                self.topics_dist,
                self.sent_dist,
                self.date_dist
            ]
        ])

        global_distance = np.mean(self.weights * combined_dist_matrix, axis=0)

        indices = np.argsort(global_distance)[:k]
        return indices
