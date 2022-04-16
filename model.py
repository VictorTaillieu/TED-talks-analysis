import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
from scipy.spatial import distance


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
    def boolean_df(item_lists, unique_items):
        bool_dict = {}

        for item in unique_items:
            bool_dict[item] = item_lists.apply(lambda x: item in x)

        return pd.DataFrame(bool_dict)

    @staticmethod
    def cum_dist(dist_matrix, history):
        return np.mean([dist_matrix[i] for i in history], axis=0)

    @staticmethod
    def scale(data):
        scaler = MinMaxScaler()
        reshaped_values = data.reshape(-1, 1)

        return scaler.fit_transform(reshaped_values).T[0]

    def train(self):
        # Descriptions
        desc_embeddings = np.load("data/embeddings/desc_embeddings.npy")
        self.desc_dist = pairwise_distances(desc_embeddings, metric="cosine")
        np.fill_diagonal(self.desc_dist, 1)

        # About speaker
        speak_embeddings = np.load("data/embeddings/speak_embeddings.npy")
        self.speak_dist = pairwise_distances(speak_embeddings, metric="cosine")
        np.fill_diagonal(self.speak_dist, 1)

        # General infos
        gen_infos = np.array([TEDflix.scale(self.df[attr].values) for attr in [
            "views", "comments", "duration", "discussion_rate"
        ]]).T
        self.gen_dist = pairwise_distances(gen_infos, metric="euclidean")
        np.fill_diagonal(self.gen_dist, np.max(self.gen_dist))  # Which value to put between a talk and itself?

        # Related distance
        self.related_dist = np.load("data/embeddings/related_distance.npy")
        np.fill_diagonal(self.related_dist, np.max(self.related_dist))

        top = self.df.topics.apply(lambda x: literal_eval(x))
        one_hot_topics = TEDflix.boolean_df(top, top.explode().unique()).astype(int)

        # Topics distance
        self.topics_dist = distance.squareform(distance.pdist(one_hot_topics, "jaccard"))
        np.fill_diagonal(self.topics_dist, np.max(self.topics_dist))

        self.sent_dist = pairwise_distances(np.array(self.df.sentiment).reshape(-1, 1), metric="l1")

        self.date_dist = pairwise_distances(np.array(self.df.recorded_date).reshape(-1, 1), metric="l1")

        self.trained = True

    def predict(self, history, k=5):
        if not self.trained:
            print("Warning: the model has not been trained.")
            return None

        # Cumulative distances
        desc_cum_dist = TEDflix.cum_dist(self.desc_dist, history)
        speak_cum_dist = TEDflix.cum_dist(self.speak_dist, history)
        gen_cum_dist = TEDflix.cum_dist(self.gen_dist, history)
        related_cum_dist = TEDflix.cum_dist(self.related_dist, history)
        topics_cum_dist = TEDflix.cum_dist(self.topics_dist, history)
        sent_cum_dist = TEDflix.cum_dist(self.sent_dist, history)
        date_cum_dist = TEDflix.cum_dist(self.date_dist, history)

        # Combine distances
        combined_dist_matrix = np.array([
            TEDflix.scale(dist_matrix) for dist_matrix in [
                desc_cum_dist,
                speak_cum_dist,
                gen_cum_dist,
                related_cum_dist,
                sent_cum_dist,
                date_cum_dist,
                topics_cum_dist
            ]
        ])

        global_distance = np.mean(self.weights * combined_dist_matrix, axis=0)

        indices = np.argsort(global_distance)[:k]
        return indices
