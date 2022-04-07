import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


class TEDflix:
    def __init__(self, weights=None):
        self.trained = False

        self.df = pd.read_csv("data/ted_talks_transformed.csv")
        
        if weights is None:
            self.weights = np.array([
                1,  # description
                1,  # about_speaker
                1,  # general_infos
                1,  # topics
                1,  # sentiment
                1   # dates
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

        # Topic distance
        self.topic_dist = np.load("data/embeddings/related_distance.npy")
        np.fill_diagonal(self.topic_dist, np.max(self.topic_dist))

        self.trained = True
    

    def predict(self, history, k=5):
        if not self.trained:
            print("Warning: the model has not been trained.")
            return None

        # Cumulative distances
        desc_cum_dist = TEDflix.cum_dist(self.desc_dist, history)
        speak_cum_dist = TEDflix.cum_dist(self.speak_dist, history)
        gen_cum_dist = TEDflix.cum_dist(self.gen_dist, history)
        topic_cum_dist = TEDflix.cum_dist(self.topic_dist, history)
        sent_cum_dist = np.mean([abs(self.df.sentiment[i] - self.df.sentiment) for i in history], axis=0)
        date_cum_dist = np.mean([abs(self.df.recorded_date[i] - self.df.recorded_date) for i in history], axis=0)

        # Combine distances
        combined_dist_matrix = np.array([
            TEDflix.scale(dist_matrix) for dist_matrix in [
                desc_cum_dist,
                speak_cum_dist,
                gen_cum_dist,
                topic_cum_dist,
                sent_cum_dist,
                date_cum_dist
            ]
        ])

        global_distance = np.mean(self.weights * combined_dist_matrix, axis=0)

        indices = np.argsort(global_distance)[:k]
        return self.df.title[indices]

if __name__ == "__main__":
    model = TEDflix()
    model.train()
    
    predictions = model.predict([1888, 3925])
    print(predictions)
