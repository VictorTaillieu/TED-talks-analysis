# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)
# Reference: https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/

from os import rename

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files("miguelcorraljr/ted-ultimate-dataset", path="data", quiet=False, unzip=True)

rename("data/2020-05-01", "data/raw")
