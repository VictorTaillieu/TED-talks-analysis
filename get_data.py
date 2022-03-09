# Authors:
# Maxime Lafont--Trevisan (lafm2724)
# Gaëtan Lounes (loug2904)
# Victor Taillieu (taiv2701)
# Luca Vaio (vail3202)
# Reference: https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/

from os import remove
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_file("miguelcorraljr/ted-ultimate-dataset",
                          "2020-05-01/ted_talks_en.csv",
                          path="data", quiet=False)

with ZipFile("data/ted_talks_en.csv.zip") as z:
    z.extractall(path="data")

remove("data/ted_talks_en.csv.zip")
