from os import rename
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files("miguelcorraljr/ted-ultimate-dataset", quiet=False, unzip=True)

rename("2020-05-01", "raw")
