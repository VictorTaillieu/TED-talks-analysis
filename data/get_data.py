import os
from json import load as json_load

with open("kaggle.json", 'r') as kaggle_credentials:
    kaggle_cred = json_load(kaggle_credentials)

os.environ["KAGGLE_USERNAME"] = kaggle_cred["username"]
os.environ["KAGGLE_KEY"] = kaggle_cred["key"]

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files("miguelcorraljr/ted-ultimate-dataset", quiet=False, unzip=True)

os.rename("2020-05-01", "raw")
