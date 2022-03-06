# TED talks analysis
IFT870 Project - TED talks analysis

## Prerequisites
To install the requirements, you can use:
```
$ pip install -r requirements.txt
```

## Get raw data
- Get a Kaggle API key at `https://www.kaggle.com/<user_name>/account`
- Put the `kaggle.json` file in `~/.kaggle` folder
- Then run `get_data.py` script:
```
$ python get_data.py
```
This will create `raw` folder containing raw data files in `data`.

## Preprocess data
Run `preprocess.py` script after getting raw data:
```
$ python preprocess.py
```
The resulting cleaned dataset will be created in `data`.
