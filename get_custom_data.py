from os import remove, rename
from wget import download
from zipfile import ZipFile


# Download embeddings and coutry mappings
download("https://airrunner-public.s3.ca-central-1.amazonaws.com/ted-talks/ted_embeddings.zip")

download("https://airrunner-public.s3.ca-central-1.amazonaws.com/ted-talks/event_country_mapping.csv")
rename("event_country_mapping.csv", "data/event_country_mapping.csv")


# Extract data
print("\n\nExtracting data...")

with ZipFile("ted_embeddings.zip") as zip:
  zip.extractall(path="data/distances/")
remove("ted_embeddings.zip")

print("-- Done --")
