from os import remove
from wget import download
from zipfile import ZipFile


# Download
download("https://airrunner-public.s3.ca-central-1.amazonaws.com/ted_embeddings.zip")

# Extract data
print("\n\nExtracting data...")

with ZipFile("ted_embeddings.zip") as zip:
  zip.extractall(path="data/embeddings/")
remove("ted_embeddings.zip")

print("-- Done --")
