import click
import pandas as pd
from tabulate import tabulate

from model import TEDflix


@click.command()
@click.argument("history", type=int, nargs=-1)
@click.option("-k", default=5, show_default=True, help="Number of recommendations to return")
def main(history, k):
    model = TEDflix()
    model.train()

    predictions = model.predict(history, k)

    df = pd.read_csv("data/ted_talks_infos.csv")[["title", "speaker_1", "published_date", "url"]]
    predicted_talks = df.iloc[predictions]

    print(tabulate(
        predicted_talks,
        ["Title", "Speaker", "Date", "URL"],
        tablefmt="psql",
        showindex=[i + 1 for i in range(k)]
    ))


if __name__ == "__main__":
    main()
