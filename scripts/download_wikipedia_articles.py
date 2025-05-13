import os
from typing import Annotated

import requests
import typer
from tqdm import tqdm


def get_random_titles(n: int):
    batch_size = 50
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    titles = []
    while len(titles) < n:
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(batch_size, n - len(titles)),
            "format": "json",
        }
        res = S.get(url=URL, params=params)
        data = res.json()
        titles += [item["title"] for item in data["query"]["random"]]
    return titles


def get_plain_text(title: str):
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    res = requests.get(URL, params=params)
    pages = res.json()["query"]["pages"]
    for page_id in pages:
        return pages[page_id].get("extract", "")


def download_wikipedia_articles(
    directory: Annotated[str, typer.Argument()] = "data/wikipedia/documents",
    num_articles: Annotated[int, typer.Option()] = 300,
):
    """
    Download a random sample of Wikipedia articles as .txt files.
    """
    os.makedirs(directory, exist_ok=True)
    titles = get_random_titles(num_articles)
    for title in tqdm(titles, desc="Downloading articles"):
        text = get_plain_text(title)
        filename = (
            "".join([c if c.isalnum() or c in " _-" else "_" for c in title]) + ".txt"
        )
        filepath = os.path.join(directory, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text or "")


if __name__ == "__main__":
    typer.run(download_wikipedia_articles)
