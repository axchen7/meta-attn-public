import os
from typing import Annotated

import torch
import typer

from mattn import EmbeddedDocument, QueryGenerator, detect_device


def load_embedded_documents(dir_path: str) -> list[EmbeddedDocument]:
    device = detect_device()
    embedded_documents: list[EmbeddedDocument] = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        embedded_document = torch.load(
            file_path, map_location=device, weights_only=False
        )
        embedded_documents.append(embedded_document)
    return embedded_documents


def generate_queries(
    input_dir: Annotated[str, typer.Argument()] = "data/wikipedia/embedded_documents",
    output_dir: Annotated[str, typer.Argument()] = "data/wikipedia/queries",
):
    os.makedirs(output_dir, exist_ok=True)
    documents = load_embedded_documents(input_dir)

    query_generator = QueryGenerator()

    queries = query_generator.generate_queries(documents)
    queries = query_generator.generate_targets(queries, documents)

    for query, document in zip(queries, documents):
        out_path = os.path.join(output_dir, f"{document.name}.pt")
        torch.save(query, out_path)


if __name__ == "__main__":
    typer.run(generate_queries)
