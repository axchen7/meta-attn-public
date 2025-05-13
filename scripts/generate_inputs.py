import os
import random
from typing import Annotated

import torch
import typer
from tqdm import tqdm

from mattn import (
    InputPair,
    InputPairWithRelevance,
    QueryAndTargets,
    RelevanceLabeler,
    detect_device,
)


def load_queries(dir_path: str) -> list[QueryAndTargets]:
    device = detect_device()
    embedded_documents: list[QueryAndTargets] = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        embedded_document = torch.load(
            file_path, map_location=device, weights_only=False
        )
        embedded_documents.append(embedded_document)
    return embedded_documents


def save_input(dir_path: str, input: InputPairWithRelevance, num: int):
    generating_document_name = input.query_and_targets.document.name
    this_document_name = input.document.name

    out_path = os.path.join(
        dir_path,
        f"{generating_document_name} <-> {this_document_name} [{num}][rel={input.relevance:.2f}].pt",
    )
    torch.save(input, out_path)


def generate_inputs(
    input_dir: Annotated[str, typer.Argument()] = "data/wikipedia/queries",
    output_dir: Annotated[str, typer.Argument()] = "data/wikipedia/inputs",
    num_other_documents_per_query: Annotated[
        int, typer.Option(help="Number of other documents to sample per query")
    ] = 2,
):
    os.makedirs(output_dir, exist_ok=True)
    queries = load_queries(input_dir)
    documents = [q.document for q in queries]

    relevance_labeler = RelevanceLabeler()

    for query_and_targets in tqdm(queries, desc="Computing relevance"):
        # save correct (query, document) pair
        input = relevance_labeler.compute_relevance(
            InputPair(
                document=query_and_targets.document,
                query_and_targets=query_and_targets,
            )
        )
        save_input(output_dir, input, 0)

        # save (query, other_document) pairs
        for i in range(num_other_documents_per_query):
            other_document = random.choice(documents)

            input = relevance_labeler.compute_relevance(
                InputPair(
                    document=other_document,
                    query_and_targets=query_and_targets,
                )
            )
            save_input(output_dir, input, i + 1)


if __name__ == "__main__":
    typer.run(generate_inputs)
