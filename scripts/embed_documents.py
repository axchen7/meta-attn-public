import os
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv

from mattn import Document, EmbeddingModel

load_dotenv()


def load_document(file_path: str, max_chars: int) -> Document:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r") as f:
        content = f.read()
        content = content[:max_chars]
    return Document(name=base_name, content=content)


def load_documents(dir_path: str, max_chars: int) -> list[Document]:
    documents = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        document = load_document(file_path, max_chars)
        documents.append(document)
    return documents


def embed_documents(
    input_dir: Annotated[str, typer.Argument()] = "data/wikipedia/documents",
    output_dir: Annotated[str, typer.Argument()] = "data/wikipedia/embedded_documents",
    max_document_chars: Annotated[int, typer.Option()] = int(
        os.getenv("MAX_DOCUMENT_CHARS", 1024)
    ),
):
    os.makedirs(output_dir, exist_ok=True)
    documents = load_documents(input_dir, max_document_chars)
    model = EmbeddingModel()
    embedded_documents = model.embed_documents(documents)
    for embedded_doc in embedded_documents:
        out_path = os.path.join(output_dir, f"{embedded_doc.name}.pt")
        torch.save(embedded_doc, out_path)


if __name__ == "__main__":
    typer.run(embed_documents)
