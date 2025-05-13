import os

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from .document import Document, EmbeddedDocument
from .query import EmbeddedQuery, Query
from .utils import create_tokenizer_and_model, detect_device

__all__ = ["EmbeddingModel"]

load_dotenv()


class EmbeddingModel:
    def __init__(
        self,
        *,
        device: torch.device = detect_device(),
        local_model: str = os.getenv("LOCAL_MODEL", "google/gemma-2-2b"),
        batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 16)),
    ):
        self.device = device
        self.batch_size = batch_size

        self.tokenizer, self.model = create_tokenizer_and_model(local_model, device)

    def __embed(self, contents: list[str]) -> list[torch.Tensor]:
        embeddings: list[torch.Tensor] = []

        for i in tqdm(
            range(0, len(contents), self.batch_size), desc="Computing embeddings"
        ):
            batch_contents = contents[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_contents, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
                attention_mask = inputs["attention_mask"]  # (batch, seq_len)
                mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                masked_hidden = last_hidden * mask  # (batch, seq_len, hidden)
                lengths = mask.sum(dim=1)  # (batch, 1)
                batch_embeddings = masked_hidden.sum(dim=1) / lengths  # (batch, hidden)

            embeddings.extend(batch_embeddings.cpu())

        return embeddings

    def embed_documents(self, documents: list[Document]) -> list[EmbeddedDocument]:
        contents = [doc.content for doc in documents]
        embeddings = self.__embed(contents)

        return [
            EmbeddedDocument(name=doc.name, content=doc.content, embedding=emb)
            for doc, emb in zip(documents, embeddings)
        ]

    def embed_queries(self, queries: list[Query]) -> list[EmbeddedQuery]:
        contents = [query.content for query in queries]
        embeddings = self.__embed(contents)

        return [
            EmbeddedQuery(content=query.content, embedding=emb)
            for query, emb in zip(queries, embeddings)
        ]
