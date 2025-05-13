from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from .input_pair import InputPairWithRelevance

__all__ = [
    "RelevanceModel",
    "FeedForwardRelevanceModel",
    "LatentEmbeddingRelevanceModel",
    "CosineSimilarityRelevanceModel",
]


class RelevanceModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def inputs_to_tensors(
        inputs: list[InputPairWithRelevance],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """returns x, y"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class FeedForwardRelevanceModel(RelevanceModel):
    """Takes the concatenation of the document and query embeddings as input."""

    def __init__(self, *, embedding_dim: int, hidden_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def inputs_to_tensors(inputs: list[InputPairWithRelevance]):
        device = inputs[0].document.embedding.device

        document_embeddings = torch.stack(
            [input.document.embedding for input in inputs]
        )
        query_embeddings = torch.stack(
            [input.query_and_targets.query.embedding for input in inputs]
        )

        x = torch.cat([document_embeddings, query_embeddings], dim=1)
        y = torch.tensor(
            [input.relevance for input in inputs], device=device, dtype=torch.float32
        )

        return x, y

    def forward(self, x: torch.Tensor):
        return self.mlp(x).squeeze()


class LatentEmbeddingRelevanceModel(RelevanceModel):
    """Takes the document and query embeddings as input."""

    def __init__(self, *, embedding_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()

        self.document_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def inputs_to_tensors(inputs: list[InputPairWithRelevance]):
        device = inputs[0].document.embedding.device

        document_embeddings = torch.stack(
            [input.document.embedding for input in inputs]
        )
        query_embeddings = torch.stack(
            [input.query_and_targets.query.embedding for input in inputs]
        )

        x = torch.cat([document_embeddings, query_embeddings], dim=1)
        y = torch.tensor(
            [input.relevance for input in inputs], device=device, dtype=torch.float32
        )

        return x, y

    def forward(self, x: torch.Tensor):
        document_embedding, query_embedding = x.chunk(2, dim=1)
        document_latent = self.document_mlp(document_embedding)
        query_latent = self.query_mlp(query_embedding)
        l2_distance = torch.norm(document_latent - query_latent, dim=1)
        return -l2_distance


class CosineSimilarityRelevanceModel(RelevanceModel):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def inputs_to_tensors(inputs: list[InputPairWithRelevance]):
        device = inputs[0].document.embedding.device

        document_embeddings = torch.stack(
            [input.document.embedding for input in inputs]
        )
        query_embeddings = torch.stack(
            [input.query_and_targets.query.embedding for input in inputs]
        )

        x = torch.cat([document_embeddings, query_embeddings], dim=1)
        y = torch.tensor(
            [input.relevance for input in inputs], device=device, dtype=torch.float32
        )

        return x, y

    def forward(self, x: torch.Tensor):
        document_embedding, query_embedding = x.chunk(2, dim=1)
        cosine_similarity = F.cosine_similarity(
            document_embedding, query_embedding, dim=1
        )
        return self.scale * (cosine_similarity - 1)
