from dataclasses import dataclass

import torch

from .document import EmbeddedDocument

__all__ = ["Query", "EmbeddedQuery", "TargetOutput", "QueryAndTargets"]


@dataclass
class Query:
    content: str


@dataclass
class EmbeddedQuery(Query):
    embedding: torch.Tensor
    """shape: (LLM hidden,)"""


@dataclass
class TargetOutput:
    content: str
    token_ids: list[int]
    """shape: (n_tokens,)"""
    logprob: float
    """Log probability of the target output, under the model used to generate it"""


@dataclass
class QueryAndTargets:
    query: EmbeddedQuery
    document: EmbeddedDocument
    """Document used to generate the query"""
    targets: list[TargetOutput]
    """At least one target exists"""
