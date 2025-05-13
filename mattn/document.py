from dataclasses import dataclass

import torch

__all__ = ["Document", "EmbeddedDocument"]


@dataclass
class Document:
    name: str
    content: str


@dataclass
class EmbeddedDocument(Document):
    embedding: torch.Tensor
    """shape: (LLM hidden,)"""
