import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

from .document import EmbeddedDocument
from .prompting import create_query_document_input
from .query import QueryAndTargets
from .utils import create_tokenizer_and_model, detect_device

load_dotenv()

__all__ = ["InputPair", "InputPairWithRelevance", "RelevanceLabeler"]


@dataclass
class InputPair:
    document: EmbeddedDocument
    query_and_targets: QueryAndTargets


@dataclass
class InputPairWithRelevance(InputPair):
    relevance: float


class RelevanceLabeler:
    def __init__(
        self,
        *,
        device: torch.device = detect_device(),
        local_model: str = os.getenv("LOCAL_MODEL", "google/gemma-2-2b"),
    ):
        self.device = device
        self.tokenizer, self.model = create_tokenizer_and_model(local_model, device)

    @staticmethod
    def compute_logprob(
        model,
        device: torch.device,
        input_token_ids: list[int],
        output_token_ids: list[int],
    ) -> float:
        # Concatenate input and output
        full_ids = input_token_ids + output_token_ids
        input_len = len(input_token_ids)
        input_tensor = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            outputs = model(input_tensor)
            # shape: (output_len, vocab_size); slice off the new token
            logits = outputs.logits[0][input_len - 1 : -1]

            logprob = 0
            for i, token_id in enumerate(output_token_ids):
                log_probs = torch.log_softmax(logits[i], dim=-1)
                logprob += log_probs[token_id].item()

        return logprob

    def compute_relevance(self, input: InputPair) -> InputPairWithRelevance:
        input_for_document = create_query_document_input(
            input.query_and_targets.query, input.document
        )
        input_ids = self.tokenizer.encode(input_for_document)

        # TODO assuming only one target; need to convert to KL divergence over all targets
        first_target = input.query_and_targets.targets[0]
        first_target_logprob = RelevanceLabeler.compute_logprob(
            self.model, self.device, input_ids, first_target.token_ids
        )
        relevance = first_target_logprob - first_target.logprob

        return InputPairWithRelevance(
            document=input.document,
            query_and_targets=input.query_and_targets,
            relevance=relevance,
        )
