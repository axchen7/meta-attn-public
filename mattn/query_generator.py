import asyncio
import os
from typing import Sequence

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from .document import Document, EmbeddedDocument
from .embedding import EmbeddingModel
from .input_pair import RelevanceLabeler
from .prompting import ask_for_query_from_document, create_query_document_input
from .query import Query, QueryAndTargets, TargetOutput
from .utils import detect_device

load_dotenv()


openai = AsyncOpenAI()

__all__ = ["QueryGenerator"]

QUERY_GENERATOR_TEMPERATURE = 0.5


class QueryGenerator:
    def __init__(
        self,
        *,
        num_targets: int = 1,
        device: torch.device = detect_device(),
        local_model: str = os.getenv("LOCAL_MODEL", "google/gemma-2-2b"),
        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_parallelism: int = 16,
        max_new_tokens: int = 50,
        batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 16)),
    ):
        self.num_targets = num_targets
        self.device = device
        self.openai_model = openai_model
        self.openai_parallelism = openai_parallelism
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        self.embedding_model = EmbeddingModel(
            device=device, local_model=local_model, batch_size=batch_size
        )
        self.tokenizer = self.embedding_model.tokenizer
        self.model = self.embedding_model.model

    async def __async_generate_query(self, document: Document):
        openai_res = await openai.responses.create(
            model=self.openai_model,
            input=ask_for_query_from_document(document),
            temperature=QUERY_GENERATOR_TEMPERATURE,
        )
        return Query(content=openai_res.output_text)

    def generate_queries(self, documents: Sequence[Document]) -> list[Query]:
        async def run_parallel():
            sem = asyncio.Semaphore(self.openai_parallelism)
            pbar = tqdm(total=len(documents), desc="Generating queries")

            async def sem_task(document):
                async with sem:
                    result = await self.__async_generate_query(document)
                    pbar.update(1)
                    return result

            tasks = [sem_task(document) for document in documents]
            all_results = await asyncio.gather(*tasks)
            pbar.close()
            return all_results

        return asyncio.run(run_parallel())

    def generate_targets(
        self, queries: list[Query], documents: list[EmbeddedDocument]
    ) -> list[QueryAndTargets]:
        assert len(queries) == len(documents)

        embedded_queries = self.embedding_model.embed_queries(queries)

        inputs = [create_query_document_input(q, d) for q, d in zip(queries, documents)]

        all_input_ids: list[torch.Tensor] = []  # shape: (n_queries, seq_len)
        sequences: list[torch.Tensor] = []  # shape: (n_queries * n_targets, seq_len)

        batch_size = self.batch_size // self.num_targets

        for i in tqdm(range(0, len(queries), batch_size), desc="Generating targets"):
            batch_inputs = inputs[i : i + batch_size]
            batch_inputs = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, padding_side="left"
            )
            batch_input_ids = batch_inputs["input_ids"].to(self.device)
            batch_attention_mask = batch_inputs["attention_mask"].to(self.device)

            # sequences shape: (batch_size * n_targets, seq_len)
            batch_sequences = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                num_return_sequences=self.num_targets,
                do_sample=self.num_targets > 1,  # greedy decode if only one target
                temperature=1.0,  # ignored if greedy decode
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            all_input_ids.extend(batch_input_ids)
            sequences.extend(batch_sequences)

        results: list[QueryAndTargets] = []

        for i in tqdm(range(len(queries)), desc="Computing logprobs"):
            targets: list[TargetOutput] = []

            for j in range(self.num_targets):
                idx = i * self.num_targets + j
                input_ids = all_input_ids[i]
                # Get the generated part (after the prompt)
                output_ids = sequences[idx][len(input_ids) :]

                pad = self.tokenizer.pad_token_id
                trimmed_input_ids = [t for t in input_ids.tolist() if t != pad]
                trimmed_output_ids = [t for t in output_ids.tolist() if t != pad]

                content = self.tokenizer.decode(trimmed_output_ids)
                logprob = RelevanceLabeler.compute_logprob(
                    self.model, self.device, trimmed_input_ids, trimmed_output_ids
                )
                targets.append(
                    TargetOutput(
                        content=content, token_ids=trimmed_output_ids, logprob=logprob
                    )
                )

            results.append(
                QueryAndTargets(
                    query=embedded_queries[i], document=documents[i], targets=targets
                )
            )

        return results
