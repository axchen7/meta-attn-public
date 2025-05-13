import os
import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from .input_pair import InputPairWithRelevance
from .relevance import RelevanceModel
from .utils import detect_device

__all__ = ["load_inputs", "train"]


def load_inputs(inputs_dir: str) -> tuple[list[InputPairWithRelevance], int]:
    """Returns inputs, embedding_dim"""
    device = detect_device()
    inputs: list[InputPairWithRelevance] = []
    for file_name in os.listdir(inputs_dir):
        file_path = os.path.join(inputs_dir, file_name)
        input = torch.load(file_path, map_location=device, weights_only=False)
        inputs.append(input)
    embedding_dim = inputs[0].document.embedding.shape[0]
    print(f"Loaded {len(inputs)} inputs.")
    return inputs, embedding_dim


@dataclass
class TrainResult:
    train_losses: list[float]
    test_losses: list[float]
    one_of_n_trains: dict[int, list[float]]
    one_of_n_tests: dict[int, list[float]]
    plot_range: range


def train(
    model: RelevanceModel,
    inputs: list[InputPairWithRelevance],
    cost_fn: Literal["l1", "mse"],
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 0,
    train_split: float = 0.9,
    one_of_n: list[int] = [],
    log_interval: int = 10,
) -> TrainResult:
    # train/test split
    inputs = inputs.copy()
    random.seed(42)
    random.shuffle(inputs)
    split_index = int(len(inputs) * train_split)
    inputs_train = inputs[:split_index]
    inputs_test = inputs[split_index:]
    print(f"Train size: {len(inputs_train)}, Test size: {len(inputs_test)}")

    model.to(detect_device())

    x_train, y_train = model.inputs_to_tensors(inputs_train)
    x_test, y_test = model.inputs_to_tensors(inputs_test)

    one_of_n_trains = [OneOfNRetrieval(model, inputs_train, N=n) for n in one_of_n]
    one_of_n_tests = [OneOfNRetrieval(model, inputs_test, N=n) for n in one_of_n]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss() if cost_fn == "l1" else nn.MSELoss()

    result = TrainResult(
        train_losses=[],
        test_losses=[],
        one_of_n_trains={n: [] for n in one_of_n},
        one_of_n_tests={n: [] for n in one_of_n},
        plot_range=range(0, epochs, log_interval),
    )

    writer = SummaryWriter()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        y_hat_train = model(x_train)
        loss = loss_fn(y_hat_train, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        result.train_losses.append(loss.item())
        writer.add_scalar("Loss/train", loss, epoch)

        with torch.no_grad():
            model.eval()
            y_hat_test = model(x_test)
            test_loss = loss_fn(y_hat_test, y_test)

            result.test_losses.append(test_loss.item())
            writer.add_scalar("Loss/test", test_loss, epoch)

        if epoch % log_interval == 0:
            for n, one_of_n_train, one_of_n_test in zip(
                one_of_n, one_of_n_trains, one_of_n_tests
            ):
                one_of_n_score_train = one_of_n_train.compute_score()
                one_of_n_score_test = one_of_n_test.compute_score()

                result.one_of_n_trains[n].append(one_of_n_score_train)
                result.one_of_n_tests[n].append(one_of_n_score_test)
                writer.add_scalar(f"OneOfN/train/{n}", one_of_n_score_train, epoch)
                writer.add_scalar(f"OneOfN/test/{n}", one_of_n_score_test, epoch)

    writer.close()
    return result


class OneOfNRetrieval:

    def __init__(
        self, model: RelevanceModel, inputs: list[InputPairWithRelevance], N: int
    ):

        random.seed(42)
        documents = [input.document for input in inputs]
        x_groups: list[torch.Tensor] = []

        for input in inputs:
            correct_document = input.query_and_targets.document
            other_documents = random.sample(documents, N - 1)
            group_documents = [correct_document] + other_documents

            inputs = [
                InputPairWithRelevance(
                    document=document,
                    query_and_targets=input.query_and_targets,  # only query is used
                    relevance=0,  # not used
                )
                for document in group_documents
            ]

            x, _ = model.inputs_to_tensors(inputs)
            x_groups.append(x)

        self.model = model
        self.N = N
        self.n_groups = len(x_groups)
        self.x_stack = torch.concat(x_groups, dim=0)
        """For each group of N, the first document is the correct one, and the rest are random."""

    @torch.no_grad()
    def compute_score(self) -> float:
        """
        For a given query and N candidate documents, what is the % chance that the
        correct document has the highest relevance score?
        """

        self.model.eval()

        y_hat = self.model(self.x_stack)
        y_hat = y_hat.view(self.n_groups, self.N)

        max_indices = torch.argmax(y_hat, dim=1)
        correct_count = (max_indices == 0).sum().item()
        score = correct_count / self.n_groups
        return score
