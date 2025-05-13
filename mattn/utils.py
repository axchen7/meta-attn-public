import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

__all__ = ["detect_device"]


def detect_device() -> torch.device:
    """
    Use the GPU if available, otherwise use the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_tokenizer_and_model(local_model: str, device: torch.device):
    """Returns tokenizer, model"""

    tokenizer = AutoTokenizer.from_pretrained(local_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(local_model, device_map=device)
    model.eval()

    return tokenizer, model
