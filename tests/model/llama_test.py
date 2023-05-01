from openthaigpt_pretraining_model.models.llama.model import (
    ModelArgs,
    Transformer,
    ORIGIN_ATTENTION_MODE,
    PYTORCH_ATTENTION_MODE,
    XFORMER_ATTENTION_MODE,
)
import torch
import pytest


device = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 12
LLAMA_TEST_CASES = [
    torch.randint(0, VOCAB_SIZE, (1, 1)).to(device),
    torch.randint(0, VOCAB_SIZE, (12, 1)).to(device),
    torch.randint(0, VOCAB_SIZE, (1, 12)).to(device),
    torch.randint(0, VOCAB_SIZE, (12, 12)).to(device),
    torch.randint(0, VOCAB_SIZE, (6, 4)).to(device),
]
KEYERROR_TEST_CASE = ["", "torch", "formger", "false", "true"]


def test_llama_efficient_for_torch():
    base_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode=ORIGIN_ATTENTION_MODE)
    torch_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode=PYTORCH_ATTENTION_MODE)

    base_model = Transformer(base_args).to(device)
    torch_model = Transformer(torch_args).to(device)

    torch_model.load_state_dict(base_model.state_dict())

    with torch.no_grad():
        for test_case in LLAMA_TEST_CASES:
            assert torch.all(
                torch.abs(
                    base_model(test_case.to(device), 0)
                    - torch_model(test_case.to(device), 0)
                )
                <= 1e-6
            )


def test_llama_efficient_for_xformer():
    base_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode=ORIGIN_ATTENTION_MODE)
    xformer_args = ModelArgs(
        vocab_size=VOCAB_SIZE, attention_mode=XFORMER_ATTENTION_MODE
    )

    base_model = Transformer(base_args).to(device)
    xformer_model = Transformer(xformer_args).to(device)

    xformer_model.load_state_dict(base_model.state_dict())
    with torch.no_grad():
        for test_case in LLAMA_TEST_CASES:
            assert torch.all(
                torch.abs(
                    base_model(test_case.to(device), 0)
                    - xformer_model(test_case.to(device), 0)
                )
                <= 1e-6
            )


def test_key_error():
    for key in KEYERROR_TEST_CASE:
        with pytest.raises(KeyError):
            args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode=key)
            Transformer(args)
