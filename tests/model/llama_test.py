from openthaigpt_pretraining_model.llama.model import ModelArgs, Transformer
import numpy as np
import torch


VOCAB_SIZE = 12
LLAMA_TEST_CASES = [
    torch.randint(0, VOCAB_SIZE, (1, 1)),
    torch.randint(0, VOCAB_SIZE, (12, 1)),
    torch.randint(0, VOCAB_SIZE, (1, 12)),
    torch.randint(0, VOCAB_SIZE, (12, 12)),
    torch.randint(0, VOCAB_SIZE, (6, 4)),
]


def test_llama_efficient_for_torch():
    base_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode='origin')
    torch_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode='pytorch')

    base_model = Transformer(base_args)
    torch_model = Transformer(torch_args)

    for test_case in LLAMA_TEST_CASES:
        np.testing.assert_almost_equal(base_model(test_case), torch_model(test_case), decimal=6)


def test_llama_efficient_for_xformer():
    base_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode='origin')
    xformer_args = ModelArgs(vocab_size=VOCAB_SIZE, attention_mode='xformer')

    base_model = Transformer(base_args)
    xformer_model = Transformer(xformer_args)

    for test_case in LLAMA_TEST_CASES:
        np.testing.assert_almost_equal(base_model(test_case), xformer_model(test_case), decimal=6)