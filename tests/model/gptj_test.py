from transformers import AutoTokenizer
from openthaigpt_pretraining_model.gptj.gptj_model_xformers import GPTJModel, GPTJConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

configuration_use_xformers = GPTJConfig.from_pretrained(
    "hf-internal-testing/tiny-random-gptj", use_xformers=True
)

configuration = GPTJConfig.from_pretrained(
    "hf-internal-testing/tiny-random-gptj", use_xformers=False
)


def test_gptj_xformers_attention():
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
    model1 = GPTJModel.from_pretrained(
        "hf-internal-testing/tiny-random-gptj", config=configuration
    )
    model2 = GPTJModel.from_pretrained(
        "hf-internal-testing/tiny-random-gptj", config=configuration_use_xformers
    )

    inputs_1 = tokenizer("Hello", return_tensors="pt")

    with torch.no_grad():
        assert torch.all(
            torch.abs(
                model1(**inputs_1).last_hidden_state
                - model2(**inputs_1).last_hidden_state
            )
            <= 1e-6
        )
