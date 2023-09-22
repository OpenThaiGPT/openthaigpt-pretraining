"""
from openthaigpt_pretraining_model.models.gptj.gptj_model_xformers import (
    XFORMER_ATTENTION_MODE,
    GPTJForCausalLMWithCheckpointing,
)
import torch
from transformers import AutoTokenizer, GPTJConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_output(pretrained_name, use_xformers, input_text):
    config = GPTJConfig.from_pretrained(pretrained_name)
    config.attention_mode = XFORMER_ATTENTION_MODE if use_xformers else "original"
    model = GPTJForCausalLMWithCheckpointing(config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model(inputs.input_ids)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states


def test_gptj_xformers_attention():
    pretrained_name = "hf-internal-testing/tiny-random-gptj"
    input_text = "Yo"
    output_xformers = get_output(
        pretrained_name=pretrained_name, use_xformers=True, input_text=input_text
    )
    output_originals = get_output(
        pretrained_name=pretrained_name, use_xformers=False, input_text=input_text
    )

    with torch.no_grad():
        assert torch.all(torch.abs(output_xformers - output_originals) <= 1e-6)
"""
