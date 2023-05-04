from openthaigpt_pretraining_model.gptj.gptj_model_xformers import (
    _attn_xformers,
    _attn_xformers_cpu,
)
import torch
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJModel,
)
from transformers import AutoTokenizer

_attn_orig = GPTJAttention._attn
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_output(pretrained_name, use_xformers, input_text):
    model = GPTJModel.from_pretrained(pretrained_name).to(device)
    if use_xformers:
        print("Use xFormers")
        if device == "cpu":
            GPTJAttention._attn = _attn_xformers_cpu
        else:
            GPTJAttention._attn = _attn_xformers
    else:
        print("Use original")
        GPTJAttention._attn = _attn_orig

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
