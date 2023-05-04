import torch
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJModel,
)
from transformers import AutoTokenizer
import xformers.ops as xops

_attn_orig = GPTJAttention._attn

device = "cuda" if torch.cuda.is_available() else "cpu"


def _attn_xformers(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None,
):
    if attention_mask is not None:
        raise TypeError("Not support manual attention mask")

    if head_mask is not None:
        raise TypeError("Not support head_mask")

    # Attention output
    attn_output = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        xops.LowerTriangularMask(),
    )

    return attn_output, None


def get_output(pretrained_name, use_xformers, input_text):
    model = GPTJModel.from_pretrained(pretrained_name).to(device)
    if use_xformers:
        print("Use xFormers")
        GPTJAttention._attn = _attn_xformers
    else:
        print("Use original")
        GPTJAttention._attn = _attn_orig

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model(inputs.input_ids)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
