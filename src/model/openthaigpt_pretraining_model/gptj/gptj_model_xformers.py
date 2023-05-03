import torch
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJModel,
)
from transformers import AutoTokenizer
from xformers.components.attention.core import scaled_query_key_softmax, bmm

_attn_orig = GPTJAttention._attn


def _attn_xformers(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None,
):
    # compute causal mask from causal mask buffer
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[
        :, :, key_length - query_length : key_length, :key_length  # noqa: E203
    ]
    attention_mask = torch.where(causal_mask, 0.0, -torch.inf)
    # Keep the attention weights computation in fp32
    # to avoid overflow issues
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    attn_weights = scaled_query_key_softmax(query, key, attention_mask)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    # attn_output = bmm(attn_weights, value)
    attn_output = bmm(attn_weights, value)

    return attn_output, attn_weights


def get_output(pretrained_name, use_xformers, input_text):
    model = GPTJModel.from_pretrained(pretrained_name)
    if use_xformers:
        print("Use xFormers")
        GPTJAttention._attn = _attn_xformers
    else:
        print("Use original")
        GPTJAttention._attn = _attn_orig

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states
