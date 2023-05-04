import torch
import xformers.ops as xops
from xformers.components.attention.core import scaled_query_key_softmax, bmm


def _attn_xformers_cpu(
    self,
    query,
    key,
    value,
    attention_mask=None,
    head_mask=None,
):
    # compute causal mask from causal mask buffer
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    attention_mask = torch.where(causal_mask, 0.0, -torch.inf)
    # Keep the attention weights computation in fp32 to avoid overflow issues
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
