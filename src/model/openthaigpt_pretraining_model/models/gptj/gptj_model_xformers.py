import torch
import xformers.ops as xops
from xformers.components.attention.core import scaled_query_key_softmax, bmm
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
)
from transformers import GPTJConfig, GPTJForCausalLM


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
    attn_output = xops.memory_efficient_attention(
        query.transpose(2, 1),
        key.transpose(2, 1),
        value.transpose(2, 1),
        xops.LowerTriangularMask(),
    ).transpose(2, 1)

    return attn_output, None


def make_model_gptj(
    vocab_size, context_length, use_xformers, use_checkpointing, device: str = "cuda"
):
    config = GPTJConfig(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_embd=1536,
        n_layer=12,
        n_head=8,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
    )
    model = GPTJForCausalLM(config)

    if use_xformers:
        print("Use xFormers")
        if device == "cpu":
            GPTJAttention._attn = _attn_xformers_cpu
        else:
            GPTJAttention._attn = _attn_xformers
    else:
        print("Use original")

    # https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
    if use_checkpointing:
        model.gradient_checkpointing_enable()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPTJ size: {model_size/1000**2:.1f}M parameters")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPTJ size requires_grad: {model_size/1000**2:.1f}M parameters")

    return model
