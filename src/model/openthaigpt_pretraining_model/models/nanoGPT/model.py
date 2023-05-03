import torch.nn.functional as F
import torch.backends.cuda as cuda
from transformers import AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention


_attn_orig = GPT2Attention._attn


# patch GPT2Attention to use flash_sdp, disable it when doing the inference
def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    if head_mask is not None:
        raise NotImplementedError("head_mask is not implemented for flash_sdp")
    is_causal = attention_mask is None
    with cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ):
        attn_out = F.scaled_dot_product_attention(
            query=query.half(),
            key=key.half(),
            value=value.half(),
            is_causal=is_causal,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p,
        ).float()
    return attn_out, None


def make_model(
    pretrained_name, max_tokens, tokenizer, use_flash, use_checkpointing, device
):
    config = AutoConfig.from_pretrained(
        pretrained_name,
        vocab_size=len(tokenizer),
        n_ctx=max_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        optimize_cuda_cache=True,
    )
    model = GPT2LMHeadModel(config).to(device)
    GPT2Attention._attn = _attn_orig
    if use_flash:
        print("Use Flash Attention")
        GPT2Attention._attn = _attn_wrapper

    model.resize_token_embeddings(len(tokenizer))

    # https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
    if use_checkpointing:
        model.gradient_checkpointing_enable()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPT-2 size requires_grad: {model_size/1000**2:.1f}M parameters")

    return model
