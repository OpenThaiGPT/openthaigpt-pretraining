from typing import Optional
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import xformers.ops as xops

from llama.model import RMSNorm, apply_rotary_emb, precompute_freqs_cis  # type: ignore
from typing import NamedTuple

ORIGIN_ATTENTION_MODE = "origin"
PYTORCH_ATTENTION_MODE = "pytorch"
XFORMERS_ATTENTION_MODE = "xformers"


@dataclass
class LLaMAArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    attention_mode: str = ORIGIN_ATTENTION_MODE  # pytorch, xformers
    use_checkpointing: bool = False
    checkpoint_only_attention: bool = False


class OutputModel(NamedTuple):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Attention(nn.Module):
    def __init__(self, args: LLaMAArgs):
        super().__init__()

        self.head_dim = args.dim // args.n_heads

        self.n_local_heads = args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        if not (
            args.attention_mode == ORIGIN_ATTENTION_MODE
            or args.attention_mode == PYTORCH_ATTENTION_MODE  # noqa: W503
            or args.attention_mode == XFORMERS_ATTENTION_MODE  # noqa: W503
        ):
            raise KeyError(
                f'attention mode must be "{ORIGIN_ATTENTION_MODE}", "{XFORMERS_ATTENTION_MODE}" or "{PYTORCH_ATTENTION_MODE}"'  # noqa: E501
            )

        self.mode = args.attention_mode
        self.checkpoint_only_attention = False
        if args.use_checkpointing and args.checkpoint_only_attention:
            self.checkpoint_only_attention = args.checkpoint_only_attention

    def compute_attention(self, xq, keys, values, mask):
        if self.mode == ORIGIN_ATTENTION_MODE or xq.device == torch.device("cpu"):
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)
            output = output.transpose(1, 2)
        elif self.mode == PYTORCH_ATTENTION_MODE:
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            output = F.scaled_dot_product_attention(xq, keys, values, mask)
            output = output.transpose(1, 2)
        elif self.mode == XFORMERS_ATTENTION_MODE:
            output = xops.memory_efficient_attention(
                xq, keys, values, attn_bias=xops.LowerTriangularMask()
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        if self.checkpoint_only_attention:
            output = torch.utils.checkpoint.checkpoint(
                self.compute_attention, xq, keys, values, mask
            )
        else:
            output = self.compute_attention(xq, keys, values, mask)

        output = output.contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LLaMAArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMA(nn.Module):
    def __init__(self, params: LLaMAArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.use_checkpointing = False
        if params.use_checkpointing:
            if params.checkpoint_only_attention:
                print("use gradient checkpointing only attentions")
            else:
                self.use_checkpointing = params.use_checkpointing
                print("use gradient checkpointing")

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        labels: Optional[torch.Tensor] = None,
    ) -> OutputModel:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # noqa: E203

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            if self.use_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, start_pos, freqs_cis, mask
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :, 1:].contiguous()

            loss_fn = CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return OutputModel(logits=logits[:, -1, :], loss=loss)
