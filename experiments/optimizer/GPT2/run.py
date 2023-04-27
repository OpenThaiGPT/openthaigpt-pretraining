import time

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset

import numpy as np
import random
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention
from typing import Tuple, Optional, Callable
from torch.optim.optimizer import Optimizer


_attn_orig = GPT2Attention._attn


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# functions


def exists(val):
    return val is not None


# update functions


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# class


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        # assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = {lr: lr, betas: betas, weight_decay: weight_decay}

        super().__init__(params, defaults)

        # if use_triton:
        #     from lion_pytorch.triton import update_fn as triton_update_fn
        #     self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss


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


def closest_power_of_2(x):
    return 2 ** (x - 1).bit_length()


@torch.no_grad()
def do_eval(model, loader_val, grad):
    # model.eval()
    val_loss = 0.0
    c_1 = 0
    # prog1 = tqdm(self.loader_val)
    for i1, batch1 in enumerate(loader_val):
        batch1 = batch1.cuda()
        with torch.autocast(device_type="cuda", enabled=True):
            loss1 = model(batch1, labels=batch1).loss
            # loss1 = loss1 /grad
            val_loss = float(val_loss) + float(loss1.item())
        c_1 += 1
    print(f"loss_val : {(val_loss / c_1):.3f}")
    # model.train()
    return val_loss / c_1


def make_model(pretrained_name, max_tokens, tokenizer, use_flash):
    config = AutoConfig.from_pretrained(
        pretrained_name,
        vocab_size=len(tokenizer),
        n_ctx=max_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        optimize_cuda_cache=True,
    )
    # model = GPT2LMHeadModel.from_pretrained(pretrained_name).cuda()
    model = GPT2LMHeadModel(config).cuda()
    GPT2Attention._attn = _attn_orig
    if use_flash:
        print("Use Flash Attention")
        GPT2Attention._attn = _attn_wrapper

    # model.config.update(
    #     dict(
    #         n_ctx=max_tokens,
    #         #n_positions=max_tokens,
    #         vocab_size=len(tokenizer),
    #         bos_token_id=tokenizer.bos_token_id,
    #          eos_token_id=tokenizer.eos_token_id,
    #           pad_token_id=tokenizer.pad_token_id,
    #          optimize_cuda_cache=True
    #     )
    # )
    model.resize_token_embeddings(len(tokenizer))

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPT-2 size requires_grad: {model_size/1000**2:.1f}M parameters")
    # # patch model embeddings
    # emb = model.transformer.wpe.weight.data
    # wpe = nn.Embedding(max_tokens, emb.shape[1])
    # wpe.weight.data = emb.repeat(max_tokens // emb.shape[0], 1)
    # model.transformer.wpe = wpe

    # # also increase mask size
    # for block in model.transformer.h:
    #     block.attn.bias.data = (
    #         torch.tril(torch.ones((max_tokens, max_tokens), dtype=torch.bool))
    #         .view(1, 1, max_tokens, max_tokens)
    #         .cuda()
    #     )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model


class DatasetWrapper(IterableDataset):
    def __init__(self, mode, max_tokens=256):
        self.model_name = "flax-community/gpt2-base-thai"
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            self.model_name,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )
        self.mode = mode
        self.max_tokens = max_tokens
        self.data_train = load_dataset(
            "mc4", languages=["th"], streaming=True, split="train"  # optional
        ).shuffle(buffer_size=10_000)

        self.data_validate = load_dataset(
            "mc4", languages=["th"], streaming=True, split="validation"  # optional
        )

    def __iter__(self):
        buffer = []
        iter_dataset = self.data_train
        if self.mode == "val":
            iter_dataset = self.data_validate

        for sample in iter_dataset:
            buffer += self.tokenizer(sample["text"] + self.eos_token)["input_ids"]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


# class DatasetWrapper_val(IterableDataset):
#     def __init__(self, max_tokens=256):
#         self.model_name = "flax-community/gpt2-base-thai"
#         self.bos_token = "<|startoftext|>"
#         self.eos_token = "<|endoftext|>"
#         self.pad_token = "<|pad|>"
#         self.tokenizer = GPT2TokenizerFast.from_pretrained(
#             self.model_name,
#             bos_token=self.bos_token,
#             eos_token=self.eos_token,
#             pad_token=self.pad_token,
#         )
#         self.max_tokens = max_tokens
#         self.data_validate = load_dataset(
#             "mc4", languages=["th"], streaming=True, split="validation"  # optional
#         )

#     def __iter__(self):
#         buffer = []
#         for sample in self.data_validate:
#             buffer += self.tokenizer(sample["text"] + self.eos_token)["input_ids"]
#             # buffer += [self.tokenizer.eos_token_id]
#             # print(self.tokenizer.decode(buffer))
#             while len(buffer) > self.max_tokens:
#                 yield torch.tensor(buffer[: self.max_tokens])
#                 buffer = buffer[self.max_tokens :]


class Trainer:
    def __init__(
        self,
        optimizer,
        seed,
        batch_size,
        context_length,
        max_steps,
        eval_steps,
        warmup_steps,
        model_name,
        weight_decay,
        grad,
        lr,
        do_sample,
        use_flash,
    ):
        self.max_tokens = context_length
        self.grad = grad
        self.step = 0
        self.max_steps = max_steps
        self.seed = seed
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.do_sample = do_sample
        self.dataset = DatasetWrapper("train", self.max_tokens)
        self.dataset_val = DatasetWrapper("val", self.max_tokens)
        self.use_flash = use_flash

        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
        )

        self.loader_val = DataLoader(self.dataset_val, batch_size=batch_size)

        self.scaler = torch.cuda.amp.GradScaler()
        self.model = model = make_model(
            model_name, self.max_tokens, self.tokenizer, self.use_flash
        )

        if optimizer == "lion":
            print("Use lion optimizer")
            self.opt = Lion(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == "adamw":
            print("Use AdamW optimizer")
            self.opt = optim.AdamW(
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                fused=True,
            )
        else:
            raise NotImplementedError("only support lion or AdamW")
        self.model = torch.compile(model)

    def train_step(self, batch):
        batch = batch.cuda()
        with torch.autocast(device_type="cuda", enabled=True):
            loss = self.model(batch, labels=batch).loss
            loss = loss / self.grad
        self.scaler.scale(loss).backward()
        return loss

    def val_step(self):
        self.model.eval()
        prog = tqdm(self.loader_val)
        for i, batch in enumerate(prog):
            batch = batch.cuda()
            with torch.autocast(device_type="cuda", enabled=True):
                loss = self.model(batch, labels=batch).loss
                loss = loss / self.grad

            prog.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()

        return loss

    def generate_samples(self, n_samples=8):
        GPT2Attention._attn = _attn_orig  # back to faster but more memory consuming
        model = self.model
        x = torch.tensor([[self.tokenizer.eos_token_id]] * n_samples).cuda()
        t0 = time.time()
        model.eval()
        y = model.generate(
            inputs=x,
            max_length=self.max_tokens,
            do_sample=True,
        ).tolist()
        model.train()
        t1 = time.time()
        t = [self.tokenizer.decode(z) for z in y]
        for u in range(len(t)):
            print("samples = ", t[u])
        # t = "<hr>".join(f"<p>{c}</p>" for c in t)
        # html = WANDB_STYLE + t
        # wandb.log({"samples": wandb.Html(html)}, step=self.step)
        print(f"Generated in {t1-t0:.3f}s")
        if self.use_flash:
            GPT2Attention._attn = _attn_wrapper

    def train(self):
        # wandb.init(
        #     project="long-gptx",
        #     entity="_",
        # )

        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")
            # wandb.log(
            #     {
            #         "loss": loss.item(),
            #         "grad": self.grad,
            #     },
            #     step=i,
            # )

            if i % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if i % self.eval_steps == 0 and i != 0:
                print("Step =", self.step)
                # loss_val = self.val_step()
                self.model.eval()
                val_loss = do_eval(self.model, self.loader_val, self.grad)
                self.model.train()
                print(f"loss_val: {val_loss.item():.3f}")
                if self.do_sample:
                    self.generate_samples(6)

            self.grad = max(1, closest_power_of_2(i + 1) // 32)
            if self.step > self.max_steps:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--seed", type=int, default=42, help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=256, help="seq")
    parser.add_argument("--lr", type=float, default=5e-4, help="lr")
    parser.add_argument("--max_steps", type=int, default=800, help="max steps")
    parser.add_argument("--eval_steps", type=int, default=400, help="eval steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="warmup steps")
    parser.add_argument("--use_flash", default=False, action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="{gpt2|gpt2-medium|gpt2-large|gpt2-xl}",
    )
    parser.add_argument("--do_sample", default=False, action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="weight decay")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="gradient acc",
    )

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    trainer = Trainer(
        optimizer=args.optimizer,
        seed=args.seed,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        model_name=args.model_name,
        weight_decay=args.weight_decay,
        grad=args.gradient_accumulation_steps,
        lr=args.lr,
        do_sample=args.do_sample,
        use_flash=args.use_flash,
    )
    trainer.train()
