from contextlib import nullcontext
import time
import os

import torch

import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
import random
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from lion_pytorch import Lion
from bitsandbytes.optim import Adam8bit
import deepspeed

from openthaigpt_pretraining_model.models.nanoGPT.model import make_model, _attn_wrapper
from .constants import (
    DTYPE_CHOICE,
    MODEL_NAME,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    DATASET_NAME,
    SPLIT_VAL,
    SPLIT_TRAIN,
    LANGUAGE_DATASET,
)

_attn_orig = GPT2Attention._attn


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def closest_power_of_2(x):
    return 2 ** (x - 1).bit_length()


@torch.no_grad()
def do_eval(model, loader_val, ctx, device):
    val_loss = 0.0
    c_1 = 0
    for i1, batch1 in enumerate(loader_val):
        batch1 = batch1.to(device)
        with ctx:
            loss1 = model(batch1, labels=batch1).loss
            val_loss = float(val_loss) + float(loss1.item())
        c_1 += 1
    # print(f"loss_val : {(val_loss / c_1):.3f}")
    return val_loss / c_1


def get_torch_context(dtype: str):
    device_type = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # for later use in torch.autocast

    if dtype not in DTYPE_CHOICE.keys():
        raise NotImplementedError(
            f"dtype: {dtype} is not available. Only supports bfloat16|float32|float16"
        )

    ptdtype = DTYPE_CHOICE[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
    )
    return ctx


class DatasetWrapper(IterableDataset):
    def __init__(self, mode, max_tokens=256):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            MODEL_NAME,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
        )
        self.mode = mode
        self.max_tokens = max_tokens

        if mode == "val":
            self.data_set = load_dataset(
                DATASET_NAME,
                languages=[LANGUAGE_DATASET],
                streaming=True,
                split=SPLIT_VAL,  # optional
            )
        elif mode == "train":
            self.data_set = load_dataset(
                DATASET_NAME,
                languages=[LANGUAGE_DATASET],
                streaming=True,
                split=SPLIT_TRAIN,  # optional
            ).shuffle(buffer_size=10_000)
        else:
            raise NotImplementedError("only support Train,Val")

    def __iter__(self):
        buffer = []
        iter_dataset = self.data_set

        for sample in iter_dataset:
            buffer += self.tokenizer(sample["text"] + EOS_TOKEN)["input_ids"]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


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
        use_checkpointing,
        dtype: str,
        use_rotary,
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
        self.use_checkpointing = use_checkpointing
        self.use_rotary = use_rotary
        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
        )

        self.loader_val = DataLoader(self.dataset_val, batch_size=batch_size)

        self.backend = "nccl"

        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.backend)
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.ctx = get_torch_context(dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
        self.model = model = make_model(
            model_name,
            self.max_tokens,
            self.tokenizer,
            self.use_flash,
            self.use_checkpointing,
            self.device,
            self.use_rotary,
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
            self.opt = optim.AdamW(  # type: ignore
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                fused=True,
            )
        elif optimizer == "adam8bit":
            assert self.device != "cpu", "Adam8bit need GPU to execute"
            print("Use Adam8bit optimizer")
            self.opt = Adam8bit(
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
        elif optimizer == "adam1bit":
            print("Use Adam1bit optimizer")
            config_params = {
                "train_batch_size": batch_size,
                "optimizer": {
                    "type": "OneBitAdam",
                    "params": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "betas": (0.9, 0.95),
                    },
                },
            }
            self.model, self.opt, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=config_params,
            )
        else:
            raise NotImplementedError("only support lion or AdamW")
        self.model = torch.compile(model)  # type: ignore
        if self.ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

    def train_step(self, batch):
        batch = batch.to(self.device)
        with self.ctx:
            loss = self.model(batch, labels=batch).loss
            loss = loss / self.grad
        self.scaler.scale(loss).backward()
        return loss

    def val_step(self):
        self.model.eval()
        prog = tqdm(self.loader_val)
        for i, batch in enumerate(prog):
            batch = batch.to(self.device)
            with self.ctx:
                loss = self.model(batch, labels=batch).loss
                loss = loss / self.grad

            prog.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()

        return loss

    def generate_samples(self, n_samples=8):
        GPT2Attention._attn = _attn_orig  # back to faster but more memory consuming
        model = self.model
        x = torch.tensor([[self.tokenizer.eos_token_id]] * n_samples).to(self.device)
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
        print(f"Generated in {t1-t0:.3f}s")
        if self.use_flash:
            GPT2Attention._attn = _attn_wrapper

    def train(self):
        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            if self.ddp:
                self.model.require_backward_grad_sync = i % self.grad != 0

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")

            if i % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if i % self.eval_steps == 0 and i != 0:
                print("Step =", self.step)
                # loss_val = self.val_step()
                self.model.eval()
                val_loss = do_eval(self.model, self.loader_val, self.ctx, self.device)
                self.model.train()
                print(f"loss_val : {val_loss:.3f}")
                if self.do_sample:
                    self.generate_samples(6)

            self.grad = max(1, closest_power_of_2(i + 1) // 32)
            if self.step > self.max_steps:
                break

        if self.ddp:
            destroy_process_group()
