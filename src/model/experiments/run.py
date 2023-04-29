import time

import argparse
import torch

import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import numpy as np
import random
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from lion_pytorch import Lion

from openthaigpt_pretraining_model.nanoGPT.model import (
    make_model,
    _attn_wrapper,
    _attn_orig,
)
from openthaigpt_pretraining_model.nanoGPT.constants import (
    MODEL_NAME,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    DATASET_NAME,
    CUDA,
    SPLIT_VAL,
    SPLIT_TRAIN,
    LANGUAGE_DATASET,
)


# _attn_orig = GPT2Attention._attn
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
def do_eval(model, loader_val):
    val_loss = 0.0
    c_1 = 0
    for i1, batch1 in enumerate(loader_val):
        batch1 = batch1.cuda()
        with torch.autocast(device_type=CUDA, enabled=True):
            loss1 = model(batch1, labels=batch1).loss
            val_loss = float(val_loss) + float(loss1.item())
        c_1 += 1
    print(f"loss_val : {(val_loss / c_1):.3f}")
    return val_loss / c_1


class DatasetWrapper(IterableDataset):
    def __init__(self, mode, max_tokens=256):
        self.model_name = MODEL_NAME
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            self.model_name,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
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
            buffer += self.tokenizer(sample["text"] + self.eos_token)["input_ids"]
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
        with torch.autocast(device_type=CUDA, enabled=True):
            loss = self.model(batch, labels=batch).loss
            loss = loss / self.grad
        self.scaler.scale(loss).backward()
        return loss

    def val_step(self):
        self.model.eval()
        prog = tqdm(self.loader_val)
        for i, batch in enumerate(prog):
            batch = batch.cuda()
            with torch.autocast(device_type=CUDA, enabled=True):
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
        print(f"Generated in {t1-t0:.3f}s")
        if self.use_flash:
            GPT2Attention._attn = _attn_wrapper

    def train(self):
        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

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
                val_loss = do_eval(self.model, self.loader_val)
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
