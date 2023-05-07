from datasets import load_dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset
from lion_pytorch import Lion
from typing import List, Union
from transformers import (
    AutoTokenizer,
)
from .constants import (
    DATASET_NAME,
    SPLIT_VAL,
    SPLIT_TRAIN,
    LANGUAGE_DATASET,
)
from openthaigpt_pretraining_model.models.llama.model import (
    ModelArgs,
    Transformer,
    ORIGIN_ATTENTION_MODE,
)


class DatasetWrapper(IterableDataset):
    def __init__(self, mode, model, max_tokens=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.mode = mode
        self.max_tokens = max_tokens

        if mode == "val":
            self.data_set = load_dataset(
                DATASET_NAME,
                LANGUAGE_DATASET,
                streaming=True,
                split=SPLIT_VAL,
            )
        elif mode == "train":
            self.data_set = load_dataset(
                DATASET_NAME,
                LANGUAGE_DATASET,
                streaming=True,
                split=SPLIT_TRAIN,
            ).shuffle(buffer_size=10_000)
        else:
            raise NotImplementedError("only support Train,Val")

    def __iter__(self):
        buffer = []
        iter_dataset = self.data_set

        for sample in iter_dataset:
            buffer += self.tokenizer(sample["text"])["input_ids"]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def do_eval(model, loader_val, device):
    val_loss = 0.0
    c_1 = 0
    for i1, batch1 in enumerate(loader_val):
        batch1 = batch1.to(device)
        loss1 = model(batch1, labels=batch1).loss
        val_loss = float(val_loss) + float(loss1.item())
        c_1 += 1
    print(f"loss_val : {(val_loss / c_1):.3f}")
    return val_loss / c_1


class Trainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        seed: int = 42,
        batch_size: int = 8,
        # grad: int = 4,
        context_length: int = 256,
        # max_steps: int = 1000,
        # eval_steps: int = 100,
        # warmup_steps: int = 2,
        model_name: str = "decapoda-research/llama-7b-hf",
        optimizer: str = "adamw",
        weight_decay: float = 1e-2,
        lr: float = 1e-4,
        # do_sample,
        # use_flash,
        # use_checkpointing,
    ):
        self.max_tokens = context_length
        self.step = 0
        self.seed = seed
        # self.max_steps = max_steps
        # self.warmup_steps = warmup_steps
        # self.eval_steps = eval_steps
        # self.do_sample = do_sample
        # self.use_flash = use_flash
        # self.use_checkpointing = use_checkpointing
        # self.fabric = L.Fabric(accelerator="cuda", devices=2, precision="16-mixed", strategy="ddp")
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
        )
        self.fabric.launch()

        self.dataset = DatasetWrapper("train", model_name, self.max_tokens)
        self.dataset_val = DatasetWrapper("val", model_name, self.max_tokens)
        self.tokenizer = self.dataset.tokenizer
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
        )

        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size)
        # self.model = model = createmodel(
        #     model_name,
        #     self.max_tokens,
        #     self.tokenizer,
        #     self.use_flash,
        #     self.use_checkpointing,
        #     self.device,
        # )
        cfg = ModelArgs(
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=32000,
            multiple_of=256,
            norm_eps=1e-5,
            max_batch_size=32,
            max_seq_len=2048,
            attention_mode=ORIGIN_ATTENTION_MODE,
        )
        self.model = model = Transformer(cfg)

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

        model, self.opt = self.fabric.setup(model, self.opt)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloder_val = self.fabric.setup_dataloaders(self.dataloader_val)

    def train_step(self, batch):
        loss = self.model(batch, labels=batch).loss
        self.fabric.backward(loss)
        return loss

    def val_step(self):
        self.model.eval()
        progress_bar = tqdm(self.dataloader_val)
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                # batch = batch.to(self.device)
                loss = self.model(batch, labels=batch).loss
            progress_bar.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()
        return loss

    def train(self):
        progress_bar = tqdm(self.dataloader)
        self.opt.zero_grad()

        for i, batch in enumerate(progress_bar):
            loss = self.train_step(batch)

            progress_bar.set_description(f"loss: {loss.item():.3f}")
            self.fabric.backward(loss)
            self.opt.step()
            self.opt.zero_grad()

        # self.model.eval()
        # val_loss = do_eval(self.model, self.data
        # loader_val, self.device)
        # self.model.train()
        val_loss = self.val_step()
        print(f"loss_val: {val_loss.item():.3f}")

        # if self.ddp:
        #     destroy_process_group()
