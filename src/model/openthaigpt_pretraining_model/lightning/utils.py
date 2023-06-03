from datasets import load_dataset
import numpy as np
import random
from tqdm import tqdm
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from lion_pytorch import Lion
from typing import List, Union
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)
from .constants import (
    DATASET_NAME,
    SPLIT_VAL,
    SPLIT_TRAIN,
    LANGUAGE_DATASET,
    LLAMA_MODEL,
    GPTJ_MODEL,
)
from openthaigpt_pretraining_model.models.gptj.gptj_model_xformers import (
    make_model_gptj,
)
from openthaigpt_pretraining_model.models.llama.model import make_model_llama
from openthaigpt_pretraining_model.models.llama_hf.model import (
    make_model_llama_hf,
)
from lightning.fabric.strategies import DeepSpeedStrategy
import deepspeed
import wandb
import os

# os.environ["WANDB_API_KEY"] = "<your-api-key>"
os.environ["WANDB_MODE"] = "offline"


class DatasetWrapper(IterableDataset):
    def __init__(self, mode, model, max_tokens=256):
        if model != "decapoda-research/llama-7b-hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(model)
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


def compute_perplexity(loss: torch.Tensor) -> float:
    return torch.exp(loss).item()


class Trainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        stage: int = 2,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        seed: int = 42,
        batch_size: int = 8,
        num_workers: int = 2,
        grad: int = 4,
        context_length: int = 256,
        model_name: str = "llama",
        optimizer: str = "adamw",
        weight_decay: float = 1e-2,
        lr: float = 1e-4,
        vocab_size: int = 50400,
        attention_mode: str = "origin",
        checkpoint: bool = False,
        checkpoint_only_attention: bool = False,
        num_nodes: int = 1,
    ):
        if torch.cuda.get_device_name(0) == "NVIDIA A100-SXM4-40GB":
            torch.set_float32_matmul_precision("medium")  # high
        self.wandb = None
        self.max_tokens = context_length
        self.step = 0
        self.seed = seed
        self.grad = grad
        if strategy == "deepspeed":
            strategy = DeepSpeedStrategy(
                stage=stage,
                offload_optimizer=offload_optimizer,
                offload_parameters=offload_parameters,
            )
        elif offload_optimizer or offload_parameters:
            raise NotImplementedError("offload only support for deepspeed strategy")
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            loggers=self.wandb,
            num_nodes=num_nodes,
        )
        self.fabric.launch()
        print(f"device:{self.fabric.device}")
        if self.fabric.global_rank == 0:
            self.wandb = wandb.init(project="Fabric")

        if model_name == "llama":
            model_name = LLAMA_MODEL  # for tokenizer
            self.model = make_model_llama(
                vocab_size=vocab_size,
                context_length=context_length,
                atention_mode=attention_mode,
                use_checkpointing=checkpoint,
                checkpoint_only_attention=checkpoint_only_attention,
            )
        elif model_name == "llama_hf":
            model_name = LLAMA_MODEL  # for tokenizer
            self.model = make_model_llama_hf(
                vocab_size=vocab_size,
                context_length=context_length,
                use_checkpointing=checkpoint,
                checkpoint_only_attention=checkpoint_only_attention,
            )
        elif model_name == "gptj":
            model_name = GPTJ_MODEL  # for tokenizer
            self.model = make_model_gptj(
                vocab_size=vocab_size,
                context_length=context_length,
                attention_mode=attention_mode,
                use_checkpointing=checkpoint,
                checkpoint_only_attention=checkpoint_only_attention,
                device=self.fabric.device,
            )
        else:
            raise NotImplementedError("only support Llama, llama_hf or GPTJ")

        self.dataset = DatasetWrapper("train", model_name, self.max_tokens)
        self.dataset_val = DatasetWrapper("val", model_name, self.max_tokens)
        self.tokenizer = self.dataset.tokenizer
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size)
        if offload_optimizer or offload_parameters:
            if optimizer == "adamw":
                print("Use AdamW optimizer")
                self.opt = deepspeed.ops.adam.DeepSpeedCPUAdam(
                    self.model.parameters(),
                    lr=lr,
                    bias_correction=True,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                    amsgrad=False,
                    adamw_mode=True,
                    fp32_optimizer_states=True,
                )
            else:
                raise NotImplementedError("only support AdamW")
        else:
            if optimizer == "lion":
                print("Use lion optimizer")
                self.opt = Lion(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            elif optimizer == "adamw":
                print("Use AdamW optimizer")
                self.opt = optim.AdamW(
                    params=self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                )
            else:
                raise NotImplementedError("only support lion or AdamW")

        self.model, self.opt = self.fabric.setup(self.model, self.opt)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloder_val = self.fabric.setup_dataloaders(self.dataloader_val)

    def log(self, data):
        if self.wandb is not None:
            self.wandb.log(data)

    def train_step(self, batch):
        loss = self.model(batch, labels=batch).loss
        return loss

    def val_step(self):
        self.model.eval()
        progress_bar = tqdm(self.dataloader_val)
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                loss = self.model(batch, labels=batch).loss
                perplexity = compute_perplexity(loss)
                self.log({"val_loss": loss.item(), "val_perplexity": perplexity})
            progress_bar.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()
        return loss

    def train(self):
        progress_bar = tqdm(self.dataloader, disable=(self.fabric.global_rank != 0))
        self.opt.zero_grad()

        for i, batch in enumerate(progress_bar):
            is_accumulating = (i + 1) % self.grad != 0

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss = self.train_step(batch)

                self.fabric.backward(loss)
                perplexity = compute_perplexity(loss)
                self.log({"train_loss": loss.item(), "train_perplexity": perplexity})
                progress_bar.set_description(f"loss: {loss.item():.3f}")

            if not is_accumulating:
                self.opt.step()
                self.opt.zero_grad()

        val_loss = self.val_step()
        print(f"loss_val: {val_loss.item():.3f}")

        self.run.finish()
