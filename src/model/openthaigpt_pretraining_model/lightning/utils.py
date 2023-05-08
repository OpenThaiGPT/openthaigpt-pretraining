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
    GPTJConfig,
    GPTJForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from .constants import (
    DATASET_NAME,
    SPLIT_VAL,
    SPLIT_TRAIN,
    LANGUAGE_DATASET,
    LLAMA_MODEL,
    GPTJ_MODEL,
)


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


class Trainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        seed: int = 42,
        batch_size: int = 8,
        grad: int = 4,
        context_length: int = 256,
        model_name: str = "llama",
        optimizer: str = "adamw",
        weight_decay: float = 1e-2,
        lr: float = 1e-4,
    ):
        self.max_tokens = context_length
        self.step = 0
        self.seed = seed
        self.grad = grad
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
        )
        self.fabric.launch()
        if model_name == "llama":
            model_name = LLAMA_MODEL  # for tokenizer
            cfg = LlamaConfig(
                vocab_size=32000,
                hidden_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                hidden_act="silu",
                max_position_embeddings=2048,  # sequence length
            )
            self.model = model = LlamaForCausalLM(cfg)
        elif model_name == "gptj":
            model_name = GPTJ_MODEL  # for tokenizer
            cfg = GPTJConfig(
                vocab_size=50400,
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
            self.model = model = GPTJForCausalLM(cfg)
        else:
            raise NotImplementedError("only support LlaMa or gptj")

        self.dataset = DatasetWrapper("train", model_name, self.max_tokens)
        self.dataset_val = DatasetWrapper("val", model_name, self.max_tokens)
        self.tokenizer = self.dataset.tokenizer
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
        )

        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size)

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
            )
        else:
            raise NotImplementedError("only support lion or AdamW")

        model, self.opt = self.fabric.setup(model, self.opt)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloder_val = self.fabric.setup_dataloaders(self.dataloader_val)

    def train_step(self, batch):
        loss = self.model(batch, labels=batch).loss
        return loss

    def val_step(self):
        self.model.eval()
        progress_bar = tqdm(self.dataloader_val)
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
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
            if (i + 1) % self.grad == 0:
                self.opt.step()
                self.opt.zero_grad()

        val_loss = self.val_step()
        print(f"loss_val: {val_loss.item():.3f}")
