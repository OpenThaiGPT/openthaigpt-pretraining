from datasets import load_dataset, load_from_disk
import numpy as np
import random
from tqdm import tqdm
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, Dataset, DataLoader
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
import wandb
import os
from re import findall

# os.environ["WANDB_API_KEY"] = "<your-api-key>"
os.environ["WANDB_MODE"] = "offline"


class TokenizedDataset:
    def __init__(
        self,
        mode: str,
        model_or_path: str,
        max_tokens: int = 256,
        save_path: str = "./",
        chunk_size: int = 1024 * 1024,
        batch_size: int = 10000,
        num_proc: int = 16,
        use_cache: bool = True,
        dataset_name: str = "oscar",
        dataset_dir: str = "unshuffled_deduplicated_th",
    ):
        if len(findall("llama", model_or_path)):
            self.tokenizer = LlamaTokenizer.from_pretrained(model_or_path)
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)

        self.mode = mode
        self.max_tokens = max_tokens
        self.save_path = save_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        if use_cache:
            if mode == "val":
                self.data_set = load_dataset(
                    dataset_name,
                    dataset_dir,
                    split=SPLIT_TRAIN,
                )
            elif mode == "train":
                self.data_set = load_dataset(
                    dataset_name,
                    dataset_dir,
                    split=SPLIT_TRAIN,
                )
            else:
                raise NotImplementedError("only support Train,Val")
            self.num_shards = (len(self.data_set) + chunk_size - 1) // chunk_size

        self.tokenized_data = None

    def tokenize_data(self):
        def tokenize_function(examples):
            tokenized_text = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_tokens,
            )["input_ids"]
            return {"tokenized_text": tokenized_text}

        for i in tqdm(range(self.num_shards)):
            chunk = self.data_set.shard(self.num_shards, i)  # split chunk
            tokenized_dataset = chunk.map(
                tokenize_function,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
            )
            print(f"save {self.mode}_chunk_{i}")
            tokenized_dataset.save_to_disk(
                os.path.join(self.save_path, f"{self.mode}_chunk_{i}")
            )


class ChunkedDatasetWrapper(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset
        self.file_paths = []
        self.chunk_lengths = []
        self.total_length = 0
        self.loaded_chunk = None
        self.loaded_chunk_start_index = 0
        self.loaded_chunk_end_index = 0

        chunk_count = 0
        file_path = os.path.join(
            self.tokenized_dataset.save_path,
            f"{self.tokenized_dataset.mode}_chunk_{chunk_count}",
        )
        while os.path.exists(file_path):
            dataset = load_from_disk(file_path)
            self.file_paths.append(file_path)
            self.chunk_lengths.append(len(dataset))
            self.total_length += len(dataset)
            chunk_count += 1
            file_path = os.path.join(
                self.tokenized_dataset.save_path,
                f"{self.tokenized_dataset.mode}_chunk_{chunk_count}",
            )
        # Pre-calculate
        self.cumulative_chunk_lengths = [0]
        for chunk_length in self.chunk_lengths:
            self.cumulative_chunk_lengths.append(
                self.cumulative_chunk_lengths[-1] + chunk_length
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if not self.loaded_chunk_start_index <= idx < self.loaded_chunk_end_index:
            # Calculate the chunk index
            file_index = 0
            for i, chunk_length in enumerate(self.chunk_lengths):
                if idx < chunk_length:
                    file_index = i
                    break
                idx -= chunk_length
            self.loaded_chunk = load_from_disk(self.file_paths[file_index])
            self.loaded_chunk_start_index = sum(self.chunk_lengths[:file_index])
            self.loaded_chunk_end_index = (
                self.loaded_chunk_start_index + self.chunk_lengths[file_index]
            )
        idx -= self.loaded_chunk_start_index
        return torch.tensor(self.loaded_chunk[idx]["tokenized_text"])


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


def compute_perplexity(loss):
    return torch.exp(loss).item()


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

        self.dataset = TokenizedDataset(
            "train",
            model_name,
            max_tokens=self.max_tokens,
            save_path="/project/lt200056-opgpth/lightning/tokendata/oscar",
            use_cache=False,
        )
        self.dataset_val = TokenizedDataset(
            "val",
            model_name,
            max_tokens=self.max_tokens,
            save_path="/project/lt200056-opgpth/lightning/tokendata/oscar",
            use_cache=False,
        )
        self.tokenizer = self.dataset.tokenizer
        self.dataset = ChunkedDatasetWrapper(self.dataset)
        self.dataset_val = ChunkedDatasetWrapper(self.dataset_val)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=8,
        )
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size)

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
        self.opt.zero_grad()
        progress_bar = tqdm(
            self.dataloader,
            disable=(self.fabric.global_rank != 0),
        )

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

        self.wandb.finish()
