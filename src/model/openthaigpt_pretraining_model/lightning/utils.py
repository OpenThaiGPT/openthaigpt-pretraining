from tqdm import tqdm
import lightning as L
import torch
from torch.utils.data import DataLoader

from ..utils import compute_perplexity
from ..data_wrapper import (
    DatasetWrapper,
    load_token_dataset,
    HF_TOKENIZER_INPUT_IDS_NAME,
)
from ..datasets import get_dataset
from ..optimizers import get_optimizer
from ..models import load_model_and_tokenizer

from lightning.fabric.strategies import DeepSpeedStrategy
import wandb
import os

# os.environ["WANDB_API_KEY"] = "<your-api-key>"
os.environ["WANDB_MODE"] = "offline"


class Trainer:
    def __init__(
        self,
        training_configuration,
    ):
        if torch.cuda.get_device_name(0) == "NVIDIA A100-SXM4-40GB":
            torch.set_float32_matmul_precision("medium")  # high
        config = training_configuration.training
        self.wandb = None
        self.max_tokens = config.max_tokens
        self.step = 0
        self.seed = config.seed
        self.grad = config.grad
        if config.strategy == "deepspeed":
            strategy = DeepSpeedStrategy(
                stage=config.stage,
                offload_optimizer=config.offload_optimizer,
                offload_parameters=config.offload_parameters,
            )
        elif config.offload_optimizer or config.offload_parameters:
            raise NotImplementedError("offload only support for deepspeed strategy")
        self.fabric = L.Fabric(
            accelerator=config.accelerator,
            strategy=strategy,
            devices=config.devices,
            precision=config.precision,
            loggers=self.wandb,
            num_nodes=config.num_nodes,
        )
        self.fabric.launch()
        print(f"device:{self.fabric.device}")
        if self.fabric.global_rank == 0:
            self.wandb = wandb.init(project="Fabric")

        self.tokenizer, self.model = load_model_and_tokenizer(
            training_configuration.model,
        )
        if training_configuration.dataset.tokenized.path is None:
            train_dataset = get_dataset(training_configuration.dataset.train)
            val_dataset = get_dataset(training_configuration.dataset.eval)
            self.dataset = DatasetWrapper(
                self.tokenizer, train_dataset, self.max_tokens
            )
            self.dataset_val = DatasetWrapper(
                self.tokenizer, val_dataset, self.max_tokens
            )
        else:
            self.dataset = load_token_dataset(
                dataset_path=training_configuration.dataset.tokenized.path,
                num_shards=config.num_shards,
                split=training_configuration.dataset.tokenized.train_split,
            )
            self.dataset_val = load_token_dataset(
                dataset_path=training_configuration.dataset.tokenized.path,
                num_shards=config.num_shards,
                split=training_configuration.dataset.tokenized.eval_split,
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=config.batch_size)

        self.model = self.model.to("cuda")
        self.model, self.opt = get_optimizer(
            model=self.model,
            optimizer_configuration=training_configuration.optimizer,
            batch_size=config.batch_size,
            offload_optimizer=config.offload_optimizer,
            offload_parameters=config.offload_parameters,
        )
        self.model, self.opt = self.fabric.setup(self.model, self.opt)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloder_val = self.fabric.setup_dataloaders(self.dataloader_val)

    def log(self, data):
        if self.wandb is not None:
            self.wandb.log(data)

    def train_step(self, batch):
        inputs = batch[HF_TOKENIZER_INPUT_IDS_NAME]
        loss = self.model(inputs, labels=inputs).loss
        return loss

    def val_step(self):
        self.model.eval()
        progress_bar = tqdm(self.dataloader_val)
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                inputs = batch[HF_TOKENIZER_INPUT_IDS_NAME]
                loss = self.model(inputs, labels=inputs).loss
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

        self.wandb.finish()
