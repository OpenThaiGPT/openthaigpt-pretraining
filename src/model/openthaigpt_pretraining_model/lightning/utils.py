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
        configuration,
    ):
        if torch.cuda.get_device_name(0) == "NVIDIA A100-SXM4-40GB":
            torch.set_float32_matmul_precision("medium")  # high
        training_configuration = configuration.training
        self.wandb = None
        self.max_tokens = training_configuration.max_tokens
        self.step = 0
        self.seed = training_configuration.seed
        self.grad = training_configuration.grad
        strategy = training_configuration.strategy
        self.epochs = training_configuration.epochs
        self.save_steps = training_configuration.save_steps
        self.save_paths = training_configuration.save_paths
        if strategy == "deepspeed":
            strategy = DeepSpeedStrategy(
                stage=training_configuration.stage,
                offload_optimizer=training_configuration.offload_optimizer,
                offload_parameters=training_configuration.offload_parameters,
            )
        elif (
            training_configuration.offload_optimizer
            or training_configuration.offload_parameters
        ):
            raise NotImplementedError("offload only support for deepspeed strategy")
        self.fabric = L.Fabric(
            accelerator=training_configuration.accelerator,
            strategy=strategy,
            devices=training_configuration.num_gpus,
            precision=training_configuration.precision,
            loggers=self.wandb,
            num_nodes=training_configuration.num_nodes,
        )
        self.fabric.launch()
        print(f"device:{self.fabric.device}")
        if self.fabric.global_rank == 0:
            self.wandb = wandb.init(project="Fabric")

        self.tokenizer, self.model = load_model_and_tokenizer(
            configuration.model,
        )
        if configuration.dataset.tokenized.path is None:
            train_dataset = get_dataset(configuration.dataset.train)
            val_dataset = get_dataset(configuration.dataset.eval)
            self.dataset = DatasetWrapper(
                self.tokenizer, train_dataset, self.max_tokens
            )
            self.dataset_val = DatasetWrapper(
                self.tokenizer, val_dataset, self.max_tokens
            )
        else:
            self.dataset = load_token_dataset(
                dataset_path=configuration.dataset.tokenized.path,
                num_shards=training_configuration.num_shards,
                split=configuration.dataset.tokenized.train_split,
            )
            self.dataset_val = load_token_dataset(
                dataset_path=configuration.dataset.tokenized.path,
                num_shards=training_configuration.num_shards,
                split=configuration.dataset.tokenized.eval_split,
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=training_configuration.batch_size,
            num_workers=training_configuration.num_workers,
        )
        self.dataloader_val = DataLoader(
            self.dataset_val, batch_size=training_configuration.batch_size
        )

        self.model = self.model.to("cuda")
        self.model, self.opt = get_optimizer(
            model=self.model,
            optimizer_configuration=configuration.optimizer,
            batch_size=training_configuration.batch_size,
            offload_optimizer=training_configuration.offload_optimizer,
            offload_parameters=training_configuration.offload_parameters,
        )
        self.model, self.opt = self.fabric.setup(self.model, self.opt)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloder_val = self.fabric.setup_dataloaders(self.dataloader_val)

    def log(self, data):
        if self.wandb is not None:
            self.wandb.log(data)

    def save_checkpoint(self, epoch, loss, perplexity):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "loss": loss,
                "perplexity": perplexity,
            },
            f"{self.save_paths}_{epoch}.pt",
        )

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            print(f"=> loading checkpoint {path}")
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(f"=> no checkpoint found {path}")

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
        return (loss, perplexity)

    def train(self):
        progress_bar = tqdm(self.dataloader, disable=(self.fabric.global_rank != 0))
        self.opt.zero_grad()
        for epoch in range(self.epochs):
            for i, batch in enumerate(progress_bar):
                is_accumulating = (i + 1) % self.grad != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.train_step(batch)

                    self.fabric.backward(loss)
                    perplexity = compute_perplexity(loss)
                    self.log(
                        {"train_loss": loss.item(), "train_perplexity": perplexity}
                    )
                    progress_bar.set_description(f"loss: {loss.item():.3f}")

                if not is_accumulating:
                    self.opt.step()
                    self.opt.zero_grad()

                if (i + 1) % self.save_steps == 0:
                    (val_loss, val_perplexity) = self.val_step()
                    self.save_checkpoint(epoch, val_loss, val_perplexity)

            (val_loss, val_perplexity) = self.val_step()
            print(f"val_loss: {val_loss.item():.3f}, perplexity: {val_perplexity:.3f})")

        self.wandb.finish()
