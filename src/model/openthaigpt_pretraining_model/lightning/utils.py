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
from ..models import load_model_and_tokenizer, load_lora

from lightning.fabric.strategies import DeepSpeedStrategy
import wandb
import math
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
        self.model_name = configuration.model.name
        self.wandb = None
        self.max_tokens = training_configuration.max_tokens
        self.seed = training_configuration.seed
        self.grad = training_configuration.grad
        strategy = training_configuration.strategy
        self.epochs = training_configuration.epochs
        self.start_epochs = training_configuration.start_epochs
        self.start_steps = training_configuration.start_steps
        self.eval_steps = training_configuration.eval_steps
        self.global_steps = 0
        self.save_steps = training_configuration.save_steps
        self.save_paths = training_configuration.save_paths
        self.deepspeed = False
        if strategy == "deepspeed":
            self.deepspeed = True
            strategy = DeepSpeedStrategy(
                stage=training_configuration.stage,
                offload_optimizer=training_configuration.offload_optimizer,
                offload_parameters=training_configuration.offload_parameters,
            )
            strategy.config[
                "gradient_clipping"
            ] = training_configuration.gradient_clipping
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

        with self.fabric.device:
            self.tokenizer, self.model = load_model_and_tokenizer(
                configuration.model,
                training_configuration.get("load_in_4bit", False),
                training_configuration.get("load_in_8bit", False),
            )

        if configuration.get("lora", None) is not None:
            self.model = load_lora(
                self.model,
                configuration.lora,
                self.model_name,
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
        self.model, self.opt = get_optimizer(
            model=self.model,
            optimizer_configuration=configuration.optimizer,
            batch_size=training_configuration.batch_size,
            offload_optimizer=training_configuration.offload_optimizer,
            offload_parameters=training_configuration.offload_parameters,
        )
        self.model, self.opt = self.fabric.setup(self.model, self.opt)
        if training_configuration.get("load_weight_path", False):
            self.load_checkpoint(training_configuration.load_weight_path)
            self.fabric.print(
                f"loading weight from {training_configuration.load_weight_path} success"
            )

        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)
        self.dataloader_val = self.fabric.setup_dataloaders(self.dataloader_val)

        self.decay_lr = training_configuration.decay_lr
        self.warmup_iters = training_configuration.warmup_iters
        self.lr_decay_iters = training_configuration.lr_decay_iters
        self.min_lr = training_configuration.min_lr
        self.learning_rate = configuration.optimizer.hyps.lr

    def log(self, data):
        if self.wandb is not None:
            self.wandb.log(data)

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def save_checkpoint(self, step, epoch):
        if self.deepspeed:
            self.fabric.save(
                f"{self.save_paths}/{self.model_name}_{epoch}_{step}",
                {
                    "model": self.model,
                    "_optimizer": self.opt,
                    "_global_steps": self.global_steps,
                    "local_step": step,
                    "epoch": epoch,
                },
            )
        else:
            self.fabric.save(
                f"{self.save_paths}/{self.model_name}_{epoch}_{step}.pt",
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "_global_steps": self.global_steps,
                    "local_step": step,
                    "epoch": epoch,
                },
            )
        self.fabric.barrier()

    def load_checkpoint(self, path):
        if self.deepspeed:
            state = {
                "model": self.model,
                "_optimizer": self.opt,
                "_global_steps": self.global_steps,
                "epoch": self.start_epochs,
                "local_step": self.start_steps,
            }

            self.fabric.load(path, state)

            self.start_epochs = state["epoch"]
            self.start_steps = state["local_step"]
            self.global_steps = state["_global_steps"]
        else:
            checkpoint = torch.load(path)
            with self.fabric.device:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epochs = checkpoint.get("epoch", 0)
            self.start_steps = checkpoint.get("local_step", 0)
            self.global_steps = checkpoint.get("_global_steps", 0)

        self.fabric.barrier()

    def train_step(self, batch):
        inputs = batch[HF_TOKENIZER_INPUT_IDS_NAME]
        loss = self.model(inputs, labels=inputs).loss
        return loss

    def val_step(self):
        self.model.eval()
        progress_bar = tqdm(self.dataloader_val, disable=(self.fabric.global_rank != 0))
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                inputs = batch[HF_TOKENIZER_INPUT_IDS_NAME]
                loss = self.model(inputs, labels=inputs).loss
                perplexity = compute_perplexity(loss)
                self.log({"val_loss": loss.item(), "val_perplexity": perplexity})
            progress_bar.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()
        self.fabric.barrier()
        return (loss, perplexity)

    def train(self):
        self.opt.zero_grad()
        for epoch in range(self.start_epochs, self.epochs):
            progress_bar = tqdm(self.dataloader, disable=(self.fabric.global_rank != 0))
            for i, batch in enumerate(progress_bar):
                if i < self.start_steps and epoch == self.start_epochs:
                    continue
                self.global_steps += 1

                if self.decay_lr:
                    lr = self.get_lr(self.global_steps)
                else:
                    lr = self.learning_rate
                for param_group in self.opt.param_groups:
                    param_group["lr"] = lr

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

                if (i + 1) % self.eval_steps == 0:
                    (val_loss, perplexity) = self.val_step()
                    self.fabric.print(
                        f"val_loss: {val_loss.item():.3f}, "
                        f"perplexity: {perplexity:.3f})"
                    )

                if (i + 1) % self.save_steps == 0:
                    self.fabric.print(f"Saving weights : {self.save_paths}")
                    self.save_checkpoint(i + 1, epoch)

        if self.fabric.global_rank == 0:
            self.wandb.finish()
