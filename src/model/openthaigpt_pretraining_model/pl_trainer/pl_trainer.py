import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

from datasets import load_from_disk

from ..models import load_model_and_tokenizer, load_lora
from ..utils import compute_perplexity

from .optimizer import (
    get_optimizer,
    get_scheduler,
)
from ..data_wrapper import (
    HF_TOKENIZER_INPUT_IDS_NAME,
)

from .constants import (
    TRAIN_LOSS_MONITOR,
    VAL_LOSS_MONITOR,
    TRAIN_PERPLEXITY_MONITOR,
    VAL_PERPLEXITY_MONITOR,
    LR_STEP_INTERVAL,
)


class CausalModelPL(pl.LightningModule):
    def __init__(
        self,
        trainer_config=None,
        model_config=None,
        lora_config=None,
        optimizer_config=None,
        scheduler_config=None,
    ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        _, self.model = load_model_and_tokenizer(model_config)
        if lora_config is not None:
            self.model = load_lora(
                self.model,
                lora_config,
                model_config.name,
            )

    def forward(
        self,
        input_ids,
    ):
        return self.model(
            input_ids=input_ids,
            labels=input_ids,
        )

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_config,
        )
        optimizers_config = {
            "optimizer": optimizer,
            "monitor": VAL_LOSS_MONITOR,
        }
        if self.hparams.scheduler_config is not None:
            optimizers_config["lr_scheduler"] = {
                "scheduler": get_scheduler(
                    optimizer,
                    base_lr=self.hparams.optimizer_config.hyps.lr,
                    total_steps=self.hparams.trainer_config.max_steps,
                    scheduler_config=self.hparams.scheduler_config,
                ),
                "interval": LR_STEP_INTERVAL,
            }
        return optimizers_config

    def training_step(self, batch, batch_idx):
        input_ids = batch[HF_TOKENIZER_INPUT_IDS_NAME]

        output = self.forward(
            input_ids,
        )
        loss = output.loss
        perplexity = compute_perplexity(loss)
        self.log(TRAIN_LOSS_MONITOR, loss, prog_bar=True, sync_dist=True)
        self.log(TRAIN_PERPLEXITY_MONITOR, perplexity, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[HF_TOKENIZER_INPUT_IDS_NAME]

        output = self.forward(
            input_ids,
        )
        loss = output.loss
        perplexity = compute_perplexity(loss)
        self.log(VAL_LOSS_MONITOR, loss, prog_bar=True, sync_dist=True)
        self.log(VAL_PERPLEXITY_MONITOR, perplexity, prog_bar=True, sync_dist=True)


def causal_pretraining(
    name,
    model_config,
    optimizer_config,
    scheduler_config,
    trainer_config,
    lora_config,
    deepspeed_config,
    dataset_config,
    checkpointing_config,
    logging_config,
):
    """
    Args:
        name: Name of experiments
        model_config: Model configuration
        optimizer_config: Optimizer configuration
        scheduler_config: Scheduler configuration
        trainer_config: Trainer configuration
        lora_config: LORA configuration
        deepspeed_config: Deepspeed configuration
        dataset_config: Dataset configuration
        checkpointing_config: Checkpointing configuration
        logging_config: Logging configuration
    """

    torch.manual_seed(dataset_config.shuffle_seed)

    hf_train_dataset = load_from_disk(dataset_config.training_dataset).with_format(
        "torch"
    )

    train_dataloaders = torch.utils.data.DataLoader(
        hf_train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=dataset_config.num_workers,
        shuffle=True,
    )
    if dataset_config.validation_dataset:
        hf_validation_dataset = load_from_disk(
            dataset_config.validation_dataset
        ).with_format("torch")
        validation_dataloaders = torch.utils.data.DataLoader(
            hf_validation_dataset,
            batch_size=trainer_config.batch_size,
            num_workers=dataset_config.num_workers,
        )
    else:
        validation_dataloaders = None

    if trainer_config.resume_from_checkpoint:
        model = CausalModelPL.load_from_checkpoint(
            trainer_config.resume_from_checkpoint
        )
    else:
        model = CausalModelPL(
            trainer_config=trainer_config,
            model_config=model_config,
            lora_config=lora_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )
    loggers = []
    if logging_config.log_dir is not None:
        loggers.append(TensorBoardLogger(logging_config.log_dir, name=name))
    if logging_config.use_wandb:
        loggers.append(WandbLogger(name=name, offline=True))

    lr_monitor = (
        LearningRateMonitor(logging_interval=LR_STEP_INTERVAL)
        if logging_config.log_dir or logging_config.use_wandb
        else None
    )
    checkpoint_step_callback = (
        ModelCheckpoint(
            dirpath=os.path.join(checkpointing_config.save_dir, "steps"),
            filename=name + "-{epoch:03d}-{step:07d}",
            every_n_train_steps=checkpointing_config.save_steps,
            save_top_k=checkpointing_config.save_top_k_steps,
            monitor=VAL_LOSS_MONITOR,
        )
        if checkpointing_config.save_dir and checkpointing_config.save_steps
        else None
    )
    checkpoint_epoch_callback = (
        ModelCheckpoint(
            dirpath=os.path.join(checkpointing_config.save_dir, "epoch"),
            filename=name + "-{epoch:03d}",
        )
        if checkpointing_config.save_dir
        else None
    )
    callbacks = [checkpoint_step_callback, checkpoint_epoch_callback, lr_monitor]
    if deepspeed_config is None:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = DeepSpeedStrategy(
            **deepspeed_config,
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        precision=trainer_config.precision,
        max_steps=trainer_config.max_steps,
        num_nodes=trainer_config.num_nodes,
        devices=trainer_config.num_gpus,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        logger=loggers,
        log_every_n_steps=logging_config.log_steps,
        callbacks=[callback for callback in callbacks if callback],
        strategy=strategy,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=validation_dataloaders,
        ckpt_path=trainer_config.resume_from_checkpoint,
    )
    # Save last chpt
    trainer.save_checkpoint(
        os.path.join(
            checkpointing_config.save_dir,
            f"{name}-last.ckpt",
        )
    )
