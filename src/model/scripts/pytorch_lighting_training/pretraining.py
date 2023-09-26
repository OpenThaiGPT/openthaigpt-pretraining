import hydra
from openthaigpt_pretraining_model.pl_trainer import causal_pretraining


@hydra.main(
    version_base=None,
    config_path="./pl_configurations",
    config_name="config",
)
def pretraining(cfg):
    causal_pretraining(
        name=cfg.name,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.get("scheduler", None),
        trainer_config=cfg.trainer,
        lora_config=cfg.get("lora", None),
        deepspeed_config=cfg.get("deepspeed", None),
        dataset_config=cfg.dataset,
        checkpointing_config=cfg.checkpointing,
        logging_config=cfg.logging,
    )


if __name__ == "__main__":
    pretraining()
