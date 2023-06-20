import argparse
from openthaigpt_pretraining_model.lightning.utils import (
    Trainer,
)
from openthaigpt_pretraining_model.utils import seed_everything

import hydra

@hydra.main(version_base=None, config_path="../../configuration_example/", config_name="config")
def main(cfg):
    seed_everything(cfg.training.seed)
    trainer = Trainer(
        training_configuration=cfg,
    )
    trainer.train()

if __name__ == "__main__":
    main() # type: ignore
