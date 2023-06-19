import argparse
from openthaigpt_pretraining_model.lightning.utils import (
    Trainer,
)
from openthaigpt_pretraining_model.utils import (
    seed_everything,
    load_hydra_config,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_configuration",
        type=str,
        default="../configuration_example/config.yaml",
    )
    args = parser.parse_args()
    print(args)

    training_configuration = load_hydra_config(args.training_configuration)
    seed_everything(training_configuration.training.seed)
    trainer = Trainer(
        training_configuration=training_configuration,
    )
    trainer.train()
