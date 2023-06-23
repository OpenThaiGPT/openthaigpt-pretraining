from openthaigpt_pretraining_data.blind_pdpa import (
    blind_pdpa,
)

import hydra


@hydra.main(version_base=None, config_path="./config", config_name="blind_pdpa")
def main(cfg):
    blind_pdpa(cfg.train_dataset, cfg.blind_config)


if __name__ == "__main__":
    main()  # type: ignore
