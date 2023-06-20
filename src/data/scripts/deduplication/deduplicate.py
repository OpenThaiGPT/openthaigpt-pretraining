from openthaigpt_pretraining_data.deduplication.deduplication import deduplicate
from openthaigpt_pretraining_data.deduplication.generate_minhash import (
    generate_minhash,
)

import hydra


@hydra.main(version_base=None, config_path="./config", config_name="deduplication")
def main(cfg):
    generate_minhash(cfg.train_dataset, cfg.minhash, cfg.global_config)
    deduplicate(cfg.train_dataset, cfg.deduplication, cfg.global_config)


if __name__ == "__main__":
    main()  # type: ignore
