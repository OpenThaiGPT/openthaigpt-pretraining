import random
import numpy as np
import torch
import hydra
from pathlib import Path


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def compute_perplexity(loss: torch.Tensor) -> float:
    return torch.exp(loss).item()


def load_hydra_config(config_file: str):
    config_file_path = Path(config_file)
    with hydra.initialize(
        config_path=str(config_file_path.parent),
        job_name=config_file_path.stem,
        version_base=None,
    ):
        config = hydra.compose(config_name=config_file_path.stem)
    return config
