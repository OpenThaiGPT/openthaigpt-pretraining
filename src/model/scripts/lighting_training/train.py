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
    parser.add_argument("--accelerator", type=str, default="cuda", help="cpu | cuda")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="dp | ddp | ddp_spawn | xla | deepspeed | fsdp",
    )
    parser.add_argument("--stage", type=int, default=2, help="stage of deepspeed")
    parser.add_argument(
        "--offload_opt", action="store_true", help="offload optimizer of deepspeed"
    )
    parser.add_argument(
        "--offload_par", action="store_true", help="offload parameters of deepspeed"
    )
    parser.add_argument("--devices", type=int, default=1, help="number of GPUS")
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="32-true | 32 | 16-mixed | bf16-mixed | 64-true",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
    )
    parser.add_argument("--num_shards", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad", type=int, default=4, help="gradient accumulate")
    parser.add_argument("--context_length", type=int, default=256, help="seq")
    parser.add_argument(
        "--training_configuration",
        type=str,
        default="src/model/configuration_example/config.yaml",
    )

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    training_configuration = load_hydra_config(args.training_configuration)
    trainer = Trainer(
        training_configuration=training_configuration,
    )
    trainer.train()
