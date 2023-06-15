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
    parser.add_argument("--seed", type=int, default=42, help="{13|21|42|87|100}")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--dataset_name_or_path", type=str, default="./tokendata")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad", type=int, default=4, help="gradient accumulate")
    parser.add_argument("--context_length", type=int, default=256, help="seq")
    parser.add_argument(
        "--training_configuration",
        type=str,
        default="src/model/configuration_example/config.yaml",
    )
    parser.add_argument(
        "--save_steps", type=int, default=10000, help="save every n steps"
    )
    parser.add_argument("--save_paths", type=str, default=".")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    training_configuration = load_hydra_config(args.training_configuration)
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        stage=args.stage,
        offload_optimizer=args.offload_opt,
        offload_parameters=args.offload_par,
        devices=args.devices,
        precision=args.precision,
        num_nodes=args.num_nodes,
        seed=args.seed,
        training_configuration=training_configuration,
        streaming=args.streaming,
        dataset_name_or_path=args.dataset_name_or_path,
        batch_size=args.batch_size,
        grad=args.grad,
        context_length=args.context_length,
        save_steps=args.save_steps,
        save_paths=args.save_paths,
    )
    trainer.train()
