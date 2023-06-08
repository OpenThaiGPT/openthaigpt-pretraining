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
    """
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        num_nodes: int = 1,
        seed: int = 42,
        streaming: bool = False,
        data_path: str = "./tokendata",
        dataset_name: str = DATASET_NAME,
        batch_size: int = 8,
        grad: int = 4,
        context_length: int = 256,
        model_name: str = "llama",
        optimizer: str = "adamw",
        weight_decay: float = 1e-2,
        lr: float = 1e-4,
        vocab_size: int = 50400,
        attention_mode: str = "origin",
        checkpoint: bool = False,
        checkpoint_only_attention: bool = False,
    """
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
    parser.add_argument("--optimizer", type=str, default="adamw", help="adamw | lion")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--lr", type=float, default=5e-4, help="lr")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama",
        help="{llama | llama_hf | gptj}",
    )
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument(
        "--attention",
        type=str,
        default="origin",
        help="origin | pytorch (support only llama) | xformers (llama_hf only support origin)",  # noqa
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="gradient checkpointing",
    )
    parser.add_argument(
        "--checkpoint_only_attention",
        action="store_true",
        help="False (model) | True (self-attentions only)",
    )
    parser.add_argument(
        "src/model/configuration_example/config.yaml",
        type=str,
    )

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
        model_name=args.model_name,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        lr=args.lr,
        vocab_size=args.vocab_size,
        attention_mode=args.attention,
        checkpoint=args.checkpoint,
        checkpoint_only_attention=args.checkpoint_only_attention,
    )
    trainer.train()
