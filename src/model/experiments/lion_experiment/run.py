import argparse
from openthaigpt_pretraining_model.optimizer.lion.utils import (
    seed_everything,
    Trainer,
)
import os

# https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186#issuecomment-1518694203
if os.name == "posix":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--seed", type=int, default=42, help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=256, help="seq")
    parser.add_argument("--lr", type=float, default=5e-4, help="lr")
    parser.add_argument("--max_steps", type=int, default=800, help="max steps")
    parser.add_argument("--eval_steps", type=int, default=400, help="eval steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="warmup steps")
    parser.add_argument("--use_flash", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="{gpt2|gpt2-medium|gpt2-large|gpt2-xl,cerebras/Cerebras-GPT-2.7B}",
    )
    parser.add_argument("--do_sample", default=False, action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="weight decay")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="gradient acc",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="{bfloat16|float32|float16}",
    )

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    trainer = Trainer(
        optimizer=args.optimizer,
        seed=args.seed,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        model_name=args.model_name,
        weight_decay=args.weight_decay,
        grad=args.gradient_accumulation_steps,
        lr=args.lr,
        do_sample=args.do_sample,
        use_flash=args.use_flash,
        use_checkpointing=args.use_checkpointing,
        dtype=args.dtype,
    )
    trainer.train()
