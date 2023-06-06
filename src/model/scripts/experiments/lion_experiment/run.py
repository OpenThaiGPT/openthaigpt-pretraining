import argparse
from openthaigpt_pretraining_model.utils import (
    seed_everything,
    load_hydra_config,
)
import os
from contextlib import nullcontext
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm

from transformers import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from openthaigpt_pretraining_model.models.nanoGPT.model import make_model, _attn_wrapper
from openthaigpt_pretraining_model.data_wrapper import DatasetWrapper
from openthaigpt_pretraining_model.optimizers import get_optimizer
from openthaigpt_pretraining_model.datasets import get_dataset
from openthaigpt_pretraining_model.datasets.constants import SPLIT_TRAIN, SPLIT_VAL

# https://github.com/d8ahazard/sd_dreambooth_extension/pull/1186#issuecomment-1518694203
if os.name == "posix":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

_attn_orig = GPT2Attention._attn

MODEL_NAME = "flax-community/gpt2-base-thai"
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
DATASET_NAME = "mc4"

DTYPE_CHOICE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def closest_power_of_2(x):
    return 2 ** (x - 1).bit_length()


@torch.no_grad()
def do_eval(model, loader_val, ctx, device):
    val_loss = 0.0
    c_1 = 0
    for i1, batch1 in enumerate(loader_val):
        batch1 = batch1.to(device)
        with ctx:
            loss1 = model(batch1, labels=batch1).loss
            val_loss = float(val_loss) + float(loss1.item())
        c_1 += 1
    # print(f"loss_val : {(val_loss / c_1):.3f}")
    return val_loss / c_1


def get_torch_context(dtype: str):
    device_type = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # for later use in torch.autocast

    if dtype not in DTYPE_CHOICE.keys():
        raise NotImplementedError(
            f"dtype: {dtype} is not available. Only supports bfloat16|float32|float16"
        )

    ptdtype = DTYPE_CHOICE[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
    )
    return ctx


class Trainer:
    def __init__(
        self,
        training_configuration,
        seed,
        batch_size,
        context_length,
        max_steps,
        eval_steps,
        warmup_steps,
        model_name,
        grad,
        do_sample,
        use_flash,
        use_checkpointing,
        dtype: str,
        use_rotary,
    ):
        self.max_tokens = context_length
        self.grad = grad
        self.step = 0
        self.max_steps = max_steps
        self.seed = seed
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.do_sample = do_sample
        tokenizer = GPT2TokenizerFast.from_pretrained(
            MODEL_NAME,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
        )
        dataset_train = get_dataset(DATASET_NAME, split=SPLIT_TRAIN, shuffle=True)
        dataset_val = get_dataset(DATASET_NAME, split=SPLIT_VAL)
        self.dataset = DatasetWrapper(tokenizer, dataset_train, self.max_tokens)
        self.dataset_val = DatasetWrapper(tokenizer, dataset_val, self.max_tokens)
        self.use_flash = use_flash
        self.use_checkpointing = use_checkpointing
        self.use_rotary = use_rotary
        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
        )

        self.loader_val = DataLoader(self.dataset_val, batch_size=batch_size)

        self.backend = "nccl"

        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.backend)
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.ctx = get_torch_context(dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
        self.model = make_model(
            model_name,
            self.max_tokens,
            self.tokenizer,
            self.use_flash,
            self.use_checkpointing,
            self.device,
            self.use_rotary,
        )
        self.model, self.opt = get_optimizer(
            self.model,
            optimizer_configuration=training_configuration.optimizer,
        )
        self.model = torch.compile(self.model)  # type: ignore
        if self.ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

    def train_step(self, batch):
        batch = batch.to(self.device)
        with self.ctx:
            loss = self.model(batch, labels=batch).loss
            loss = loss / self.grad
        self.scaler.scale(loss).backward()
        return loss

    def val_step(self):
        self.model.eval()
        prog = tqdm(self.loader_val)
        for i, batch in enumerate(prog):
            batch = batch.to(self.device)
            with self.ctx:
                loss = self.model(batch, labels=batch).loss
                loss = loss / self.grad

            prog.set_description(f"loss_val: {loss.item():.3f}")
        self.model.train()

        return loss

    def generate_samples(self, n_samples=8):
        GPT2Attention._attn = _attn_orig  # back to faster but more memory consuming
        model = self.model
        x = torch.tensor([[self.tokenizer.eos_token_id]] * n_samples).to(self.device)
        t0 = time.time()
        model.eval()
        y = model.generate(
            inputs=x,
            max_length=self.max_tokens,
            do_sample=True,
        ).tolist()
        model.train()
        t1 = time.time()
        t = [self.tokenizer.decode(z) for z in y]
        for u in range(len(t)):
            print("samples = ", t[u])
        print(f"Generated in {t1-t0:.3f}s")
        if self.use_flash:
            GPT2Attention._attn = _attn_wrapper

    def train(self):
        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            if self.ddp:
                self.model.require_backward_grad_sync = i % self.grad != 0

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")

            if i % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if i % self.eval_steps == 0 and i != 0:
                print("Step =", self.step)
                # loss_val = self.val_step()
                self.model.eval()
                val_loss = do_eval(self.model, self.loader_val, self.ctx, self.device)
                self.model.train()
                print(f"loss_val : {val_loss:.3f}")
                if self.do_sample:
                    self.generate_samples(6)

            self.grad = max(1, closest_power_of_2(i + 1) // 32)
            if self.step > self.max_steps:
                break

        if self.ddp:
            destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_configuration", type=str, default="./lion_optimizer_experiment.yml"
    )
    parser.add_argument("--seed", type=int, default=42, help="{13|21|42|87|100}")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=256, help="seq")
    parser.add_argument("--max_steps", type=int, default=800, help="max steps")
    parser.add_argument("--eval_steps", type=int, default=400, help="eval steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="warmup steps")
    parser.add_argument("--use_flash", default=False, action="store_true")
    parser.add_argument("--use_rotary", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="{gpt2|gpt2-medium|gpt2-large|gpt2-xl,cerebras/Cerebras-GPT-2.7B}",
    )
    parser.add_argument("--do_sample", default=False, action="store_true")
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
    training_configuration = load_hydra_config(args.training_configuration)
    trainer = Trainer(
        training_configuration=training_configuration,
        seed=args.seed,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        model_name=args.model_name,
        grad=args.gradient_accumulation_steps,
        do_sample=args.do_sample,
        use_flash=args.use_flash,
        use_checkpointing=args.use_checkpointing,
        dtype=args.dtype,
        use_rotary=args.use_rotary,
    )
    trainer.train()
