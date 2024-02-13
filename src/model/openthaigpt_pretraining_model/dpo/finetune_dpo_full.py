import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="merge.jsonl")
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--eval_steps", type=int, default=200)
# parser.add_argument("--flash_attn", action='store_true', default=False)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
# parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--max_len", type=int, default=1024)
parser.add_argument("--max_prompt_len", type=int, default=256)
parser.add_argument("--micro_bsz", type=int, default=2)
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
# parser.add_argument("--num_warmup_steps", type=int, default=100)
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
# parser.add_argument("--peft", action="store_true", default=False)
parser.add_argument("--save_steps", type=int, default=200)
# parser.add_argument("--tensorboard_name", type=str, default="run")
parser.add_argument("--val_size", type=float, default=0.1)
# parser.add_argument("--tokenizer_path", type=str, required=True)
# parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_name", type=str, default="OTG")
parser.add_argument("--warmup_steps", type=int, default=0)

args = parser.parse_args()

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = args.wandb_name


# -------- prompt -----------------------------------------------------------------
def generate_prompt_and_responses(data_point):
    system_prompt = "You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด"
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    data_point["input"] = (
        ""  # the jsonl file doesn't have 'input' field, so add here to avoid key error below
    )
    user_prompt = (
        (
            f"""<s>[INST] <<SYS>>
{system_prompt}
<<SYS>>

{data_point['prompt']}###{data_point['input']} [/INST]"""
        )
        if data_point["input"] != ""
        else (
            f"""<s>[INST] <<SYS>>
{system_prompt}
<<SYS>>

{data_point['prompt']} [/INST]"""
        )
    )
    return {
        "prompt": user_prompt,
        "chosen": data_point["good"] + "</s>",
        "rejected": data_point["bad"] + "</s>",
    }


# ---------------------------------------------------------------------------------

# -------- model ------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # ),
)
model.config.use_cache = False

model_ref = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------------

# -------- data -------------------------------------------------------------------
# need this block because some decoding error in the data
list_of_dicts = []

with open(args.data_path, "r") as file:
    for line in file:
        try:  # some of the lines have json decoding error
            data = json.loads(line)
            list_of_dicts.append(data)
        except:
            continue

data = Dataset.from_list(list_of_dicts)

train_val = data.train_test_split(test_size=args.val_size, shuffle=True, seed=42)

train_data = train_val["train"].shuffle().map(generate_prompt_and_responses)
val_data = train_val["test"].shuffle().map(generate_prompt_and_responses)
# ---------------------------------------------------------------------------------


# peft_config = LoraConfig(
#     r=args.lora_r,
#     lora_alpha=args.lora_r * 2,
#     lora_dropout=0.05,
#     target_modules=[
#         "q_proj",
#         "v_proj",
#         "k_proj",
#         "out_proj",
#         "fc_in",
#         "fc_out",
#         "wte",
#     ],
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# https://github.com/huggingface/trl/issues/831#issuecomment-1792611218
training_args = TrainingArguments(
    bf16=True,
    eval_steps=args.eval_steps,
    evaluation_strategy="steps",
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=args.lr,
    logging_steps=10,
    lr_scheduler_type="cosine",
    num_train_epochs=args.epoch,
    optim="paged_adamw_32bit",
    output_dir=args.output_path,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=args.micro_bsz,
    remove_unused_columns=False,
    report_to="wandb",
    save_steps=args.save_steps,
    warmup_steps=args.warmup_steps,
)

# 5. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    # peft_config=peft_config,
    max_prompt_length=args.max_prompt_len,
    max_length=args.max_len,
)

# 6. train
dpo_trainer.train()
dpo_trainer.save_model(args.output_path)
