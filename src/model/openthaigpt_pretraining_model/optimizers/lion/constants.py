import torch


MODEL_NAME = "flax-community/gpt2-base-thai"
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
DATASET_NAME = "mc4"
SPLIT_VAL = "validation"
SPLIT_TRAIN = "train"
LANGUAGE_DATASET = "th"

DTYPE_CHOICE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
