import os

FILE_DIR = os.path.dirname(__file__)

GPT2_REPO = "flax-community/gpt2-base-thai"
GPTJ_REPO = "EleutherAI/gpt-j-6B"
GPT2_LOCAL_DIR = f"{FILE_DIR}/GPT2_merge_rule/"
GPTJ_LOCAL_DIR = f"{FILE_DIR}/GPTJ_merge_rule/"
GPT2_MERGE_DIR = f"{FILE_DIR}/GPT2_merge_rule/merges.txt"
GPTJ_MERGE_DIR = f"{FILE_DIR}/GPTJ_merge_rule/merges.txt"
OUTPUT_HF_DIR = f"{FILE_DIR}/merged_GPTJ_tokenizer_hf"
NEW_TOKEN_DIR = f"{FILE_DIR}/merged_GPTJ_tokenizer_hf/tokenizer.json"
