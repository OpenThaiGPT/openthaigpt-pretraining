import os


FILE_DIR = os.path.dirname(__file__)
LLAMA_TOKENIZER_DIR = "decapoda-research/llama-7b-hf"
THAI_SP_MODEL_DIR = f"{FILE_DIR}/thai_tokenizer/trained_bpe.model"
OUTPUT_SP_DIR = f"{FILE_DIR}/merged_tokenizer_sp"
OUTPUT_HF_DIR = f"{FILE_DIR}/merged_tokenizer_hf"
