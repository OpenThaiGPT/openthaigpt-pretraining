import os


FILE_DIR = os.path.dirname(__file__)
LLAMA_TOKENIZER_DIR = f"{FILE_DIR}/llama_tokenizer"
THAI_SP_MODEL_DIR = f"{FILE_DIR}/thai_tokenizer/thai_sentencepiece.bpe.model"
OUTPUT_SP_DIR = f"{FILE_DIR}/merged_tokenizer_sp"
OUTPUT_HF_DIR = f"{FILE_DIR}/merged_tokenizer_hf"
ENGTHAI_LLAMA_TOKENIZER_DIR = OUTPUT_HF_DIR
