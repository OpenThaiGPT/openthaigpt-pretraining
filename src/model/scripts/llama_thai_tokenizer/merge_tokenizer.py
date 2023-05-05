from openthaigpt_pretraining_model.llama_thai_tokenizer.merge import merge
from openthaigpt_pretraining_model.llama_thai_tokenizer.constants import (
    OUTPUT_HF_DIR,
)

tokenizer = merge()
tokenizer.save_pretrained(OUTPUT_HF_DIR)
