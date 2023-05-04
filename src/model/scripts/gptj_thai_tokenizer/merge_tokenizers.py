from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.merge import merge
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.constants import (
    OUTPUT_HF_DIR,
)

tokenizer = merge()
tokenizer.save_pretrained(OUTPUT_HF_DIR)
