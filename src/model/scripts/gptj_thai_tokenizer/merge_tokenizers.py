from transformers import AutoTokenizer, GPT2TokenizerFast
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.constants import (
    GPT2_MERGE_DIR,
    GPTJ_MERGE_DIR,
)
import json
from typing import Dict

# load tokenizer
gptj_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt2_tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-base-thai")

# retrieve the vocabs
gptj_vocab = gptj_tokenizer.get_vocab()
gpt2_vocab = gpt2_tokenizer.get_vocab()

# Create a new vocabulary by merging vocabs
new_vocab: Dict[str, int] = {}
idx = 0
for word in gptj_vocab.keys():
    if word not in new_vocab.keys():
        new_vocab[word] = gptj_vocab[word]
        idx += 1

# Add words from second tokenizer
for word in gpt2_vocab.keys():
    if word not in new_vocab.keys():
        new_vocab[word] = idx
        idx += 1

# convert dictionary to json
new_vocab_json = json.dumps(new_vocab, ensure_ascii=False)

# write json to a file
with open("merge_vocab.json", "w", encoding="utf-8") as outfile:
    outfile.write(new_vocab_json)

# merge merged rule
with open(GPT2_MERGE_DIR, "r", encoding="utf-8") as f1, open(
    GPTJ_MERGE_DIR, "r", encoding="utf-8"
) as f2, open("new_merged_rule.txt", "w", encoding="utf-8") as out_file:
    # Ignore first line of each input file
    next(f1)
    next(f2)

    # Read the remaining lines of each file and write them to the output file
    lines = set()
    for line in f1:
        if line not in lines:
            out_file.write(line)
            lines.add(line)
    for line in f2:
        if line not in lines:
            out_file.write(line)
            lines.add(line)

merge_tokenizer = GPT2TokenizerFast(
    vocab_file="merge_vocab.json", merges_file="new_merged_rule.txt"
)

merge_tokenizer.save_pretrained("merged_GPTJ_tokenizer_hf")
