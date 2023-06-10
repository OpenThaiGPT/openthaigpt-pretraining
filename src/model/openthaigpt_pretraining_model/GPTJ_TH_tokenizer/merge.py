from transformers import AutoTokenizer, GPT2TokenizerFast
import json
import os
from typing import Dict


def merge(tokenizer_dir_1, tokenizer_dir_2, merge_file_1, merge_file_2):
    # load tokenizer
    gptj_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_1)
    gpt2_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_2)

    # retrieve the vocabs
    gptj_vocab = gptj_tokenizer.get_vocab()
    gpt2_vocab = gpt2_tokenizer.get_vocab()

    # create folder to keep new vocab and merge
    folder_path = "./temp"
    os.mkdir(folder_path)

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
    vocab_file_path = os.path.join(folder_path, "merge_vocab.json")

    with open(vocab_file_path, "w", encoding="utf-8") as outfile:
        outfile.write(new_vocab_json)

    # merge merged rule
    merge_file_path = os.path.join(folder_path, "new_merged_rule.txt")

    with open(merge_file_1, "r", encoding="utf-8") as f1, open(
        merge_file_2, "r", encoding="utf-8"
    ) as f2, open(merge_file_path, "w", encoding="utf-8") as out_file:
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
        vocab_file=vocab_file_path, merges_file=merge_file_path
    )

    return merge_tokenizer
