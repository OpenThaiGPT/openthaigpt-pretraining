# Tokenizer Experiment

merge [GPTJ Tokenizer](https://huggingface.co/EleutherAI/gpt-j-6b/tree/main) with [gpt2_base_thai Tokenizer](https://huggingface.co/flax-community/gpt2-base-thai/tree/main) Tokenizer by extent vocabulary following [working example](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md) 

## Objective
1. extended Thai vocabulary on top of GPTJ tokenizer
2. still have similar tokenized output when running with English sentence.

## To test merged method
1.  already have [GPTJ Tokenizer](https://huggingface.co/EleutherAI/gpt-j-6b/tree/main) and [gpt2_base_thai Tokenizer](https://huggingface.co/flax-community/gpt2-base-thai/tree/main)
2.  run merge_tokenizers.py
``` python3 scripts/gptj_thai_tokenizer/merge_tokenizers.py```
3. merge tokenizer will save path following constants.py

## To test merged tokenizer

1.  run llama_thai_token_test.py and 
```python3 tests/model/GPTJ_th_tokenizer_test.py```

### Results
- The new tokenizer can tokenize Thai text and the tokenized outputs are the same as gpt2_base_th on Thai text
- The new tokenizer has the same tokenized outputs with GPTJ on English text


