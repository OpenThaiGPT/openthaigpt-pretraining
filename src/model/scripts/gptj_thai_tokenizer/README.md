# Tokenizer Experiment

merge [GPTJ Tokenizer](https://huggingface.co/EleutherAI/gpt-j-6b/tree/main) with [gpt2_base_thai Tokenizer](https://huggingface.co/flax-community/gpt2-base-thai/tree/main) Tokenizer by extent vocabulary following [working example](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md) 

## Objective
1. extended Thai vocabulary on top of GPTJ tokenizer
2. still have similar tokenized output when running with English sentence.

## To test merged method
1. run run merge.py
``` python3 src/model/openthaigpt_pretraining_model/GPTJ_TH_tokenizer/merge.py```
2. then run merge_tokenizers.py
``` python3 scripts/gptj_thai_tokenizer/merge_tokenizers.py```
3. merge tokenizer will save path following constants.py

## To test merged tokenizer

1.  run GPTJ_th_tokenizer_test.py  
```python3 tests/model/GPTJ_th_tokenizer_test.py```

### Results
- The new tokenizer can tokenize Thai text and the tokenized outputs are the same as gpt2_base_th on Thai text
- The new tokenizer has the same tokenized outputs with GPTJ on English text


