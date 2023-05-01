# Tokenizer Experiment

merge LLaMa Tokenizer with [wangchanberta](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main) Tokenizer by extent vocabulary following [working example](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md) 

## Objective
1. have faster iteration when tokenize Thai and English sentence.
2. still have similar tokenized output when running with English sentence.

## To test merged method
1.  already have LLaMa Tokenizer and [wangchanberta tokenizer](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased/tree/main)
2.  run merge_tokenizer.py
``` python3 scripts/llama_thai_tokenizer/merge_tokenizer.py```
3. merge tokenizer will save path following constants.py

## To test merged tokenizer

1.  run llama_thai_token_test.py and inference time checked
```python3 scripts/llama_thai_token_test.py```

### Results
- Text : การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่
    - Tokenized by LLaMA tokenizer: ['▁', 'ก', 'า', 'ร', '<0xE0>', '<0xB9>', '<0x83>', 'ช', '้', 'ง', 'า', 'น', 'ห', 'ล', 'ั', 'ก', 'ข', 'อ', 'ง', '▁L', 'La', 'MA', '▁', 'ค', 'ื', 'อ', 'ก', 'า', 'ร', 'ว', 'ิ', 'จ', 'ั', 'ย', 'เ', 'ก', 'ี', '่', 'ย', 'ว', 'ก', 'ั', 'บ', 'ร', 'ู', 'ป', 'แ', 'บ', 'บ', 'ภ', 'า', 'ษ', 'า', 'ท', 'ี', '่', '<0xE0>', '<0xB9>', '<0x83>', 'ห', 'ญ', '่']
    - Tokenized by English-Thai LLaMA tokenizer: ['▁', 'ก', 'า', 'ร', 'ใช้', 'ง', 'า', 'น', 'หล', 'ั', 'ก', 'ของ', '▁L', 'La', 'MA', '▁คือ', 'ก', 'า', 'ร', 'วิจัย', 'เกี่ยวกับ', 'รูปแบบ', 'ภ', 'า', 'ษ', 'า', 'ที่', 'ให', 'ญ', '่']
    - EngOnly time: 0.00027060508728027344
    - EngThai time: 0.00016260147094726562
    - time diff: 0.00010800361633300781

