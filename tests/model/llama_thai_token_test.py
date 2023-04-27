from transformers import LlamaTokenizer
import time

llama_tokenizer_dir = "/root/openthaigpt-pretraining/src/model/openthaigpt_pretraining_model/llama_thai_tokenizer/llama_tokenizer"
EngTh_llama_dir = "/root/openthaigpt-pretraining/src/model/openthaigpt_pretraining_model/llama_thai_tokenizer/merged_tokenizer"

llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
EngTh_llama_tokenizer = LlamaTokenizer.from_pretrained(EngTh_llama_dir)

text = "LLaMa สำหรับ use of LLaMA is research on large language models, including การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
start = time.time()
print(f"Tokenized by LLaMA tokenizer: {llama_tokenizer.tokenize(text)}")
time1 = time.time() - start

start2 = time.time()
print(
    f"Tokenized by English-Thai LLaMA tokenizer: {EngTh_llama_tokenizer.tokenize(text)}"
)
time2 = time.time() - start2

print(
    f"time llama tokenizer: {time1}\ntime english-thai:    {time2}\ntime diff: {time1 - time2}"
)
