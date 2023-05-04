import time
from openthaigpt_pretraining_model.llama_thai_tokenizer.tokenizer import (
    LLaMaToken,
    EngThaiLLaMaToken,
)

llama_tokenizer = LLaMaToken()
EngTh_llama_tokenizer = EngThaiLLaMaToken()

text = "การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
start = time.time()
print(f"Tokenized by LLaMA tokenizer: {llama_tokenizer.tokenize(text)}")
t1 = time.time() - start

start2 = time.time()
print(
    f"Tokenized by English-Thai LLaMA tokenizer: {EngTh_llama_tokenizer.tokenize(text)}"
)
t2 = time.time() - start2

print(f"EngOnly time: {t1}\nEngThai time: {t2}\ntime diff:{t1 - t2}")
