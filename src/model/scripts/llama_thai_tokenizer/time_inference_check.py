import time
import argparse
from openthaigpt_pretraining_model.llama_thai_tokenizer.tokenizer import (
    LLaMaToken,
    EngThaiLLaMaToken,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llama_path",
        type=str,
        help="path to llama tokenizer",
    )
    parser.add_argument(
        "--thai_sp_path",
        type=str,
        help="path to Thai tokenizer",
    )

    args = parser.parse_args()
    llama_tokenizer = LLaMaToken(args.llama_path)
    merge_tokenizer = EngThaiLLaMaToken(args.llama_path, args.thai_sp_path)

    text = "การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
    start = time.time()
    print(f"Tokenized by LLaMA tokenizer: {llama_tokenizer.tokenize(text)}")
    t1 = time.time() - start

    start2 = time.time()
    print(f"Tokenized by English-Thai LLaMA tokenizer:{merge_tokenizer.tokenize(text)}")
    t2 = time.time() - start2

    print(f"EngOnly time: {t1}\nEngThai time: {t2}\ntime diff:{t1 - t2}")
