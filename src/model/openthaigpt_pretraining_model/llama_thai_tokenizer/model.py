from transformers import LlamaTokenizer
from pathlib import Path

llama_tokenizer_dir = "./llama_thai_tokenizer"
english_thai_llama_dir = "./merged_tokenizer"

class LLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
        pass

    def predict(self, x: str) -> str:
        return self.token.tokenize(x)


class EngThaiLLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(english_thai_llama_dir)
        pass

    def predict(self, x: str) -> str:
        return self.token.tokenize(x)


token = LLaMaToken()
text = "including การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
print(token.predict(text))
