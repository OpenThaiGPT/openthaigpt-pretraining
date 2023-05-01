from transformers import LlamaTokenizer
from .constants import LLAMA_TOKENIZER_DIR, ENGTHAI_LLAMA_TOKENIZER_DIR


class LLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)


class EngThaiLLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(ENGTHAI_LLAMA_TOKENIZER_DIR)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)


# text = "including การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
# token = EngThaiLLaMaToken()
# print(token.tokenize(text))
