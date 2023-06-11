from transformers import LlamaTokenizer
from .merge import merge


class LLaMaToken:
    def __init__(self, llama_tokenizer_dir):
        self.token = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)

    def decode(self, x: str) -> str:
        return self.token.decode(x)

    def encode(self, x: str) -> str:
        return self.token.encode(x)


class EngThaiLLaMaToken:
    def __init__(self, llama_tokenizer_dir, thai_sp_model_dir):
        self.token = merge(llama_tokenizer_dir, thai_sp_model_dir)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)

    def decode(self, x: str) -> str:
        return self.token.decode(x)

    def encode(self, x: str) -> str:
        return self.token.encode(x)
