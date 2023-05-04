from transformers import LlamaTokenizer
from .constants import LLAMA_TOKENIZER_DIR
from .merge import merge


class LLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)

    def decode(self, x: str) -> str:
        return self.token.decode(x)

    def encode(self, x: str) -> str:
        return self.token.encode(x)


class EngThaiLLaMaToken:
    def __init__(self):
        self.token = merge()

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)

    def decode(self, x: str) -> str:
        return self.token.decode(x)

    def encode(self, x: str) -> str:
        return self.token.encode(x)
