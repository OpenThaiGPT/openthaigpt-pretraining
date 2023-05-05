from transformers import AutoTokenizer
from .merge import merge
from typing import Any, List, Union


class GPTJToken:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class GPT2Token:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-base-thai")

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class MergedToken:
    def __init__(self):
        self.tokenizer = merge()

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


# text = "รายละเอียดและหลักเกณฑ์การคัดเลือก AI Startup Incubation by AIEAT"
# tokens = MergedToken()
# print(tokens.tokenize(text))
# print(tokens.encode(text))
# print([tokens.decode([token]) for token in tokens.encode(text)])
