from transformers import PreTrainedTokenizerFast
from constants import GPTJ_TOKEN_DIR, NEW_TOKEN_DIR
from typing import Any, List


class GPTJToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=GPTJ_TOKEN_DIR)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: int) -> str:
        return self.tokenizer.decode(x)


class MergedToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=NEW_TOKEN_DIR)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: int) -> str:
        return self.tokenizer.decode(x)


text = "รายละเอียดและหลักเกณฑ์การคัดเลือก AI Startup Incubation​ by AIEAT"
tokens = MergedToken()
print(tokens.tokenize(text))
print(tokens.encode(text))
print([tokens.decode([token]) for token in tokens.encode(text)])
