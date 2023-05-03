from transformers import PreTrainedTokenizerFast
from .constants import GPTJ_TOKEN_DIR, TH_TOKEN_DIR, NEW_TOKEN_DIR
from typing import Any, List, Union


class GPTJToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=GPTJ_TOKEN_DIR)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class GPT2Token:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=TH_TOKEN_DIR)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class MergedToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=NEW_TOKEN_DIR)

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
