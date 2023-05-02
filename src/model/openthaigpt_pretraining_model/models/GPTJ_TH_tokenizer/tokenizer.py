from transformers import PreTrainedTokenizerFast
from constants import GPTJ_TOKEN_DIR, NEW_TOKEN_DIR
from typing import Union, List


class GPTJToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=GPTJ_TOKEN_DIR)

    def tokenize(self, x: str) -> str:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> str:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        if isinstance(x, int):
            return self.tokenizer.decode(x)
        else:
            return [self.tokenizer.decode(i) for i in x]


class MergedToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=NEW_TOKEN_DIR)

    def tokenize(self, x: str) -> str:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> str:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        if isinstance(x, int):
            return self.tokenizer.decode(x)
        else:
            return [self.tokenizer.decode(i) for i in x]


text = "รายละเอียดและหลักเกณฑ์การคัดเลือก AI Startup Incubation​ by AIEAT"
tokens = MergedToken()
print(tokens.tokenize(text))
print(tokens.encode(text))
print([tokens.decode([token]) for token in tokens.encode(text)])
