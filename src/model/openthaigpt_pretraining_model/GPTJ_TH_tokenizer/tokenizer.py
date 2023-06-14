from transformers import PreTrainedTokenizerFast
from .merge import merge
from typing import Any, List, Union


class GPTJToken:
    def __init__(self, tokenizer_dir):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class GPT2Token:
    def __init__(self, tokenizer_dir):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    def tokenize(self, x: str) -> List[Any]:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> List[Any]:
        return self.tokenizer.encode(x)

    def decode(self, x: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(x)


class MergedToken:
    def __init__(self, tokenizer_dir_1, tokenizer_dir_2, merge_file_1, merge_file_2):
        self.tokenizer = merge(
            tokenizer_dir_1, tokenizer_dir_2, merge_file_1, merge_file_2
        )

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
