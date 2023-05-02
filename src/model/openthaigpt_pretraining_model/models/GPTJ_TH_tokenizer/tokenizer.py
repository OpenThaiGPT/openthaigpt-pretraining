from transformers import PreTrainedTokenizerFast
from constants import GPTJ_TOKENIZER_DIR, OUTPUT_TOKENIZER_DIR


class GPTJToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=GPTJ_TOKENIZER_DIR)

    def tokenize(self, x: str) -> str:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> str:
        return self.tokenizer.encode(x)

    def decode(self, x: int) -> int:
        return self.tokenizer.decode(x)


class MergedToken:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=OUTPUT_TOKENIZER_DIR)

    def tokenize(self, x: str) -> str:
        return self.tokenizer.tokenize(x)

    def encode(self, x: str) -> str:
        return self.tokenizer.encode(x)

    def decode(self, x: int) -> int:
        return self.tokenizer.decode(x)


# text = "รายละเอียดและหลักเกณฑ์การคัดเลือก AI Startup Incubation​ by AIEAT"
# tokens = MergedToken()
# print(tokens.tokenize(text))
# print(tokens.encode(text))
# print([tokens.decode([token]) for token in tokens.encode(text)])
