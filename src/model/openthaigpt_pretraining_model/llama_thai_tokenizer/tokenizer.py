from transformers import LlamaTokenizer  # type: ignore

llama_tokenizer_dir = (
    "/root/openthaigpt-pretraining/src/model/"
    "openthaigpt_pretraining_model/llama_thai_tokenizer/llama_tokenizer"
)
english_thai_llama_dir = (
    "/root/openthaigpt-pretraining/src/model/"
    "openthaigpt_pretraining_model/llama_thai_tokenizer/merged_tokenizer"
)


class LLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)


class EngThaiLLaMaToken:
    def __init__(self):
        self.token = LlamaTokenizer.from_pretrained(english_thai_llama_dir)

    def tokenize(self, x: str) -> str:
        return self.token.tokenize(x)


# text = "including การใช้งานหลักของ LLaMA คือการวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่"
# token = EngThaiLLaMaToken()
# print(token.tokenize(text))
