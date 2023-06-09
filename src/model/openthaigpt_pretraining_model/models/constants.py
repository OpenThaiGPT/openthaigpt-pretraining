from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, GPTJForCausalLM, GPTJConfig
from .falcon.model import RWForCausalLM
from .falcon.configuration_RW import RWConfig
from .llama_hf.model import LlamaForCausalLMNewCheckpoint

# from .llama.model import make_model_llama

TOKENIZERS = {
    "AutoTokenizer": AutoTokenizer,
    "LlamaTokenizer": LlamaTokenizer,
}

MODELS = {
    "falcon": RWForCausalLM,
    "llama_hf": LlamaForCausalLMNewCheckpoint,
    "gptj": GPTJForCausalLM,
}

MODEL_CONFIGS = {
    "falcon": RWConfig,
    "llama_hf": LlamaConfig,
    "gptj": GPTJConfig,
}
