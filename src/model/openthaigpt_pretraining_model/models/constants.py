from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
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
    "llama_hf": LlamaForCausalLM,
    "llama_hf_gradient_attention_checkpoint": LlamaForCausalLMNewCheckpoint,
}

MODEL_CONFIGS = {
    "falcon": RWConfig,
    "llama_hf": LlamaConfig,
    "llama_hf_gradient_attention_checkpoint": LlamaConfig,
}
