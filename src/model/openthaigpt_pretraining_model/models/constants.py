from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, GPTJConfig
from .falcon.model import RWForCausalLMwithCheckpointing
from .falcon.configuration_RW import RWConfig
from .llama.model import LLaMAArgs, LLaMA
from .llama_hf.model import LlamaForCausalLMWithCheckpointing
from .gptj.gptj_model_xformers import GPTJForCausalLMWithCheckpointing

TOKENIZERS = {
    "AutoTokenizer": AutoTokenizer,
    "LlamaTokenizer": LlamaTokenizer,
}

MODELS = {
    "falcon": RWForCausalLMwithCheckpointing,  # type: ignore
    "llama_hf": LlamaForCausalLMWithCheckpointing,
    "gptj": GPTJForCausalLMWithCheckpointing,
    "llama": LLaMA,
}

MODEL_CONFIGS = {
    "falcon": RWConfig,
    "llama_hf": LlamaConfig,
    "gptj": GPTJConfig,
    "llama": LLaMAArgs,
}
LORA_MODEL = {
    "llama_hf": True,
    "gptj": True,
    "falcon": True,
    "llama": False,
}
