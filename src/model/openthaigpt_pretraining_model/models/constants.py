from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, GPTJConfig
from .falcon.model import RWForCausalLM
from .falcon.configuration_RW import RWConfig
from .llama.model import LLaMAArgs, LLaMA
from .llama_hf.model import LlamaForCausalLMNewCheckpoint
from .gptj.gptj_model_xformers import change_attn, GPTJForCausalLMWithCheckpointing
from peft import LoraConfig, get_peft_model

TOKENIZERS = {
    "AutoTokenizer": AutoTokenizer,
    "LlamaTokenizer": LlamaTokenizer,
}

MODELS = {
    "falcon": RWForCausalLM,  # type: ignore
    "llama_hf": LlamaForCausalLMNewCheckpoint,
    "gptj": GPTJForCausalLMWithCheckpointing,
    "llama": LLaMA,
}

MODEL_CONFIGS = {
    "falcon": RWConfig,
    "llama_hf": LlamaConfig,
    "gptj": GPTJConfig,
    "llama": LLaMAArgs,
}

ATTENTION_MODE = {
    "gptj": change_attn,
}

GRADIENT_CHECKPOINTING = {
    "llama_hf": True,
    "gptj": True,
    "llama": False,
}

LORA_CONFIG = {
    "llama_hf": LoraConfig,
    "gptj": LoraConfig,
}
LORA_MODEL = {
    "llama_hf": get_peft_model,
    "gptj": get_peft_model,
}
