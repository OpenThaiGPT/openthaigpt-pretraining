from transformers import AutoTokenizer, LlamaTokenizer
from .falcon.model import RWForCausalLM
from .falcon.configuration_RW import RWConfig

TOKENIZERS = {
    "AutoTokenizer": AutoTokenizer,
    "LlamaTokenizer": LlamaTokenizer,
}

MODELS = {"falcon": RWForCausalLM}

MODEL_CONFIGS = {"falcon": RWConfig}
