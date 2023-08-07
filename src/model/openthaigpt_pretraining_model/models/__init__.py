from .constants import (
    TOKENIZERS,
    MODELS,
    MODEL_CONFIGS,
    LORA_MODEL,
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

import torch


def load_model_and_tokenizer(model_config, load_in_4bit=False, load_in_8bit=False):
    tokenizer = load_tokenizer(model_config.tokenizer)
    model = load_model(model_config, tokenizer, load_in_4bit, load_in_8bit)
    return tokenizer, model


def load_model(model_config, tokenizer=None, load_in_4bit=False, load_in_8bit=False):
    model_config_object = MODEL_CONFIGS.get(model_config.name, None)
    if model_config_object is None:
        raise NotImplementedError(f"No model name: {model_config.name}")
    model_object = MODELS.get(model_config.name, None)
    if model_object is None:
        raise NotImplementedError(f"No model name: {model_config.name}")

    model_pretrained = model_config.pretrained_model_name_or_path

    if model_pretrained is None:
        model_config_args = model_config.args
        if tokenizer is not None:
            model_config_args = {
                **model_config_args,
                "vocab_size": len(tokenizer),
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        config = model_config_object(
            **model_config.args,
        )
        model = model_object(config)
    else:
        config = model_config_object(
            **model_config.args,
        )

        if load_in_8bit and load_in_4bit:
            raise ValueError(
                "You can't load the model in 8 bits and 4 bits at the same time"
            )
        elif load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
            torch_dtype = torch.bfloat16
        else:
            quantization_config = None
            torch_dtype = None

        model = model_object.from_pretrained(
            model_pretrained,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        if tokenizer is not None and model.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size/1000**2:.1f}M parameters")
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model size requires_grad: {model_size/1000**2:.1f}M parameters")
    return model


def load_tokenizer(tokenizer_config):
    tokenizer_object = TOKENIZERS.get(tokenizer_config.tokenizer_class, None)
    if tokenizer_object is None:
        raise NotImplementedError(
            f"No tokenizer name: {tokenizer_config.tokenizer_class}"
        )
    tokenizer = tokenizer_object.from_pretrained(
        tokenizer_config.pretrained_model_name_or_path
    )
    return tokenizer


def load_lora(model, lora_config, model_name):
    lora_support = LORA_MODEL.get(model_name, False)
    if lora_support is False:
        raise NotImplementedError(f"No lora_avaiable for {model_name}")
    lora_config = lora_config.lora
    lora_config = LoraConfig(**lora_config)
    model = get_peft_model(model, lora_config)

    return model
