from .constants import TOKENIZERS, MODELS, MODEL_CONFIGS


def load_model(model_config):
    tokenizer_object = TOKENIZERS.get(model_config.tokenizer.tokenizer_class, None)
    if tokenizer_object is None:
        raise NotImplementedError(
            f"No tokenizer name: {model_config.tokenizer.tokenizer_class}"
        )
    model_config_object = MODEL_CONFIGS.get(model_config.name, None)
    if model_config_object is None:
        raise NotImplementedError(f"No model name: {model_config.name}")
    model_object = MODELS.get(model_config.name, None)
    if model_object is None:
        raise NotImplementedError(f"No model name: {model_config.name}")

    tokenizer = tokenizer_object.from_pretrained(model_config.tokenizer.pretrained)
    model_pretrained = model_config.pretrained
    if model_pretrained is None:
        config = model_config_object(
            **model_config.args,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = model_object(config)
    else:
        model = model_object.from_pretrained(model_pretrained)
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
