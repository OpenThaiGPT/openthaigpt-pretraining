from .constants import TOKENIZERS, MODELS, MODEL_CONFIGS


def load_model_and_tokenizer(model_config):
    tokenizer = load_tokenizer(model_config.tokenizer)
    model = load_model(model_config, tokenizer)
    return tokenizer, model


def load_model(model_config, tokenizer=None):
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
        model = model_object.from_pretrained(model_pretrained)
        if tokenizer is not None and model.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    
    use_checkpointing = model_config.args.get('use_checkpointing', False)
    checkpoint_only_attention = model_config.args.get('checkpoint_only_attention', False)
    if use_checkpointing:
        model.gradient_checkpointing_enable()
        print("use gradient checkpointing")
        if checkpoint_only_attention:
            print("use gradient checkpointing only attentions")

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
