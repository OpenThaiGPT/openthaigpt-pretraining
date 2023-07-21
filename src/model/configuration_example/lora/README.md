# Config

- r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters.
- alpha: LoRA scaling factor.
- bias: Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
- task_type: task of model
- target_modules: The modules (for example, attention blocks) to apply the LoRA update matrices.
- modules_to_save: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
- layers_to_transform: List of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed.
- layers_pattern: Pattern to match layer names in target_modules, if layers_to_transform is specified. By default PeftModel will look at common layer pattern (layers, h, blocks, etc.), use it for exotic and custom models.


read more on reference https://huggingface.co/docs/peft/conceptual_guides/lora