# Falcon model

## How to use

```python
import torch
from transformers import AutoTokenizer
from openthaigpt_pretraining_model.models.falcon.model import RWForCausalLM

model_name = "tiiuae/falcon-7b"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
model = RWForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to(device)

text = "Q: Hi, how are you?\nA:"
encoding = tokenizer(text, return_tensors="pt")
del encoding["token_type_ids"]
encoding = {k: v.to(device) for k, v in encoding.items()}

output = model.generate(**encoding, max_length=200, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0]))
```
