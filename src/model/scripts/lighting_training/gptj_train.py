from tqdm import tqdm
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPTJConfig,
    GPTJForCausalLM,
)
from openthaigpt_pretraining_model.lightning.utils import DatasetWrapper, seed_everything

cfg = GPTJConfig(
    vocab_size=50400,
    n_positions=2048,
    n_embd=1536,
    n_layer=12,
    n_head=8,
    rotary_dim=64,
    n_inner=None,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=50256,
    eos_token_id=50256,
    tie_word_embeddings=False,
)
seed_everything(69)
dataset = DatasetWrapper("train", "EleutherAI/gpt-j-6B")
train_loader = DataLoader(dataset, batch_size=2, num_workers=2)

fabric = L.Fabric(accelerator="cuda", devices=2, precision="16-mixed", strategy="ddp")
fabric.launch()
model = GPTJForCausalLM(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()
model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(train_loader)

for idx, batch in enumerate(tqdm(dataloader)):
    loss = model(batch, labels=batch).loss
    fabric.backward(loss)

    optimizer.step()
    optimizer.zero_grad()
