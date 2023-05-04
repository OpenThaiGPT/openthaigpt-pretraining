from tqdm import tqdm
import lightning as L
import torch
from torch.utils.data import DataLoader
from openthaigpt_pretraining_model.models.llama.model import (
    ModelArgs,
    Transformer,
    ORIGIN_ATTENTION_MODE,
)
from openthaigpt_pretraining_model.lightning.utils import DatasetWrapper

CFG = ModelArgs(
    dim=512,
    n_layers=8,
    n_heads=8,
    vocab_size=-1,
    multiple_of=256,
    norm_eps=1e-5,
    max_batch_size=32,
    max_seq_len=2048,
    attention_mode=ORIGIN_ATTENTION_MODE,
)

dataset = DatasetWrapper("train")
train_loader = DataLoader(dataset, batch_size=2, num_workers=2)

fabric = L.Fabric(accelerator="cuda", devices=2, precision="16-mixed", strategy="ddp")
fabric.launch()
model = Transformer(CFG)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()
model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(train_loader)

for idx, batch in enumerate(tqdm(dataloader)):
    loss = model(batch, labels=batch).loss
    fabric.backward(loss)

    optimizer.step()
    optimizer.zero_grad()
