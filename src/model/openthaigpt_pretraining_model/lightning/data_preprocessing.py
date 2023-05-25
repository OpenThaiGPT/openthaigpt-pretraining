from openthaigpt_pretraining_model.lightning.utils import TokenizedDataset

dataset = TokenizedDataset(mode="train", model="decapoda-research/llama-7b-hf")
dataset.save_data()  # to save tokenized data
