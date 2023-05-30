from openthaigpt_pretraining_model.lightning.utils import (
    # ChunkedDatasetWrapper,
    TokenizedDataset,
)

# from torch.utils.data import IterableDataset, Dataset, DataLoader
# from tqdm import tqdm


dataset = TokenizedDataset(
    mode="train",
    model_or_path="/home/swongpra/thai_llama_tokenizer",
    max_tokens=2048,
    save_path="/project/lt200056-opgpth/lightning/tokendata/oscar",
    chunk_size=1024 * 1024,
    batch_size=25000,
    num_proc=128,
    use_cache=True,
    dataset_name="oscar",
    dataset_dir="unshuffled_deduplicated_th",
)
dataset.tokenize_data()
# dataset = ChunkedDatasetWrapper(dataset)
# dataloader = DataLoader(dataset, num_workers=8)
# for i, data in enumerate(tqdm(dataloader)):
#     # print(data)
#     print(i)
