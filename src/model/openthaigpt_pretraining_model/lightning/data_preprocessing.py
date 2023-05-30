from openthaigpt_pretraining_model.lightning.utils import (
    # ChunkedDatasetWrapper,
    TokenizedDataset,
)

dataset = TokenizedDataset(
    mode="train",
    model_or_path="/path/to/model",
    max_tokens=2048,
    save_path="/path/to/save/data",
    chunk_size=1024 * 1024,
    batch_size=25000,
    num_proc=128,
    use_cache=True,
    dataset_name="oscar",
    dataset_dir="unshuffled_deduplicated_th",
)
dataset.tokenize_data()
