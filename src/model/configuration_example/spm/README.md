# Config

- output_path: folder to save tokenizer
- vocab_size: vocab size of tokenizer
- streaming: streaming mode
- load_dataset_path: dataset name for load from huggingface don't need if have load_dataset_local_path
- load_dataset_name: split of dataset use for load_dataset_path
- load_dataset_local_path: path to load dataset from local if null will use load_dataset_path instead
- load_dataset_data_type: format type of dataset (ex: json, csv) if null will be huggingface format
- large_corpus: use true when want to train large dataset
- mode: mode of train tpkenizer sentencepiece (spm: sentencepiece, bpe: byte pair encoding)
