# OpenThaiGPT tokenizer training pipeline

## Description

The OpenThaiGPT tokenizer training pipeline is a pipeline for training tokenizer specially for Thai language.

## how to train

1. you need to config argument first [configuration_example/spm/training_v1.yaml](../../configuration_example/spm/training_v1.yaml)

   - output_path (str): The path and prefix to use when saving the trained tokenizer.
   - vocab_size (int): The size of the vocabulary to use when training the tokenizer. Defaults to the number of available CPU cores.
   - is_slurm (bool): Whether the code is running on a Slurm cluster. Defaults to False.
   - load_dataset_path (str): The name of the Hugging Face dataset to load. Defaults to "oscar".
   - load_dataset_name (str): The name of the dataset split to use. Defaults to "unshuffled_deduplicated_th".
   - load_dataset_local_path (str): The path to a local directory containing the input data. If specified, the Hugging Face dataset is not used. Defaults to None.
   - load_dataset_data_type (str): The file type of the input data if using a local directory
   - large_corpus (bool): Whether to use a large corpus. Defaults to False.
   - mode(spm | bpe): type of tokenizer

2. train tokenizer by running following script, don't forget to check <output_path> and <vocab_size>

   ```bash
   python train.py
   ```

3. View the Results
   result will be in <output_path>
