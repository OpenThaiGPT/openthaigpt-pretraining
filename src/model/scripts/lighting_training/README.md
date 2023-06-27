## Preprocess dataset if don't want to use raw dataset
```python src/model/scripts/lighting_training/data_preprocessing.py```

change config in ../../configuration_example/data_preprocess.yaml
note: after preprocess you should update config in dataset in term tokenized to use preprocessed dataset

## Load model

use load model to local for use in lanta
```python src/model/scripts/lighting_training/load_model.py --model_name --output_path```

model_name: model name in huggingface
output_path: path to save model

## Check Tokenizer

use check tokenizer information
```python src/model/scripts/lighting_training/tokenizer_info.py --tokenizer```

tokenizer: path to tokenizer
note: if vocab size or special token id of tokenizer don't match with model config you shold update model config 

## Train
```python src/model/scripts/lighting_training/train.py --model_name llama```

change config in ../../configuration_example/config.yaml