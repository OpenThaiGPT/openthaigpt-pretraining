## Install

```bash
ml Miniconda3

conda create -n <your env name> python=3.10

conda activate <your env name>

pip install -e ./src/core
pip install -e ./src/model
```

## Setup config

Setup config in src\model\configuration_example

## Preprocess Dataset

```bash
sbatch src\model\scripts\sh_file\lighting_training\dataprocess.sh
```

after you proprecess dataset you should add dataset path after preprocess to dataset config

## Train

```bash
sbatch src\model\scripts\sh_file\lighting_training\train.sh
```