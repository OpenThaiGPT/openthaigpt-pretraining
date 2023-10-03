#!/bin/bash
#SBATCH -p memory
#SBATCH -N 1 -c 128
#SBATCH --ntasks-per-node=1
#SBATCH -t  20:00:00
#SBATCH -A lt200056
#SBATCH -J test

ml Miniconda3

conda deactivate
conda activate oscar_collosal

export HF_DATASETS_CACHE="/project/lt200056-opgpth/openthaigpt-refactor/.cache"

python ./src/data/scripts/deduplication/deduplicate.py
python ./src/data/scripts/decontamination/decontaminate.py