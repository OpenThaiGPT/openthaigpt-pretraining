#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1 -c 128
#SBATCH --ntasks-per-node=1
#SBATCH -t  30:00:00
#SBATCH -A lt200056
#SBATCH -J test

module purge
module load Apptainer

# module load Miniconda3/22.11.1-1
# conda activate /project/lt200056-opgpth/decontamination/.conda/env

export HF_DATASETS_CACHE="/project/lt200056-opgpth/openthaigpt-refactor/.cache"

proxy_server=172.23.0.21:9631
export http_proxy=http://$proxy_server
export HTTP_PROXY=$http_proxy
export https_proxy=http://$proxy_server
export HTTPS_PROXY=$https_proxy

# apptainer run -B /scratch --home /project/lt200056-opgpth/openthaigpt-refactor image_sandbox python ./src/data/scripts/deduplication/deduplicate.py
apptainer run -B /scratch --home /project/lt200056-opgpth/openthaigpt-refactor image_sandbox python ./src/data/scripts/deduplication/deduplicate.py
apptainer run -B /scratch --home /project/lt200056-opgpth/openthaigpt-refactor image_sandbox python ./src/data/scripts/decontamination/decontaminate.py
# python ./src/data/scripts/deduplication/deduplicate.py
