#!/bin/bash

########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name rollout-diffusion-amsr-rothera
#SBATCH --account=gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1

########## Command Lines to Run ##########

export PYTHONPATH="/data/hpcdata/users/anddon76/icenet/icenet-experimental/"
export WANDB_API_KEY="c9359bdb7a98988cb4d1b0a92098e2a8f6bda29a"

export TRANSFORMERS_CACHE="/data/hpcdata/users/anddon76/.cache/huggingface"
export HF_DATASETS_CACHE="/data/hpcdata/users/anddon76/.cache/huggingface"
export MPLCONFIGDIR="/data/hpcdata/users/anddon76/.cache/matplotlib"
export WANDB_CACHE_DIR="/data/hpcdata/users/anddon76/.cache/wandb"
export WANDB_DATA_DIR="/data/hpcdata/users/anddon76/.cache/wandb"

cd /data/hpcdata/users/anddon76/icenet/icenet-experimental/amsr
micromamba activate icenet-3.11
nvidia-smi

python lightning_rollout.py --name="lowernoise_sep2021_30day_32member" --start_date=20210901

scontrol show job $SLURM_JOB_ID