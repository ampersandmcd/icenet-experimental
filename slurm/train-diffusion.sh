#!/bin/bash

########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name lightning-train-big
#SBATCH --account=gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1

########## Command Lines to Run ##########

export PYTHONPATH="/data/hpcdata/users/anddon76/icenet/icenet-experimental/"
export WANDB_API_KEY="c9359bdb7a98988cb4d1b0a92098e2a8f6bda29a"

export HF_DATASETS_CACHE="/data/hpcdata/users/anddon76/.cache/huggingface"
export MPLCONFIGDIR="/data/hpcdata/users/anddon76/.cache/matplotlib"
export WANDB_CACHE_DIR="/data/hpcdata/users/anddon76/.cache/wandb"
export WANDB_DATA_DIR="/data/hpcdata/users/anddon76/.cache/wandb"

cd /data/hpcdata/users/anddon76/icenet/icenet-experimental
micromamba activate icenet-3.11
nvidia-smi

python backbone-v2/lightning-train.py --batch_size=12 --model=unet # --prediction_type="epsilon" # > results/nonlatent-train-93-only/nonlatent-train-93-only.out 2>&1 &

scontrol show job $SLURM_JOB_ID