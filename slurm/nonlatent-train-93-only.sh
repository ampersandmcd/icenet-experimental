#!/bin/bash

########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name nonlatent-train-93-only
#SBATCH --account=gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1

########## Command Lines to Run ##########

export PYTHONPATH="/data/hpcdata/users/anddon76/icenet/icenet-experimental/"
cd /data/hpcdata/users/anddon76/icenet/icenet-experimental
# mamba activate icenet-3.11

python3 backbone/nonlatent-train-93-only.py # > results/nonlatent-train-93-only/nonlatent-train-93-only.out 2>&1 &

scontrol show job $SLURM_JOB_ID