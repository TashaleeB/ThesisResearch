#!/bin/bash
#SBATCH -J optv9nowedge
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 48:00:00
#SBATCH --gres gpu:k80:4
#SBATCH -o /pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/opt_v9_nowedge.log
#SBATCH --mail-type ALL
#SBATCH --mail-user tashalee@sas.upenn.edu

# Start GPU environment
#conda activate dl-gpu
conda activate hp_opt

# echo the starting time
date

# Compile and run job
python /pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/opt_versions/opt_v9_nowedge.py

# Echo the finishing time
date
