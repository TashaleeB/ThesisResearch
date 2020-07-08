#!/bin/bash
#SBATCH -J VStackSnapshots
#SBATCH -p LM
#SBATCH -t 72:00:00
#SBATCH --mem=1530GB
#SBATCH -o /pylon5/as5phnp/tbilling/data/vstack_snapshots.log
#SBATCH -A as5fp4p
#SBATCH --mail-type ALL
#SBATCH --mail-user tashalee@sas.upenn.edu
# Intel

# Start GPU environment
conda activate hp_opt
# echo the starting time
date

# Compile and run job
python /pylon5/as5phnp/tbilling/data/vstack_snapshots.py

# Echo the finishing time
date
