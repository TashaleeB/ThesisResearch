#!/bin/bash
#SBATCH -J NoWedge
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -p EM
#SBATCH -A ast180004p
#SBATCH -o /ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/load_ml_p1_model_stream_v12Data_DenseVar_layer_end/nowedge.log
#SBATCH --mail-type ALL
#SBATCH --mail-user tashalee@sas.upenn.edu

#echo commands to stdout
set -x

# echo each command to standard out before running it


# run the Unix 'date' command
date

# activate conda environment to work in
conda activate hp_opt

echo "Running load_ml_p1_model_stream_v12Data_DenseVar_layer_end.py"
# run the Unix 'echo' command

python load_ml_p1_model_stream_v12Data_DenseVar_layer_end.py
