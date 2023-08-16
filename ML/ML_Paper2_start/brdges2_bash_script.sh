#!/bin/bash
#SBATCH -J NoWedge
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:16
#SBATCH -A ast190007p
#SBATCH -o /ocean/projects/ast180004p/tbilling/sandbox/bayesian/denseflipout/sandbox/load_ml_p1_model_stream_v12Data_DenseVar_layer_end/nowedge.log
#SBATCH --mail-type ALL
#SBATCH --mail-user tashalee@sas.upenn.edu

--gres=gpu:v100-32:8 -N 2 -t 3:00:00

#echo commands to stdout
set -x

# echo each command to standard out before running it


# run the Unix 'date' command
date

# activate conda environment to work in
conda activate hp_opt
date

# run the Unix 'echo' command
echo "Running load_ml_p1_model_stream_v12Data_DenseVar_layer_end.py"

# run python script
python load_ml_p1_model_stream_v12Data_DenseVar_layer_end.py
