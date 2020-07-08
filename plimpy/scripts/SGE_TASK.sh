#!/bin/sh

#  SGE_TASK.sh
#  
#
#  Created by Tashalee Billings on 2/10/19.
#  
!/bin/sh
$ -N SGE_TASK # Some name.
$ -cwd # Current Working Directory
$ -V # Copy Environment
$ -j y
$ -pe omp 8 # Number of Cores
$ -R y
$ -l h_vmem=7000M
python imageing_pipeline.py $SGE_TASK_ID 8
