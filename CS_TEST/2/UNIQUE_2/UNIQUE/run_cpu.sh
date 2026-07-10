#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -J test_allo
#SBATCH -o ll_out
#SBATCH -A m4397

module load python/3.10
conda activate /global/common/software/m4397/naman/conda_pkgs/mattergen
export PYTHONPATH=/global/common/software/m4397/naman/code:$PYTHONPATH

python3 new_optimization.py . > out 
