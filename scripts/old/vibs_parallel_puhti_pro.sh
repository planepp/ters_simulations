#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p small
#SBATCH --time=2-0:0:00
#SBATCH -J launcher
#SBATCH -o out/launcher_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1

python -u test1.py "$@"

