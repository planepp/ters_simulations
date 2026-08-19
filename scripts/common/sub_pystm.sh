#!/bin/bash -l
#SBATCH --job-name=stm1
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out                # Output file based on job name and job ID
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

cd "$SLURM_SUBMIT_DIR"

if [ -z "$1" ]; then
    echo "Por favor, proporciona un voltaje como primer argumento."
    exit 1
fi

V=$1
export VASP_PP_PATH=/projappl/project_2001912
SCRIPT="$VASP_PP_PATH/pyFDBM"

python "$SCRIPT/pystm.py" --dir sample -b $V  



