#!/bin/bash -l
#SBATCH --job-name=ph1
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03

export VASP_PP_PATH=/projappl/project_2001912
SCRIPT="$VASP_PP_PATH/pyFDBM"
VASPCMD="srun -n 16 $VASP_PP_PATH/vasp-env/view/bin/vasp_std"

python "$SCRIPT/prepare_calculation.py" --vaspcmd "$VASPCMD" -s -t -g -i input.in > prep.out
python "$SCRIPT/density_interactions.py" -i input.in > dens.out

source "$VASP_PP_PATH/env_pyfdbm/bin/activate"
srun -n 16 python "$SCRIPT/dftd3-py.py" -i input.in > d3.out
deactivate

source "$VASP_PP_PATH/env_pyfdbm2/bin/activate"
srun -n 16 python "$SCRIPT/relax_xy.py" -i input.in -m powell -k 0.4 > relax.out
deactivate

