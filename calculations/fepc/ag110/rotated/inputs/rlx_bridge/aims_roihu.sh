#!/bin/bash
#SBATCH -J ag110_rotated
#SBATCH --account=project_2001912
#SBATCH -p small
#SBATCH --time=36:00:00
#SBATCH -o out_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

module load gcc/15.2.0 openmpi/5.0.10 openblas/0.3.30 csc-tools StdEnv netlib-scalapack/2.2.2 python-data/3.10-06.07
export OMP_NUM_THREADS=1
ulimit -s unlimited

# Path to aims binary and species defaults
AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"

# Prepare control.in
# the argument is for example: C tight Mo really_tight
# all the others will be light by default
#control_speciesdefaults "$@"

# Run FHI-aims
srun "$AIMS_BIN" > aims.out
