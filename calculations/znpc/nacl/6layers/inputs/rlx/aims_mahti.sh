#!/bin/bash
#SBATCH -J newslab_znpcnacl
#SBATCH --account=project_2001912
#SBATCH -p medium # test - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours
#SBATCH --time=18:0:00
#SBATCH -o out_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128

module load gcc/11.2.0 openmpi/4.1.2 openblas/0.3.18-omp csc-tools StdEnv netlib-scalapack/2.1.0 python-data/3.8-22.10
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
