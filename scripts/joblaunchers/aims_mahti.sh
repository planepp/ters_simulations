#!/bin/bash
#SBATCH -J test
#SBATCH --account=project_2001912
#SBATCH -p test # test - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours
#SBATCH --time=1:00:00
#SBATCH -o out/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
##SBATCH --cpus-per-task=4

module load gcc/11.2.0 openmpi/4.1.2 openblas/0.3.18-omp csc-tools StdEnv netlib-scalapack/2.1.0 python-data/3.8-22.10
export OMP_NUM_THREADS=1
ulimit -s unlimited

# Path to aims binary and species defaults
AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"

# Run FHI-aims
srun "$AIMS_BIN" > aims.out
