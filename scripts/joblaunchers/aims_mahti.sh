#!/bin/bash
#SBATCH -J test
#SBATCH --account=project_2001912
#SBATCH -p test # test - 1h ; medium - 20 nodes/36 hours ; large - 20-200 nodes/36 hours
#SBATCH --time=1:00:00
#SBATCH -o out/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

module load gcc/11.2.0 openmpi/4.1.2 openblas/0.3.18-omp csc-tools StdEnv netlib-scalapack/2.1.0 python-data/3.8-22.10
export OMP_NUM_THREADS=1
ulimit -s unlimited

AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"
srun "$AIMS_BIN" > aims.out
