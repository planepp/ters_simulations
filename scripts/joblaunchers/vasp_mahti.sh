#!/bin/bash
#SBATCH -J test
#SBATCH --account=project_2001912
#SBATCH -p test # test - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours
#SBATCH --time=1:00:00
#SBATCH -o out/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
##SBATCH --cpus-per-task=4

module load gcc/11.2.0 openmpi/4.1.2 openblas/0.3.18-omp StdEnv netlib-scalapack/2.1.0 hdf5/1.10.7-mpi fftw/3.3.10-mpi

export OMP_NUM_THREADS=1
ulimit -s unlimited

srun /projappl/project_2001912/plane_vasp.6.2.1/bin/vasp_std

