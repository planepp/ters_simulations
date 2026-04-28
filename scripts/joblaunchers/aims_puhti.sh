#!/bin/bash
#SBATCH -J test
#SBATCH --account=project_2001912
#SBATCH -p test # test - 10 min, 4 nodes ; small - 3 days, 40 nodes ; large - 3 days, 1040 nodes
#SBATCH --time=0:10:00
#SBATCH -o out/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=4

module purge
module load StdEnv  csc-tools  python-data/3.8-22.10  intel-oneapi-mkl/2022.1.0  intel-oneapi-compilers/2022.1.0  intel-oneapi-mpi/2021.6.0
export OMP_NUM_THREADS=1
ulimit -s unlimited

# Path to aims binary and species defaults
AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"

### Run the calculation
srun $AIMS_BIN geometry.in > aims.out
