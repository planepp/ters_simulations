#!/bin/bash
#SBATCH -J test
#SBATCH -o out/%j.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2000

module purge
module load triton/2024.1-gc aims/250320
export OMP_NUM_THREADS=1
ulimit -s unlimited

# Path to aims binary and species defaults
AIMS_BIN="aims.250320.scalapack.mpi.x"

### Run the calculation
srun $AIMS_BIN geometry.in > aims.out
