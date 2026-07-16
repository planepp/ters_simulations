#!/bin/bash
#SBATCH -J gaussian
#SBATCH -o gaussian.out
#SBATCH -e gaussian.err
#SBATCH -t 4:00:00
#SBATCH --account=project_2001912
#SBATCH -N 1
#SBATCH --mem=3500
#SBATCH -c 24

module purge
module load gaussian anaconda
export g16root=/appl/soft/manual/chem/x86_64/gaussian/G16RevC.02_bin
source $g16root/g16/bsd/g16.profile
export OMP_NUM_THREADS=1
export GAUSS_SCRDIR=$PWD/$SLURM_JOB_ID.GAUSS_SCRDIR
mkdir -p $GAUSS_SCRDIR
lfs setstripe -S4M -E 64M -c 1 -E 2G -c 4 -E -1 -c 8 $GAUSS_SCRDIR

which g16

# Run Gaussian
g16 < mgp.com > mgp.log

# Cleanup
rm -rf "$GAUSS_SCRDIR"
