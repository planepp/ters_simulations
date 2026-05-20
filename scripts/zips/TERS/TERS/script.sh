#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p medium # test - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours #
#SBATCH --time=36:00:00      # dd-hh:mm:ss
#SBATCH -J AtifAIMS        # name 
#SBATCH -o sbatch-%j.out # where the outputs & errors are written
#SBATCH --nodes=4                  # N nodes to run (N x 64 = n); max 192 ; max debug 48
#SBATCH --ntasks-per-node=128      # n processes to run (N x 64 = n); max 192 ; max debug 48
##SBATCH --cpus-per-task=4

module load gcc/11.2.0   openmpi/4.1.2   openblas/0.3.18-omp   csc-tools   StdEnv   netlib-scalapack/2.1.0 python-data/3.8-22.10
export OMP_NUM_THREADS=1
ulimit -s unlimited

# Path to your aims binary
AIMS_BIN="/projappl/project_2001912/fhi-aims.250822/build/aims.250822.scalapack.mpi.x"

# Loop over all mode_* directories
for mode_dir in mode_*; do
    for disp in positive_displacement negative_displacement; do
        for field in field_on zero_field; do
            target_dir="$mode_dir/$disp/$field"
            if [ -d "$target_dir" ]; then
                if [ ! -f "$target_dir/aims.out" ]; then
                    echo "Running in $target_dir"
                    cd "$target_dir" || exit 1
                    srun "$AIMS_BIN" > aims.out
                    cd - >/dev/null
                else
                    echo "Skipping $target_dir (aims.out already exists)"
                fi
            fi
        done
    done
done

