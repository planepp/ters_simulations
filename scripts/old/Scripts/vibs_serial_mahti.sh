#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p medium # medium - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours #
#SBATCH --time=24:00:00   # dd-hh:mm:ss
#SBATCH -J vibs   # name
#SBATCH -o out/%j.out # where the outputs & errors are written
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=128
##SBATCH --cpus-per-task=4

module load gcc/11.2.0  openmpi/4.1.2  openblas/0.3.18-omp  csc-tools  StdEnv  netlib-scalapack/2.1.0 python-data/3.8-22.10
export OMP_NUM_THREADS=1
ulimit -s unlimited

start_time=$(date +%s)
echo "Job started at $(date)"

# Path to your aims binary
AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"

# Check argument
if [ -z "$1" ]; then
    echo "Usage: sbatch run_aims_conditional.sh [nmodes|ters]"
    exit 1
fi

mode=$1

run_aims() {
    local dir="$1"
    local aims_out="$dir/aims.out"

    # Run if aims.out doesn't exist OR it exists but is incomplete
    if [ ! -f "$aims_out" ] || ! tail -n 2 "$aims_out" | grep -q "Have a nice day"; then
        echo "Running in $dir"
        cd "$dir" || exit 1
        srun "$AIMS_BIN" > aims.out

        #Print if the run finished successfully
        if tail -n 2 aims.out | grep -Eq "Have a nice day|Invalid ovlp_type"; then
            tail -n 2 aims.out
        else
            echo "ERROR: $dir did not finish properly — check aims.out"
            echo "Stopping calculations"
            cd - >/dev/null
            exit 1
        fi
        cd - >/dev/null
    else
        echo "Skipping $dir (converged aims.out already exists)"
    fi
}

if [ "$mode" == "nmodes" ]; then
    # Run in *.i_atom_* directories
    dirs=( nmodes* )
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "ERROR: No directories found matching 'nmodes*'" >&2
        exit 1
    fi
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
		run_aims "$dir"
	fi
    done

elif [ "$mode" == "ters1d" ]; then
    # Run in mode_* / positive_displacement|negative_displacement / field_on|zero_field
    for mode_dir in ters1d*; do
        for disp in positive_displacement negative_displacement; do
            for field in field_on zero_field; do
                target_dir="$mode_dir/$disp/$field"
                if [ -d "$target_dir" ]; then
                        run_aims "$target_dir"
                fi
            done
        done
    done

elif [ "$mode" == "ters2d" ]; then
    # Run in calc2D_* / tippos* / positive_displacement|negative_displacement / field_on|zero_field
    for calc_dir in ters2d*; do
        for tippos_dir in "$calc_dir"/tippos*; do
            [ -d "$tippos_dir" ] || continue
            for disp in positive_displacement negative_displacement; do
                for field in field_on zero_field; do
                    target_dir="$tippos_dir/$disp/$field"
                    if [ -d "$target_dir" ]; then
                        run_aims "$target_dir"
                    fi
                done
            done
        done
    done

else
    echo "ERROR: Unknown mode '$mode'. Use 'nmodes', 'ters1d' or 'ters2d'." >&2
    exit 1
fi

end_time=$(date +%s)
echo "Job ended at $(date)"

total_time=$((end_time - start_time))
total_mins=$(echo "scale=2; $total_time / 60" | bc)
echo "Total time used: $total_mins minutes"


