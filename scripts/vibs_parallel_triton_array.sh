#!/bin/bash
#SBATCH -J aims_array
#SBATCH -o out/array_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2000
# (array is set dynamically below)

AIMS_BIN="aims.250320.scalapack.mpi.x"
GROUP_SIZE=1

########################################
# 🔹 LAUNCHER MODE (no array task ID)
########################################
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then

    if [ -z "$1" ]; then
        echo "Usage: sbatch this_script.sh [nmodes|ters1d|ters2d]"
        exit 1
    fi

    mode=$1
    mkdir -p out

    echo "Preparing directory list..."

    # Find directories
    all_dirs=( $(find "${mode}"* -type d ! -exec sh -c 'ls -A "{}"/*/ >/dev/null 2>&1' \; -print) )
    dirs=()

    for d in "${all_dirs[@]}"; do
        if [ ! -f "$d/aims.out" ] || ! grep -Eq "Have a nice day|Invalid ovlp_type" "$d/aims.out"; then
            dirs+=( "$d" )
        fi
    done

    num_dirs=${#dirs[@]}

    if [ "$num_dirs" -eq 0 ]; then
        echo "Nothing to run!"
        exit 0
    fi

    # Save list
    listfile="dirs_list_${SLURM_JOB_ID}.txt"
    printf "%s\n" "${dirs[@]}" > "$listfile"

    # Compute number of groups
    num_groups=$(( (num_dirs + GROUP_SIZE - 1) / GROUP_SIZE ))

    echo "Total dirs: $num_dirs"
    echo "Group size: $GROUP_SIZE"
    echo "Array jobs: $num_groups"

    # Submit array job (self-submission)
    sbatch --array=0-$((num_groups-1)) \
           --export=ALL,LISTFILE="$listfile",GROUP_SIZE="$GROUP_SIZE",AIMS_BIN="$AIMS_BIN" \
           "$0"

    exit 0
fi

########################################
# 🔹 ARRAY TASK MODE
########################################

module load triton/2024.1-gc aims/250320
export OMP_NUM_THREADS=1
ulimit -s unlimited

mapfile -t dirs < "$LISTFILE"
num_dirs=${#dirs[@]}

start=$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE ))
end=$(( start + GROUP_SIZE ))

if (( start >= num_dirs )); then
    echo "Task $SLURM_ARRAY_TASK_ID: nothing to do"
    exit 0
fi

if (( end > num_dirs )); then
    end=$num_dirs
fi

echo "Task $SLURM_ARRAY_TASK_ID handling dirs $start to $((end-1))"

for ((i=start; i<end; i++)); do
    d="${dirs[$i]}"
    echo "Running in $d"
    cd "$d"

    if [ ! -f "aims.out" ] || ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then
        srun $AIMS_BIN >> aims.out 2>> aims.err

        if ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then
            echo "ERROR: $d failed"
            exit 1
        else
            echo "Calculation successful"
        fi
    fi

    cd - >/dev/null
done
