#!/bin/bash
#SBATCH -J aims_array
#SBATCH -o out/array_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2000
# (array indices and concurrency limit set dynamically at submission)

AIMS_BIN="aims.250320.scalapack.mpi.x"
MAX_CONCURRENT=50   # max simultaneously running array tasks

########################################
# 🔹 LAUNCHER MODE (no array task ID)
########################################
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    if [ -z "$1" ]; then
        echo "Usage: sbatch $0 [nmodes|ters1d|ters2d]"
        exit 1
    fi
    mode=$1
    mkdir -p out

    echo "Scanning directories..."

    # Collect leaf directories (no subdirectories) that haven't finished
    mapfile -t dirs < <(
        find "${mode}"* -mindepth 1 -type d |
        while IFS= read -r d; do
            # Skip if it has subdirectories (not a leaf)
            find "$d" -mindepth 1 -maxdepth 1 -type d | read -r && continue
            # Skip if already finished
            [ -f "$d/aims.out" ] &&
                grep -Eq "Have a nice day|Invalid ovlp_type" "$d/aims.out" && continue
            echo "$d"
        done | sort
    )

    num_dirs=${#dirs[@]}
    if [ "$num_dirs" -eq 0 ]; then
        echo "Nothing to run — all calculations finished."
        exit 0
    fi

    # Write the directory list, tagged with the launcher job ID
    listfile="dirs_list_${SLURM_JOB_ID}.txt"
    printf "%s\n" "${dirs[@]}" > "$listfile"

    echo "Total dirs   : $num_dirs"
    echo "Max concurrent: $MAX_CONCURRENT"
    echo "List file    : $listfile"

    # One task per directory; %MAX_CONCURRENT caps simultaneous tasks
    sbatch \
        --array=0-$((num_dirs - 1))%${MAX_CONCURRENT} \
        --export=ALL,LISTFILE="$listfile",AIMS_BIN="$AIMS_BIN" \
        "$0" "$mode"
    exit 0
fi

########################################
# 🔹 ARRAY TASK MODE (one dir per task)
########################################
module load triton/2024.1-gc aims/250320
export OMP_NUM_THREADS=1
ulimit -s unlimited

mapfile -t dirs < "$LISTFILE"
d="${dirs[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$d" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: index out of range, nothing to do."
    exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID → $d"

# Double-check: another task (or a previous run) may have finished this already
if [ -f "$d/aims.out" ] && grep -Eq "Have a nice day|Invalid ovlp_type" "$d/aims.out"; then
    echo "Already finished, skipping."
    exit 0
fi

cd "$d" || { echo "ERROR: cannot cd into $d"; exit 1; }

srun "$AIMS_BIN" >> aims.out 2>> aims.err

if grep -Eq "Have a nice day|Invalid ovlp_type" aims.out; then
    echo "Calculation successful."
else
    echo "ERROR: calculation in $d did not finish cleanly." >&2
    exit 1
fi
