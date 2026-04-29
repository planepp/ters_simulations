#!/bin/bash
#SBATCH -J aims_array
#SBATCH --account=project_2001912
#SBATCH -p test # test - 10 min, 4 nodes ; small - 3 days, 40 nodes ; large - 3 days, 1040 nodes
#SBATCH --time=0:10:00
#SBATCH -o out/array_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40


AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"
MAX_CONCURRENT=100
CHUNK_SIZE=5

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
    mapfile -t dirs < <(python3 collect_dirs.py "$mode")

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
    num_tasks=$(( (num_dirs + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    sbatch \
       --array=0-$((num_tasks - 1))%${MAX_CONCURRENT} \
       --export=ALL,LISTFILE="$listfile",AIMS_BIN="$AIMS_BIN",CHUNK_SIZE="$CHUNK_SIZE" \
       "$0" "$mode"

    exit 0
fi

########################################
# 🔹 ARRAY TASK MODE (one dir per task)
########################################
module load StdEnv  csc-tools  python-data/3.8-22.10  intel-oneapi-mkl/2022.1.0  intel-oneapi-compilers/2022.1.0  intel-oneapi-mpi/2021.6.0
export OMP_NUM_THREADS=1
ulimit -s unlimited

mapfile -t dirs < "$LISTFILE"
start=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
end=$(( start + CHUNK_SIZE ))
if [ "$end" -gt "${#dirs[@]}" ]; then
    end=${#dirs[@]}
fi

echo "Task $SLURM_ARRAY_TASK_ID → $d"

for ((i=start; i<end; i++)); do
    d="${dirs[$i]}"
    echo "Task $SLURM_ARRAY_TASK_ID → $d"

    cd "$d" || { echo "ERROR: cannot cd into $d"; continue; }

    srun "$AIMS_BIN" >> aims.out 2>> aims.err

    if grep -Eq "Have a nice day|Invalid ovlp_type" aims.out; then
        echo "[$d] OK"
    else
        echo "[$d] FAILED" >&2
    fi

    cd - > /dev/null
done
