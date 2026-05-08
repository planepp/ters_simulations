#!/bin/bash

########################################
# User settings
########################################

AIMS_BIN="aims.250320.scalapack.mpi.x"
GROUP_SIZE=5
JOBNAME="neglines"
TIME="10:00:00"

########################################
# Parse arguments
########################################

mode=$1
shift

pos_filter=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pos)
            IFS=',' read -ra pos_filter <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$mode" ]; then
    echo "Usage: ./submit_vibs.sh [nmodes|ters1d|ters2d] [--pos 1,2,3]"
    exit 1
fi

mkdir -p out dirs_lists

########################################
# Find unfinished leaf directories
########################################

echo "Preparing directory list..."

mapfile -t all_dirs < <(
    find "${mode}"* -type d ! -exec sh -c 'ls -A "{}"/*/ >/dev/null 2>&1' \; -print | sort
)

dirs=()

for d in "${all_dirs[@]}"; do

    if [ ! -f "$d/aims.out" ] || \
       ! grep -Eq "Have a nice day|Invalid ovlp_type" "$d/aims.out"; then
        dirs+=( "$d" )
    fi

done

########################################
# Optional filtering by tip positions
########################################

if [ ${#pos_filter[@]} -gt 0 ]; then

    filtered=()

    for d in "${dirs[@]}"; do

        # Always include zerofield
        if [[ "$d" == *"zerofield"* ]]; then
            filtered+=( "$d" )
            continue
        fi

        tipdir=$(echo "$d" | cut -d'/' -f3)

        for i in "${pos_filter[@]}"; do

            target="tippos_$(printf '%03d' "$i")"

            if [[ "$tipdir" == "$target" ]]; then
                filtered+=( "$d" )
                break
            fi

        done

    done

    dirs=( "${filtered[@]}" )

fi

########################################
# Final checks
########################################

num_dirs=${#dirs[@]}

if [ "$num_dirs" -eq 0 ]; then
    echo "Nothing to run!"
    exit 0
fi

########################################
# Save directory list
########################################

timestamp=$(date +%Y%m%d_%H%M%S)
listfile="dirs_lists/dirs_${JOBNAME}_${timestamp}.txt"

printf "%s\n" "${dirs[@]}" > "$listfile"

########################################
# Compute array size
########################################

num_groups=$(( (num_dirs + GROUP_SIZE - 1) / GROUP_SIZE ))

echo "Total dirs: $num_dirs"
echo "Group size: $GROUP_SIZE"
echo "Array jobs: $num_groups"

########################################
# Submit array job
########################################

sbatch \
    --job-name="$JOBNAME" \
    --time="$TIME" \
    --output="out/${JOBNAME}_%A_%a.out" \
    --array=0-$((num_groups - 1)) \
    --export=ALL,LISTFILE="$listfile",GROUP_SIZE="$GROUP_SIZE",AIMS_BIN="$AIMS_BIN" \
<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2000

module load triton/2024.1-gc aims/250320

export OMP_NUM_THREADS=1
ulimit -s unlimited

########################################
# Load directory list
########################################

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

echo "Task $SLURM_ARRAY_TASK_ID handling dirs $start to $((end - 1))"

########################################
# Run calculations
########################################

for ((i = start; i < end; i++)); do

    d="${dirs[$i]}"

    echo "========================================"
    echo "Running in: $d"
    echo "========================================"

    cd "$d" || {
        echo "ERROR: Could not enter $d"
        continue
    }

    if [ ! -f "aims.out" ] || \
       ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then

        srun "$AIMS_BIN" >> aims.out 2>> aims.err

        if grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then
            echo "Calculation successful"
        else
            echo "ERROR: Calculation failed in $d"
        fi

    else

        echo "Already completed"

    fi

    cd - >/dev/null

done
EOF
