#!/bin/bash

########################################
# User settings
########################################

AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"
group_size=3
JOBNAME="modesmariana"
TIME="8:00:00"

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
    echo "Usage: ./submit_vibs.sh [nmodes|ters1d|ters2d] [group_size] [hours] [--pos 1,2,3]"
    exit 1
fi

mkdir -p out out/dirs_lists

########################################
# Find unfinished leaf directories
########################################

echo "Preparing directory list..."


#mapfile -t all_dirs < <(
#    find "${mode}"* -type d ! -exec sh -c 'ls -A "{}"/*/ >/dev/null 2>&1' \; -print | sort
#)

mapfile -t all_dirs < <(get_leaf_dirs.py "${mode}" )
echo "Found ${#all_dirs[@]} directories"

#check_zerofield.py ${all_dirs[0]}

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
listfile="./out/dirs_lists/dirs_${JOBNAME}_${timestamp}.txt"

printf "%s\n" "${dirs[@]}" > "$listfile"

########################################
# Compute array size
########################################

num_groups=$(( (num_dirs + group_size - 1) / group_size ))

echo "Total dirs: $num_dirs"
echo "Group size: $group_size"
echo "Array jobs: $num_groups"
echo "Allocated time: $TIME"

########################################
# Submit array job
########################################

echo "Starting $JOBNAME"

: '
# -EOF = tab friendly, EOF is just convention, and not using the quoted 'EOF' allows for variables evaluation
#sh -s $listfile $group_size $AIMS_BIN \ # debug run
'
sbatch \
    --job-name="$JOBNAME" \
    --time="$TIME" \
    --output="out/${JOBNAME}/%A_%a.out" \
    --array=0-$((num_groups - 1)) \
    --export=ALL,LISTFILE="$listfile",GROUP_SIZE="$group_size",AIMS_BIN="$AIMS_BIN" \
<<-EOF
#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p small # test - 10 min, 4 nodes ; small - 3 days, 40 nodes ; large - 3 days, 1040 nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

module load StdEnv  csc-tools  python-data/3.8-22.10  intel-oneapi-mkl/2022.1.0  intel-oneapi-compilers/2022.1.0  intel-oneapi-mpi/2021.6.0
export OMP_NUM_THREADS=1
ulimit -s unlimited

########################################
# Load directory list
########################################

mapfile -t dirs < "\$LISTFILE"

num_dirs=\${#dirs[@]}

start=\$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE ))
end=\$(( start + GROUP_SIZE ))

if (( start >= num_dirs )); then
    echo "Task \$SLURM_ARRAY_TASK_ID: nothing to do"
    #check_zerofield.py \${dirs[0]} print_missing
    exit 0
fi

if (( end > num_dirs )); then
    end=\$num_dirs
fi

echo "Task \$SLURM_ARRAY_TASK_ID handling dirs \$start to \$((end - 1))"

########################################
# Run calculations
########################################

#check_zerofield.py \${dirs[0]}

start_time=\$(date +%s)
echo "STARTED at $(date '+%F %T')"
n=\$((end - start))

iter_start=\$(date +%s)

for ((i = start; i < end; i++)); do

    d="\${dirs[\$i]}"

    echo "========================================"
    echo "\$i/\$((end-1)) running in: \$d"

    cd "\$d" || {
        echo "ERROR: Could not enter \$d"
        continue
    }

    if [ ! -f "aims.out" ] || \
       ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then

        srun "\$AIMS_BIN" >> aims.out 2>> aims.err

        ### ETA calculator
        iter_end=\$(date +%s)
        iter_dt=\$((iter_end - iter_start))

        done_count=\$((i - start + 1))
        elapsed=\$((iter_end - start_time))
        avg=\$((elapsed / done_count))
        remaining=\$((avg * (n - done_count)))
        eta_epoch=\$((iter_end + remaining))

        eta_local=\$(date -d "@\$eta_epoch" '+%F %T')
        rem_h=\$((remaining / 3600))
        rem_m=\$(((remaining % 3600) / 60))
        rem_s=\$((remaining % 60))

        iter_start=\$(date +%s) # for next loop
        ###
        if ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then # something in aims.out failed
            printf "ERROR: %s failed after %d s. ETA: %s. You will need to run it again later.\n" "\$d" "\$iter_dt" "\$eta_local"
            #exit 1
        else
            printf "Calculated in %d s. Remaining: %02d:%02d:%02d, ETA: %s\n" "\$iter_dt" "\$rem_h" "\$rem_m" "\$rem_s" "\$eta_local"
        fi
        
    else
        echo "Already completed"
    fi


    cd - >/dev/null

done

#check_zerofield.py \${dirs[0]}

echo "DONE at \$(date '+%F %T')"

EOF
