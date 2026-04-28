#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p small
#SBATCH --time=10:00:00
#SBATCH -J launcher
#SBATCH -o out/launcher_%j.out
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4

AIMS_BIN="/projappl/project_2001912/aims.250822.scalapack.mpi.x"

if [ -z "$1" ]; then
    echo "Usage: sbatch submit_groups_stop_on_error.sh [nmodes|ters1d|ters2d]"
    exit 1
fi

start_time=$(date +%s)
echo "Launcher started at $(date)"

### Find all leaf directories
mode=$1
all_dirs=( $(find "${mode}"* -type d ! -exec sh -c 'ls -A "{}"/*/ >/dev/null 2>&1' \; -print) )
dirs=()
for d in "${all_dirs[@]}"; do
    if [ ! -f "$d/aims.out" ] || ! grep -Eq "Have a nice day|Invalid ovlp_type" "$d/aims.out"; then
        dirs+=( "$d" )
    fi
done

### Divide the directories in groups
group_size=1
num_dirs=${#dirs[@]}
num_groups=$(( (num_dirs + group_size - 1) / group_size ))

# Create output directories if they don't exist
mkdir -p out
mkdir -p job_launchers
job_ids=()

### Submit one job per group
for ((g=0; g<num_groups; g++)); do
    start=$(( g * group_size ))
    if (( start + group_size > num_dirs )); then
        length=$(( num_dirs - start ))
    else
        length=$group_size
    fi
    group_dirs=( "${dirs[@]:start:length}" )

    jobname="group_$g"
    jobfile="job_launchers/job_${jobname}.sh"

    # Create the SLURM job script
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p small
#SBATCH --time=2:00:00   
#SBATCH -J ${jobname}
#SBATCH -o out/${jobname}_%j.out
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=40
##SBATCH --cpus-per-task=4

module load StdEnv  csc-tools  python-data/3.8-22.10  intel-oneapi-mkl/2022.1.0  intel-oneapi-compilers/2022.1.0  intel-oneapi-mpi/2021.6.0
ulimit -s unlimited

# Run in each directory in the group
for d in ${group_dirs[@]}; do
	echo "Running in \$d"
	cd "\$d"

	if [ ! -f "aims.out" ] || ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then
		srun $AIMS_BIN >> aims.out 2>> aims.err

		if ! grep -Eq "Have a nice day|Invalid ovlp_type" "aims.out"; then
			echo "ERROR: \$d failed - check aims.out"
			echo "Stopping calculations"
			cd - >/dev/null
			exit 1
		else
			echo "Calculation succesful"
			echo ""
		fi
		cd - >/dev/null
	fi
done
EOF

    # Submit the group job
    jid=$(sbatch "$jobfile" | awk '{print $4}')
    echo "Submitted $jobname with Job ID $jid"
    for d in "${group_dirs[@]}"; do
        echo "  $d"
    done
    echo ""
    job_ids+=("$jid")
done

### Hold launcher until all jobs are done
echo "Waiting for all jobs to finish..."
for jid in "${job_ids[@]}"; do
    while squeue -j "$jid" 2>/dev/null | grep -q "^ *$jid "; do
        sleep 10
    done

    # Check final state of all the jobs
    state=$(sacct -j "${jid}.batch" --format=State%20 -n | tr -d ' ')
    if [[ "$state" == "TIMEOUT" ]]; then
        echo "WARNING: Job $jid hit the time limit!"
    elif [[ "$state" != "COMPLETED" ]]; then
        echo "WARNING: Job $jid did not complete successfully!"
    fi
done

# Measure running time
end_time=$(date +%s)
echo "Last job ended at $(date)"

total_time=$((end_time - start_time))
total_mins=$(echo "scale=2; $total_time / 60" | bc)
echo "Total running time: $total_mins minutes"
