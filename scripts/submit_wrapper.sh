#!/bin/bash
# submit_wrapper.sh
CONFIG=${1:-"config.triton.sh"}
source "$CONFIG"

mkdir -p out

sbatch \
  -J $JOB_NAME \
  -o $OUTPUT \
  --time=$TIME \
  --nodes=$NODES \
  --ntasks-per-node=$NTASKS_PER_NODE \
  --mem-per-cpu=$MEM_PER_CPU \
  --export=ALL,CONFIG=$CONFIG \
  aims.sh
