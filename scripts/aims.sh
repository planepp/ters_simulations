#!/bin/bash
source "$CONFIG"

module purge
module load $MODULES
export OMP_NUM_THREADS=1
ulimit -s unlimited

echo "======================================"
echo "Cluster:      $CLUSTER"
echo "Host:         $(hostname)"
echo "Modules:      $MODULES"
echo "AIMS binary:  $AIMS_BIN"
echo "======================================"

srun $AIMS_BIN geometry.in > aims.out
