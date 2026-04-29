# config.triton.sh
CLUSTER="triton"
MODULES="triton/2024.1-gc aims/250320"
AIMS_BIN="aims.250320.scalapack.mpi.x"

# SLURM resources
JOB_NAME="test"
OUTPUT="out/%j.out"
TIME="3:00:00"
NODES=1
NTASKS_PER_NODE=24
MEM_PER_CPU=2000
