#!/usr/bin/env bash

#SBATCH --account=teaching
#SBATCH --reservation=teaching
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=my_job.out

mpirun -n $SLURM_NTASKS time python -m mpi4py main_parallel.py