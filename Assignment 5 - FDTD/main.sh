#!/usr/bin/env bash
N_vals=(1 2 4 8 16)
for N in ${N_vals[@]}; do
  sbatch --ntasks=${N} --output=${N}.out slurm.sh
done
