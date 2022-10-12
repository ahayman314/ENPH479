#!/usr/bin/env bash
N_vals=(1 2 4 8 16)
Nodes=1
for N in ${N_vals[@]}; do
  sbatch --ntasks=${N} --nodes=${Nodes} --output=${Nodes}_${N}.out slurm_script.sh
done

N_vals=(2 4 8 16 32)
Nodes=2

for N in ${N_vals[@]}; do
  sbatch --ntasks=${N} --output=${Nodes}_${N}.out slurm_script.sh
done
