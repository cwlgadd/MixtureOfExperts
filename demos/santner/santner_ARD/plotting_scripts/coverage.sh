#!/bin/sh
#SBATCH --array=1,2,5,10,15,20

module load anaconda3/5.1.0-cpu.lua

cd /l/gaddc1/Dropbox/MixtureOfExperts/demos/santner/Apr15_fromTriton

python /l/gaddc1/Dropbox/MixtureOfExperts/demos/santner/plotting_scripts/coverage.py ${SLURM_ARRAY_TASK_ID} hpd_union
