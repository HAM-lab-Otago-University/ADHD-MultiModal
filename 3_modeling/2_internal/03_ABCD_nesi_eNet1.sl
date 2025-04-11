#!/bin/bash -e

#SBATCH --job-name=abcd_nesi_eNet1_cntr
#SBATCH --time=06:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=30
#SBATCH --account=uoo03493
#SBATCH --profile task
#SBATCH --output=/nesi/nobackup/uoo03493/farzane/abcd/main_std/%x/%x_%j_%a.out

#SBATCH --mail-user=lalfa602@student.otago.ac.nz
#SBATCH --mail-type=ALL

#SBATCH --array=0-20


module purge
module load Python/3.10.5-gimkl-2022a
source /nesi/project/uoo03493/PLS_venv/bin/activate

export PYTHONNOUSERSITE=1

python ./abcd_nesi_eNet1_cntr.py "${SLURM_ARRAY_TASK_ID}"
