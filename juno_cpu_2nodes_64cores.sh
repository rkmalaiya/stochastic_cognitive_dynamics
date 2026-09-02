#!/bin/bash
#SBATCH --job-name=cme_cpu
#SBATCH --partition=normal
# #SBATCH --nodes=4
# #SBATCH --ntasks=8
# #SBATCH --ntasks-per-node=2
# #SBATCH --cpus-per-task=10
# #SBATCH --mem=32G
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=cme_cpu_%j.out
#SBATCH --error=cme_cpu_%j.err
#SBATCH --open-mode=append

module load miniconda
source ~/.bashrc
conda activate ds

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

cd "${SLURM_SUBMIT_DIR}"

srun --cpu-bind=cores python fit_emotion_labeling.py
