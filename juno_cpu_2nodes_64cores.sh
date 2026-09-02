#!/bin/bash
#SBATCH --job-name=cme_cpu
#SBATCH --partition=normal
# One JAX process needs 4 cores: 4 XLA devices (one per MCMC chain) x 1 thread each.
# The matrices are 51x51, far too small for intra-op threading to pay for itself, so
# extra cores per task do nothing. 16 tasks x 4 cores fills a 64-core node exactly.
# 8 nodes x 16 = 128 processes, so each one runs ~3 fits instead of ~46 - that is what
# keeps XLA from accumulating enough compiled executables to hit the mmap limit.
# 8 nodes is Juno's per-job maximum. mem-per-cpu 4G -> 256 GB/node of the 384 available.
# Original 2-tasks-per-node, 10-cores-per-task layout retained for reference:
# #SBATCH --nodes=4
# #SBATCH --ntasks=8
# #SBATCH --ntasks-per-node=2
# #SBATCH --cpus-per-task=10
# #SBATCH --mem=32G
# Four-node version retained for reference:
# #SBATCH --nodes=4
# #SBATCH --ntasks=64
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

# JAX/Eigen size their thread pool from the machine's core count (64), NOT from what
# SLURM allocated to the task. Without these caps every process started ~64 threads on
# a 4-core slice - that was the oversubscription. One thread per device is right here
# because a 51x51 matrix is too small to split across cores: measured on the laptop,
# going from 1 to 8 threads per fit changed nothing (21.3 vs 21.7 ms/it), while turning
# Eigen's multithreading off was the fastest of all (20.9 ms/it).
# OMP/MKL/OPENBLAS govern numpy and scipy. They do NOT control JAX's own compute.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# This is the knob that actually controls JAX/XLA on CPU.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

cd "${SLURM_SUBMIT_DIR}"

srun --cpu-bind=cores python fit_emotion_labeling.py
