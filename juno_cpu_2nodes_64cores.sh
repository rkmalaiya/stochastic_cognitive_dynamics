#!/bin/bash
#SBATCH --job-name=cme_cpu
#SBATCH --partition=normal
#SBATCH --nodes=2
# Original SLURM task configuration retained for reference:
# #SBATCH --ntasks-per-node=64
# Run one coordinating Python process on each allocated node.
#SBATCH --ntasks-per-node=1
# Give that coordinator, and its local Joblib workers, all 64 node cores.
#SBATCH --cpus-per-task=64
# Original memory request retained for reference. Juno rejected this because
# schedulable memory is lower than the node's advertised physical memory.
# #SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --output=cme_cpu_%j.out
#SBATCH --error=cme_cpu_%j.err

module load miniconda
source ~/.bashrc
conda activate ds

# Original per-batch core budget retained for reference. SLURM now runs one
# JAX process per node and does not start local Joblib children.
# export CME_CORES_PER_BATCH=10

# Original single-thread restrictions retained for reference:
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# Original explicit thread limits retained for reference. JAX and the native
# math libraries now use the CPUs made available by SLURM.
# export OMP_NUM_THREADS="${CME_CORES_PER_BATCH}"
# export MKL_NUM_THREADS="${CME_CORES_PER_BATCH}"
# export OPENBLAS_NUM_THREADS="${CME_CORES_PER_BATCH}"

cd "${SLURM_SUBMIT_DIR}"
# Original example entry point retained for reference:
# srun --cpu-bind=cores python examples/fit_data.py
srun --cpu-bind=cores python fit_emotion_labeling.py
