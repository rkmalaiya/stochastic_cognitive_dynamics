#!/bin/bash
#SBATCH --job-name=cme_cpu
#SBATCH --partition=normal
# Original multi-node allocation retained for reference:
# #SBATCH --nodes=2
# Each array task requests one node and runs one Python process.
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Two independent processes replace the original two-node job.
# Change 0-1 to 0-(N-1) to request N independent processes.
#SBATCH --array=0-1
# Keep separate array tasks on separate physical nodes.
#SBATCH --exclusive
# Permit a failed array task to be submitted again.
#SBATCH --requeue
# Original SLURM task configuration retained for reference:
# #SBATCH --ntasks-per-node=64
# Original CPU allocation retained for reference:
# #SBATCH --cpus-per-task=64
# Give the single JAX process 10 CPU cores on its node.
#SBATCH --cpus-per-task=10
# Original memory request retained for reference. Juno rejected this because
# schedulable memory is lower than the node's advertised physical memory.
# #SBATCH --mem=384G
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
# Original non-array output names retained for reference:
# #SBATCH --output=cme_cpu_%j.out
# #SBATCH --error=cme_cpu_%j.err
#SBATCH --output=cme_cpu_%A_%a.out
#SBATCH --error=cme_cpu_%A_%a.err
#SBATCH --open-mode=append

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
# Original multi-node launch retained for reference:
# srun --cpu-bind=cores python fit_emotion_labeling.py

srun --ntasks=1 --cpu-bind=cores env \
    SLURM_PROCID="${SLURM_ARRAY_TASK_ID}" \
    SLURM_STEP_NUM_TASKS="${SLURM_ARRAY_TASK_COUNT}" \
    python fit_emotion_labeling.py
status=$?

restart_count="${SLURM_RESTART_COUNT:-0}"
if [ "${status}" -ne 0 ] && [ "${restart_count}" -lt 3 ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} failed with status ${status}; requeueing restart $((restart_count + 1))/3."
    if scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; then
        exit 0
    fi
fi

exit "${status}"
