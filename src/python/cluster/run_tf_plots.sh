#!/bin/bash
#SBATCH --job-name=tf_plots
#SBATCH --account=st-ssfels-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_tf_plots/%x_%A_%a.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_tf_plots/%x_%A_%a.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --array=2-5

# ---------------------------------------------------------------------------
# CONFIGURE: edit subjects to run
# ---------------------------------------------------------------------------
SUBJECTS=(
    "subj-02"
    "subj-03"
    "subj-04"
    "subj-05"
    "subj-06"
    "subj-07"
)
 
N_SUBJ=${#SUBJECTS[@]}
 
# Validate that the task ID is within range (safety check)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_SUBJ}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= N_SUBJ=${N_SUBJ}. Exiting."
    exit 1
fi
 
SUBJ_IDX=${SLURM_ARRAY_TASK_ID}
SUBJ="${SUBJECTS[$SUBJ_IDX]}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_tf_plots

export MPLCONFIGDIR=/scratch/st-ssfels-1/vickywx/cache/matplotlib
export FONTCONFIG_CACHE=/scratch/st-ssfels-1/vickywx/cache/fontconfig
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_CACHE}"

source ~/python_3_12_env/bin/activate

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node:             $(hostname)"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:    ${SLURM_ARRAY_TASK_ID}  (${SUBJ})"
echo "CPUs:             ${SLURM_CPUS_PER_TASK}"
echo "Mem:              ${SLURM_MEM_PER_NODE} MB"

python /scratch/st-ssfels-1/vickywx/B2S_data_analysis/src/python/tf_plots.py \
    --subj "${SUBJ}" \
    --input-dir  /scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/04_processed/ \
    --output-dir /scratch/st-ssfels-1/vickywx/B2S_data_analysis/results/tf_plots/ \
    --figure-dir-name all_ICs_all_epochs

EXIT_CODE=$?
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}