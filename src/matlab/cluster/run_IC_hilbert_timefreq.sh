#!/bin/bash
#SBATCH --job-name=eeg_hilbert
#SBATCH --account=st-ssfels-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_hilbert/%x_%A_%a.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_hilbert/%x_%A_%a.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL
 
# ---------------------------------------------------------------------------
# CONFIGURE: edit subjects and speech types to run
# ---------------------------------------------------------------------------
SUBJECTS=(
    "subj-02"
    "subj-03"
    "subj-04"
    "subj-05"
    "subj-06"
    "subj-07"
)
 
SPEECH_TYPES=(
    "sp"
    "im"
)
 
# ---------------------------------------------------------------------------
# Build (subject, speech_type) pairs and map task ID to a pair
#
# Task array is submitted with --array=0-N where N = (nSubjs * nTypes - 1)
# Pair index layout (row-major over subjects):
#   task 0 -> subj-02, sp
#   task 1 -> subj-02, im
#   task 2 -> subj-03, sp
#   task 3 -> subj-03, im   ...etc
# ---------------------------------------------------------------------------
 
N_SUBJ=${#SUBJECTS[@]}
N_TYPE=${#SPEECH_TYPES[@]}
N_TOTAL=$(( N_SUBJ * N_TYPE ))
 
# Validate that the task ID is within range (safety check)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= N_TOTAL=${N_TOTAL}. Exiting."
    exit 1
fi
 
SUBJ_IDX=$(( SLURM_ARRAY_TASK_ID / N_TYPE ))
TYPE_IDX=$(( SLURM_ARRAY_TASK_ID % N_TYPE ))
 
export SUBJ="${SUBJECTS[$SUBJ_IDX]}"
export SPEECH_TYPE="${SPEECH_TYPES[$TYPE_IDX]}"
 
# ---------------------------------------------------------------------------
# MATLAB environment setup
# ---------------------------------------------------------------------------
module load gcc/9.4.0
module load matlab/R2024b
 
export MATLAB_PREFDIR=/scratch/st-ssfels-1/vickywx/matlab_prefs
mkdir -p "${MATLAB_PREFDIR}"
 
export XDG_CACHE_HOME=/scratch/st-ssfels-1/vickywx/cache
mkdir -p "${XDG_CACHE_HOME}"
 
# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_hilbert
mkdir -p /scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/04_processed
 
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node:             $(hostname)"
echo "Array job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:    ${SLURM_ARRAY_TASK_ID}  (${SUBJ} | ${SPEECH_TYPE})"
echo "CPUs:             ${SLURM_CPUS_PER_TASK}"
echo "Mem:              ${SLURM_MEM_PER_NODE} MB"
 
matlab -nodisplay -nosplash -nodesktop -r \
    "try; run('/scratch/st-ssfels-1/vickywx/B2S_data_analysis/IC_hilbert_timefreq.m'); catch e; fprintf('ERROR: %s\n', e.message); exit(1); end; exit(0);"
 
EXIT_CODE=$?
 
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}