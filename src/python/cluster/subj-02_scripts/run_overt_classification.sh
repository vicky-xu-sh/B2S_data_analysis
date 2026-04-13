#!/bin/bash
#SBATCH --job-name=classification_overt
#SBATCH --account=st-ssfels-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
SUBJ="subj-02"
OVERT_KEEP_ICS="2 3 4 5 7 12 14 20 21 22 23 26 27 28 29 30 31 32 33 36 38"
OVERT_BRAIN_ICS="4 7 14 21 22 32"
OVERT_BAD_EPOCHS="1 111"

# Speech window in ms — set after first run to match what classification_overt.py reports.
# Leave empty to auto-derive (both scripts will independently auto-derive the same value).
SPEECH_WINDOW_MS=""

SCRIPT_DIR=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/src/python
INPUT_DIR=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/04_processed
OUTPUT_DIR=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/results/classification

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_classification

export MPLCONFIGDIR=/scratch/st-ssfels-1/vickywx/cache/matplotlib
export FONTCONFIG_CACHE=/scratch/st-ssfels-1/vickywx/cache/fontconfig
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_CACHE}"

source ~/python_3_12_env/bin/activate

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Mem:    ${SLURM_MEM_PER_NODE} MB"
echo "Subj:   ${SUBJ}"

# Build optional --speech-window-ms arg
SPEECH_WIN_ARG=""
if [ -n "${SPEECH_WINDOW_MS}" ]; then
    SPEECH_WIN_ARG="--speech-window-ms ${SPEECH_WINDOW_MS}"
fi

# --- Step 1: main classification (W1, W2, W3) ---
echo ""
echo "--- Step 1: classification_overt.py ---"
python "${SCRIPT_DIR}/classification_overt.py" \
    --subj             "${SUBJ}" \
    --input-dir        "${INPUT_DIR}" \
    --output-dir       "${OUTPUT_DIR}" \
    --overt-keep-ics   ${OVERT_KEEP_ICS} \
    --overt-brain-ics  ${OVERT_BRAIN_ICS} \
    --overt-bad-epochs ${OVERT_BAD_EPOCHS} \
    ${SPEECH_WIN_ARG}

if [ $? -ne 0 ]; then
    echo "[ERROR] classification_overt.py failed — aborting."
    exit 1
fi

# --- Step 2: W4 pre-speech sweep ---
echo ""
echo "--- Step 2: classification_overt_W4_sweep.py ---"
python "${SCRIPT_DIR}/classification_overt_W4_sweep.py" \
    --subj             "${SUBJ}" \
    --input-dir        "${INPUT_DIR}" \
    --output-dir       "${OUTPUT_DIR}" \
    --overt-brain-ics  ${OVERT_BRAIN_ICS} \
    --overt-bad-epochs ${OVERT_BAD_EPOCHS} \
    ${SPEECH_WIN_ARG}

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}