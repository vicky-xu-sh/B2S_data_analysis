#!/bin/bash
#SBATCH --job-name=covert_exp
#SBATCH --account=st-ssfels-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL

# ---------------------------------------------------------------------------
# CONFIGURE — fill in after reviewing overt results
# ---------------------------------------------------------------------------
SUBJ="subj-02"
OVERT_KEEP_ICS="2 3 4 5 7 12 14 20 21 22 23 26 27 28 29 30 31 32 33 36 38"
OVERT_MATCHED_ICS="7 7 7 21"
OVERT_BAD_EPOCHS="1 111"

COVERT_KEEP_ICS="3 4 5 7 13 14 15 16 18 19 20 22 23 24 25 28 29 34 36 39 42 46"
COVERT_MATCHED_ICS="3 4 5 7"
COVERT_BAD_EPOCHS="9 85 95 96 97 110"

# From overt W4 sweep output:
#   Best overall pre-onset 
BEST_OVERALL_PRE_ONSET_MS=350

#   Consonant-specific pre-onsets, ordered: stop(gi/gu) nasal(mi/mu) fricative(si/su)
CONSONANT_PRE_ONSET_MS="250 350 200"

#   Speech window — must match what was used in the overt scripts
#   Leave empty to derive automatically
SPEECH_WINDOW_MS=""

# If want to skip overt onset check, set to true
SKIP_OVERT_ONSET_CHECK=false

# ---------------------------------------------------------------------------

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
echo "Best overall pre-onset: ${BEST_OVERALL_PRE_ONSET_MS} ms"
echo "Consonant pre-onsets:   ${CONSONANT_PRE_ONSET_MS}"
echo "Speech window:          ${SPEECH_WINDOW_MS} ms"

# Build optional --speech-window-ms arg
SPEECH_WIN_ARG=""
if [ -n "${SPEECH_WINDOW_MS}" ]; then
    SPEECH_WIN_ARG="--speech-window-ms ${SPEECH_WINDOW_MS}"
fi

# Build optional --skip-sanity-check arg
SKIP_ONSET_CHECK_ARG=""
if [ "${SKIP_OVERT_ONSET_CHECK}" = true ]; then
    SKIP_ONSET_CHECK_ARG="--skip-sanity-check"
fi

python "${SCRIPT_DIR}/classification_covert.py" \
    --subj                       "${SUBJ}" \
    --input-dir                  "${INPUT_DIR}" \
    --output-dir                 "${OUTPUT_DIR}" \
    --overt-keep-ics             ${OVERT_KEEP_ICS} \
    --overt-matched-ics          ${OVERT_MATCHED_ICS} \
    --overt-bad-epochs           ${OVERT_BAD_EPOCHS} \
    --covert-keep-ics            ${COVERT_KEEP_ICS} \
    --covert-matched-ics         ${COVERT_MATCHED_ICS} \
    --covert-bad-epochs          ${COVERT_BAD_EPOCHS} \
    --best-overall-pre-onset-ms  ${BEST_OVERALL_PRE_ONSET_MS} \
    --consonant-pre-onset-ms     ${CONSONANT_PRE_ONSET_MS} \
    ${SPEECH_WIN_ARG} \
    ${SKIP_ONSET_CHECK_ARG} \
    --n-permutations 200


EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
