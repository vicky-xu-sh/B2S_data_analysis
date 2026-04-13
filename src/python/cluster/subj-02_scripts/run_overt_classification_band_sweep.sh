#!/bin/bash
#SBATCH --job-name=band_sweep
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
# CONFIGURE — fill in after reviewing W4 sweep results
# ---------------------------------------------------------------------------
SUBJ="subj-02"
OVERT_BRAIN_ICS="4 7 14 21 22 32"
OVERT_BAD_EPOCHS="1 111"

# From W4 sweep output:
#   Best overall pre-onset (Exp A)
BEST_OVERALL_PRE_ONSET_MS=300

#   Consonant-specific pre-onsets (Exp B), ordered: stop(gi/gu) nasal(mi/mu) fricative(si/su)
CONSONANT_PRE_ONSET_MS="300 300 100"

#   Speech window — must match what was used in classification_overt.py and W4 sweep
SPEECH_WINDOW_MS=500

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

python "${SCRIPT_DIR}/classification_overt_band_sweep.py" \
    --subj                       "${SUBJ}" \
    --input-dir                  "${INPUT_DIR}" \
    --output-dir                 "${OUTPUT_DIR}" \
    --overt-brain-ics            ${OVERT_BRAIN_ICS} \
    --overt-bad-epochs           ${OVERT_BAD_EPOCHS} \
    --best-overall-pre-onset-ms  ${BEST_OVERALL_PRE_ONSET_MS} \
    --consonant-pre-onset-ms     ${CONSONANT_PRE_ONSET_MS} \
    --speech-window-ms           ${SPEECH_WINDOW_MS}

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}