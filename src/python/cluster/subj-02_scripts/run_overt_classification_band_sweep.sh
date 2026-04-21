#!/bin/bash
#SBATCH --job-name=band_sweep
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
# CONFIGURE — fill in after reviewing W4 sweep results
# ---------------------------------------------------------------------------
SUBJ="subj-02"
OVERT_KEEP_ICS="2 3 4 5 7 12 14 20 21 22 23 26 27 28 29 30 31 32 33 36 38"
OVERT_BRAIN_ICS="4 7 14 21 22 32"
OVERT_BAD_EPOCHS="1 111"

# From W4 sweep output:
#   Best overall pre-onset 
BEST_OVERALL_PRE_ONSET_MS=350

#   Consonant-specific pre-onsets, ordered: stop(gi/gu) nasal(mi/mu) fricative(si/su)
CONSONANT_PRE_ONSET_MS="250 350 200"

#   Speech window — must match what was used in classification_overt.py and W4 sweep 
#   Leave empty to derive automatically
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
echo "Best overall pre-onset: ${BEST_OVERALL_PRE_ONSET_MS} ms"
echo "Consonant pre-onsets:   ${CONSONANT_PRE_ONSET_MS}"
echo "Speech window:          ${SPEECH_WINDOW_MS} ms"

# --- Parse recommended fixed params saved ---
FIXED_PARAMS_CSV="${OUTPUT_DIR}/${SUBJ}/spoken/baseline_windows/${SUBJ}_sp_recommended_fixed_params.csv"

if [ ! -f "${FIXED_PARAMS_CSV}" ]; then
    echo "[ERROR] Recommended fixed params CSV not found: ${FIXED_PARAMS_CSV}"
    echo "[ERROR] Cannot pass fixed params to W4 sweep — aborting."
    exit 1
fi

# Extract each param value robustly (strip carriage returns from any Windows line endings)
get_csv_value() {
    local key="$1"
    awk -F',' -v key="$key" 'NR > 1 && $1 == key { print $2 }' "${FIXED_PARAMS_CSV}" \
        | tr -d '\r\n'
}

FIX_SVM_C=$(get_csv_value "fix_svm_C")
FIX_RF_DEPTH=$(get_csv_value "fix_rf_depth")
FIX_RF_FEATURES=$(get_csv_value "fix_rf_features")
FIX_RF_ESTIMATORS=$(get_csv_value "fix_rf_estimators")
FIX_RF_SPLIT=$(get_csv_value "fix_rf_split")

# Verify all values were extracted (FIX_RF_DEPTH may be the string "None" — that is valid)
if [ -z "${FIX_SVM_C}" ] || [ -z "${FIX_RF_DEPTH}" ] || \
   [ -z "${FIX_RF_FEATURES}" ] || [ -z "${FIX_RF_ESTIMATORS}" ] || \
   [ -z "${FIX_RF_SPLIT}" ]; then
    echo "[ERROR] Failed to parse all fixed params from CSV: ${FIXED_PARAMS_CSV}"
    echo "  fix_svm_C='${FIX_SVM_C}'  fix_rf_depth='${FIX_RF_DEPTH}'"
    echo "  fix_rf_features='${FIX_RF_FEATURES}'  fix_rf_estimators='${FIX_RF_ESTIMATORS}'"
    echo "  fix_rf_split='${FIX_RF_SPLIT}'"
    exit 1
fi

echo ""
echo "  Fixed params parsed from W1a:"
echo "    SVM:  C=${FIX_SVM_C}"
echo "    RF:   max_depth=${FIX_RF_DEPTH}  max_features=${FIX_RF_FEATURES}"
echo "          n_estimators=${FIX_RF_ESTIMATORS}  min_samples_split=${FIX_RF_SPLIT}"

# Build optional --speech-window-ms arg
SPEECH_WIN_ARG=""
if [ -n "${SPEECH_WINDOW_MS}" ]; then
    SPEECH_WIN_ARG="--speech-window-ms ${SPEECH_WINDOW_MS}"
fi

python "${SCRIPT_DIR}/classification_overt_band_sweep.py" \
    --subj                       "${SUBJ}" \
    --input-dir                  "${INPUT_DIR}" \
    --output-dir                 "${OUTPUT_DIR}" \
    --overt-keep-ics             ${OVERT_KEEP_ICS} \
    --overt-bad-epochs           ${OVERT_BAD_EPOCHS} \
    --best-overall-pre-onset-ms  ${BEST_OVERALL_PRE_ONSET_MS} \
    --consonant-pre-onset-ms     ${CONSONANT_PRE_ONSET_MS} \
    ${SPEECH_WIN_ARG} \
    --fix-svm-C                  "${FIX_SVM_C}" \
    --fix-rf-depth               "${FIX_RF_DEPTH}" \
    --fix-rf-features            "${FIX_RF_FEATURES}" \
    --fix-rf-estimators          "${FIX_RF_ESTIMATORS}" \
    --fix-rf-split               "${FIX_RF_SPLIT}" \
    --n-jobs                     ${SLURM_CPUS_PER_TASK}

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
