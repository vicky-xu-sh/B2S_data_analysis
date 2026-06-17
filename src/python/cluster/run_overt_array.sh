#!/bin/bash
# Overt classification array job — one task per subject.
#
# Subject index mapping (use --array to select which to run):
#   0=subj-01  1=subj-02  2=subj-03  3=subj-04  4=subj-05
#   5=subj-06  6=subj-07  7=subj-08  8=subj-11  9=subj-12
#
# Currently available subjects (non-empty in subject_config.csv):
#   --array=0-1,3,5-7,9
#
# Example — run all available subjects:
#   sbatch --array=0-1,3,5-7,9 run_overt_array.sh
#
# Example — run a single subject (subj-06, index 5):
#   sbatch --array=5 run_overt_array.sh
#SBATCH --job-name=overt_classification
#SBATCH --account=st-ssfels-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%A_%a.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%A_%a.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL

SUBJECTS=(subj-01 subj-02 subj-03 subj-04 subj-05 subj-06 subj-07 subj-08 subj-11 subj-12)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

BASE=/scratch/st-ssfels-1/vickywx/B2S_data_analysis
CONFIG_CSV="${BASE}/subject_config.csv"
SCRIPT_DIR="${BASE}/src/python"
INPUT_DIR="${BASE}/data/04_processed"
OUTPUT_DIR="${BASE}/results/overt_classification"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_classification

export MPLCONFIGDIR=/scratch/st-ssfels-1/vickywx/cache/matplotlib
export FONTCONFIG_CACHE=/scratch/st-ssfels-1/vickywx/cache/fontconfig
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_CACHE}"

source ~/python_3_12_env/bin/activate

# ---------------------------------------------------------------------------
# Read sp config for this subject from subject_config.csv
# ---------------------------------------------------------------------------
read_sp_field() {
    python3 -c "
import csv, sys
with open('${CONFIG_CSV}') as f:
    for row in csv.DictReader(f):
        if row['subject'] == '${SUBJ}' and row['condition'] == 'sp':
            print(row['$1'].strip())
            sys.exit(0)
print('')
"
}

OVERT_KEEP_ICS=$(read_sp_field keep_ics)
OVERT_BAD_EPOCHS=$(read_sp_field bad_epochs)

# Skip subjects whose data aren't processed yet (empty keep_ics)
if [ -z "${OVERT_KEEP_ICS}" ]; then
    echo "No sp data for ${SUBJ} in subject_config.csv — skipping."
    exit 0
fi

# "None" means no bad epochs to exclude
[ "${OVERT_BAD_EPOCHS}" = "None" ] && OVERT_BAD_EPOCHS=""

# Build optional CLI args
BAD_EPOCHS_ARG=""
[ -n "${OVERT_BAD_EPOCHS}" ] && BAD_EPOCHS_ARG="--overt-bad-epochs ${OVERT_BAD_EPOCHS}"

echo "===== Job started: $(date) ====="
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "Subj:        ${SUBJ}"
echo "Keep ICs:    ${OVERT_KEEP_ICS}"
echo "Bad epochs:  ${OVERT_BAD_EPOCHS:-<none>}"

# ---------------------------------------------------------------------------
# Step 1: classification_overt.py — W1a grid search + W1b–e, W2, W3
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: classification_overt.py ---"
python "${SCRIPT_DIR}/classification_overt.py" \
    --subj             "${SUBJ}" \
    --input-dir        "${INPUT_DIR}" \
    --output-dir       "${OUTPUT_DIR}" \
    --overt-keep-ics   ${OVERT_KEEP_ICS} \
    ${BAD_EPOCHS_ARG}

[ $? -ne 0 ] && { echo "[ERROR] classification_overt.py failed — aborting."; exit 1; }

# ---------------------------------------------------------------------------
# Parse fixed hyperparams written by Step 1
# ---------------------------------------------------------------------------
FIXED_PARAMS_CSV="${OUTPUT_DIR}/${SUBJ}/baseline_windows/${SUBJ}_sp_recommended_fixed_params.csv"

if [ ! -f "${FIXED_PARAMS_CSV}" ]; then
    echo "[ERROR] Fixed params CSV not found: ${FIXED_PARAMS_CSV}"
    exit 1
fi

get_fixed_param() {
    awk -F',' -v key="$1" 'NR > 1 && $1 == key { print $2 }' "${FIXED_PARAMS_CSV}" \
        | tr -d '\r\n'
}

FIX_SVM_C=$(get_fixed_param "fix_svm_C")
FIX_RF_DEPTH=$(get_fixed_param "fix_rf_depth")
FIX_RF_FEATURES=$(get_fixed_param "fix_rf_features")
FIX_RF_ESTIMATORS=$(get_fixed_param "fix_rf_estimators")
FIX_RF_SPLIT=$(get_fixed_param "fix_rf_split")

if [ -z "${FIX_SVM_C}" ] || [ -z "${FIX_RF_DEPTH}" ] || \
   [ -z "${FIX_RF_FEATURES}" ] || [ -z "${FIX_RF_ESTIMATORS}" ] || \
   [ -z "${FIX_RF_SPLIT}" ]; then
    echo "[ERROR] Failed to parse all fixed params from ${FIXED_PARAMS_CSV}"
    exit 1
fi

echo ""
echo "Fixed params from W1a:"
echo "  SVM: C=${FIX_SVM_C}"
echo "  RF:  max_depth=${FIX_RF_DEPTH}  max_features=${FIX_RF_FEATURES}"
echo "       n_estimators=${FIX_RF_ESTIMATORS}  min_samples_split=${FIX_RF_SPLIT}"

# ---------------------------------------------------------------------------
# Step 2: classification_overt_W4_sweep.py — pre-speech onset sweep
# Also writes {subj}_sp_W4_recommended_pre_onsets.csv for the covert pipeline
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: classification_overt_W4_sweep.py ---"
python "${SCRIPT_DIR}/classification_overt_W4_sweep.py" \
    --subj              "${SUBJ}" \
    --input-dir         "${INPUT_DIR}" \
    --output-dir        "${OUTPUT_DIR}" \
    --overt-keep-ics    ${OVERT_KEEP_ICS} \
    ${BAD_EPOCHS_ARG} \
    --fix-svm-C         "${FIX_SVM_C}" \
    --fix-rf-depth      "${FIX_RF_DEPTH}" \
    --fix-rf-features   "${FIX_RF_FEATURES}" \
    --fix-rf-estimators "${FIX_RF_ESTIMATORS}" \
    --fix-rf-split      "${FIX_RF_SPLIT}" \
    --n-jobs            ${SLURM_CPUS_PER_TASK}

[ $? -ne 0 ] && { echo "[ERROR] classification_overt_W4_sweep.py failed — aborting."; exit 1; }

# ---------------------------------------------------------------------------
# Parse W4 recommended pre-onsets (written by Step 2) for band sweep
# ---------------------------------------------------------------------------
W4_REC_CSV="${OUTPUT_DIR}/${SUBJ}/W4_sweep/${SUBJ}_sp_W4_recommended_pre_onsets.csv"
if [ ! -f "${W4_REC_CSV}" ]; then
    echo "[ERROR] W4 recommended pre-onsets not found: ${W4_REC_CSV}"
    exit 1
fi

get_pre_onset() {
    awk -F',' -v key="$1" 'NR > 1 && $1 == key { print $2 }' "${W4_REC_CSV}" \
        | tr -d '\r\n'
}

BEST_OVERALL_MS=$(get_pre_onset "best_overall_pre_onset_ms")
STOP_MS=$(get_pre_onset "consonant_stop_ms")
NASAL_MS=$(get_pre_onset "consonant_nasal_ms")
FRICATIVE_MS=$(get_pre_onset "consonant_fricative_ms")

if [ -z "${BEST_OVERALL_MS}" ] || [ -z "${STOP_MS}" ] || \
   [ -z "${NASAL_MS}" ] || [ -z "${FRICATIVE_MS}" ]; then
    echo "[ERROR] Failed to parse pre-onsets from ${W4_REC_CSV}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: classification_overt_band_sweep.py — frequency band sweep
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: classification_overt_band_sweep.py ---"
python "${SCRIPT_DIR}/classification_overt_band_sweep.py" \
    --subj                      "${SUBJ}" \
    --input-dir                 "${INPUT_DIR}" \
    --output-dir                "${OUTPUT_DIR}" \
    --overt-keep-ics            ${OVERT_KEEP_ICS} \
    ${BAD_EPOCHS_ARG} \
    --best-overall-pre-onset-ms "${BEST_OVERALL_MS}" \
    --consonant-pre-onset-ms    ${STOP_MS} ${NASAL_MS} ${FRICATIVE_MS} \
    --fix-svm-C                 "${FIX_SVM_C}" \
    --fix-rf-depth              "${FIX_RF_DEPTH}" \
    --fix-rf-features           "${FIX_RF_FEATURES}" \
    --fix-rf-estimators         "${FIX_RF_ESTIMATORS}" \
    --fix-rf-split              "${FIX_RF_SPLIT}" \
    --n-jobs                    ${SLURM_CPUS_PER_TASK}

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
