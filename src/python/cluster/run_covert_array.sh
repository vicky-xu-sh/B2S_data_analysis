#!/bin/bash
# Covert classification array job — one task per subject, one job per condition.
#
# MATCH_COND must be passed via --export at submit time (see submit_pipeline.sh).
#
# Conditions (MATCH_COND values):
#   corr0.8_allranks        corr0.8_allranks_brain
#   corr0.7_allranks        corr0.7_allranks_brain
#   corr0.8_rank1           corr0.8_rank1_brain
#   corr0.7_rank1           corr0.7_rank1_brain
#
# Subject index mapping:
#   0=subj-01  1=subj-02  2=subj-03  3=subj-04  4=subj-05
#   5=subj-06  6=subj-07  7=subj-08  8=subj-11  9=subj-12
#
# Results are saved to:
#   results/covert_classification_{MATCH_COND}/{subj}/imagined/covert_classification/
#
# Example — submit one condition manually:
#   sbatch --export=MATCH_COND=corr0.8_allranks \
#          --job-name=covert_corr0.8_allranks \
#          --array=0-1,3,5-7,9 run_covert_array.sh
#SBATCH --job-name=covert_classification
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
CORRMAP_DIR="${BASE}/data/06_corrmap_IC_match"
INPUT_DIR="${BASE}/data/04_processed"
OVERT_OUTPUT_DIR="${BASE}/results/overt_classification"
OUTPUT_DIR="${BASE}/results/covert_classification_${MATCH_COND}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_classification

export MPLCONFIGDIR=/scratch/st-ssfels-1/vickywx/cache/matplotlib
export FONTCONFIG_CACHE=/scratch/st-ssfels-1/vickywx/cache/fontconfig
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_CACHE}"

source ~/python_3_12_env/bin/activate

# ---------------------------------------------------------------------------
# Validate MATCH_COND
# ---------------------------------------------------------------------------
if [ -z "${MATCH_COND}" ]; then
    echo "[ERROR] MATCH_COND not set. Submit with --export=MATCH_COND=corr0.8_allranks etc."
    exit 1
fi

# Parse MATCH_COND → derive_covert_ic_matches.py flags
# Format: corr{threshold}_{allranks|rank1}[_brain]
CORR_THRESHOLD=$(echo "${MATCH_COND}" | grep -oP '\d+\.\d+')
RANK1_ONLY_ARG=""
echo "${MATCH_COND}" | grep -q "rank1" && RANK1_ONLY_ARG="--rank1-only"
BRAIN_ARG=""
echo "${MATCH_COND}" | grep -q "brain" && BRAIN_ARG="--brain-ics-only"

# ---------------------------------------------------------------------------
# Read sp and im config for this subject
# ---------------------------------------------------------------------------
read_config_field() {
    local cond="$1" field="$2"
    python3 -c "
import csv, sys
with open('${CONFIG_CSV}') as f:
    for row in csv.DictReader(f):
        if row['subject'] == '${SUBJ}' and row['condition'] == '$cond':
            print(row['$field'].strip())
            sys.exit(0)
print('')
"
}

OVERT_KEEP_ICS=$(read_config_field sp keep_ics)
OVERT_BAD_EPOCHS=$(read_config_field sp bad_epochs)
COVERT_KEEP_ICS=$(read_config_field im keep_ics)
COVERT_BAD_EPOCHS=$(read_config_field im bad_epochs)

# Skip subjects whose data aren't processed yet
if [ -z "${OVERT_KEEP_ICS}" ] || [ -z "${COVERT_KEEP_ICS}" ]; then
    echo "No sp/im data for ${SUBJ} in subject_config.csv — skipping."
    exit 0
fi

# "None" means no bad epochs
[ "${OVERT_BAD_EPOCHS}" = "None" ]  && OVERT_BAD_EPOCHS=""
[ "${COVERT_BAD_EPOCHS}" = "None" ] && COVERT_BAD_EPOCHS=""

OVERT_BAD_EPOCHS_ARG=""
[ -n "${OVERT_BAD_EPOCHS}" ]  && OVERT_BAD_EPOCHS_ARG="--overt-bad-epochs ${OVERT_BAD_EPOCHS}"
COVERT_BAD_EPOCHS_ARG=""
[ -n "${COVERT_BAD_EPOCHS}" ] && COVERT_BAD_EPOCHS_ARG="--covert-bad-epochs ${COVERT_BAD_EPOCHS}"

# ---------------------------------------------------------------------------
# Derive matched IC pairs from corrmap file
# ---------------------------------------------------------------------------
IC_MATCH_OUTPUT=$(python3 "${SCRIPT_DIR}/derive_covert_ic_matches.py" \
    --subj           "${SUBJ}" \
    --corrmap-dir    "${CORRMAP_DIR}" \
    --config-csv     "${CONFIG_CSV}" \
    --corr-threshold "${CORR_THRESHOLD}" \
    ${RANK1_ONLY_ARG} \
    ${BRAIN_ARG})

if [ $? -ne 0 ]; then
    echo "[ERROR] derive_covert_ic_matches.py failed for ${SUBJ} / ${MATCH_COND} — see stderr above."
    exit 1
fi

# Sets OVERT_MATCHED and COVERT_MATCHED in this shell
eval "${IC_MATCH_OUTPUT}"

# ---------------------------------------------------------------------------
# Read pre-onsets written by run_overt_array.sh
# ---------------------------------------------------------------------------
W4_REC_CSV="${OVERT_OUTPUT_DIR}/${SUBJ}/W4_sweep/${SUBJ}_sp_W4_recommended_pre_onsets.csv"
if [ ! -f "${W4_REC_CSV}" ]; then
    echo "[ERROR] W4 recommended pre-onsets not found: ${W4_REC_CSV}"
    echo "[ERROR] Run run_overt_array.sh first."
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
# Read fixed hyperparams written by classification_overt.py
# ---------------------------------------------------------------------------
FIXED_PARAMS_CSV="${OVERT_OUTPUT_DIR}/${SUBJ}/baseline_windows/${SUBJ}_sp_recommended_fixed_params.csv"
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
    echo "[ERROR] Failed to parse fixed params from ${FIXED_PARAMS_CSV}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node:           $(hostname)"
echo "Job ID:         ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Subj:           ${SUBJ}"
echo "Match cond:     ${MATCH_COND}  (corr>=${CORR_THRESHOLD})"
echo "Overt matched:  ${OVERT_MATCHED}"
echo "Covert matched: ${COVERT_MATCHED}"
echo "Best overall:   ${BEST_OVERALL_MS} ms"
echo "Consonant:      stop=${STOP_MS} nasal=${NASAL_MS} fricative=${FRICATIVE_MS} ms"
echo "Output dir:     ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Run covert classification
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/classification_covert.py" \
    --subj                       "${SUBJ}" \
    --input-dir                  "${INPUT_DIR}" \
    --output-dir                 "${OUTPUT_DIR}" \
    --overt-keep-ics             ${OVERT_KEEP_ICS} \
    --overt-matched-ics          ${OVERT_MATCHED} \
    ${OVERT_BAD_EPOCHS_ARG} \
    --covert-keep-ics            ${COVERT_KEEP_ICS} \
    --covert-matched-ics         ${COVERT_MATCHED} \
    ${COVERT_BAD_EPOCHS_ARG} \
    --best-overall-pre-onset-ms  ${BEST_OVERALL_MS} \
    --consonant-pre-onset-ms     ${STOP_MS} ${NASAL_MS} ${FRICATIVE_MS} \
    --fix-svm-C                  "${FIX_SVM_C}" \
    --fix-rf-depth               "${FIX_RF_DEPTH}" \
    --fix-rf-features            "${FIX_RF_FEATURES}" \
    --fix-rf-estimators          "${FIX_RF_ESTIMATORS}" \
    --fix-rf-split               "${FIX_RF_SPLIT}" \
    --n-permutations             200

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
