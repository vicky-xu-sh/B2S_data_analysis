#!/bin/bash
#SBATCH --job-name=tf_plots_good_data
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
#SBATCH --array=0

# ---------------------------------------------------------------------------
# CONFIGURE: edit subjects to run (update --array above to match N_SUBJ - 1)
# ---------------------------------------------------------------------------
SUBJECTS=(
    "subj-02"
    "subj-03"
    "subj-04"
    "subj-05"
    "subj-06"
    "subj-07"
)

CONFIG_CSV=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/subject_config.csv

INPUT_DIR=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/04_processed/
OUTPUT_DIR=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/results/tf_plots/
FIGURE_DIR_NAME=good_data_trials_ICs

SCRIPT=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/src/python/tf_plots.py

# ---------------------------------------------------------------------------
# Resolve subject from task array index
# ---------------------------------------------------------------------------
N_SUBJ=${#SUBJECTS[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_SUBJ}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= N_SUBJ=${N_SUBJ}. Exiting."
    exit 1
fi

SUBJ_IDX=${SLURM_ARRAY_TASK_ID}
SUBJ="${SUBJECTS[$SUBJ_IDX]}"

# ---------------------------------------------------------------------------
# Parse subject config from CSV
# CSV format (no brackets, space-separated values inside cells):
#   subject,condition,bad_epochs,keep_ics
#   subj-02,sp,1 111,2 3 4 5 7 ...
# ---------------------------------------------------------------------------
parse_config() {
    local subj=$1
    local cond=$2
    awk -F',' -v subj="$subj" -v cond="$cond" '
        NR>1 && $1==subj && $2==cond {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            print $3 "|" $4
        }
    ' "${CONFIG_CSV}"
}

OVERT_ROW=$(parse_config "${SUBJ}" "sp")
COVERT_ROW=$(parse_config "${SUBJ}" "im")

if [ -z "${OVERT_ROW}" ]; then
    echo "[ERROR] No overt (sp) config found for ${SUBJ} in ${CONFIG_CSV}"
    exit 1
fi
if [ -z "${COVERT_ROW}" ]; then
    echo "[ERROR] No covert (im) config found for ${SUBJ} in ${CONFIG_CSV}"
    exit 1
fi

OVERT_BAD=$(echo "${OVERT_ROW}"   | cut -d'|' -f1)
OVERT_ICS=$(echo "${OVERT_ROW}"   | cut -d'|' -f2)
COVERT_BAD=$(echo "${COVERT_ROW}" | cut -d'|' -f1)
COVERT_ICS=$(echo "${COVERT_ROW}" | cut -d'|' -f2)

# ---------------------------------------------------------------------------
# Environment setup
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
echo "Overt  bad epochs: ${OVERT_BAD}"
echo "Overt  keep ICs:   ${OVERT_ICS}"
echo "Covert bad epochs: ${COVERT_BAD}"
echo "Covert keep ICs:   ${COVERT_ICS}"

ARGS=(
    --subj "${SUBJ}"
    --input-dir  "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --figure-dir-name "${FIGURE_DIR_NAME}"
    --overt-bad-epochs  ${OVERT_BAD}
    --overt-keep-ics    ${OVERT_ICS}
    --covert-bad-epochs ${COVERT_BAD}
    --covert-keep-ics   ${COVERT_ICS}
)

python "${SCRIPT}" "${ARGS[@]}"

EXIT_CODE=$?
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}