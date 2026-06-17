#!/bin/bash
# Plot classification summary figures for overt results — one task per subject.
# Submitted automatically by submit_pipeline.sh after the overt array completes.
#
# Subject index mapping:
#   0=subj-01  1=subj-02  2=subj-03  3=subj-04  4=subj-05
#   5=subj-06  6=subj-07  7=subj-08  8=subj-11  9=subj-12
#SBATCH --job-name=plot_overt_summary
#SBATCH --account=st-ssfels-1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%A_%a.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%A_%a.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL

SUBJECTS=(subj-01 subj-02 subj-03 subj-04 subj-05 subj-06 subj-07 subj-08 subj-11 subj-12)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
COND="sp"

BASE=/scratch/st-ssfels-1/vickywx/B2S_data_analysis
SCRIPT="${BASE}/src/python/plot_classification_summary.py"
SUBJ_DIR="${BASE}/results/overt_classification/${SUBJ}"
OUTPUT_DIR="${SUBJ_DIR}/summary_figures"

export MPLCONFIGDIR=/scratch/st-ssfels-1/vickywx/cache/matplotlib
export FONTCONFIG_CACHE=/scratch/st-ssfels-1/vickywx/cache/fontconfig
mkdir -p "${MPLCONFIGDIR}" "${FONTCONFIG_CACHE}" "${OUTPUT_DIR}"

source ~/python_3_12_env/bin/activate

# Skip if overt results don't exist for this subject
if [ ! -d "${SUBJ_DIR}/baseline_windows" ]; then
    echo "No overt results for ${SUBJ} — skipping."
    exit 0
fi

echo "===== Job started: $(date) ====="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Subj:   ${SUBJ}"

# --- Plot 1: W1 / W2 / W3 baselines ---
echo ""
echo "--- Plot 1: baseline windows ---"
python "${SCRIPT}" \
    --summary-csv "${SUBJ_DIR}/baseline_windows/${SUBJ}_${COND}_classification_summary.csv" \
    --recall-csv  "${SUBJ_DIR}/baseline_windows/${SUBJ}_${COND}_per_class_recall.csv" \
    --output-dir  "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "W1_W2_W3"

# --- Plot 2: W4 sweep ---
echo ""
echo "--- Plot 2: W4 sweep ---"
python "${SCRIPT}" \
    --summary-csv "${SUBJ_DIR}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_sweep.csv" \
    --recall-csv  "${SUBJ_DIR}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_sweep.csv" \
    --output-dir  "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "W4_sweep"

# --- Plot 3: all combined (baseline + W4 final) ---
echo ""
echo "--- Plot 3: all combined ---"
python "${SCRIPT}" \
    --summary-csv \
        "${SUBJ_DIR}/baseline_windows/${SUBJ}_${COND}_classification_summary.csv" \
        "${SUBJ_DIR}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_final.csv" \
    --recall-csv \
        "${SUBJ_DIR}/baseline_windows/${SUBJ}_${COND}_per_class_recall.csv" \
        "${SUBJ_DIR}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_final.csv" \
    --output-dir  "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "all"

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
