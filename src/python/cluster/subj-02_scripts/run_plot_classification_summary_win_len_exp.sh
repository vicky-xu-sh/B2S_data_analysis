#!/bin/bash
#SBATCH --job-name=plot_summary
#SBATCH --account=st-ssfels-1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_classification/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=FAIL

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
SUBJ="subj-02"
COND="sp"

BASE=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/results/classification/${SUBJ}/spoken
SCRIPT=/scratch/st-ssfels-1/vickywx/B2S_data_analysis/src/python/plot_classification_summary.py
OUTPUT_DIR=${BASE}/summary_figures

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
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
echo "Subj:   ${SUBJ} | ${COND}"

# --- Plot 1: W1 / W2 / W3 baselines ---
echo ""
echo "--- Plot 1: baseline windows ---"
python "${SCRIPT}" \
    --summary-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_classification_summary.csv" \
    --recall-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_per_class_recall.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "W1_W2_W3"

# --- Plot 2: W4 experiments ---
echo ""
echo "--- Plot 2: W4 experiments ---"
python "${SCRIPT}" \
    --summary-csv \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_sweep.csv" \
    --recall-csv \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_sweep.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "W4_sweep"

# --- Plot 3: W1 / W2 / W3 / W4 final experiments combined ---
echo ""
echo "--- Plot 3: W1 / W2 / W3 / W4 final experiments combined ---"
python "${SCRIPT}" \
    --summary-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_classification_summary.csv" \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_final.csv" \
    --recall-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_per_class_recall.csv" \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_final.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "all"

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}