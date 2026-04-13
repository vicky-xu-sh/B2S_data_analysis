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

module load python/3.12
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

# --- Plot 2: W4 final experiments ---
echo ""
echo "--- Plot 2: W4 final experiments ---"
python "${SCRIPT}" \
    --summary-csv \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_final.csv" \
    --recall-csv \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_final.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "W4_final"

# --- Plot 3: all experiments combined ---
echo ""
echo "--- Plot 3: all experiments combined ---"
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

# --- Plot 4: key experiments only (brain ICs, zpower) ---
echo ""
echo "--- Plot 4: key experiments (brain ICs, zpower) ---"
python "${SCRIPT}" \
    --summary-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_classification_summary.csv" \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_classification_summary_W4_final.csv" \
    --recall-csv \
        "${BASE}/baseline_windows/${SUBJ}_${COND}_per_class_recall.csv" \
        "${BASE}/W4_sweep/${SUBJ}_${COND}_per_class_recall_W4_final.csv" \
    --experiments \
        W1b_brainIC_zpower_full \
        W2b_brainIC_zpower_speech500ms \
        W3a_brainIC_zpower_prespeech500ms \
        W4_ExpA_RF_brainIC_zpower_pre300ms_speech500ms \
        W4_ExpA_combined_brainIC_zpower_pre300ms_speech500ms \
        W4_ExpB_RF_brainIC_zpower_vel300ms_bil300ms_alv350ms \
        W4_ExpB_combined_brainIC_zpower_vel300ms_bil300ms_alv100ms \
    --output-dir "${OUTPUT_DIR}" \
    --subj "${SUBJ}" --cond "${COND}" \
    --tag "key_brainIC"

EXIT_CODE=$?
echo ""
echo "===== Job finished: $(date) ====="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}