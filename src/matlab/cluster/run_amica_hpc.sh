#!/bin/bash
#SBATCH --job-name=eeg_amica
#SBATCH --account=st-ssfels-1
#SBATCH --time=3:00:00                 # AMICA on 256ch EEG can take several hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # passed to AMICA as max_threads
#SBATCH --mem=64G
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ---------------------------------------------------------------------------
# Input/output paths and parameters — edit these for each run
# ---------------------------------------------------------------------------
SUBJ="subj-02"
SPEECH_TYPE="sp"
BASE="/scratch/st-ssfels-1/vickywx/EEG_preproc_output/${SUBJ}/${SPEECH_TYPE}"

export MAX_ITER="3000"
export N_PCA="70"

export INPUT_DIR="${BASE}/datasets"
export INPUT_SET_NAME="subj-02_pilot_sp_bp_1_150hz_bad_data_removed_full_transferred_ICs_marked_non_brain" 

export OUTPUT_DIR="${BASE}/datasets"
export OUTPUT_SET_NAME="subj-02_pilot_sp_bp_1_150hz_bad_data_removed_cleaned_2ndICA_AMICA_${N_PCA}comps"

export AMICA_DIR="${BASE}/amica_dir2_${N_PCA}comps"

# ---------------------------------------------------------------------------
# MATLAB environment setup
# ---------------------------------------------------------------------------
module load gcc/9.4.0
module load matlab/R2024b

export MATLAB_PREFDIR=/scratch/st-ssfels-1/vickywx/matlab_prefs
mkdir -p $MATLAB_PREFDIR

export XDG_CACHE_HOME=/scratch/st-ssfels-1/vickywx/cache
mkdir -p $XDG_CACHE_HOME

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${AMICA_DIR}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Input:  ${INPUT_DIR}/${INPUT_SET_NAME}.set"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_SET_NAME}.set"

matlab -nodisplay -nosplash -nodesktop -r \
    "try; run('/scratch/st-ssfels-1/vickywx/run_amica_hpc.m'); catch e; fprintf('ERROR: %s\n', e.message); exit(1); end; exit(0);"

echo "===== Job finished: $(date) ====="