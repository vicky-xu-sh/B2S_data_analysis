#!/bin/bash
#SBATCH --job-name=eeg_preproc
#SBATCH --account=st-ssfels-1
#SBATCH --time=12:00:00                 # adjust based on AMICA runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # passed to AMICA as max_threads
#SBATCH --mem=64G                       # AMICA on 256-ch EEG needs ~32-64GB
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_preproc/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_preproc/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ---------------------------------------------------------------------------
# Subject and condition — edit here to run a specific subject
# ---------------------------------------------------------------------------
export SUBJ="subj-04"
export SPEECH_TYPE="im"      # "sp" = spoken/overt | "im" = imagined/covert

# ---------------------------------------------------------------------------
# MATLAB environment setup
# ---------------------------------------------------------------------------
module load gcc/9.4.0
module load matlab/R2024b

# Redirect MATLAB prefs to scratch (home is read-only on compute nodes)
export MATLAB_PREFDIR=/scratch/st-ssfels-1/vickywx/matlab_prefs
mkdir -p $MATLAB_PREFDIR

export XDG_CACHE_HOME=/scratch/st-ssfels-1/vickywx/cache
mkdir -p $XDG_CACHE_HOME

# ---------------------------------------------------------------------------
# Create output log directory
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_preproc

# ---------------------------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node: $(hostname)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "Subject: ${SUBJ} | Speech type: ${SPEECH_TYPE}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

matlab -nodisplay -nosplash -nodesktop -r \
    "try; run('/scratch/st-ssfels-1/vickywx/preproc_hpc.m'); catch e; fprintf('ERROR: %s\n', e.message); exit(1); end; exit(0);"

echo "===== Job finished: $(date) ====="