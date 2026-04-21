#!/usr/bin/env python3
"""
constants.py — Shared constants for EEG syllable classification pipeline.

Imported by utils.py and all classification scripts.
Edit here to change hyperparameter grids, band definitions, or sweep ranges.
"""

# ---------------------------------------------------------------------------
# EEG / experiment
# ---------------------------------------------------------------------------

SYLLABLES         = ["gi", "gu", "mi", "mu", "si", "su"]
FS                = 500       # sampling rate (Hz)
LP_CUTOFF_HZ      = 10        # low-pass cutoff for envelope smoothing (Hz)
LP_ORDER          = 4         # Butterworth filter order
BASELINE_START_MS = -450      # baseline window start (ms, relative to stimulus onset)
BASELINE_END_MS   = 0         # baseline window end (ms)
BAND_NAMES        = ['Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma']

# ---------------------------------------------------------------------------
# Classifier hyperparameter grids
# ---------------------------------------------------------------------------

C_PARAMS          = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]
KERNEL_PARAMS     = ['linear']
ESTIMATORS_PARAMS = [200, 300, 400, 500, 600]      
DEPTH_PARAMS      = [5, 10, None]
SPLITS_PARAMS     = [2, 5, 10]
MAX_FEATURES_PARAMS = ['sqrt', 'log2']

# ---------------------------------------------------------------------------
# Band index sets for band sweep experiments
# Keys are used directly in experiment names and filenames.
# ---------------------------------------------------------------------------

BAND_SETS = {
    'all_bands':       [0, 1, 2, 3, 4],
    'theta':           [0],
    'alpha':           [1],
    'beta':            [2],
    'gamma':           [3],
    'high_gamma':      [4],
    'mu':              [1, 2],       # alpha + beta
    'wide_gamma':      [3, 4],       # gamma + high gamma
    'beta_wide_gamma': [2, 3, 4],    # beta + gamma + high gamma
}

# ---------------------------------------------------------------------------
# Window experiment parameters
# ---------------------------------------------------------------------------

# W3: pre-speech only window
W3_PRE_ONSET_MS = 500    # window: [onset − 500ms, onset]

# W4: pre-speech sweep values prepended to speech window
W4_SWEEP_MS     = [100, 150, 200, 250, 300, 350, 400]

# Consonant groups for class-specific pre-onset experiment (Exp B)
# Keys are group names, values are (label_indices, description)
# Label indices are 1-indexed matching SYLLABLES list
CONSONANT_GROUPS = {
    'velar stop': [1, 2],           # gi, gu
    'bilabial nasal': [3, 4],       # mi, mu
    'alveolar fricative': [5, 6],   # si, su
}

# Minimum guaranteed post-onset duration for class-specific window (ms)
# Total window = max(pre_onsets) + CONSONANT_POST_BASE_MS
CONSONANT_POST_BASE_MS = 300
