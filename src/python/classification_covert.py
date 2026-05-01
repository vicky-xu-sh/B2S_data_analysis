#!/usr/bin/env python3
"""
classification_covert.py — EEG syllable classification (covert/imagined speech)

Pipeline overview
-----------------
1. Load and preprocess overt + covert analytic signals (same Hilbert/z-score/smooth
   pipeline as overt scripts).
2. Overt sanity check: build class templates on k-1 folds of overt trials, apply
   sliding Pearson correlation to held-out fold, compare estimated vs. true acoustic
   onset — reports MAE, median AE, % within ±50 ms.
3. For each window condition x band set x IC condition:
     a. Build class-specific overt neural templates (mean z_power per class).
     b. Slide each template across the corresponding covert trial using Pearson
        correlation; peak position = estimated imagery onset.
     c. Extract covert feature window anchored at estimated onset.
     d. Classify with SVM + RF 
4. Baseline: direct classification on full covert epoch (0 → 1950 ms).

Window conditions (matching overt experiments)
----------------------------------------------
  W1  - full epoch baseline: template/window = [0, 1950 ms]
  W2  — speech only:         template/window = [onset, onset + N ms]
  W3  — pre-speech only:     template/window = [onset - 500 ms, onset]
  W4u — uniform pre-onset:   template/window = [onset - p* ms, onset + N ms]
  W4c — consonant-specific:  per-group pre-onset, total window fixed

IC conditions for classification feature extraction
----------------------------------------------------
  keep    — all covert keep ICs (--covert-keep-ics)
  matched — topology-matched covert ICs only (--covert-matched-ics)

  Onset estimation always uses topology-matched ICs (--covert-matched-ics),
  since overt templates are built from overt brain ICs (--overt-matched-ics) and
  these map 1-to-1 via CORRMAP.  The number of --covert-matched-ics must equal
  the number of --overt-matched-ics.

Band sets
---------
  Subset of BAND_SETS from constants.py (default: all_bands, beta, gamma,
  high_gamma, wide_gamma, beta_wide_gamma).  The same band set is used for
  both template construction (onset estimation) and covert feature extraction.

Outputs
-------
  {output_dir}/{subj}/imagined/covert_classification/
    *_classification_summary_*.csv
    *_per_class_recall_*.csv
    *_onset_sanity_*.csv
    *_onset_sanity_*.png
    *_{SVM|RF}_cm.png
    *_RF_importance_*.png

Usage
-----
  python classification_covert.py \\
      --subj subj-02 \\
      --input-dir  /path/to/04_processed/ \\
      --output-dir       /path/to/results/ \\
      --overt-matched-ics   ... \\
      --overt-bad-epochs  ... \\
      --covert-keep-ics   ... \\
      --covert-matched-ics ... \\
      --covert-bad-epochs  ... \\
      --speech-window-ms  ... \\
      --best-overall-pre-onset-ms ... \\
      --consonant-pre-onset-ms ... \\
      --fix-svm-C 0.01 \\
      --fix-rf-depth None \\
      --fix-rf-features sqrt \\
      --fix-rf-estimators 300 \\
      --fix-rf-split 2

Notes
-----
- Topology matching (CORRMAP) is performed in MATLAB; matched IC indices are
  supplied here as constants via --covert-matched-ics.
- TODO: Permutation baseline 
"""

from constants import (
    SYLLABLES, FS, LP_CUTOFF_HZ, W3_PRE_ONSET_MS,
    BAND_SETS, CONSONANT_GROUPS, CONSONANT_POST_BASE_MS,
)
from utils import (
    load_analytic, compute_features, reject_epochs, compute_speech_window_tp,
    derive_epoch_indices, lowpass_smooth,
    build_X, build_X_speech_window,
    run_classifiers, plot_feature_importance, print_and_save_summary,
)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from joblib import Parallel, delayed

CONSONANT_GROUP_NAMES = list(CONSONANT_GROUPS.keys())

# Default band sets for covert experiments (subset of BAND_SETS)
DEFAULT_COVERT_BAND_SETS = [
    'all_bands', 'beta', 'gamma', 'high_gamma', 'wide_gamma', 'beta_wide_gamma'
]

# Search region (relative to t = 0 ms in epoch)
SEARCH_START_MS_DEFAULT = 0     # ms
SEARCH_END_MS_DEFAULT   = 1950  # ms  

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_group_name(cls_label_1idx):
    """Return consonant group name for a 1-indexed class label."""
    for group_name, labels in CONSONANT_GROUPS.items():
        if cls_label_1idx in labels:
            return group_name
    return None


def _safe_tag(s):
    for ch in (' ', '|', '/', '(', ')', '[', ']', ',', '+', '→', '-'):
        s = s.replace(ch, '_')
    return s.strip('_')


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def build_class_templates(z_power, y, onset_tps,
                           ic_0idx, band_idx,
                           pre_tp, post_tp,
                           trial_mask=None):
    """
    Build class-specific mean neural templates from (a subset of) overt trials.

    Template window per trial: [onset_tp - pre_tp,  onset_tp + post_tp]

    Parameters
    ----------
    z_power    : np.ndarray [ICs x bands x time x trials]
    y          : np.ndarray [trials], 1-indexed int class labels
    onset_tps  : np.ndarray [trials], sample index of acoustic onset
    ic_0idx    : list of int, 0-indexed IC indices (overt brain ICs)
    band_idx   : list of int, band indices
    pre_tp     : int, samples before onset included in template
    post_tp    : int, samples after onset included in template
    trial_mask : np.ndarray bool [trials] or None — use all trials if None

    Returns
    -------
    templates : dict {class_label (int): np.ndarray [nICs x nBands x win_len]}
                Zero array for any class with no usable trials.
    win_len   : int, template length in samples (pre_tp + post_tp)
    """
    nICs    = len(ic_0idx)
    nBands  = len(band_idx)
    win_len = pre_tp + post_tp
    total_T = z_power.shape[2]

    if trial_mask is None:
        trial_mask = np.ones(z_power.shape[-1], dtype=bool)

    templates = {}
    for cls in range(1, len(SYLLABLES) + 1):
        cls_mask = trial_mask & (y == cls)
        cls_idx  = np.where(cls_mask)[0]

        accum = []
        for i in cls_idx:
            start = int(onset_tps[i]) - pre_tp
            end   = int(onset_tps[i]) + post_tp
            if start < 0 or end > total_T:
                continue
            win = z_power[np.ix_(ic_0idx, band_idx,
                                  list(range(start, end)), [i])][:, :, :, 0]
            accum.append(win)

        templates[cls] = (np.mean(accum, axis=0)
                          if len(accum) > 0
                          else np.zeros((nICs, nBands, win_len)))

    return templates, win_len


def build_class_templates_consonant(z_power, y, onset_tps,
                                     ic_0idx, band_idx,
                                     consonant_pre_tps, post_tp_base,
                                     trial_mask=None):
    """
    Build class-specific templates for the W4-consonant condition.

    Each consonant group has its own pre_tp, so template win_len varies by class.

    Parameters
    ----------
    consonant_pre_tps : dict {group_name: pre_tp (samples)}
    post_tp_base      : int, post-onset length in samples (same for all groups;
                        only the pre-onset varies per consonant group)

    Returns
    -------
    templates    : dict {class_label: np.ndarray [nICs x nBands x group_win_len]}
    group_pre_tps: dict {group_name: pre_tp} (echo for callers)
    """
    total_T = z_power.shape[2]
    nICs    = len(ic_0idx)
    nBands  = len(band_idx)

    if trial_mask is None:
        trial_mask = np.ones(z_power.shape[-1], dtype=bool)

    templates = {}
    for cls in range(1, len(SYLLABLES) + 1):
        group_name = _get_group_name(cls)
        pre_tp     = consonant_pre_tps.get(group_name, 0)
        win_len    = pre_tp + post_tp_base

        cls_mask = trial_mask & (y == cls)
        cls_idx  = np.where(cls_mask)[0]

        accum = []
        for i in cls_idx:
            start = int(onset_tps[i]) - pre_tp
            end   = int(onset_tps[i]) + post_tp_base
            if start < 0 or end > total_T:
                continue
            win = z_power[np.ix_(ic_0idx, band_idx,
                                  list(range(start, end)), [i])][:, :, :, 0]
            accum.append(win)

        templates[cls] = (np.mean(accum, axis=0)
                          if len(accum) > 0
                          else np.zeros((nICs, nBands, win_len)))

    return templates, consonant_pre_tps


# ---------------------------------------------------------------------------
# Sliding Pearson correlation
# ---------------------------------------------------------------------------

def _pearson_r(a, b):
    """Fast Pearson r between two 1-D arrays (zero-variance → 0.0)."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    return float(np.dot(a_c, b_c) / denom) if denom > 1e-10 else 0.0


def slide_template(trial_3d, template_flat, win_len,
                   search_start, search_end, step=1,
                   prominence_threshold=0.05,
                   min_peak_distance_ms=200):
    """
    Slide a flattened template across a 3-D trial [ICs x bands x time].

    Parameters
    ----------
    trial_3d      : np.ndarray [nICs x nBands x total_T]
    template_flat : np.ndarray [nICs * nBands * win_len]
    win_len       : int
    search_start  : int, first valid window start (absolute sample index)
    search_end    : int, exclusive upper bound for window start
                    (caller ensures search_end - win_len ≥ search_start)
    step          : int, stride for sliding (default 1)

    prominence_threshold   : float, minimum prominence to be considered a 
                             real peak (filters noise peaks). Default 0.05.
    min_peak_distance_ms   : int, minimum separation between peaks in ms.
                             Peaks closer than this are treated as one broad 
                             peak. Default 200ms (= 100 samples at 500Hz, 
                             step=1). Prevents double-counting a single broad 
                             peak split by a tiny dip.

    Returns
    -------
    raw_corr      : np.ndarray [n_positions], unsmoothed Pearson r at each start position
    smoothed_corr : np.ndarray [n_positions], Gaussian-smoothed Pearson r at each start position
    peak_s        : int, absolute sample index of peak correlation window start
    """
    positions  = range(search_start, search_end - win_len + 1, step)
    t_flat_std = np.std(template_flat)

    if t_flat_std < 1e-10 or len(positions) == 0:
        z = np.zeros(max(1, len(positions)))
        return z, z, search_start

    corr_vals = np.empty(len(positions))
    for j, s in enumerate(positions):
        window = trial_3d[:, :, s:s + win_len].flatten()
        corr_vals[j] = _pearson_r(template_flat, window)

    smoothed_corr = gaussian_filter1d(corr_vals, sigma=5)

    # peaks, _ = find_peaks(smoothed_corr, prominence=0.05, width=(10, win_len * 2))
    # if len(peaks) > 0:
    #     peak_j = peaks[int(np.argmax(smoothed_corr[peaks]))]
    # else:
    #     peak_j = int(np.nanargmax(smoothed_corr))

    # peak_s = search_start + peak_j * step
    # return corr_vals, smoothed_corr, peak_s

    # --- NEW PEAK SELECTION ---
    min_distance_samples = int(min_peak_distance_ms / 1000 * FS / step)
    
    peaks, props = find_peaks(
        smoothed_corr,
        prominence=prominence_threshold,
        distance=min_distance_samples   # enforce minimum separation
    )
    
    if len(peaks) == 0:
        # No peaks above threshold — fall back to global max
        peak_j = int(np.nanargmax(smoothed_corr))
        peak_prominence = 0.0
        n_competing = 0
        
    elif len(peaks) == 1:
        # Unambiguous single peak
        peak_j = peaks[0]
        peak_prominence = float(props['prominences'][0])
        n_competing = 0
        
    else:
        # Multiple peaks — rank by prominence, not height
        prominences = props['prominences']
        ranked      = np.argsort(prominences)[::-1]   # descending
        peak_j      = peaks[ranked[0]]
        peak_prominence      = float(prominences[ranked[0]])
        second_prominence    = float(prominences[ranked[1]])
        
        # If top-2 prominences are within 20% of each other, break the tie
        # by choosing the peak with the higher correlation value
        n_competing = int(np.sum(prominences >= 0.8 * peak_prominence))
        if n_competing > 1:
            competing_idx = np.where(prominences >= 0.8 * peak_prominence)[0]
            competing_peaks = peaks[competing_idx]
            peak_j = competing_peaks[int(np.argmax(smoothed_corr[competing_peaks]))]
            peak_prominence = float(prominences[competing_idx[np.argmax(smoothed_corr[competing_peaks])]])
    
    peak_s = search_start + peak_j * step
    return corr_vals, smoothed_corr, peak_s, peak_prominence, n_competing


# ---------------------------------------------------------------------------
# Onset estimation (standard and consonant-specific)
# ---------------------------------------------------------------------------

def estimate_covert_onsets(z_power_covert, y_covert,
                            templates, win_len,
                            matched_ic_0idx, band_idx,
                            pre_tp, search_start, search_end, step=1):
    """
    Estimate speech onset for each covert trial (standard window conditions).

    Uses the class-specific template (true label from experimental paradigm).
    Onset estimation always performed on topology-matched covert ICs.

    Parameters
    ----------
    z_power_covert  : np.ndarray [ICs x bands x time x trials], full covert data
    y_covert        : np.ndarray [trials], 1-indexed class labels
    templates       : dict {class_label: np.ndarray [nICs x nBands x win_len]}
    win_len         : int, template length (same for all classes here)
    matched_ic_0idx : list of int, 0-indexed topology-matched covert IC indices
    band_idx        : list of int
    pre_tp          : int, samples before onset encoded at template start position
                      (= 0 for W2, = pre_tp for W3 and W4u)
    search_start    : int, absolute sample index — first valid window start
    search_end      : int, exclusive upper bound for window start
    step            : int

    Returns
    -------
    estimated_onset_tps : np.ndarray [trials], estimated onset (absolute sample index)
    peak_start_tps      : np.ndarray [trials], start of peak-corr window (= estimated_onset - pre_tp)
    peak_corrs          : np.ndarray [trials], Pearson r at peak
    """
    nTrials = z_power_covert.shape[-1]
    estimated_onset_tps = np.zeros(nTrials, dtype=int)
    peak_start_tps      = np.zeros(nTrials, dtype=int)
    peak_corrs          = np.zeros(nTrials)

    for i in range(nTrials):
        cls         = int(y_covert[i])
        template_3d = templates.get(cls)

        if template_3d is None or np.all(template_3d == 0):
            # Degenerate template — fall back to search_start
            peak_start_tps[i]      = search_start
            estimated_onset_tps[i] = search_start + pre_tp
            peak_corrs[i]          = 0.0
            continue

        template_flat = template_3d.flatten()
        trial_3d      = z_power_covert[
            np.ix_(matched_ic_0idx, band_idx,
                   list(range(z_power_covert.shape[2])), [i])][:, :, :, 0]

        _, _, peak_s, peak_prominence, n_competing = slide_template(trial_3d, template_flat, win_len,
                                      search_start, search_end, step)

        peak_start_tps[i]      = peak_s
        estimated_onset_tps[i] = peak_s + pre_tp
        peak_corrs[i]          = _pearson_r(
            template_flat,
            trial_3d[:, :, peak_s:peak_s + win_len].flatten())

    return estimated_onset_tps, peak_start_tps, peak_corrs


def estimate_covert_onsets_consonant(z_power_covert, y_covert,
                                      templates_consonant,
                                      consonant_pre_tps, post_tp_base,
                                      matched_ic_0idx, band_idx,
                                      search_start, search_end, step=1):
    """
    Estimate onsets for W4-consonant condition (group-specific template size).

    Parameters
    ----------
    templates_consonant : dict {class_label: np.ndarray [nICs x nBands x group_win_len]}
    consonant_pre_tps   : dict {group_name: pre_tp (samples)}
    post_tp_base        : int, post-onset samples (same for all groups; pre-onset varies per group)

    Returns
    -------
    estimated_onset_tps : np.ndarray [trials]
    peak_start_tps      : np.ndarray [trials]
    peak_corrs          : np.ndarray [trials]
    """
    nTrials = z_power_covert.shape[-1]
    estimated_onset_tps = np.zeros(nTrials, dtype=int)
    peak_start_tps      = np.zeros(nTrials, dtype=int)
    peak_corrs          = np.zeros(nTrials)

    for i in range(nTrials):
        cls         = int(y_covert[i])
        group_name  = _get_group_name(cls)
        pre_tp      = consonant_pre_tps.get(group_name, 0)
        win_len     = pre_tp + post_tp_base

        template_3d = templates_consonant.get(cls)
        if template_3d is None or np.all(template_3d == 0):
            peak_start_tps[i]      = search_start
            estimated_onset_tps[i] = search_start + pre_tp
            peak_corrs[i]          = 0.0
            continue

        template_flat = template_3d.flatten()
        trial_3d      = z_power_covert[
            np.ix_(matched_ic_0idx, band_idx,
                   list(range(z_power_covert.shape[2])), [i])][:, :, :, 0]

        _, _, peak_s, peak_prominence, n_competing = slide_template(trial_3d, template_flat, win_len,
                                      search_start, search_end, step)

        peak_start_tps[i]      = peak_s
        estimated_onset_tps[i] = peak_s + pre_tp
        peak_corrs[i]          = _pearson_r(
            template_flat,
            trial_3d[:, :, peak_s:peak_s + win_len].flatten())

    return estimated_onset_tps, peak_start_tps, peak_corrs


# ---------------------------------------------------------------------------
# Covert feature extraction (W4-consonant)
# ---------------------------------------------------------------------------

def build_X_covert_consonant(z_power_covert, y_covert,
                              estimated_onset_tps,
                              ic_0idx, band_idx,
                              consonant_pre_tps, post_tp_base):
    """
    Build equal-length feature matrix for W4-consonant condition.

    Window layout (matches overt W4c):
      total_win  = max_pre_tp + post_tp_base   (same for every trial)
      pre_tp     = consonant_pre_tps[group]    (group-specific)
      post_tp    = total_win - pre_tp          (≥ post_tp_base for all groups;
                                                exactly post_tp_base for the
                                                group with the longest pre-onset)
      start = estimated_onset - pre_tp
      end   = estimated_onset + post_tp

    Parameters
    ----------
    z_power_covert      : np.ndarray [ICs x bands x time x trials]
    y_covert            : np.ndarray [trials], 1-indexed class labels (to look
                          up each trial's consonant group)
    estimated_onset_tps : np.ndarray [trials], estimated onset sample indices
    ic_0idx             : list of int (all-keep or matched, depending on condition)
    band_idx            : list of int
    consonant_pre_tps   : dict {group_name: pre_tp (samples)}
    post_tp_base        : int, minimum post-onset samples guaranteed for every trial

    Returns
    -------
    X            : np.ndarray [trials x features]
    total_win_tp : int
    """
    max_pre_tp  = max(consonant_pre_tps.values())
    total_win   = max_pre_tp + post_tp_base
    nTrials     = z_power_covert.shape[-1]
    nICs        = len(ic_0idx)
    nBands      = len(band_idx)
    total_T     = z_power_covert.shape[2]

    print(f'  Building W4c covert feature matrix: {nICs} ICs, {nBands} bands, '
          f'{total_win} samples (max_pre={max_pre_tp}, min_post={post_tp_base})')

    X = np.zeros((nICs, nBands, total_win, nTrials))
    for i in range(nTrials):
        group_name = _get_group_name(int(y_covert[i]))
        pre_tp_i   = consonant_pre_tps.get(group_name, 0)
        post_tp_i  = total_win - pre_tp_i   # absorbs remaining samples after group pre-onset
        start = int(estimated_onset_tps[i]) - pre_tp_i
        end   = int(estimated_onset_tps[i]) + post_tp_i
        if start < 0 or end > total_T:
            print(f'    Trial {i} ({group_name}): window [{start}, {end}] OOB — zero-filled')
            continue
        X[:, :, :, i] = z_power_covert[
            np.ix_(ic_0idx, band_idx, list(range(start, end)), [i])][:, :, :, 0]

    X = X.transpose(3, 0, 1, 2)
    return X.reshape(nTrials, -1), total_win


# ---------------------------------------------------------------------------
# Overt sanity check (held-out CV)
# ---------------------------------------------------------------------------

def sanity_check_overt_onset(z_power_overt, y_overt, onset_tps,
                              ic_0idx, band_idx,
                              pre_tp, post_tp,
                              search_start, search_end,
                              idx_0ms,
                              n_folds=5, step=1,
                              save_dir=None, subj=None,
                              cond_code='sp', tag=''):
    """
    Validate overt onset estimation via stratified k-fold held-out CV.

    In each fold: templates built from training trials, sliding correlation
    applied to held-out trials where the true acoustic onset is known.

    Parameters
    ----------
    z_power_overt : np.ndarray [ICs x bands x time x trials]
    y_overt       : np.ndarray [trials], 1-indexed labels
    onset_tps     : np.ndarray [trials], true onset sample indices (absolute)
    ic_0idx       : list of int (list of overt match ICs — same as used for templates)
    band_idx      : list of int
    pre_tp        : int, samples before onset in template
    post_tp       : int, samples after onset in template
    search_start  : int, absolute sample index for search start
    search_end    : int, exclusive bound for window start
    idx_0ms       : int, sample index of t = 0 ms in epoch (for ms conversion)
    n_folds       : int
    step          : int, slide step size
    save_dir      : str or None
    subj          : str or None
    cond_code     : str
    tag           : str, label for filenames

    Returns
    -------
    metrics : dict with MAE_ms, median_AE_ms, pct_within_50ms, n_valid
    """
    win_len  = pre_tp + post_tp
    nTrials  = len(y_overt)
    skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_errors_ms    = np.full(nTrials, np.nan)
    all_estimated_ms = np.full(nTrials, np.nan)

    # True onset in ms relative to t=0
    true_onset_ms = (onset_tps - idx_0ms) / FS * 1000

    trial_records = []  # collect per-trial corr curves for example plots

    print(f'\n  Overt sanity check — {tag}  '
          f'(win_len={win_len} samples, {n_folds}-fold CV)')

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(nTrials), y_overt)):
        train_mask = np.isin(np.arange(nTrials), train_idx)

        templates_fold, _ = build_class_templates(
            z_power_overt, y_overt, onset_tps,
            ic_0idx, band_idx, pre_tp, post_tp,
            trial_mask=train_mask)

        for i in test_idx:
            cls         = int(y_overt[i])
            template_3d = templates_fold.get(cls)
            if template_3d is None or np.all(template_3d == 0):
                continue

            trial_3d = z_power_overt[
                np.ix_(ic_0idx, band_idx,
                       list(range(z_power_overt.shape[2])), [i])][:, :, :, 0]

            raw_corr, smoothed_corr, peak_s, peak_prominence, n_competing = slide_template(
                trial_3d, template_3d.flatten(), win_len,
                search_start, search_end, step)

            est_onset_s       = peak_s + pre_tp
            true_onset_s      = int(onset_tps[i])
            error_ms          = abs(est_onset_s - true_onset_s) / FS * 1000

            all_errors_ms[i]    = error_ms
            all_estimated_ms[i] = (est_onset_s - idx_0ms) / FS * 1000

            n_pos = len(smoothed_corr)
            positions_ms = (search_start + np.arange(n_pos) * step - idx_0ms) / FS * 1000
            trial_records.append({
                'trial_idx':    i,
                'class_label':  cls,
                'error_ms':     error_ms,
                'true_ms':      (true_onset_s - idx_0ms) / FS * 1000,
                'estimated_ms': (est_onset_s  - idx_0ms) / FS * 1000,
                'raw_corr':     raw_corr.copy(),
                'corr_vals':    smoothed_corr.copy(),
                'positions_ms': positions_ms,
            })

        print(f'    Fold {fold_i + 1}/{n_folds}: {len(test_idx)} held-out trials done')

    valid         = ~np.isnan(all_errors_ms)
    errors_ms     = all_errors_ms[valid]
    estimated_ms  = all_estimated_ms[valid]
    true_ms_valid = true_onset_ms[valid]

    mae_ms   = float(np.mean(errors_ms))
    med_ms   = float(np.median(errors_ms))
    pct_50   = float(np.mean(errors_ms <= 50) * 100)
    n_valid  = int(valid.sum())

    sep = '-' * 60
    print(f'\n  {sep}')
    print(f'  Sanity check results  [{tag}]')
    print(f'  {sep}')
    print(f'  n trials evaluated : {n_valid}')
    print(f'  MAE                : {mae_ms:.1f} ms')
    print(f'  Median AE          : {med_ms:.1f} ms')
    print(f'  % within ±50 ms   : {pct_50:.1f} %')
    print(f'  {sep}\n')

    if save_dir and subj:
        _plot_onset_sanity(true_ms_valid, estimated_ms, errors_ms,
                           save_dir, subj, cond_code, tag, mae_ms, pct_50)
        _plot_onset_curves_examples(trial_records, save_dir, subj, cond_code, tag)

    return {
        'tag':             tag,
        'win_condition':   tag,
        'MAE_ms':          round(mae_ms, 2),
        'median_AE_ms':    round(med_ms, 2),
        'pct_within_50ms': round(pct_50, 2),
        'n_valid':         n_valid,
    }


def _plot_onset_sanity(true_ms, estimated_ms, errors_ms,
                        save_dir, subj, cond_code, tag, mae_ms, pct_50):
    """Two-panel onset sanity check diagnostic figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: true vs estimated
    ax1.scatter(true_ms, estimated_ms, alpha=0.6, s=25, zorder=3)
    lims = [min(true_ms.min(), estimated_ms.min()) - 60,
            max(true_ms.max(), estimated_ms.max()) + 60]
    ax1.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect estimation')
    ax1.set_xlabel('True acoustic onset (ms from t=0)')
    ax1.set_ylabel('Estimated onset (ms from t=0)')
    ax1.set_title('True vs estimated onset (overt held-out)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    # Histogram of absolute errors
    ax2.hist(errors_ms, bins=20, edgecolor='k', color='steelblue', alpha=0.85)
    ax2.axvline(mae_ms, color='red', linestyle='--',
                label=f'MAE = {mae_ms:.1f} ms')
    ax2.axvline(50, color='orange', linestyle=':', linewidth=1.5,
                label=f'±50 ms  ({pct_50:.1f}% within)')
    ax2.set_xlabel('|Estimated - True onset| (ms)')
    ax2.set_ylabel('Trial count')
    ax2.set_title('Onset estimation error distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)

    fig.suptitle(f'{subj} | Overt onset sanity check — {tag}', fontsize=11)
    plt.tight_layout()

    fpath = os.path.join(save_dir,
                         f'{subj}_spoken_onset_check_{_safe_tag(tag)}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved sanity plot: {fpath}')


def _plot_onset_curves_examples(trial_records, save_dir, subj, cond_code, tag, n_each=5):
    """Plot n_each good and n_each terrible onset correlation curves on one figure."""
    if not trial_records:
        return

    sorted_recs  = sorted(trial_records, key=lambda r: r['error_ms'])
    good         = sorted_recs[:n_each]
    terrible     = sorted_recs[-n_each:]

    n_rows = max(len(good), len(terrible))
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    col_labels = ['Good estimates (lowest error)', 'Terrible estimates (highest error)']

    for col, records in enumerate([good, terrible]):
        for row, rec in enumerate(records):
            ax = axes[row, col]
            ax.plot(rec['positions_ms'], rec['raw_corr'],  color='steelblue', lw=0.7,
                    alpha=0.4, label='Raw r'      if row == 0 else '_nolegend_')
            ax.plot(rec['positions_ms'], rec['corr_vals'], color='steelblue', lw=1.4,
                    label='Smoothed r' if row == 0 else '_nolegend_')
            ax.axvline(rec['true_ms'],      color='green', linestyle='--', lw=1.5,
                       label='True onset'  if row == 0 else '_nolegend_')
            ax.axvline(rec['estimated_ms'], color='red',   linestyle=':',  lw=1.5,
                       label='Est. onset'  if row == 0 else '_nolegend_')
            syl = SYLLABLES[rec['class_label'] - 1] if rec['class_label'] <= len(SYLLABLES) else '?'
            title = f'Trial {rec["trial_idx"] + 1} | {syl} | err={rec["error_ms"]:.0f} ms'
            if row == 0:
                title = f'{col_labels[col]}\n{title}'
            ax.set_title(title, fontsize=9)
            ax.set_ylabel('Pearson r', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('Position (ms from t=0)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)

        # hide unused rows if good/terrible have fewer than n_rows entries
        for row in range(len(records), n_rows):
            axes[row, col].set_visible(False)

    fig.suptitle(f'{subj} | Overt onset curve examples — {tag}', fontsize=11)
    plt.tight_layout()

    fpath = os.path.join(save_dir,
                         f'{subj}_spoken_onset_curves_{_safe_tag(tag)}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved onset curve examples: {fpath}')


def save_sanity_metrics(metrics_list, save_dir, subj, cond_code):
    """Save all window sanity check metrics to CSV."""
    df = pd.DataFrame(metrics_list)
    fpath = os.path.join(save_dir, f'{subj}_spoken_onset_check.csv')
    df.to_csv(fpath, index=False)
    print(f'  Sanity check metrics saved: {fpath}')


# ---------------------------------------------------------------------------
# Permutation baseline
# ---------------------------------------------------------------------------

def _classify_cv_fast(X, y, fp, n_splits=5, random_state=42):
    """Cross-validated SVM+RF with fixed params. No plots. Returns (svm_acc, svm_bal, rf_acc, rf_bal)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=fp['fix_svm_C'], kernel='linear')),
    ])
    svm_pipe.fit(X, y)
    y_pred_svm = cross_val_predict(svm_pipe, X, y, cv=cv, n_jobs=1)
    svm_acc = float((y_pred_svm == y).mean())
    svm_bal = float(balanced_accuracy_score(y, y_pred_svm))

    rf_pipe = Pipeline([
        ('rf', RandomForestClassifier(
            random_state=random_state,
            n_estimators=fp['fix_rf_estimators'],
            max_depth=fp['fix_rf_depth'],
            min_samples_split=fp['fix_rf_split'],
            max_features=fp['fix_rf_features'],
        )),
    ])
    rf_pipe.fit(X, y)
    y_pred_rf = cross_val_predict(rf_pipe, X, y, cv=cv, n_jobs=1)
    rf_acc = float((y_pred_rf == y).mean())
    rf_bal = float(balanced_accuracy_score(y, y_pred_rf))

    return svm_acc, svm_bal, rf_acc, rf_bal


def _run_one_permutation(
    seed,
    z_power_overt_smooth, y_overt, onset_tps,
    z_power_covert_smooth, y_covert,
    overt_matched_0idx, covert_matched_0idx,
    covert_keep_0idx,
    band_idx, win_cond,
    pre_tp, post_tp,
    search_start_idx, search_end_bound, slide_step,
    fp,
    consonant_pre_tps=None, post_tp_base_w4c=None,
):
    """
    One permutation iteration: shuffle overt class-template assignments,
    run full pipeline (onset estimation → feature extraction → classification),
    return (svm_acc, svm_bal, rf_acc, rf_bal).
    """
    rng     = np.random.default_rng(seed)
    classes = list(range(1, len(SYLLABLES) + 1))
    perm    = rng.permutation(classes)
    # perm_map[orig_class] = new_label: template originally for orig_class
    # is now assigned to new_label, so covert trials of new_label will use
    # the wrong (shuffled) overt template for onset estimation.
    perm_map = {cls: int(perm[i]) for i, cls in enumerate(classes)}

    if win_cond in ('W2', 'W3', 'W4u'):
        templates, win_len = build_class_templates(
            z_power_overt_smooth, y_overt, onset_tps,
            overt_matched_0idx, band_idx,
            pre_tp=pre_tp, post_tp=post_tp)

        # Reassign: label perm_map[c] now carries template originally built for c
        shuffled = {perm_map[c]: templates[c] for c in classes}

        est_onset_tps, _, _ = estimate_covert_onsets(
            z_power_covert_smooth, y_covert,
            shuffled, win_len,
            covert_matched_0idx, band_idx,
            pre_tp=pre_tp,
            search_start=search_start_idx,
            search_end=search_end_bound,
            step=slide_step)

        X = build_X_speech_window(
            z_power_covert_smooth, covert_keep_0idx, est_onset_tps,
            pre_onset_tp=pre_tp, post_onset_tp=post_tp,
            band_idx=band_idx)

    else:  # W4c — consonant-specific template sizes
        templates_c, _ = build_class_templates_consonant(
            z_power_overt_smooth, y_overt, onset_tps,
            overt_matched_0idx, band_idx,
            consonant_pre_tps, post_tp_base_w4c)

        shuffled_c = {perm_map[c]: templates_c[c] for c in classes}

        # Inline onset estimation: derive win_len from the shuffled template's
        # actual shape to avoid size mismatches from cross-group shuffling.
        nTrials       = z_power_covert_smooth.shape[-1]
        total_T       = z_power_covert_smooth.shape[2]
        est_onset_tps = np.zeros(nTrials, dtype=int)
        for i in range(nTrials):
            cls  = int(y_covert[i])
            tmpl = shuffled_c.get(cls)
            if tmpl is None or np.all(tmpl == 0):
                est_onset_tps[i] = search_start_idx
                continue
            win_len_i = tmpl.shape[2]
            pre_tp_i  = win_len_i - post_tp_base_w4c
            trial_3d  = z_power_covert_smooth[
                np.ix_(covert_matched_0idx, band_idx,
                       list(range(total_T)), [i])][:, :, :, 0]
            _, _, peak_s, _, _ = slide_template(
                trial_3d, tmpl.flatten(), win_len_i,
                search_start_idx, search_end_bound, slide_step)
            est_onset_tps[i] = peak_s + pre_tp_i

        X, _ = build_X_covert_consonant(
            z_power_covert_smooth, y_covert, est_onset_tps,
            covert_keep_0idx, band_idx,
            consonant_pre_tps, post_tp_base_w4c)

    return _classify_cv_fast(X, y_covert, fp)


def _compute_X_true(
    cfg,
    z_power_overt_smooth, y_overt, onset_tps,
    z_power_covert_smooth, y_covert,
    overt_matched_0idx, covert_matched_0idx,
    covert_keep_0idx,
    search_start_idx, search_end_bound, slide_step,
    consonant_pre_tps=None, post_tp_base_w4c=None,
):
    """Re-extract covert feature matrix using true (non-shuffled) overt templates."""
    win_cond = cfg['win_cond']
    band_idx = cfg['band_idx']
    pre_tp   = cfg['pre_tp']
    post_tp  = cfg['post_tp']

    if win_cond in ('W2', 'W3', 'W4u'):
        templates, win_len = build_class_templates(
            z_power_overt_smooth, y_overt, onset_tps,
            overt_matched_0idx, band_idx, pre_tp=pre_tp, post_tp=post_tp)
        est_onset_tps, _, _ = estimate_covert_onsets(
            z_power_covert_smooth, y_covert, templates, win_len,
            covert_matched_0idx, band_idx, pre_tp=pre_tp,
            search_start=search_start_idx, search_end=search_end_bound,
            step=slide_step)
        X = build_X_speech_window(
            z_power_covert_smooth, covert_keep_0idx, est_onset_tps,
            pre_onset_tp=pre_tp, post_onset_tp=post_tp, band_idx=band_idx)
    else:  # W4c
        templates_c, _ = build_class_templates_consonant(
            z_power_overt_smooth, y_overt, onset_tps,
            overt_matched_0idx, band_idx, consonant_pre_tps, post_tp_base_w4c)
        est_onset_tps, _, _ = estimate_covert_onsets_consonant(
            z_power_covert_smooth, y_covert, templates_c,
            consonant_pre_tps, post_tp_base_w4c,
            covert_matched_0idx, band_idx,
            search_start=search_start_idx, search_end=search_end_bound,
            step=slide_step)
        X, _ = build_X_covert_consonant(
            z_power_covert_smooth, y_covert, est_onset_tps,
            covert_keep_0idx, band_idx, consonant_pre_tps, post_tp_base_w4c)
    return X


def _plot_permutation_null(
    null_tmpl_svm_acc, null_tmpl_svm_bal, null_tmpl_rf_acc, null_tmpl_rf_bal,
    null_lbl_svm_acc,  null_lbl_svm_bal,  null_lbl_rf_acc,  null_lbl_rf_bal,
    true_svm_acc, true_svm_bal, true_rf_acc, true_rf_bal,
    p_tmpl_svm_acc, p_tmpl_svm_bal, p_tmpl_rf_acc, p_tmpl_rf_bal,
    p_lbl_svm_acc,  p_lbl_svm_bal,  p_lbl_rf_acc,  p_lbl_rf_bal,
    tag, save_dir, subj, cond_code, n_perms,
):
    """
    2-row × 4-col null distribution figure.
    Row 0: template-label permutation (tests onset estimation specificity).
    Row 1: covert-trial-label permutation (tests classifier performance).
    """
    chance = 1.0 / len(SYLLABLES)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    rows = [
        (0, null_tmpl_svm_acc, null_tmpl_svm_bal, null_tmpl_rf_acc, null_tmpl_rf_bal,
         p_tmpl_svm_acc, p_tmpl_svm_bal, p_tmpl_rf_acc, p_tmpl_rf_bal,
         'Template-label permutation\n(onset estimation specificity)',
         'darkorange'),
        (1, null_lbl_svm_acc,  null_lbl_svm_bal,  null_lbl_rf_acc,  null_lbl_rf_bal,
         p_lbl_svm_acc,  p_lbl_svm_bal,  p_lbl_rf_acc,  p_lbl_rf_bal,
         'Covert-trial-label permutation\n(classifier above-chance check)',
         'steelblue'),
    ]
    col_labels = ['SVM accuracy', 'SVM balanced accuracy', 'RF accuracy', 'RF balanced accuracy']

    for row_i, n0, n1, n2, n3, p0, p1, p2, p3, row_title, color in rows:
        for col_i, (null_dist, true_val, pval, metric_name) in enumerate(zip(
            [n0, n1, n2, n3],
            [true_svm_acc, true_svm_bal, true_rf_acc, true_rf_bal],
            [p0, p1, p2, p3],
            col_labels,
        )):
            ax = axes[row_i, col_i]
            ax.hist(null_dist, bins=40, color=color, edgecolor='k', alpha=0.70,
                    label=f'Null (N={n_perms})\nmean={null_dist.mean():.3f}')
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                       label=f'True = {true_val:.3f}')
            ax.axvline(chance, color='gray', linestyle=':', linewidth=1.2,
                       label=f'Chance = {chance:.3f}')
            ax.set_xlabel(metric_name, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_title(f'p = {pval:.4f}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.35)
            if col_i == 0:
                ax.set_ylabel(row_title + '\n\nCount', fontsize=8)

    fig.suptitle(f'{subj} | Permutation baseline — {tag}', fontsize=11)
    plt.tight_layout()
    fpath = os.path.join(save_dir, f'{subj}_{cond_code}_permutation_{_safe_tag(tag)}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved permutation plot: {fpath}')


def run_permutation_baseline(
    top_k_configs,
    z_power_overt_smooth, y_overt, onset_tps,
    z_power_covert_smooth, y_covert,
    overt_matched_0idx, covert_matched_0idx,
    covert_keep_0idx,
    search_start_idx, search_end_bound, slide_step,
    fp,
    n_perms,
    exp_results,
    save_dir, subj, cond_code,
    consonant_pre_tps=None, post_tp_base_w4c=None,
):
    """
    Two permutation tests for each of the top-k covert classification experiments:

    1. Template-label permutation: overt class-template assignments are shuffled,
       then the full pipeline (onset estimation → window extraction → classification)
       is re-run N times.  Tests whether correct template-class matching contributes
       class-specific temporal structure to the onset estimate.

    2. Covert-trial-label permutation: covert trial labels are shuffled while keeping
       the feature matrix X (extracted with true templates) fixed, and the classifier
       is re-run N times.  Standard check that the classifier performs above chance.

    p-value = proportion of permutations >= true accuracy for both tests.
    Saves per-experiment 2×4 panel figures and a combined CSV.
    """
    exp_lookup = {r.name: r for r in exp_results}
    perm_rows  = []
    sep        = '-' * 60

    for cfg in top_k_configs:
        win_cond  = cfg['win_cond']
        band_name = cfg['band_name']
        band_idx  = cfg['band_idx']
        pre_tp    = cfg['pre_tp']
        post_tp   = cfg['post_tp']
        exp_name  = cfg['exp_name']
        exp_tag   = cfg['tag']

        true_result = exp_lookup.get(exp_name)
        if true_result is None:
            print(f'\n  Permutation baseline [{exp_tag}]: '
                  f'no matching experiment result ({exp_name}), skipping.')
            continue

        true_svm_acc = true_result.svm_accuracy
        true_svm_bal = true_result.svm_bal_acc
        true_rf_acc  = true_result.rf_accuracy
        true_rf_bal  = true_result.rf_bal_acc

        print(f'\n  {sep}')
        print(f'  Permutation baseline: {exp_tag}')
        print(f'  N = {n_perms}  |  experiment: {exp_name}')
        print(f'  True — SVM acc={true_svm_acc:.3f} bal={true_svm_bal:.3f}  '
              f'RF acc={true_rf_acc:.3f} bal={true_rf_bal:.3f}')

        # --- (1) Template-label permutation ---
        print(f'  Running template-label permutation ({n_perms} iterations)...')
        tmpl_results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_run_one_permutation)(
                seed,
                z_power_overt_smooth, y_overt, onset_tps,
                z_power_covert_smooth, y_covert,
                overt_matched_0idx, covert_matched_0idx,
                covert_keep_0idx,
                band_idx, win_cond,
                pre_tp, post_tp,
                search_start_idx, search_end_bound, slide_step,
                fp,
                consonant_pre_tps if win_cond == 'W4c' else None,
                post_tp_base_w4c  if win_cond == 'W4c' else None,
            )
            for seed in range(n_perms)
        )

        null_tmpl_svm_acc = np.array([r[0] for r in tmpl_results])
        null_tmpl_svm_bal = np.array([r[1] for r in tmpl_results])
        null_tmpl_rf_acc  = np.array([r[2] for r in tmpl_results])
        null_tmpl_rf_bal  = np.array([r[3] for r in tmpl_results])

        p_tmpl_svm_acc = float(np.mean(null_tmpl_svm_acc >= true_svm_acc))
        p_tmpl_svm_bal = float(np.mean(null_tmpl_svm_bal >= true_svm_bal))
        p_tmpl_rf_acc  = float(np.mean(null_tmpl_rf_acc  >= true_rf_acc))
        p_tmpl_rf_bal  = float(np.mean(null_tmpl_rf_bal  >= true_rf_bal))

        print(f'  Template perm null — '
              f'SVM acc={null_tmpl_svm_acc.mean():.3f}±{null_tmpl_svm_acc.std():.3f}  '
              f'RF acc={null_tmpl_rf_acc.mean():.3f}±{null_tmpl_rf_acc.std():.3f}')
        print(f'  Template perm p    — '
              f'SVM acc={p_tmpl_svm_acc:.4f} bal={p_tmpl_svm_bal:.4f}  '
              f'RF acc={p_tmpl_rf_acc:.4f} bal={p_tmpl_rf_bal:.4f}')

        # --- (2) Covert-trial-label permutation ---
        # Re-extract X once with true templates, then only shuffle y N times.
        print(f'  Re-extracting X with true templates for label permutation...')
        X_true = _compute_X_true(
            cfg,
            z_power_overt_smooth, y_overt, onset_tps,
            z_power_covert_smooth, y_covert,
            overt_matched_0idx, covert_matched_0idx,
            covert_keep_0idx,
            search_start_idx, search_end_bound, slide_step,
            consonant_pre_tps if win_cond == 'W4c' else None,
            post_tp_base_w4c  if win_cond == 'W4c' else None,
        )

        print(f'  Running covert-label permutation ({n_perms} iterations)...')
        lbl_results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_classify_cv_fast)(
                X_true,
                np.random.default_rng(seed).permutation(y_covert),
                fp,
            )
            for seed in range(n_perms)
        )

        null_lbl_svm_acc = np.array([r[0] for r in lbl_results])
        null_lbl_svm_bal = np.array([r[1] for r in lbl_results])
        null_lbl_rf_acc  = np.array([r[2] for r in lbl_results])
        null_lbl_rf_bal  = np.array([r[3] for r in lbl_results])

        p_lbl_svm_acc = float(np.mean(null_lbl_svm_acc >= true_svm_acc))
        p_lbl_svm_bal = float(np.mean(null_lbl_svm_bal >= true_svm_bal))
        p_lbl_rf_acc  = float(np.mean(null_lbl_rf_acc  >= true_rf_acc))
        p_lbl_rf_bal  = float(np.mean(null_lbl_rf_bal  >= true_rf_bal))

        print(f'  Label perm null    — '
              f'SVM acc={null_lbl_svm_acc.mean():.3f}±{null_lbl_svm_acc.std():.3f}  '
              f'RF acc={null_lbl_rf_acc.mean():.3f}±{null_lbl_rf_acc.std():.3f}')
        print(f'  Label perm p       — '
              f'SVM acc={p_lbl_svm_acc:.4f} bal={p_lbl_svm_bal:.4f}  '
              f'RF acc={p_lbl_rf_acc:.4f} bal={p_lbl_rf_bal:.4f}')

        perm_rows.append({
            'exp_name':                  exp_name,
            'win_cond':                  win_cond,
            'band_name':                 band_name,
            'true_svm_acc':              round(true_svm_acc, 4),
            'true_svm_bal':              round(true_svm_bal, 4),
            'true_rf_acc':               round(true_rf_acc,  4),
            'true_rf_bal':               round(true_rf_bal,  4),
            # template permutation
            'tmpl_null_svm_acc_mean':    round(float(null_tmpl_svm_acc.mean()), 4),
            'tmpl_null_rf_acc_mean':     round(float(null_tmpl_rf_acc.mean()),  4),
            'tmpl_null_svm_acc_std':     round(float(null_tmpl_svm_acc.std()),  4),
            'tmpl_null_rf_acc_std':      round(float(null_tmpl_rf_acc.std()),   4),
            'p_tmpl_svm_acc':            round(p_tmpl_svm_acc, 4),
            'p_tmpl_svm_bal':            round(p_tmpl_svm_bal, 4),
            'p_tmpl_rf_acc':             round(p_tmpl_rf_acc,  4),
            'p_tmpl_rf_bal':             round(p_tmpl_rf_bal,  4),
            # label permutation
            'lbl_null_svm_acc_mean':     round(float(null_lbl_svm_acc.mean()),  4),
            'lbl_null_rf_acc_mean':      round(float(null_lbl_rf_acc.mean()),   4),
            'lbl_null_svm_acc_std':      round(float(null_lbl_svm_acc.std()),   4),
            'lbl_null_rf_acc_std':       round(float(null_lbl_rf_acc.std()),    4),
            'p_lbl_svm_acc':             round(p_lbl_svm_acc, 4),
            'p_lbl_svm_bal':             round(p_lbl_svm_bal, 4),
            'p_lbl_rf_acc':              round(p_lbl_rf_acc,  4),
            'p_lbl_rf_bal':              round(p_lbl_rf_bal,  4),
            'n_permutations':            n_perms,
        })

        _plot_permutation_null(
            null_tmpl_svm_acc, null_tmpl_svm_bal, null_tmpl_rf_acc, null_tmpl_rf_bal,
            null_lbl_svm_acc,  null_lbl_svm_bal,  null_lbl_rf_acc,  null_lbl_rf_bal,
            true_svm_acc, true_svm_bal, true_rf_acc, true_rf_bal,
            p_tmpl_svm_acc, p_tmpl_svm_bal, p_tmpl_rf_acc, p_tmpl_rf_bal,
            p_lbl_svm_acc,  p_lbl_svm_bal,  p_lbl_rf_acc,  p_lbl_rf_bal,
            exp_tag, save_dir, subj, cond_code, n_perms)

    if perm_rows:
        df    = pd.DataFrame(perm_rows)
        fpath = os.path.join(save_dir, f'{subj}_{cond_code}_permutation_baseline.csv')
        df.to_csv(fpath, index=False)
        print(f'\n  Permutation baseline CSV saved: {fpath}')

        print(f'\n  Permutation baseline summary:')
        hdr = f'  {"Experiment":<50} {"p_tmpl(SVM)":<13} {"p_tmpl(RF)":<13} {"p_lbl(SVM)":<13} {"p_lbl(RF)"}'
        print(hdr)
        print(f'  {"-"*len(hdr)}')
        for row in perm_rows:
            print(f'  {row["exp_name"]:<50} '
                  f'{row["p_tmpl_svm_acc"]:<13.4f} {row["p_tmpl_rf_acc"]:<13.4f} '
                  f'{row["p_lbl_svm_acc"]:<13.4f} {row["p_lbl_rf_acc"]:.4f}')

    return perm_rows


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------

def plot_covert_summary_figures(all_results, save_dir, subj, cond_code):
    """
    One 2×2 summary figure per window condition (baseline / W2 / W3 / W4u / W4c).

    Layout mirrors classification_overt_band_sweep.py:
      Top-left  — accuracy bar chart (SVM acc, SVM bal, RF acc, RF bal)
      Top-right — RF per-class recall heatmap  (experiments × syllables)
      Bot-left  — SVM per-class recall heatmap
      Bot-right — Mean(RF+SVM) per-class recall heatmap
    """
    if not all_results:
        return

    chance = 1.0 / len(SYLLABLES)

    def _short_label(r):
        if '_est_with_' in r.name:
            # name format: {Wx}_est_with_{band_name}_{ic_tag}IC[_...suffix]
            after_win = r.name.split('_est_with_', 1)[1]
            # strip IC tag and anything after it
            for ic_tag in ('_all_keepIC', '_matchedIC'):
                if ic_tag in after_win:
                    band_part = after_win.split(ic_tag, 1)[0]
                    break
            else:
                band_part = after_win
            return band_part
        return r.name

    def _plot_win_summary(results, win_tag, win_label):
        if not results:
            return

        labels      = [_short_label(r) for r in results]
        n           = len(results)
        svm_accs    = [r.svm_accuracy for r in results]
        rf_accs     = [r.rf_accuracy  for r in results]
        svm_bals    = [r.svm_bal_acc  for r in results]
        rf_bals     = [r.rf_bal_acc   for r in results]
        rf_recall   = np.array([[r.rf_per_class.get(s, 0)  for s in SYLLABLES] for r in results])
        svm_recall  = np.array([[r.svm_per_class.get(s, 0) for s in SYLLABLES] for r in results])
        mean_recall = (rf_recall + svm_recall) / 2

        fig_h = max(12, n * 0.35 + 4)
        fig, axes = plt.subplots(2, 2, figsize=(18, fig_h))
        ax_acc, ax_rf, ax_svm, ax_mean = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        x = np.arange(n)
        w = 0.2
        ax_acc.bar(x - 1.5*w, svm_accs, w, label='SVM acc',     color='steelblue',  alpha=0.9)
        ax_acc.bar(x - 0.5*w, svm_bals, w, label='SVM bal acc', color='steelblue',  alpha=0.45)
        ax_acc.bar(x + 0.5*w, rf_accs,  w, label='RF acc',      color='darkorange', alpha=0.9)
        ax_acc.bar(x + 1.5*w, rf_bals,  w, label='RF bal acc',  color='darkorange', alpha=0.45)
        ax_acc.axhline(chance, color='gray', linestyle=':', linewidth=0.8,
                       label=f'Chance ({chance:.2f})')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax_acc.set_ylabel('Cross-validated accuracy')
        ax_acc.set_title('Accuracy by experiment')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(axis='y', alpha=0.4)
        ax_acc.set_ylim(0, 1)

        def _heatmap(ax, matrix, title):
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(range(len(SYLLABLES)))
            ax.set_xticklabels(SYLLABLES, fontsize=10)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel('Syllable')
            ax.set_ylabel('Experiment')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall')
            for i in range(n):
                for j in range(len(SYLLABLES)):
                    val   = matrix[i, j]
                    color = 'white' if val < 0.25 or val > 0.75 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=color)

        _heatmap(ax_rf,   rf_recall,   'RF per-class recall')
        _heatmap(ax_svm,  svm_recall,  'SVM per-class recall')
        _heatmap(ax_mean, mean_recall, 'Mean(RF+SVM) per-class recall')

        fig.suptitle(
            f'{subj} | Covert — {win_label}',
            fontsize=12)
        plt.tight_layout()
        fpath = os.path.join(save_dir, f'{subj}_{cond_code}_{win_tag}_summary.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved summary: {fpath}')

    for win_tag, win_label, pfx in [
        ('baseline', 'baseline (full epoch)',    'baseline'),
        ('W2',  'W2 — speech window',            'W2_'),
        ('W3',  'W3 — pre-speech',               'W3_'),
        ('W4u', 'W4u — uniform pre-onset',       'W4u_'),
        ('W4c', 'W4c — consonant-specific',      'W4c_'),
    ]:
        subset = [r for r in all_results
                  if (pfx == 'baseline' and r.name.startswith('baseline'))
                  or (pfx != 'baseline' and r.name.startswith(pfx))]
        _plot_win_summary(subset, win_tag, win_label)


# ---------------------------------------------------------------------------
# Single covert experiment runner
# ---------------------------------------------------------------------------

def run_covert_exp(name, X, y, save_dir, subj, cond_code,
                   ic_set, band_set, feature, window,
                   inner_jobs, band_idx_vec=None,
                   fix_svm_C=None, fix_rf_depth=None,
                   fix_rf_features=None, fix_rf_estimators=None,
                   fix_rf_split=None,
                   nICs=None, nBands=None, nTime=None,
                   ic_labels=None, time_vec=None):
    """
    Classify one covert experiment and optionally save RF importance plots.

    Parameters mirror run_classifiers() with additional shape info for plots.
    """
    print(f'\n  [{name}]  X={X.shape}')
    result, rf_model = run_classifiers(
        name, X, y, save_dir, subj, cond_code,
        ic_set=ic_set, band_set=band_set,
        feature=feature, window=window,
        inner_n_jobs=inner_jobs,
        fix_svm_C=fix_svm_C, fix_rf_depth=fix_rf_depth,
        fix_rf_features=fix_rf_features,
        fix_rf_estimators=fix_rf_estimators,
        fix_rf_split=fix_rf_split)

    if nICs is not None and nBands is not None and nTime is not None:
        plot_feature_importance(
            rf_model, nICs, nBands, nTime,
            ic_labels or [f'IC{i}' for i in range(nICs)],
            time_vec if time_vec is not None else np.arange(nTime),
            save_dir, subj, cond_code, name,
            band_idx_vec=band_idx_vec)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EEG syllable classification — covert/imagined condition')

    # -----------------------------------------------------------------------
    # Subject / paths
    # -----------------------------------------------------------------------
    parser.add_argument('--subj', required=True, type=str,
                        help='Subject ID, e.g. subj-02')
    parser.add_argument('--input-dir', required=True, type=str,
                        help='Directory with .mat files')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Output root directory')

    # -----------------------------------------------------------------------
    # Overt IC / epoch args
    # -----------------------------------------------------------------------
    parser.add_argument('--overt-keep-ics', required=True, type=int, nargs='+',
                        help='1-indexed overt keep ICs (all artefact-free ICs). Must be the same set used for overt classification.')
    parser.add_argument('--overt-matched-ics', required=True, type=int, nargs='+',
                        help='1-indexed overt brain ICs used for template construction')
    parser.add_argument('--overt-bad-epochs', default=[], type=int, nargs='*',
                        help='1-indexed bad overt epochs to reject')

    # -----------------------------------------------------------------------
    # Covert IC / epoch args
    # -----------------------------------------------------------------------
    parser.add_argument('--covert-keep-ics', required=True, type=int, nargs='+',
                        help='1-indexed covert keep ICs (all artefact-free ICs)')
    parser.add_argument('--covert-matched-ics', required=True, type=int, nargs='+',
                        help='1-indexed covert ICs matched to overt matched ICs via CORRMAP. '
                             'Count must equal --overt-matched-ics count.')
    parser.add_argument('--covert-bad-epochs', default=[], type=int, nargs='*',
                        help='1-indexed bad covert epochs to reject')

    # -----------------------------------------------------------------------
    # Window parameters (from overt results)
    # -----------------------------------------------------------------------
    parser.add_argument('--speech-window-ms', default=None, type=int,
                        help='Must be the same as overt W2. Override auto-computed speech window length (ms). '
                             'Default: ceil(mean speech duration / 100ms) * 100ms')
    parser.add_argument('--best-overall-pre-onset-ms', default=None, type=int,
                        help='Uniform pre-onset p* (ms) for W4u. '
                             'If omitted, W4u experiments are skipped.')
    parser.add_argument('--consonant-pre-onset-ms', default=None,
                        type=int, nargs=3,
                        metavar=('STOP_MS', 'NASAL_MS', 'FRICATIVE_MS'),
                        help='Per-group pre-onset (ms) for W4c, ordered: '
                             'stop(gi/gu) nasal(mi/mu) fricative(si/su). '
                             'If omitted, W4c experiments are skipped.')

    # -----------------------------------------------------------------------
    # Search region
    # -----------------------------------------------------------------------
    parser.add_argument('--search-start-ms', default=SEARCH_START_MS_DEFAULT,
                        type=int,
                        help=f'Template search start, ms from t=0 '
                             f'(default: {SEARCH_START_MS_DEFAULT})')
    parser.add_argument('--search-end-ms', default=SEARCH_END_MS_DEFAULT,
                        type=int,
                        help=f'Template search end, ms from t=0 — '
                             f'win_len will be subtracted automatically '
                             f'(default: {SEARCH_END_MS_DEFAULT})')
    parser.add_argument('--slide-step', default=5, type=int,
                        help='Sliding step size in samples (default: 5 = 10 ms at 500 Hz)')

    # -----------------------------------------------------------------------
    # Band sets
    # -----------------------------------------------------------------------
    parser.add_argument('--band-sets', default=list(BAND_SETS.keys()),
                        type=str, nargs='+',
                        help=f'Band sets to test. Choices: {list(BAND_SETS.keys())}. '
                             f'Default: all band sets')

    # -----------------------------------------------------------------------
    # Sanity check
    # -----------------------------------------------------------------------
    parser.add_argument('--sanity-folds', default=5, type=int,
                        help='Number of folds for overt onset estimation sanity check (default: 5)')
    parser.add_argument('--skip-sanity-check', action='store_true', default=False,
                        help='Skip the overt onset estimation sanity check')
    parser.add_argument('--n-permutations', default=1000, type=int,
                        help='Number of permutations for permutation baseline (default: 1000)')
    parser.add_argument('--permutation-top-k', default=5, type=int,
                        help='Number of top sanity-check experiments to run permutation baseline on (default: 5)')
    parser.add_argument('--skip-permutation-test', action='store_true', default=False,
                        help='Skip permutation baseline (requires sanity check to have run)')

    # -----------------------------------------------------------------------
    # Fixed classifier hyperparameters (if want to test specific values)
    # -----------------------------------------------------------------------
    parser.add_argument('--fix-svm-C', type=float, default=None)
    parser.add_argument('--fix-rf-depth', type=str, default=None, help='int or "None" for no limit')
    parser.add_argument('--fix-rf-features', type=str, default=None)
    parser.add_argument('--fix-rf-estimators', type=int, default=None)
    parser.add_argument('--fix-rf-split', type=int, default=None)

    # -----------------------------------------------------------------------
    # Window condition selector
    # -----------------------------------------------------------------------
    parser.add_argument('--window-conditions', default=['W2', 'W3', 'W4u', 'W4c'],
                        type=str, nargs='+',
                        choices=['W2', 'W3', 'W4u', 'W4c'],
                        help='Which window conditions to run (default: all)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if len(args.overt_matched_ics) != len(args.covert_matched_ics):
        parser.error('--overt-matched-ics and --covert-matched-ics must have '
                     'the same number of entries (1-to-1 CORRMAP correspondence).')

    # Validate band sets
    invalid_bands = [b for b in args.band_sets if b not in BAND_SETS]
    if invalid_bands:
        parser.error(f'Unknown band sets: {invalid_bands}. '
                     f'Valid options: {list(BAND_SETS.keys())}')

    subj        = args.subj
    cond_code   = 'im'
    cond_label  = 'imagined'
    inner_jobs  = -1 

    # Fixed params
    fix_rf_depth = (None if args.fix_rf_depth in (None, 'None')
                    else int(args.fix_rf_depth))
    fp = dict(
        fix_svm_C        = args.fix_svm_C,
        fix_rf_depth     = fix_rf_depth,
        fix_rf_features  = args.fix_rf_features,
        fix_rf_estimators= args.fix_rf_estimators,
        fix_rf_split     = args.fix_rf_split,
    )

    # IC indices (0-based)
    overt_keep_0idx   = [ic - 1 for ic in args.overt_keep_ics]
    overt_matched_0idx   = [ic - 1 for ic in args.overt_matched_ics]
    covert_keep_0idx   = [ic - 1 for ic in args.covert_keep_ics]
    covert_matched_0idx= [ic - 1 for ic in args.covert_matched_ics]

    # Window parameters in samples
    pre_onset_w3_tp    = int(W3_PRE_ONSET_MS / 1000 * FS)

    run_w4u = ('W4u' in args.window_conditions
               and args.best_overall_pre_onset_ms is not None)
    run_w4c = ('W4c' in args.window_conditions
               and args.consonant_pre_onset_ms is not None)

    if 'W4u' in args.window_conditions and args.best_overall_pre_onset_ms is None:
        print('  WARNING: W4u requested but --best-overall-pre-onset-ms not provided. '
              'Skipping W4u.')
    if 'W4c' in args.window_conditions and args.consonant_pre_onset_ms is None:
        print('  WARNING: W4c requested but --consonant-pre-onset-ms not provided. '
              'Skipping W4c.')

    if run_w4u:
        pre_onset_w4u_tp = int(args.best_overall_pre_onset_ms / 1000 * FS)

    if run_w4c:
        consonant_pre_tps = {
            name: int(args.consonant_pre_onset_ms[i] / 1000 * FS)
            for i, name in enumerate(CONSONANT_GROUP_NAMES)
        }
        post_tp_base_w4c = int(CONSONANT_POST_BASE_MS / 1000 * FS)

    # Output dir
    save_dir_root = os.path.join(args.output_dir, subj, cond_label)
    save_dir_onset_overt_test = os.path.join(save_dir_root, 'overt_onset_test')
    os.makedirs(save_dir_onset_overt_test, exist_ok=True)
    save_dir_covert = os.path.join(save_dir_root, 'covert_classification')
    os.makedirs(save_dir_covert, exist_ok=True)
    save_dir_figures = os.path.join(save_dir_root, 'summary_figures')
    os.makedirs(save_dir_figures, exist_ok=True)

    # Band sets to iterate
    band_items = [(name, BAND_SETS[name]) for name in args.band_sets]

    # IC label helpers (overt labels not needed — templates built internally)
    ic_labels_covert_keep    = [f'IC{ic}' for ic in args.covert_keep_ics]
    ic_labels_covert_matched = [f'IC{ic}' for ic in args.covert_matched_ics]

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  Subject:     {subj}')
    print(f'  Condition:   {cond_label} ({cond_code})')
    print(f'  Overt keep ICs (1-idx): {args.overt_keep_ics}')
    print(f'  Overt matched ICs (1-idx): {args.overt_matched_ics}')
    print(f'  Covert keep ICs (1-idx): {args.covert_keep_ics}')
    print(f'  Covert matched ICs (1-idx): {args.covert_matched_ics}')
    if run_w4u:
        print(f'  W4u pre-onset:           {args.best_overall_pre_onset_ms} ms')
    if run_w4c:
        for i, name in enumerate(CONSONANT_GROUP_NAMES):
            print(f'  W4c {name} pre-onset: {args.consonant_pre_onset_ms[i]} ms')
    print(f'  Band sets:   {args.band_sets}')
    print(f'  Window conds:{args.window_conditions}')
    print(f'  Search region: {args.search_start_ms} → {args.search_end_ms} ms from t=0')
    print(f'  Slide step:  {args.slide_step} samples')
    if any(v is not None for v in fp.values()):
        print(f'  Classifier hyperparams fixed from inputs.')
        print(f'  SVM: C={fp["fix_svm_C"]}  kernel=linear')
        print(f'  RF:  max_depth={fp["fix_rf_depth"]}  '              
              f'max_features={fp["fix_rf_features"]}  '              
              f'n_estimators={fp["fix_rf_estimators"]}  '              
              f'min_samples_split={fp["fix_rf_split"]}')
    print(f'{sep}')

    # ==================================================================
    # [1/5] Load and preprocess OVERT data
    # ==================================================================
    print(f'\n{sep}')
    print('  [1/5] Loading and preprocessing OVERT data...')
    print(f'{sep}')

    overt_path = os.path.join(
        args.input_dir, f'{subj}_sp_eeg_analytic.mat')
    onset_path = os.path.join(
        args.input_dir, f'{subj}_speech_onset_offset.mat')

    Z_overt, labels_overt, times_overt = load_analytic(overt_path)
    print(f'  Overt analytic shape: {Z_overt.shape}')
    idx_0ms_overt, _ = derive_epoch_indices(times_overt)

    _, _, _, z_power_overt = compute_features(Z_overt, times_overt)

    speech_data  = loadmat(onset_path)
    onset_times  = speech_data['onset_latencies'].squeeze()
    offset_times = speech_data['offset_latencies'].squeeze()

    if len(args.overt_bad_epochs) > 0:
        print(f'  Rejecting {len(args.overt_bad_epochs)} overt bad epoch(s): '
              f'{args.overt_bad_epochs}')
        _, (z_power_overt, labels_overt,
            onset_times, offset_times) = reject_epochs(
            args.overt_bad_epochs,
            z_power_overt, labels_overt, onset_times, offset_times)
    else:
        print(f'  No bad overt epochs — keeping all {z_power_overt.shape[-1]} trials')

    print(f'  Smoothing overt (LP {LP_CUTOFF_HZ} Hz)...')
    z_power_overt_smooth = lowpass_smooth(z_power_overt)

    # Sample indices for onset/offset
    onset_tps  = np.rint(onset_times  / 1000 * FS + idx_0ms_overt).astype(int)
    offset_tps = np.rint(offset_times / 1000 * FS + idx_0ms_overt).astype(int)

    invalid = np.where(offset_tps < onset_tps)[0]
    if len(invalid) > 0:
        print(f'  WARNING: {len(invalid)} trials have offset < onset')

    # Speech window length (W2)
    if args.speech_window_ms is not None:
        speech_window_tp = int(args.speech_window_ms / 1000 * FS)
        print(f'  Speech window overridden: {args.speech_window_ms} ms '
              f'= {speech_window_tp} samples')
    else:
        speech_window_tp = compute_speech_window_tp(onset_tps, offset_tps)
    # Display label for headers and output names (int ms, works whether auto or overridden)
    speech_window_ms_label = int(round(speech_window_tp / FS * 1000))

    y_overt = np.array(labels_overt).astype(int)

    print(f'  Overt trials after rejection: {len(y_overt)}')

    # ==================================================================
    # [2/5] Load and preprocess COVERT data
    # ==================================================================
    print(f'\n{sep}')
    print('  [2/5] Loading and preprocessing COVERT data...')
    print(f'{sep}')

    covert_path = os.path.join(
        args.input_dir, f'{subj}_im_eeg_analytic.mat')

    Z_covert, labels_covert, times_covert = load_analytic(covert_path)
    print(f'  Covert analytic shape: {Z_covert.shape}')
    idx_0ms_covert, idx_1950ms_covert = derive_epoch_indices(times_covert)

    _, _, _, z_power_covert = compute_features(Z_covert, times_covert)

    if len(args.covert_bad_epochs) > 0:
        print(f'  Rejecting {len(args.covert_bad_epochs)} covert bad epoch(s): '
              f'{args.covert_bad_epochs}')
        _, (z_power_covert, labels_covert) = reject_epochs(
            args.covert_bad_epochs, z_power_covert, labels_covert)
    else:
        print(f'  No bad covert epochs — keeping all {z_power_covert.shape[-1]} trials')

    print(f'  Smoothing covert (LP {LP_CUTOFF_HZ} Hz)...')
    z_power_covert_smooth = lowpass_smooth(z_power_covert)

    y_covert = np.array(labels_covert).astype(int)
    n_covert = len(y_covert)
    print(f'  Covert trials after rejection: {n_covert}')

    # Search region in absolute sample indices (covert epoch)
    search_start_idx = idx_0ms_covert + int(args.search_start_ms / 1000 * FS)
    search_end_bound = idx_0ms_covert + int(args.search_end_ms   / 1000 * FS)
    # Note: actual search_end passed to slide_template is search_end_bound;
    # slide_template subtracts win_len internally.

    print(f'  Search region: samples [{search_start_idx}, '
          f'{search_end_bound}] '
          f'({args.search_start_ms}–{args.search_end_ms} ms from t=0)')

    # ==================================================================
    # [3/5] Overt onset estimation sanity check
    # ==================================================================
    sanity_metrics_list = []

    if not args.skip_sanity_check:
        print(f'\n{sep}')
        print('  [3/5] Overt onset estimation sanity check...')
        print(f'{sep}')

        # Run sanity check for each requested window condition x band set
        sanity_conditions = []
        if 'W2' in args.window_conditions:
            sanity_conditions.append(
                ('W2', 0, speech_window_tp, 'W2 (onset → onset+N)'))
        if 'W3' in args.window_conditions:
            sanity_conditions.append(
                ('W3', pre_onset_w3_tp, 0, 'W3 (onset-500ms → onset)'))
        if run_w4u:
            sanity_conditions.append(
                ('W4u', pre_onset_w4u_tp, speech_window_tp,
                 f'W4u (onset-{args.best_overall_pre_onset_ms}ms → onset+N)'))

        for band_name_s, band_idx_s in band_items:
            for win_tag, s_pre_tp, s_post_tp, s_label in sanity_conditions:
                if s_pre_tp + s_post_tp == 0:
                    print(f'  Skipping {win_tag} [{band_name_s}] sanity check (zero-length window)')
                    continue
                tag_s = f'{s_label} [{band_name_s}]'

                metrics = sanity_check_overt_onset(
                    z_power_overt_smooth, y_overt, onset_tps,
                    overt_keep_0idx, band_idx_s,
                    pre_tp=s_pre_tp, post_tp=s_post_tp,
                    search_start=search_start_idx,
                    search_end=search_end_bound,
                    idx_0ms=idx_0ms_overt,
                    n_folds=args.sanity_folds,
                    step=args.slide_step,
                    save_dir=save_dir_onset_overt_test, subj=subj,
                    cond_code=cond_code, tag=tag_s)
                metrics['win_condition'] = win_tag   # overwrite full tag with short code
                metrics['band_name']     = band_name_s
                metrics['band_idx']      = band_idx_s
                metrics['pre_tp']        = s_pre_tp
                metrics['post_tp']       = s_post_tp
                sanity_metrics_list.append(metrics)

        # W4c sanity check (consonant-specific windows, all band sets)
        if run_w4c:
            for band_name_s, band_idx_s in band_items:
                print(f'\n  Overt sanity check — W4c [{band_name_s}] (consonant-specific windows)')
                nTrials_overt = len(y_overt)
                skf_c = StratifiedKFold(
                    n_splits=args.sanity_folds, shuffle=True, random_state=42)

                all_errors_c    = np.full(nTrials_overt, np.nan)
                all_estimated_c = np.full(nTrials_overt, np.nan)
                true_onset_ms_c = (onset_tps - idx_0ms_overt) / FS * 1000
                trial_records_c = []

                for fold_i, (train_idx, test_idx) in enumerate(
                        skf_c.split(np.zeros(nTrials_overt), y_overt)):
                    train_mask = np.isin(np.arange(nTrials_overt), train_idx)
                    templates_c, _ = build_class_templates_consonant(
                        z_power_overt_smooth, y_overt, onset_tps,
                        overt_keep_0idx, band_idx_s,
                        consonant_pre_tps, post_tp_base_w4c,
                        trial_mask=train_mask)

                    for i in test_idx:
                        cls        = int(y_overt[i])
                        grp        = _get_group_name(cls)
                        pre_tp_grp = consonant_pre_tps[grp]
                        win_len_grp= pre_tp_grp + post_tp_base_w4c
                        tmpl_3d    = templates_c.get(cls)
                        if tmpl_3d is None or np.all(tmpl_3d == 0):
                            continue

                        trial_3d = z_power_overt_smooth[
                            np.ix_(overt_keep_0idx, band_idx_s,
                                   list(range(z_power_overt_smooth.shape[2])),
                                   [i])][:, :, :, 0]

                        raw_corr_c, smoothed_corr_c, peak_s, peak_prominence, n_competing = slide_template(
                            trial_3d, tmpl_3d.flatten(), win_len_grp,
                            search_start_idx, search_end_bound, args.slide_step)

                        est_s = peak_s + pre_tp_grp
                        error_ms_c = abs(est_s - onset_tps[i]) / FS * 1000
                        all_errors_c[i]    = error_ms_c
                        all_estimated_c[i] = (est_s - idx_0ms_overt) / FS * 1000

                        n_pos_c = len(smoothed_corr_c)
                        positions_ms_c = (search_start_idx + np.arange(n_pos_c) * args.slide_step
                                          - idx_0ms_overt) / FS * 1000
                        trial_records_c.append({
                            'trial_idx':    i,
                            'class_label':  cls,
                            'error_ms':     error_ms_c,
                            'true_ms':      (int(onset_tps[i]) - idx_0ms_overt) / FS * 1000,
                            'estimated_ms': (est_s - idx_0ms_overt) / FS * 1000,
                            'raw_corr':     raw_corr_c.copy(),
                            'corr_vals':    smoothed_corr_c.copy(),
                            'positions_ms': positions_ms_c,
                        })

                    print(f'    Fold {fold_i + 1}/{args.sanity_folds}: '
                          f'{len(test_idx)} held-out trials done')

                valid_c    = ~np.isnan(all_errors_c)
                errs_c     = all_errors_c[valid_c]
                mae_c      = float(np.mean(errs_c))
                med_c      = float(np.median(errs_c))
                pct_50_c   = float(np.mean(errs_c <= 50) * 100)
                print(f'\n  W4c sanity check [{band_name_s}]: MAE={mae_c:.1f} ms  '
                      f'Median={med_c:.1f} ms  %≤50ms={pct_50_c:.1f}%')

                tag_w4c = f'W4c_consonant_specific [{band_name_s}]'
                _plot_onset_sanity(
                    true_onset_ms_c[valid_c], all_estimated_c[valid_c], errs_c,
                    save_dir_onset_overt_test, subj, cond_code,
                    tag_w4c, mae_c, pct_50_c)
                _plot_onset_curves_examples(
                    trial_records_c, save_dir_onset_overt_test, subj, cond_code, tag_w4c)

                sanity_metrics_list.append({
                    'tag':             tag_w4c,
                    'win_condition':   'W4c',
                    'MAE_ms':          round(mae_c, 2),
                    'median_AE_ms':    round(med_c, 2),
                    'pct_within_50ms': round(pct_50_c, 2),
                    'n_valid':         int(valid_c.sum()),
                    'band_name':       band_name_s,
                    'band_idx':        band_idx_s,
                    'pre_tp':          None,   # consonant-specific, not a single value
                    'post_tp':         post_tp_base_w4c,
                })

        if sanity_metrics_list:
            save_sanity_metrics(sanity_metrics_list, save_dir_onset_overt_test, subj, cond_code)

            top5 = sorted(sanity_metrics_list, key=lambda m: m['MAE_ms'])[:5]
            print(f'\n  Top-5 window/band conditions by MAE:')
            print(f'  {"Rank":<5} {"Tag":<55} {"MAE (ms)":<12} {"% ≤50ms"}')
            print(f'  {"-"*90}')
            for rank, m in enumerate(top5, 1):
                print(f'  {rank:<5} {m["tag"]:<55} {m["MAE_ms"]:<12.1f} {m["pct_within_50ms"]:.1f}%')
    else:
        print(f'\n  [3/5] Skipping sanity check (--skip-sanity-check)')

    # ==================================================================
    # [4/5] Baseline — full epoch, all-keep covert ICs, all bands
    # ==================================================================
    print(f'\n{sep}')
    print('  [4/5] Baseline — full covert epoch (0 → 1950 ms), all-keep ICs, all bands')
    print(f'{sep}')

    all_results = []
    t_full      = slice(idx_0ms_covert, idx_1950ms_covert)
    n_full_tp   = idx_1950ms_covert - idx_0ms_covert
    time_vec_full = times_covert[idx_0ms_covert:idx_1950ms_covert]

    X_baseline = build_X(
        z_power_covert_smooth, covert_keep_0idx,
        band_idx=None, time_slice=t_full)

    baseline_result = run_covert_exp(
        'baseline_keepIC_all_bands_full',
        X_baseline, y_covert, save_dir_covert, subj, cond_code,
        ic_set='all_keep', band_set='all_bands',
        feature='z_power_smooth', window='0-1950ms',
        inner_jobs=inner_jobs,
        nICs=len(covert_keep_0idx),
        nBands=z_power_covert_smooth.shape[1],
        nTime=n_full_tp,
        ic_labels=ic_labels_covert_keep,
        time_vec=time_vec_full,
        **fp)
    all_results.append(baseline_result)

    # Extract best hyperparameters from baseline for all subsequent experiments
    if fp['fix_svm_C'] is None:
        fp = dict(
            fix_svm_C        = baseline_result.svm_best_params.get('svm__C'),
            fix_rf_depth     = baseline_result.rf_best_params.get('rf__max_depth'),
            fix_rf_features  = baseline_result.rf_best_params.get('rf__max_features'),
            fix_rf_estimators= baseline_result.rf_best_params.get('rf__n_estimators'),
            fix_rf_split     = baseline_result.rf_best_params.get('rf__min_samples_split'),
        )

        fsep = '=' * 60
        print(f'\n{fsep}')
        print(f'  RECOMMENDED FIXED PARAMS (from covert baseline) — {subj} | {cond_code}')
        print(fsep)
        print(f'  SVM: C={fp["fix_svm_C"]}  kernel=linear')
        print(f'  RF:  max_depth={fp["fix_rf_depth"]}  '              f'max_features={fp["fix_rf_features"]}  '              f'n_estimators={fp["fix_rf_estimators"]}  '              f'min_samples_split={fp["fix_rf_split"]}')
        print('  (derived from covert baseline: all-keep ICs, z-power, full epoch)')
        print('  Note: params fixed across all experiments for fair cross-experiment comparison')
        print(fsep)

        fixed_params_path = os.path.join(
            save_dir_covert, f'{subj}_{cond_code}_recommended_fixed_params.csv')
        pd.DataFrame([
            {'param': 'fix_svm_C',         'value': fp['fix_svm_C']},
            {'param': 'fix_rf_depth',
                'value': 'None' if fp['fix_rf_depth'] is None else fp['fix_rf_depth']},
            {'param': 'fix_rf_features',    'value': fp['fix_rf_features']},
            {'param': 'fix_rf_estimators',  'value': fp['fix_rf_estimators']},
            {'param': 'fix_rf_split',       'value': fp['fix_rf_split']},
        ]).to_csv(fixed_params_path, index=False)
        print(f'  Fixed params saved: {fixed_params_path}\n')

    # ==================================================================
    # [5/5] Template-matching experiments
    #        Phase A (sequential): build overt templates + estimate covert onsets
    #                              + extract covert feature matrices for every
    #                              window condition × band set × IC condition.
    #        Phase B (parallel):   run all SVM/RF classifiers simultaneously.
    #        With hyperparams fixed from baseline, GridSearchCV is skipped in
    #        every template-matching call, so inner_jobs=1 avoids over-subscription.
    # ==================================================================
    print(f'\n{sep}')
    print('  [5/5] Template-matching experiments')
    print(f'{sep}')

    ic_blocks = [
        ('all_keep',    covert_keep_0idx,    ic_labels_covert_keep),
    ]
    pending    = []   # delayed(run_covert_exp)(...) for every template-matching exp
    exp_configs = {}  # exp_name → config dict for permutation test reconstruction

    # ------------------------------------------------------------------
    # W2: speech window [onset, onset + N]
    # ------------------------------------------------------------------
    if 'W2' in args.window_conditions:
        print(f'\n  --- W2 — speech window (onset → onset+{speech_window_ms_label}ms) ---')

        for band_name, band_idx in band_items:
            print(f'\n  W2 templates + covert onset estimation: {band_name}')
            templates_w2, win_len_w2 = build_class_templates(
                z_power_overt_smooth, y_overt, onset_tps,
                overt_matched_0idx, band_idx,
                pre_tp=0, post_tp=speech_window_tp)

            est_onset_tps, _, peak_corrs = estimate_covert_onsets(
                z_power_covert_smooth, y_covert,
                templates_w2, win_len_w2,
                covert_matched_0idx, band_idx,
                pre_tp=0,  # template starts AT onset
                search_start=search_start_idx,
                search_end=search_end_bound,
                step=args.slide_step)

            print(f'  Peak correlation: '
                  f'mean={np.mean(peak_corrs):.3f}  '
                  f'min={np.min(peak_corrs):.3f}  '
                  f'max={np.max(peak_corrs):.3f}')

            print(f'  Estimated covert articulation onsets (ms from stimulus onset): '
                  f'mean={np.mean(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'min={np.min(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'max={np.max(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}')

            time_vec = np.arange(0, speech_window_tp) / FS * 1000

            for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
                _ename = f'W2_est_with_{band_name}_{ic_tag}IC'
                X = build_X_speech_window(
                    z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
                    pre_onset_tp=0, post_onset_tp=speech_window_tp,
                    band_idx=band_idx)
                pending.append(delayed(run_covert_exp)(
                    _ename,
                    X, y_covert, save_dir_covert, subj, cond_code,
                    ic_set=ic_tag, band_set=band_name,
                    feature='z_power_smooth',
                    window=f'est_covert_articulation_window_{speech_window_tp}tp',
                    inner_jobs=1, band_idx_vec=band_idx,
                    nICs=len(ic_0idx_feats), nBands=len(band_idx),
                    nTime=speech_window_tp,
                    ic_labels=ic_lbl_list, time_vec=time_vec, **fp))
                exp_configs[_ename] = dict(
                    win_cond='W2', band_name=band_name, band_idx=band_idx,
                    pre_tp=0, post_tp=speech_window_tp)

            # for feat_band_name, feat_band_idx in band_items:
            #     for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
            #         X = build_X_speech_window(
            #             z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
            #             pre_onset_tp=0, post_onset_tp=speech_window_tp,
            #             band_idx=feat_band_idx)
            #         pending.append(delayed(run_covert_exp)(
            #             f'W2_est_with_{band_name}_features_{feat_band_name}_{ic_tag}IC',
            #             X, y_covert, save_dir_covert, subj, cond_code,
            #             ic_set=ic_tag, band_set=feat_band_name,
            #             feature='z_power_smooth',
            #             window=f'est_covert_articulation_window_{speech_window_tp}tp',
            #             inner_jobs=1, band_idx_vec=feat_band_idx,
            #             nICs=len(ic_0idx_feats), nBands=len(feat_band_idx),
            #             nTime=speech_window_tp,
            #             ic_labels=ic_lbl_list, time_vec=time_vec, **fp))

    # ------------------------------------------------------------------
    # W3: pre-speech only [onset − 500ms, onset]
    # ------------------------------------------------------------------
    if 'W3' in args.window_conditions:
        print(f'\n  --- W3 — pre-speech (onset−{W3_PRE_ONSET_MS}ms → onset) ---')

        for band_name, band_idx in band_items:
            print(f'\n  W3 templates + covert onset estimation: {band_name}')
            templates_w3, win_len_w3 = build_class_templates(
                z_power_overt_smooth, y_overt, onset_tps,
                overt_matched_0idx, band_idx,
                pre_tp=pre_onset_w3_tp, post_tp=0)

            est_onset_tps, _, peak_corrs = estimate_covert_onsets(
                z_power_covert_smooth, y_covert,
                templates_w3, win_len_w3,
                covert_matched_0idx, band_idx,
                pre_tp=pre_onset_w3_tp,  # template starts 500ms BEFORE onset
                search_start=search_start_idx,
                search_end=search_end_bound,
                step=args.slide_step)

            print(f'  Peak correlation: '
                  f'mean={np.mean(peak_corrs):.3f}  '
                  f'min={np.min(peak_corrs):.3f}  '
                  f'max={np.max(peak_corrs):.3f}')

            print(f'  Estimated covert articulation onsets (ms from stimulus onset): '
                  f'mean={np.mean(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'min={np.min(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'max={np.max(est_onset_tps - idx_0ms_covert) / FS * 1000:.1f}')

            time_vec = np.arange(-pre_onset_w3_tp, 0) / FS * 1000

            for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
                _ename = f'W3_est_with_{band_name}_{ic_tag}IC_pre{W3_PRE_ONSET_MS}ms'
                X = build_X_speech_window(
                    z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
                    pre_onset_tp=pre_onset_w3_tp, post_onset_tp=0,
                    band_idx=band_idx)
                pending.append(delayed(run_covert_exp)(
                    _ename,
                    X, y_covert, save_dir_covert, subj, cond_code,
                    ic_set=ic_tag, band_set=band_name,
                    feature='z_power_smooth',
                    window=f'est_onset-{pre_onset_w3_tp}tp_to_est_onset+0tp',
                    inner_jobs=1, band_idx_vec=band_idx,
                    nICs=len(ic_0idx_feats), nBands=len(band_idx),
                    nTime=pre_onset_w3_tp,
                    ic_labels=ic_lbl_list, time_vec=time_vec, **fp))
                exp_configs[_ename] = dict(
                    win_cond='W3', band_name=band_name, band_idx=band_idx,
                    pre_tp=pre_onset_w3_tp, post_tp=0)

            # for feat_band_name, feat_band_idx in band_items:
            #     for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
            #         X = build_X_speech_window(
            #             z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
            #             pre_onset_tp=pre_onset_w3_tp, post_onset_tp=0,
            #             band_idx=feat_band_idx)
            #         pending.append(delayed(run_covert_exp)(
            #             f'W3_est_with_{band_name}_features_{feat_band_name}_{ic_tag}IC_pre{W3_PRE_ONSET_MS}ms',
            #             X, y_covert, save_dir_covert, subj, cond_code,
            #             ic_set=ic_tag, band_set=feat_band_name,
            #             feature='z_power_smooth',
            #             window=f'est_onset-{pre_onset_w3_tp}tp_to_est_onset+0tp',
            #             inner_jobs=1, band_idx_vec=feat_band_idx,
            #             nICs=len(ic_0idx_feats), nBands=len(feat_band_idx),
            #             nTime=pre_onset_w3_tp,
            #             ic_labels=ic_lbl_list, time_vec=time_vec, **fp))

    # ------------------------------------------------------------------
    # W4u: uniform pre-onset [onset − p*, onset + N]
    # ------------------------------------------------------------------
    if run_w4u:
        p_star = args.best_overall_pre_onset_ms
        print(f'\n  --- W4u — uniform pre-onset '
              f'(onset-{p_star}ms → onset+{speech_window_ms_label}ms) ---')

        for band_name, band_idx in band_items:
            print(f'\n  W4u templates + covert onset estimation: {band_name}')
            templates_w4u, win_len_w4u = build_class_templates(
                z_power_overt_smooth, y_overt, onset_tps,
                overt_matched_0idx, band_idx,
                pre_tp=pre_onset_w4u_tp, post_tp=speech_window_tp)

            est_onset_tps, _, peak_corrs = estimate_covert_onsets(
                z_power_covert_smooth, y_covert,
                templates_w4u, win_len_w4u,
                covert_matched_0idx, band_idx,
                pre_tp=pre_onset_w4u_tp,
                search_start=search_start_idx,
                search_end=search_end_bound,
                step=args.slide_step)

            print(f'  Peak correlation: '
                  f'mean={np.mean(peak_corrs):.3f}  '
                  f'min={np.min(peak_corrs):.3f}  '
                  f'max={np.max(peak_corrs):.3f}')

            print(f'  Estimated covert pre-articulation brain activation onsets (ms from stimulus onset): '
                  f'mean={np.mean(est_onset_tps-pre_onset_w4u_tp - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'min={np.min(est_onset_tps-pre_onset_w4u_tp - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'max={np.max(est_onset_tps-pre_onset_w4u_tp - idx_0ms_covert) / FS * 1000:.1f}')

            nTime_w4u = pre_onset_w4u_tp + speech_window_tp
            time_vec  = np.arange(-pre_onset_w4u_tp, speech_window_tp) / FS * 1000

            for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
                _ename = (f'W4u_est_with_{band_name}_{ic_tag}IC'
                          f'_pre{p_star}ms_speech{speech_window_ms_label}ms')
                X = build_X_speech_window(
                    z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
                    pre_onset_tp=pre_onset_w4u_tp, post_onset_tp=speech_window_tp,
                    band_idx=band_idx)
                pending.append(delayed(run_covert_exp)(
                    _ename,
                    X, y_covert, save_dir_covert, subj, cond_code,
                    ic_set=ic_tag, band_set=band_name,
                    feature='z_power_smooth',
                    window=f'est_onset-{pre_onset_w4u_tp}tp_to_est_onset+{speech_window_tp}tp',
                    inner_jobs=1, band_idx_vec=band_idx,
                    nICs=len(ic_0idx_feats), nBands=len(band_idx), nTime=nTime_w4u,
                    ic_labels=ic_lbl_list, time_vec=time_vec, **fp))
                exp_configs[_ename] = dict(
                    win_cond='W4u', band_name=band_name, band_idx=band_idx,
                    pre_tp=pre_onset_w4u_tp, post_tp=speech_window_tp)

            # for feat_band_name, feat_band_idx in band_items:
            #     for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
            #         X = build_X_speech_window(
            #             z_power_covert_smooth, ic_0idx_feats, est_onset_tps,
            #             pre_onset_tp=pre_onset_w4u_tp, post_onset_tp=speech_window_tp,
            #             band_idx=feat_band_idx)
            #         pending.append(delayed(run_covert_exp)(
            #             f'W4u_est_with_{band_name}_features_{feat_band_name}_{ic_tag}IC_pre{p_star}ms_speech{speech_window_ms_label}ms',
            #             X, y_covert, save_dir_covert, subj, cond_code,
            #             ic_set=ic_tag, band_set=feat_band_name,
            #             feature='z_power_smooth',
            #             window=f'est_onset-{pre_onset_w4u_tp}tp_to_est_onset+{speech_window_tp}tp',
            #             inner_jobs=1, band_idx_vec=feat_band_idx,
            #             nICs=len(ic_0idx_feats), nBands=len(feat_band_idx), nTime=nTime_w4u,
            #             ic_labels=ic_lbl_list, time_vec=time_vec, **fp))

    # ------------------------------------------------------------------
    # W4c: consonant-group-specific pre-onset
    # ------------------------------------------------------------------
    if run_w4c:
        print('\n  --- W4c — consonant-specific pre-onset ---')
        for name, pre_tp in consonant_pre_tps.items():
            syls = [SYLLABLES[lbl - 1] for lbl in CONSONANT_GROUPS[name]]
            print(f'    {name:20s} ({"/".join(syls)}):  '
                  f'pre={pre_tp} samples ({pre_tp/FS*1000:.0f} ms)  '
                  f'post={post_tp_base_w4c} samples')

        max_pre_tp_w4c   = max(consonant_pre_tps.values())
        total_win_tp_w4c = max_pre_tp_w4c + post_tp_base_w4c
        total_win_ms_w4c = int(total_win_tp_w4c / FS * 1000)
        t_vec_w4c        = np.arange(0, total_win_tp_w4c) / FS * 1000

        for band_name, band_idx in band_items:
            print(f'\n  W4c templates + covert onset estimation: {band_name}')
            templates_w4c, _ = build_class_templates_consonant(
                z_power_overt_smooth, y_overt, onset_tps,
                overt_matched_0idx, band_idx,
                consonant_pre_tps, post_tp_base_w4c)

            est_onset_tps_c, _, peak_corrs_c = estimate_covert_onsets_consonant(
                z_power_covert_smooth, y_covert,
                templates_w4c, consonant_pre_tps, post_tp_base_w4c,
                covert_matched_0idx, band_idx,
                search_start=search_start_idx,
                search_end=search_end_bound,
                step=args.slide_step)

            print(f'  Peak correlation: '
                  f'mean={np.mean(peak_corrs_c):.3f}  '
                  f'min={np.min(peak_corrs_c):.3f}  '
                  f'max={np.max(peak_corrs_c):.3f}')

            print(f'  Estimated consonant-group-specific covert articulation onsets (ms from stimulus onset): '
                  f'mean={np.mean(est_onset_tps_c - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'min={np.min(est_onset_tps_c - idx_0ms_covert) / FS * 1000:.1f}  '
                  f'max={np.max(est_onset_tps_c - idx_0ms_covert) / FS * 1000:.1f}')

            for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
                _ename = f'W4c_est_with_{band_name}_{ic_tag}IC_total{total_win_ms_w4c}ms'
                X_w4c, _ = build_X_covert_consonant(
                    z_power_covert_smooth, y_covert,
                    est_onset_tps_c,
                    ic_0idx_feats, band_idx,
                    consonant_pre_tps, post_tp_base_w4c)
                pending.append(delayed(run_covert_exp)(
                    _ename,
                    X_w4c, y_covert, save_dir_covert, subj, cond_code,
                    ic_set=ic_tag, band_set=band_name,
                    feature='z_power_smooth',
                    window=f'consonant_specific_total{total_win_ms_w4c}ms',
                    inner_jobs=1, band_idx_vec=band_idx,
                    nICs=len(ic_0idx_feats), nBands=len(band_idx),
                    nTime=total_win_tp_w4c,
                    ic_labels=ic_lbl_list, time_vec=t_vec_w4c, **fp))
                exp_configs[_ename] = dict(
                    win_cond='W4c', band_name=band_name, band_idx=band_idx,
                    pre_tp=None, post_tp=post_tp_base_w4c)

            # for feat_band_name, feat_band_idx in band_items:
            #     for ic_tag, ic_0idx_feats, ic_lbl_list in ic_blocks:
            #         X_w4c, _ = build_X_covert_consonant(
            #             z_power_covert_smooth, y_covert,
            #             est_onset_tps_c,
            #             ic_0idx_feats, feat_band_idx,
            #             consonant_pre_tps, post_tp_base_w4c)
            #         pending.append(delayed(run_covert_exp)(
            #             f'W4c_est_with_{band_name}_features_{feat_band_name}_{ic_tag}IC_total{total_win_ms_w4c}ms',
            #             X_w4c, y_covert, save_dir_covert, subj, cond_code,
            #             ic_set=ic_tag, band_set=feat_band_name,
            #             feature='z_power_smooth',
            #             window=f'consonant_specific_total{total_win_ms_w4c}ms',
            #             inner_jobs=1, band_idx_vec=feat_band_idx,
            #             nICs=len(ic_0idx_feats), nBands=len(feat_band_idx),
            #             nTime=total_win_tp_w4c,
            #             ic_labels=ic_lbl_list, time_vec=t_vec_w4c, **fp))

    # ------------------------------------------------------------------
    # Phase B: run all template-matching experiments in parallel
    # ------------------------------------------------------------------
    print(f'\n  Launching {len(pending)} template-matching experiments in parallel '
          f'(threading, n_jobs=-1)...')
    exp_results = Parallel(n_jobs=-1, prefer='threads')(pending)
    all_results.extend(exp_results)

    # Per-condition summaries
    if 'W2' in args.window_conditions:
        print_and_save_summary(
            [r for r in exp_results if 'W2_' in r.name],
            save_dir_covert, subj, cond_code, tag='W2')
    if 'W3' in args.window_conditions:
        print_and_save_summary(
            [r for r in exp_results if 'W3_' in r.name],
            save_dir_covert, subj, cond_code, tag='W3')
    if run_w4u:
        print_and_save_summary(
            [r for r in exp_results if 'W4u_' in r.name],
            save_dir_covert, subj, cond_code, tag='W4u')
    if run_w4c:
        print_and_save_summary(
            [r for r in exp_results if 'W4c_' in r.name],
            save_dir_covert, subj, cond_code, tag='W4c')

    # ==================================================================
    # Combined summary (all experiments)
    # ==================================================================
    print_and_save_summary(all_results, save_dir_covert, subj, cond_code,
                           tag='all_covert_exps')

    # ==================================================================
    # Summary figures
    # ==================================================================
    print(f'\n{sep}')
    print('  Generating summary figures...')
    print(f'{sep}')
    plot_covert_summary_figures(all_results, save_dir_figures, subj, cond_code)

    # ==================================================================
    # [6/6] Permutation baseline (top-k covert classification experiments)
    # ==================================================================
    if not args.skip_permutation_test and exp_results:
        print(f'\n{sep}')
        print(f'  [6/6] Permutation baseline — top {args.permutation_top_k} covert '
              f'classification experiments (N={args.n_permutations})')
        print(f'{sep}')

        save_dir_perm = os.path.join(save_dir_root, 'permutation_baseline')
        os.makedirs(save_dir_perm, exist_ok=True)

        # Rank template-matching experiments by mean balanced accuracy (SVM + RF).
        # Exclude the full-epoch baseline (no onset estimation to test).
        template_exps = [r for r in exp_results if not r.name.startswith('baseline')]
        top_k_exps = sorted(
            template_exps,
            key=lambda r: (r.svm_bal_acc + r.rf_bal_acc) / 2,
            reverse=True,
        )[:args.permutation_top_k]

        print(f'  Top-{args.permutation_top_k} experiments by mean balanced accuracy:')
        print(f'  {"Rank":<5} {"Experiment":<55} {"SVM bal":<10} {"RF bal"}')
        print(f'  {"-"*85}')
        for rank, r in enumerate(top_k_exps, 1):
            print(f'  {rank:<5} {r.name:<55} {r.svm_bal_acc:<10.3f} {r.rf_bal_acc:.3f}')

        top_k_configs = []
        for r in top_k_exps:
            cfg = exp_configs.get(r.name)
            if cfg is None:
                print(f'  WARNING: no config found for {r.name}, skipping.')
                continue
            top_k_configs.append(dict(cfg, exp_name=r.name, tag=r.name))

        if top_k_configs:
            run_permutation_baseline(
                top_k_configs,
                z_power_overt_smooth, y_overt, onset_tps,
                z_power_covert_smooth, y_covert,
                overt_matched_0idx, covert_matched_0idx,
                covert_keep_0idx,
                search_start_idx, search_end_bound, args.slide_step,
                fp,
                n_perms=args.n_permutations,
                exp_results=exp_results,
                save_dir=save_dir_perm, subj=subj, cond_code=cond_code,
                consonant_pre_tps=consonant_pre_tps if run_w4c else None,
                post_tp_base_w4c=post_tp_base_w4c  if run_w4c else None,
            )
        else:
            print('  No configs found for any top-k experiment.')

    elif args.skip_permutation_test:
        print(f'\n  [6/6] Skipping permutation baseline (--skip-permutation-test)')
    else:
        print(f'\n  [6/6] Skipping permutation baseline (no template-matching results)')

    print(f'\n{sep}')
    print(f'  Done. Results in: {save_dir_covert}')
    print(f'{sep}\n')


if __name__ == '__main__':
    main()
