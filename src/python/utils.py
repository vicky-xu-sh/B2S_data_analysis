#!/usr/bin/env python3
"""
utils.py — Shared utilities for EEG syllable classification scripts.

Imported by:
  - classification_overt.py
  - classification_overt_W4_sweep.py
  - classification_overt_band_sweep.py
"""

from constants import (
    SYLLABLES, FS, LP_CUTOFF_HZ, LP_ORDER,
    BASELINE_START_MS, BASELINE_END_MS, BAND_NAMES,
    C_PARAMS, KERNEL_PARAMS, ESTIMATORS_PARAMS, DEPTH_PARAMS, SPLITS_PARAMS,
)

import math
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy.signal import butter, sosfiltfilt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score

warnings.filterwarnings('ignore', category=UserWarning)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name:            str
    feature:         str
    ic_set:          str
    band_set:        str
    window:          str
    n_features:      int
    svm_best_params: dict = field(default_factory=dict)
    svm_accuracy:    float = 0.0
    svm_bal_acc:     float = 0.0
    svm_per_class:   dict = field(default_factory=dict)
    svm_precision:   dict = field(default_factory=dict)
    rf_best_params:  dict = field(default_factory=dict)
    rf_accuracy:     float = 0.0
    rf_bal_acc:      float = 0.0
    rf_per_class:    dict = field(default_factory=dict)
    rf_precision:    dict = field(default_factory=dict)
    runtime_s:       float = 0.0
 
 
# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
 
def lowpass_smooth(data, cutoff_hz=LP_CUTOFF_HZ, fs=FS, order=LP_ORDER, axis=2):
    """
    Zero-phase Butterworth low-pass filter along the time axis.
    Reportable as: "4th-order zero-phase Butterworth low-pass filter at 10 Hz"
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, data, axis=axis)
 
 
def load_analytic(filepath):
    """
    Load analytic signal, labels, and times from HDF5 .mat file.
 
    Returns
    -------
    Z      : np.ndarray complex [comps x bands x time x trials]
    labels : np.ndarray [trials]
    times  : np.ndarray [time], in ms
    """
    with h5py.File(filepath, 'r') as f:
        Z_raw  = f['analytic_signal'][:]
        labels = f['labels'][:].squeeze()
        times  = f['times'][:].squeeze()
    # h5py transposes MATLAB arrays: restore [comps x bands x time x trials]
    Z_raw = Z_raw.transpose(3, 2, 1, 0)
    Z = Z_raw['real'] + 1j * Z_raw['imag']
    return Z, labels, times
 
 
def compute_features(Z, times):
    """
    Compute instantaneous power, phase, frequency, and z-scored power.
 
    Parameters
    ----------
    Z     : np.ndarray complex [comps x bands x time x trials]
    times : np.ndarray [time], in ms
 
    Returns
    -------
    power, phase, inst_freq, z_power — all np.ndarray
    """
    power     = np.abs(Z) ** 2
    phase     = np.angle(Z)
    inst_freq = np.diff(np.unwrap(phase, axis=2), axis=2) / (2 * np.pi) * FS
 
    baseline_mask = (times >= BASELINE_START_MS) & (times <= BASELINE_END_MS)
    mu      = power[..., baseline_mask, :].mean(axis=-2, keepdims=True)
    std     = power[..., baseline_mask, :].std(axis=-2, keepdims=True)
    z_power = (power - mu) / (std + 1e-8)
 
    return power, phase, inst_freq, z_power
 
 
def reject_epochs(bad_epochs_1idx, *arrays):
    """
    Remove bad epochs (1-indexed) from arrays with trials on the last axis.
 
    Returns
    -------
    good_mask : np.ndarray bool [trials]
    arrays    : tuple of np.ndarray with bad trials removed
    """
    n_trials  = arrays[0].shape[-1]
    good_mask = np.ones(n_trials, dtype=bool)
    if len(bad_epochs_1idx) > 0:
        good_mask[np.array(bad_epochs_1idx) - 1] = False
    return good_mask, tuple(a[..., good_mask] for a in arrays)
 
 
def compute_speech_window_tp(onset_tps, offset_tps):
    """
    Compute subject-specific speech window length in samples.
    Rounds UP to nearest 50 samples (= 100ms at 500Hz).
 
    Parameters
    ----------
    onset_tps  : np.ndarray [trials], sample indices of speech onset
    offset_tps : np.ndarray [trials], sample indices of speech offset
 
    Returns
    -------
    speech_window_tp : int
    """
    mean_len = np.mean(offset_tps - onset_tps)
    max_len = np.max(offset_tps - onset_tps)
    min_len = np.min(offset_tps - onset_tps)
    speech_window_tp = int(math.ceil(mean_len / 50) * 50)
    print('\n  Speech window (W2):')
    print(f'    Mean speech duration:  {mean_len:.1f} samples ({mean_len/FS*1000:.0f} ms)')
    print(f'    Max speech duration:   {max_len:.1f} samples ({max_len/FS*1000:.0f} ms)')
    print(f'    Min speech duration:   {min_len:.1f} samples ({min_len/FS*1000:.0f} ms)')
    print(f'    Rounded window length: {speech_window_tp} samples ({speech_window_tp/FS*1000:.0f} ms)')
    print('    Override with --speech-window-ms if needed\n')
    return speech_window_tp
 
 
def derive_epoch_indices(times):
    """
    Derive sample indices for key epoch timepoints from the times array.
    More robust than hardcoding POST_STIM_0MS etc.
 
    Returns
    -------
    idx_0ms   : int, sample index of t=0ms
    idx_1950ms: int, sample index closest to t=1950ms
    """
    idx_0ms    = int(np.searchsorted(times, 0))
    idx_1950ms = int(np.searchsorted(times, 1950))
    assert times[idx_0ms] == 0, \
        f"Expected t=0ms at index {idx_0ms}, got {times[idx_0ms]:.1f}ms"
    print(f'  Epoch indices: t=0ms → sample {idx_0ms}, '
          f't={times[idx_1950ms]:.0f}ms → sample {idx_1950ms}')
    return idx_0ms, idx_1950ms
 
 
# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
 
def build_X(data, ic_0idx, band_idx=None, time_slice=None):
    """
    Extract and flatten feature matrix from [ICs x bands x time x trials].
 
    Parameters
    ----------
    data       : np.ndarray [ICs x bands x time x trials]
    ic_0idx    : list of int, 0-indexed IC indices
    band_idx   : list of int or None — which bands to include (None = all)
    time_slice : slice or None — which time samples to include (None = all)
 
    Returns
    -------
    X : np.ndarray [trials x features]
    """
    time_slice_str = f"{time_slice.start}:{time_slice.stop}" if time_slice is not None else "all"
    print(f"  Building feature matrix with {len(ic_0idx)} ICs,"
          f" {len(band_idx) if band_idx is not None else data.shape[1]} bands,"
          f" {data.shape[2]} time samples (time slice: {time_slice_str})")
    sliced = data[np.ix_(ic_0idx,
                         band_idx if band_idx is not None else list(range(data.shape[1])),
                         list(range(data.shape[2])),
                         list(range(data.shape[3])))]
    if time_slice is not None:
        sliced = sliced[:, :, time_slice, :]
    # [ICs x bands x time x trials] → [trials x ICs x bands x time]
    sliced = sliced.transpose(3, 0, 1, 2)
    return sliced.reshape(sliced.shape[0], -1)
 
 
def build_X_speech_window(data, ic_0idx, onset_tps,
                           pre_onset_tp=0, post_onset_tp=300,
                           band_idx=None):
    """
    Extract per-trial speech-aligned feature matrix.
 
    Window per trial: [onset_tp - pre_onset_tp, onset_tp + post_onset_tp]
    Trials where the window falls outside available data are zero-filled.
 
    Parameters
    ----------
    data          : np.ndarray [ICs x bands x time x trials]
    ic_0idx       : list of int, 0-indexed IC indices
    onset_tps     : np.ndarray [trials], sample index of speech onset per trial
    pre_onset_tp  : int, samples before onset to include
    post_onset_tp : int, samples after onset to include
    band_idx      : list of int or None
 
    Returns
    -------
    X : np.ndarray [trials x features]
    """
    nTrials = data.shape[-1]
    nICs    = len(ic_0idx)
    bands   = band_idx if band_idx is not None else list(range(data.shape[1]))
    nBands  = len(bands)
    win_len = pre_onset_tp + post_onset_tp
    total_T = data.shape[2]
 
    print(f"  Building speech-aligned feature matrix with {nICs} ICs, {nBands} bands,"
          f" {win_len} time samples (pre-onset: {pre_onset_tp} samples, post-onset: {post_onset_tp} samples)")
 
    out = np.zeros((nICs, nBands, win_len, nTrials))
 
    for i in range(nTrials):
        start = int(onset_tps[i]) - pre_onset_tp
        end   = int(onset_tps[i]) + post_onset_tp
 
        if start < 0 or end > total_T:
            print(f"  Trial {i}: window [{start}, {end}] out of bounds — zero-filled")
            continue
 
        out[:, :, :, i] = data[np.ix_(ic_0idx, bands,
                                       list(range(start, end)),
                                       [i])][:, :, :, 0]
 
    # [ICs x bands x time x trials] → [trials x features]
    out = out.transpose(3, 0, 1, 2)
    return out.reshape(nTrials, -1)
 
 
# ---------------------------------------------------------------------------
# Classifier pipelines
# ---------------------------------------------------------------------------
 
def _make_svm_pipeline(inner_n_jobs):
    pipeline   = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
    param_grid = {'svm__C': C_PARAMS, 'svm__kernel': KERNEL_PARAMS}
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy',
                        n_jobs=inner_n_jobs, verbose=0), cv
 
 
def _make_rf_pipeline(inner_n_jobs):
    pipeline   = Pipeline([('rf', RandomForestClassifier(random_state=42))])
    param_grid = {
        'rf__n_estimators':      ESTIMATORS_PARAMS,
        'rf__max_depth':         DEPTH_PARAMS,
        'rf__min_samples_split': SPLITS_PARAMS,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy',
                        n_jobs=inner_n_jobs, verbose=0), cv
 
 
def _per_class_recall(y_true, y_pred):
    """Return dict {syllable: recall} from cross-validated predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, 7)))
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.where(cm.sum(axis=1) > 0,
                          cm.diagonal() / cm.sum(axis=1), 0.0)
    return {syl: float(recall[i]) for i, syl in enumerate(SYLLABLES)}
 
 
def _per_class_precision(y_true, y_pred):
    """Return dict {syllable: precision} from cross-validated predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, 7)))
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(cm.sum(axis=0) > 0,
                             cm.diagonal() / cm.sum(axis=0), 0.0)
    return {syl: float(precision[i]) for i, syl in enumerate(SYLLABLES)}
 
 
def _safe_name(s):
    """Convert experiment name to a filesystem-safe string."""
    for ch in (' ', '|', '/', '(', ')', ',', '+', '-', '→'):
        s = s.replace(ch, '_')
    return s.strip('_')
 
 
# ---------------------------------------------------------------------------
# Run classifiers
# ---------------------------------------------------------------------------
 
def run_classifiers(name, X, y, save_dir, subj, cond,
                    ic_set, band_set, feature, window,
                    inner_n_jobs=-1):
    """
    Run SVM + RF gridsearch, save confusion matrices, return ExperimentResult
    and the fitted RF model (for feature importance).
 
    Parameters
    ----------
    name          : str, experiment label
    X             : np.ndarray [trials x features]
    y             : np.ndarray [trials], integer class labels (1-indexed)
    save_dir      : str, output directory
    subj          : str
    cond          : str, condition code e.g. 'sp'
    ic_set        : str, label for IC set used
    band_set      : str, label for band set used
    feature       : str, label for feature type
    window        : str, label for time window
    inner_n_jobs  : int, n_jobs for GridSearchCV
 
    Returns
    -------
    result    : ExperimentResult
    rf_model  : fitted RandomForestClassifier (named_steps['rf'])
    """
    t0     = time.time()
    result = ExperimentResult(
        name=name, feature=feature, ic_set=ic_set,
        band_set=band_set, window=window, n_features=X.shape[1])
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
    # --- SVM ---
    print(f"    [SVM] {name} ...")
    gs_svm, _ = _make_svm_pipeline(inner_n_jobs)
    gs_svm.fit(X, y)
    best_svm   = gs_svm.best_estimator_
    y_pred_svm = cross_val_predict(best_svm, X, y, cv=cv, n_jobs=inner_n_jobs)
    acc_svm    = (y_pred_svm == y).mean()
    bal_svm    = balanced_accuracy_score(y, y_pred_svm)
 
    result.svm_best_params = gs_svm.best_params_
    result.svm_accuracy    = acc_svm
    result.svm_bal_acc     = bal_svm
    result.svm_per_class   = _per_class_recall(y, y_pred_svm)
    result.svm_precision   = _per_class_precision(y, y_pred_svm)
 
    _save_confusion_matrix(
        y, y_pred_svm, save_dir,
        title=(f'{subj} | {name}\n'
               f'SVM  acc={acc_svm:.3f}  bal={bal_svm:.3f} | {gs_svm.best_params_}'),
        fname=f'{subj}_{cond}_{_safe_name(name)}_SVM_cm.png')
 
    # --- RF ---
    print(f"    [RF]  {name} ...")
    gs_rf, _ = _make_rf_pipeline(inner_n_jobs)
    gs_rf.fit(X, y)
    best_rf   = gs_rf.best_estimator_
    y_pred_rf = cross_val_predict(best_rf, X, y, cv=cv, n_jobs=inner_n_jobs)
    acc_rf    = (y_pred_rf == y).mean()
    bal_rf    = balanced_accuracy_score(y, y_pred_rf)
 
    result.rf_best_params = gs_rf.best_params_
    result.rf_accuracy    = acc_rf
    result.rf_bal_acc     = bal_rf
    result.rf_per_class   = _per_class_recall(y, y_pred_rf)
    result.rf_precision   = _per_class_precision(y, y_pred_rf)
 
    _save_confusion_matrix(
        y, y_pred_rf, save_dir,
        title=(f'{subj} | {name}\n'
               f'RF   acc={acc_rf:.3f}  bal={bal_rf:.3f} | {gs_rf.best_params_}'),
        fname=f'{subj}_{cond}_{_safe_name(name)}_RF_cm.png')
 
    result.runtime_s = time.time() - t0
    rf_model = best_rf.named_steps['rf']
    return result, rf_model
 
 
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
 
def _save_confusion_matrix(y_true, y_pred, save_dir, title, fname):
    os.makedirs(save_dir, exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(1, 7)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SYLLABLES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(save_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      Saved: {fpath}")
 
 
def plot_feature_importance(rf_model, nICs, nBands, nTime,
                             ic_labels, time_vector,
                             save_dir, subj, cond, exp_name):
    """
    Save band×time heatmap and IC importance bar chart for a fitted RF model.
 
    Parameters
    ----------
    rf_model    : fitted RandomForestClassifier
    nICs        : int
    nBands      : int
    nTime       : int
    ic_labels   : list of str, e.g. ['IC4', 'IC7', ...]
    time_vector : np.ndarray [nTime], time axis in ms
    save_dir    : str
    subj        : str
    cond        : str
    exp_name    : str
    """
    importances = rf_model.feature_importances_
    imp_3d      = importances.reshape((nICs, nBands, nTime))
 
    # Band × Time heatmap
    imp_band_time = imp_3d.mean(axis=0)
    step          = 100
    tick_locs     = np.searchsorted(time_vector, np.arange(time_vector[0], time_vector[-1], step))
    tick_labs     = [f"{int(time_vector[i])}" for i in tick_locs]
 
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(imp_band_time, cmap='viridis', ax=ax,
                yticklabels=BAND_NAMES[:nBands],
                xticklabels=False,
                cbar_kws={'label': 'Feature Importance'})
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labs, rotation=45, fontsize=8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency Band')
    ax.set_title(f'{subj} | {exp_name} — RF Feature Importance: Band × Time')
    plt.tight_layout()
    fpath = os.path.join(save_dir,
        f'{subj}_{cond}_{_safe_name(exp_name)}_RF_importance_band_time.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      Saved: {fpath}")
 
    # IC importance bar chart
    imp_per_ic = imp_3d.mean(axis=(1, 2))
    sorted_idx = np.argsort(imp_per_ic)[::-1]
    fig, ax    = plt.subplots(figsize=(max(8, nICs), 4))
    ax.bar(range(nICs), imp_per_ic[sorted_idx],
           tick_label=[ic_labels[i] for i in sorted_idx])
    ax.set_xlabel('Independent Component')
    ax.set_ylabel('Average Feature Importance')
    ax.set_title(f'{subj} | {exp_name} — RF Feature Importance: IC')
    plt.tight_layout()
    fpath = os.path.join(save_dir,
        f'{subj}_{cond}_{_safe_name(exp_name)}_RF_importance_ic.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      Saved: {fpath}")
 
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
 
def print_and_save_summary(results, save_dir, subj, cond, tag=''):
    """
    Print and save classification summary CSV and per-class recall CSV.
 
    Parameters
    ----------
    results  : list of ExperimentResult
    save_dir : str
    subj     : str
    cond     : str
    tag      : str, optional suffix for filenames (e.g. 'W4_sweep')
    """
    rows = []
    for r in results:
        rows.append({
            'experiment':   r.name,
            'feature':      r.feature,
            'ic_set':       r.ic_set,
            'band_set':     r.band_set,
            'window':       r.window,
            'n_features':   r.n_features,
            'SVM_accuracy': f'{r.svm_accuracy:.3f}',
            'SVM_bal_acc':  f'{r.svm_bal_acc:.3f}',
            'SVM_params':   str(r.svm_best_params),
            'RF_accuracy':  f'{r.rf_accuracy:.3f}',
            'RF_bal_acc':   f'{r.rf_bal_acc:.3f}',
            'RF_params':    str(r.rf_best_params),
            'runtime_s':    f'{r.runtime_s:.1f}',
        })
 
    df  = pd.DataFrame(rows)
    sep = '=' * 120
    suffix = f'_{tag}' if tag else ''
 
    print(f'\n{sep}')
    print(f'  CLASSIFICATION SUMMARY — {subj} | {cond}{f" | {tag}" if tag else ""}')
    print(sep)
    print(df.to_string(index=False))
 
    csv_path = os.path.join(save_dir, f'{subj}_{cond}_classification_summary{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n  Summary saved: {csv_path}')
 
    # Per-class recall (SVM + RF) — printed
    print(f'\n{sep}')
    print('  PER-CLASS RECALL — SVM and RF (cross-validated)')
    print(sep)
    rows_pc = []
    for r in results:
        row = {'experiment': r.name}
        row['SVM_accuracy'] = f'{r.svm_accuracy:.3f}'
        row['RF_accuracy'] = f'{r.rf_accuracy:.3f}'
        row.update({f'SVM_recall_{syl}': f"{r.svm_per_class.get(syl, 0):.3f}"
                    for syl in SYLLABLES})
        row.update({f'RF_recall_{syl}': f"{r.rf_per_class.get(syl, 0):.3f}"
                    for syl in SYLLABLES})
        rows_pc.append(row)
    df_pc = pd.DataFrame(rows_pc)
    print(df_pc.to_string(index=False))
 
    # Precision added to CSV only (not printed — use confusion matrices for full picture)
    for row, r in zip(rows_pc, results):
        row.update({f'SVM_precision_{syl}': f"{r.svm_precision.get(syl, 0):.3f}"
                    for syl in SYLLABLES})
        row.update({f'RF_precision_{syl}': f"{r.rf_precision.get(syl, 0):.3f}"
                    for syl in SYLLABLES})
    df_pc = pd.DataFrame(rows_pc)
 
    csv_pc = os.path.join(save_dir, f'{subj}_{cond}_per_class_recall{suffix}.csv')
    df_pc.to_csv(csv_pc, index=False)
    print(f'\n  Per-class recall + precision saved: {csv_pc}')
    print(f'\n{sep}\n')

