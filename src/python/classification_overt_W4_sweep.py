#!/usr/bin/env python3
"""
classification_overt_W4_sweep.py — Pre-speech onset sweep (overt condition)

W4 — Sweep over pre-speech window prepended to speech segment:
  Window per step: [onset - X ms, onset + speech_window_ms]
  X ∈ W4_SWEEP_MS (defined in constants.py)

After the sweep, four final experiments are run using pre-onset values derived
from the sweep results using two different recommendation strategies:

  RF-only strategy   : pre-onset that maximises RF recall
  Combined strategy  : pre-onset that maximises mean(RF + SVM) recall

  Exp A_RF       — uniform best overall pre-onset (RF-only derived)
  Exp A_combined — uniform best overall pre-onset (combined derived)
  Exp B_RF       — consonant-group-specific pre-onset (RF-only derived)
  Exp B_combined — consonant-group-specific pre-onset (combined derived)

  For Exp B: total window = max(group_pre_onsets) + CONSONANT_POST_BASE_MS
             post-onset per group = total_window - group_pre_onset

IC set : brain ICs only
Feature: z_power_smooth
Bands  : all bands

Run order: classification_overt.py → this script → classification_overt_band_sweep.py

Usage
-----
    # Full sweep + auto-derived final experiments
    python classification_overt_W4_sweep.py \\
        --subj subj-02 \\
        --input-dir  /path/to/04_processed/ \\
        --output-dir /path/to/results/ \\
        --overt-brain-ics 4 7 14 21 22 32 \\
        --overt-bad-epochs 1 111 \\
        --speech-window-ms 500

    # Skip sweep, run Exp A + Exp B with manually specified values
    python classification_overt_W4_sweep.py \\
        --subj subj-02 ... \\
        --run-final-only \\
        --best-overall-pre-onset-ms 250 \\
        --consonant-pre-onset-ms 250 350 100

    # Parallel sweep steps
    python classification_overt_W4_sweep.py ... --n-jobs 4
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import joblib

from constants import (
    SYLLABLES, FS, W4_SWEEP_MS,
    CONSONANT_GROUPS, CONSONANT_POST_BASE_MS,
)
from utils import (
    LP_CUTOFF_HZ,
    load_analytic, compute_features, reject_epochs,
    compute_speech_window_tp, derive_epoch_indices,
    lowpass_smooth,
    build_X_speech_window,
    run_classifiers,
    plot_feature_importance,
    print_and_save_summary,
)


# ---------------------------------------------------------------------------
# Consonant group helpers
# ---------------------------------------------------------------------------

CONSONANT_GROUP_NAMES = list(CONSONANT_GROUPS.keys())  # ['stop', 'nasal', 'fricative']


def _group_mean_recall(result, group_labels_1idx, strategy='rf'):
    """
    Mean recall across syllables in a consonant group.

    Parameters
    ----------
    result           : ExperimentResult
    group_labels_1idx: list of int, 1-indexed class labels
    strategy         : 'rf' | 'combined' — which classifier(s) to use

    Returns
    -------
    float
    """
    recalls = []
    for lbl in group_labels_1idx:
        syl   = SYLLABLES[lbl - 1]
        rf_r  = result.rf_per_class.get(syl, 0)
        svm_r = result.svm_per_class.get(syl, 0)
        recalls.append(rf_r if strategy == 'rf' else (rf_r + svm_r) / 2)
    return float(np.mean(recalls))


def derive_consonant_pre_onsets(sweep_results, strategy='rf'):
    """
    Best pre-onset per consonant group.

    Parameters
    ----------
    sweep_results : list of (pre_ms, ExperimentResult)
    strategy      : 'rf' | 'combined'

    Returns
    -------
    dict: {group_name: best_pre_onset_ms}
    """
    pre_arr = np.array([r[0] for r in sweep_results])
    best    = {}
    for group_name, label_1idx in CONSONANT_GROUPS.items():
        group_recalls = np.array([
            _group_mean_recall(r, label_1idx, strategy) for _, r in sweep_results
        ])
        best[group_name] = int(pre_arr[np.argmax(group_recalls)])
    return best


def derive_best_overall_pre_onset(sweep_results, strategy='rf'):
    """
    Best pre-onset maximising mean recall across all classes.

    Parameters
    ----------
    sweep_results : list of (pre_ms, ExperimentResult)
    strategy      : 'rf' | 'combined'

    Returns
    -------
    int: best pre-onset in ms
    """
    pre_arr     = np.array([r[0] for r in sweep_results])
    mean_recall = np.array([
        np.mean([
            r.rf_per_class.get(syl, 0) if strategy == 'rf'
            else (r.rf_per_class.get(syl, 0) + r.svm_per_class.get(syl, 0)) / 2
            for syl in SYLLABLES
        ])
        for _, r in sweep_results
    ])
    return int(pre_arr[np.argmax(mean_recall)])


# ---------------------------------------------------------------------------
# W4 sweep summary plot
# ---------------------------------------------------------------------------

def plot_W4_sweep_summary(sweep_results, save_dir, subj, cond, speech_window_ms):
    """
    Two separate figures:
      Figure 1 — overall accuracy (SVM + RF) vs. pre-onset
      Figure 2 — per-class recall vs. pre-onset, 1×3 subplots by consonant group
                 (gi/gu | mi/mu | si/su), solid=RF dashed=SVM
 
    Parameters
    ----------
    sweep_results    : list of (pre_ms, ExperimentResult)
    save_dir         : str
    subj             : str
    cond             : str
    speech_window_ms : int
    """
    pre_vals = [r[0] for r in sweep_results]
    svm_accs = [r[1].svm_accuracy for r in sweep_results]
    rf_accs  = [r[1].rf_accuracy  for r in sweep_results]
    svm_bals = [r[1].svm_bal_acc  for r in sweep_results]
    rf_bals  = [r[1].rf_bal_acc   for r in sweep_results]
 
    title_base = (f'{subj} | W4 pre-speech sweep — brain ICs, z_power_smooth\n'
                  f'Window end fixed at onset + {speech_window_ms} ms')
 
    # -------------------------------------------------------------------
    # Figure 1 — overall accuracy
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pre_vals, svm_accs, 'o-',  label='SVM acc',     color='steelblue')
    ax.plot(pre_vals, svm_bals, 'o--', label='SVM bal acc', color='steelblue',  alpha=0.5)
    ax.plot(pre_vals, rf_accs,  's-',  label='RF acc',      color='darkorange')
    ax.plot(pre_vals, rf_bals,  's--', label='RF bal acc',  color='darkorange', alpha=0.5)
    ax.axhline(1/6, color='gray', linestyle=':', linewidth=0.8, label='Chance (1/6)')
    ax.set_xlabel('Pre-speech onset included (ms)')
    ax.set_ylabel('Cross-validated accuracy')
    ax.set_title('Overall accuracy vs. pre-speech onset')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1)
    fig.suptitle(title_base, fontsize=10)
    plt.tight_layout()
 
    fpath1 = os.path.join(save_dir, f'{subj}_{cond}_W4_sweep_accuracy.png')
    fig.savefig(fpath1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved sweep accuracy plot: {fpath1}')
 
    # -------------------------------------------------------------------
    # Figure 2 — per-class recall, 1×3 by consonant group
    # -------------------------------------------------------------------
    consonant_groups = [
        ('Velar stops',         ['gi', 'gu'], ['#1f77b4', '#aec7e8']),
        ('Bilabial nasals',     ['mi', 'mu'], ['#2ca02c', '#98df8a']),
        ('Alveolar fricatives', ['si', 'su'], ['#9467bd', '#c5b0d5']),
    ]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
 
    for ax, (group_title, syls, colors) in zip(axes, consonant_groups):
        for syl, color in zip(syls, colors):
            rf_recalls  = [r[1].rf_per_class.get(syl, 0)  for r in sweep_results]
            svm_recalls = [r[1].svm_per_class.get(syl, 0) for r in sweep_results]
            ax.plot(pre_vals, rf_recalls,  'o-',  label=f'{syl} RF',
                    color=color, linewidth=1.8)
            ax.plot(pre_vals, svm_recalls, 'o--', label=f'{syl} SVM',
                    color=color, alpha=0.55, linewidth=1.4)
        ax.axhline(1/6, color='gray', linestyle=':', linewidth=0.8, label='Chance')
        ax.set_xlabel('Pre-speech onset included (ms)')
        ax.set_title(group_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(0, 1)
 
    axes[0].set_ylabel('Per-class recall  (solid=RF, dashed=SVM)')
    fig.suptitle(title_base, fontsize=10)
    plt.tight_layout()
 
    fpath2 = os.path.join(save_dir, f'{subj}_{cond}_W4_sweep_recall_by_consonant.png')
    fig.savefig(fpath2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved sweep recall plot:    {fpath2}')


# ---------------------------------------------------------------------------
# Recommendation printer
# ---------------------------------------------------------------------------

def print_W4_recommendation(sweep_results,
                             best_overall_rf, best_overall_combined,
                             consonant_rf, consonant_combined):
    """
    Print both RF-only and combined recommendations side by side.

    Parameters
    ----------
    sweep_results         : list of (pre_ms, ExperimentResult)
    best_overall_rf       : int, best overall pre-onset via RF-only
    best_overall_combined : int, best overall pre-onset via combined
    consonant_rf          : dict {group_name: best_pre_onset_ms} via RF-only
    consonant_combined    : dict {group_name: best_pre_onset_ms} via combined
    """
    pre_arr = np.array([r[0] for r in sweep_results])
    rf_arr  = np.array([[r.rf_per_class.get(syl, 0)  for syl in SYLLABLES]
                         for _, r in sweep_results])
    svm_arr = np.array([[r.svm_per_class.get(syl, 0) for syl in SYLLABLES]
                         for _, r in sweep_results])
    avg_arr = (rf_arr + svm_arr) / 2

    sep = '=' * 90
    print(f'\n{sep}')
    print('  W4 RECOMMENDATION')
    print(f'{sep}')

    # Per-consonant-group
    print('\n  Per-consonant-group (mean recall across pair):')
    print(f'  {"Group":28s}  {"RF-only best":>14s}  {"RF recall":>10s}  '
          f'{"Combined best":>14s}  {"Avg recall":>10s}')
    print(f'  {"-"*80}')
    for group_name, label_1idx in CONSONANT_GROUPS.items():
        syls = [SYLLABLES[l - 1] for l in label_1idx]
        label = f'{group_name} ({"/".join(syls)})'

        rf_ms  = consonant_rf[group_name]
        comb_ms = consonant_combined[group_name]

        rf_idx   = np.where(pre_arr == rf_ms)[0][0]
        comb_idx = np.where(pre_arr == comb_ms)[0][0]

        rf_recall   = np.mean([rf_arr[rf_idx,    l - 1] for l in label_1idx])
        comb_recall = np.mean([avg_arr[comb_idx, l - 1] for l in label_1idx])

        print(f'  {label:28s}  {rf_ms:>12d} ms  {rf_recall:>10.3f}  '
              f'{comb_ms:>12d} ms  {comb_recall:>10.3f}')

    # Per-syllable
    print('\n  Per-syllable:')
    print(f'  {"Syl":6s}  {"RF best":>10s}  {"RF recall":>10s}  '
          f'{"SVM best":>10s}  {"SVM recall":>10s}  {"Avg best":>10s}  {"Avg recall":>10s}')
    print(f'  {"-"*72}')
    for si, syl in enumerate(SYLLABLES):
        rf_best_idx  = np.argmax(rf_arr[:, si])
        svm_best_idx = np.argmax(svm_arr[:, si])
        avg_best_idx = np.argmax(avg_arr[:, si])
        print(f'  {syl:6s}  '
              f'{pre_arr[rf_best_idx]:>8d} ms  {rf_arr[rf_best_idx, si]:>10.3f}  '
              f'{pre_arr[svm_best_idx]:>8d} ms  {svm_arr[svm_best_idx, si]:>10.3f}  '
              f'{pre_arr[avg_best_idx]:>8d} ms  {avg_arr[avg_best_idx, si]:>10.3f}')

    # Overall
    rf_mean   = rf_arr.mean(axis=1)
    comb_mean = avg_arr.mean(axis=1)
    rf_overall_idx   = np.where(pre_arr == best_overall_rf)[0][0]
    comb_overall_idx = np.where(pre_arr == best_overall_combined)[0][0]
    print(f'\n  Best overall:')
    print(f'    RF-only   → {best_overall_rf} ms  '
          f'(mean RF recall = {rf_mean[rf_overall_idx]:.3f})')
    print(f'    Combined  → {best_overall_combined} ms  '
          f'(mean avg recall = {comb_mean[comb_overall_idx]:.3f})')

    # Exp B window summary
    for label, consonant_dict in [('RF-only', consonant_rf),
                                   ('Combined', consonant_combined)]:
        total_win = max(consonant_dict.values()) + CONSONANT_POST_BASE_MS
        print(f'\n  Exp B windows ({label}, total = {total_win} ms):')
        for group_name, label_1idx in CONSONANT_GROUPS.items():
            syls    = [SYLLABLES[l - 1] for l in label_1idx]
            pre_ms  = consonant_dict[group_name]
            post_ms = total_win - pre_ms
            print(f'    {group_name:12s} ({"/".join(syls)}):  '
                  f'pre={pre_ms} ms  post={post_ms} ms')

    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Single final experiment (A or B)
# ---------------------------------------------------------------------------

def run_exp_A(z_power_smooth, y, onset_tps,
              brain_ics_0idx, ic_labels_brain,
              speech_window_tp, speech_window_ms,
              best_overall_ms, strategy_tag,
              save_dir, subj, cond_code, inner_jobs):
    """
    Exp A — uniform best overall pre-onset for all trials.

    Parameters
    ----------
    strategy_tag : str, 'RF' or 'combined' — used in experiment name
    """
    pre_tp   = int(best_overall_ms / 1000 * FS)
    win_tp   = pre_tp + speech_window_tp
    exp_name = (f'W4_ExpA_{strategy_tag}_brainIC_zpower_'
                f'pre{best_overall_ms}ms_speech{speech_window_ms}ms')
    t_vec    = np.linspace(-best_overall_ms, speech_window_ms, win_tp)

    print(f'\n{"="*60}')
    print(f'  Exp A_{strategy_tag} — uniform pre-onset: {best_overall_ms} ms')
    print(f'  Window: onset−{best_overall_ms}ms → onset+{speech_window_ms}ms '
          f'({win_tp} samples)')
    print(f'{"="*60}')

    X = build_X_speech_window(
        z_power_smooth, brain_ics_0idx, onset_tps,
        pre_onset_tp=pre_tp, post_onset_tp=speech_window_tp)

    r, rf_model = run_classifiers(
        exp_name, X, y, save_dir, subj, cond_code,
        ic_set='brain', band_set='all_bands',
        feature='z_power_smooth',
        window=f'onset-{best_overall_ms}ms_to_onset+{speech_window_ms}ms',
        inner_n_jobs=inner_jobs)

    plot_feature_importance(
        rf_model, len(brain_ics_0idx), z_power_smooth.shape[1], win_tp,
        ic_labels_brain, t_vec,
        save_dir, subj, cond_code, exp_name)

    return r


def run_exp_B(z_power_smooth, y, onset_tps,
              brain_ics_0idx, ic_labels_brain,
              consonant_pre_onsets_ms, strategy_tag,
              save_dir, subj, cond_code, inner_jobs):
    """
    Exp B — consonant-group-specific pre-onset.

    Parameters
    ----------
    consonant_pre_onsets_ms : dict {group_name: pre_onset_ms}
    strategy_tag            : str, 'RF' or 'combined'
    """
    nBands      = z_power_smooth.shape[1]
    nICs        = len(brain_ics_0idx)
    nTrials     = z_power_smooth.shape[-1]
    total_T     = z_power_smooth.shape[2]
    bands       = list(range(nBands))

    max_pre_ms   = max(consonant_pre_onsets_ms.values())
    total_win_ms = max_pre_ms + CONSONANT_POST_BASE_MS
    total_win_tp = int(total_win_ms / 1000 * FS)
    # t=0 is the start of the window (earliest articulatory prep onset across groups)
    # t=total_win_ms is the end of the window
    # acoustic onset occurs at different positions per group within this window
    t_vec        = np.linspace(0, total_win_ms, total_win_tp)

    print(f'\n{"="*60}')
    print(f'  Exp B_{strategy_tag} — consonant-specific pre-onset')
    print(f'  Total window: {total_win_ms} ms ({total_win_tp} samples)')
    for group_name, pre_ms in consonant_pre_onsets_ms.items():
        post_ms = total_win_ms - pre_ms
        syls    = [SYLLABLES[l - 1] for l in CONSONANT_GROUPS[group_name]]
        print(f'    {group_name:12s} ({"/".join(syls)}):  '
              f'pre={pre_ms} ms  post={post_ms} ms')
    print(f'{"="*60}')

    # Build per-trial feature matrix
    X_b = np.zeros((nICs, nBands, total_win_tp, nTrials))

    for group_name, label_1idx in CONSONANT_GROUPS.items():
        pre_ms  = consonant_pre_onsets_ms[group_name]
        post_ms = total_win_ms - pre_ms
        pre_tp  = int(pre_ms  / 1000 * FS)
        post_tp = int(post_ms / 1000 * FS)

        trial_mask = np.isin(y, label_1idx)

        for i in np.where(trial_mask)[0]:
            start = int(onset_tps[i]) - pre_tp
            end   = int(onset_tps[i]) + post_tp

            if start < 0 or end > total_T:
                print(f'    Trial {i} ({group_name}): '
                      f'window [{start},{end}] out of bounds — zero-filled')
                continue

            X_b[:, :, :, i] = z_power_smooth[
                np.ix_(brain_ics_0idx, bands, list(range(start, end)), [i])
            ][:, :, :, 0]

    X_b = X_b.transpose(3, 0, 1, 2).reshape(nTrials, -1)

    groups_str = '_'.join([f'{g[:3]}{v}ms'
                           for g, v in consonant_pre_onsets_ms.items()])
    exp_name = f'W4_ExpB_{strategy_tag}_brainIC_zpower_{groups_str}'

    print(f'\n  [{exp_name}]  X={X_b.shape}')

    r, rf_model = run_classifiers(
        exp_name, X_b, y, save_dir, subj, cond_code,
        ic_set='brain', band_set='all_bands',
        feature='z_power_smooth',
        window=f'consonant_specific_{strategy_tag}_total{total_win_ms}ms',
        inner_n_jobs=inner_jobs)

    plot_feature_importance(
        rf_model, nICs, nBands, total_win_tp,
        ic_labels_brain, t_vec,
        save_dir, subj, cond_code, exp_name)

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EEG W4 pre-speech onset sweep — overt condition')

    parser.add_argument(
        '--subj', required=True, type=str,
        help='Subject ID, e.g. subj-02')
    parser.add_argument(
        '--input-dir', required=True, type=str,
        help='Directory containing analytic .mat files')
    parser.add_argument(
        '--output-dir', required=True, type=str,
        help='Output directory for figures and CSVs')
    parser.add_argument(
        '--overt-brain-ics', required=True, type=int, nargs='+',
        help='1-indexed brain ICs for overt condition')
    parser.add_argument(
        '--overt-bad-epochs', default=[], type=int, nargs='*',
        help='1-indexed bad epochs to reject')
    parser.add_argument(
        '--speech-window-ms', default=None, type=int,
        help='Post-onset speech window (ms). Should match classification_overt.py. '
             'Default: auto-derived from mean(offset − onset).')
    parser.add_argument(
        '--n-jobs', default=1, type=int,
        help='n_jobs=1 → sequential, GridSearchCV uses all cores. '
             'n_jobs>1 → parallel sweep steps, GridSearchCV single-threaded.')
    parser.add_argument(
        '--run-final-only', action='store_true', default=False,
        help='Skip sweep. Run Exp A + Exp B with manually specified values '
             '(treated as both RF and combined since you are picking manually).')
    parser.add_argument(
        '--best-overall-pre-onset-ms', default=None, type=int,
        help='[--run-final-only] Uniform pre-onset for Exp A (ms).')
    parser.add_argument(
        '--consonant-pre-onset-ms', default=None, type=int, nargs=3,
        metavar=('STOP_MS', 'NASAL_MS', 'FRICATIVE_MS'),
        help='[--run-final-only] Pre-onset per consonant group for Exp B (ms), '
             'ordered: stop (gi/gu), nasal (mi/mu), fricative (si/su).')

    args = parser.parse_args()

    if args.run_final_only:
        if args.best_overall_pre_onset_ms is None:
            parser.error('--best-overall-pre-onset-ms is required with --run-final-only')
        if args.consonant_pre_onset_ms is None:
            parser.error('--consonant-pre-onset-ms is required with --run-final-only')

    subj       = args.subj
    cond_code  = 'sp'
    cond_label = 'spoken'
    inner_jobs = -1 if args.n_jobs == 1 else 1

    brain_ics_1idx  = args.overt_brain_ics
    brain_ics_0idx  = [ic - 1 for ic in brain_ics_1idx]
    bad_epochs      = args.overt_bad_epochs
    ic_labels_brain = [f'IC{ic}' for ic in brain_ics_1idx]

    save_dir = os.path.join(args.output_dir, subj, cond_label, 'W4_sweep')
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load and preprocess
    # -------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f'  Subject:   {subj}')
    print(f'  Condition: {cond_label} ({cond_code})')
    print(f'  Brain ICs: {brain_ics_1idx} ({len(brain_ics_1idx)} total)')
    print(f'  Bad epochs: {bad_epochs if bad_epochs else "None"}')
    if args.run_final_only:
        print('  Mode: final experiments only (sweep skipped)')
    print(f'{"="*60}')

    data_path  = os.path.join(args.input_dir, f'{subj}_{cond_code}_eeg_analytic.mat')
    onset_path = os.path.join(args.input_dir, f'{subj}_speech_onset_offset.mat')

    print('[1/4] Loading analytic signal...')
    Z, labels, times = load_analytic(data_path)
    print(f'  Shape: {Z.shape}')

    idx_0ms, _ = derive_epoch_indices(times)

    print('[2/4] Computing features...')
    _, _, _, z_power = compute_features(Z, times)

    speech_data  = loadmat(onset_path)
    onset_times  = speech_data['onset_latencies'].squeeze()
    offset_times = speech_data['offset_latencies'].squeeze()

    if len(bad_epochs) > 0:
        print(f'[3/4] Rejecting {len(bad_epochs)} bad epoch(s): {bad_epochs}')
        good_mask, (z_power, labels, onset_times, offset_times) = \
            reject_epochs(bad_epochs, z_power, labels, onset_times, offset_times)
        print(f'  Remaining trials: {good_mask.sum()}')
    else:
        print('[3/4] Skipping bad epoch rejection - no bad epochs specified')
        good_mask = np.ones(z_power.shape[-1], dtype=bool)
        print(f'  Keeping all {good_mask.sum()} trials')

    print(f'[4/4] Smoothing (LP {LP_CUTOFF_HZ} Hz)...')
    z_power_smooth = lowpass_smooth(z_power)

    onset_tps  = np.rint(onset_times  / 1000 * FS + idx_0ms).astype(int)
    offset_tps = np.rint(offset_times / 1000 * FS + idx_0ms).astype(int)

    invalid = np.where(offset_tps < onset_tps)[0]
    if len(invalid) > 0:
        print(f'  WARNING: {len(invalid)} trials have offset < onset — check data')

    if args.speech_window_ms is not None:
        speech_window_tp = int(args.speech_window_ms / 1000 * FS)
        speech_window_ms = args.speech_window_ms
        print(f'  Speech window (from arg): {speech_window_ms} ms = {speech_window_tp} samples')
    else:
        speech_window_tp = compute_speech_window_tp(onset_tps, offset_tps)
        speech_window_ms = int(speech_window_tp / FS * 1000)

    y = np.array(labels).astype(int)

    # -------------------------------------------------------------------
    # W4 sweep
    # -------------------------------------------------------------------
    sweep_results = []

    if not args.run_final_only:
        print(f'\n{"="*60}')
        print(f'  W4 sweep — pre-onset values: {W4_SWEEP_MS} ms')
        print(f'  Window end fixed at onset + {speech_window_ms} ms')
        print(f'{"="*60}')

        def _sweep_one(pre_ms):
            pre_tp   = int(pre_ms / 1000 * FS)
            win_tp   = pre_tp + speech_window_tp
            exp_name = f'W4_brainIC_zpower_pre{pre_ms}ms_speech{speech_window_ms}ms'
            t_vec    = np.linspace(-pre_ms, speech_window_ms, win_tp)

            print(f'\n  [W4 pre={pre_ms}ms]  '
                  f'total window={win_tp} samples ({pre_ms + speech_window_ms} ms)')
            X = build_X_speech_window(
                z_power_smooth, brain_ics_0idx, onset_tps,
                pre_onset_tp=pre_tp, post_onset_tp=speech_window_tp)

            r, rf_model = run_classifiers(
                exp_name, X, y, save_dir, subj, cond_code,
                ic_set='brain', band_set='all_bands',
                feature='z_power_smooth',
                window=f'onset-{pre_ms}ms_to_onset+{speech_window_ms}ms',
                inner_n_jobs=inner_jobs)

            plot_feature_importance(
                rf_model,
                len(brain_ics_0idx), z_power_smooth.shape[1], win_tp,
                ic_labels_brain, t_vec,
                save_dir, subj, cond_code, exp_name)

            return pre_ms, r

        if args.n_jobs == 1:
            sweep_results = [_sweep_one(pre_ms) for pre_ms in W4_SWEEP_MS]
        else:
            sweep_results = joblib.Parallel(n_jobs=args.n_jobs, prefer='loky')(
                joblib.delayed(_sweep_one)(pre_ms) for pre_ms in W4_SWEEP_MS)

        plot_W4_sweep_summary(
            sweep_results, save_dir, subj, cond_code, speech_window_ms)

        # Derive both recommendation strategies
        best_overall_rf       = derive_best_overall_pre_onset(sweep_results, 'rf')
        best_overall_combined = derive_best_overall_pre_onset(sweep_results, 'combined')
        consonant_rf          = derive_consonant_pre_onsets(sweep_results, 'rf')
        consonant_combined    = derive_consonant_pre_onsets(sweep_results, 'combined')

        # Save sweep table with SVM + RF per-class recall
        sweep_rows = []
        for pre_ms, r in sweep_results:
            row = {'pre_onset_ms': pre_ms,
                   'SVM_acc':      f'{r.svm_accuracy:.3f}',
                   'SVM_bal_acc':  f'{r.svm_bal_acc:.3f}',
                   'RF_acc':       f'{r.rf_accuracy:.3f}',
                   'RF_bal_acc':   f'{r.rf_bal_acc:.3f}'}
            row.update({f'SVM_recall_{syl}': f"{r.svm_per_class.get(syl, 0):.3f}"
                        for syl in SYLLABLES})
            row.update({f'RF_recall_{syl}':  f"{r.rf_per_class.get(syl, 0):.3f}"
                        for syl in SYLLABLES})
            sweep_rows.append(row)

        df_sweep  = pd.DataFrame(sweep_rows)
        csv_sweep = os.path.join(save_dir, f'{subj}_{cond_code}_W4_sweep.csv')
        df_sweep.to_csv(csv_sweep, index=False)
        print(f'\n  Sweep table saved: {csv_sweep}')
        print(df_sweep.to_string(index=False))

        print_and_save_summary(
            [r for _, r in sweep_results],
            save_dir, subj, cond_code, tag='W4_sweep')

    # -------------------------------------------------------------------
    # Print recommendation
    # -------------------------------------------------------------------
    if sweep_results:
        print_W4_recommendation(
            sweep_results,
            best_overall_rf, best_overall_combined,
            consonant_rf, consonant_combined)

    # -------------------------------------------------------------------
    # Four final experiments
    # -------------------------------------------------------------------
    print(f'\n{"="*60}')
    print('  Running four final experiments')
    print(f'{"="*60}')

    final_results = []

    if not args.run_final_only:

        final_results.append(run_exp_A(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            speech_window_tp, speech_window_ms,
            best_overall_rf, 'RF',
            save_dir, subj, cond_code, inner_jobs))

        final_results.append(run_exp_A(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            speech_window_tp, speech_window_ms,
            best_overall_combined, 'combined',
            save_dir, subj, cond_code, inner_jobs))

        final_results.append(run_exp_B(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            consonant_rf, 'RF',
            save_dir, subj, cond_code, inner_jobs))

        final_results.append(run_exp_B(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            consonant_combined, 'combined',
            save_dir, subj, cond_code, inner_jobs))

    else:
        best_overall = args.best_overall_pre_onset_ms
        best_consonant = {
            name: args.consonant_pre_onset_ms[i]
            for i, name in enumerate(CONSONANT_GROUP_NAMES)
        }

        final_results.append(run_exp_A(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            speech_window_tp, speech_window_ms,
            best_overall, 'manual',
            save_dir, subj, cond_code, inner_jobs))

        final_results.append(run_exp_B(
            z_power_smooth, y, onset_tps,
            brain_ics_0idx, ic_labels_brain,
            best_consonant, 'manual',
            save_dir, subj, cond_code, inner_jobs))

    print_and_save_summary(
        final_results, save_dir, subj, cond_code, tag='W4_final')


if __name__ == '__main__':
    main()