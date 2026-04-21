#!/usr/bin/env python3
"""
classification_overt_band_sweep.py — Band sweep for overt condition

RQ: Which frequency bands carry the most discriminative information
for syllable classification during overt speech?

Runs the band sweep under four window definitions:

  Sweep A — Acoustic speech period (mirrors W2):
    Window: [onset, onset + speech_window_ms]
    t_vec : 0 to +speech_window_ms

  Sweep B — Pre-acoustic speech (mirrors W3):
    Window: [onset - W3_PRE_ONSET_MS, onset]
    t_vec : -W3_PRE_ONSET_MS to 0

  Sweep C — Uniform pre-onset (--best-overall-pre-onset-ms):
    Window: [onset - X ms, onset + speech_window_ms] for all trials
    t_vec : -X ms to +speech_window_ms

  Sweep D — Consonant-group-specific pre-onset (--consonant-pre-onset-ms):
    Each consonant group uses its own pre-onset.
    Total window = max(group_pre_onsets) + CONSONANT_POST_BASE_MS (fixed).
    Post-onset per group = total_window - group_pre_onset.
    t_vec : 0 to total_win_ms (t=0 = start of earliest articulatory prep window)

IC set : defined keep ICs only
Feature: z_power_smooth

Band sets (9 total, defined in constants.py: BAND_SETS):
  all_bands, theta, alpha, beta, gamma, high_gamma, mu, wide_gamma, beta_wide_gamma

Run AFTER classification_overt_W4_sweep.py.
Use its recommended pre-onset values as arguments.

Usage
-----
    python classification_overt_band_sweep.py \\
        --subj subj-02 \\
        --input-dir  /path/to/04_processed/ \\
        --output-dir /path/to/results/ \\
        --overt-keep-ics ... \\
        --overt-bad-epochs ... \\
        --best-overall-pre-onset-ms 250 \\
        --consonant-pre-onset-ms 250 350 100 \\
        --speech-window-ms 500

    # Parallel band experiments
    python classification_overt_band_sweep.py ... --n-jobs 4
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import joblib

from constants import (
    SYLLABLES, FS, BAND_NAMES, BAND_SETS,
    CONSONANT_GROUPS, CONSONANT_POST_BASE_MS,
    W3_PRE_ONSET_MS,
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

CONSONANT_GROUP_NAMES = list(CONSONANT_GROUPS.keys())


# ---------------------------------------------------------------------------
# Consonant-specific X builder with band selection
# ---------------------------------------------------------------------------

def build_X_consonant_window(z_power_smooth, y, onset_tps,
                            keep_ics_0idx, consonant_pre_onsets_ms,
                            band_idx=None):
    """
    Build feature matrix using consonant-group-specific windows.

    Each trial's window: [onset - group_pre_onset, onset + group_post_onset]
    Total window length is fixed: max(pre_onsets) + CONSONANT_POST_BASE_MS

    Parameters
    ----------
    z_power_smooth          : np.ndarray [ICs x bands x time x trials]
    y                       : np.ndarray [trials], 1-indexed integer labels
    onset_tps               : np.ndarray [trials]
    keep_ics_0idx                : list of int
    consonant_pre_onsets_ms : dict {group_name: pre_onset_ms}
    band_idx                : list of int or None (None = all bands)

    Returns
    -------
    X            : np.ndarray [trials x features]
    total_win_ms : int, total window length in ms
    total_win_tp : int, total window length in samples
    """
    nBands_all  = z_power_smooth.shape[1]
    bands       = band_idx if band_idx is not None else list(range(nBands_all))
    nBands      = len(bands)
    nICs        = len(keep_ics_0idx)
    nTrials     = z_power_smooth.shape[-1]
    total_T     = z_power_smooth.shape[2]

    max_pre_ms   = max(consonant_pre_onsets_ms.values())
    total_win_ms = max_pre_ms + CONSONANT_POST_BASE_MS
    total_win_tp = int(total_win_ms / 1000 * FS)

    X_out = np.zeros((nICs, nBands, total_win_tp, nTrials))

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
                print(f'  Trial {i} ({group_name}): '
                      f'window [{start},{end}] out of bounds — zero-filled')
                continue

            X_out[:, :, :, i] = z_power_smooth[
                np.ix_(keep_ics_0idx, bands, list(range(start, end)), [i])
            ][:, :, :, 0]

    X_out = X_out.transpose(3, 0, 1, 2)
    return X_out.reshape(nTrials, -1), total_win_ms, total_win_tp


# ---------------------------------------------------------------------------
# Band sweep summary plot
# ---------------------------------------------------------------------------

def plot_band_sweep_summary(results, band_labels, save_dir, subj, cond,
                             win_str, tag):
    svm_accs = [r.svm_accuracy for r in results]
    rf_accs  = [r.rf_accuracy  for r in results]
    svm_bals = [r.svm_bal_acc  for r in results]
    rf_bals  = [r.rf_bal_acc   for r in results]

    rf_recall   = np.array([[r.rf_per_class.get(syl, 0)  for syl in SYLLABLES] for r in results])
    svm_recall  = np.array([[r.svm_per_class.get(syl, 0) for syl in SYLLABLES] for r in results])
    mean_recall = (rf_recall + svm_recall) / 2

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax_acc  = axes[0, 0]
    ax_rf   = axes[0, 1]
    ax_svm  = axes[1, 0]
    ax_mean = axes[1, 1]

    # Accuracy bar chart
    x = np.arange(len(band_labels))
    w = 0.2
    ax_acc.bar(x - 1.5*w, svm_accs, w, label='SVM acc',     color='steelblue',  alpha=0.9)
    ax_acc.bar(x - 0.5*w, svm_bals, w, label='SVM bal acc', color='steelblue',  alpha=0.45)
    ax_acc.bar(x + 0.5*w, rf_accs,  w, label='RF acc',      color='darkorange', alpha=0.9)
    ax_acc.bar(x + 1.5*w, rf_bals,  w, label='RF bal acc',  color='darkorange', alpha=0.45)
    ax_acc.axhline(1/6, color='gray', linestyle=':', linewidth=0.8, label='Chance (1/6)')
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(band_labels, rotation=35, ha='right', fontsize=9)
    ax_acc.set_ylabel('Cross-validated accuracy')
    ax_acc.set_title('Accuracy by band set')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis='y', alpha=0.4)
    ax_acc.set_ylim(0, 1)

    def _heatmap(ax, matrix, title):
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(SYLLABLES)))
        ax.set_xticklabels(SYLLABLES, fontsize=10)
        ax.set_yticks(range(len(band_labels)))
        ax.set_yticklabels(band_labels, fontsize=9)
        ax.set_xlabel('Syllable')
        ax.set_ylabel('Band set')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall')
        for i in range(len(band_labels)):
            for j in range(len(SYLLABLES)):
                val   = matrix[i, j]
                color = 'white' if val < 0.25 or val > 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    _heatmap(ax_rf,   rf_recall,   'RF per-class recall')
    _heatmap(ax_svm,  svm_recall,  'SVM per-class recall')
    _heatmap(ax_mean, mean_recall, 'Mean(RF+SVM) per-class recall')

    fig.suptitle(
        f'{subj} | Band sweep ({tag}) — keep ICs, z_power_smooth\n'
        f'Window: {win_str}',
        fontsize=12)
    plt.tight_layout()

    fpath = os.path.join(save_dir, f'{subj}_{cond}_band_sweep_{tag}_summary.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved band sweep summary: {fpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EEG band sweep classification — overt condition')

    parser.add_argument(
        '--subj', required=True, type=str,
        help='Subject ID, e.g. subj-02')
    parser.add_argument(
        '--input-dir', required=True, type=str,
        help='Directory containing analytic .mat files')
    parser.add_argument(
        '--output-dir', required=True, type=str,
        help='Output root directory for figures and CSVs')
    parser.add_argument(
        '--overt-keep-ics', required=True, type=int, nargs='+',
        help='1-indexed keep ICs for overt condition')
    parser.add_argument(
        '--overt-bad-epochs', default=[], type=int, nargs='*',
        help='1-indexed bad epochs to reject')
    parser.add_argument(
        '--best-overall-pre-onset-ms', required=True, type=int,
        help='Uniform pre-onset for Sweep C (ms). '
             'Use W4 best overall recommendation.')
    parser.add_argument(
        '--consonant-pre-onset-ms', required=True, type=int, nargs=3,
        metavar=('STOP_MS', 'NASAL_MS', 'FRICATIVE_MS'),
        help='Pre-onset per consonant group for Sweep D (ms), '
             'ordered: stop (gi/gu), nasal (mi/mu), fricative (si/su). '
             'Use W4 consonant-group recommendation.')
    parser.add_argument(
        '--speech-window-ms', default=None, type=int,
        help='Post-onset speech window (ms). Should match W4 sweep script. '
             'Default: auto-derived from mean(offset - onset).')
    parser.add_argument(
        '--n-jobs', default=1, type=int,
        help='n_jobs=1 → sequential, GridSearchCV uses all cores. '
             'n_jobs>1 → parallel band experiments, GridSearchCV single-threaded.')

    parser.add_argument(
        '--fix-svm-C', type=float, default=None,
        help='Fixed SVM C (from W1a). Skips gridsearch when all fix-* params provided.')
    parser.add_argument(
        '--fix-rf-depth', type=str, default=None,
        help='Fixed RF max_depth (int or "None" for no limit).')
    parser.add_argument(
        '--fix-rf-features', type=str, default=None,
        help='Fixed RF max_features ("sqrt" or "log2").')
    parser.add_argument(
        '--fix-rf-estimators', type=int, default=None,
        help='Fixed RF n_estimators.')
    parser.add_argument(
        '--fix-rf-split', type=int, default=None,
        help='Fixed RF min_samples_split.')

    args = parser.parse_args()

    subj       = args.subj
    cond_code  = 'sp'
    cond_label = 'spoken'
    inner_jobs = -1 if args.n_jobs == 1 else 1

    fix_svm_C         = args.fix_svm_C
    fix_rf_depth      = (None if args.fix_rf_depth in (None, 'None')
                         else int(args.fix_rf_depth))
    fix_rf_features   = args.fix_rf_features
    fix_rf_estimators = args.fix_rf_estimators
    fix_rf_split      = args.fix_rf_split

    best_overall_ms = args.best_overall_pre_onset_ms
    consonant_pre_onsets_ms = {
        name: args.consonant_pre_onset_ms[i]
        for i, name in enumerate(CONSONANT_GROUP_NAMES)
    }

    keep_ics_1idx  = args.overt_keep_ics
    keep_ics_0idx  = [ic - 1 for ic in keep_ics_1idx]
    bad_epochs      = args.overt_bad_epochs
    ic_labels = [f'IC{ic}' for ic in keep_ics_1idx]

    save_dir = os.path.join(args.output_dir, subj, cond_label, 'band_sweep')
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Load and preprocess
    # -------------------------------------------------------------------
    print(f'\n{"="*60}')
    print(f'  Subject:        {subj}')
    print(f'  Condition:      {cond_label} ({cond_code})')
    print(f'  Keep ICs:      {keep_ics_1idx} ({len(keep_ics_1idx)} total)')
    print(f'  Bad epochs:     {bad_epochs if bad_epochs else "None"}')
    print(f'  Sweep C pre-onset (uniform):      {best_overall_ms} ms')
    for name, ms in consonant_pre_onsets_ms.items():
        print(f'  Sweep D pre-onset ({name}): {ms} ms')
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

    y           = np.array(labels).astype(int)
    band_items  = list(BAND_SETS.items())
    band_labels = list(BAND_SETS.keys())

    figure_save_dir = os.path.join(args.output_dir, subj, cond_label, 'summary_figures')
    os.makedirs(figure_save_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Sweep A — acoustic speech period (mirrors W2: onset → onset + N ms)
    # -------------------------------------------------------------------
    t_vec_a   = np.linspace(0, speech_window_ms, speech_window_tp)
    win_str_a = f'onset → onset+{speech_window_ms}ms'

    print(f'\n{"="*60}')
    print(f'  Sweep A — acoustic speech period: {win_str_a}')
    print(f'  ({speech_window_tp} samples per trial)')
    print(f'{"="*60}')

    def _band_job_A(band_name, band_idx):
        nBands_b = len(band_idx)
        exp_name = (f'BandA_{band_name}_keepIC_zpower_'
                    f'speech{speech_window_ms}ms')

        print(f'\n  [A | {band_name}]  '
              f'nICs={len(keep_ics_0idx)} nBands={nBands_b} nTime={speech_window_tp}')

        X = build_X_speech_window(
            z_power_smooth, keep_ics_0idx, onset_tps,
            pre_onset_tp=0,
            post_onset_tp=speech_window_tp,
            band_idx=band_idx)

        r, rf_model = run_classifiers(
            exp_name, X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set=band_name,
            feature='z_power_smooth',
            window=f'onset_to_onset+{speech_window_ms}ms',
            inner_n_jobs=inner_jobs,
            fix_svm_C=fix_svm_C, fix_rf_depth=fix_rf_depth,
            fix_rf_features=fix_rf_features, fix_rf_estimators=fix_rf_estimators,
            fix_rf_split=fix_rf_split)

        plot_feature_importance(
            rf_model, len(keep_ics_0idx), nBands_b, speech_window_tp,
            ic_labels, t_vec_a,
            save_dir, subj, cond_code, exp_name,
            band_idx_vec=band_idx)

        return r

    if args.n_jobs == 1:
        results_A = [_band_job_A(name, idx) for name, idx in band_items]
    else:
        results_A = joblib.Parallel(n_jobs=args.n_jobs, prefer='processes')(
            joblib.delayed(_band_job_A)(name, idx) for name, idx in band_items)

    plot_band_sweep_summary(
        results_A, band_labels, figure_save_dir, subj, cond_code,
        win_str=win_str_a, tag='speech')

    print_and_save_summary(
        results_A, save_dir, subj, cond_code,
        tag=f'band_sweep_speech{speech_window_ms}ms')

    # -------------------------------------------------------------------
    # Sweep B — pre-acoustic speech (mirrors W3: onset - W3_PRE_ONSET_MS → onset)
    # -------------------------------------------------------------------
    pre_onset_tp_b = int(W3_PRE_ONSET_MS / 1000 * FS)
    t_vec_b        = np.linspace(-W3_PRE_ONSET_MS, 0, pre_onset_tp_b)
    win_str_b      = f'onset-{W3_PRE_ONSET_MS}ms → onset'

    print(f'\n{"="*60}')
    print(f'  Sweep B — pre-acoustic speech: {win_str_b}')
    print(f'  ({pre_onset_tp_b} samples per trial)')
    print(f'{"="*60}')

    def _band_job_B(band_name, band_idx):
        nBands_b = len(band_idx)
        exp_name = (f'BandB_{band_name}_keepIC_zpower_'
                    f'pre{W3_PRE_ONSET_MS}ms')

        print(f'\n  [B | {band_name}]  '
              f'nICs={len(keep_ics_0idx)} nBands={nBands_b} nTime={pre_onset_tp_b}')

        X = build_X_speech_window(
            z_power_smooth, keep_ics_0idx, onset_tps,
            pre_onset_tp=pre_onset_tp_b,
            post_onset_tp=0,
            band_idx=band_idx)

        r, rf_model = run_classifiers(
            exp_name, X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set=band_name,
            feature='z_power_smooth',
            window=f'onset-{W3_PRE_ONSET_MS}ms_to_onset',
            inner_n_jobs=inner_jobs,
            fix_svm_C=fix_svm_C, fix_rf_depth=fix_rf_depth,
            fix_rf_features=fix_rf_features, fix_rf_estimators=fix_rf_estimators,
            fix_rf_split=fix_rf_split)

        plot_feature_importance(
            rf_model, len(keep_ics_0idx), nBands_b, pre_onset_tp_b,
            ic_labels, t_vec_b,
            save_dir, subj, cond_code, exp_name,
            band_idx_vec=band_idx)

        return r

    if args.n_jobs == 1:
        results_B = [_band_job_B(name, idx) for name, idx in band_items]
    else:
        results_B = joblib.Parallel(n_jobs=args.n_jobs, prefer='processes')(
            joblib.delayed(_band_job_B)(name, idx) for name, idx in band_items)

    plot_band_sweep_summary(
        results_B, band_labels, figure_save_dir, subj, cond_code,
        win_str=win_str_b, tag='prespeech')

    print_and_save_summary(
        results_B, save_dir, subj, cond_code,
        tag=f'band_sweep_prespeech{W3_PRE_ONSET_MS}ms')

    # -------------------------------------------------------------------
    # Sweep C — uniform pre-onset window
    # -------------------------------------------------------------------
    pre_onset_tp_c = int(best_overall_ms / 1000 * FS)
    win_tp_c       = pre_onset_tp_c + speech_window_tp
    t_vec_c        = np.linspace(-best_overall_ms, speech_window_ms, win_tp_c)
    win_str_c      = (f'onset-{best_overall_ms}ms → onset+{speech_window_ms}ms'
                      if best_overall_ms > 0 else
                      f'onset → onset+{speech_window_ms}ms')

    print(f'\n{"="*60}')
    print(f'  Sweep C — uniform window: {win_str_c}')
    print(f'  ({win_tp_c} samples per trial)')
    print(f'{"="*60}')

    def _band_job_C(band_name, band_idx):
        nBands_b = len(band_idx)
        exp_name = (f'BandC_{band_name}_keepIC_zpower_'
                    f'pre{best_overall_ms}ms_speech{speech_window_ms}ms')

        print(f'\n  [C | {band_name}]  '
              f'nICs={len(keep_ics_0idx)} nBands={nBands_b} nTime={win_tp_c}')

        X = build_X_speech_window(
            z_power_smooth, keep_ics_0idx, onset_tps,
            pre_onset_tp=pre_onset_tp_c,
            post_onset_tp=speech_window_tp,
            band_idx=band_idx)

        r, rf_model = run_classifiers(
            exp_name, X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set=band_name,
            feature='z_power_smooth',
            window=f'uniform_onset-{best_overall_ms}ms_to_onset+{speech_window_ms}ms',
            inner_n_jobs=inner_jobs,
            fix_svm_C=fix_svm_C, fix_rf_depth=fix_rf_depth,
            fix_rf_features=fix_rf_features, fix_rf_estimators=fix_rf_estimators,
            fix_rf_split=fix_rf_split)

        plot_feature_importance(
            rf_model, len(keep_ics_0idx), nBands_b, win_tp_c,
            ic_labels, t_vec_c,
            save_dir, subj, cond_code, exp_name,
            band_idx_vec=band_idx)

        return r

    if args.n_jobs == 1:
        results_C = [_band_job_C(name, idx) for name, idx in band_items]
    else:
        results_C = joblib.Parallel(n_jobs=args.n_jobs, prefer='processes')(
            joblib.delayed(_band_job_C)(name, idx) for name, idx in band_items)

    plot_band_sweep_summary(
        results_C, band_labels, figure_save_dir, subj, cond_code,
        win_str=win_str_c, tag='uniform')

    print_and_save_summary(
        results_C, save_dir, subj, cond_code,
        tag=f'band_sweep_uniform_pre{best_overall_ms}ms')

    # -------------------------------------------------------------------
    # Sweep D — consonant-specific window
    # -------------------------------------------------------------------
    max_pre_ms    = max(consonant_pre_onsets_ms.values())
    total_win_ms  = max_pre_ms + CONSONANT_POST_BASE_MS
    total_win_tp  = int(total_win_ms / 1000 * FS)
    # t=0 is start of window (earliest articulatory prep onset across groups)
    t_vec_d       = np.linspace(0, total_win_ms, total_win_tp)

    groups_str = '_'.join([f'{g[:3]}{v}ms'
                           for g, v in consonant_pre_onsets_ms.items()])
    win_str_d  = (f'consonant-specific total {total_win_ms} ms '
                  f'({groups_str})')

    print(f'\n{"="*60}')
    print('  Sweep D — consonant-specific window')
    print(f'  Total window: {total_win_ms} ms ({total_win_tp} samples)')
    for group_name, pre_ms in consonant_pre_onsets_ms.items():
        post_ms = total_win_ms - pre_ms
        syls    = [SYLLABLES[l - 1] for l in CONSONANT_GROUPS[group_name]]
        print(f'    {group_name:12s} ({"/".join(syls)}): '
              f'pre={pre_ms} ms  post={post_ms} ms')
    print(f'{"="*60}')

    def _band_job_D(band_name, band_idx):
        nBands_b = len(band_idx)
        exp_name = (f'BandD_{band_name}_keepIC_zpower_{groups_str}')

        print(f'\n  [D | {band_name}]  '
              f'nICs={len(keep_ics_0idx)} nBands={nBands_b} nTime={total_win_tp}')

        X, _, _ = build_X_consonant_window(
            z_power_smooth, y, onset_tps,
            keep_ics_0idx, consonant_pre_onsets_ms,
            band_idx=band_idx)

        r, rf_model = run_classifiers(
            exp_name, X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set=band_name,
            feature='z_power_smooth',
            window=f'consonant_specific_total{total_win_ms}ms',
            inner_n_jobs=inner_jobs,
            fix_svm_C=fix_svm_C, fix_rf_depth=fix_rf_depth,
            fix_rf_features=fix_rf_features, fix_rf_estimators=fix_rf_estimators,
            fix_rf_split=fix_rf_split)

        plot_feature_importance(
            rf_model, len(keep_ics_0idx), nBands_b, total_win_tp,
            ic_labels, t_vec_d,
            save_dir, subj, cond_code, exp_name,
            band_idx_vec=band_idx)

        return r

    if args.n_jobs == 1:
        results_D = [_band_job_D(name, idx) for name, idx in band_items]
    else:
        results_D = joblib.Parallel(n_jobs=args.n_jobs, prefer='processes')(
            joblib.delayed(_band_job_D)(name, idx) for name, idx in band_items)

    plot_band_sweep_summary(
        results_D, band_labels, figure_save_dir, subj, cond_code,
        win_str=win_str_d, tag='consonant')

    print_and_save_summary(
        results_D, save_dir, subj, cond_code,
        tag=f'band_sweep_consonant_{groups_str}')


if __name__ == '__main__':
    main()


