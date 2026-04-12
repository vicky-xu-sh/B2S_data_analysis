#!/usr/bin/env python3
"""
classification_overt.py — EEG syllable classification pipeline (overt condition)

Experiments
-----------
W1 — Full post-stimulus window (0 → 1950ms):
  W1a  all-keep ICs | z_power_smooth   | all bands
  W1b  brain ICs    | z_power_smooth   | all bands
  W1c  all-keep ICs | inst_freq_smooth | all bands
  W1d  brain ICs    | inst_freq_smooth | all bands
  W1e  brain ICs    | z_power_smooth + inst_freq_smooth (combined) | all bands

W2 — Speech window (onset → onset + N ms, N derived from mean speech duration):
  W2a  all-keep ICs | z_power_smooth  | all bands
  W2b  brain ICs    | z_power_smooth  | all bands
  W2c  brain ICs    | z_power_smooth + inst_freq_smooth (combined) | all bands

W3 — Pre-speech only (onset - 500ms → onset):
  W3a  brain ICs    | z_power_smooth  | all bands
  W3b  all-keep ICs | z_power_smooth  | all bands

Usage
-----
    python classification_overt.py \\
        --subj subj-02 \\
        --input-dir  /path/to/04_processed/ \\
        --output-dir /path/to/results/ \\
        --overt-keep-ics 2 3 4 5 7 12 14 \\
        --overt-brain-ics 4 7 14 \\
        --overt-bad-epochs 1 111

    # Use subject-specific speech window override
    python classification_overt.py ... --speech-window-ms 600
"""

import os
import argparse
import numpy as np
from scipy.io import loadmat

from utils import (
    FS, LP_CUTOFF_HZ, W3_PRE_ONSET_MS,
    load_analytic, compute_features, reject_epochs,
    compute_speech_window_tp, derive_epoch_indices,
    lowpass_smooth,
    build_X, build_X_speech_window,
    run_classifiers,
    plot_feature_importance,
    print_and_save_summary,
)


# ---------------------------------------------------------------------------
# Experiment runner (top-level so it can be imported/tested independently)
# ---------------------------------------------------------------------------

def run_exp(name, X, y, save_dir, subj, cond,
            ic_set, band_set, feature, window,
            ic_labels, time_vec, nICs, nBands, nTime,
            inner_jobs, skip_importance=False):
    """
    Run classifiers for one experiment and save feature importance plots.

    Parameters
    ----------
    name            : str, experiment label
    X               : np.ndarray [trials x features]
    y               : np.ndarray [trials]
    save_dir        : str
    subj            : str
    cond            : str
    ic_set          : str
    band_set        : str
    feature         : str
    window          : str
    ic_labels       : list of str
    time_vec        : np.ndarray, time axis in ms for importance plots
    nICs            : int
    nBands          : int
    nTime           : int
    inner_jobs      : int, n_jobs for GridSearchCV
    skip_importance : bool, skip feature importance plot (e.g. combined features)

    Returns
    -------
    ExperimentResult
    """
    print(f'\n  [{name}]  X={X.shape}')
    result, rf_model = run_classifiers(
        name, X, y, save_dir, subj, cond,
        ic_set=ic_set, band_set=band_set,
        feature=feature, window=window,
        inner_n_jobs=inner_jobs)

    if not skip_importance:
        plot_feature_importance(
            rf_model, nICs, nBands, nTime,
            ic_labels, time_vec,
            save_dir, subj, cond, name)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='EEG syllable classification — overt condition')

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
        '--overt-keep-ics', required=False, default=None, type=int, nargs='+',
        help='1-indexed ICs to use (default: all)')
    parser.add_argument(
        '--overt-brain-ics', required=False, default=None, type=int, nargs='+',
        help='1-indexed brain ICs (required for brain-IC experiments)')
    parser.add_argument(
        '--overt-bad-epochs', default=[], type=int, nargs='*',
        help='1-indexed bad epochs to reject')
    parser.add_argument(
        '--speech-window-ms', default=None, type=int,
        help='Override auto-computed speech window length (ms). '
             'Default: ceil(mean speech duration / 100ms) * 100ms')
    parser.add_argument(
        '--n-jobs', default=1, type=int,
        help='n_jobs=1 → sequential, GridSearchCV uses all cores. '
             'n_jobs>1 → parallel experiments, GridSearchCV single-threaded.')

    args = parser.parse_args()

    subj      = args.subj
    cond_code = 'sp'
    cond_label = 'spoken'

    # Parallelism: avoid nested joblib pools
    # n_jobs=1 → sequential experiments, GridSearchCV uses all cores
    # n_jobs>1 → parallel experiments, GridSearchCV single-threaded
    inner_jobs = -1 if args.n_jobs == 1 else 1

    # IC index setup
    keep_ics_1idx  = args.overt_keep_ics
    brain_ics_1idx = args.overt_brain_ics
    bad_epochs     = args.overt_bad_epochs

    run_brain_exps = brain_ics_1idx is not None
    if not run_brain_exps:
        print('  WARNING: --overt-brain-ics not provided. '
              'Brain-IC experiments (W1b, W1d, W1e, W2b, W2c, W3, band sweep) will be skipped.')

    keep_ics_0idx  = [ic - 1 for ic in keep_ics_1idx] if keep_ics_1idx else None
    brain_ics_0idx = [ic - 1 for ic in brain_ics_1idx] if run_brain_exps else None

    save_dir = os.path.join(args.output_dir, subj, cond_label, 'baseline_windows')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"="*60}\n  Subject: {subj}\n  Condition: {cond_label} ({cond_code})\n{"="*60}')
    print(f'  Keep ICs (1-indexed): {keep_ics_1idx if keep_ics_1idx else "all"} ({len(keep_ics_1idx) if keep_ics_1idx else ""} in total)')
    if run_brain_exps:
        print(f'  Brain ICs (1-indexed): {brain_ics_1idx} ({len(brain_ics_1idx)} in total)')
    else:
        print('  Brain ICs (1-indexed): N/A (brain-IC experiments will be skipped)')
    if len(bad_epochs) > 0:
        print(f'  Bad epochs (1-indexed): {bad_epochs}')  
    else:
        print('  Bad epochs (1-indexed): None (no epoch rejection will be performed)')  

    # -------------------------------------------------------------------
    # Load and preprocess
    # -------------------------------------------------------------------
    data_path  = os.path.join(args.input_dir, f'{subj}_{cond_code}_eeg_analytic.mat')
    onset_path = os.path.join(args.input_dir, f'{subj}_speech_onset_offset.mat')

    print('[1/4] Loading analytic signal...')
    Z, labels, times = load_analytic(data_path)
    print(f'  Shape: {Z.shape}')

    # Derive epoch sample indices from times array
    idx_0ms, idx_1950ms = derive_epoch_indices(times)

    print('[2/4] Computing features...')
    _, _, inst_freq, z_power = compute_features(Z, times)

    speech_data  = loadmat(onset_path)
    onset_times  = speech_data['onset_latencies'].squeeze()
    offset_times = speech_data['offset_latencies'].squeeze()

    if len(bad_epochs) > 0:
        print(f"[3/4] Rejecting {len(bad_epochs)} bad epoch(s): {bad_epochs}")
        good_mask, (z_power, inst_freq, labels, onset_times, offset_times) = \
            reject_epochs(bad_epochs, z_power, inst_freq, labels, onset_times, offset_times)
        print(f"  Remaining trials: {good_mask.sum()}")
    else:
        print("[3/4] Skipping bad epoch rejection - no bad epochs specified")
        good_mask = np.ones(z_power.shape[-1], dtype=bool)
        print(f"  Keeping all {good_mask.sum()} trials")

    # Validate trial counts
    n_trials = good_mask.sum()
    if len(onset_times) != n_trials or len(offset_times) != n_trials:
        print(f'  WARNING: trial count mismatch — '
              f'labels={n_trials}, onset={len(onset_times)}, offset={len(offset_times)}')

    print(f'[4/4] Smoothing (LP {LP_CUTOFF_HZ} Hz)...')
    z_power_smooth   = lowpass_smooth(z_power)
    inst_freq_smooth = lowpass_smooth(inst_freq)

    # Sample indices for onset/offset
    onset_tps  = np.rint(onset_times  / 1000 * FS + idx_0ms).astype(int)
    offset_tps = np.rint(offset_times / 1000 * FS + idx_0ms).astype(int)

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

    speech_window_ms = int(speech_window_tp / FS * 1000)

    y = np.array(labels).astype(int)

    # Time vectors for feature importance plots
    n_full_tp         = idx_1950ms - idx_0ms
    time_vec_full     = times[idx_0ms:idx_1950ms]
    time_vec_speech   = np.linspace(0, speech_window_tp / FS * 1000, speech_window_tp)
    time_vec_prespeech = np.linspace(-W3_PRE_ONSET_MS, 0,
                                     int(W3_PRE_ONSET_MS / 1000 * FS))

    ic_labels_keep  = [f'IC{ic}' for ic in keep_ics_1idx] if keep_ics_1idx else []
    ic_labels_brain = [f'IC{ic}' for ic in brain_ics_1idx] if run_brain_exps else []

    all_results = []
    t_full = slice(idx_0ms, idx_1950ms)

    # -------------------------------------------------------------------
    # W1 — Full post-stimulus window (0 → 1950ms)
    # -------------------------------------------------------------------
    print(f'\n{"="*60}\n  W1 — Full window (0 → 1950ms)\n{"="*60}')

    if keep_ics_0idx is not None:
        # W1a: all-keep ICs | z_power_smooth
        X = build_X(z_power_smooth, keep_ics_0idx, time_slice=t_full)
        all_results.append(run_exp(
            'W1a_allIC_zpower_full', X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set='all_bands',
            feature='z_power_smooth', window='0-1950ms',
            ic_labels=ic_labels_keep, time_vec=time_vec_full,
            nICs=len(keep_ics_0idx), nBands=z_power_smooth.shape[1], nTime=n_full_tp,
            inner_jobs=inner_jobs))

        # W1c: all-keep ICs | inst_freq_smooth
        X = build_X(inst_freq_smooth, keep_ics_0idx, time_slice=t_full)
        all_results.append(run_exp(
            'W1c_allIC_instfreq_full', X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set='all_bands',
            feature='inst_freq_smooth', window='0-1950ms',
            ic_labels=ic_labels_keep, time_vec=time_vec_full,
            nICs=len(keep_ics_0idx), nBands=inst_freq_smooth.shape[1], nTime=n_full_tp,
            inner_jobs=inner_jobs))

    if run_brain_exps:
        # W1b: brain ICs | z_power_smooth
        X = build_X(z_power_smooth, brain_ics_0idx, time_slice=t_full)
        all_results.append(run_exp(
            'W1b_brainIC_zpower_full', X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='z_power_smooth', window='0-1950ms',
            ic_labels=ic_labels_brain, time_vec=time_vec_full,
            nICs=len(brain_ics_0idx), nBands=z_power_smooth.shape[1], nTime=n_full_tp,
            inner_jobs=inner_jobs))

        # W1d: brain ICs | inst_freq_smooth
        X = build_X(inst_freq_smooth, brain_ics_0idx, time_slice=t_full)
        all_results.append(run_exp(
            'W1d_brainIC_instfreq_full', X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='inst_freq_smooth', window='0-1950ms',
            ic_labels=ic_labels_brain, time_vec=time_vec_full,
            nICs=len(brain_ics_0idx), nBands=inst_freq_smooth.shape[1], nTime=n_full_tp,
            inner_jobs=inner_jobs))

        # W1e: brain ICs | combined z_power + inst_freq
        X_pw = build_X(z_power_smooth,   brain_ics_0idx, time_slice=t_full)
        X_if = build_X(inst_freq_smooth, brain_ics_0idx, time_slice=t_full)
        X    = np.hstack([X_pw, X_if])
        all_results.append(run_exp(
            'W1e_brainIC_combined_full', X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='z_power+inst_freq', window='0-1950ms',
            ic_labels=ic_labels_brain, time_vec=time_vec_full,
            nICs=len(brain_ics_0idx), nBands=z_power_smooth.shape[1], nTime=n_full_tp,
            inner_jobs=inner_jobs, skip_importance=True))

    # -------------------------------------------------------------------
    # W2 — Speech window (onset → onset + speech_window_tp)
    # -------------------------------------------------------------------
    print(f'\n{"="*60}\n  W2 — Speech window (onset → onset+{speech_window_tp}tp = '
          f'{speech_window_tp/FS*1000:.0f}ms)\n{"="*60}')

    if keep_ics_0idx is not None:
        # W2a: all-keep ICs | z_power_smooth
        X = build_X_speech_window(z_power_smooth, keep_ics_0idx, onset_tps,
                                   pre_onset_tp=0, post_onset_tp=speech_window_tp)
        all_results.append(run_exp(
            f'W2a_allIC_zpower_speech{speech_window_ms}ms',
            X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set='all_bands',
            feature='z_power_smooth', window=f'onset+{speech_window_ms}ms',
            ic_labels=ic_labels_keep, time_vec=time_vec_speech,
            nICs=len(keep_ics_0idx), nBands=z_power_smooth.shape[1],
            nTime=speech_window_tp, inner_jobs=inner_jobs))

    if run_brain_exps:
        # W2b: brain ICs | z_power_smooth
        X = build_X_speech_window(z_power_smooth, brain_ics_0idx, onset_tps,
                                   pre_onset_tp=0, post_onset_tp=speech_window_tp)
        all_results.append(run_exp(
            f'W2b_brainIC_zpower_speech{speech_window_ms}ms',
            X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='z_power_smooth', window=f'onset+{speech_window_ms}ms',
            ic_labels=ic_labels_brain, time_vec=time_vec_speech,
            nICs=len(brain_ics_0idx), nBands=z_power_smooth.shape[1],
            nTime=speech_window_tp, inner_jobs=inner_jobs))

        # W2c: brain ICs | combined z_power + inst_freq
        X_pw = build_X_speech_window(z_power_smooth,   brain_ics_0idx, onset_tps,
                                      pre_onset_tp=0, post_onset_tp=speech_window_tp)
        X_if = build_X_speech_window(inst_freq_smooth, brain_ics_0idx, onset_tps,
                                      pre_onset_tp=0, post_onset_tp=speech_window_tp)
        X    = np.hstack([X_pw, X_if])
        all_results.append(run_exp(
            f'W2c_brainIC_combined_speech{speech_window_ms}ms',
            X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='z_power+inst_freq', window=f'onset+{speech_window_ms}ms',
            ic_labels=ic_labels_brain, time_vec=time_vec_speech,
            nICs=len(brain_ics_0idx), nBands=z_power_smooth.shape[1],
            nTime=speech_window_tp, inner_jobs=inner_jobs, skip_importance=True))

    # -------------------------------------------------------------------
    # W3 — Pre-speech only (onset - 500ms → onset)
    # -------------------------------------------------------------------
    pre_tp = int(W3_PRE_ONSET_MS / 1000 * FS)
    print(f'\n{"="*60}\n  W3 — Pre-speech only '
          f'(onset−{W3_PRE_ONSET_MS}ms → onset, {pre_tp} samples)\n{"="*60}')

    if run_brain_exps:
        # W3a: brain ICs
        X = build_X_speech_window(z_power_smooth, brain_ics_0idx, onset_tps,
                                   pre_onset_tp=pre_tp, post_onset_tp=0)
        all_results.append(run_exp(
            f'W3a_brainIC_zpower_prespeech{W3_PRE_ONSET_MS}ms',
            X, y, save_dir, subj, cond_code,
            ic_set='brain', band_set='all_bands',
            feature='z_power_smooth', window=f'onset-{W3_PRE_ONSET_MS}ms_to_onset',
            ic_labels=ic_labels_brain, time_vec=time_vec_prespeech,
            nICs=len(brain_ics_0idx), nBands=z_power_smooth.shape[1],
            nTime=pre_tp, inner_jobs=inner_jobs))

    if keep_ics_0idx is not None:
        # W3b: all-keep ICs
        X = build_X_speech_window(z_power_smooth, keep_ics_0idx, onset_tps,
                                   pre_onset_tp=pre_tp, post_onset_tp=0)
        all_results.append(run_exp(
            f'W3b_allIC_zpower_prespeech{W3_PRE_ONSET_MS}ms',
            X, y, save_dir, subj, cond_code,
            ic_set='all_keep', band_set='all_bands',
            feature='z_power_smooth', window=f'onset-{W3_PRE_ONSET_MS}ms_to_onset',
            ic_labels=ic_labels_keep, time_vec=time_vec_prespeech,
            nICs=len(keep_ics_0idx), nBands=z_power_smooth.shape[1],
            nTime=pre_tp, inner_jobs=inner_jobs))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print_and_save_summary(all_results, save_dir, subj, cond_code)


if __name__ == '__main__':
    main()
