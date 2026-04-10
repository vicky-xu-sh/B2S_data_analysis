from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for both local and HPC
import matplotlib.pyplot as plt
import os
import h5py
import argparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POST_STIM_0MS   = 250       # sample index corresponding to 0 ms post-stimulus
SYLLABLES       = ["gi", "gu", "mi", "mu", "si", "su"]
FS              = 500       # sampling rate (Hz)
LP_CUTOFF_HZ    = 10        # low-pass cutoff for smoothing (Hz) 
LP_ORDER        = 4         # Butterworth filter order


# ---------------------------------------------------------------------------
# Smoothing — low-pass Butterworth (zero-phase, sosfiltfilt)
# ---------------------------------------------------------------------------

def lowpass_smooth(data, cutoff_hz=LP_CUTOFF_HZ, fs=FS, order=LP_ORDER, axis=2):
    """
    Apply a zero-phase Butterworth low-pass filter along the time axis.

    Reportable as: "4th-order zero-phase Butterworth low-pass filter at X Hz"

    Parameters
    ----------
    data       : np.ndarray [...x time x ...]
    cutoff_hz  : float, cutoff frequency in Hz
    fs         : float, sampling rate in Hz
    order      : int, filter order (4 is standard)
    axis       : int, time axis (default=2 for [comps x bands x time x trials])

    Returns
    -------
    np.ndarray, same shape as input
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz ({cutoff_hz}) must be less than Nyquist ({nyq} Hz)")
    sos = butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, data, axis=axis)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tfa_by_class(data, labels, times, comp_num=1, time_range=None,
                      feature_name='z_power', ylabel='Power (z-score)',
                      class_names=SYLLABLES, realign_times=None,
                      save_dir=None, subj=None, cond=None):
    """
    Plot any TFA feature per class for a single IC component.

    Parameters
    ----------
    data          : np.ndarray [comps x bands x time x trials]
    labels        : np.ndarray [trials], integer class labels (1-indexed)
    times         : np.ndarray [time], in ms
    comp_num      : int, 1-indexed component number to plot
    time_range    : tuple (start_idx, end_idx) or None
    feature_name  : str, used in title and filename
    ylabel        : str, y-axis label
    class_names   : list of str
    realign_times : np.ndarray [trials] or None, event times in ms to realign to
    save_dir      : str or None — if provided, saves figure here instead of showing
    subj          : str, used in filename (required if save_dir is set)
    cond          : str, used in filename (required if save_dir is set)
    """
    band_names  = ['Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma']
    band_ranges = [(4, 7), (8, 12), (13, 30), (30, 75), (75, 150)]
    nClasses    = len(class_names)
    nICs        = data.shape[0]
    nBands      = data.shape[1]
    nTimes      = data.shape[2]
    nTrials     = data.shape[3]
    comp_idx    = comp_num - 1
    labels      = np.array(labels).squeeze().astype(int)
    times_trim  = times[:nTimes]  # handles inst_freq being 1 sample shorter

    # --- Realign trials if requested ---
    if realign_times is not None:
        realign_times = np.array(realign_times).squeeze()
        assert len(realign_times) == nTrials, \
            f"realign_times length {len(realign_times)} must match nTrials {nTrials}"

        realign_times_tps = np.rint(realign_times / 1000 * FS + POST_STIM_0MS).astype(int)
        min_tps = np.min(realign_times_tps)
        max_tps = np.max(realign_times_tps)
        print(f"  Realignment times (samples): min={min_tps}, max={max_tps}")

        window_start  = max(0, min_tps - POST_STIM_0MS)
        window_length = nTimes - max_tps
        window_end    = window_start + window_length
        print(f"  Realignment window: start={window_start}, length={window_length}, end={window_end}")

        data_realigned = np.full((nICs, nBands, window_start + window_length, nTrials), np.nan)

        for i in range(nTrials):
            tps   = realign_times_tps[i]
            shift = (min_tps - POST_STIM_0MS) - tps
            for band in range(nBands):
                trial_data = data[comp_idx, band, :, i]
                shifted    = np.roll(trial_data, shift)
                data_realigned[comp_idx, band, :, i] = shifted[0:window_end]

        data         = data_realigned
        title_suffix = ' (realigned to onset)'
        name_suffix = '_realigned'
        nTimes       = data_realigned.shape[2]
    else:
        title_suffix = ''
        name_suffix = ''

    # --- Per-class mean across trials ---
    class_means = np.zeros((nBands, nTimes, nClasses))
    for c in range(nClasses):
        idx = labels == (c + 1)
        class_means[:, :, c] = data[comp_idx][:, :, idx].mean(axis=-1)

    # --- Colors: 3 base hues, each with a lighter variant ---
    base_colors    = np.array([[0.2, 0.4, 0.8],
                                [0.2, 0.7, 0.2],
                                [0.7, 0.2, 0.7]])
    lighter_colors = base_colors * 0.35 + 0.65
    colors         = np.zeros((6, 3))
    colors[0::2]   = base_colors
    colors[1::2]   = lighter_colors

    # --- Time slice ---
    if time_range is None:
        time_range = (0, nTimes)
    else:
        time_range = (max(0, time_range[0]), min(nTimes, time_range[1]))
    t_slice = slice(time_range[0], time_range[1])
    t_axis  = times_trim[t_slice]

    # --- Plot ---
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    axes = axes.flatten()

    for b in range(nBands):
        ax = axes[b]
        for i in range(nClasses):
            ax.plot(t_axis, class_means[b, t_slice, i],
                    color=colors[i], linewidth=1.5, label=class_names[i])

        if 'freq' in feature_name.lower():
            fmin, fmax = band_ranges[b]
            ax.axhspan(fmin, fmax, alpha=0.08, color='gray',
                       label=f'Expected {fmin}–{fmax} Hz')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{band_names[b]} Band — {feature_name}')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True)

    for b in range(nBands, nrows * ncols):
        axes[b].set_visible(False)

    suptitle = f'{subj or ""} | IC {comp_num:02d} — {feature_name} by Class{title_suffix}'
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    # --- Save or show ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{subj}_{cond}_IC{comp_num:02d}_{feature_name}{name_suffix}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fpath}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_analytic(data_mat_filepath):
    """
    Load analytic signal, labels, and times from the HDF5 .mat file.

    Returns
    -------
    Z      : np.ndarray complex [comps x bands x time x trials]
    labels : np.ndarray [trials]
    times  : np.ndarray [time], in ms
    """
    with h5py.File(data_mat_filepath, 'r') as f:
        # MATLAB saved as [comps x bands x time x trials]
        # h5py reads transposed: [trials x time x bands x comps]
        Z_raw  = f['analytic_signal'][:]
        labels = f['labels'][:].squeeze()
        times  = f['times'][:].squeeze()

    # Restore MATLAB dimension order
    Z_raw = Z_raw.transpose(3, 2, 1, 0)

    # h5py stores complex as structured array with 'real' and 'imag' fields
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

    baseline_mask = (times >= -450) & (times <= 0)
    mu    = power[..., baseline_mask, :].mean(axis=-2, keepdims=True)
    std   = power[..., baseline_mask, :].std(axis=-2, keepdims=True)
    z_power = (power - mu) / (std + 1e-8)

    return power, phase, inst_freq, z_power


def reject_epochs(bad_epochs_1idx, *arrays):
    """
    Remove bad epochs (1-indexed) from any number of trailing-axis arrays.

    Parameters
    ----------
    bad_epochs_1idx : list of int, 1-indexed bad epoch numbers
    *arrays         : np.ndarray with trials on the last axis

    Returns
    -------
    good_mask : np.ndarray bool [trials]
    arrays    : tuple of np.ndarray with bad trials removed
    """
    n_trials  = arrays[0].shape[-1]
    good_mask = np.ones(n_trials, dtype=bool)
    good_mask[np.array(bad_epochs_1idx) - 1] = False
    return good_mask, tuple(a[..., good_mask] for a in arrays)


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------

def run_condition(
    input_dir, output_dir, figure_dir_name,
    subj, cond_code, cond_label,
    keep_ics, bad_epochs,
    edge_trim_ms=50,
    realign_onset=False):
    
    print(f"\n{'='*60}")
    print(f"  {subj} | {cond_label} ({cond_code})")
    print(f"{'='*60}")

    data_mat_filepath = os.path.join(input_dir, f"{subj}_{cond_code}_eeg_analytic.mat")
    onset_offset_mat_filepath = os.path.join(input_dir, f"{subj}_speech_onset_offset.mat")
    figure_dir = os.path.join(output_dir, subj, cond_label, figure_dir_name)

    for p in [data_mat_filepath, onset_offset_mat_filepath]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    os.makedirs(figure_dir, exist_ok=True)

    print("[1/4] Loading analytic signal...")
    Z, labels, times = load_analytic(data_mat_filepath)
    print(f"  Analytic signal shape: {Z.shape}")

    # --- Default keep_ics to all components if not specified ---
    nComps = Z.shape[0]
    if keep_ics is None:
        keep_ics = list(range(1, nComps + 1))
        print(f"  No ICs specified — plotting all {nComps} components")

    print("[2/4] Computing features...")
    power, phase, inst_freq, z_power = compute_features(Z, times)

    speech_data  = loadmat(onset_offset_mat_filepath)
    onset_times  = speech_data['onset_latencies'].squeeze()
    offset_times = speech_data['offset_latencies'].squeeze()

    print(f"[3/4] Rejecting {len(bad_epochs)} bad epoch(s): {bad_epochs}")
    good_mask, (power, z_power, inst_freq, labels, onset_times, offset_times) = \
        reject_epochs(bad_epochs, power, z_power, inst_freq, labels, onset_times, offset_times)
    print(f"  Remaining trials: {good_mask.sum()}")

    print(f"[4/4] Smoothing (zero-phase Butterworth LP at {LP_CUTOFF_HZ} Hz, order {LP_ORDER})...")
    z_power_smooth   = lowpass_smooth(z_power,   cutoff_hz=LP_CUTOFF_HZ, fs=FS)
    inst_freq_smooth = lowpass_smooth(inst_freq, cutoff_hz=LP_CUTOFF_HZ, fs=FS)

    # --- Edge trim: convert ms → samples, apply to both ends ---
    edge_samples = int(edge_trim_ms / 1000 * FS)
    nTimes       = z_power_smooth.shape[2]
    time_range   = (edge_samples, nTimes - edge_samples)
    print(f"  Edge trim: {edge_trim_ms} ms = {edge_samples} samples each end "
          f"→ time_range={time_range}")

    print(f"\nPlotting {len(keep_ics)} component(s) → {figure_dir}")

    for comp in keep_ics:
        print(f"\n  IC {comp:02d}")
        plot_tfa_by_class(
            z_power_smooth, labels, times, comp_num=comp,
            feature_name='z_power', ylabel='Power (z-score)',
            time_range=time_range,
            realign_times=onset_times if realign_onset else None,   
            save_dir=figure_dir, subj=subj, cond=cond_code
        )
        plot_tfa_by_class(
            inst_freq_smooth, labels, times, comp_num=comp,
            feature_name='inst_freq', ylabel='Instantaneous Frequency (Hz)',
            time_range=time_range,
            realign_times=onset_times if realign_onset else None,   
            save_dir=figure_dir, subj=subj, cond=cond_code
        )

    print(f"\nDone — {subj} | {cond_label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TFA analysis — plot IC components by class.')

    parser.add_argument(
        '--subj', required=True, type=str,
        help='Subject ID, e.g. subj-02')
    parser.add_argument('--input-dir',required=True, type=str,
        help='Input data directory. ')
    parser.add_argument('--output-dir',required=True, type=str,
        help='Output directory. ')
    parser.add_argument('--figure-dir-name', default='component_figures', type=str,
        help='Name of the figure output directory (default: component_figures)')
    parser.add_argument(
        '--overt-keep-ics', required=False, default=None, type=int, nargs='+',
        help='1-indexed ICs to plot for overt (default: all)')
    parser.add_argument(
        '--overt-bad-epochs', default=[], type=int, nargs='*',
        help='1-indexed bad epochs for overt condition')
    parser.add_argument(
        '--covert-keep-ics', required=False, default=None, type=int, nargs='+',
        help='1-indexed ICs to plot for covert (default: all)')
    parser.add_argument(
        '--covert-bad-epochs', default=[], type=int, nargs='*',
        help='1-indexed bad epochs for covert condition')
    parser.add_argument(
        '--conditions', default=['sp', 'im'], nargs='+', choices=['sp', 'im'],
        help='Which conditions to run (default: both)')
    parser.add_argument(
        '--edge-trim-ms', default=50, type=int,
        help='Samples to trim from each end of plots to hide filter edge artifacts (default: 50 ms)')
    parser.add_argument(
        '--realign-onset', action='store_true', default=False,
        help='Realign trials to speech onset times before plotting')


    args = parser.parse_args()

    cond_map = {'sp': 'spoken', 'im': 'imagined'}

    for cond_code in args.conditions:
        keep_ics   = args.overt_keep_ics   if cond_code == 'sp' else args.covert_keep_ics
        bad_epochs = args.overt_bad_epochs if cond_code == 'sp' else args.covert_bad_epochs

        run_condition(
            subj         = args.subj,
            cond_code    = cond_code,
            cond_label   = cond_map[cond_code],
            keep_ics     = keep_ics,
            bad_epochs   = bad_epochs,
            input_dir    = args.input_dir,
            output_dir   = args.output_dir,
            figure_dir_name = args.figure_dir_name,
            edge_trim_ms = args.edge_trim_ms,
            realign_onset = args.realign_onset,
        )