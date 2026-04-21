#!/usr/bin/env python3
"""
plot_classification_summary.py — Visualise classification experiment results
 
Reads the classification summary and per-class recall CSVs and generates a 2x2 figure:
 
  Top-left     : Overall accuracy bar chart (SVM + RF, grouped by experiment)
  Top-right    : RF per-class recall heatmap (experiments x syllables)
  Bottom-left  : SVM per-class recall heatmap
  Bottom-right : Mean(RF+SVM) per-class recall heatmap
 
Usage
-----
    python plot_classification_summary.py \\
        --summary-csv  /path/to/subj-02_sp_classification_summary.csv \\
        --recall-csv   /path/to/subj-02_sp_per_class_recall.csv \\
        --output-dir   /path/to/figures/ \\
        --subj subj-02 --cond sp
 
    # Filter to specific experiments
    python plot_classification_summary.py ... --experiments W1a W1b W2a W2b W3a
"""
 
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
 
SYLLABLES = ["gi", "gu", "mi", "mu", "si", "su"]
CHANCE    = 1 / 6
 
 
# ---------------------------------------------------------------------------
# Name formatter
# ---------------------------------------------------------------------------
 
def _short_label(exp_name):
    """
    Convert experiment name to readable short label.
 
    Examples
    --------
    W1a_keepIC_zpower_full            → W1a (keepIC, zpower)
    W2b_brainIC_zpower_speech500ms   → W2b (brainIC, zpower)
    W1e_brainIC_combined_full        → W1e (brainIC, combined)
    W3a_brainIC_zpower_prespeech500ms → W3a (brainIC, zpower)
    Band_beta_brainIC_zpower_...     → Band beta (brainIC, zpower)
    W4_ExpA_RF_brainIC_...           → W4 ExpA RF
    """
    parts = exp_name.split('_')
 
    # Band sweep experiments: Band_{band_name}_brainIC_...
    if parts[0] == 'Band' or (len(parts) > 1 and parts[0] in ('BandA', 'BandB')):
        prefix   = parts[0]                          # Band, BandA, BandB
        band     = parts[1]                          # e.g. beta, all, wide
        ic_part  = 'brainIC' if 'brainIC' in exp_name else 'keepIC'
        return f'{prefix} {band} ({ic_part})'
 
    # W4 final experiments: W4_ExpA_RF_... or W4_ExpB_combined_...
    if parts[0] == 'W4' and len(parts) > 2 and parts[1].startswith('Exp'):
        return f'W4 {parts[1]} {parts[2]}'
 
    # W4 sweep: W4_brainIC_zpower_pre250ms_...
    if parts[0] == 'W4' and len(parts) > 2 and parts[1] in ('brainIC', 'keepIC'):
        pre_part = next((p for p in parts if p.startswith('pre')), '')
        return f'W4 ({pre_part})'
 
    # Standard experiments: W{n}{letter}_{ic}_{feature}_{window}
    exp_id  = parts[0]                               # W1a, W2b, W3a ...
    ic_raw  = parts[1] if len(parts) > 1 else ''
    ic_str  = 'brainIC' if 'brain' in ic_raw else 'keepIC'
    feat    = parts[2] if len(parts) > 2 else ''
    feat_map = {'zpower': 'zpower', 'instfreq': 'instfreq', 'combined': 'combined'}
    feat_str = feat_map.get(feat, feat)
 
    return f'{exp_id} ({ic_str}, {feat_str})'
 
 
# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
 
def plot_summary(summary_df, recall_df, subj, cond,
                 output_dir, experiments=None, tag=''):
    """
    2x2 summary figure.
 
    Parameters
    ----------
    summary_df   : pd.DataFrame, from classification_summary CSV
    recall_df    : pd.DataFrame, from per_class_recall CSV
    subj         : str
    cond         : str
    output_dir   : str
    experiments  : list of str or None — filter to specific experiment names
    tag          : str, optional filename suffix
    """
    # Filter experiments if requested
    if experiments:
        summary_df = summary_df[summary_df['experiment'].isin(experiments)].copy()
        recall_df  = recall_df[recall_df['experiment'].isin(experiments)].copy()
 
    # Align both dataframes to same experiment order
    exp_order  = summary_df['experiment'].tolist()
    recall_df  = recall_df.set_index('experiment').reindex(exp_order).reset_index()
 
    n_exp      = len(exp_order)
    short_lbls = [_short_label(e) for e in exp_order]
 
    # Parse accuracy values
    svm_acc  = summary_df['SVM_accuracy'].astype(float).values
    svm_bal  = summary_df['SVM_bal_acc'].astype(float).values
    rf_acc   = summary_df['RF_accuracy'].astype(float).values
    rf_bal   = summary_df['RF_bal_acc'].astype(float).values
 
    # Parse per-class recall matrices
    rf_recall  = np.array([[float(recall_df.loc[i, f'RF_recall_{s}'])
                             for s in SYLLABLES]
                            for i in range(n_exp)])
    svm_recall = np.array([[float(recall_df.loc[i, f'SVM_recall_{s}'])
                             for s in SYLLABLES]
                            for i in range(n_exp)])
    mean_recall = (rf_recall + svm_recall) / 2
 
    # -------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35)
 
    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_rf   = fig.add_subplot(gs[0, 1])
    ax_svm  = fig.add_subplot(gs[1, 0])
    ax_mean = fig.add_subplot(gs[1, 1])
 
    x = np.arange(n_exp)
    w = 0.18
 
    # -------------------------------------------------------------------
    # Top-left: overall accuracy bar chart
    # -------------------------------------------------------------------
    ax_acc.bar(x - 1.5*w, svm_acc, w, label='SVM acc',     color='steelblue',  alpha=0.9)
    ax_acc.bar(x - 0.5*w, svm_bal, w, label='SVM bal acc', color='steelblue',  alpha=0.45)
    ax_acc.bar(x + 0.5*w, rf_acc,  w, label='RF acc',      color='darkorange', alpha=0.9)
    ax_acc.bar(x + 1.5*w, rf_bal,  w, label='RF bal acc',  color='darkorange', alpha=0.45)
    ax_acc.axhline(CHANCE, color='gray', linestyle=':', linewidth=0.9,
                   label=f'Chance ({CHANCE:.2f})')
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(short_lbls, rotation=40, ha='right', fontsize=8)
    ax_acc.set_ylabel('Cross-validated accuracy')
    ax_acc.set_title('Overall accuracy by experiment')
    ax_acc.legend(fontsize=8, loc='upper left')
    ax_acc.grid(axis='y', alpha=0.35)
    ax_acc.set_ylim(0, 1)
 
    # -------------------------------------------------------------------
    # Heatmap helper
    # -------------------------------------------------------------------
    def _heatmap(ax, matrix, title):
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(SYLLABLES)))
        ax.set_xticklabels(SYLLABLES, fontsize=10)
        ax.set_yticks(range(n_exp))
        ax.set_yticklabels(short_lbls, fontsize=8)
        ax.set_xlabel('Syllable')
        ax.set_ylabel('Experiment')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall')
 
        # Annotate cells
        for i in range(n_exp):
            for j in range(len(SYLLABLES)):
                val   = matrix[i, j]
                color = 'white' if val < 0.25 or val > 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
 
        # Chance line marker on colorbar not possible cleanly —
        # add a horizontal dashed line at chance level on a twin axis instead
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        ax2.set_yticklabels([])
 
    _heatmap(ax_rf,   rf_recall,   'RF per-class recall')
    _heatmap(ax_svm,  svm_recall,  'SVM per-class recall')
    _heatmap(ax_mean, mean_recall, 'Mean(RF+SVM) per-class recall')
 
    fig.suptitle(
        f'{subj} | {cond} — Classification experiment summary\n'
        f'Chance = {CHANCE:.3f}   (n_exp={n_exp})',
        fontsize=13, y=0.98)
 
    os.makedirs(output_dir, exist_ok=True)
    suffix  = f'_{tag}' if tag else ''
    fpath   = os.path.join(output_dir,
                           f'{subj}_{cond}_classification_summary_plot{suffix}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description='Plot classification experiment summary (2×2 figure)')
 
    parser.add_argument(
        '--summary-csv', required=True, type=str, nargs='+',
        help='One or more *_classification_summary*.csv files to combine')
    parser.add_argument(
        '--recall-csv', required=True, type=str, nargs='+',
        help='One or more *_per_class_recall*.csv files to combine')
    parser.add_argument(
        '--output-dir', required=True, type=str,
        help='Directory to save the figure')
    parser.add_argument(
        '--subj', required=True, type=str,
        help='Subject ID, e.g. subj-02')
    parser.add_argument(
        '--cond', default='sp', type=str,
        help='Condition code, e.g. sp or im (default: sp)')
    parser.add_argument(
        '--experiments', default=None, type=str, nargs='+',
        help='Filter to specific experiment names (default: all)')
    parser.add_argument(
        '--tag', default='', type=str,
        help='Optional suffix for the output filename')
 
    args = parser.parse_args()
 
    summary_df = pd.concat([pd.read_csv(f, dtype=str) for f in args.summary_csv], ignore_index=True)
    recall_df  = pd.concat([pd.read_csv(f, dtype=str) for f in args.recall_csv], ignore_index=True)
 
    print(f'Loaded {len(summary_df)} experiments from {args.summary_csv}')
 
    plot_summary(
        summary_df, recall_df,
        subj=args.subj, cond=args.cond,
        output_dir=args.output_dir,
        experiments=args.experiments,
        tag=args.tag)
 
 
if __name__ == '__main__':
    main()