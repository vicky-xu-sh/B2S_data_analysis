#!/usr/bin/env python3
"""
group_analysis.py — Group-level statistical analysis of covert speech classification.

Works through analyses in order:
  #1  Baseline vs. chance (Wilcoxon + t-test)
  #2  (to be added) Improvement over baseline — per-subject and fixed condition
  #3  (to be added) Factor decomposition: window × band × template
  #4  (to be added) Per-class recall analysis
  #5  (to be added) Consistency and reliability across subjects
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "covert_classification_corr0.8_rank1_brain"
SUBJECTS    = ["subj-01", "subj-02", "subj-03", "subj-04", "subj-05", "subj-06", "subj-07", "subj-08", "subj-11", "subj-12"]
CHANCE      = 1 / 6          # 6-class classification
BASELINE_EXP = "baseline_keepIC_all_bands_full"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_exps(subject: str) -> pd.DataFrame:
    path = RESULTS_DIR / subject / "covert_classification" / f"{subject}_im_classification_summary_all_covert_exps.csv"
    return pd.read_csv(path)


def cohens_d_one_sample(data: np.ndarray, mu: float) -> float:
    return (np.mean(data) - mu) / np.std(data, ddof=1)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def report_test(label: str, values: np.ndarray, chance: float = CHANCE):
    mean, sd = np.mean(values), np.std(values, ddof=1)
    d = cohens_d_one_sample(values, chance)

    t_stat, t_p = stats.ttest_1samp(values, chance)
    w_stat, w_p = stats.wilcoxon(values - chance, alternative='greater')

    print(f"\n  [{label}]")
    print(f"    Values per subject : {np.round(values, 4).tolist()}")
    print(f"    Mean ± SD          : {mean:.4f} ± {sd:.4f}")
    print(f"    Chance level       : {chance:.4f}")
    print(f"    Cohen's d          : {d:.4f}")
    print(f"    t-test (1-samp)    : t = {t_stat:.4f},  p = {t_p:.4f}")
    print(f"    Wilcoxon (>chance) : W = {w_stat:.1f},   p = {w_p:.4f}")
    n_above = np.sum(values > chance)
    print(f"    N subjects > chance: {n_above}/{len(values)}")


# ---------------------------------------------------------------------------
# Analysis #1 — Baseline vs. chance
# ---------------------------------------------------------------------------

def analysis_1_baseline_vs_chance():
    print_section("Analysis #1 — Baseline vs. Chance (1/6 ≈ 0.167)")

    svm_vals, rf_vals, mean_vals = [], [], []

    print(f"\n  {'Subject':<12} {'SVM_bal':>10} {'RF_bal':>10} {'Mean_bal':>10}")
    print(f"  {'-'*44}")

    for subj in SUBJECTS:
        df = load_all_exps(subj)
        row = df[df["experiment"] == BASELINE_EXP]
        if row.empty:
            print(f"  {subj:<12} {'MISSING':>10}")
            continue
        svm = float(row["SVM_bal_acc"].iloc[0])
        rf  = float(row["RF_bal_acc"].iloc[0])
        mean = (svm + rf) / 2
        svm_vals.append(svm)
        rf_vals.append(rf)
        mean_vals.append(mean)
        print(f"  {subj:<12} {svm:>10.4f} {rf:>10.4f} {mean:>10.4f}")

    svm_vals  = np.array(svm_vals)
    rf_vals   = np.array(rf_vals)
    mean_vals = np.array(mean_vals)

    report_test("SVM balanced accuracy",  svm_vals)
    report_test("RF balanced accuracy",   rf_vals)
    report_test("Mean (SVM+RF)/2",        mean_vals)


# ---------------------------------------------------------------------------
# Analysis #2 — Δ matrix: all est_with experiments vs. baseline
# ---------------------------------------------------------------------------

WINDOWS = ["W2", "W3", "W4c", "W4u"]
BANDS   = ["theta", "alpha", "beta", "alpha_beta", "gamma", "wide_gamma",
           "beta_wide_gamma", "high_gamma", "all_bands"]


def find_exp(df: pd.DataFrame, window: str, band: str) -> str | None:
    """Return the experiment name matching '{window}_est_with_{band}_all_keepIC*'.
    W4c/W4u suffixes are subject-dependent, so we match by prefix."""
    prefix = f"{window}_est_with_{band}_all_keepIC"
    matches = [e for e in df.index if e.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def mean_bal_acc(row: pd.Series) -> float:
    return (float(row["SVM_bal_acc"]) + float(row["RF_bal_acc"])) / 2


def build_delta_matrix():
    """
    Returns:
        delta_df     : Δ mean(SVM+RF)/2 bal_acc
        svm_delta_df : Δ SVM bal_acc
        rf_delta_df  : Δ RF bal_acc
        pct_df       : % gain on mean
        svm_pct_df   : % gain on SVM
        rf_pct_df    : % gain on RF
        abs_df       : raw mean bal_acc
    """
    conditions = [f"{w}_{b}" for w in WINDOWS for b in BANDS]
    delta_data, svm_delta_data, rf_delta_data = {}, {}, {}
    pct_data, svm_pct_data, rf_pct_data       = {}, {}, {}
    abs_data                                   = {}

    for subj in SUBJECTS:
        df = load_all_exps(subj)
        df = df.set_index("experiment")

        baseline_row = df.loc[BASELINE_EXP]
        baseline_acc = mean_bal_acc(baseline_row)
        baseline_svm = float(baseline_row["SVM_bal_acc"])
        baseline_rf  = float(baseline_row["RF_bal_acc"])

        deltas, svm_deltas, rf_deltas = [], [], []
        pcts, svm_pcts, rf_pcts       = [], [], []
        abss                           = []

        for w in WINDOWS:
            for b in BANDS:
                name = find_exp(df, w, b)
                if name is not None:
                    row  = df.loc[name]
                    acc  = mean_bal_acc(row)
                    svm  = float(row["SVM_bal_acc"])
                    rf   = float(row["RF_bal_acc"])
                    deltas.append(acc - baseline_acc)
                    svm_deltas.append(svm - baseline_svm)
                    rf_deltas.append(rf - baseline_rf)
                    pcts.append((acc - baseline_acc) / baseline_acc * 100)
                    svm_pcts.append((svm - baseline_svm) / baseline_svm * 100)
                    rf_pcts.append((rf - baseline_rf) / baseline_rf * 100)
                    abss.append(acc)
                else:
                    deltas.append(np.nan)
                    svm_deltas.append(np.nan)
                    rf_deltas.append(np.nan)
                    pcts.append(np.nan)
                    svm_pcts.append(np.nan)
                    rf_pcts.append(np.nan)
                    abss.append(np.nan)

        delta_data[subj]     = deltas
        svm_delta_data[subj] = svm_deltas
        rf_delta_data[subj]  = rf_deltas
        pct_data[subj]       = pcts
        svm_pct_data[subj]   = svm_pcts
        rf_pct_data[subj]    = rf_pcts
        abs_data[subj]       = abss

    delta_df     = pd.DataFrame(delta_data,     index=conditions)
    svm_delta_df = pd.DataFrame(svm_delta_data, index=conditions)
    rf_delta_df  = pd.DataFrame(rf_delta_data,  index=conditions)
    pct_df       = pd.DataFrame(pct_data,       index=conditions)
    svm_pct_df   = pd.DataFrame(svm_pct_data,   index=conditions)
    rf_pct_df    = pd.DataFrame(rf_pct_data,    index=conditions)
    abs_df       = pd.DataFrame(abs_data,       index=conditions)
    return delta_df, svm_delta_df, rf_delta_df, pct_df, svm_pct_df, rf_pct_df, abs_df


def ranked_sort_index(delta_df: pd.DataFrame) -> pd.Index:
    """Sort by N>base descending, then mean Δ descending."""
    mean_delta = delta_df.mean(axis=1)
    n_pos      = (delta_df > 0).sum(axis=1)
    rank_df    = pd.DataFrame({"n_pos": n_pos, "mean_delta": mean_delta})
    return rank_df.sort_values(["n_pos", "mean_delta"], ascending=[False, False]).index


def plot_delta_heatmap(delta_df: pd.DataFrame, save_dir: Path,
                       pct_df: pd.DataFrame | None = None,
                       title: str = "Δ bal acc vs. baseline (est_with, per subject)",
                       filename: str = "delta_heatmap_est_with_vs_baseline.png"):
    mean_delta = delta_df.mean(axis=1)
    n_pos      = (delta_df > 0).sum(axis=1)

    sort_order   = ranked_sort_index(delta_df)
    delta_sorted = delta_df.loc[sort_order]
    mean_sorted  = mean_delta.loc[sort_order]
    npos_sorted  = n_pos.loc[sort_order]
    mean_pct_sorted = pct_df.mean(axis=1).loc[sort_order] if pct_df is not None else None

    row_labels = [c.replace("_", " | ", 1) for c in sort_order]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 10),
        gridspec_kw={"width_ratios": [len(SUBJECTS), 3]},
    )

    vmax = np.nanmax(np.abs(delta_df.values))

    # left panel: per-subject Δ
    sns.heatmap(
        delta_sorted.values,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=[s.replace("subj-", "S") for s in SUBJECTS],
        yticklabels=row_labels,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Δ bal acc (exp − baseline)", "shrink": 0.6},
        annot=True, fmt=".2f", annot_kws={"size": 7},
    )
    axes[0].set_title(title, fontsize=11)
    axes[0].tick_params(axis="y", labelsize=8)

    # right panel: Mean Δ | % gain | N > base
    if mean_pct_sorted is not None:
        summary_vals  = np.column_stack([
            mean_sorted.values,
            mean_pct_sorted.values,
            npos_sorted.values.astype(float),
        ])
        summary_annot = np.array([
            [f"{m:.2f}", f"{p:+.1f}%", f"{int(n)}/{len(SUBJECTS)}"]
            for m, p, n in zip(mean_sorted, mean_pct_sorted, npos_sorted)
        ])
        col_labels = ["Mean Δ", "% gain", "N > base"]
    else:
        summary_vals  = np.column_stack([mean_sorted.values, npos_sorted.values.astype(float)])
        summary_annot = np.array(
            [[f"{m:.2f}", f"{int(n)}/{len(SUBJECTS)}"] for m, n in zip(mean_sorted, npos_sorted)]
        )
        col_labels = ["Mean Δ", "N > base"]

    sns.heatmap(
        summary_vals,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=col_labels,
        yticklabels=[""] * len(sort_order),
        linewidths=0.3,
        linecolor="white",
        cbar=False,
        annot=summary_annot, fmt="", annot_kws={"size": 8},
    )
    axes[1].set_title("Summary", fontsize=11)

    plt.tight_layout()
    out_path = save_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Heatmap saved → {out_path}")


def analysis_2_delta_matrix():
    print_section("Analysis #2 — Δ bal acc: est_with experiments vs. baseline")

    delta_df, svm_delta_df, rf_delta_df, pct_df, svm_pct_df, rf_pct_df, _ = build_delta_matrix()

    mean_delta = delta_df.mean(axis=1)
    mean_pct   = pct_df.mean(axis=1)
    n_pos      = (delta_df > 0).sum(axis=1)

    sort_order = ranked_sort_index(delta_df)

    print(f"\n  {'Condition':<40} {'Mean Δ':>8} {'% gain':>8} {'N>base':>8}")
    print(f"  {'-'*66}")
    for cond in sort_order:
        print(f"  {cond:<40} {mean_delta[cond]:>+8.4f} {mean_pct[cond]:>+7.1f}% {n_pos[cond]:>6}/{len(SUBJECTS)}")

    # save heatmaps
    save_dir = RESULTS_DIR.parent / "group_analysis_figures"
    save_dir.mkdir(exist_ok=True)
    plot_delta_heatmap(delta_df, save_dir, pct_df=pct_df)
    plot_delta_heatmap(
        svm_delta_df, save_dir,
        pct_df=svm_pct_df,
        title="Δ SVM bal acc vs. baseline (est_with, per subject)",
        filename="delta_heatmap_est_with_vs_baseline_SVM_only.png",
    )
    plot_delta_heatmap(
        rf_delta_df, save_dir,
        pct_df=rf_pct_df,
        title="Δ RF bal acc vs. baseline (est_with, per subject)",
        filename="delta_heatmap_est_with_vs_baseline_RF_only.png",
    )

    return delta_df, svm_delta_df, rf_delta_df, pct_df


# ---------------------------------------------------------------------------
# Analysis #2b — Formal paired tests: W4c bands vs. baseline (+ top cross-window)
# ---------------------------------------------------------------------------

def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n     = len(pvalues)
    order = np.argsort(pvalues)
    adj   = pvalues[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result        = np.empty(n)
    result[order] = adj
    return result


def cohens_d_paired(delta: np.ndarray) -> float:
    return np.mean(delta) / np.std(delta, ddof=1)


def analysis_2_paired_tests(delta_df: pd.DataFrame, pct_df: pd.DataFrame):
    print_section("Analysis #2b — Paired tests: all windows × focus bands vs. baseline (FDR corrected)")

    # All W4c bands + focus bands for W2/W3/W4u, all FDR-corrected together
    w4c_conditions   = [f"W4c_{b}" for b in BANDS]
    extra_conditions = [f"{w}_{b}" for w in ["W3", "W2", "W4u"] for b in FOCUS_BANDS]
    test_conditions  = w4c_conditions + extra_conditions

    labels, mean_deltas, mean_pcts, sds, ds, pvals = [], [], [], [], [], []

    for cond in test_conditions:
        if cond not in delta_df.index:
            continue
        delta = delta_df.loc[cond].values.astype(float)
        pct   = pct_df.loc[cond].values.astype(float)
        valid = ~np.isnan(delta)
        delta, pct = delta[valid], pct[valid]
        if len(delta) < 3:
            continue

        w_stat, p = stats.wilcoxon(delta, alternative="greater")
        labels.append(cond)
        mean_deltas.append(np.mean(delta))
        mean_pcts.append(np.mean(pct))
        sds.append(np.std(delta, ddof=1))
        ds.append(cohens_d_paired(delta))
        pvals.append(p)

    pvals_arr  = np.array(pvals)
    padj       = fdr_bh(pvals_arr)

    n_w4c = sum(1 for label in labels if label.startswith("W4c"))

    print(f"\n  {'Condition':<30} {'Mean Δ':>8} {'% gain':>8} {'SD':>7} {'Cohen d':>8} {'p (Wilc)':>10} {'p (FDR)':>10}  sig")
    print(f"  {'-'*93}")

    for i, (lbl, m, pct, sd, d, p, pa) in enumerate(
            zip(labels, mean_deltas, mean_pcts, sds, ds, pvals_arr, padj)):
        if i == n_w4c:
            print("  --- W2 / W3 / W4u × focus bands ---")
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else ("†" if p < 0.05 else "")
        print(f"  {lbl:<30} {m:>+8.4f} {pct:>+7.1f}% {sd:>7.4f} {d:>8.3f} {p:>10.4f} {pa:>10.4f}  {sig}")


# ---------------------------------------------------------------------------
# Analysis #2c — Window comparison at fixed bands
# ---------------------------------------------------------------------------

FOCUS_BANDS = ["high_gamma", "gamma", "wide_gamma", "beta"]
WIN_COLORS  = {"W2": "#4C72B0", "W3": "#DD8452", "W4c": "#55A868", "W4u": "#C44E52"}


def analysis_2_window_summary(delta_df: pd.DataFrame, save_dir: Path):
    print_section("Analysis #2c — Window comparison at fixed bands (W2 vs W3 vs W4c vs W4u)")

    n_pairs = len(WINDOWS) * (len(WINDOWS) - 1) // 2   # 6 pairs
    pairs   = [(i, j) for i in range(len(WINDOWS)) for j in range(i + 1, len(WINDOWS))]

    fig, axes = plt.subplots(1, len(FOCUS_BANDS), figsize=(4 * len(FOCUS_BANDS), 5), sharey=True)

    for ax, band in zip(axes, FOCUS_BANDS):
        print(f"\n  Band: {band}")
        print(f"  {'Window':<8} {'Mean Δ':>8} {'SD':>7} {'Cohen d':>8}")
        print(f"  {'-'*34}")

        window_deltas = []
        for wi, w in enumerate(WINDOWS):
            cond  = f"{w}_{band}"
            delta = delta_df.loc[cond].values.astype(float) if cond in delta_df.index \
                    else np.full(len(SUBJECTS), np.nan)
            window_deltas.append(delta)
            valid = delta[~np.isnan(delta)]
            print(f"  {w:<8} {np.nanmean(delta):>+8.4f} {np.nanstd(delta, ddof=1):>7.4f} "
                  f"{cohens_d_paired(valid):>8.3f}")

            col    = WIN_COLORS[w]
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(delta))
            ax.scatter(wi + jitter, delta, color=col, alpha=0.75, s=40, zorder=3)
            ax.bar(wi, np.nanmean(delta), width=0.5, color=col, alpha=0.25, zorder=2)
            ax.errorbar(wi, np.nanmean(delta), yerr=np.nanstd(delta, ddof=1),
                        color=col, fmt="none", capsize=5, linewidth=2, zorder=4)

        # Friedman test across 4 windows
        stacked = np.column_stack(window_deltas)
        valid   = ~np.any(np.isnan(stacked), axis=1)
        if valid.sum() >= 3:
            f_stat, f_p = stats.friedmanchisquare(*[stacked[valid, i] for i in range(len(WINDOWS))])
            print(f"\n  Friedman test: χ² = {f_stat:.3f}, p = {f_p:.4f}")

            print(f"  {'Pair':<14} {'W':>6} {'p':>8} {'p×6 (Bonf)':>12}  sig")
            print(f"  {'-'*46}")
            for i, j in pairs:
                diff   = stacked[valid, i] - stacked[valid, j]
                w_stat, p = stats.wilcoxon(diff)
                p_bonf = min(p * n_pairs, 1.0)
                sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 \
                      else "*" if p_bonf < 0.05 else ("†" if p < 0.05 else "")
                print(f"  {WINDOWS[i]} vs {WINDOWS[j]:<6} {w_stat:>6.1f} {p:>8.4f} {p_bonf:>12.4f}  {sig}")

            ax.set_title(f"{band}\nFriedman p={f_p:.3f}", fontsize=10)
        else:
            ax.set_title(band, fontsize=10)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(WINDOWS)))
        ax.set_xticklabels(WINDOWS, fontsize=9)
        ax.set_xlabel("Window", fontsize=9)

    axes[0].set_ylabel("Δ mean bal acc (exp − baseline)", fontsize=10)
    fig.suptitle("Window comparison at fixed bands (each dot = one subject)", fontsize=11)
    plt.tight_layout()
    out_path = save_dir / "window_comparison_fixed_bands.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Analysis #3 — Template type comparison: est_with vs gen_tmpl vs grp_tmpl
# ---------------------------------------------------------------------------

TMPL_TYPES  = ["est_with", "gen_tmpl", "grp_tmpl"]
TMPL_COLORS = {"est_with": "#2196F3", "gen_tmpl": "#FF9800", "grp_tmpl": "#4CAF50"}
TMPL_LABELS = {"est_with": "est_with\n(subject onset)", "gen_tmpl": "gen_tmpl\n(generic)", "grp_tmpl": "grp_tmpl\n(group)"}


def find_tmpl_exp(df: pd.DataFrame, window: str, band: str, tmpl_type: str) -> str | None:
    """Return the experiment name matching '{window}_{tmpl_type}_{band}_all_keepIC*'."""
    prefix = f"{window}_{tmpl_type}_{band}_all_keepIC"
    matches = [e for e in df.index if e.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def analysis_3_template_comparison(save_dir: Path):
    print_section("Analysis #3 — Template type: est_with vs gen_tmpl vs grp_tmpl")

    # For each subject × window × band × tmpl_type → mean bal acc
    # Then Δ vs baseline per subject
    subject_dfs = {}
    baselines   = {}
    for subj in SUBJECTS:
        df = load_all_exps(subj).set_index("experiment")
        subject_dfs[subj] = df
        if BASELINE_EXP in df.index:
            baselines[subj] = mean_bal_acc(df.loc[BASELINE_EXP])

    # Build delta table: index = (window, band, tmpl_type), columns = subjects
    records = {}
    for window in WINDOWS:
        for band in FOCUS_BANDS:
            for tmpl in TMPL_TYPES:
                key = (window, band, tmpl)
                vals = []
                for subj in SUBJECTS:
                    df  = subject_dfs[subj]
                    exp = find_tmpl_exp(df, window, band, tmpl)
                    if exp and subj in baselines:
                        vals.append(mean_bal_acc(df.loc[exp]) - baselines[subj])
                    else:
                        vals.append(np.nan)
                records[key] = vals

    # ---- Summary table: per template type, mean Δ averaged over bands & windows ----
    print(f"\n  Overall summary (mean Δ across {FOCUS_BANDS} × {WINDOWS}):")
    print(f"  {'Template':<12} {'Mean Δ':>8} {'SD':>7} {'N>0':>6} {'Cohen d':>8}")
    print(f"  {'-'*44}")
    overall = {}
    for tmpl in TMPL_TYPES:
        # per-subject mean across all focus band × window combos
        per_subj = []
        for si, subj in enumerate(SUBJECTS):
            vals = [records[(w, b, tmpl)][si]
                    for w in WINDOWS for b in FOCUS_BANDS
                    if not np.isnan(records[(w, b, tmpl)][si])]
            per_subj.append(np.mean(vals) if vals else np.nan)
        per_subj = np.array(per_subj)
        overall[tmpl] = per_subj
        valid = per_subj[~np.isnan(per_subj)]
        n_pos = int(np.sum(valid > 0))
        print(f"  {tmpl:<12} {np.mean(valid):>+8.4f} {np.std(valid, ddof=1):>7.4f} "
              f"{n_pos:>4}/{len(valid)} {cohens_d_paired(valid):>8.3f}")

    # Pairwise Wilcoxon on overall (3 pairs, Bonferroni)
    print("\n  Overall pairwise Wilcoxon (Bonferroni n=3):")
    print(f"  {'Pair':<26} {'W':>6} {'p':>8} {'p×3':>8}  sig")
    print(f"  {'-'*54}")
    pairs = [("est_with", "gen_tmpl"), ("est_with", "grp_tmpl"), ("gen_tmpl", "grp_tmpl")]
    for a, b in pairs:
        diff = overall[a] - overall[b]
        valid = ~np.isnan(diff)
        if valid.sum() >= 3:
            w_stat, p = stats.wilcoxon(diff[valid])
            p_b = min(p * 3, 1.0)
            sig = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ("†" if p < 0.05 else "")
            print(f"  {a} vs {b:<12} {w_stat:>6.1f} {p:>8.4f} {p_b:>8.4f}  {sig}")

    # ---- Per-window breakdown ----
    print("\n  Per-window breakdown (Δ averaged over focus bands, Friedman across templates):")
    n_pairs_pw = 3
    for window in WINDOWS:
        print(f"\n  Window: {window}")
        print(f"  {'Template':<12} {'Mean Δ':>8} {'SD':>7} {'Cohen d':>8}")
        print(f"  {'-'*38}")
        per_win = {}
        for tmpl in TMPL_TYPES:
            per_subj = []
            for si in range(len(SUBJECTS)):
                vals = [records[(window, b, tmpl)][si]
                        for b in FOCUS_BANDS
                        if not np.isnan(records[(window, b, tmpl)][si])]
                per_subj.append(np.mean(vals) if vals else np.nan)
            per_subj = np.array(per_subj)
            per_win[tmpl] = per_subj
            valid = per_subj[~np.isnan(per_subj)]
            print(f"  {tmpl:<12} {np.nanmean(per_subj):>+8.4f} {np.nanstd(per_subj, ddof=1):>7.4f} "
                  f"{cohens_d_paired(valid):>8.3f}")

        stacked = np.column_stack([per_win[t] for t in TMPL_TYPES])
        valid   = ~np.any(np.isnan(stacked), axis=1)
        if valid.sum() >= 3:
            f_stat, f_p = stats.friedmanchisquare(*[stacked[valid, i] for i in range(3)])
            print(f"  Friedman: χ²={f_stat:.3f}  p={f_p:.4f}")
            for a, b in pairs:
                ai, bi = TMPL_TYPES.index(a), TMPL_TYPES.index(b)
                diff = stacked[valid, ai] - stacked[valid, bi]
                w_stat, p = stats.wilcoxon(diff)
                p_b = min(p * n_pairs_pw, 1.0)
                sig = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ("†" if p < 0.05 else "")
                print(f"  {a} vs {b:<12} W={w_stat:.1f}  p={p:.4f}  p×3={p_b:.4f}  {sig}")

    # ---- W4c × high_gamma specific breakdown ----
    print("\n  W4c × high_gamma specific (template comparison at best condition):")
    print(f"  {'Template':<12} {'Mean Δ':>8} {'SD':>7} {'N>0':>6} {'Cohen d':>8}")
    print(f"  {'-'*44}")
    w4c_hg = {}
    for tmpl in TMPL_TYPES:
        per_subj = []
        for si, subj in enumerate(SUBJECTS):
            df  = subject_dfs[subj]
            exp = find_tmpl_exp(df, "W4c", "high_gamma", tmpl)
            val = (mean_bal_acc(df.loc[exp]) - baselines[subj]) if (exp and subj in baselines) else np.nan
            per_subj.append(val)
        per_subj = np.array(per_subj)
        w4c_hg[tmpl] = per_subj
        valid = per_subj[~np.isnan(per_subj)]
        n_pos = int(np.sum(valid > 0))
        print(f"  {tmpl:<12} {np.mean(valid):>+8.4f} {np.std(valid, ddof=1):>7.4f} "
              f"{n_pos:>4}/{len(valid)} {cohens_d_paired(valid):>8.3f}")
    print("\n  Pairwise Wilcoxon (Bonferroni n=3):")
    print(f"  {'Pair':<26} {'W':>6} {'p':>8} {'p×3':>8}  sig")
    print(f"  {'-'*54}")
    for a, b in pairs:
        diff = w4c_hg[a] - w4c_hg[b]
        valid = ~np.isnan(diff)
        if valid.sum() >= 3:
            w_stat, p = stats.wilcoxon(diff[valid])
            p_b = min(p * 3, 1.0)
            sig = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ("†" if p < 0.05 else "")
            print(f"  {a} vs {b:<12} {w_stat:>6.1f} {p:>8.4f} {p_b:>8.4f}  {sig}")

    # ---- Figure: 4 panels (one per window), 3 bars per panel (template types) ----
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(4 * len(WINDOWS), 5), sharey=True)

    for ax, window in zip(axes, WINDOWS):
        per_win = {}
        for tmpl in TMPL_TYPES:
            per_subj = np.array([
                np.nanmean([records[(window, b, tmpl)][si] for b in FOCUS_BANDS])
                for si in range(len(SUBJECTS))
            ])
            per_win[tmpl] = per_subj

        for xi, tmpl in enumerate(TMPL_TYPES):
            d   = per_win[tmpl]
            col = TMPL_COLORS[tmpl]
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
            ax.scatter(xi + jitter, d, color=col, alpha=0.75, s=40, zorder=3)
            ax.bar(xi, np.nanmean(d), width=0.55, color=col, alpha=0.25, zorder=2)
            ax.errorbar(xi, np.nanmean(d), yerr=np.nanstd(d, ddof=1),
                        color=col, fmt="none", capsize=5, linewidth=2, zorder=4)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(TMPL_TYPES)))
        ax.set_xticklabels([TMPL_LABELS[t] for t in TMPL_TYPES], fontsize=8)
        ax.set_title(window, fontsize=11)
        ax.set_xlabel("Template type", fontsize=9)

    axes[0].set_ylabel("Δ mean bal acc vs baseline\n(avg over focus bands)", fontsize=10)
    fig.suptitle("Template type comparison per window (each dot = one subject)", fontsize=11)
    plt.tight_layout()
    out_path = save_dir / "template_type_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Analysis #4 — Per-class recall
# ---------------------------------------------------------------------------

CLASSES = ["gi", "gu", "mi", "mu", "si", "su"]


def load_per_class_recall(subject: str) -> pd.DataFrame:
    path = (RESULTS_DIR / subject / "covert_classification"
            / f"{subject}_im_per_class_recall_all_covert_exps.csv")
    return pd.read_csv(path).set_index("experiment")


def mean_recall(row: pd.Series, cls: str) -> float:
    return (row[f"SVM_recall_{cls}"] + row[f"RF_recall_{cls}"]) / 2


def analysis_4_per_class_recall(save_dir: Path):
    print_section("Analysis #4 — Per-class recall (W4c_est_with_high_gamma vs baseline)")

    recall_matrix   = np.full((len(SUBJECTS), len(CLASSES)), np.nan)
    baseline_recall = np.full((len(SUBJECTS), len(CLASSES)), np.nan)
    found_subjs     = []

    for si, subj in enumerate(SUBJECTS):
        df_cls = load_per_class_recall(subj)
        df_exp = load_all_exps(subj).set_index("experiment")
        exp    = find_exp(df_exp, "W4c", "high_gamma")
        if exp is None or exp not in df_cls.index:
            print(f"  {subj}: experiment not found, skipping")
            continue
        for ci, cls in enumerate(CLASSES):
            recall_matrix[si, ci] = mean_recall(df_cls.loc[exp], cls)
        if BASELINE_EXP in df_cls.index:
            for ci, cls in enumerate(CLASSES):
                baseline_recall[si, ci] = mean_recall(df_cls.loc[BASELINE_EXP], cls)
        found_subjs.append(si)

    # Per-class summary
    print(f"\n  {'Class':<8} {'Mean recall':>12} {'SD':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*48}")
    mean_per_class = []
    for ci, cls in enumerate(CLASSES):
        vals = recall_matrix[found_subjs, ci]
        mean_per_class.append(np.mean(vals))
        print(f"  {cls:<8} {np.mean(vals):>12.3f} {np.std(vals, ddof=1):>8.3f} "
              f"{np.min(vals):>8.3f} {np.max(vals):>8.3f}")

    # Rank consistency: rank 1 = highest recall per subject, averaged across subjects
    print("\n  Rank consistency (1=best per subject, averaged across subjects):")
    print(f"  {'Class':<8} {'Mean rank':>10} {'SD rank':>9}")
    print(f"  {'-'*30}")
    rank_matrix = np.full_like(recall_matrix, np.nan)
    for si in found_subjs:
        row   = recall_matrix[si]
        valid = ~np.isnan(row)
        ranks = np.full(len(CLASSES), np.nan)
        ranks[valid] = stats.rankdata(-row[valid])
        rank_matrix[si] = ranks
    for ci, cls in enumerate(CLASSES):
        r = rank_matrix[found_subjs, ci]
        print(f"  {cls:<8} {np.mean(r):>10.2f} {np.std(r, ddof=1):>9.2f}")

    # Figure: bar chart + subject heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x  = np.arange(len(CLASSES))
    base_means = [np.mean(baseline_recall[found_subjs, ci]) for ci in range(len(CLASSES))]
    ax.bar(x - 0.2, mean_per_class, width=0.35, label="W4c est_with high_gamma",
           color="#2196F3", alpha=0.75)
    ax.bar(x + 0.2, base_means, width=0.35, label="Baseline (all bands)",
           color="#9E9E9E", alpha=0.75)
    for ci in range(len(CLASSES)):
        ax.errorbar(ci - 0.2, mean_per_class[ci],
                    yerr=np.std(recall_matrix[found_subjs, ci], ddof=1),
                    fmt="none", color="#1565C0", capsize=4, linewidth=1.5)
        ax.errorbar(ci + 0.2, base_means[ci],
                    yerr=np.std(baseline_recall[found_subjs, ci], ddof=1),
                    fmt="none", color="#424242", capsize=4, linewidth=1.5)
    ax.axhline(CHANCE, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({CHANCE:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Mean recall (SVM+RF)/2", fontsize=10)
    ax.set_title("Per-class recall: W4c_high_gamma vs baseline", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    ax2 = axes[1]
    hmap         = recall_matrix[found_subjs]
    subj_labels  = [SUBJECTS[si] for si in found_subjs]
    im = ax2.imshow(hmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels(CLASSES, fontsize=10)
    ax2.set_yticks(range(len(found_subjs)))
    ax2.set_yticklabels(subj_labels, fontsize=9)
    for si_local in range(len(found_subjs)):
        for ci in range(len(CLASSES)):
            val = hmap[si_local, ci]
            ax2.text(ci, si_local, f"{val:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if 0.25 < val < 0.80 else "white")
    plt.colorbar(im, ax=ax2, label="Recall")
    ax2.set_title("Per-subject per-class recall\n(W4c est_with high_gamma)", fontsize=10)

    plt.tight_layout()
    out_path = save_dir / "per_class_recall.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Analysis #5 — Consistency and reliability across subjects
# ---------------------------------------------------------------------------

KEY_CONDITIONS = [
    ("W4c", "high_gamma"),
    ("W4c", "gamma"),
    ("W4c", "wide_gamma"),
    ("W4c", "beta"),
]


def analysis_5_consistency(delta_df: pd.DataFrame, save_dir: Path):
    print_section("Analysis #5 — Consistency and reliability across subjects")

    cond_labels = [f"{w}_{b}" for w, b in KEY_CONDITIONS]
    cond_deltas = {}
    for w, b in KEY_CONDITIONS:
        key = f"{w}_{b}"
        cond_deltas[key] = delta_df.loc[key].values.astype(float) \
            if key in delta_df.index else np.full(len(SUBJECTS), np.nan)

    # Per-condition: N>0 + binomial test
    print(f"\n  {'Condition':<22} {'Mean Δ':>8} {'SD':>7} {'N>0':>6}  Binomial p")
    print(f"  {'-'*58}")
    from scipy.stats import binomtest
    for key in cond_labels:
        vals  = cond_deltas[key]
        valid = vals[~np.isnan(vals)]
        n_pos = int(np.sum(valid > 0))
        n_tot = len(valid)
        b_p   = binomtest(n_pos, n_tot, p=0.5, alternative="greater").pvalue
        sig   = "*" if b_p < 0.05 else ""
        print(f"  {key:<22} {np.mean(valid):>+8.4f} {np.std(valid, ddof=1):>7.4f} "
              f"{n_pos:>4}/{n_tot}  {b_p:.4f} {sig}")

    # Spearman ρ between key conditions
    print("\n  Spearman ρ between key conditions:")
    print(f"  {'Pair':<38} {'ρ':>7} {'p':>8}")
    print(f"  {'-'*56}")
    for i, ca in enumerate(cond_labels):
        for cb in cond_labels[i + 1:]:
            a, b   = cond_deltas[ca], cond_deltas[cb]
            valid  = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() >= 4:
                rho, p_r = stats.spearmanr(a[valid], b[valid])
                print(f"  {ca} vs {cb:<16} {rho:>7.3f} {p_r:>8.4f}")

    # Figure: per-subject strip plot for each key condition
    subj_markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]
    palette      = plt.cm.tab10(np.linspace(0, 0.6, len(SUBJECTS)))

    fig, axes = plt.subplots(1, len(KEY_CONDITIONS),
                             figsize=(3 * len(KEY_CONDITIONS), 5), sharey=True)
    for ax, (w, b) in zip(axes, KEY_CONDITIONS):
        key  = f"{w}_{b}"
        vals = cond_deltas[key]
        for si, val in enumerate(vals):
            if not np.isnan(val):
                ax.scatter(0, val, color=palette[si], marker=subj_markers[si],
                           s=70, zorder=3, label=SUBJECTS[si])
        valid = vals[~np.isnan(vals)]
        ax.bar(0, np.mean(valid), width=0.5, color="#4C72B0", alpha=0.2, zorder=2)
        ax.errorbar(0, np.mean(valid), yerr=np.std(valid, ddof=1),
                    color="#4C72B0", fmt="none", capsize=6, linewidth=2, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks([0])
        ax.set_xticklabels([key], fontsize=8, rotation=15, ha="right")
        ax.set_xlim(-0.6, 0.6)

    axes[0].set_ylabel("Δ mean bal acc (exp − baseline)", fontsize=10)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=8,
               title="Subject", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Per-subject Δ for key conditions", fontsize=11)
    plt.tight_layout()
    out_path = save_dir / "subject_consistency.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Overt analysis — Window × band comparison vs. baseline
# ---------------------------------------------------------------------------

OVERT_RESULTS_DIR  = RESULTS_DIR.parent / "overt_classification"
OVERT_BASELINE_EXP = "W1a_keepIC_zpower_full"
OVERT_WIN_PREFIXES = {"W2": "BandA", "W3": "BandB", "W4c": "BandD", "W4u": "BandC"}


def load_overt_subject(subject: str) -> pd.DataFrame:
    subj_dir = OVERT_RESULTS_DIR / subject
    baseline_csv = subj_dir / "baseline_windows" / f"{subject}_sp_classification_summary.csv"
    dfs = [pd.read_csv(baseline_csv).set_index("experiment")]
    for csv in (subj_dir / "band_sweep").glob(
            f"{subject}_sp_classification_summary_band_sweep_*.csv"):
        dfs.append(pd.read_csv(csv).set_index("experiment"))
    return pd.concat(dfs)


def overt_mean_bal(row: pd.Series) -> float:
    return (row["SVM_bal_acc"] + row["RF_bal_acc"]) / 2


def analysis_overt_window_band(save_dir: Path):
    print_section("Analysis OVERT — Window × band vs. baseline (FDR corrected)")

    rows = []
    for subj in SUBJECTS:
        try:
            df = load_overt_subject(subj)
        except FileNotFoundError:
            print(f"  [skip] {subj} — overt results not found")
            continue

        if OVERT_BASELINE_EXP not in df.index:
            print(f"  [skip] {subj} — baseline experiment not found")
            continue
        baseline_bal = overt_mean_bal(df.loc[OVERT_BASELINE_EXP])

        for win, prefix in OVERT_WIN_PREFIXES.items():
            for band in FOCUS_BANDS:
                matches = [e for e in df.index if e.startswith(f"{prefix}_{band}_keepIC")]
                if not matches:
                    continue
                bal = overt_mean_bal(df.loc[matches[0]])
                rows.append({
                    "subject": subj, "window": win, "band": band,
                    "bal_acc": bal, "baseline": baseline_bal,
                    "delta": bal - baseline_bal,
                })

    data_df = pd.DataFrame(rows)

    # Collect stats for FDR
    labels, mean_bals, mean_deltas, sds, ds, n_pos_list, pvals = [], [], [], [], [], [], []

    for win in WINDOWS:
        for band in FOCUS_BANDS:
            subset = data_df[(data_df.window == win) & (data_df.band == band)]
            if len(subset) < 3:
                continue
            delta = subset["delta"].values
            bal   = subset["bal_acc"].values
            n_pos = int((delta > 0).sum())
            d     = cohens_d_paired(delta)
            _, p  = stats.wilcoxon(delta, alternative="greater")
            lbl   = f"{win}_{band}"
            labels.append(lbl)
            mean_bals.append(np.mean(bal))
            mean_deltas.append(np.mean(delta))
            sds.append(np.std(delta, ddof=1))
            ds.append(d)
            n_pos_list.append(n_pos)
            pvals.append(p)

    padj = fdr_bh(np.array(pvals))

    # Baseline row for reference
    baseline_vals = data_df.drop_duplicates("subject")["baseline"].values
    print("\n  Baseline (W1a, full epoch, all bands)")
    print(f"  Mean bal acc = {np.mean(baseline_vals):.4f}  SD = {np.std(baseline_vals, ddof=1):.4f}\n")

    print(f"  {'Condition':<20} {'Bal acc':>8} {'Mean Δ':>8} {'SD':>7} {'N>base':>7} {'Cohen d':>8} {'p(Wilc)':>9} {'p(FDR)':>9}  sig")
    print(f"  {'-'*95}")

    prev_win = None
    for lbl, bal, m, sd, d, n_pos, p, pa in zip(
            labels, mean_bals, mean_deltas, sds, ds, n_pos_list, pvals, padj):
        win = lbl.split("_")[0]
        if win != prev_win:
            print(f"  --- {win} ---")
            prev_win = win
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else ("†" if p < 0.05 else "")
        print(f"  {lbl:<20} {bal:>8.4f} {m:>+8.4f} {sd:>7.4f} {n_pos:>5}/{len(SUBJECTS)} {d:>8.3f} {p:>9.4f} {pa:>9.4f}  {sig}")

    # True best per subject — search all bands in all loaded CSVs
    print("\n  True best per subject (all bands, all windows):")
    all_rows = []
    for subj in SUBJECTS:
        try:
            df = load_overt_subject(subj)
        except FileNotFoundError:
            continue
        if OVERT_BASELINE_EXP not in df.index:
            continue
        baseline_bal = overt_mean_bal(df.loc[OVERT_BASELINE_EXP])
        for win, prefix in OVERT_WIN_PREFIXES.items():
            matches = [e for e in df.index if e.startswith(prefix + "_")]
            for exp in matches:
                band_name = exp.replace(f"{prefix}_", "").split("_keepIC")[0]
                bal = overt_mean_bal(df.loc[exp])
                all_rows.append({
                    "subject": subj, "window": win, "band": band_name,
                    "experiment": exp, "bal_acc": bal,
                    "delta": bal - baseline_bal,
                })
    all_df = pd.DataFrame(all_rows)
    true_best = all_df.loc[all_df.groupby("subject")["bal_acc"].idxmax()]
    for _, r in true_best.iterrows():
        print(f"    {r.subject}:  {r.window}_{r.band}  bal_acc={r.bal_acc:.4f}  Δ={r.delta:+.4f}")

    # Figure
    fig, axes = plt.subplots(1, len(FOCUS_BANDS), figsize=(4 * len(FOCUS_BANDS), 5), sharey=True)
    for ax, band in zip(axes, FOCUS_BANDS):
        vals = [
            data_df[(data_df.window == w) & (data_df.band == band)]["delta"].values
            for w in WINDOWS
        ]
        ax.bar(WINDOWS, [np.mean(v) for v in vals],
                    color=[WIN_COLORS[w] for w in WINDOWS],
                    yerr=[np.std(v, ddof=1) for v in vals],
                    capsize=4, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(band.replace("_", " "), fontsize=10)
        ax.set_xlabel("Window")
        if ax == axes[0]:
            ax.set_ylabel("Δ bal acc vs. baseline")
    fig.suptitle("Overt: Δ balanced accuracy vs. W1a baseline (est_with focus bands)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / "overt_window_band_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Figure saved → overt_window_band_comparison.png")


# ---------------------------------------------------------------------------
# Analysis #6 — Feature count vs. Δbal_acc (dimensionality confound check)
# ---------------------------------------------------------------------------

def analysis_6_feature_count(save_dir: Path):
    print_section("Analysis #6 — Feature count vs. Δ balanced accuracy")

    # Build full experiment table across all subjects
    rows = []
    for subj in SUBJECTS:
        df = load_all_exps(subj)
        df = df.set_index("experiment")
        if BASELINE_EXP not in df.index:
            continue
        baseline_svm  = float(df.loc[BASELINE_EXP, "SVM_bal_acc"])
        baseline_rf   = float(df.loc[BASELINE_EXP, "RF_bal_acc"])
        baseline_mean = (baseline_svm + baseline_rf) / 2

        for exp, row in df.iterrows():
            if exp == BASELINE_EXP:
                continue
            svm  = float(row["SVM_bal_acc"])
            rf   = float(row["RF_bal_acc"])
            mean = (svm + rf) / 2
            rows.append({
                "subject":    subj,
                "experiment": exp,
                "n_features": int(row["n_features"]),
                "delta_svm":  svm  - baseline_svm,
                "delta_rf":   rf   - baseline_rf,
                "delta_mean": mean - baseline_mean,
            })

    data = pd.DataFrame(rows)

    # Spearman correlations
    print("\n  Spearman ρ (n_features vs. Δbal_acc):")
    for col, label in [("delta_svm", "SVM"), ("delta_rf", "RF"), ("delta_mean", "Mean")]:
        r, p = stats.spearmanr(data["n_features"], data[col])
        print(f"    {label:<6}: ρ = {r:+.3f},  p = {p:.4f}")

    # Highlighted conditions: W4c × {high_gamma, gamma, wide_gamma}
    hl_specs = [
        ("W4c_high_gamma", "#E74C3C", "W4c_est_with_high_gamma_all_keepIC"),
        ("W4c_gamma",      "#3498DB", "W4c_est_with_gamma_all_keepIC"),
        ("W4c_wide_gamma", "#2ECC71", "W4c_est_with_wide_gamma_all_keepIC"),
    ]
    hl_data = {lbl: (data[data["experiment"].str.startswith(pfx)], col)
               for lbl, col, pfx in hl_specs}

    # Three scatter plots
    configs = [
        ("delta_svm",  "SVM only",        "Δ SVM bal acc vs. baseline"),
        ("delta_rf",   "RF only",         "Δ RF bal acc vs. baseline"),
        ("delta_mean", "Mean (SVM+RF)/2", "Δ Mean bal acc vs. baseline"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (col, title, ylabel) in zip(axes, configs):
        ax.scatter(data["n_features"], data[col],
                   color="grey", alpha=0.15, s=8, linewidths=0, zorder=1)
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        for lbl, (subset, color) in hl_data.items():
            if not subset.empty:
                ax.scatter(subset["n_features"], subset[col],
                           color=color, s=70, zorder=3,
                           label=lbl, edgecolors="white", linewidths=0.5)
        r, p = stats.spearmanr(data["n_features"], data[col])
        ax.text(0.97, 0.97, f"ρ={r:+.3f}, p={p:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.set_xlabel("n_features")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Feature count vs. Δ balanced accuracy (covert, all experiments × subjects)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / "feature_count_vs_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Figure saved → feature_count_vs_delta.png")

    # ------------------------------------------------------------------
    # Overt version
    # ------------------------------------------------------------------
    print("\n  --- Overt ---")
    ov_rows = []
    for subj in SUBJECTS:
        try:
            df = load_overt_subject(subj)
        except FileNotFoundError:
            continue
        if OVERT_BASELINE_EXP not in df.index:
            continue
        base_svm  = float(df.loc[OVERT_BASELINE_EXP, "SVM_bal_acc"])
        base_rf   = float(df.loc[OVERT_BASELINE_EXP, "RF_bal_acc"])
        base_mean = (base_svm + base_rf) / 2
        for exp, row in df.iterrows():
            if exp == OVERT_BASELINE_EXP:
                continue
            svm  = float(row["SVM_bal_acc"])
            rf   = float(row["RF_bal_acc"])
            mean = (svm + rf) / 2
            ov_rows.append({
                "subject":    subj,
                "experiment": exp,
                "n_features": int(row["n_features"]),
                "delta_svm":  svm  - base_svm,
                "delta_rf":   rf   - base_rf,
                "delta_mean": mean - base_mean,
            })
    ov_data = pd.DataFrame(ov_rows)

    print("\n  Spearman ρ (n_features vs. Δbal_acc) — overt:")
    for col, label in [("delta_svm", "SVM"), ("delta_rf", "RF"), ("delta_mean", "Mean")]:
        r, p = stats.spearmanr(ov_data["n_features"], ov_data[col])
        print(f"    {label:<6}: ρ = {r:+.3f},  p = {p:.4f}")

    ov_hl_specs = [
        ("W4c_high_gamma", "#E74C3C", "BandD_high_gamma_keepIC"),
        ("W4c_gamma",      "#3498DB", "BandD_gamma_keepIC"),
        ("W4c_wide_gamma", "#2ECC71", "BandD_wide_gamma_keepIC"),
    ]
    ov_hl_data = {lbl: (ov_data[ov_data["experiment"].str.startswith(pfx)], col)
                  for lbl, col, pfx in ov_hl_specs}

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (col, title, ylabel) in zip(axes2, configs):
        ax.scatter(ov_data["n_features"], ov_data[col],
                   color="grey", alpha=0.15, s=8, linewidths=0, zorder=1)
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        for lbl, (subset, color) in ov_hl_data.items():
            if not subset.empty:
                ax.scatter(subset["n_features"], subset[col],
                           color=color, s=70, zorder=3,
                           label=lbl, edgecolors="white", linewidths=0.5)
        r, p = stats.spearmanr(ov_data["n_features"], ov_data[col])
        ax.text(0.97, 0.97, f"ρ={r:+.3f}, p={p:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.set_xlabel("n_features")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig2.suptitle("Feature count vs. Δ balanced accuracy (overt, all experiments × subjects)", fontsize=11)
    plt.tight_layout()
    fig2.savefig(save_dir / "feature_count_vs_delta_overt.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("\n  Figure saved → feature_count_vs_delta_overt.png")


# ---------------------------------------------------------------------------
# Per-subject classifier summary — SVM vs RF, covert and overt
# ---------------------------------------------------------------------------

def _fmt_params(params_str: str) -> str:
    """Extract key hyperparameter values from the stored params string."""
    import ast
    try:
        p = ast.literal_eval(str(params_str))
    except Exception:
        return str(params_str)
    return "  ".join(f"{k}={v}" for k, v in p.items())


def _print_best(label: str, exp: str, svm: float, rf: float, mean: float,
                svm_params: str, rf_params: str):
    print(f"  {label}")
    print(f"    Experiment : {exp}")
    print(f"    SVM bal acc: {svm:.4f}    params: {_fmt_params(svm_params)}")
    print(f"    RF  bal acc: {rf:.4f}    params: {_fmt_params(rf_params)}")
    print(f"    Mean bal acc: {mean:.4f}")


FOCUS_BANDS_ORDERED = ["high_gamma", "gamma", "wide_gamma", "beta", "alpha", "theta"]


def _get_w4c_exp(w4c_df: pd.DataFrame, band: str) -> pd.Series | None:
    """Return the est_with single-band W4c row for the given band, or None."""
    mask = (w4c_df["experiment"].str.contains("est_with") &
            w4c_df["experiment"].str.contains(f"_{band}_") &
            ~w4c_df["experiment"].str.contains("all_bands|alpha_beta|beta_wide"))
    rows = w4c_df[mask]
    if rows.empty:
        return None
    rows = rows.copy()
    rows["mean_bal"] = (rows["SVM_bal_acc"] + rows["RF_bal_acc"]) / 2
    return rows.loc[rows["mean_bal"].idxmax()]


_OVERT_BAND_TO_WIN = {"BandA": "W2", "BandB": "W3", "BandC": "W4u", "BandD": "W4c"}


def _shorten_exp(exp: str) -> str:
    """Compact experiment label for table display."""
    for band, win in _OVERT_BAND_TO_WIN.items():
        exp = exp.replace(band, win)
    exp = re.sub(r"_all_keepIC", "", exp)
    exp = re.sub(r"_zpower", "", exp)
    exp = re.sub(r"_est_with", "", exp)
    exp = re.sub(r"_keepIC", "", exp)
    exp = re.sub(r"_total\d+ms", "", exp)
    exp = re.sub(r"_pre\d+ms(_speech\d+ms)?", "", exp)
    exp = re.sub(r"_vel\d+ms_bil\d+ms_alv\d+ms", "", exp)
    return exp


def report_per_subject_classifiers():
    print_section("Per-subject classifier summary")

    # Collect data for all subjects first
    rows = []
    for subj in SUBJECTS:
        row = {"subj": subj}
        try:
            cov_df = load_all_exps(subj).set_index("experiment")
            cov_df["mean_bal"] = (cov_df["SVM_bal_acc"] + cov_df["RF_bal_acc"]) / 2
            base_mean = float(cov_df.loc[BASELINE_EXP, "mean_bal"]) if BASELINE_EXP in cov_df.index else np.nan
            row["cov_base"] = base_mean

            best_cov = cov_df.loc[cov_df["mean_bal"].idxmax()]
            row["cov_best_exp"]   = _shorten_exp(best_cov.name)
            row["cov_best_mean"]  = float(best_cov["mean_bal"])
            row["cov_best_delta"] = float(best_cov["mean_bal"]) - base_mean

            w4c_df = pd.read_csv(
                RESULTS_DIR / subj / "covert_classification"
                / f"{subj}_im_classification_summary_W4c.csv"
            )
            w4c_df["mean_bal"] = (w4c_df["SVM_bal_acc"] + w4c_df["RF_bal_acc"]) / 2
            for band in ["high_gamma", "wide_gamma"]:
                r = _get_w4c_exp(w4c_df, band)
                row[f"cov_w4c_{band}_mean"]  = float(r["mean_bal"]) if r is not None else np.nan
                row[f"cov_w4c_{band}_delta"] = float(r["mean_bal"]) - base_mean if r is not None else np.nan
        except Exception as e:
            row["cov_best_exp"] = f"ERR:{e}"

        try:
            ov_df = load_overt_subject(subj)
            ov_base = float(overt_mean_bal(ov_df.loc[OVERT_BASELINE_EXP])) if OVERT_BASELINE_EXP in ov_df.index else np.nan
            row["ov_base"] = ov_base

            ov_rows = []
            for win, prefix in OVERT_WIN_PREFIXES.items():
                for exp in [e for e in ov_df.index if e.startswith(prefix + "_")]:
                    ov_rows.append({"experiment": exp,
                                    "mean_bal": overt_mean_bal(ov_df.loc[exp])})
            ov_best_df = pd.DataFrame(ov_rows)
            best_ov = ov_best_df.loc[ov_best_df["mean_bal"].idxmax()]
            row["ov_best_exp"]   = _shorten_exp(best_ov["experiment"])
            row["ov_best_mean"]  = float(best_ov["mean_bal"])
            row["ov_best_delta"] = float(best_ov["mean_bal"]) - ov_base

            for band, prefix in [("high_gamma", "BandD_high_gamma_keepIC"),
                                  ("wide_gamma",  "BandD_wide_gamma_keepIC")]:
                matches = [e for e in ov_df.index if e.startswith(prefix)]
                if matches:
                    mean_b = overt_mean_bal(ov_df.loc[matches[0]])
                    row[f"ov_w4c_{band}_mean"]  = mean_b
                    row[f"ov_w4c_{band}_delta"] = mean_b - ov_base
                else:
                    row[f"ov_w4c_{band}_mean"]  = np.nan
                    row[f"ov_w4c_{band}_delta"] = np.nan
        except Exception as e:
            row["ov_best_exp"] = f"ERR:{e}"

        rows.append(row)

    # --- Covert table ---
    print("\n  COVERT")
    hdr = f"  {'Subject':<10} {'Best experiment':<32} {'Best':>6} {'Δ best':>7} {'W4c_HG':>7} {'ΔHG':>7} {'W4c_WG':>7} {'ΔWG':>7}"
    print(f"\n{hdr}")
    print(f"  {'-'*86}")
    for r in rows:
        print(f"  {r['subj']:<10} {r.get('cov_best_exp','?'):<32} "
              f"{r.get('cov_best_mean', np.nan):>6.3f} {r.get('cov_best_delta', np.nan):>+7.3f} "
              f"{r.get('cov_w4c_high_gamma_mean', np.nan):>7.3f} {r.get('cov_w4c_high_gamma_delta', np.nan):>+7.3f} "
              f"{r.get('cov_w4c_wide_gamma_mean', np.nan):>7.3f} {r.get('cov_w4c_wide_gamma_delta', np.nan):>+7.3f}")

    # --- Overt table ---
    print("\n\n  OVERT")
    print(f"\n{hdr}")
    print(f"  {'-'*86}")
    for r in rows:
        print(f"  {r['subj']:<10} {r.get('ov_best_exp','?'):<32} "
              f"{r.get('ov_best_mean', np.nan):>6.3f} {r.get('ov_best_delta', np.nan):>+7.3f} "
              f"{r.get('ov_w4c_high_gamma_mean', np.nan):>7.3f} {r.get('ov_w4c_high_gamma_delta', np.nan):>+7.3f} "
              f"{r.get('ov_w4c_wide_gamma_mean', np.nan):>7.3f} {r.get('ov_w4c_wide_gamma_delta', np.nan):>+7.3f}")

    # --- Loss summary ---
    def _loss_summary(label: str, best_key: str, hg_key: str, wg_key: str):
        losses_hg = np.array([r.get(best_key, np.nan) - r.get(hg_key, np.nan) for r in rows])
        losses_wg = np.array([r.get(best_key, np.nan) - r.get(wg_key, np.nan) for r in rows])
        print(f"\n  {label} — loss vs individual best (best − W4c_band):")
        print(f"  {'Subject':<10} {'loss_HG':>9} {'loss_WG':>9}")
        print(f"  {'-'*30}")
        for r, lhg, lwg in zip(rows, losses_hg, losses_wg):
            print(f"  {r['subj']:<10} {lhg:>+9.3f} {lwg:>+9.3f}")
        valid_hg = losses_hg[~np.isnan(losses_hg)]
        valid_wg = losses_wg[~np.isnan(losses_wg)]
        print(f"  {'Mean':<10} {np.mean(valid_hg):>+9.3f} {np.mean(valid_wg):>+9.3f}")
        print(f"  {'SD':<10} {np.std(valid_hg, ddof=1):>9.3f} {np.std(valid_wg, ddof=1):>9.3f}")
        print(f"  {'Max loss':<10} {np.max(valid_hg):>+9.3f} {np.max(valid_wg):>+9.3f}")
        n_hg_best = int(np.sum(losses_hg <= 0.005))
        n_wg_best = int(np.sum(losses_wg <= 0.005))
        print(f"  W4c_HG ≈ best for {n_hg_best}/{len(rows)} subjects (loss ≤ 0.005)")
        print(f"  W4c_WG ≈ best for {n_wg_best}/{len(rows)} subjects (loss ≤ 0.005)")

    _loss_summary("COVERT", "cov_best_mean", "cov_w4c_high_gamma_mean", "cov_w4c_wide_gamma_mean")
    _loss_summary("OVERT",  "ov_best_mean",  "ov_w4c_high_gamma_mean",  "ov_w4c_wide_gamma_mean")


# ---------------------------------------------------------------------------
# W4c window parameters — loaded from overt W4_sweep CSVs
# ---------------------------------------------------------------------------

def load_w4c_windows() -> dict:
    windows = {}
    for subj in SUBJECTS:
        # Pre-onset times from overt W4_sweep
        csv = (OVERT_RESULTS_DIR / subj / "W4_sweep"
               / f"{subj}_sp_W4_recommended_pre_onsets.csv")
        df = pd.read_csv(csv).set_index("param")["value"]
        stop = int(df["consonant_stop_ms"])
        nasal = int(df["consonant_nasal_ms"])
        fric  = int(df["consonant_fricative_ms"])
        w4u_pre = int(df["best_overall_pre_onset_ms"])

        # W4c post-onset: back-calculate from experiment name (total - max_pre)
        w4c_csv = (RESULTS_DIR / subj / "covert_classification"
                   / f"{subj}_im_classification_summary_W4c.csv")
        w4c_exp = pd.read_csv(w4c_csv)["experiment"]
        w4c_total_str = next(
            (e for e in w4c_exp if "est_with_high_gamma_all_keepIC_total" in e), None
        )
        w4c_total = int(re.search(r"total(\d+)ms", w4c_total_str).group(1))
        w4c_post = w4c_total - max(stop, nasal, fric)

        # W4u speech window: from experiment name (pre…ms_speech…ms)
        w4u_csv = (RESULTS_DIR / subj / "covert_classification"
                   / f"{subj}_im_classification_summary_W4u.csv")
        w4u_exp = pd.read_csv(w4u_csv)["experiment"]
        w4u_exp_str = next(
            (e for e in w4u_exp if "est_with_high_gamma_all_keepIC" in e), None
        )
        w4u_speech = int(re.search(r"speech(\d+)ms", w4u_exp_str).group(1))
        w4u_total = w4u_pre + w4u_speech

        windows[subj] = {
            "velar": stop, "bilabial": nasal, "alveolar": fric,
            "w4c_post": w4c_post, "w4c_total": w4c_total,
            "w4u_pre": w4u_pre, "w4u_speech": w4u_speech, "w4u_total": w4u_total,
        }
    return windows


def analysis_w4c_windows():
    print_section("W4c Window Parameters — consonant-specific pre-onset (from W4_sweep CSVs)")

    w4c_windows = load_w4c_windows()
    groups = ["velar", "bilabial", "alveolar"]

    # W4c table: per-consonant pre + post (post varies to keep total fixed)
    print("\n  W4c: consonant-specific windows (total fixed per subject = max_pre + 300ms)")
    print(f"\n  {'Subject':<10} "
          f"{'Vel pre':>8} {'Vel post':>9} "
          f"{'Bil pre':>8} {'Bil post':>9} "
          f"{'Alv pre':>8} {'Alv post':>9} "
          f"{'Total':>7}")
    print(f"  {'-'*72}")
    for subj in SUBJECTS:
        w = w4c_windows[subj]
        t = w["w4c_total"]
        vel_post = t - w["velar"]
        bil_post = t - w["bilabial"]
        alv_post = t - w["alveolar"]
        note = " ← W4c=W4u" if w["velar"] == w["bilabial"] == w["alveolar"] else ""
        print(f"  {subj:<10} "
              f"{w['velar']:>8} {vel_post:>9} "
              f"{w['bilabial']:>8} {bil_post:>9} "
              f"{w['alveolar']:>8} {alv_post:>9} "
              f"{t:>7}{note}")

    print(f"\n  {'Mean pre':<10}", end="")
    for g in groups:
        vals = [w4c_windows[s][g] for s in SUBJECTS]
        print(f" {np.mean(vals):>8.1f} {'':>9}", end="")
    print()
    print(f"  {'Mean post':<10}", end="")
    for g in groups:
        post_vals = [w4c_windows[s]["w4c_total"] - w4c_windows[s][g] for s in SUBJECTS]
        print(f" {'':>8} {np.mean(post_vals):>9.1f}", end="")
    print()

    # Group-level trend across consonant groups
    group_means = {g: np.mean([w4c_windows[s][g] for s in SUBJECTS]) for g in groups}
    data = np.array([[w4c_windows[s][g] for g in groups] for s in SUBJECTS])
    f_stat, f_p = stats.friedmanchisquare(*[data[:, i] for i in range(3)])
    print(f"\n  Group-level pre-onset trend: velar={group_means['velar']:.0f}ms, "
          f"bilabial={group_means['bilabial']:.0f}ms, alveolar={group_means['alveolar']:.0f}ms")
    print(f"  Bilabial had more pre-onset included on average (+{group_means['bilabial']-group_means['alveolar']:.0f}ms vs alveolar, "
          f"+{group_means['bilabial']-group_means['velar']:.0f}ms vs velar); "
          f"Friedman χ²={f_stat:.3f}, p={f_p:.3f} (descriptive trend, not significant at N=10)")

    # W4u table: pre + speech (varies per subject)
    print("\n\n  W4u: uniform pre-onset + subject-specific speech window")
    print(f"\n  {'Subject':<10} {'Pre (ms)':>9} {'Speech (ms)':>12} {'Total (ms)':>11}")
    print(f"  {'-'*45}")
    for subj in SUBJECTS:
        w = w4c_windows[subj]
        print(f"  {subj:<10} {w['w4u_pre']:>9} {w['w4u_speech']:>12} {w['w4u_total']:>11}")
    print(f"\n  {'Mean':<10} {np.mean([w4c_windows[s]['w4u_pre'] for s in SUBJECTS]):>9.1f} "
          f"{np.mean([w4c_windows[s]['w4u_speech'] for s in SUBJECTS]):>12.1f} "
          f"{np.mean([w4c_windows[s]['w4u_total'] for s in SUBJECTS]):>11.1f}")

    flat = [s for s in SUBJECTS if len({w4c_windows[s][g] for g in groups}) == 1]
    if flat:
        verb = "have" if len(flat) > 1 else "has"
        print(f"\n  Note: {', '.join(flat)} {verb} identical pre-onset across all groups → W4c ≡ W4u.")

    # --- Do window parameters predict which band is best? ---
    print("\n\n  Are window parameters different for subjects where W4c high_gamma is best?")

    # Determine per-subject best W4c band (est_with only, single bands)
    FOCUS_BANDS = ["high_gamma", "gamma", "wide_gamma", "beta", "alpha", "theta"]
    best_bands = {}
    for subj in SUBJECTS:
        w4c_csv = (RESULTS_DIR / subj / "covert_classification"
                   / f"{subj}_im_classification_summary_W4c.csv")
        df = pd.read_csv(w4c_csv)
        mask = (df["experiment"].str.contains("est_with") &
                ~df["experiment"].str.contains("all_bands|alpha_beta|beta_wide"))
        df = df[mask].copy()
        df["mean_bal"] = (df["SVM_bal_acc"] + df["RF_bal_acc"]) / 2
        best_exp = df.loc[df["mean_bal"].idxmax(), "experiment"]
        best_bands[subj] = next((b for b in FOCUS_BANDS if f"_{b}_" in best_exp), "?")

    hg_best = [s for s in SUBJECTS if best_bands[s] == "high_gamma"]
    other   = [s for s in SUBJECTS if best_bands[s] != "high_gamma"]
    print(f"\n  high_gamma best (N={len(hg_best)}): {', '.join(hg_best)}")
    print(f"  other best    (N={len(other)}): " +
          ", ".join(f"{s}({best_bands[s]})" for s in other))

    # Compare window metrics between the two groups
    metrics = {
        "total (ms)":    lambda s: w4c_windows[s]["w4c_total"],
        "velar pre":     lambda s: w4c_windows[s]["velar"],
        "bilabial pre":  lambda s: w4c_windows[s]["bilabial"],
        "alveolar pre":  lambda s: w4c_windows[s]["alveolar"],
        "pre range":     lambda s: max(w4c_windows[s][g] for g in groups) - min(w4c_windows[s][g] for g in groups),
    }

    print(f"\n  {'Metric':<16} {'HG-best mean':>13} {'Other mean':>11} {'U':>6} {'p':>8}  sig")
    print(f"  {'-'*60}")
    for label, fn in metrics.items():
        hg_vals   = [fn(s) for s in hg_best]
        oth_vals  = [fn(s) for s in other]
        u, p = stats.mannwhitneyu(hg_vals, oth_vals, alternative="two-sided")
        sig = "*" if p < 0.05 else ("†" if p < 0.10 else "")
        print(f"  {label:<16} {np.mean(hg_vals):>13.1f} {np.mean(oth_vals):>11.1f} "
              f"{u:>6.0f} {p:>8.4f}  {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analysis_1_baseline_vs_chance()
    delta_df, svm_delta_df, rf_delta_df, pct_df = analysis_2_delta_matrix()
    analysis_2_paired_tests(delta_df, pct_df)
    save_dir = RESULTS_DIR.parent / "group_analysis_figures"
    save_dir.mkdir(exist_ok=True)
    analysis_2_window_summary(delta_df, save_dir)
    analysis_3_template_comparison(save_dir)
    analysis_4_per_class_recall(save_dir)
    analysis_5_consistency(delta_df, save_dir)
    analysis_overt_window_band(save_dir)
    analysis_w4c_windows()
    analysis_6_feature_count(save_dir)
    report_per_subject_classifiers()
