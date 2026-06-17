#!/usr/bin/env python3
"""
Derive CORRMAP-matched IC pairs for the covert classification pipeline.

Reads subject_config.csv (for brain_ics) and {subj}_corrmap_matches.csv,
applies threshold / rank / brain-IC filters, and prints two shell-eval-able
lines to stdout:

    OVERT_MATCHED="ic1 ic2 ..."
    COVERT_MATCHED="ic1 ic2 ..."

The two lists are parallel: OVERT_MATCHED[i] is the overt IC paired with
COVERT_MATCHED[i].  Multiple rows from the corrmap file can map to the same
overt IC (many covert ICs may match one overt IC), and vice versa.

Usage (from bash):
    eval $(python derive_covert_ic_matches.py \\
               --subj            subj-02 \\
               --corrmap-dir     .../data/06_corrmap_IC_match \\
               --config-csv      .../subject_config.csv \\
               --corr-threshold  0.8)
    # OVERT_MATCHED and COVERT_MATCHED are now set in the shell

Flags:
    --rank1-only      Keep only rank=1 matches (best match per covert IC).
                      Without this flag ALL rows above the threshold are kept.
    --brain-ics-only  Keep only pairs where both the covert IC is in im brain_ics
                      AND the overt IC is in sp brain_ics (from subject_config.csv).
"""
import argparse
import csv
import sys


def load_brain_ics(config_csv: str, subj: str) -> tuple[set[int], set[int]]:
    sp_brain: set[int] = set()
    im_brain: set[int] = set()
    with open(config_csv) as f:
        for row in csv.DictReader(f):
            if row['subject'] != subj:
                continue
            raw = row.get('brain_ics', '').strip()
            ics = {int(x) for x in raw.split()} if raw else set()
            if row['condition'] == 'sp':
                sp_brain = ics
            elif row['condition'] == 'im':
                im_brain = ics
    return sp_brain, im_brain


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Derive corrmap IC match pairs for the covert pipeline.')
    parser.add_argument('--subj',           required=True,
                        help='Subject ID, e.g. subj-02')
    parser.add_argument('--corrmap-dir',    required=True,
                        help='Directory containing {subj}_corrmap_matches.csv files')
    parser.add_argument('--config-csv',     required=True,
                        help='Path to subject_config.csv')
    parser.add_argument('--corr-threshold', type=float, required=True,
                        help='Minimum correlation to keep a match (e.g. 0.8)')
    parser.add_argument('--rank1-only',     action='store_true',
                        help='Keep only rank=1 (best) match per covert IC')
    parser.add_argument('--brain-ics-only', action='store_true',
                        help='Keep only pairs where both ICs are in brain_ics')
    args = parser.parse_args()

    sp_brain, im_brain = load_brain_ics(args.config_csv, args.subj)

    if args.brain_ics_only and (not sp_brain or not im_brain):
        print(f'[ERROR] --brain-ics-only requested but brain_ics are empty '
              f'for {args.subj} in {args.config_csv}', file=sys.stderr)
        sys.exit(1)

    corrmap_path = f'{args.corrmap_dir}/{args.subj}_corrmap_matches.csv'
    overt_list:  list[int] = []
    covert_list: list[int] = []

    try:
        with open(corrmap_path) as f:
            for row in csv.DictReader(f):
                if float(row['correlation']) < args.corr_threshold:
                    continue
                if args.rank1_only and int(row['rank']) != 1:
                    continue
                covert_ic = int(row['covert_ic'])
                overt_ic  = int(row['overt_ic_original'])
                if args.brain_ics_only:
                    if covert_ic not in im_brain or overt_ic not in sp_brain:
                        continue
                covert_list.append(covert_ic)
                overt_list.append(overt_ic)
    except FileNotFoundError:
        print(f'[ERROR] Corrmap file not found: {corrmap_path}', file=sys.stderr)
        sys.exit(1)

    if not overt_list:
        print(
            f'[ERROR] No IC matches found for {args.subj} with '
            f'corr>={args.corr_threshold}, rank1={args.rank1_only}, '
            f'brain_only={args.brain_ics_only}',
            file=sys.stderr)
        sys.exit(1)

    print(f'OVERT_MATCHED="{" ".join(str(x) for x in overt_list)}"')
    print(f'COVERT_MATCHED="{" ".join(str(x) for x in covert_list)}"')


if __name__ == '__main__':
    main()
