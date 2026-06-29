%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================

% SUBJ = 'subj-01';
% OVERT_KEEP_ICS  = [3 4 5 6 10 13 18 19 20 21];
% COVERT_KEEP_ICS = [1 2 3 4 5 6 7 9 12 13 15 18 20 21 23 24 25 27];

% SUBJ            = 'subj-02';
% OVERT_KEEP_ICS  = [2 3 4 5 7 12 14 20 21 22 23 26 27 28 29 30 31 32 33 36 38];
% COVERT_KEEP_ICS = [3 4 5 7 13 14 15 16 18 19 20 22 23 24 25 28 29 34 36 39 42 46];

% SUBJ            = 'subj-03';
% OVERT_KEEP_ICS  = [3 6 16 19 21 31];
% COVERT_KEEP_ICS = [4 5 8 9 10 16 17 22 35 43];

% SUBJ            = 'subj-04';
% OVERT_KEEP_ICS  = [1 2 3 5 6 7 8 10 15 17 18 20 22 23 24 25];
% COVERT_KEEP_ICS = [1 2 3 4 5 7 8 10 11 12 15 17 24];

% SUBJ            = 'subj-05';
% OVERT_KEEP_ICS  = [4 16 17 19 20 21 22 25 34 39 46 48];
% COVERT_KEEP_ICS = [2 3 10 11 14 17 18 20 23 25 28 29 30];

% SUBJ = 'subj-06';
% OVERT_KEEP_ICS  = [2 3 4 5 8 9 10 11 12 13 14 15 16 17 24 29 30 31 32 33 34 36 37 39 40 42 44 45 48 49 50 53];
% COVERT_KEEP_ICS = [2 3 4 5 6 7 8 9 10 12 13 17 18 20 22 27 30 41];
% 
% SUBJ = 'subj-07';
% OVERT_KEEP_ICS  = [1 2 3 4 9 11 13 15 21 25 26 30 31 35];
% COVERT_KEEP_ICS = [2 4 6 8 9 10 11 12 13 14 18 21 22 23 26 32 35 36 38 39 40 42 44 46];
% 
% SUBJ = 'subj-08';
% OVERT_KEEP_ICS  = [5 6 7 11 12 13 18 19 22 24 28];
% COVERT_KEEP_ICS = [2 5 6 11 13 14 15 16 18 25 27 29 34 38 41];

SUBJ = 'subj-11';
OVERT_KEEP_ICS  = [5 11 16 17 19 25 27];
COVERT_KEEP_ICS = [1 2 4 5 6 7 10 11 12 13 15 18 21 22 29 30 31 32 36 38];
% 
% SUBJ = 'subj-12';
% OVERT_KEEP_ICS  = [1 2 3 4 7 8 13 14 15 19 22 24 25 26 29 30 31 35];
% COVERT_KEEP_ICS = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 19 24 25 26 27 34 36 38 39 40 45];



CORRMAP_THRESH  = 0.6;   % topology correlation threshold

BASE_PATH     = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/data';
SPOKEN_PATH   = fullfile(BASE_PATH, '03_interim_cluster', SUBJ, 'spoken',   'datasets');
IMAGINED_PATH = fullfile(BASE_PATH, '03_interim_cluster', SUBJ, 'imagined', 'datasets');

eeglab;
global ALLEEG EEG CURRENTSET CURRENTSTUDY STUDY;

fprintf('\n========================================================\n');
fprintf('  Subject:          %s\n', SUBJ);
fprintf('  Spoken data dir:  %s\n', SPOKEN_PATH);
fprintf('  Imagined data dir:%s\n', IMAGINED_PATH);
fprintf('  CORRMAP threshold:%.2f\n', CORRMAP_THRESH);
fprintf('========================================================\n\n');

%% =========================================================================
%  Load datasets
% ==========================================================================
speech_types = {'sp', 'im'};
file_paths   = cell(1, 2);

for cond_idx = 1:2
    if strcmp(speech_types{cond_idx}, 'sp')
        INPUT_DIR = SPOKEN_PATH;
    else
        INPUT_DIR = IMAGINED_PATH;
    end

    setname  = [SUBJ, '_pilot_', speech_types{cond_idx}, '_cleaned_2ndICA_epoched'];
    filename = [setname, '.set'];
    file_paths{cond_idx} = fullfile(INPUT_DIR, filename);

    fprintf('Loading dataset: %s\n', filename);
    EEG = pop_loadset('filename', filename, 'filepath', INPUT_DIR);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    fprintf('  → Loaded: %s\n',       EEG.setname);
    fprintf('  → Channels:       %d\n',   EEG.nbchan);
    fprintf('  → ICA components: %d\n',   size(EEG.icaweights, 1));
    fprintf('  → Trials:         %d\n',   EEG.trials);
    fprintf('  → Epoch length:   %.0f samples (%.3f s)\n', EEG.pnts, EEG.pnts / EEG.srate);
    fprintf('  → Sampling rate:  %.0f Hz\n\n', EEG.srate);
end

%% =========================================================================
%  Create subsetted overt dataset containing only OVERT_KEEP_ICS
%
%  CORRMAP scans all components in a dataset regardless of which components
%  are selected in the STUDY. To restrict matching to OVERT_KEEP_ICS only,
%  we create a new dataset with those ICs extracted via pop_subcomp.
%
%  pop_subcomp renumbers kept ICs as 1..N in sorted order, so we track:
%    new index k → original IC number = overt_ic_map(k) = sort(OVERT_KEEP_ICS)(k)
% ==========================================================================
fprintf('Creating subsetted overt dataset (OVERT_KEEP_ICS only)...\n');

EEG_overt_full = ALLEEG(1);   % full overt dataset

% pop_subcomp takes a list of components TO REMOVE
all_comps    = 1:size(EEG_overt_full.icaweights, 1);
comps_remove = setdiff(all_comps, OVERT_KEEP_ICS);

EEG_overt_sub          = pop_subcomp(EEG_overt_full, comps_remove, 0);
EEG_overt_sub.setname  = [SUBJ, '_sp_corrmap_subset'];

% Resave as dataset 1 in ALLEEG
%   ALLEEG(1) = subsetted overt 
%   ALLEEG(2) = full covert
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG_overt_sub, 1);
OVERT_SUB_SET_NUM = CURRENTSET;   % = 1

fprintf('  → Subsetted overt: %d ICs (from %d total)\n', ...
    size(EEG_overt_sub.icaweights, 1), size(EEG_overt_full.icaweights, 1));
fprintf('  → Stored as ALLEEG(%d)\n\n', OVERT_SUB_SET_NUM);

% Mapping from subsetted IC index back to original IC number.
% pop_subcomp keeps ICs in sorted order, so:
%   subsetted IC 1 = original IC overt_ic_map(1), etc.
overt_ic_map = sort(OVERT_KEEP_ICS);

%% =========================================================================
%  Create STUDY using subsetted overt dataset
%  Study set 1 = subsetted overt  (ALLEEG(OVERT_SUB_SET_NUM))
%  Study set 2 = covert           (ALLEEG(2))
% ==========================================================================
STUDY = [];

% Build study pointing to the subsetted overt dataset in memory.
% We pass the subsetted EEG directly via eeg_store and assign dataset index.
[STUDY, ALLEEG] = std_editset(STUDY, ALLEEG, ...
    'name',     [SUBJ, '_spoken_vs_imagined'], ...
    'commands', { ...
        {'index', 1, 'subject', SUBJ, 'condition', 'spoken_sub', ...
         'comps', 1:length(OVERT_KEEP_ICS)}, ...
        {'index', 2, 'subject', SUBJ, 'condition', 'imagined', ...
         'comps', COVERT_KEEP_ICS}, ...
    }, ...
    'updatedat', 'off');

[STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);
CURRENTSTUDY = 1;
EEG          = ALLEEG;
CURRENTSET   = 1:length(ALLEEG);
eeglab redraw;
fprintf('STUDY created: %s\n\n', STUDY.name);

%% =========================================================================
%  Run CORRMAP
%  Template: each covert keep IC (study set index 2)
%  Target:   subsetted overt     (study set index 1)
%
%  CORRMAP returns IC indices in the subsetted overt (1..N).
%  We remap to original IC numbers via overt_ic_map.
% ==========================================================================
TEMPLATE_SET_NUM = 2;   % covert
TARGET_SET_NUM   = 1;   % subsetted overt

overt_matched_ics     = [];   % original overt IC numbers (remapped)
overt_matched_ics_sub = [];   % subsetted overt IC indices (raw from CORRMAP)
covert_matched_ics    = [];   % covert template IC numbers
match_correlations    = [];

fprintf('Running CORRMAP (threshold = %.2f)...\n', CORRMAP_THRESH);
fprintf('Template: study set %d (covert) | Target: study set %d (subsetted overt)\n\n', ...
    TEMPLATE_SET_NUM, TARGET_SET_NUM);

for i = 1:length(COVERT_KEEP_ICS)
    template_ic = COVERT_KEEP_ICS(i);
    fprintf('  Testing covert IC %d (%d/%d)...\n', ...
        template_ic, i, length(COVERT_KEEP_ICS));

    try
        CORRMAP = corrmap(STUDY, ALLEEG, TEMPLATE_SET_NUM, template_ic, ...
            'th',            num2str(CORRMAP_THRESH), ...
            'ics',           2, ...       % return up to 2 matches per dataset
            'resetclusters', 'off', ...
            'pl',            'none');

        if ~isfield(CORRMAP, 'corr') || isempty(CORRMAP.corr.sets{1})
            fprintf('    → No match above threshold\n');
            continue;
        end

        matched_sets = CORRMAP.corr.sets{1};
        matched_ics  = CORRMAP.corr.ics{1};
        matched_corr = CORRMAP.corr.abs_values{1};

        % Filter to overt set and above threshold
        valid = (matched_sets == TARGET_SET_NUM) & (matched_corr >= CORRMAP_THRESH);
        matched_ics  = matched_ics(valid);
        matched_corr = matched_corr(valid);

        if isempty(matched_ics)
            fprintf('    → No match above threshold in overt set\n');
            continue;
        end

        % Sort by correlation descending, take top 2
        [sorted_corr, order] = sort(matched_corr, 'descend');
        sorted_ics = matched_ics(order);
        n_report = min(2, length(sorted_ics));

        for r = 1:n_report
            matched_sub_idx   = sorted_ics(r);
            original_overt_ic = overt_ic_map(matched_sub_idx);

            overt_matched_ics_sub = [overt_matched_ics_sub, matched_sub_idx];
            overt_matched_ics     = [overt_matched_ics,     original_overt_ic];
            covert_matched_ics    = [covert_matched_ics,    template_ic];
            match_correlations    = [match_correlations,    sorted_corr(r)];

            fprintf('    → Rank %d: covert IC %d ↔ overt IC %d  (corr=%.3f)\n', ...
                r, template_ic, original_overt_ic, sorted_corr(r));
        end

    catch ME
        fprintf('    → CORRMAP error for IC %d: %s\n', template_ic, ME.message);
    end
end

%% =========================================================================
%  Print and save match summary (sorted by correlation, highest → lowest)
% ==========================================================================
fprintf('\n========================================================\n');
fprintf('  CORRMAP RESULTS — %s  (threshold = %.2f, up to 2 matches per covert IC)\n', SUBJ, CORRMAP_THRESH);
fprintf('========================================================\n');
fprintf('  Total covert ICs tested:  %d\n', length(COVERT_KEEP_ICS));
fprintf('  Covert ICs with ≥1 match: %d\n', length(unique(covert_matched_ics)));
fprintf('  Total match entries:      %d\n', length(covert_matched_ics));
fprintf('\n');

% Compute rank within each covert IC BEFORE sorting (matches are pushed
% in correlation-descending order per covert IC, so consecutive same-IC
% entries are already correctly ranked 1, 2, ...)
ranks       = zeros(1, length(covert_matched_ics));
prev_covert = -1;
rank        = 0;
for i = 1:length(covert_matched_ics)
    if covert_matched_ics(i) ~= prev_covert
        rank        = 1;
        prev_covert = covert_matched_ics(i);
    else
        rank = rank + 1;
    end
    ranks(i) = rank;
end

% Sort all entries by correlation descending
[~, sort_order]       = sort(match_correlations, 'descend');
covert_sorted         = covert_matched_ics(sort_order);
overt_sorted          = overt_matched_ics(sort_order);
overt_sub_sorted      = overt_matched_ics_sub(sort_order);
corr_sorted           = match_correlations(sort_order);
ranks_sorted          = ranks(sort_order);

% Print
fprintf('  %-12s %-6s %-22s %-12s\n', 'Covert IC', 'Rank', 'Overt IC (original)', 'Correlation');
fprintf('  %s\n', repmat('-', 1, 56));
for i = 1:length(covert_sorted)
    fprintf('  %-12d %-6d %-22d %.3f\n', ...
        covert_sorted(i), ranks_sorted(i), overt_sorted(i), corr_sorted(i));
end

fprintf('\n  → For covert ICs with 2 matches, visually compare scalp topographies\n');
fprintf('    in EEGLAB before selecting the final match.\n');
fprintf('    Enter chosen matches into subject_config.csv.\n');
fprintf('========================================================\n\n');

% Save CSV (same sorted order)
output_dir  = fullfile(BASE_PATH, '06_corrmap_IC_match');
output_file = fullfile(output_dir, [SUBJ, '_corrmap_matches.csv']);
fid = fopen(output_file, 'w');
fprintf(fid, 'covert_ic,rank,overt_ic_original,overt_ic_subset_idx,correlation\n');
for i = 1:length(covert_sorted)
    fprintf(fid, '%d,%d,%d,%d,%.4f\n', ...
        covert_sorted(i), ranks_sorted(i), overt_sorted(i), ...
        overt_sub_sorted(i), corr_sorted(i));
end
fclose(fid);
fprintf('  Match table saved: %s\n\n', output_file);