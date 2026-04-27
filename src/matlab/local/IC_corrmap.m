%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================
SUBJ            = 'subj-02';
OVERT_KEEP_ICS  = [2 3 4 5 7 12 14 20 21 22 23 26 27 28 29 30 31 32 33 36 38];
COVERT_KEEP_ICS = [3 4 5 7 13 14 15 16 18 19 20 22 23 24 25 28 29 34 36 39 42 46];
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
            'ics',           1, ...       % max 1 match per dataset
            'resetclusters', 'off', ...
            'pl',            'none');

        matched_sets = CORRMAP.corr.sets{1};
        matched_ics  = CORRMAP.corr.ics{1};

        overt_match_idx = find(matched_sets == TARGET_SET_NUM);

        if ~isempty(overt_match_idx)
            % Raw index within subsetted overt dataset (1..N)
            matched_sub_idx = matched_ics(overt_match_idx);

            % Remap to original IC number in full overt dataset
            original_overt_ic = overt_ic_map(matched_sub_idx);

            overt_matched_ics_sub = [overt_matched_ics_sub, matched_sub_idx];
            overt_matched_ics     = [overt_matched_ics,     original_overt_ic];
            covert_matched_ics    = [covert_matched_ics,    template_ic];
            match_correlations    = [match_correlations, ...
                CORRMAP.corr.abs_values{1}(overt_match_idx)];

            fprintf('    → Match: covert IC %d ↔ overt IC %d  (subset idx=%d, corr=%.3f)\n', ...
                template_ic, original_overt_ic, matched_sub_idx, ...
                CORRMAP.corr.abs_values{1}(overt_match_idx));
        else
            fprintf('    → No match above threshold\n');
        end

    catch ME
        fprintf('    → CORRMAP error for IC %d: %s\n', template_ic, ME.message);
    end
end

%% =========================================================================
%  Print and save match summary
% ==========================================================================
fprintf('\n========================================================\n');
fprintf('  CORRMAP RESULTS — %s  (threshold = %.2f)\n', SUBJ, CORRMAP_THRESH);
fprintf('========================================================\n');
fprintf('  Total covert ICs tested:  %d\n', length(COVERT_KEEP_ICS));
fprintf('  Matches found:            %d\n', length(covert_matched_ics));
fprintf('\n');
fprintf('  %-15s %-22s %-15s\n', 'Covert IC', 'Overt IC (original)', 'Correlation');
fprintf('  %s\n', repmat('-', 1, 54));
for i = 1:length(covert_matched_ics)
    fprintf('  %-15d %-22d %.3f\n', ...
        covert_matched_ics(i), overt_matched_ics(i), match_correlations(i));
end

fprintf('\n  → Visually verify each matched pair in EEGLAB.\n');
fprintf('    Enter final matches into subject_config.csv.\n');
fprintf('========================================================\n\n');

% Save CSV — includes both original IC numbers and subset indices for traceability
output_dir  = fullfile(BASE_PATH, '04_processed');
output_file = fullfile(output_dir, [SUBJ, '_corrmap_matches.csv']);
fid = fopen(output_file, 'w');
fprintf(fid, 'covert_ic,overt_ic_original,overt_ic_subset_idx,correlation\n');
for i = 1:length(covert_matched_ics)
    fprintf(fid, '%d,%d,%d,%.4f\n', ...
        covert_matched_ics(i), overt_matched_ics(i), ...
        overt_matched_ics_sub(i), match_correlations(i));
end
fclose(fid);
fprintf('  Match table saved: %s\n\n', output_file);