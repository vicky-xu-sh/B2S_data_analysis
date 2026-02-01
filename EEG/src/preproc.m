% Preprocessing

eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

% CHANGE THIS
SPEECH_TYPE = 'sp';
SUBJ = 'subj-02';
SESS = 'sess-02';

if SPEECH_TYPE == 'sp'
    dataset_path = [dataset_path,'/',SUBJ,'/',SESS,'/spoken'];  % make sure datapath exists
else 
    dataset_path = [dataset_path,'/',SUBJ,'/',SESS,'/imagined'];
end


if exist(dataset_path, 'dir') == 7
    disp('Path exists.');
else
    disp('Path does not exist.');
end
%% Import/load dataset

% === Import mff and save as dataset (Only run this if not saved as dataset) ===
% import mff and save as dataset
raw_eeg_datafile = '/Users/vickyxu/Desktop/B2S/raw_EEG_data/subj-02/sess-02/B2S_P02_JB_2025-11-27_imagine-1_20250625_101753.mff';
EEG = pop_mffimport({raw_eeg_datafile},{'classid','code','description','label','mffkeys','name'},0,0);
EEG.setname = [SUBJ,'_',SESS,'_pilot_',SPEECH_TYPE,'_raw'];
filename = [EEG.setname, '.set'];
EEG = pop_saveset(EEG, 'filename', filename, 'filepath', dataset_path);
eeglab redraw; % refresh GUI

% === Load the raw dataset ===
% filename = [SUBJ,'_',SESS,'_pilot_',SPEECH_TYPE,'_raw.set'];
% EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
% % updates data structure
% [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
% eeglab redraw; % refresh GUI
% EEG_orig = EEG; % keep a copy of the original

%% Import voice onset/offset events (run only for overt speech)
% Import events (commands not really working?)
audio_processing_path = '/Users/vickyxu/Desktop/B2S/B2S_EEG_Analysis/overt_audio_processing';
speech_events_path =[audio_processing_path,'/',SUBJ,'_',SESS,'/',SUBJ,'_',SESS,'_speech_events.txt'];
EEG = pop_importevent(EEG, 'event', speech_events_path,'fields',{'latency','type'},'timeunit',1, 'align', NaN, 'append', 'yes');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

% Check events
pop_eegplot(EEG, 1, 1, 1);

%% Import channel locations

EEG = pop_chanedit(EEG, 'load',{'/Users/vickyxu/Desktop/B2S/raw_EEG_data/subj-02/sess-02/11-27-2025_P02.sfp','filetype','sfp'});
EEG = pop_chanedit(EEG, 'changefield',{258,'labels','Nz'},'changefield',{259,'labels','LPA'},'changefield',{260,'labels','RPA'});
EEG = pop_chanedit(EEG, 'eval','chans = pop_chancenter( chans, [],[]);');   % optimize head centre to better plot on 2D
figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);

%% Delete bad channels (obvious channels not on the scalp, if needed)

% EEG = pop_select(EEG, 'nochannel', {'E102', 'E111', 'E120', 'E133', 'E122', 'E145', 'E165', 'E174', 'E187', 'E199', 'E208'});

%% Visualize raw dataset
figure;  title('Raw data power spectra');
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 200],...
    'electrodes','off');

%% Save set

setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_raw_edited'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

EEG_orig = EEG; % keep a copy of the original

%% Highpass at 1 or 2 Hz and remove line noise

if strcmp(SPEECH_TYPE, 'sp')
    hp_cutoff = 2;
else 
    hp_cutoff = 1;
end

EEG = pop_eegfiltnew(EEG, 'locutoff', hp_cutoff); 

% % Remove 60 Hz line noise and its harmonics
EEG = pop_eegfiltnew(EEG, 'locutoff',59,'hicutoff',61,'revfilt',1); % set 'plotfreqz' to 1 if want freq/phase response graph
EEG = pop_eegfiltnew(EEG, 'locutoff',119,'hicutoff',121,'revfilt',1);
EEG = pop_eegfiltnew(EEG, 'locutoff',179,'hicutoff',181,'revfilt',1);

% Visualize
figure; title(sprintf('Power spectra after %d Hz high-pass and line noise removed',hp_cutoff));
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 250],...
    'electrodes','off');

%% Remove bad channels 

EEG = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion',5, ...
    'ChannelCriterion',0.8, ...
    'LineNoiseCriterion',4, ...
    'Highpass','off', ...
    'BurstCriterion','off', ...
    'WindowCriterion','off', ...
    'BurstRejection','off', ...
    'Distance','Euclidian');

% % Visualize the removed channels
removed_chans = EEG_orig.chanlocs(~EEG.etc.clean_channel_mask); 
removed_labels = {removed_chans.labels};

% Create a subset EEG to visualize the removed channels
EEG_removed_only = pop_select(EEG_orig, 'channel', removed_labels);
pop_eegplot(EEG_removed_only, 1, 1, 1);  % Plot removed channels only 

figure; 
title('Power spectra after filtered and bad channels removed');
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 250],...
    'electrodes','off');

%% 

% save set
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_raw_',num2str(hp_cutoff),'hz_hp_badchan_removed'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI


%% Another 60Hz notch filter with wider passband (if needed)

EEG = pop_eegfiltnew(EEG, 'locutoff',58,'hicutoff',62,'revfilt',1);

% Visualize
figure; title(sprintf('Power spectra after a second wider 60Hz notch filter',hp_cutoff));
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 250],...
    'electrodes','off');

%% Re-reference to average and resample

EEG = pop_reref( EEG, []);
EEG = pop_resample( EEG, 500);

%% Manually inpspect the data and reject bad data segments

% Stack the channels to easily spot bad data segments

%% Define PCA function

function n_pca = run_PCA(EEG, run)
% Reshape EEG data to 2D (channels x time)
data = double(EEG.data);
data_2d = reshape(data, EEG.nbchan, []);

% Center data
data_2d = data_2d - mean(data_2d, 2);

% Run SVD (PCA)
[U, S, ~] = svd(data_2d, 'econ');
explained = diag(S).^2 / sum(diag(S).^2);
cumulative_explained = cumsum(explained);

n_pca = find(cumulative_explained >= 0.99, 1);
fprintf('Number of PCs that explain 99%% variance: %d\n', n_pca);

n_pca_999 = find(cumulative_explained >= 0.999, 1);
fprintf('Number of PCs that explain 99.9%% variance: %d\n', n_pca_999);

% output 99.9% variance number of pcs if not enough 99% pcs
if run == 1 && n_pca < 60
    n_pca = n_pca_999;
elseif run == 2 && n_pca < 30
    n_pca = n_pca_999;
end

% Plot cumulative explained variance
figure;
plot(cumulative_explained * 100, 'LineWidth', 2); % convert to percent
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA: Cumulative Explained Variance');
grid on;

hold on;
yline(99, '--r', '99% threshold');
hold on;
yline(99.9, '--g', '99.9% threshold');
end

%% Run PCA to get num PCs that explain 99% (or 99.9%) var

n_pca = run_PCA(EEG, 1);
disp(n_pca)

%% Run ICA (AMICA) 

% type “help runamica15()” for a full list and explanation of the parameters
% define parameters
numprocs = 1;       % # of nodes (default = 1)
max_threads = 4;    % # of threads per node
num_models = 1;     % # of models of mixture ICA
max_iter = 2000;    % max number of learning steps
% ===== EDIT THIS PARAMETER IF NEEDED ==== %
pcakeep = n_pca;       % EDIT NUM of PCs to keep

% run amica
outdir = [ pwd filesep 'amicaouttmp' filesep ];
[weights,sphere,mods] = runamica15(EEG.data, 'num_models',num_models, 'outdir',outdir, ...
    'numprocs', numprocs, 'max_threads', max_threads, 'max_iter',max_iter, 'pcakeep', pcakeep);

% Assign weights and sphere to EEG
EEG.icaweights = weights;
EEG.icasphere = sphere;

% Store number of components
EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;  % assuming all channels used
%% 

% save set
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_raw_',num2str(hp_cutoff),'hz_hp_badchan_removed_reref_resampled_seg_removed_PCA',num2str(pcakeep),'_AMICA'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Prepare analysis dataset (bandpass 1-200Hz)

% save ICA EEG set
EEG_ICA = EEG;

% Filtering Bandpass 1-200 Hz and remove line noise

% Band-pass at 1-200 Hz
EEG = pop_eegfiltnew(EEG_orig, 1, 200);

% Remove 60 Hz line noise and its harmonics
EEG = pop_eegfiltnew(EEG, 'locutoff',59,'hicutoff',61,'revfilt',1); 
EEG = pop_eegfiltnew(EEG, 'locutoff',119,'hicutoff',121,'revfilt',1);
EEG = pop_eegfiltnew(EEG, 'locutoff',179,'hicutoff',181,'revfilt',1);

% Remove the same bad channels
removed_chans = EEG.chanlocs(~EEG_ICA.etc.clean_channel_mask); 
removed_labels = {removed_chans.labels};
EEG = pop_select(EEG, 'nochannel', removed_labels);

% check channel consistency
if EEG_ICA.nbchan ~= EEG.nbchan
    error('Channel count does not match between datasets.');
elseif ~isequal({EEG_ICA.chanlocs.labels}, {EEG.chanlocs.labels})
    error('Channel labels or order do not match — cannot safely transfer ICA.');
else
    disp('Channel count and order are consistent.');
end
%% 

% Extra 60Hz line noise filtering (if needed)
% EEG = pop_eegfiltnew(EEG, 'locutoff',58,'hicutoff',62,'revfilt',1); 

% re-referenve and resample
EEG = pop_reref( EEG, []);
EEG = pop_resample( EEG, 500);

% Remove the same bad data segments; GET FROM HISTORY
% CHANGE THIS

% OVERT/SPOKEN DATASET
% subj-01, sess-01
% EEG = eeg_eegrej( EEG, [11 277; 234690 235274]);
% EEG = eeg_eegrej( EEG, [11 277;26672 27449;75704 76504;83332 84133;119318 120017;134020 134706;139461 139999;156043 156609;163150 163665;181993 182452;190899 191468;234690 235274]);

% subj-01, sess-02
% EEG = eeg_eegrej( EEG, [12650 14210;164692 165299;238172 238382]);

% COVERT/IMAGINED DATASET
% subj-01, sess-01
% EEG = eeg_eegrej( EEG, [1 427;237514 238432]);

% subj-01, sess-02
% EEG = eeg_eegrej( EEG, [4 177;52445 53406;63556 64031;229668 230084]);

% subj-02, sess-01, spoken
% EEG = eeg_eegrej( EEG, [2 97;231504 231924]);

% subj-02, sess-01, imagined
% EEG = eeg_eegrej( EEG, [64873 66730;233173 233545]);

% subj-03, sess-01, imagined
% EEG = eeg_eegrej( EEG, [2 436;236847 237847]);

% subj-03, sess-01, spoken
% EEG = eeg_eegrej( EEG, [2 513;238809 239116]);

% subj-04, sess-01, spoken
% EEG = eeg_eegrej( EEG, [3 387;232743 233165]);

% subj-04, sess-01, imagined
% EEG = eeg_eegrej( EEG, [3 844;242627 243410]);

% subj-02, sess-02, spoken
EEG = eeg_eegrej( EEG, [2 220;237245 237746]);

% subj-02, sess-02, imagined
% EEG = eeg_eegrej( EEG, [2 358;240867 241720]);

%% Transfer ICA weights to 1-200Hz dataset for analysis

% ica_weights: [n_ICs × n_PCA_components]
% icasphere: [n_PCA_components × n_channels]

% Reconstruct the full unmixing matrix W
W = EEG_ICA.icaweights * EEG_ICA.icasphere;  % shape: [n_ICs × n_channels]

% Transfer ICA to EEG_analysis
EEG.icaweights  = W;
EEG.icawinv     = pinv(W);
EEG.icasphere   = eye(size(W, 2));  % full-rank, no sphering
EEG.icachansind = 1:size(W, 2);
EEG.icaact      = [];

% Check 
EEG = eeg_checkset(EEG);

% Run ICLabel classification
EEG = pop_iclabel(EEG, 'lite');
% view components
pop_viewprops(EEG, 0, 1:size(EEG.icaweights,1), {'freqrange' [2 200]});

%% Mark non-brain ICs to be rejected (but don't reject yet)

% Extract ICLabel probabilities
% Columns order (default ICLabel output):
% 1: Brain, 2: Muscle, 3: Eye, 4: Heart, 5: Line Noise, 6: Channel Noise, 7: Other

ic_probs = EEG.etc.ic_classification.ICLabel.classifications;

% Define muscle ICs to have prob > 0.5
all_muscle_ICs = find(ic_probs(:,2) > 0.5);

% The 7th column corresponds to the "other" class
other_probs = ic_probs(:,7);

% Find indices where "other" is the most likely class
[~, max_classes] = max(ic_probs, [], 2);
other_ICs = find(max_classes == 7);

% find all ICs with brain probability < thresh
thresh = 0.05;
non_brain_ICs = find(ic_probs(:,1) < thresh);
fprintf('\nNon-brain ICs: %s\n', num2str(non_brain_ICs'));

% find ICs that are non brain but not marked as other
ICs_to_reject = setdiff(non_brain_ICs, other_ICs);
fprintf('\nNon-brain but not marked as others ICs: %s\n', num2str(ICs_to_reject'));

% Mark non-brain ICs for rejection
EEG.reject.gcompreject(ICs_to_reject) = 1;

rejected_muscle_ICs = intersect(all_muscle_ICs, ICs_to_reject);
fprintf('\nRejected muscle ICs: %s\n', num2str(rejected_muscle_ICs'));

% visualize the components
pop_selectcomps(EEG, [1:size(EEG.icaweights,1)] );

%% 

% save set
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_bp_1_200hz_bad_data_removed_full_transferred_ICs_two_notch_filtering_mark_non_brain'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% subtract rejected ICs

EEG = pop_subcomp(EEG, [], 1);

%% Interpolate removed bad channels (NOT USING)

% EEG = pop_interp(EEG, EEG_orig.chanlocs, 'spherical');
% [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

%% re-reference to average

EEG = pop_reref( EEG, []);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

%% Run PCA to determine # PCs to keep

n_pca = run_PCA(EEG, 2);
disp(n_pca)

%% Second ICA run

numprocs = 1;       % # of nodes (default = 1)
max_threads = 4;    % # of threads per node
num_models = 1;     % # of models of mixture ICA
max_iter = 2000;    % max number of learning steps
% ===== EDIT THIS PARAMETER IF NEEDED ==== %
pcakeep = n_pca;       % EDIT NUM of PCs to keep

% run amica
outdir = [ pwd filesep 'amicaouttmp' filesep ];
[weights,sphere,mods] = runamica15(EEG.data, 'num_models',num_models, 'outdir',outdir, ...
    'numprocs', numprocs, 'max_threads', max_threads, 'max_iter',max_iter, 'pcakeep', pcakeep);

% Assign weights and sphere to EEG
EEG.icaweights = weights;
EEG.icasphere = sphere;

% Store number of components
EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;  % assuming all channels used

% Run ICLabel classification
EEG = pop_iclabel(EEG, 'default');
% view components
pop_viewprops(EEG, 0, 1:size(EEG.icaweights,1), {'freqrange' [2 200]});

%% save
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_bp_1_200hz_bad_data_removed_cleaned_2ndICA' ...
    ''];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Epoch

max_epoch_length = 1.5;

% Spoken/overt

% mark non-brain (NOT USING)
% CHANGE THIS
% ICs_to_remove = [1,3,4,5,7,10,12,14];
% EEG.reject.gcompreject(ICs_to_remove) = 1;

% setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
% 
% EEG = pop_epoch( EEG, {  'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _guSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _miSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _muSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _siSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _suSP_[]_ECI TCP-IP 55513'  }, ...
%                          [-0.5 max_epoch_length], ...
%                          'newname', setname, ...
%                          'epochinfo', 'yes');
% 
% % save to disk
% filename = [setname, '.set'];
% EEG = pop_saveset(EEG, 'filename', filename, 'filepath', dataset_path);

% Imagined/covert

% mark non-brain (NOT USING)
% CHANGE THIS
% ICs_to_ignore = [1,2,3,5,7,8,14:34];
% EEG.reject.gcompreject(ICs_to_ignore) = 1;
% 
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
EEG = pop_epoch( EEG, {  'EVNT_STIM_    _giIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _guIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _miIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _muIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _siIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _suIM_[]_ECI TCP-IP 55513'  }, ...
                         [-0.5 max_epoch_length], ...
                         'newname', setname, ...
                         'epochinfo', 'yes');

% save to disk
filename = [setname, '.set'];
EEG = pop_saveset(EEG, 'filename', filename, 'filepath', dataset_path);

%% Compute voice onset and offset relative mean time (overt dataset only)

nEpochs = length(EEG.epoch);
onset_latencies  = nan(1, nEpochs);
offset_latencies = nan(1, nEpochs);
cond_labels = {'giSP', 'guSP', 'miSP', 'muSP', 'siSP', 'suSP'};

% Initialize condition-specific latency storage
cond_onsets  = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));
cond_offsets = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));

for i = 1:nEpochs
    etype  = EEG.epoch(i).eventtype;
    elat   = EEG.epoch(i).eventlatency;
    elabel = EEG.epoch(i).eventlabel;

    % Convert to cell arrays if needed
    if ~iscell(etype),  etype  = {etype};  end
    if ~iscell(elat),   elat   = {elat};   end
    if ~iscell(elabel), elabel = {elabel}; end

    % Get indices for 'onset' and 'offset'
    onset_idx  = strcmp(etype, 'onset');
    offset_idx = strcmp(etype, 'offset');

    % Save latencies
    if any(onset_idx)
        onset_latencies(i) = elat{onset_idx};
    end
    if any(offset_idx)
        offset_latencies(i) = elat{offset_idx};
    end

    % Determine condition label (assumes only one condition label per epoch)
    cond_match = intersect(elabel, cond_labels);
    if ~isempty(cond_match)
        label = cond_match{1};
        if any(onset_idx)
            cond_onsets(label) = [cond_onsets(label), elat{onset_idx}];
        end
        if any(offset_idx)
            cond_offsets(label) = [cond_offsets(label), elat{offset_idx}];
        end
    end
end

% ---- Overall Stats ----
fprintf('--- Overall Latency Statistics ---\n');
fprintf('Onset:  Mean = %.2f ms, Min = %.2f ms, Max = %.2f ms\n', ...
    mean(onset_latencies, 'omitnan'), min(onset_latencies, [], 'omitnan'), max(onset_latencies, [], 'omitnan'));
fprintf('Offset: Mean = %.2f ms, Min = %.2f ms, Max = %.2f ms\n', ...
    mean(offset_latencies, 'omitnan'), min(offset_latencies, [], 'omitnan'), max(offset_latencies, [], 'omitnan'));

% ---- Per-Condition Stats ----
fprintf('\n--- Per-Condition Latency Statistics ---\n');
for k = 1:length(cond_labels)
    label = cond_labels{k};
    onsets  = cond_onsets(label);
    offsets = cond_offsets(label);

    fprintf('%s:\n', label);
    if ~isempty(onsets)
        fprintf('  Onset:  Mean = %.2f ms, Min = %.2f, Max = %.2f\n', ...
            mean(onsets), min(onsets), max(onsets));
    else
        fprintf('  Onset:  No data\n');
    end
    if ~isempty(offsets)
        fprintf('  Offset: Mean = %.2f ms, Min = %.2f, Max = %.2f\n', ...
            mean(offsets), min(offsets), max(offsets));
    else
        fprintf('  Offset: No data\n');
    end
end

%% Save speech onset offset for overt dataset

speech_timing_filename = [SUBJ,'_',SESS, '_speech_onset_offset.mat'];
save(speech_timing_filename, 'onset_latencies', 'offset_latencies');

