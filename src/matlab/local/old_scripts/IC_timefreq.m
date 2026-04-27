% Time-frequency analysis (using Hilbert transform)

eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

SPEECH_TYPE = 'sp';
SUBJ = 'subj-02';

if SPEECH_TYPE == 'sp'
    dataset_path = [dataset_path,'/',SUBJ,'/spoken'];
else 
    dataset_path = [dataset_path,'/',SUBJ,'/imagined'];
end

%% Load cleaned and epoched dataset

setname = [SUBJ,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname,'.set'];
EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
% updates data structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

eeglab redraw; % refresh GUI


%% Calculate icaact for the dataset

for i = 1:len(ALLEEG)
    ALLEEG(i).icaact = (ALLEEG(i).icaweights * ALLEEG(i).icasphere) * ALLEEG(i).data(ALLEEG(i).icachansind, :);
    % Reshape to 3D: [ICs x time x trials]
    ALLEEG(i).icaact = reshape(ALLEEG(i).icaact, size(ALLEEG(i).icaweights, 1), ALLEEG(i).pnts, ALLEEG(i).trials);
end

%% Hilbert transform 

EEG = ALLEEG(1);
fs = EEG.srate;
times = EEG.times;

theta_band      = [4 7];
alpha_band      = [8 12];
beta_band       = [13 30];
gamma_band      = [30 75];
high_gamma_band = [75 160];

bands = {theta_band, alpha_band, beta_band, gamma_band, high_gamma_band};
band_names = {'Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma'};

% Split into different bands before computing Hilbert transform
narrowband_EEG = cell(1,length(bands));

for b = 1:length(band_names)
    band_range = bands{b};
    narrowband_EEG{b} = pop_eegfiltnew(EEG, 'locutoff',band_range(1),'hicutoff',band_range(2));

    narrowband_EEG{b}.icaweights = EEG.icaweights;
    narrowband_EEG{b}.icasphere  = EEG.icasphere;
    narrowband_EEG{b}.icachansind = EEG.icachansind;

     % Use the same ICA weights and channel indices from the broadband data
    narrowband_EEG{b}.icaact = (narrowband_EEG{b}.icaweights * narrowband_EEG{b}.icasphere) * narrowband_EEG{b}.data(narrowband_EEG{b}.icachansind, :);
    
    % Reshape to [ICs x time x trials]
    narrowband_EEG{b}.icaact = reshape(narrowband_EEG{b}.icaact, size(narrowband_EEG{b}.icaweights,1), narrowband_EEG{b}.pnts, narrowband_EEG{b}.trials);
end

%% Save the raw hilbert analytic signals

% Get trial class labels

nEpochs = length(EEG.epoch);

classes = {'gi', 'gu', 'mi', 'mu', 'si', 'su'};
labels = nan(1, nEpochs);

for i = 1:nEpochs
    etype = EEG.epoch(i).eventtype;

    if iscell(etype)
        % get class labels for the trials
        cond_idx = find(contains(etype, 'EVNT_STIM_'));
        cond_str = etype{cond_idx};
        class_match_idx = find(cellfun(@(c) contains(cond_str, c), classes));
        labels(i) = class_match_idx;
    end
end

% Preallocate: [components x bands x time x trials] — complex
nBands   = length(narrowband_EEG);
nTrials  = size(narrowband_EEG{1}.icaact, 3);
nTimes   = size(narrowband_EEG{1}.icaact, 2);
nComps   = size(narrowband_EEG{1}.icaact, 1);

analytic_signal = complex(nan(nComps, nBands, nTimes, nTrials));

for b = 1:nBands
    data_band = narrowband_EEG{b}.icaact;
    for c = 1:nComps
        for tr = 1:nTrials
            signal = squeeze(data_band(c, :, tr));
            analytic_signal(c, b, :, tr) = hilbert(signal);
        end
    end
end

% Save both the analytic signal and metadata
filename = [SUBJ,'_',SESS,'_', SPEECH_TYPE, '_eeg_analytic.mat'];
save(filename, 'analytic_signal', 'labels', 'times', 'band_names', 'bands', '-v7.3');


%% Save scalp topologies (icawinv OLD CODE)

% % NEED TO INTERPOLATE TO HAVE FULL HEAD!
% 
% % icawinv: 257 * #ICs
% icawinv = ALLEEG(1).icawinv;
% filename = [SUBJ,'_',SESS,'_', SPEECH_TYPE, '_icawinv.mat'];
% save(filename, 'icawinv');


%% Compute Hilbert transform and remove single-trial baseline for all components (OLD CODE)

nBands   = length(narrowband_EEG);
nTrials  = size(narrowband_EEG{1}.icaact, 3);
nTimes   = size(narrowband_EEG{1}.icaact, 2);
nComps   = size(narrowband_EEG{1}.icaact, 1);

% Define baseline window
baseline_idx = find(times >= -450 & times <= 0);  % taking -450ms to exclude edge artifacts at -500ms

% Preallocate: [components x bands x time x trials]
z_power = nan(nComps, nBands, nTimes, nTrials);

for b = 1:nBands
    data_band = narrowband_EEG{b}.icaact;  % [components x time x trials]

    for c = 1:nComps
        for tr = 1:nTrials
            signal = squeeze(data_band(c, :, tr));    % [1 x time]
            analytic = hilbert(signal);               % complex
            power = abs(analytic).^2;                 % [1 x time]

            % Baseline mean & std
            mu_baseline  = mean(power(baseline_idx));
            std_baseline = std(power(baseline_idx));

            % Z-score normalization
            z_power(c, b, :, tr) = (power - mu_baseline) / std_baseline;
        end
    end
end

% Get trial class labels

nEpochs = length(EEG.epoch);

classes = {'gi', 'gu', 'mi', 'mu', 'si', 'su'};
labels = nan(1, nEpochs);

for i = 1:nEpochs
    etype = EEG.epoch(i).eventtype;

    if iscell(etype)
        % get class labels for the trials
        cond_idx = find(contains(etype, 'EVNT_STIM_'));
        cond_str = etype{cond_idx};
        class_match_idx = find(cellfun(@(c) contains(cond_str, c), classes));
        labels(i) = class_match_idx;
    end
end


% Save the time-frequency data

filename = [SUBJ,'_',SESS,'_', SPEECH_TYPE, '_eeg_data_time_freq_z_power_labels.mat'];
save(filename, 'z_power', 'labels');

%% Plot average power (to compare with wavelet results OLD CODE)

% CHANGE - SELECT IC TO PLOT
comp_num = 10;

nClasses = 6;
[comp, nFreqs, nTimes, nTrials] = size(z_power(comp_num,:,:,:));

EEG = ALLEEG(1);
fs = EEG.srate;
times = EEG.times;

class_means = zeros(nFreqs, nTimes, nClasses);

for c = 1:nClasses
    idx = labels == c;  % Find trials belonging to class c
    class_means(:,:,c) = mean(z_power(comp_num,:,:,idx), 4);  % Mean over trials
end

% plot
time_range = [225 975];  % select time range to plot
%time_range = [1 length(times)];

band_names = {'Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma'};

% Base colors (dark versions)
base_colors = [
    0.2 0.4 0.8;   % blue
    0.2 0.7 0.2;   % green
    0.7 0.2 0.7    % purple
];

% Generate lighter versions by mixing with white
lighter_colors = base_colors * 0.35 + 0.65;  % simple linear blend with white

% Combine into 6 colors: dark, light, dark, light, ...
colors = zeros(6,3);
colors(1:2:end, :) = base_colors;
colors(2:2:end, :) = lighter_colors;


nrows = 3;
ncols = 2;
figure;

for b = 1:length(band_names)
    subplot(nrows, ncols, b);  % choose subplot location
    hold on;
    for i = 1:6
        plot(times((time_range(1)):(time_range(2))), class_means(b,(time_range(1)):(time_range(2)),i), 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    xlabel('Time (ms)');
    ylabel('Power (z-score)');
    title(sprintf('%s Band Power', band_names{b}));
    legend({"gi", "gu", "mi", "mu", "si", "su"}, 'Location', 'best');
    grid on;
end

sgtitle(sprintf('Overt IC %d Band power comparison', comp_num));
