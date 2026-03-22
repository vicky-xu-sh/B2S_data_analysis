% EEGLAB ERSP (wavelet)

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

%% Load seperated overt datasets (if exists)

filenames = ["sp_gi.set", "sp_gu.set", "sp_mi.set", "sp_mu.set", "sp_si.set", "sp_su.set"];
for i = 1:6
    filename = char(filenames(i));
    EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
end

eeglab redraw; % refresh GUI

%% Load seperated covert datasets (if exists)

filenames = ["im_gi.set", "im_gu.set", "im_mi.set", "im_mu.set", "im_si.set", "im_su.set"];
for i = 1:6
    filename = char(filenames(i));
    EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
end

eeglab redraw; % refresh GUI

%% Save each syllable condition as seperate datasets (if not already)

EEG_all = EEG;
setname = 'sp_gi';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'sp_gu';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _guSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'sp_mi';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _miSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'sp_mu';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _muSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 5,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'sp_si';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _siSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 6,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'sp_su';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _suSP_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 7,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 
eeglab redraw;

%% Covert

EEG_all = EEG;
setname = 'im_gi';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _giIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'im_gu';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _guIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'im_mi';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _miIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'im_mu';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _muIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 5,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'im_si';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _siIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 6,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 

setname = 'im_su';
filename = [setname, '.set'];
EEG = pop_selectevent( EEG_all, 'type',{'EVNT_STIM_    _suIM_[]_ECI TCP-IP 55513'},'deleteevents','off','deleteepochs','on','invertepochs','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 7,'setname',setname,'savenew',fullfile(dataset_path, filename),'gui','off'); 
eeglab redraw;


%% Calculate icaact for each dataset

for i = 1:7
    ALLEEG(i).icaact = (ALLEEG(i).icaweights * ALLEEG(i).icasphere) * ALLEEG(i).data(ALLEEG(i).icachansind, :);
    % Reshape to 3D: [ICs x time x trials]
    ALLEEG(i).icaact = reshape(ALLEEG(i).icaact, size(ALLEEG(i).icaweights, 1), ALLEEG(i).pnts, ALLEEG(i).trials);
end

%% Generate ERSP plots for all ICs and CV conditions

% CHANGE THIS
ICs = [3,10,12,14,15,20,24];

freqs = exp(linspace(log(4), log(150), 100));

% Construct cycles vector matched to freqs
cycles = zeros(size(freqs));
for i = 1:length(freqs)
    if freqs(i) <= 45
        % linear scale from 1 to 8 in the lower part
        cycles(i) = 1 + (8 - 1) * (log(freqs(i)) - log(4)) / (log(45) - log(4));
    else
        cycles(i) = 8;
    end
end

syllables = ["all", "gi", "gu", "mi", "mu", "si", "su"];

nrows = 3;
ncols = 2;

for k = 1:length(ICs)
    comp_num = ICs(k);

    figure;
    for i = 2:7
        curr_cv = syllables(i);
        caption = sprintf('IC %d broadband ERSP p<0.05 FDR corrected, single-trial norm + STD baseline norm', comp_num);
        %caption = sprintf('IC %d broadband ERSP, single-trial norm + STD baseline norm', comp_num);
        titletext = sprintf('%s', curr_cv);
        EEG = ALLEEG(i);

        subplot(nrows, ncols, i-1);  % choose subplot location

        % 'mcorrect', 'fdr', 'alpha', 0.05, % if using statistical testing fdr correction
        newtimef( EEG.icaact(comp_num, :, :), EEG.pnts, [-500  1498], EEG.srate, cycles, ...
            'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
            'baseline',0, 'freqs', freqs, ... 
            'plotphase', 'on', 'plotitc', 'off', ...
            'freqscale', 'log', 'ntimesout', 200, ...
            'title', titletext, 'caption', caption, ...
            'mcorrect', 'fdr', 'alpha', 0.05, ...
            'basenorm', 'on', 'trialbase', 'full'); 

    end
end

%% Save ERSP for each syllables

% CHANGE THIS
ICs = [5,7,9,11,12,16,19]; % subj-02 covert

% CHANGE THIS
max_time_ms = 1998;

freqs = exp(linspace(log(4), log(150), 100));

% Construct cycles vector matched to freqs
cycles = zeros(size(freqs));
for i = 1:length(freqs)
    if freqs(i) <= 45
        % linear scale from 1 to 8 in the lower part
        cycles(i) = 1 + (8 - 1) * (log(freqs(i)) - log(4)) / (log(45) - log(4));
    else
        cycles(i) = 8;
    end
end

syllables = ["all", "gi", "gu", "mi", "mu", "si", "su"];

all_ersp = cell(1,6);
all_times = cell(1,6);

for k = 1:length(ICs)
    comp_num = ICs(k);
    for i = 2:7
        EEG = ALLEEG(i);
        [ersp,itc,powbase,times,freqs,erspboot,itcboot, tfdata] = newtimef( EEG.icaact(comp_num, :, :), EEG.pnts, [-500  max_time_ms], EEG.srate, cycles, ...
            'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
            'baseline',[-450 0], 'freqs', freqs, ... 
            'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 200, ...
            'plotitc', 'off', 'plotersp', 'off', ...
            'basenorm', 'on', 'trialbase', 'full'); 
        all_ersp{i-1} = ersp;
        all_times{i-1} = times;
    end

    plot_band_power_all(all_ersp, all_times, freqs, comp_num);
end

%% Generate ERSP Band power comparison plots
function plot_band_power_all(all_ersp, all_times, freqs, comp_num)
% Plot band power curves across all CV syllables for theta, alpha, beta, gamma, and high gamma

theta_band      = [4 8];
alpha_band      = [8 12];
beta_band       = [13 30];
gamma_band      = [30 75];
high_gamma_band = [75 160];

bands = {theta_band, alpha_band, beta_band, gamma_band, high_gamma_band};
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
    band_range = bands{b};

    subplot(nrows, ncols, b);  % choose subplot location
    hold on;

    for i = 1:6
        ersp = all_ersp{i};     % [freqs x time]
        times = all_times{i};   % [1 x time]
        
        % Find frequency indices
        band_idx = find(freqs >= band_range(1) & freqs <= band_range(2));
        if isempty(band_idx)
            warning('No frequencies found for %s band in [%g %g] Hz — skipping.', band_names{b}, band_range(1), band_range(2));
            continue;
        end
        
        % Average over frequency band
        band_power = mean(ersp(band_idx, :), 1);  % [1 x time]

        plot(times, band_power, 'Color', colors(i,:), 'LineWidth', 1.5);
    end

    xlabel('Time (ms)');
    ylabel('Power (z-score)');
    title(sprintf('%s Band Power', band_names{b}));
    legend({"gi", "gu", "mi", "mu", "si", "su"}, 'Location', 'best');
    grid on;
end

sgtitle(sprintf('IC %d Band power comparison', comp_num));
end



%% Exploring different time-frequency parameters

% EEG = EEG_all;
% comp_num = 2;
% figure; pop_newtimef( EEG, 0, comp_num, [-500  1498], [8] , 'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
%     'baseline',[0], 'freqs', [85 175], ...
%     'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 400, 'padratio', 1, ...
%     'caption', 'IC 2 high-gamma band ERSP 8-cycle wavelet, padratio 1');
% 
% figure; pop_newtimef( EEG, 0, comp_num, [-500  1498], [8] , 'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
%     'baseline',[0], 'freqs', [85 175], ...
%     'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 400, 'padratio', 2, ...
%     'caption', 'IC 2 high-gamma band ERSP padratio 2');
% 
% figure; pop_newtimef( EEG, 0, comp_num, [-500  1498], [8] , 'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
%     'baseline',[0], 'alpha',0.05, 'freqs', [85 175], ...
%     'mcorrect', 'fdr', 'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 400, 'padratio', 2, ...
%     'caption', 'IC 2 high-gamma band ERSP padratio 2, p<0.05 FDR corrected');
% 
% figure; pop_newtimef( EEG, 0, comp_num, [-500  1498], [8] , 'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
%     'baseline',[0], 'alpha',0.05, 'freqs', [85 175], ...
%     'mcorrect', 'fdr', 'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 400, 'padratio', 2, ...
%     'caption', 'IC 2 high-gamma band ERSP padratio 2, p<0.05 FDR corrected, STD baseline normalization', ...
%     'basenorm', 'on'); % STD baseline (z-score)
% 
% % 'trialbase' 'full' is an option that perform single trial normalization (or simple division based on the 
% % 'basenorm' input over the full trial length before performing standard baseline removal. It has been
% % shown to be less sensitive to noisy trials.
% figure; pop_newtimef( EEG, 0, comp_num, [-500  1498], [8] , 'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
%     'baseline',[0], 'alpha',0.05, 'freqs', [85 175], ...
%     'mcorrect', 'fdr', 'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 400, 'padratio', 2, ...
%     'caption', 'IC 2 high-gamma band ERSP padratio 2, p<0.05 FDR corrected, single-trial norm + STD baseline norm', ...
%     'basenorm', 'on', 'trialbase', 'full'); 

%% Find all voice onsets and labels for all trials
% 
% nEpochs = length(EEG.epoch);
% onset_latencies  = nan(1, nEpochs);
% 
% classes = {'gi', 'gu', 'mi', 'mu', 'si', 'su'};
% labels = nan(1, nEpochs);
% 
% for i = 1:nEpochs
%     etype = EEG.epoch(i).eventtype;
%     elat  = EEG.epoch(i).eventlatency;
% 
%     if iscell(etype)
%         % get vector of onset latencies
%         onset_idx  = strcmp(etype, 'onset'); 
%         onset_latencies(i)  = elat{onset_idx};
% 
%         % get class labels for the trials
%         cond_idx = find(contains(etype, 'EVNT_STIM_'));
%         cond_str = etype{cond_idx};
%         class_match_idx = find(cellfun(@(c) contains(cond_str, c), classes));
%         labels(i) = class_match_idx;
%     end
% end

%% Try to plot power for each trials (using ERAPIMAGE)

% comp_num = 8;
% classes = {'gi', 'gu', 'mi', 'mu', 'si', 'su'};
% 
% for b = 1:length(band_names)
%     nrows = 3;
%     ncols = 2;
%     figure;
% 
%     for c = 1:nClasses
%         idx = labels == c;  % Find trials belonging to class c
% 
%         subplot(nrows, ncols, c);  % choose subplot location
% 
%         erpimage(z_power(comp_num,b,:,idx), onset_latencies(idx), [-500 nTimes fs], ...
%             sprintf('%s %s Band Power', classes{c}, band_names{b}), 1, 1, ...
%             'erp', 'on', 'cbar', 'on');
%     end
% end


%% Phase and amplitube analysis
% 
% nBands   = length(narrowband_EEG);
% nTrials  = size(narrowband_EEG{1}.icaact, 3);
% nTimes   = size(narrowband_EEG{1}.icaact, 2);
% nComps   = size(narrowband_EEG{1}.icaact, 1);
% 
% % Preallocate
% amp_data   = nan(nComps, nBands, nTimes, nTrials);   % amplitude envelope
% phase_data = nan(nComps, nBands, nTimes, nTrials);   % phase in radians
% 
% for b = 1:nBands
%     data_band = narrowband_EEG{b}.icaact;  % [components x time x trials]
% 
%     for c = 1:nComps
%         for tr = 1:nTrials
%             signal = squeeze(data_band(c, :, tr));    % [1 x time]
%             analytic = hilbert(signal);               % complex analytic signal
% 
%             amp_data(c, b, :, tr)   = abs(analytic);   % amplitude
%             phase_data(c, b, :, tr) = angle(analytic); % phase
%         end
%     end
% end
% 
% % Save to .mat
% filename = [SUBJ,'_',SESS,'_', SPEECH_TYPE, '_IC_amp_phase.mat'];
% save(filename, 'amp_data', 'phase_data');