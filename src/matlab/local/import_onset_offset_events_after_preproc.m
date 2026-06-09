eeglab; % launch EEGLAB

%% Load dataset from step 1 preprocessed EEG 

% Check events
pop_eegplot(EEG, 1, 1, 1);

% Align with the cleaned EEG dataset (resample + trim)
EEG = pop_resample( EEG, 500);

trim_samples = floor(1 * EEG.srate);
start_trim   = 1;
end_trim     = EEG.pnts - trim_samples;
EEG = eeg_eegrej(EEG, [start_trim, trim_samples; end_trim, EEG.pnts]);
fprintf('Trimmed samples 1-%d (start) and %d-%d (end)\n', trim_samples, end_trim, EEG.pnts);

%% 

% Set output filename
filename = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/subj-12_speech_onset_offset_events_cleaned_EEG_aligned.txt';
fid = fopen(filename, 'w');

% Loop through EEG events
for i = 1:length(EEG.event)
    eventType = EEG.event(i).type;
    
    % Check if event is 'onset' or 'offset'
    if ischar(eventType) && (strcmp(eventType, 'onset') || strcmp(eventType, 'offset'))
        % Convert latency (in samples) to seconds
        time_sec = EEG.event(i).latency / EEG.srate;
        fprintf(fid, '%.6f\t%s\n', time_sec, eventType);
    end
end

% Close file
fclose(fid);