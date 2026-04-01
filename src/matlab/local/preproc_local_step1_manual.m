% preproc_local_step1_manual.m
%
% PURPOSE: Manual local preprocessing steps to be run interactively in EEGLAB.
%   1. Import raw .mff file
%   2. Import voice onset/offset events (overt speech only)
%   3. Import channel locations
%   4. Optionally delete obviously bad channels (non-scalp electrodes)
%   5. Visualize raw data and save as *_raw_edited.set
%
% OUTPUT: saves to data/02_interim_local/{SUBJ}/{spoken|imagined}/
%
% Run BEFORE src/matlab/local/preproc_local_step2.m (locally) or
% src/matlab/cluster/preproc_hpc.sh (on HPC)


%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================

% CHANGE THESE
SUBJ        = 'subj-05';
SPEECH_TYPE = 'im';   % 'sp' = spoken/overt | 'im' = imagined/covert

BASE_PATH  = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/data';
RAW_PATH   = fullfile(BASE_PATH, '01_raw/', SUBJ);

% Build I/O paths
if strcmp(SPEECH_TYPE, 'sp')
    OUTPUT_DIR = fullfile(BASE_PATH, '02_interim_local', SUBJ, 'spoken');
else
    OUTPUT_DIR = fullfile(BASE_PATH, '02_interim_local', SUBJ, 'imagined');
end

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
    fprintf('Created output directory: %s\n', OUTPUT_DIR);
else
    fprintf('Output directory: %s\n', OUTPUT_DIR);
end

% Launch EEGLAB
eeglab;
global ALLEEG EEG CURRENTSET;

%% =========================================================================
%  IMPORT RAW .mff FILE
%  Only run this section if the dataset has not been saved yet.
% ==========================================================================

if strcmp(SPEECH_TYPE, 'sp')
    raw_mff_filename_pattern = fullfile(RAW_PATH, '*spoken*.mff');
else
    raw_mff_filename_pattern = fullfile(RAW_PATH, '*imagined*.mff');
end

raw_mff_file_matches = dir(raw_mff_filename_pattern);
if ~isempty(raw_mff_file_matches)
    [~, mff_filename, ~] = fileparts(raw_mff_file_matches(1).name);
    raw_mff_file = fullfile(RAW_PATH, [mff_filename, '.mff']);
else
    error('Raw mff file not found.')
end

fprintf('Importing the raw mff file: %s\n', raw_mff_file);
EEG = pop_mffimport({raw_mff_file}, {'classid','code','description','label','mffkeys','name'}, 0, 0);
EEG.setname = [SUBJ, '_pilot_', SPEECH_TYPE, '_raw'];

[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw;

fprintf('Imported the raw EEG set: %s\n', EEG.setname);


%% =========================================================================
%  IMPORT VOICE ONSET / OFFSET EVENTS  (overt speech only)
% ==========================================================================

if strcmp(SPEECH_TYPE, 'sp')
    audio_path   = fullfile(BASE_PATH, '02_interim_local', SUBJ, 'spoken');
    speech_events_file = fullfile(audio_path, [SUBJ, '_speech_events.txt']);
    fprintf('Importing the speech onset/offset events: %s\n', speech_events_file);

    EEG = pop_importevent(EEG, ...
        'event',    speech_events_file, ...
        'fields',   {'latency', 'type'}, ...
        'timeunit', 1, ...
        'align',    NaN, ...
        'append',   'yes');

    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
    eeglab redraw;

    % Visually verify events are in the right places
    pop_eegplot(EEG, 1, 1, 1);
end


%% =========================================================================
%  IMPORT CHANNEL LOCATIONS
% ==========================================================================

sfp_filename_pattern = fullfile(RAW_PATH, '*.sfp');

sfp_file_matches = dir(sfp_filename_pattern);
if ~isempty(sfp_file_matches)
    [~, sfp_filename, ~] = fileparts(sfp_file_matches(1).name);
    sfp_file = fullfile(RAW_PATH, [sfp_filename, '.sfp']);
else
    error('Channel location .sfp file not found.')
end

fprintf('Importing %s channel location .sfp file: %s\n', SUBJ, sfp_file);
EEG = pop_chanedit(EEG, 'load', {sfp_file, 'filetype', 'sfp'});

% Label fiducials
EEG = pop_chanedit(EEG, ...
    'changefield', {258, 'labels', 'Nz'}, ...
    'changefield', {259, 'labels', 'LPA'}, ...
    'changefield', {260, 'labels', 'RPA'});

% Align electrode locations in 3D space — three sequential steps:
%   1. Translate:  pop_chancenter moves the origin to the best-fit head centre
%   2. Tilt fix:   rotate so Cz sits exactly on the +Z axis  (X=0, Y=0 from above)
%   3. Spin fix:   rotate around Z so Nz sits on the +X axis (Y=0, nose forward)
%   4. Translate:  pop_chancenter again after corrections

EEG = pop_chanedit(EEG, 'eval', 'chans = pop_chancenter(chans, [], []);');
EEG = align_chanlocs(EEG);
EEG = pop_chanedit(EEG, 'eval', 'chans = pop_chancenter(chans, [], []);');
 
% Visualize — before vs after
figure;
subplot(1,2,1);
topoplot([], EEG.chanlocs, 'style', 'blank', 'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
title('Channel locations (aligned)');
subplot(1,2,2);
chan_xyz = [[EEG.chanlocs.X]; [EEG.chanlocs.Y]; [EEG.chanlocs.Z]];
scatter3(chan_xyz(1,:), chan_xyz(2,:), chan_xyz(3,:), 20, 'b', 'filled');
axis equal; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D electrode positions (check Cz top, Nz front)');

[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;


%% =========================================================================
%  DELETE OBVIOUSLY BAD CHANNELS (non-scalp electrodes, if needed)
%  Uncomment and edit the channel list as needed.
% ==========================================================================

% EEG = pop_select(EEG, 'nochannel', {'E102','E111','E120','E133','E122', ...
%                                      'E145','E165','E174','E187','E199','E208'});
% EEG = pop_select(EEG, 'nochannel', {'E92','E163','E209'});

% EEG = pop_select(EEG, 'nochannel', {'E82','E91','E92','E102','E103','E111','E112',...
%     'E120','E121','E133','E134','E145','E146','E156','E165','E166','E174',...
%     'E175','E187','E188','E199','E200','E208','E209','E216','E256'});
% EEG = pop_select(EEG, 'nochannel', {'E219'});
EEG = pop_select(EEG, 'nochannel', {'E91','E92','E102','E103','E111','E112',...
    'E120','E121','E122','E133','E134','E145','E146','E156','E165','E166','E174',...
    'E175','E187','E188','E199'});
% EEG = pop_select(EEG, 'nochannel', {'E217'});
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET+1);
eeglab redraw;

%% =========================================================================
%  FIX WRONG STIM EVENT LABELS
%  NOTE: subj-03 im dataset had stimulus events incorrectly marked as SP
% ==========================================================================
% for i = 1:length(EEG.event)
%     % Fix event type string e.g. 'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513'
%     EEG.event(i).type  = strrep(EEG.event(i).type,  'SP_[]', 'IM_[]');
%     % Fix label field e.g. 'giSP' -> 'giIM'
%     if isfield(EEG.event, 'label')
%         EEG.event(i).label = strrep(EEG.event(i).label, 'SP', 'IM');
%     end
% end
% 
% % Verify fix
% fprintf('Unique event types after fix:\n');
% disp(unique({EEG.event.type})');
% if isfield(EEG.event, 'label')
%     fprintf('Unique event labels after fix:\n');
%     disp(unique({EEG.event.label})');
% end
%
% [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
% eeglab redraw;

%% =========================================================================
%  VISUALIZE RAW DATA
% ==========================================================================

figure;
pop_spectopo(EEG, 1, [0 EEG.pnts], 'EEG', ...
    'freq', [10 20 80], 'freqrange', [0.1 200], 'electrodes', 'off');
sgtitle('Raw data power spectra');


%% =========================================================================
%  SAVE AS *_raw_edited.set  →  02_interim_local
% ==========================================================================

setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_raw_edited'];
filename = [setname, '.set'];
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname',  setname, ...
    'savenew',  fullfile(OUTPUT_DIR, filename), ...
    'gui',      'off');
eeglab redraw;

fprintf('\n=== Step 1 complete. Saved: %s ===\n', fullfile(OUTPUT_DIR, filename));
fprintf('Next: run preproc_local_step2.m locally or preproc_hpc.sh on HPC.\n');


%% =========================================================================
%  LOCAL FUNCTIONS
% ==========================================================================
 
function EEG = align_chanlocs(EEG)
% ALIGN_CHANLOCS  Rigidly aligns EEG electrode locations in 3D space.
%
%   After pop_chancenter (which only centres the origin), two residual
%   misalignments can remain:
%     - TILT:  Cz is not directly above the origin (X≠0 or Y≠0 from above).
%     - SPIN:  Nz is not in the +X direction (Y≠0 when viewed from above).
%
%   This function corrects both using Rodrigues rotation:
%     Step 1 — Rotate so Cz aligns with the +Z axis  → fixes tilt.
%     Step 2 — Rotate around Z so Nz aligns with +X  → fixes spin / nose direction.
%
%   All coordinate representations (Cartesian, spherical, 2D projected) are
%   recomputed from the corrected X/Y/Z values via EEGLAB's convertlocs.
%
%   Usage:
%       EEG = align_chanlocs(EEG);
 
    % ---- Extract Cartesian coordinates (N x 3) ----
    coords = [[EEG.chanlocs.X]; [EEG.chanlocs.Y]; [EEG.chanlocs.Z]]';
 
    % ---- Find landmark indices ----
    % Cz is a data channel — lives in EEG.chanlocs
    cz_idx = find(strcmpi({EEG.chanlocs.labels}, 'Cz'), 1);
 
    % Nz is a fiducial — EEGLAB stores fiducials in EEG.chaninfo.nodatchans,
    % NOT in EEG.chanlocs. Search there first; fall back to chanlocs.
    nz_idx  = [];
    nz_xyz  = [];
    if isfield(EEG, 'chaninfo') && isfield(EEG.chaninfo, 'nodatchans') && ~isempty(EEG.chaninfo.nodatchans)
        nz_fid = find(strcmpi({EEG.chaninfo.nodatchans.labels}, 'Nz'), 1);
        if ~isempty(nz_fid)
            nd = EEG.chaninfo.nodatchans(nz_fid);
            nz_xyz = [nd.X, nd.Y, nd.Z];
            fprintf('align_chanlocs: Nz found in chaninfo.nodatchans at [%.3f, %.3f, %.3f]\n', nz_xyz);
        end
    end
    if isempty(nz_xyz)
        % Fallback: Nz in chanlocs (shouldn't normally happen but just in case)
        nz_idx = find(strcmpi({EEG.chanlocs.labels}, 'Nz'), 1);
        if ~isempty(nz_idx)
            nz_xyz = [EEG.chanlocs(nz_idx).X, EEG.chanlocs(nz_idx).Y, EEG.chanlocs(nz_idx).Z];
            fprintf('align_chanlocs: Nz found in chanlocs at [%.3f, %.3f, %.3f]\n', nz_xyz);
        end
    end
 
    % ================================================================
    %  STEP 1: Align Cz → +Z axis  (removes head tilt)
    %  Rotate so that the vector pointing to Cz coincides with [0 0 1].
    % ================================================================
    if ~isempty(cz_idx)
        cz_vec  = coords(cz_idx, :);
        cz_unit = cz_vec / norm(cz_vec);
        z_hat   = [0, 0, 1];
 
        rot_axis = cross(cz_unit, z_hat);
        sin_a    = norm(rot_axis);
        cos_a    = dot(cz_unit, z_hat);
 
        if sin_a > 1e-10   % non-zero rotation needed
            rot_axis = rot_axis / sin_a;
            % Rodrigues' rotation formula
            K  = skew(rot_axis);
            R1 = eye(3) + sin_a * K + (1 - cos_a) * (K * K);
            coords = (R1 * coords')';
            % Also rotate the fiducial so Step 2 uses the post-tilt Nz position
            if ~isempty(nz_xyz)
                nz_xyz = (R1 * nz_xyz')';
            end
        end
 
        fprintf('align_chanlocs: Cz aligned to +Z (was [%.3f, %.3f, %.3f])\n', cz_vec);
    else
        warning('align_chanlocs: Cz not found — tilt correction skipped.');
    end
 
    % ================================================================
    %  STEP 2: Align Nz → +X axis  (removes spin; nose points forward)
    %  In EEGLAB's convention the nose is along +X, so Nz should be at Y=0.
    %  Rotate around Z so the XY projection of Nz lands on +X (Y=0).
    % ================================================================
    if ~isempty(nz_xyz)
        nz_xy    = nz_xyz(1:2);                  % XY projection of Nz (post-tilt frame)
        spin_ang = atan2(nz_xy(2), nz_xy(1));    % current angle from +X (want Y=0)
 
        if abs(spin_ang) > 1e-10
            c  = cos(-spin_ang);
            s  = sin(-spin_ang);
            R2 = [c, -s, 0;
                  s,  c, 0;
                  0,  0, 1];
            coords = (R2 * coords')';
            nz_xyz = (R2 * nz_xyz')';   % keep fiducial in sync
        end
 
        fprintf('align_chanlocs: Nz aligned to +X / Y=0 (spin correction: %.1f deg)\n', rad2deg(spin_ang));
    else
        warning('align_chanlocs: Nz not found in chanlocs or chaninfo.nodatchans — spin correction skipped.');
    end
 
    % ---- Write corrected X/Y/Z back into chanlocs ----
    for i = 1:length(EEG.chanlocs)
        EEG.chanlocs(i).X = coords(i, 1);
        EEG.chanlocs(i).Y = coords(i, 2);
        EEG.chanlocs(i).Z = coords(i, 3);
    end
 
    % ---- Write corrected Nz back into chaninfo.nodatchans (if that's where it lives) ----
    if ~isempty(nz_xyz) && isfield(EEG, 'chaninfo') && isfield(EEG.chaninfo, 'nodatchans')
        nz_fid = find(strcmpi({EEG.chaninfo.nodatchans.labels}, 'Nz'), 1);
        if ~isempty(nz_fid)
            EEG.chaninfo.nodatchans(nz_fid).X = nz_xyz(1);
            EEG.chaninfo.nodatchans(nz_fid).Y = nz_xyz(2);
            EEG.chaninfo.nodatchans(nz_fid).Z = nz_xyz(3);
        end
    end
 
    % ---- Recompute all other coordinate representations from X/Y/Z ----
    % (spherical: sph_theta, sph_phi, sph_radius; 2D: theta, radius)
    EEG.chanlocs = convertlocs(EEG.chanlocs, 'cart2all');
 
    fprintf('align_chanlocs: Done. All coordinate representations updated.\n');
end
 
 
function K = skew(v)
% SKEW  3x3 skew-symmetric (cross-product) matrix for vector v.
    K = [ 0,    -v(3),  v(2);
          v(3),  0,    -v(1);
         -v(2),  v(1),  0   ];
end