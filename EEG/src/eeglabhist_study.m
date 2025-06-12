% EEGLAB history file generated on the 01-Jun-2025
% ------------------------------------------------
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
eeglab('redraw');
EEG = pop_loadset('filename','pilot_sp_cleaned_2ndICA_dipole_fit_voice_marked_epoched.set','filepath','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets/');
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
[STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'name','pilot_sp_ICs','task','Overt spoken task: 6 CV syllables','commands',{{'inbrain','on','dipselect',0.4},{'index',1,'subject','1','session',1,'run',1}},'updatedat','on','rmclust','on' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
STUDY = std_makedesign(STUDY, ALLEEG, 1, 'name','STUDY.design 1','delfiles','off','defaultdesign','off','variable1','label','values1',{'giSP','guSP','miSP','muSP','siSP','suSP'},'vartype1','categorical');
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'components','savetrials','on','recompute','on','ersp','on','erspparams',{'baseline',0,'alpha',0.05,'freqs',[85 175] ,'mcorrect','fdr','freqscale','log','plotphase','off','ntimesout',400,'padratio',2,'basenorm','on','trialbase','full'});

STUDY = pop_erspparams(STUDY, 'timerange',[-500 1500] ,'freqrange',[85 175] );
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'components','savetrials','on','recompute','on','ersp','on','erspparams',{'cycles',8,'freqs',[85 175] ,'ntimesout',400,'padratio',2});
[STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename','pilot_sp_study_all.study','filepath','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets/','resavedatasets','on');
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
CURRENTSTUDY = 0;[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'retrieve',1,'study',1); 
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'retrieve',1,'study',1); CURRENTSTUDY = 1;
eeglab redraw;
