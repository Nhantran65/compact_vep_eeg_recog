function Preprocess_Step2
% Step 2: Average re-referencing + Bandpass filtering 0.1Hz to 50Hz
% Input  : EEGLAB-SET_STEP1_ICA/<class>/<phase>/*_ICAclean.set
% Output : EEGLAB-SET_STEP2_FILTERED/<class>/<phase>/*_ICAclean_filtered.set
% Report : filtering_report.csv at output root

clc; fprintf('\n=== STEP 2: Re-referencing + Bandpass Filtering (0.1-50 Hz) ===\n');

inRoot  = 'EEGLAB-SET_STEP1_ICA';
outRoot = 'EEGLAB-SET_STEP2_FILTERED';
lowFreq = 0.1;   % High-pass filter cutoff (Hz)
highFreq = 50;   % Low-pass filter cutoff (Hz)

assert(~isempty(which('eeglab')), 'Please add EEGLAB to the MATLAB path.');
eeglab nogui;

sets = dir(fullfile(inRoot, '**', '*.set'));
if isempty(sets)
    error('No .set files found under %s', inRoot);
end

% report
repHeaders = {'rel_path','nbchan','pnts','srate','orig_srate','lowcut','highcut','status'};
rep = strings(0, numel(repHeaders));

for i = 1:numel(sets)
    inPath  = fullfile(sets(i).folder, sets(i).name);
    
    % Get relative path properly
    fullInRoot = fullfile(pwd, inRoot);
    if startsWith(sets(i).folder, fullInRoot)
        relPath = erase(sets(i).folder, [fullInRoot filesep]);
    else
        relPath = erase(sets(i).folder, [inRoot filesep]);
    end

    fprintf('>> %s\n', inPath);

    try
        % Load cleaned data from Step 1
        EEG = pop_loadset(inPath);
        EEG = eeg_checkset(EEG);
        
        origSrate = EEG.srate;
        
        % ---- Re-reference to average reference ----
        fprintf('   Re-referencing to average reference...\n');
        EEG = pop_reref(EEG, []);  % [] means average reference of all channels
        EEG = eeg_checkset(EEG);
        
        % ---- Bandpass filter: 0.1 - 50 Hz ----
        % Using pop_eegfiltnew (firfilt plugin) - recommended for EEGLAB
        % If firfilt is not available, it will fall back to basic filtering
        
        if ~isempty(which('pop_eegfiltnew'))
            % Use firfilt plugin (better, FIR filter)
            fprintf('   Applying bandpass filter: %.1f - %.1f Hz (FIR)...\n', lowFreq, highFreq);
            EEG = pop_eegfiltnew(EEG, 'locutoff', lowFreq, 'hicutoff', highFreq);
        else
            % Fallback to basic EEGLAB filter
            fprintf('   Applying bandpass filter: %.1f - %.1f Hz (basic)...\n', lowFreq, highFreq);
            EEG = pop_eegfilt(EEG, lowFreq, 0);      % high-pass
            EEG = pop_eegfilt(EEG, 0, highFreq);     % low-pass
        end
        
        EEG = eeg_checkset(EEG);
        
        % ---- Optional: Notch filter for line noise (50Hz or 60Hz) ----
        % Uncomment if you see 50Hz/60Hz noise in your data
        % fprintf('   Applying notch filter at 50 Hz...\n');
        % EEG = pop_eegfiltnew(EEG, 'locutoff', 48, 'hicutoff', 52, 'revfilt', 1);
        % EEG = eeg_checkset(EEG);

        % ---- Save filtered set ----
        outDir = fullfile(outRoot, relPath);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        
        [~, base] = fileparts(inPath);
        % Remove _ICAclean suffix and add _filtered
        base = strrep(base, '_ICAclean', '');
        EEG.setname = [base '_ICAclean_filtered'];
        outFile = fullfile(outDir, [EEG.setname '.set']);

        pop_saveset(EEG, 'filename', outFile, ...
                    'savemode', 'onefile', 'version', '7.3');

        % ---- Append to report ----
        rep(end+1, :) = [ fullfile(relPath, sets(i).name), ...
                          string(EEG.nbchan), string(EEG.pnts), ...
                          string(EEG.srate), string(origSrate), ...
                          string(lowFreq), string(highFreq), "success" ];

    catch ME
        fprintf(2, '   !! ERROR: %s\n', ME.message);
        % Log the failure
        rep(end+1, :) = [ fullfile(relPath, sets(i).name), "-", "-", "-", "-", ...
                          string(lowFreq), string(highFreq), "error: " + string(ME.message) ];
        continue
    end
end

% Write report
repFile = fullfile(outRoot, 'filtering_report.csv');
writecell([repHeaders; cellstr(rep)], repFile);
fprintf('\nDone. Filtered sets -> %s\nReport -> %s\n', outRoot, repFile);
fprintf('Total processed: %d files\n', size(rep, 1));
end
