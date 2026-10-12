function Preprocess_Step3
% Step 3: Sliding window epoching (for continuous EEG analysis)
% Input  : EEGLAB-SET_STEP2_FILTERED/<class>/<phase>/*_filtered.set
% Output : EEGLAB-SET_STEP3_EPOCHED/<class>/<phase>/*_epoched.set
% Report : epoching_report.csv at output root
%
% Creates overlapping epochs for continuous EEG analysis
% Use when: stimulus markers are unavailable or for spectral/ML features

clc; fprintf('\n=== STEP 3: Sliding Window Epoching ===\n');

inRoot  = 'EEGLAB-SET_STEP2_FILTERED';
outRoot = 'EEGLAB-SET_STEP3_EPOCHED';

% Sliding window parameters
windowSize = 1.0;    % 1 second window
overlap = 0.5;       % 50% overlap (0.5 sec step)
                     % Set to 0.0 for non-overlapping windows (recommended for ML/Foundation Models)

assert(~isempty(which('eeglab')), 'Please add EEGLAB to the MATLAB path.');
eeglab nogui;

sets = dir(fullfile(inRoot, '**', '*.set'));
if isempty(sets)
    error('No .set files found under %s', inRoot);
end

% report
repHeaders = {'rel_path','nbchan','orig_pnts','window_sec','overlap_pct','n_epochs','status'};
rep = strings(0, numel(repHeaders));

for i = 1:numel(sets)
    inPath  = fullfile(sets(i).folder, sets(i).name);
    
    % Get relative path
    fullInRoot = fullfile(pwd, inRoot);
    if startsWith(sets(i).folder, fullInRoot)
        relPath = erase(sets(i).folder, [fullInRoot filesep]);
    else
        relPath = erase(sets(i).folder, [inRoot filesep]);
    end

    fprintf('>> %s\n', inPath);

    try
        % Load filtered data from Step 2
        EEG = pop_loadset(inPath);
        EEG = eeg_checkset(EEG);
        
        origPnts = EEG.pnts;
        srate = EEG.srate;
        
        % ---- Create artificial events for sliding windows ----
        windowSamples = round(windowSize * srate);
        stepSamples = round(windowSize * srate * (1 - overlap));
        
        % Calculate number of possible epochs
        nEpochs = floor((EEG.pnts - windowSamples) / stepSamples) + 1;
        
        fprintf('   Window: %.1f sec, Overlap: %.0f%%, Step: %.2f sec\n', ...
                windowSize, overlap*100, stepSamples/srate);
        fprintf('   Creating %d epochs\n', nEpochs);
        
        % Clear existing events
        EEG.event = [];
        
        % Create artificial events at each window start
        for j = 1:nEpochs
            latency = (j-1) * stepSamples + 1;
            if latency + windowSamples - 1 <= EEG.pnts
                EEG.event(j).type = 'window';
                EEG.event(j).latency = latency;
                EEG.event(j).duration = 1;
            end
        end
        
        EEG = eeg_checkset(EEG, 'eventconsistency');
        
        % ---- Extract epochs ----
        % Extract full 1-second windows (not 0.5s!)
        epochLimits = [0 windowSize];  % 0 to 1.0 seconds
        EEG = pop_epoch(EEG, {'window'}, epochLimits, 'epochinfo', 'yes');
        EEG = eeg_checkset(EEG);
        
        actualEpochs = EEG.trials;
        fprintf('   Created %d epochs\n', actualEpochs);
        
        % ---- Optional: Remove bad epochs ----
        % Reject epochs with extreme values (> ±150 µV)
        % Increased from ±100 to ±150 to reduce overly aggressive rejection
        EEG = pop_eegthresh(EEG, 1, 1:EEG.nbchan, -150, 150, 0, windowSize, 0, 1);
        EEG = eeg_checkset(EEG);
        
        finalEpochs = EEG.trials;
        if finalEpochs < actualEpochs
            fprintf('   Rejected %d bad epochs (kept %d)\n', actualEpochs - finalEpochs, finalEpochs);
        end

        % ---- Save epoched set ----
        outDir = fullfile(outRoot, relPath);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        
        [~, base] = fileparts(inPath);
        base = strrep(base, '_ICAclean_filtered', '');
        EEG.setname = [base '_epoched'];
        outFile = fullfile(outDir, [EEG.setname '.set']);

        pop_saveset(EEG, 'filename', outFile, 'savemode', 'onefile', 'version', '7.3');

        % ---- Append to report ----
        rep(end+1, :) = [ fullfile(relPath, sets(i).name), ...
                          string(EEG.nbchan), string(origPnts), ...
                          string(windowSize), string(overlap*100), ...
                          string(finalEpochs), "success" ];

    catch ME
        fprintf(2, '   !! ERROR: %s\n', ME.message);
        rep(end+1, :) = [ fullfile(relPath, sets(i).name), "-", "-", ...
                          string(windowSize), string(overlap*100), "-", ...
                          "error: " + string(ME.message) ];
        continue
    end
end

% Write report
repFile = fullfile(outRoot, 'epoching_report.csv');
writecell([repHeaders; cellstr(rep)], repFile);
fprintf('\nDone. Epoched sets -> %s\nReport -> %s\n', outRoot, repFile);
fprintf('Total processed: %d files\n', size(rep, 1));
end
