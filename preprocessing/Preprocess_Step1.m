function preprocess
% Step 1: ICA -> ICLabel -> remove Eye/Muscle >= 0.90
% Input  : EEGLAB-SET/<class>/<phase>/*.set
% Output : EEGLAB-SET_STEP1_ICA/<class>/<phase>/*_ICAclean.set
% Report : iclabel_removal_report.csv at output root

clc; fprintf('\n=== STEP 1: ICA + ICLabel (Eye/Muscle >= 0.90 removal) ===\n');

inRoot  = 'EEGLAB-SET';
outRoot = 'EEGLAB-SET_STEP1_ICA';
thresh  = 0.90;                   % probability threshold

assert(~isempty(which('eeglab')), 'Please add EEGLAB to the MATLAB path.');
eeglab nogui;

% quick check ICLabel is available
if isempty(which('iclabel'))
    error('ICLabel plugin not found. In EEGLAB: File -> Manage extensions -> Install "ICLabel".');
end

sets = dir(fullfile(inRoot, '**', '*.set'));
if isempty(sets)
    error('No .set files found under %s', inRoot);
end

% report
repHeaders = {'rel_path','nbchan','pnts','nIC','removed_idx','removed_reason'};
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
        EEG = pop_loadset(inPath);
        EEG = eeg_checkset(EEG);

        % ---- Check if data is loaded ----
        if isempty(EEG.data)
            error('No data loaded from file');
        end

        % ---- Check/fix sampling rate for ICLabel (requires >= 100 Hz) ----
        if isempty(EEG.srate) || EEG.srate < 100
            if isempty(EEG.srate) || EEG.srate == 0
                fprintf('   Warning: srate not set. Setting to 128 Hz and resampling...\n');
                EEG.srate = 128;
            else
                fprintf('   Warning: srate = %.1f Hz < 100 Hz. Resampling to 128 Hz...\n', EEG.srate);
            end
            EEG = pop_resample(EEG, 128);
            EEG = eeg_checkset(EEG);
        end

        % ---- ICA (runica) on current data ----
        % (no filtering/re-ref here per your request)
        EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'off');
        EEG = eeg_checkset(EEG);

        % ---- ICLabel classification ----
        EEG = iclabel(EEG, 'default');
        probs = EEG.etc.ic_classification.ICLabel.classifications;
        % ICLabel class order: [Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other]
        isMuscle = probs(:,2) >= thresh;
        isEye    = probs(:,3) >= thresh;

        rej = find(isMuscle | isEye);

        reason = strings(size(rej));
        for k = 1:numel(rej)
            if isMuscle(rej(k)) && isEye(rej(k))
                reason(k) = 'Muscle+Eye';
            elseif isMuscle(rej(k))
                reason(k) = 'Muscle';
            else
                reason(k) = 'Eye';
            end
        end

        if ~isempty(rej)
            EEG = pop_subcomp(EEG, rej, 0);   % remove bad ICs
            EEG = eeg_checkset(EEG);
        end

        % ---- save cleaned set ----
        outDir = fullfile(outRoot, relPath);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        [~, base] = fileparts(inPath);
        EEG.setname  = [base '_ICAclean'];
        outFile = fullfile(outDir, [base '_ICAclean.set']);

        pop_saveset(EEG, 'filename', outFile, ...
                    'savemode', 'onefile', 'version', '7.3');

        % ---- append to report ----
        if isempty(rej)
            rep(end+1, :) = [ fullfile(relPath, sets(i).name), ...
                              string(EEG.nbchan), string(EEG.pnts), ...
                              string(size(probs,1)), "[]", "-" ];
        else
            rep(end+1, :) = [ fullfile(relPath, sets(i).name), ...
                              string(EEG.nbchan), string(EEG.pnts), ...
                              string(size(probs,1)), ...
                              "[" + strjoin(string(rej.'), ',') + "]", ...
                              "[" + strjoin(reason.', ',') + "]" ];
        end

    catch ME
        fprintf(2, '   !! ERROR: %s\n', ME.message);
        % still log the failure
        rep(end+1, :) = [ fullfile(relPath, sets(i).name), "-", "-", "-", "error", string(ME.message) ];
        continue
    end
end

% write report
repFile = fullfile(outRoot, 'iclabel_removal_report.csv');
writecell([repHeaders; cellstr(rep)], repFile);
fprintf('\nDone. Cleaned sets -> %s\nReport -> %s\n', outRoot, repFile);
end
