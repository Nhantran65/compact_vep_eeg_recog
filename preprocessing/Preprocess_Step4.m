function Preprocess_Step4
% Step 4: Export epochs to CSV format for SVM classification
% Input  : EEGLAB-SET_STEP3_EPOCHED/<class>/<phase>/*_epoched.set
% Output : CSV-FEATURES/<class>/<phase>/*.csv
%
% Each CSV file contains:
% - Rows: epochs
% - Columns: [Channel1_t1, Channel1_t2, ..., Channel14_t128, Class, Phase, Subject]
% - Flattened time series per channel per epoch

clc; fprintf('\n=== STEP 4: Export Epochs to CSV for SVM ===\n');

inRoot  = 'EEGLAB-SET_STEP3_EPOCHED';
outRoot = 'CSV-FEATURES';

assert(~isempty(which('eeglab')), 'Please add EEGLAB to the MATLAB path.');
eeglab nogui;

sets = dir(fullfile(inRoot, '**', '*.set'));
if isempty(sets)
    error('No .set files found under %s', inRoot);
end

% Classes mapping
classMap = containers.Map({'Apple', 'Car', 'Flower', 'Human Face'}, [1, 2, 3, 4]);

fprintf('Found %d files to process\n\n', numel(sets));

for i = 1:numel(sets)
    inPath  = fullfile(sets(i).folder, sets(i).name);
    
    % Get relative path
    fullInRoot = fullfile(pwd, inRoot);
    if startsWith(sets(i).folder, fullInRoot)
        relPath = erase(sets(i).folder, [fullInRoot filesep]);
    else
        relPath = erase(sets(i).folder, [inRoot filesep]);
    end

    fprintf('>> [%d/%d] %s\n', i, numel(sets), inPath);

    try
        % Load epoched data from Step 3
        EEG = pop_loadset(inPath);
        EEG = eeg_checkset(EEG);
        
        nEpochs = EEG.trials;
        nChannels = EEG.nbchan;
        nTimepoints = EEG.pnts;
        
        fprintf('   Epochs: %d, Channels: %d, Timepoints: %d\n', nEpochs, nChannels, nTimepoints);
        
        % Extract class and phase from path
        pathParts = split(relPath, filesep);
        className = pathParts{1};
        phaseName = pathParts{2};
        
        % Extract subject number from filename
        [~, fname] = fileparts(sets(i).name);
        subjectMatch = regexp(fname, 'sub(\d+)', 'tokens');
        if ~isempty(subjectMatch)
            subjectNum = str2double(subjectMatch{1}{1});
        else
            subjectNum = 0;
        end
        
        % Get class label
        if isKey(classMap, className)
            classLabel = classMap(className);
        else
            classLabel = 0;
        end
        
        % ---- Prepare data matrix ----
        % Flatten each epoch: [channels x timepoints] -> [1 x (channels*timepoints)]
        dataMatrix = zeros(nEpochs, nChannels * nTimepoints + 3); % +3 for class, phase, subject
        
        for ep = 1:nEpochs
            % Get epoch data: [channels x timepoints]
            epochData = EEG.data(:, :, ep);
            
            % Flatten: reshape to [1 x (channels*timepoints)]
            flatData = reshape(epochData', 1, []);  % transpose first to keep time-major order
            
            % Add metadata
            dataMatrix(ep, :) = [flatData, classLabel, str2double(phaseName(2)), subjectNum];
        end
        
        % ---- Create column headers ----
        headers = cell(1, nChannels * nTimepoints + 3);
        idx = 1;
        for ch = 1:nChannels
            chLabel = EEG.chanlocs(ch).labels;
            for t = 1:nTimepoints
                headers{idx} = sprintf('%s_t%d', chLabel, t);
                idx = idx + 1;
            end
        end
        headers{end-2} = 'Class';
        headers{end-1} = 'Phase';
        headers{end} = 'Subject';
        
        % ---- Save to CSV ----
        outDir = fullfile(outRoot, className, phaseName);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        
        [~, base] = fileparts(inPath);
        base = strrep(base, '_epoched', '');
        csvFile = fullfile(outDir, [base '_epochs.csv']);
        
        % Write CSV
        dataTable = array2table(dataMatrix, 'VariableNames', headers);
        writetable(dataTable, csvFile);
        
        fprintf('   Saved: %s (%d epochs)\n', csvFile, nEpochs);

    catch ME
        fprintf(2, '   !! ERROR: %s\n', ME.message);
        continue
    end
end

fprintf('\n=== Done! CSV files saved to: %s ===\n', outRoot);
fprintf('Total files processed: %d\n', numel(sets));

% ---- Create summary CSV with all metadata ----
fprintf('\nCreating summary file...\n');
summaryFile = fullfile(outRoot, 'dataset_summary.csv');
createSummary(inRoot, summaryFile);

end

%% Helper function to create dataset summary
function createSummary(inRoot, summaryFile)
sets = dir(fullfile(inRoot, '**', '*.set'));
summary = cell(numel(sets)+1, 6);
summary(1, :) = {'Filename', 'Class', 'Phase', 'Subject', 'nEpochs', 'nChannels'};

for i = 1:numel(sets)
    try
        inPath = fullfile(sets(i).folder, sets(i).name);
        EEG = pop_loadset(inPath);
        
        fullInRoot = fullfile(pwd, inRoot);
        if startsWith(sets(i).folder, fullInRoot)
            relPath = erase(sets(i).folder, [fullInRoot filesep]);
        else
            relPath = erase(sets(i).folder, [inRoot filesep]);
        end
        
        pathParts = split(relPath, filesep);
        className = pathParts{1};
        phaseName = pathParts{2};
        
        [~, fname] = fileparts(sets(i).name);
        subjectMatch = regexp(fname, 'sub(\d+)', 'tokens');
        if ~isempty(subjectMatch)
            subjectNum = str2double(subjectMatch{1}{1});
        else
            subjectNum = 0;
        end
        
        summary(i+1, :) = {sets(i).name, className, phaseName, subjectNum, EEG.trials, EEG.nbchan};
    catch
        summary(i+1, :) = {sets(i).name, 'ERROR', '', 0, 0, 0};
    end
end

writecell(summary, summaryFile);
fprintf('Summary saved to: %s\n', summaryFile);
end
