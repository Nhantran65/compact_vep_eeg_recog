function ImportRawToSET
% ImportRawToSET.m — Import EDF (Emotiv 14ch) and save only .SET files
%   No preprocessing, no reref, no filtering
%   Only keeps 14 EEG channels, no EDF copies.

clc; fprintf('\n=== Importing EDF → .SET (RAW 14ch, no EDF duplication) ===\n');

% ---------- CONFIG ----------
dataRoot = 'VEP-EDF';           % source EDF folders
outRoot  = 'EEGLAB-SET';        % destination for .set
classes  = {'Apple','Car','Flower','Human Face'};
phases   = {{'A1','A2'},{'C1','C2'},{'F1','F2'},{'P1','P2'}};

% Emotiv EPOC-X channel names (order to keep)
emotiv14 = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};

% ---------- Initialize EEGLAB ----------
assert(~isempty(which('eeglab')), 'Add EEGLAB to MATLAB path first.');
eeglab nogui;
addpath(fullfile(fileparts(which('eeglab.m')),'plugins','biosig')); % ensure BIOSIG available

for i = 1:numel(classes)
    for j = 1:numel(phases{i})
        inDir  = fullfile(dataRoot, classes{i}, phases{i}{j});
        if ~exist(inDir,'dir'), continue; end
        outDir = fullfile(outRoot, classes{i}, phases{i}{j});
        if ~exist(outDir,'dir'), mkdir(outDir); end

        edfs = dir(fullfile(inDir, '*.edf'));
        if isempty(edfs)
            fprintf('!! No EDF in %s\n', inDir);
            continue;
        end

        for k = 1:numel(edfs)
            inFile = fullfile(edfs(k).folder, edfs(k).name);
            [~, base, ~] = fileparts(inFile);
            fprintf('>> Importing: %s\n', inFile);

            try
                % --- Import EDF (no event extraction needed yet)
                EEG = pop_biosig(inFile, 'importevent','on', 'rmeventchan','on');
                EEG = eeg_checkset(EEG);

                % --- Keep only 14 EEG channels
                EEG = keep14Emotiv(EEG, emotiv14);

                % --- Assign channel locations for visualization
                EEG = pop_chanedit(EEG, 'lookup', fullfile(fileparts(which('eeglab.m')),...
                    'plugins','dipfit','standard_BEM','elec','standard_1005.elc'));

                % --- Ensure sample rate present
                if isempty(EEG.srate) || EEG.srate==0
                    EEG.srate = 128;
                end

                % --- Save .set only (no EDF duplication)
                EEG.setname  = sprintf('%s_%s_%s_RAW14', classes{i}, phases{i}{j}, base);
                EEG.filename = [EEG.setname '.set'];
                EEG.filepath = outDir;
                pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath, ...
                                'savemode','onefile', 'version','7.3');  % <- no .fdt


            catch ME
                fprintf(2, '   !! ERROR %s: %s\n', base, ME.message);
            end
        end
    end
end

fprintf('\n=== DONE. Only .SET files saved under: %s ===\n', outRoot);
end


% ---------- Helper: select only Emotiv 14 EEG channels ----------
function EEG = keep14Emotiv(EEG, emotiv14)
labels = string({EEG.chanlocs.labels});
want   = upper(string(emotiv14));
idx    = zeros(1, numel(want));
for c = 1:numel(want)
    hit = find(upper(labels) == want(c), 1);
    if ~isempty(hit), idx(c) = hit; end
end
idx(idx==0) = [];
EEG = pop_select(EEG, 'channel', idx);
for c = 1:min(14, EEG.nbchan)
    EEG.chanlocs(c).labels = emotiv14{c};
end
EEG = eeg_checkset(EEG);
end
