

function ScanEDFMarkers(edfRoot)
% Quick report of marker presence in all EDFs
if nargin==0, edfRoot = 'VEP-EDF'; end
eeglab nogui;
addpath(fullfile(fileparts(which('eeglab.m')),'plugins','biosig'));

edfs = dir(fullfile(edfRoot,'**','*.edf'));
fprintf('file, has_MarkerValueInt, nonzeros, has_MarkerIndex, nonzeros, has_MarkerType\n');

for i = 1:numel(edfs)
    f = fullfile(edfs(i).folder, edfs(i).name);
    try
        [dat, hdr] = sload(f);                % N x Ch
        labs = string(hdr.Label);

        ixVal  = find(labs=="MarkerValueInt",1);
        ixIdx  = find(labs=="MarkerIndex",   1);
        ixType = find(labs=="MarkerType",    1);

        hasVal = ~isempty(ixVal);
        hasIdx = ~isempty(ixIdx);
        hasTyp = ~isempty(ixType);

        nzVal = 0; nzIdx = 0;
        if hasVal, nzVal = nnz(~isnan(dat(:,ixVal)) & dat(:,ixVal)~=0); end
        if hasIdx, nzIdx = nnz(~isnan(dat(:,ixIdx)) & dat(:,ixIdx)>0);  end

        fprintf('%s,%d,%d,%d,%d,%d\n', f, hasVal, nzVal, hasIdx, nzIdx, hasTyp);
    catch ME
        fprintf('%s,ERROR,%s\n', f, ME.message);
    end
end
end

