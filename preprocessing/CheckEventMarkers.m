[dat, hdr] = sload('VEP-EDF/Apple/A1/sub3_A1.edf');
labs = string(hdr.Label);
ixIdx = find(labs=="MarkerIndex",1);
ixType = find(labs=="MarkerType",1);
ixVal = find(labs=="MarkerValueInt",1);
fprintf('Unique counts: Index=%d, Type=%d, Value=%d\n', ...
    numel(unique(dat(:,ixIdx))), numel(unique(dat(:,ixType))), numel(unique(dat(:,ixVal))));
