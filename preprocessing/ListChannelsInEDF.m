function ListChannelsInEDF(edfRoot)
% Liệt kê các kênh (cột) có trong mỗi file EDF
% Dùng cho dataset EEG (VEP-EDF)
%
% Ví dụ: ListChannelsInEDF('VEP-EDF/Apple/A1')

if nargin==0
    edfRoot = 'VEP-EDF/Apple/A1';
end

addpath(fullfile(fileparts(which('eeglab.m')),'plugins','biosig'));
edfs = dir(fullfile(edfRoot,'*.edf'));

fprintf('File, NumChannels, ChannelLabels\n');
fprintf('-------------------------------------------\n');

[dat, hdr] = sload('VEP-EDF/Apple/A1/sub3_A1.edf');
disp(string(hdr.Label)');
if isfield(hdr,'EVENT') && isfield(hdr.EVENT,'TYP') && ~isempty(hdr.EVENT.TYP)
    % hdr.EVENT.TYP  : mã sự kiện (vector int16/string)
    % hdr.EVENT.POS  : vị trí theo mẫu (sample index, 1-based)
    % hdr.EVENT.DUR  : độ dài theo mẫu
    ev = table(hdr.EVENT.POS(:), hdr.EVENT.TYP(:), hdr.EVENT.DUR(:), ...
        'VariableNames', {'pos','typ','dur'});
    disp(ev(1:min(10,height(ev)),:));
    % Sau đó đẩy vào EEGLAB:
    EEG = pop_biosig(edfFile);
    % (EEGLAB đã tự thêm EEG.event từ hdr.EVENT)
    return;
else
    disp("empty");
end
end
