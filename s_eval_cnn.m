%% s_eval_cnn.m — Evaluate CNNs (RAW and HOG)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

%% Load splits & feature sets  
S = load('splits/splits.mat'); 
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');  
H = load('features/hog/features_hog.mat');   

MR = load('models/cnn/modelCNN_raw.mat');   
MH = load('models/cnn/modelCNN_hog.mat');    
netCNN_raw = MR.netCNN_raw;
rawHW      = MR.rawHW;

netCNN_hog = MH.netCNN_hog;
nBlocksY   = MH.nBlocksY;
nBlocksX   = MH.nBlocksX;
channels   = MH.channels;

%%   Prepare RAW test data  
y_all = double(R.y(:));
X_all_raw = double(R.X_raw);

yte = y_all(testIdx);
Xte_raw_flat = X_all_raw(testIdx,:);
Ntest = numel(yte);

fprintf('Test set: %d samples\n', Ntest);

Xte_raw_4d = reshape(Xte_raw_flat', rawHW(1), rawHW(2), 1, Ntest);
Xte_raw_4d = mat2gray(Xte_raw_4d);

%%   Prepare HOG test data  
X_all_hog = double(H.X_hog);
y_all_hog = double(H.y(:)); 

Xte_hog_flat = X_all_hog(testIdx,:);
Ntest_hog = numel(y_all_hog(testIdx));

Xte_hog_4d = reshape(Xte_hog_flat', channels, nBlocksX, nBlocksY, Ntest_hog);
Xte_hog_4d = permute(Xte_hog_4d, [3 2 1 4]);  
Xte_hog_4d = mat2gray(Xte_hog_4d);

%%   Predict 
fprintf('\nEvaluating CNN models...\n');

% RAW CNN
tic;
ypred_raw_cat = classify(netCNN_raw, Xte_raw_4d);
t_raw = toc / Ntest;
ypred_raw = double(ypred_raw_cat == '1');

% HOG CNN
tic;
ypred_hog_cat = classify(netCNN_hog, Xte_hog_4d);
t_hog = toc / Ntest_hog;
ypred_hog = double(ypred_hog_cat == '1');

%%   Metrics helpers 
metrics = @(y,p) struct( ...
    'acc', mean(p==y)*100, ...
    'tp', sum(p==1 & y==1), ...
    'tn', sum(p==0 & y==0), ...
    'fp', sum(p==1 & y==0), ...
    'fn', sum(p==0 & y==1) ...
);

calcF1 = @(m) struct( ...
    'prec', m.tp / (m.tp + m.fp + eps), ...
    'rec',  m.tp / (m.tp + m.fn + eps) ...
);

%%   Compute metrics  
M.raw = metrics(yte, ypred_raw);
M.hog = metrics(yte, ypred_hog);

% RAW
f_raw = calcF1(M.raw);
M.raw.prec = f_raw.prec;
M.raw.rec  = f_raw.rec;
M.raw.f1   = 2*f_raw.prec*f_raw.rec/(f_raw.prec+f_raw.rec+eps);

% HOG
f_hog = calcF1(M.hog);
M.hog.prec = f_hog.prec;
M.hog.rec  = f_hog.rec;
M.hog.f1   = 2*f_hog.prec*f_hog.rec/(f_hog.prec+f_hog.rec+eps);

%%   Performance summary  
fprintf('\nCNN RESULTS\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | Time(s/sample)\n');
fprintf('                     --\n');
fprintf('RAW (CNN)      :  %6.2f  | %.3f  | %.3f  | %.3f  | %.4fs\n', ...
    M.raw.acc, M.raw.prec, M.raw.rec, M.raw.f1, t_raw);
fprintf('HOG (CNN)      :  %6.2f  | %.3f  | %.3f  | %.3f  | %.4fs\n', ...
    M.hog.acc, M.hog.prec, M.hog.rec, M.hog.f1, t_hog);

%%   NEW: TP/TN/FP/FN Summary Table  
fprintf('\nTP/TN/FP/FN SUMMARY\n');
fprintf('Descriptor     |   TP   |   TN   |   FP   |   FN\n');
fprintf('                \n');
fprintf('RAW (CNN)      : %6d | %6d | %6d | %6d\n', M.raw.tp, M.raw.tn, M.raw.fp, M.raw.fn);
fprintf('HOG (CNN)      : %6d | %6d | %6d | %6d\n', M.hog.tp, M.hog.tn, M.hog.fp, M.hog.fn);

%%   Confusion matrices  
figure('Name','CNN Confusion Matrices','Position',[100 100 800 400]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
confusionchart(yte, ypred_raw, 'Title','CNN (RAW)', ...
    'RowSummary','off','ColumnSummary','off');

nexttile;
confusionchart(yte, ypred_hog, 'Title','CNN (HOG)', ...
    'RowSummary','off','ColumnSummary','off');

%%   TP/TN/FP/FN Image Grids  
rawHW = rawHW;
maxShow = 64;

Xte_raw_vis = Xte_raw_flat;

% RAW
idxTP_raw = find(ypred_raw==1 & yte==1);
idxTN_raw = find(ypred_raw==0 & yte==0);
idxFP_raw = find(ypred_raw==1 & yte==0);
idxFN_raw = find(ypred_raw==0 & yte==1);

% HOG
idxTP_hog = find(ypred_hog==1 & yte==1);
idxTN_hog = find(ypred_hog==0 & yte==0);
idxFP_hog = find(ypred_hog==1 & yte==0);
idxFN_hog = find(ypred_hog==0 & yte==1);

showIndexGridFromFeatures(Xte_raw_vis, idxTP_raw, maxShow, 'CNN RAW — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxTN_raw, maxShow, 'CNN RAW — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFP_raw, maxShow, 'CNN RAW — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFN_raw, maxShow, 'CNN RAW — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw_vis, idxTP_hog, maxShow, 'CNN HOG — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxTN_hog, maxShow, 'CNN HOG — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFP_hog, maxShow, 'CNN HOG — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFN_hog, maxShow, 'CNN HOG — False Negatives', rawHW);

%%   Helper  
function showIndexGridFromFeatures(X, idx, maxShow, figTitle, imgHW)
    figure('Name',figTitle,'NumberTitle','off');
    nShow = min(maxShow, numel(idx));
    if nShow == 0
        sgtitle([figTitle ' (none)']);
        return;
    end
    nCols = 8;
    nRows = ceil(nShow / nCols);
    tiledlayout(nRows, nCols, 'Padding','compact','TileSpacing','compact');
    for k = 1:nShow
        nexttile;
        img = reshape(X(idx(k),:), imgHW);
        imshow(mat2gray(img), []);
        axis off;
    end
    sgtitle(figTitle);
end