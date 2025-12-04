%% p_eval_bagging.m — Evaluate Bagging (RAW / HOG / PCA / LDA / LBP / EDGE / FUSION)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

%%  Load test indices and features 
S = load('splits/splits.mat'); 
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');                 % R.X_raw, R.y
H = load('features/hog/features_hog.mat');                 % H.X_hog
P = load('features/pca/features_pca.mat');                 % P.Xte_pca
L = load('features/lda/features_lda.mat');                 % L.Xte_lda
B = load('features/lbp/features_lbp.mat');                 % B.X_lbp
E = load('features/edge/features_edgehog.mat');            % E.X_edge
F = load('features/fusion/features_fusion_hog_lbp.mat');   % F.X_fusion

%%  Load Bagging models 
MR = load('models/bagging/modelBag_raw.mat');       % MR.modelBag_raw,    MR.bestRaw
MH = load('models/bagging/modelBag_hog.mat');       % MH.modelBag_hog,    MH.bestHog
MP = load('models/bagging/modelBag_pca.mat');       % MP.modelBag_pca,    MP.bestPca
ML = load('models/bagging/modelBag_lda.mat');       % ML.modelBag_lda,    ML.bestLda
MB = load('models/bagging/modelBag_lbp.mat');       % MB.modelBag_lbp,    MB.bestLbp
ME = load('models/bagging/modelBag_edge.mat');      % ME.modelBag_edge,   ME.bestEdge
MF = load('models/bagging/modelBag_fusion.mat');    % MF.modelBag_fusion, MF.bestFusion

%%  Prepare test data 
yte = double(R.y(testIdx));

Xte_raw    = double(R.X_raw(testIdx,:));
Xte_hog    = double(H.X_hog(testIdx,:));
Xte_pca    = double(P.Xte_pca);
Xte_lda    = double(L.Xte_lda);
Xte_lbp    = double(B.X_lbp(testIdx,:));
Xte_edge   = double(E.X_edge(testIdx,:));
Xte_fusion = double(F.X_fusion(testIdx,:));

fprintf('Test set: %d samples\n', numel(yte));
fprintf('  RAW   : %d dims\n', size(Xte_raw,2));
fprintf('  HOG   : %d dims\n', size(Xte_hog,2));
fprintf('  PCA   : %d dims\n', size(Xte_pca,2));
fprintf('  LDA   : %d dims\n', size(Xte_lda,2));
fprintf('  LBP   : %d dims\n', size(Xte_lbp,2));
fprintf('  EDGE  : %d dims\n', size(Xte_edge,2));
fprintf('  FUSION: %d dims\n', size(Xte_fusion,2));
%%  Predict  
fprintf('\nEvaluating Bagging models...\n');
[pred_raw,    t_raw]    = timedPredictBag(MR.modelBag_raw,    Xte_raw);
[pred_hog,    t_hog]    = timedPredictBag(MH.modelBag_hog,    Xte_hog);
[pred_pca,    t_pca]    = timedPredictBag(MP.modelBag_pca,    Xte_pca);
[pred_lda,    t_lda]    = timedPredictBag(ML.modelBag_lda,    Xte_lda);
[pred_lbp,    t_lbp]    = timedPredictBag(MB.modelBag_lbp,    Xte_lbp);
[pred_edge,   t_edge]   = timedPredictBag(ME.modelBag_edge,   Xte_edge);
[pred_fusion, t_fusion] = timedPredictBag(MF.modelBag_fusion, Xte_fusion);

%%  Compute metrics helper 
metrics = @(y,p) struct( ...
    'acc', mean(p==y)*100, ...
    'tp', sum(p==1 & y==1), ...
    'fp', sum(p==1 & y==0), ...
    'fn', sum(p==0 & y==1) ...
);

calcF1 = @(m) struct( ...
    'prec', m.tp / (m.tp + m.fp + eps), ...
    'rec',  m.tp / (m.tp + m.fn + eps) ...
);

%%  Compute metrics for each 
M.raw    = metrics(yte,pred_raw);
M.hog    = metrics(yte,pred_hog);
M.pca    = metrics(yte,pred_pca);
M.lda    = metrics(yte,pred_lda);
M.lbp    = metrics(yte,pred_lbp);
M.edge   = metrics(yte,pred_edge);
M.fusion = metrics(yte,pred_fusion);

fields = fieldnames(M);
for i = 1:numel(fields)
    Ff = fields{i};
    f  = calcF1(M.(Ff));
    M.(Ff).prec = f.prec;
    M.(Ff).rec  = f.rec;
    M.(Ff).f1   = 2*f.prec*f.rec/(f.prec+f.rec+eps);
end

%%  Display results summary 
fprintf('\nBAGGING (Bagged Trees) RESULTS\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | Trees | Leaf | Time(s)\n');
fprintf('----------------------------------------------------------------------------\n');
fprintf('RAW            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.raw.acc, M.raw.prec, M.raw.rec, M.raw.f1, MR.bestRaw.T,    MR.bestRaw.leaf,    t_raw);
fprintf('HOG            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.hog.acc, M.hog.prec, M.hog.rec, M.hog.f1, MH.bestHog.T,    MH.bestHog.leaf,    t_hog);
fprintf('PCA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.pca.acc, M.pca.prec, M.pca.rec, M.pca.f1, MP.bestPca.T,    MP.bestPca.leaf,    t_pca);
fprintf('LDA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.lda.acc, M.lda.prec, M.lda.rec, M.lda.f1, ML.bestLda.T,    ML.bestLda.leaf,    t_lda);
fprintf('LBP            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.lbp.acc, M.lbp.prec, M.lbp.rec, M.lbp.f1, MB.bestLbp.T,    MB.bestLbp.leaf,    t_lbp);
fprintf('EDGE           :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.edge.acc, M.edge.prec, M.edge.rec, M.edge.f1, ME.bestEdge.T,  ME.bestEdge.leaf,  t_edge);
fprintf('LBP+HOG        :  %6.2f  | %.3f  | %.3f  | %.3f  | %-5d | %-4d | %.4fs\n', ...
    M.fusion.acc, M.fusion.prec, M.fusion.rec, M.fusion.f1, MF.bestFusion.T, MF.bestFusion.leaf, t_fusion);

%%  Confusion matrices 
figure('Name','Bagging Confusion Matrices','Position',[100 100 1600 800]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

titles = {'RAW','HOG','PCA','LDA','LBP','EDGE','LBP+HOG'};
preds  = {pred_raw,pred_hog,pred_pca,pred_lda,pred_lbp,pred_edge,pred_fusion};

for i = 1:7
    nexttile;
    confusionchart(yte, preds{i}, ...
        'Title', sprintf('Bagging (%s)', titles{i}), ...
        'RowSummary', 'off', ...
        'ColumnSummary', 'off');
end

%%  TP/TN/FP/FN image grids (for RAW and HOG, using RAW images) 
rawHW  = [128 64];  % image size used for raw features
maxShow = 64;

Xte_raw_full = double(R.X_raw(testIdx,:));   % for visualization

% indices for RAW
idxTP_raw = find(pred_raw==1 & yte==1);
idxTN_raw = find(pred_raw==0 & yte==0);
idxFP_raw = find(pred_raw==1 & yte==0);
idxFN_raw = find(pred_raw==0 & yte==1);

% indices for HOG
idxTP_hog = find(pred_hog==1 & yte==1);
idxTN_hog = find(pred_hog==0 & yte==0);
idxFP_hog = find(pred_hog==1 & yte==0);
idxFN_hog = find(pred_hog==0 & yte==1);

showIndexGridFromFeatures(Xte_raw_full, idxTP_raw, maxShow, 'Bagging RAW — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxTN_raw, maxShow, 'Bagging RAW — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFP_raw, maxShow, 'Bagging RAW — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFN_raw, maxShow, 'Bagging RAW — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw_full, idxTP_hog, maxShow, 'Bagging HOG — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxTN_hog, maxShow, 'Bagging HOG — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFP_hog, maxShow, 'Bagging HOG — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFN_hog, maxShow, 'Bagging HOG — False Negatives', rawHW);

%%  Local helper: showIndexGridFromFeatures 
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

%%  Local helper: timed prediction for Bagging 
function [pred, t] = timedPredictBag(model, X)
    tic;
    ypredCell = predict(model, X);   % TreeBagger returns cell array of class labels
    t = toc / size(X,1);
    if iscell(ypredCell)
        pred = str2double(ypredCell);
    else
        pred = ypredCell;
    end
end