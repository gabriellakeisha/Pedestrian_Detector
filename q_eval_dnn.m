%% q_eval_dnn.m — Evaluate Deep Neural Network (RAW / HOG / PCA / LDA / LBP / EDGE / FUSION)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

%%  Load test indices and features  
S = load('splits/splits.mat'); 
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P = load('features/pca/features_pca.mat');
L = load('features/lda/features_lda.mat');
B = load('features/lbp/features_lbp.mat');
E = load('features/edge/features_edgehog.mat');
F = load('features/fusion/features_fusion_hog_lbp.mat');

%%  Load DNN models  
MR = load('models/dnn/modelDNN_raw.mat');
MH = load('models/dnn/modelDNN_hog.mat');
MP = load('models/dnn/modelDNN_pca.mat');
ML = load('models/dnn/modelDNN_lda.mat');
MB = load('models/dnn/modelDNN_lbp.mat');
ME = load('models/dnn/modelDNN_edge.mat');
MF = load('models/dnn/modelDNN_fusion.mat');

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

%%  Normalize using training stats  
Xte_raw_norm    = (Xte_raw    - MR.mu_raw_full)    ./ MR.sigma_raw_full;
Xte_hog_norm    = (Xte_hog    - MH.mu_hog_full)    ./ MH.sigma_hog_full;
Xte_lbp_norm    = (Xte_lbp    - MB.mu_lbp_full)    ./ MB.sigma_lbp_full;
Xte_edge_norm   = (Xte_edge   - ME.mu_edge_full)   ./ ME.sigma_edge_full;
Xte_fusion_norm = (Xte_fusion - MF.mu_fusion_full) ./ MF.sigma_fusion_full;

%%  Predict  
fprintf('\nEvaluating DNN models...\n');
[pred_raw,    t_raw]    = timedPredictDNN(MR.modelDNN_raw,    Xte_raw_norm);
[pred_hog,    t_hog]    = timedPredictDNN(MH.modelDNN_hog,    Xte_hog_norm);
[pred_pca,    t_pca]    = timedPredictDNN(MP.modelDNN_pca,    Xte_pca);
[pred_lda,    t_lda]    = timedPredictDNN(ML.modelDNN_lda,    Xte_lda);
[pred_lbp,    t_lbp]    = timedPredictDNN(MB.modelDNN_lbp,    Xte_lbp_norm);
[pred_edge,   t_edge]   = timedPredictDNN(ME.modelDNN_edge,   Xte_edge_norm);
[pred_fusion, t_fusion] = timedPredictDNN(MF.modelDNN_fusion, Xte_fusion_norm);

%%  Metrics helpers  
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

%%  Compute metrics  
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
fprintf('\nDEEP NEURAL NETWORK RESULTS\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | Layers         | Time(s)\n');
fprintf('                         -\n');
printRow('RAW',     M.raw,    MR.bestRaw.layers,    t_raw);
printRow('HOG',     M.hog,    MH.bestHog.layers,    t_hog);
printRow('PCA',     M.pca,    MP.bestPca.layers,    t_pca);
printRow('LDA',     M.lda,    ML.bestLda.layers,    t_lda);
printRow('LBP',     M.lbp,    MB.bestLbp.layers,    t_lbp);
printRow('EDGE',    M.edge,   ME.bestEdge.layers,   t_edge);
printRow('LBP+HOG', M.fusion, MF.bestFusion.layers, t_fusion);

%%  TP/TN/FP/FN counts summary  
fprintf('\nTP/TN/FP/FN COUNTS\n');
fprintf('Descriptor     |   TP   |   TN   |   FP   |   FN\n');
fprintf('                \n');
printConfRow('RAW',     M.raw);
printConfRow('HOG',     M.hog);
printConfRow('PCA',     M.pca);
printConfRow('LDA',     M.lda);
printConfRow('LBP',     M.lbp);
printConfRow('EDGE',    M.edge);
printConfRow('LBP+HOG', M.fusion);

%%  Confusion matrices  
figure('Name','DNN Confusion Matrices','Position',[100 100 1600 800]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

titles = {'RAW','HOG','PCA','LDA','LBP','EDGE','LBP+HOG'};
preds  = {pred_raw,pred_hog,pred_pca,pred_lda,pred_lbp,pred_edge,pred_fusion};

for i = 1:7
    nexttile;
    confusionchart(yte, preds{i}, ...
        'Title', sprintf('DNN (%s)', titles{i}), ...
        'RowSummary', 'off', ...
        'ColumnSummary', 'off');
end

%%  TP/TN/FP/FN image grids for RAW and HOG (using RAW images)  
rawHW  = [128 64];
maxShow = 64;

Xte_raw_full = double(R.X_raw(testIdx,:));   

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

showIndexGridFromFeatures(Xte_raw_full, idxTP_raw, maxShow, 'DNN RAW — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxTN_raw, maxShow, 'DNN RAW — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFP_raw, maxShow, 'DNN RAW — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFN_raw, maxShow, 'DNN RAW — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw_full, idxTP_hog, maxShow, 'DNN HOG — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxTN_hog, maxShow, 'DNN HOG — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFP_hog, maxShow, 'DNN HOG — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_full, idxFN_hog, maxShow, 'DNN HOG — False Negatives', rawHW);

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

%%  Local helper: timed prediction for DNN  
function [pred, t] = timedPredictDNN(net, X)
    tic;
    Y = net(X');                
    t = toc / size(X,1);
    [~, predIdx] = max(Y, [], 1);
    pred = predIdx' - 1;       
end

%%  Local helper: print main metric row  
function printRow(name, M, layers, t)
    fprintf('%-14s:  %6.2f  | %.3f  | %.3f  | %.3f  | %-13s | %.4fs\n', ...
        name, M.acc, M.prec, M.rec, M.f1, mat2str(layers), t);
end

%%  Local helper: print TP/TN/FP/FN row  
function printConfRow(name, M)
    fprintf('%-14s: %6d | %6d | %6d | %6d\n', ...
        name, M.tp, M.tn, M.fp, M.fn);
end