% j_eval_svm_rbf.m — Evaluate SVM-RBF (RAW / HOG / PCA / LDA / LBP / EDGE / FUSION)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

% Load splits and features 
S = load('splits/splits.mat'); 
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P = load('features/pca/features_pca.mat');
L = load('features/lda/features_lda.mat');
B = load('features/lbp/features_lbp.mat');
E = load('features/edge/features_edgehog.mat');
F = load('features/fusion/features_fusion_hog_lbp.mat');

% Load models 
MR = load('models/svm/modelSVM_rbf_raw.mat');
MH = load('models/svm/modelSVM_rbf_hog.mat');
MP = load('models/svm/modelSVM_rbf_pca.mat');
ML = load('models/svm/modelSVM_rbf_lda.mat');
MB = load('models/svm/modelSVM_rbf_lbp.mat');
ME = load('models/svm/modelSVM_rbf_edge.mat');
MF = load('models/svm/modelSVM_rbf_fusion.mat');

% Prepare test data 
yte = double(R.y(testIdx));
Xte_raw    = double(R.X_raw(testIdx,:));
Xte_hog    = double(H.X_hog(testIdx,:));
Xte_pca    = double(P.Xte_pca);
Xte_lda    = double(L.Xte_lda);
Xte_lbp    = double(B.X_lbp(testIdx,:));
Xte_edge   = double(E.X_edge(testIdx,:));
Xte_fusion = double(F.X_fusion(testIdx,:));

% Normalise 
Xte_raw_norm    = (Xte_raw    - MR.mu_raw_full)    ./ MR.sigma_raw_full;
Xte_hog_norm    = (Xte_hog    - MH.mu_hog_full)    ./ MH.sigma_hog_full;
Xte_lbp_norm    = (Xte_lbp    - MB.mu_lbp_full)    ./ MB.sigma_lbp_full;
Xte_edge_norm   = (Xte_edge   - ME.mu_edge_full)   ./ ME.sigma_edge_full;
Xte_fusion_norm = (Xte_fusion - MF.mu_fusion_full) ./ MF.sigma_fusion_full;

% Predict 
fprintf('\nEvaluating SVM-RBF models...\n');
[pred_raw, score_raw]       = predict(MR.modelSVM_raw,    Xte_raw_norm);
[pred_hog, score_hog]       = predict(MH.modelSVM_hog,    Xte_hog_norm);
[pred_pca, score_pca]       = predict(MP.modelSVM_pca,    Xte_pca);
[pred_lda, score_lda]       = predict(ML.modelSVM_lda,    Xte_lda);
[pred_lbp, score_lbp]       = predict(MB.modelSVM_lbp,    Xte_lbp_norm);
[pred_edge, score_edge]     = predict(ME.modelSVM_edge,   Xte_edge_norm);
[pred_fusion, score_fusion] = predict(MF.modelSVM_fusion, Xte_fusion_norm);


% Compute metrics 
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

M.raw    = metrics(yte,pred_raw);
M.hog    = metrics(yte,pred_hog);
M.pca    = metrics(yte,pred_pca);
M.lda    = metrics(yte,pred_lda);
M.lbp    = metrics(yte,pred_lbp);
M.edge   = metrics(yte,pred_edge);
M.fusion = metrics(yte,pred_fusion);

fields = fieldnames(M);
for i = 1:numel(fields)
    F = fields{i};
    f = calcF1(M.(F));
    M.(F).prec = f.prec;
    M.(F).rec  = f.rec;
    M.(F).f1   = 2*f.prec*f.rec/(f.prec+f.rec+eps);
end

% Compute ROC-AUC curve 
[~,~,~,auc_raw]    = perfcurve(yte, score_raw(:,2), 1);
[~,~,~,auc_hog]    = perfcurve(yte, score_hog(:,2), 1);
[~,~,~,auc_pca]    = perfcurve(yte, score_pca(:,2), 1);
[~,~,~,auc_lda]    = perfcurve(yte, score_lda(:,2), 1);
[~,~,~,auc_lbp]    = perfcurve(yte, score_lbp(:,2), 1);
[~,~,~,auc_edge]   = perfcurve(yte, score_edge(:,2), 1);
[~,~,~,auc_fusion] = perfcurve(yte, score_fusion(:,2), 1);

fprintf('\nOPTIMIZED RBF-SVM RESULTS (with AUC)\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | AUC   | C     | KS    \n');
fprintf('--------------------------------------------------------------------------\n');
fprintf('RAW            :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.raw.acc, M.raw.prec, M.raw.rec, M.raw.f1, auc_raw, MR.bestRaw.C, MR.bestRaw.ks);
fprintf('HOG            :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.hog.acc, M.hog.prec, M.hog.rec, M.hog.f1, auc_hog, MH.bestHog.C, MH.bestHog.ks);
fprintf('PCA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.pca.acc, M.pca.prec, M.pca.rec, M.pca.f1, auc_pca, MP.bestPca.C, MP.bestPca.ks);
fprintf('LDA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.lda.acc, M.lda.prec, M.lda.rec, M.lda.f1, auc_lda, ML.bestLda.C, ML.bestLda.ks);
fprintf('LBP            :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.lbp.acc, M.lbp.prec, M.lbp.rec, M.lbp.f1, auc_lbp, MB.bestLbp.C, MB.bestLbp.ks);
fprintf('EDGE           :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.edge.acc, M.edge.prec, M.edge.rec, M.edge.f1, auc_edge, ME.bestEdge.C, ME.bestEdge.ks);
fprintf('LBP+HOG        :  %6.2f  | %.3f  | %.3f  | %.3f  | %.3f | %-5g | %-6.2f\n', ...
    M.fusion.acc, M.fusion.prec, M.fusion.rec, M.fusion.f1, auc_fusion, MF.bestFusion.C, MF.bestFusion.ks);

% Plot ROC-AUC curve 
figure('Name','SVM-RBF ROC Curves','Position',[100 100 800 600]);
hold on; grid on;
plotROC(yte, score_raw(:,2), 'RAW');
plotROC(yte, score_hog(:,2), 'HOG');
plotROC(yte, score_pca(:,2), 'PCA');
plotROC(yte, score_fusion(:,2), 'LBP+HOG');
legend show; xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('SVM-RBF ROC Curves');
hold off;

function plotROC(y, score, name)
    [X,Y,~,AUC] = perfcurve(y, score, 1);
    plot(X,Y,'LineWidth',1.5,'DisplayName',sprintf('%s (AUC=%.3f)',name,AUC));
end

% Confusion matrices 
figure('Name','SVM-RBF Confusion Matrices','Position',[100 100 1600 800]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

titles = {'RAW','HOG','PCA','LDA','LBP','EDGE','LBP+HOG'};
preds  = {pred_raw,pred_hog,pred_pca,pred_lda,pred_lbp,pred_edge,pred_fusion};

for i = 1:7
    nexttile;
    confusionchart(yte, preds{i}, ...
        'Title', sprintf('SVM-RBF (%s)', titles{i}), ...
        'RowSummary', 'off', ...        
        'ColumnSummary', 'off');        
end

% TP/TN/FP/FN Image grids 
rawHW  = [128 64];  % image size used for raw features
maxShow = 64;

% Indices for RAW
idxTP_raw = find(pred_raw==1 & yte==1);
idxTN_raw = find(pred_raw==0 & yte==0);
idxFP_raw = find(pred_raw==1 & yte==0);
idxFN_raw = find(pred_raw==0 & yte==1);

% Indices for HOG
idxTP_hog = find(pred_hog==1 & yte==1);
idxTN_hog = find(pred_hog==0 & yte==0);
idxFP_hog = find(pred_hog==1 & yte==0);
idxFN_hog = find(pred_hog==0 & yte==1);

% indices for FUSION
idxTP_fusion = find(pred_fusion==1 & yte==1);
idxTN_fusion = find(pred_fusion==0 & yte==0);
idxFP_fusion = find(pred_fusion==1 & yte==0);
idxFN_fusion = find(pred_fusion==0 & yte==1);

% Display grids
% showIndexGridFromFeatures(Xte_raw, idxTP_raw, maxShow, 'SVM-RBF RAW — True Positives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxTN_raw, maxShow, 'SVM-RBF RAW — True Negatives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxFP_raw, maxShow, 'SVM-RBF RAW — False Positives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxFN_raw, maxShow, 'SVM-RBF RAW — False Negatives', rawHW);
% 
% showIndexGridFromFeatures(Xte_raw, idxTP_hog, maxShow, 'SVM-RBF HOG — True Positives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxTN_hog, maxShow, 'SVM-RBF HOG — True Negatives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxFP_hog, maxShow, 'SVM-RBF HOG — False Positives', rawHW);
% showIndexGridFromFeatures(Xte_raw, idxFN_hog, maxShow, 'SVM-RBF HOG — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw, idxTP_fusion, maxShow, 'SVM-RBF FUSION — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxTN_fusion, maxShow, 'SVM-RBF FUSION — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFP_fusion, maxShow, 'SVM-RBF FUSION — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFN_fusion, maxShow, 'SVM-RBF FUSION — False Negatives', rawHW);

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

function [pred, t] = timedPredict(model, X)
    tic; pred = predict(model, X); t = toc / size(X,1);
end
