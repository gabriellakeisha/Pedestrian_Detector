% l_eval_svm_linear.m — Evaluate SVM-Linear classifier with RAW/HOG/PCA/LDA/LBP/EDGE/FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf("\n=== Evaluating SVM-LINEAR Models ===\n");

% Load split and features 
S = load('splits/splits.mat');
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P = load('features/pca/features_pca.mat');
L = load('features/lda/features_lda.mat');
B = load('features/lbp/features_lbp.mat');
E = load('features/edge/features_edgehog.mat');
F = load('features/fusion/features_fusion_hog_lbp.mat');

yte        = double(R.y(testIdx));
Xte_raw    = double(R.X_raw(testIdx,:));
Xte_hog    = double(H.X_hog(testIdx,:));
Xte_pca    = double(P.Xte_pca);
Xte_lda    = double(L.Xte_lda);
Xte_lbp    = double(B.X_lbp(testIdx,:));
Xte_edge   = double(E.X_edge(testIdx,:));
Xte_fusion = double(F.X_fusion(testIdx,:));

% Load models 
MR = load('models/svm/modelSVM_linear_raw.mat');     
MH = load('models/svm/modelSVM_linear_hog.mat');     
MP = load('models/svm/modelSVM_linear_pca.mat');      
ML = load('models/svm/modelSVM_linear_lda.mat');      
MB = load('models/svm/modelSVM_linear_lbp.mat');      
ME = load('models/svm/modelSVM_linear_edge.mat');     
MF = load('models/svm/modelSVM_linear_fusion.mat');   

% Normalisation 
applyNorm = @(X,N) (X - N.mu) ./ N.sigma;

Xte_raw_norm    = applyNorm(Xte_raw,    MR.NR_raw_full);
Xte_hog_norm    = applyNorm(Xte_hog,    MH.NR_hog_full);
Xte_lbp_norm    = applyNorm(Xte_lbp,    MB.NR_lbp_full);
Xte_edge_norm   = applyNorm(Xte_edge,   ME.NR_edge_full);
Xte_fusion_norm = applyNorm(Xte_fusion, MF.NR_fusion_full);

% PCA and LDA
Xte_pca_norm = Xte_pca;
Xte_lda_norm = Xte_lda;

% Predict 
fprintf("\nPredicting...\n");

[pred_raw,    score_raw]    = predict(MR.modelSVM_raw,    Xte_raw_norm);
[pred_hog,    score_hog]    = predict(MH.modelSVM_hog,    Xte_hog_norm);
[pred_pca,    score_pca]    = predict(MP.modelSVM_pca,    Xte_pca_norm);
[pred_lda,    score_lda]    = predict(ML.modelSVM_lda,    Xte_lda_norm);
[pred_lbp,    score_lbp]    = predict(MB.modelSVM_lbp,    Xte_lbp_norm);
[pred_edge,   score_edge]   = predict(ME.modelSVM_edge,   Xte_edge_norm);
[pred_fusion, score_fusion] = predict(MF.modelSVM_fusion, Xte_fusion_norm);

% Metrics 
metrics = @(y,p) struct( ...
    'acc', mean(p==y)*100, ...
    'tp', sum(p==1 & y==1), ...
    'fp', sum(p==1 & y==0), ...
    'fn', sum(p==0 & y==1) );

calcF1 = @(m) struct( ...
    'prec', m.tp/(m.tp+m.fp+eps), ...
    'rec',  m.tp/(m.tp+m.fn+eps) );

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

% AUC-ROC curve 
[~,~,~,auc_raw]    = perfcurve(yte, score_raw(:,2), 1);
[~,~,~,auc_hog]    = perfcurve(yte, score_hog(:,2), 1);
[~,~,~,auc_pca]    = perfcurve(yte, score_pca(:,2), 1);
[~,~,~,auc_lda]    = perfcurve(yte, score_lda(:,2), 1);
[~,~,~,auc_lbp]    = perfcurve(yte, score_lbp(:,2), 1);
[~,~,~,auc_edge]   = perfcurve(yte, score_edge(:,2), 1);
[~,~,~,auc_fusion] = perfcurve(yte, score_fusion(:,2), 1);

% Print results 
fprintf('\nOPTIMIZED LINEAR-SVM RESULTS (with AUC)\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | AUC   | C\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('RAW            : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.raw.acc, M.raw.prec, M.raw.rec, M.raw.f1, auc_raw, MR.bestRaw.C);
fprintf('HOG            : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.hog.acc, M.hog.prec, M.hog.rec, M.hog.f1, auc_hog, MH.bestHog.C);
fprintf('PCA            : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.pca.acc, M.pca.prec, M.pca.rec, M.pca.f1, auc_pca, MP.bestPca.C);
fprintf('LDA            : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.lda.acc, M.lda.prec, M.lda.rec, M.lda.f1, auc_lda, ML.bestLda.C);
fprintf('LBP            : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.lbp.acc, M.lbp.prec, M.lbp.rec, M.lbp.f1, auc_lbp, MB.bestLbp.C);
fprintf('EDGE           : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.edge.acc, M.edge.prec, M.edge.rec, M.edge.f1, auc_edge, ME.bestEdge.C);
fprintf('LBP+HOG        : %6.2f | %.3f | %.3f | %.3f | %.3f | %g\n', ...
    M.fusion.acc, M.fusion.prec, M.fusion.rec, M.fusion.f1, auc_fusion, MF.bestFusion.C);

% AUC-ROC figure 
figure('Name','SVM-LINEAR ROC Curves','Position',[100 100 800 600]);
hold on; grid on;

% Helper for selecting correct positive-class score
getPosScore = @(model,score) score(:, model.ClassNames==1);

plotROC(yte, getPosScore(MR.modelSVM_raw,    score_raw),    'RAW');
plotROC(yte, getPosScore(MH.modelSVM_hog,    score_hog),    'HOG');
plotROC(yte, getPosScore(MP.modelSVM_pca,    score_pca),    'PCA');
plotROC(yte, getPosScore(ML.modelSVM_lda,    score_lda),    'LDA');
plotROC(yte, getPosScore(MB.modelSVM_lbp,    score_lbp),    'LBP');
plotROC(yte, getPosScore(ME.modelSVM_edge,   score_edge),   'EDGE');
plotROC(yte, getPosScore(MF.modelSVM_fusion, score_fusion), 'LBP+HOG');

legend show;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('SVM-LINEAR — ROC Curves');
hold off;

function plotROC(y, scorePos, name)
    [FPR,TPR,~,AUC] = perfcurve(y, scorePos, 1);
    plot(FPR, TPR, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (AUC=%.3f)', name, AUC));
end

% Confusion matrices 
figure('Name','SVM-LINEAR Confusion Matrices','Position',[100 100 1600 800]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

titles = {'RAW','HOG','PCA','LDA','LBP','EDGE','FUSION'};
preds  = {pred_raw,pred_hog,pred_pca,pred_lda,pred_lbp,pred_edge,pred_fusion};

for i = 1:7
    nexttile;
    confusionchart(yte, preds{i}, ...
        'Title', sprintf('SVM-Linear (%s)', titles{i}), ...
        'RowSummary','off', 'ColumnSummary','off');
end

% TP/TN/FP/FN Image grids for RAW and HOG 
rawHW = [128 64];
maxShow = 64;

idxTP_raw = find(pred_raw==1 & yte==1);
idxTN_raw = find(pred_raw==0 & yte==0);
idxFP_raw = find(pred_raw==1 & yte==0);
idxFN_raw = find(pred_raw==0 & yte==1);

idxTP_hog = find(pred_hog==1 & yte==1);
idxTN_hog = find(pred_hog==0 & yte==0);
idxFP_hog = find(pred_hog==1 & yte==0);
idxFN_hog = find(pred_hog==0 & yte==1);

showIndexGridFromFeatures(Xte_raw, idxTP_raw, maxShow, 'SVM-LINEAR RAW — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxTN_raw, maxShow, 'SVM-LINEAR RAW — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFP_raw, maxShow, 'SVM-LINEAR RAW — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFN_raw, maxShow, 'SVM-LINEAR RAW — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw, idxTP_hog, maxShow, 'SVM-LINEAR HOG — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxTN_hog, maxShow, 'SVM-LINEAR HOG — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFP_hog, maxShow, 'SVM-LINEAR HOG — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw, idxFN_hog, maxShow, 'SVM-LINEAR HOG — False Negatives', rawHW);

function showIndexGridFromFeatures(X, idx, maxShow, figTitle, imgHW)
    figure('Name',figTitle,'NumberTitle','off');
    nShow = min(maxShow, numel(idx));
    if nShow == 0
        sgtitle([figTitle ' (none)']); 
        return;
    end
    nCols = 8;
    nRows = ceil(nShow/nCols);
    tiledlayout(nRows, nCols, 'Padding','compact','TileSpacing','compact');
    for k = 1:nShow
        nexttile;
        img = reshape(X(idx(k),:), imgHW);
        imshow(mat2gray(img),[]); axis off;
    end
    sgtitle(figTitle);
end
