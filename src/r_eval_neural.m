%% r_eval_nn.m — Evaluate Neural Network (RAW / HOG / PCA / LDA / LBP / EDGE / FUSION)
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

%%   Load test indices and features  
S = load('splits/splits.mat'); 
testIdx = S.testIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P = load('features/pca/features_pca.mat');
L = load('features/lda/features_lda.mat');
B = load('features/lbp/features_lbp.mat');
E = load('features/edge/features_edgehog.mat');
F = load('features/fusion/features_fusion_hog_lbp.mat');

%%   Load NN models  
MR = load('models/neural/modelNN_raw.mat');
MH = load('models/neural/modelNN_hog.mat');
MP = load('models/neural/modelNN_pca.mat');
ML = load('models/neural/modelNN_lda.mat');
MB = load('models/neural/modelNN_lbp.mat');
ME = load('models/neural/modelNN_edge.mat');
MF = load('models/neural/modelNN_fusion.mat');

%%   Prepare test data  
yte = double(R.y(testIdx));

Xte_raw    = double(R.X_raw(testIdx,:));
Xte_hog    = double(H.X_hog(testIdx,:));
Xte_pca    = double(P.Xte_pca);
Xte_lda    = double(L.Xte_lda);
Xte_lbp    = double(B.X_lbp(testIdx,:));
Xte_edge   = double(E.X_edge(testIdx,:));
Xte_fusion = double(F.X_fusion(testIdx,:));

fprintf('Test set: %d samples\n', numel(yte));

%%   Normalize using training stats 
Xte_raw_norm    = (Xte_raw    - MR.mu_raw_full)    ./ MR.sigma_raw_full;
Xte_hog_norm    = (Xte_hog    - MH.mu_hog_full)    ./ MH.sigma_hog_full;
Xte_lbp_norm    = (Xte_lbp    - MB.mu_lbp_full)    ./ MB.sigma_lbp_full;
Xte_edge_norm   = (Xte_edge   - ME.mu_edge_full)   ./ ME.sigma_edge_full;
Xte_fusion_norm = (Xte_fusion - MF.mu_fusion_full) ./ MF.sigma_fusion_full;

%%   Predict  
fprintf('\nEvaluating NN models...\n');
[pred_raw,    t_raw]    = timedPredictNN(MR.modelNN_raw,    Xte_raw_norm);
[pred_hog,    t_hog]    = timedPredictNN(MH.modelNN_hog,    Xte_hog_norm);
[pred_pca,    t_pca]    = timedPredictNN(MP.modelNN_pca,    Xte_pca);
[pred_lda,    t_lda]    = timedPredictNN(ML.modelNN_lda,    Xte_lda);
[pred_lbp,    t_lbp]    = timedPredictNN(MB.modelNN_lbp,    Xte_lbp_norm);
[pred_edge,   t_edge]   = timedPredictNN(ME.modelNN_edge,   Xte_edge_norm);
[pred_fusion, t_fusion] = timedPredictNN(MF.modelNN_fusion, Xte_fusion_norm);

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

%%   Display results summary  
fprintf('\nNEURAL NETWORK RESULTS\n');
fprintf('Descriptor     | Acc(%%) | Prec  | Rec   | F1    | Hidden | Time(s)\n');
fprintf('                      -\n');
fprintf('RAW            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.raw.acc, M.raw.prec, M.raw.rec, M.raw.f1, MR.bestRaw.H,       t_raw);
fprintf('HOG            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.hog.acc, M.hog.prec, M.hog.rec, M.hog.f1, MH.bestHog.H,       t_hog);
fprintf('PCA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.pca.acc, M.pca.prec, M.pca.rec, M.pca.f1, MP.bestPca.H,       t_pca);
fprintf('LDA            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.lda.acc, M.lda.prec, M.lda.rec, M.lda.f1, ML.bestLda.H,       t_lda);
fprintf('LBP            :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.lbp.acc, M.lbp.prec, M.lbp.rec, M.lbp.f1, MB.bestLbp.H,       t_lbp);
fprintf('EDGE           :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.edge.acc, M.edge.prec, M.edge.rec, M.edge.f1, ME.bestEdge.H,  t_edge);
fprintf('LBP+HOG        :  %6.2f  | %.3f  | %.3f  | %.3f  | %-6d | %.4fs\n', ...
    M.fusion.acc, M.fusion.prec, M.fusion.rec, M.fusion.f1, MF.bestFusion.H, t_fusion);

%%   TP/TN/FP/FN summary table  
fprintf('\nTP/TN/FP/FN COUNTS\n');
fprintf('Descriptor     |   TP   |   TN   |   FP   |   FN\n');
fprintf('                \n');
fprintf('RAW            : %6d | %6d | %6d | %6d\n', ...
    M.raw.tp, M.raw.tn, M.raw.fp, M.raw.fn);
fprintf('HOG            : %6d | %6d | %6d | %6d\n', ...
    M.hog.tp, M.hog.tn, M.hog.fp, M.hog.fn);
fprintf('PCA            : %6d | %6d | %6d | %6d\n', ...
    M.pca.tp, M.pca.tn, M.pca.fp, M.pca.fn);
fprintf('LDA            : %6d | %6d | %6d | %6d\n', ...
    M.lda.tp, M.lda.tn, M.lda.fp, M.lda.fn);
fprintf('LBP            : %6d | %6d | %6d | %6d\n', ...
    M.lbp.tp, M.lbp.tn, M.lbp.fp, M.lbp.fn);
fprintf('EDGE           : %6d | %6d | %6d | %6d\n', ...
    M.edge.tp, M.edge.tn, M.edge.fp, M.edge.fn);
fprintf('LBP+HOG        : %6d | %6d | %6d | %6d\n', ...
    M.fusion.tp, M.fusion.tn, M.fusion.fp, M.fusion.fn);

%%   Confusion matrices  
figure('Name','NN Confusion Matrices','Position',[100 100 1600 800]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

titles = {'RAW','HOG','PCA','LDA','LBP','EDGE','LBP+HOG'};
preds  = {pred_raw,pred_hog,pred_pca,pred_lda,pred_lbp,pred_edge,pred_fusion};

for i = 1:7
    nexttile;
    confusionchart(yte, preds{i}, ...
        'Title', sprintf('NN (%s)', titles{i}), ...
        'RowSummary', 'off', ...
        'ColumnSummary', 'off');
end

%%   TP/TN/FP/FN image grids for RAW and HOG  
rawHW  = [128 64];
maxShow = 64;

Xte_raw_vis = double(R.X_raw(testIdx,:));  

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

showIndexGridFromFeatures(Xte_raw_vis, idxTP_raw, maxShow, 'NN RAW — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxTN_raw, maxShow, 'NN RAW — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFP_raw, maxShow, 'NN RAW — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFN_raw, maxShow, 'NN RAW — False Negatives', rawHW);

showIndexGridFromFeatures(Xte_raw_vis, idxTP_hog, maxShow, 'NN HOG — True Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxTN_hog, maxShow, 'NN HOG — True Negatives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFP_hog, maxShow, 'NN HOG — False Positives', rawHW);
showIndexGridFromFeatures(Xte_raw_vis, idxFN_hog, maxShow, 'NN HOG — False Negatives', rawHW);

%%   Local helper: showIndexGridFromFeatures  
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

%%   Local helper: timed prediction for NN  
function [pred, t] = timedPredictNN(net, X)
    tic;
    Y = net(X');              
    t = toc / size(X,1);
    [~, predIdx] = max(Y, [], 1);
    pred = predIdx' - 1;       
end