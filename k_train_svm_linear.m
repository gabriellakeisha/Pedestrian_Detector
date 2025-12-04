% i_train_svm_linear.m — Train Linear SVM classifier with RAW / HOG / PCA / LDA / LBP / EDGE / FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/svm','dir'), mkdir('models/svm'); end

fprintf("=== LINEAR SVM Training (RAW/HOG/PCA/LDA/LBP/EDGE/FUSION) ===\n");

% Load splits and features 
S = load('splits/splits.mat'); 
trainIdx = S.trainIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
P = load('features/pca/features_pca.mat');
L = load('features/lda/features_lda.mat');
B = load('features/lbp/features_lbp.mat');
E = load('features/edge/features_edgehog.mat');
F = load('features/fusion/features_fusion_hog_lbp.mat');

ytr        = double(R.y(trainIdx));
Xtr_raw    = double(R.X_raw(trainIdx,:));
Xtr_hog    = double(H.X_hog(trainIdx,:));
Xtr_pca    = double(P.Xtr_pca);          
Xtr_lda    = double(L.Xtr_lda);          
Xtr_lbp    = double(B.X_lbp(trainIdx,:));
Xtr_edge   = double(E.X_edge(trainIdx,:));
Xtr_fusion = double(F.X_fusion(trainIdx,:));

fprintf("Training samples: %d\n", numel(ytr));
fprintf("Dims — RAW:%d HOG:%d PCA:%d LDA:%d LBP:%d EDGE:%d FUSION:%d\n", ...
    size(Xtr_raw,2), size(Xtr_hog,2), size(Xtr_pca,2), size(Xtr_lda,2), ...
    size(Xtr_lbp,2), size(Xtr_edge,2), size(Xtr_fusion,2));

N = numel(ytr);
p = randperm(N);
Nval = round(0.2*N); 
valSel = p(1:Nval);
subSel = p(Nval+1:end);

% Helper for splitting 
split = @(X) deal(X(subSel,:), X(valSel,:));

[Xsub_raw,    Xval_raw]    = split(Xtr_raw);
[Xsub_hog,    Xval_hog]    = split(Xtr_hog);
[Xsub_pca,    Xval_pca]    = split(Xtr_pca);
[Xsub_lda,    Xval_lda]    = split(Xtr_lda);
[Xsub_lbp,    Xval_lbp]    = split(Xtr_lbp);
[Xsub_edge,   Xval_edge]   = split(Xtr_edge);
[Xsub_fusion, Xval_fusion] = split(Xtr_fusion);

ysub = ytr(subSel);
yval = ytr(valSel);

% Normalisation 
normFeat = @(Xsub) struct( ...
    'mu', mean(Xsub), ...
    'sigma', max(std(Xsub), eps) );

NR_raw    = normFeat(Xsub_raw);
NR_hog    = normFeat(Xsub_hog);
NR_lbp    = normFeat(Xsub_lbp);
NR_edge   = normFeat(Xsub_edge);
NR_fusion = normFeat(Xsub_fusion);

applyNorm = @(X,NR) (X - NR.mu) ./ NR.sigma;

Xsub_raw_n    = applyNorm(Xsub_raw,    NR_raw);
Xval_raw_n    = applyNorm(Xval_raw,    NR_raw);
Xsub_hog_n    = applyNorm(Xsub_hog,    NR_hog);
Xval_hog_n    = applyNorm(Xval_hog,    NR_hog);
Xsub_lbp_n    = applyNorm(Xsub_lbp,    NR_lbp);
Xval_lbp_n    = applyNorm(Xval_lbp,    NR_lbp);
Xsub_edge_n   = applyNorm(Xsub_edge,   NR_edge);
Xval_edge_n   = applyNorm(Xval_edge,   NR_edge);
Xsub_fusion_n = applyNorm(Xsub_fusion, NR_fusion);
Xval_fusion_n = applyNorm(Xval_fusion, NR_fusion);

% PCA and LDA
Xsub_pca_n = Xsub_pca;
Xval_pca_n = Xval_pca;
Xsub_lda_n = Xsub_lda;
Xval_lda_n = Xval_lda;

% Hyperparameter tuning 
Cset = [0.01 0.1 1 10 100];

train_linear = @(Xtr, Xva, name) tune_linear_svm(Xtr, ysub, Xva, yval, Cset, name);

bestRaw    = train_linear(Xsub_raw_n,    Xval_raw_n,    "RAW");
bestHog    = train_linear(Xsub_hog_n,    Xval_hog_n,    "HOG");
bestPca    = train_linear(Xsub_pca_n,    Xval_pca_n,    "PCA");
bestLda    = train_linear(Xsub_lda_n,    Xval_lda_n,    "LDA");
bestLbp    = train_linear(Xsub_lbp_n,    Xval_lbp_n,    "LBP");
bestEdge   = train_linear(Xsub_edge_n,   Xval_edge_n,   "EDGE");
bestFusion = train_linear(Xsub_fusion_n, Xval_fusion_n, "FUSION");

% Train final models 
fprintf("\nTraining final models on FULL training set...\n");

% Normalise train set
NR_raw_full    = normFeat(Xtr_raw);
NR_hog_full    = normFeat(Xtr_hog);
NR_lbp_full    = normFeat(Xtr_lbp);
NR_edge_full   = normFeat(Xtr_edge);
NR_fusion_full = normFeat(Xtr_fusion);

Xtr_raw_n    = applyNorm(Xtr_raw,    NR_raw_full);
Xtr_hog_n    = applyNorm(Xtr_hog,    NR_hog_full);
Xtr_lbp_n    = applyNorm(Xtr_lbp,    NR_lbp_full);
Xtr_edge_n   = applyNorm(Xtr_edge,   NR_edge_full);
Xtr_fusion_n = applyNorm(Xtr_fusion, NR_fusion_full);

% Final SVM training
modelSVM_raw    = fitcsvm(Xtr_raw_n,    ytr, 'KernelFunction','linear','BoxConstraint',bestRaw.C);
modelSVM_hog    = fitcsvm(Xtr_hog_n,    ytr, 'KernelFunction','linear','BoxConstraint',bestHog.C);
modelSVM_pca    = fitcsvm(Xtr_pca,      ytr, 'KernelFunction','linear','BoxConstraint',bestPca.C);
modelSVM_lda    = fitcsvm(Xtr_lda,      ytr, 'KernelFunction','linear','BoxConstraint',bestLda.C);
modelSVM_lbp    = fitcsvm(Xtr_lbp_n,    ytr, 'KernelFunction','linear','BoxConstraint',bestLbp.C);
modelSVM_edge   = fitcsvm(Xtr_edge_n,   ytr, 'KernelFunction','linear','BoxConstraint',bestEdge.C);
modelSVM_fusion = fitcsvm(Xtr_fusion_n, ytr, 'KernelFunction','linear','BoxConstraint',bestFusion.C);

% Save models 
save('models/svm/modelSVM_linear_raw.mat',    'modelSVM_raw',    'bestRaw',    'NR_raw_full',    '-v7.3');
save('models/svm/modelSVM_linear_hog.mat',    'modelSVM_hog',    'bestHog',    'NR_hog_full',    '-v7.3');
save('models/svm/modelSVM_linear_pca.mat',    'modelSVM_pca',    'bestPca',    'P',              '-v7.3');
save('models/svm/modelSVM_linear_lda.mat',    'modelSVM_lda',    'bestLda',    'L',              '-v7.3');
save('models/svm/modelSVM_linear_lbp.mat',    'modelSVM_lbp',    'bestLbp',    'NR_lbp_full',    '-v7.3');
save('models/svm/modelSVM_linear_edge.mat',   'modelSVM_edge',   'bestEdge',   'NR_edge_full',   '-v7.3');
save('models/svm/modelSVM_linear_fusion.mat', 'modelSVM_fusion', 'bestFusion', 'NR_fusion_full', '-v7.3');

fprintf("\nAll LINEAR SVM models trained and saved!\n");

function best = tune_linear_svm(Xsub, ysub, Xval, yval, Cset, desc)
    fprintf("\n[%s] Linear SVM tuning...\n", desc);
    best.acc = -inf; best.C = NaN;

    for C = Cset
        t = tic;
        mdl = fitcsvm(Xsub, ysub, ...
            'KernelFunction','linear', ...
            'BoxConstraint',C, ...
            'ClassNames',[0 1], ...
            'Standardize',false);

        pred = predict(mdl, Xval);
        acc = mean(pred == yval) * 100;

        fprintf("  C=%g -> %.2f%%  [%.2fs]\n", C, acc, toc(t));

        if acc > best.acc
            best = struct('acc',acc,'C',C);
            fprintf("    -> NEW BEST for %s\n", desc);
        end
    end
end
