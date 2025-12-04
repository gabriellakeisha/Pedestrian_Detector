% i_train_svm_rbf.m — Training RBF-SVM classifier with RAW / HOG / PCA / LDA / LBP / EDGE / FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/svm','dir'), mkdir('models/svm'); end

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

fprintf('Training: %d samples | dims — RAW:%d HOG:%d PCA:%d LDA:%d LBP:%d EDGE:%d LBP+HOG:%d\n', ...
    numel(ytr), size(Xtr_raw,2), size(Xtr_hog,2), size(Xtr_pca,2), size(Xtr_lda,2), ...
    size(Xtr_lbp,2), size(Xtr_edge,2), size(Xtr_fusion,2));

% Normalise stats 
mu_raw_full    = mean(Xtr_raw);    sigma_raw_full    = std(Xtr_raw);    sigma_raw_full(   sigma_raw_full<eps)    = 1;
mu_hog_full    = mean(Xtr_hog);    sigma_hog_full    = std(Xtr_hog);    sigma_hog_full(   sigma_hog_full<eps)    = 1;
mu_lbp_full    = mean(Xtr_lbp);    sigma_lbp_full    = std(Xtr_lbp);    sigma_lbp_full(   sigma_lbp_full<eps)    = 1;
mu_edge_full   = mean(Xtr_edge);   sigma_edge_full   = std(Xtr_edge);   sigma_edge_full(  sigma_edge_full<eps)   = 1;
mu_fusion_full = mean(Xtr_fusion); sigma_fusion_full = std(Xtr_fusion); sigma_fusion_full(sigma_fusion_full<eps) = 1;

Xtr_raw_norm    = (Xtr_raw    - mu_raw_full)    ./ sigma_raw_full;
Xtr_hog_norm    = (Xtr_hog    - mu_hog_full)    ./ sigma_hog_full;
Xtr_lbp_norm    = (Xtr_lbp    - mu_lbp_full)    ./ sigma_lbp_full;
Xtr_edge_norm   = (Xtr_edge   - mu_edge_full)   ./ sigma_edge_full;
Xtr_fusion_norm = (Xtr_fusion - mu_fusion_full) ./ sigma_fusion_full;

% Hyperparameter tuning 
fprintf('\nRBF-SVM with auto KernelScale — hyperparameter search\n');
Cset = [0.1 1 10 100 1000];

bestRaw    = tune_rbf_svm(Xtr_raw_norm,    ytr, Cset, 'RAW');
bestHog    = tune_rbf_svm(Xtr_hog_norm,    ytr, Cset, 'HOG');
bestPca    = tune_rbf_svm(Xtr_pca,         ytr, Cset, 'PCA');
bestLda    = tune_rbf_svm(Xtr_lda,         ytr, Cset, 'LDA');
bestLbp    = tune_rbf_svm(Xtr_lbp_norm,    ytr, Cset, 'LBP');
bestEdge   = tune_rbf_svm(Xtr_edge_norm,   ytr, Cset, 'EDGE');
bestFusion = tune_rbf_svm(Xtr_fusion_norm, ytr, Cset, 'LBP+HOG');

fprintf('\nBEST (validation split inside tuner)\n');
fprintf('  RAW     : C=%g, ks=%.3g -> %.2f%%\n', bestRaw.C,    bestRaw.ks,    bestRaw.acc);
fprintf('  HOG     : C=%g, ks=%.3g -> %.2f%%\n', bestHog.C,    bestHog.ks,    bestHog.acc);
fprintf('  PCA     : C=%g, ks=%.3g -> %.2f%%\n', bestPca.C,    bestPca.ks,    bestPca.acc);
fprintf('  LDA     : C=%g, ks=%.3g -> %.2f%%\n', bestLda.C,    bestLda.ks,    bestLda.acc);
fprintf('  LBP     : C=%g, ks=%.3g -> %.2f%%\n', bestLbp.C,    bestLbp.ks,    bestLbp.acc);
fprintf('  EDGE    : C=%g, ks=%.3g -> %.2f%%\n', bestEdge.C,   bestEdge.ks,   bestEdge.acc);
fprintf('  LBP+HOG : C=%g, ks=%.3g -> %.2f%%\n', bestFusion.C, bestFusion.ks, bestFusion.acc);

% Train final models using best C and KS
fprintf('\nTraining final models on full training set...\n');

modelSVM_raw = fitPosterior(fitcsvm( Xtr_raw_norm, ytr, ...
    'KernelFunction','rbf','KernelScale',bestRaw.ks, ...
    'BoxConstraint',bestRaw.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_hog = fitPosterior(fitcsvm( Xtr_hog_norm, ytr, ...
    'KernelFunction','rbf','KernelScale',bestHog.ks, ...
    'BoxConstraint',bestHog.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_pca = fitPosterior(fitcsvm( Xtr_pca, ytr, ...
    'KernelFunction','rbf','KernelScale',bestPca.ks, ...
    'BoxConstraint',bestPca.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_lda = fitPosterior(fitcsvm( Xtr_lda, ytr, ...
    'KernelFunction','rbf','KernelScale',bestLda.ks, ...
    'BoxConstraint',bestLda.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_lbp = fitPosterior(fitcsvm( Xtr_lbp_norm, ytr, ...
    'KernelFunction','rbf','KernelScale',bestLbp.ks, ...
    'BoxConstraint',bestLbp.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_edge = fitPosterior(fitcsvm( Xtr_edge_norm, ytr, ...
    'KernelFunction','rbf','KernelScale',bestEdge.ks, ...
    'BoxConstraint',bestEdge.C,'ClassNames',[0 1],'Standardize',false ));

modelSVM_fusion = fitPosterior(fitcsvm( Xtr_fusion_norm, ytr, ...
    'KernelFunction','rbf','KernelScale',bestFusion.ks, ...
    'BoxConstraint',bestFusion.C,'ClassNames',[0 1],'Standardize',false ));

% Save models 
save('models/svm/modelSVM_rbf_raw.mat',    'modelSVM_raw',    'bestRaw', ...
     'mu_raw_full',    'sigma_raw_full',    '-v7.3');
save('models/svm/modelSVM_rbf_hog.mat',    'modelSVM_hog',    'bestHog', ...
     'mu_hog_full',    'sigma_hog_full',    '-v7.3');
save('models/svm/modelSVM_rbf_pca.mat',    'modelSVM_pca',    'bestPca', 'P', '-v7.3');
save('models/svm/modelSVM_rbf_lda.mat',    'modelSVM_lda',    'bestLda', 'L', '-v7.3');
save('models/svm/modelSVM_rbf_lbp.mat',    'modelSVM_lbp',    'bestLbp', ...
     'mu_lbp_full',    'sigma_lbp_full',    '-v7.3');
save('models/svm/modelSVM_rbf_edge.mat',   'modelSVM_edge',   'bestEdge', ...
     'mu_edge_full',   'sigma_edge_full',   '-v7.3');
save('models/svm/modelSVM_rbf_fusion.mat', 'modelSVM_fusion', 'bestFusion', ...
     'mu_fusion_full', 'sigma_fusion_full', '-v7.3');

fprintf('\nAll SVM-RBF models trained, calibrated, and saved.\n');

function best = tune_rbf_svm(X, y, Cset, desc)
    fprintf('\n[%s] tuning...\n', desc);
    N = numel(y);
    p = randperm(N);
    Nval = round(0.2*N);
    valSel = p(1:Nval); subSel = p(Nval+1:end);
    Xsub = X(subSel,:); ysub = y(subSel);
    Xval = X(valSel,:); yval = y(valSel);

    best.acc = -inf; best.C = NaN; best.ks = NaN;
    for C = Cset
        t = tic;
        mdl = fitcsvm(Xsub, ysub, ...
            'KernelFunction','rbf', 'KernelScale','auto', ...
            'BoxConstraint',C, 'ClassNames',[0 1], 'Standardize',false);
        pred = predict(mdl, Xval);
        acc = mean(pred==yval) * 100;
        ks  = mdl.KernelParameters.Scale;
        fprintf('  C=%-5g -> %.2f%% (ks=%g)  [%.1fs]\n', C, acc, ks, toc(t));
        if acc > best.acc
            best = struct('acc',acc,'C',C,'ks',ks);
            fprintf('  ↳ NEW BEST for %s\n', desc);
        end
    end
end
