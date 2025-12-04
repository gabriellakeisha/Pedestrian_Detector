%% o_train_rf.m — Random Forest with RAW / HOG / PCA / LDA / LBP / EDGE / FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/rf','dir'), mkdir('models/rf'); end

%%  Load splits & features (TRAIN only) 
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

%% ===============================================================
% COMPUTE NORMALISATION FOR HOG (REQUIRED FOR DETECTOR)
mu_hog_full = mean(Xtr_hog, 1);
sigma_hog_full = std(Xtr_hog, 0, 1);
sigma_hog_full(sigma_hog_full == 0) = 1;

fprintf("\nComputed HOG normalisation → mean/std ready for detector\n");

%%  Hyperparameter tuning 
fprintf('\nRandom Forest (TreeBagger) — hyperparameter search\n');

numTreesSet = [50 100 200 400];   
leafSizeSet = [1 5 10 20];        

bestRaw    = tune_rf(Xtr_raw,    ytr, numTreesSet, leafSizeSet, 'RAW');
bestHog    = tune_rf(Xtr_hog,    ytr, numTreesSet, leafSizeSet, 'HOG');
bestPca    = tune_rf(Xtr_pca,    ytr, numTreesSet, leafSizeSet, 'PCA');
bestLda    = tune_rf(Xtr_lda,    ytr, numTreesSet, leafSizeSet, 'LDA');
bestLbp    = tune_rf(Xtr_lbp,    ytr, numTreesSet, leafSizeSet, 'LBP');
bestEdge   = tune_rf(Xtr_edge,   ytr, numTreesSet, leafSizeSet, 'EDGE');
bestFusion = tune_rf(Xtr_fusion, ytr, numTreesSet, leafSizeSet, 'LBP+HOG');

fprintf('\nBEST (validation split inside tuner)\n');
fprintf('  RAW     : T=%d, leaf=%d → %.2f%%\n', bestRaw.T,    bestRaw.leaf,    bestRaw.acc);
fprintf('  HOG     : T=%d, leaf=%d → %.2f%%\n', bestHog.T,    bestHog.leaf,    bestHog.acc);
fprintf('  PCA     : T=%d, leaf=%d → %.2f%%\n', bestPca.T,    bestPca.leaf,    bestPca.acc);
fprintf('  LDA     : T=%d, leaf=%d → %.2f%%\n', bestLda.T,    bestLda.leaf,    bestLda.acc);
fprintf('  LBP     : T=%d, leaf=%d → %.2f%%\n', bestLbp.T,    bestLbp.leaf,    bestLbp.acc);
fprintf('  EDGE    : T=%d, leaf=%d → %.2f%%\n', bestEdge.T,   bestEdge.leaf,   bestEdge.acc);
fprintf('  LBP+HOG : T=%d, leaf=%d → %.2f%%\n', bestFusion.T, bestFusion.leaf, bestFusion.acc);

%% ===============================================================
% TRAIN FINAL MODELS
fprintf('\nTraining final Random Forest models...\n');

modelRF_raw = TreeBagger(bestRaw.T, Xtr_raw, ytr, ...
    'Method','classification','MinLeafSize',bestRaw.leaf,'OOBPrediction','On');

modelRF_hog = TreeBagger(bestHog.T, Xtr_hog, ytr, ...
    'Method','classification','MinLeafSize',bestHog.leaf,'OOBPrediction','On');

modelRF_pca = TreeBagger(bestPca.T, Xtr_pca, ytr, ...
    'Method','classification','MinLeafSize',bestPca.leaf,'OOBPrediction','On');

modelRF_lda = TreeBagger(bestLda.T, Xtr_lda, ytr, ...
    'Method','classification','MinLeafSize',bestLda.leaf,'OOBPrediction','On');

modelRF_lbp = TreeBagger(bestLbp.T, Xtr_lbp, ytr, ...
    'Method','classification','MinLeafSize',bestLbp.leaf,'OOBPrediction','On');

modelRF_edge = TreeBagger(bestEdge.T, Xtr_edge, ytr, ...
    'Method','classification','MinLeafSize',bestEdge.leaf,'OOBPrediction','On');

modelRF_fusion = TreeBagger(bestFusion.T, Xtr_fusion, ytr, ...
    'Method','classification','MinLeafSize',bestFusion.leaf,'OOBPrediction','On');

%% ===============================================================
% SAVE MODELS (NOW WITH HOG NORMALISATION!)

save('models/rf/modelRF_raw.mat',    'modelRF_raw',    'bestRaw',    '-v7.3');

save('models/rf/modelRF_hog.mat', ...
    'modelRF_hog', 'bestHog', ...
    'mu_hog_full', 'sigma_hog_full', ... 
    '-v7.3');

save('models/rf/modelRF_pca.mat',    'modelRF_pca',    'bestPca',    'P', '-v7.3');
save('models/rf/modelRF_lda.mat',    'modelRF_lda',    'bestLda',    'L', '-v7.3');
save('models/rf/modelRF_lbp.mat',    'modelRF_lbp',    'bestLbp',    '-v7.3');
save('models/rf/modelRF_edge.mat',   'modelRF_edge',   'bestEdge',   '-v7.3');
save('models/rf/modelRF_fusion.mat', 'modelRF_fusion', 'bestFusion', '-v7.3');

fprintf('\n  All Random Forest models trained and saved in models/rf/.\n');


%% =================================================================
function best = tune_rf(X, y, numTreesSet, leafSizeSet, desc)
    fprintf('\n[%s] tuning Random Forest...\n', desc);
    N = numel(y);
    p = randperm(N);
    Nval = round(0.2*N);
    valSel = p(1:Nval);
    subSel = p(Nval+1:end);

    Xsub = X(subSel,:);  ysub = y(subSel);
    Xval = X(valSel,:);  yval = y(valSel);

    best.acc  = -inf;
    best.T    = NaN;
    best.leaf = NaN;

    for T = numTreesSet
        for leaf = leafSizeSet
            t = tic;
            mdl = TreeBagger(T, Xsub, ysub, ...
                'Method','classification','MinLeafSize',leaf,'OOBPrediction','Off');

            ypredCell = predict(mdl, Xval);

            if iscell(ypredCell)
                ypred = str2double(ypredCell);
            else
                ypred = ypredCell;
            end

            acc = mean(ypred == yval) * 100;
            fprintf('  T=%-4d leaf=%-3d → %.2f%%  [%.1fs]\n', T, leaf, acc, toc(t));

            if acc > best.acc
                best.acc  = acc;
                best.T    = T;
                best.leaf = leaf;
                fprintf('  ↳ NEW BEST for %s\n', desc);
            end
        end
    end
end
