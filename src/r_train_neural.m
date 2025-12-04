%% r_train_neural.m — Neural Network with RAW / HOG / PCA / LDA / LBP / EDGE / FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/neural','dir'), mkdir('models/neural'); end

%% --- Load splits & features (TRAIN only) ---
S = load('splits/splits.mat'); 
trainIdx = S.trainIdx;

R = load('features/raw/features_raw.mat');           % R.X_raw, R.y
H = load('features/hog/features_hog.mat');           % H.X_hog
P = load('features/pca/features_pca.mat');           % P.Xtr_pca
L = load('features/lda/features_lda.mat');           % L.Xtr_lda
B = load('features/lbp/features_lbp.mat');           % B.X_lbp
E = load('features/edge/features_edgehog.mat');      % E.X_edge
F = load('features/fusion/features_fusion_hog_lbp.mat'); % F.X_fusion

ytr        = double(R.y(trainIdx));

Xtr_raw    = double(R.X_raw(trainIdx,:));
Xtr_hog    = double(H.X_hog(trainIdx,:));
Xtr_pca    = double(P.Xtr_pca);              % already train split
Xtr_lda    = double(L.Xtr_lda);              % already train split
Xtr_lbp    = double(B.X_lbp(trainIdx,:));
Xtr_edge   = double(E.X_edge(trainIdx,:));
Xtr_fusion = double(F.X_fusion(trainIdx,:));

fprintf('Training: %d samples | dims — RAW:%d HOG:%d PCA:%d LDA:%d LBP:%d EDGE:%d LBP+HOG:%d\n', ...
    numel(ytr), size(Xtr_raw,2), size(Xtr_hog,2), size(Xtr_pca,2), size(Xtr_lda,2), ...
    size(Xtr_lbp,2), size(Xtr_edge,2), size(Xtr_fusion,2));

%% --- Normalization stats on FULL TRAIN ---
% We normalize non-DR features (RAW, HOG, LBP, EDGE, FUSION).
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

%% --- Hyperparameter tuning: number of hidden units ---
fprintf('\nNeural Network (patternnet) — hyperparameter search (hidden units)\n');
hiddenSet = [10 20 50 100];

bestRaw    = tune_nn(Xtr_raw_norm,    ytr, hiddenSet, 'RAW');
bestHog    = tune_nn(Xtr_hog_norm,    ytr, hiddenSet, 'HOG');
bestPca    = tune_nn(Xtr_pca,         ytr, hiddenSet, 'PCA');   % PCA / LDA left as-is
bestLda    = tune_nn(Xtr_lda,         ytr, hiddenSet, 'LDA');
bestLbp    = tune_nn(Xtr_lbp_norm,    ytr, hiddenSet, 'LBP');
bestEdge   = tune_nn(Xtr_edge_norm,   ytr, hiddenSet, 'EDGE');
bestFusion = tune_nn(Xtr_fusion_norm, ytr, hiddenSet, 'LBP+HOG');

fprintf('\nBEST (validation split inside tuner)\n');
fprintf('  RAW     : H=%d → %.2f%%\n', bestRaw.H,    bestRaw.acc);
fprintf('  HOG     : H=%d → %.2f%%\n', bestHog.H,    bestHog.acc);
fprintf('  PCA     : H=%d → %.2f%%\n', bestPca.H,    bestPca.acc);
fprintf('  LDA     : H=%d → %.2f%%\n', bestLda.H,    bestLda.acc);
fprintf('  LBP     : H=%d → %.2f%%\n', bestLbp.H,    bestLbp.acc);
fprintf('  EDGE    : H=%d → %.2f%%\n', bestEdge.H,   bestEdge.acc);
fprintf('  LBP+HOG : H=%d → %.2f%%\n', bestFusion.H, bestFusion.acc);

%% --- Train final NNs on FULL TRAIN using best hidden units ---
fprintf('\nTraining final NN models on full training set...\n');

modelNN_raw = train_nn_classifier(Xtr_raw_norm, ytr, bestRaw.H);
modelNN_hog = train_nn_classifier(Xtr_hog_norm, ytr, bestHog.H);
modelNN_pca = train_nn_classifier(Xtr_pca,      ytr, bestPca.H);
modelNN_lda = train_nn_classifier(Xtr_lda,      ytr, bestLda.H);
modelNN_lbp = train_nn_classifier(Xtr_lbp_norm, ytr, bestLbp.H);
modelNN_edge   = train_nn_classifier(Xtr_edge_norm,   ytr, bestEdge.H);
modelNN_fusion = train_nn_classifier(Xtr_fusion_norm, ytr, bestFusion.H);

%% --- Save models (+ stats the eval script expects) ---
save('models/neural/modelNN_raw.mat',    'modelNN_raw',    'bestRaw', ...
     'mu_raw_full',    'sigma_raw_full',    '-v7.3');

save('models/neural/modelNN_hog.mat',    'modelNN_hog',    'bestHog', ...
     'mu_hog_full',    'sigma_hog_full',    '-v7.3');

save('models/neural/modelNN_pca.mat',    'modelNN_pca',    'bestPca', 'P', '-v7.3');
save('models/neural/modelNN_lda.mat',    'modelNN_lda',    'bestLda', 'L', '-v7.3');

save('models/neural/modelNN_lbp.mat',    'modelNN_lbp',    'bestLbp', ...
     'mu_lbp_full',    'sigma_lbp_full',    '-v7.3');

save('models/neural/modelNN_edge.mat',   'modelNN_edge',   'bestEdge', ...
     'mu_edge_full',   'sigma_edge_full',   '-v7.3');

save('models/neural/modelNN_fusion.mat', 'modelNN_fusion', 'bestFusion', ...
     'mu_fusion_full', 'sigma_fusion_full', '-v7.3');

fprintf('\nAll neural network models trained and saved in models/neural/.\n');

%% =======================
%% Local helper: tune_nn
%% =======================
function best = tune_nn(X, y, hiddenSet, desc)
    fprintf('\n[%s] tuning neural network (hidden units)...\n', desc);
    N = numel(y);
    p = randperm(N);
    Nval = round(0.2*N);
    valSel = p(1:Nval);
    subSel = p(Nval+1:end);

    Xsub = X(subSel,:);  ysub = y(subSel);
    Xval = X(valSel,:);  yval = y(valSel);

    % One-hot targets for classification (2 classes: 0,1)
    Tsub = full(ind2vec(ysub' + 1));   % size [2 x Ns]
    Tval = full(ind2vec(yval' + 1));   % size [2 x Nv]

    best.acc = -inf;
    best.H   = NaN;

    for H = hiddenSet
        t = tic;
        net = patternnet(H);
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = true;
        net.trainParam.epochs = 200;
        net.divideFcn = 'dividetrain';  % we already have our own val split

        net = train(net, Xsub', Tsub);  % inputs: [dims x N], targets: [2 x N]
        Yval = net(Xval');              % [2 x Nv]
        [~, predIdx] = max(Yval, [], 1);
        ypred = predIdx' - 1;           % back to {0,1}

        acc = mean(ypred == yval) * 100;
        fprintf('  H=%-4d → %.2f%%  [%.1fs]\n', H, acc, toc(t));

        if acc > best.acc
            best.acc = acc;
            best.H   = H;
            fprintf('  ↳ NEW BEST for %s\n', desc);
        end
    end
end

%% =======================
%% Local helper: train_nn_classifier
%% =======================
function net = train_nn_classifier(X, y, H)
    fprintf('  → Training final neural network (H=%d)...\n', H);
    T = full(ind2vec(y' + 1));  % one-hot labels
    net = patternnet(H);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;
    net.trainParam.epochs = 300;
    net.divideFcn = 'dividetrain';  % train on all provided samples
    net = train(net, X', T);
end