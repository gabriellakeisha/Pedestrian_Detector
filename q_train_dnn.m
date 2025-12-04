%% q_train_dnn.m — Deep Neural Network with RAW / HOG / PCA / LDA / LBP / EDGE / FUSION
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/dnn','dir'), mkdir('models/dnn'); end

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

%%  Normalization stats on FULL TRAIN  
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

%%  Deep NN architectures to try (hidden layers)  
% Each entry is a vector: [layer1 layer2 ...]
archSet = { [64 32], [128 64], [256 128 64] };

fprintf('\nDeep Neural Network (patternnet) — hyperparameter search (layer sizes)\n');

bestRaw    = tune_dnn(Xtr_raw_norm,    ytr, archSet, 'RAW');
bestHog    = tune_dnn(Xtr_hog_norm,    ytr, archSet, 'HOG');
bestPca    = tune_dnn(Xtr_pca,         ytr, archSet, 'PCA');   
bestLda    = tune_dnn(Xtr_lda,         ytr, archSet, 'LDA');
bestLbp    = tune_dnn(Xtr_lbp_norm,    ytr, archSet, 'LBP');
bestEdge   = tune_dnn(Xtr_edge_norm,   ytr, archSet, 'EDGE');
bestFusion = tune_dnn(Xtr_fusion_norm, ytr, archSet, 'LBP+HOG');

fprintf('\nBEST (validation split inside tuner)\n');
printBest('RAW',     bestRaw);
printBest('HOG',     bestHog);
printBest('PCA',     bestPca);
printBest('LDA',     bestLda);
printBest('LBP',     bestLbp);
printBest('EDGE',    bestEdge);
printBest('LBP+HOG', bestFusion);

%%  Train final DNNs on FULL TRAIN using best architectures  
fprintf('\nTraining final DNN models on full training set...\n');

modelDNN_raw    = train_dnn_classifier(Xtr_raw_norm,    ytr, bestRaw.layers);
modelDNN_hog    = train_dnn_classifier(Xtr_hog_norm,    ytr, bestHog.layers);
modelDNN_pca    = train_dnn_classifier(Xtr_pca,         ytr, bestPca.layers);
modelDNN_lda    = train_dnn_classifier(Xtr_lda,         ytr, bestLda.layers);
modelDNN_lbp    = train_dnn_classifier(Xtr_lbp_norm,    ytr, bestLbp.layers);
modelDNN_edge   = train_dnn_classifier(Xtr_edge_norm,   ytr, bestEdge.layers);
modelDNN_fusion = train_dnn_classifier(Xtr_fusion_norm, ytr, bestFusion.layers);

%%  Save models (+ stats the eval script expects)  
save('models/dnn/modelDNN_raw.mat',    'modelDNN_raw',    'bestRaw', ...
     'mu_raw_full',    'sigma_raw_full',    '-v7.3');

save('models/dnn/modelDNN_hog.mat',    'modelDNN_hog',    'bestHog', ...
     'mu_hog_full',    'sigma_hog_full',    '-v7.3');

save('models/dnn/modelDNN_pca.mat',    'modelDNN_pca',    'bestPca', 'P', '-v7.3');
save('models/dnn/modelDNN_lda.mat',    'modelDNN_lda',    'bestLda', 'L', '-v7.3');

save('models/dnn/modelDNN_lbp.mat',    'modelDNN_lbp',    'bestLbp', ...
     'mu_lbp_full',    'sigma_lbp_full',    '-v7.3');

save('models/dnn/modelDNN_edge.mat',   'modelDNN_edge',   'bestEdge', ...
     'mu_edge_full',   'sigma_edge_full',   '-v7.3');

save('models/dnn/modelDNN_fusion.mat', 'modelDNN_fusion', 'bestFusion', ...
     'mu_fusion_full', 'sigma_fusion_full', '-v7.3');

fprintf('\n All DNN models trained and saved in models/dnn/.\n');

%% Local helpers

function best = tune_dnn(X, y, archSet, desc)
    fprintf('\n[%s] tuning DNN (layer configs)...\n', desc);
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

    best.acc   = -inf;
    best.layers = [];
    best.archIdx = NaN;

    for a = 1:numel(archSet)
        layers = archSet{a};
        t = tic;
        net = patternnet(layers);
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = true;
        net.trainParam.epochs = 300;
        net.divideFcn = 'dividetrain';  

        net = train(net, Xsub', Tsub);  
        Yval = net(Xval');              
        [~, predIdx] = max(Yval, [], 1);
        ypred = predIdx' - 1;         

        acc = mean(ypred == yval) * 100;
        fprintf('  Layers [%s] → %.2f%%  [%.1fs]\n', ...
            num2str(layers), acc, toc(t));

        if acc > best.acc
            best.acc    = acc;
            best.layers = layers;
            best.archIdx = a;
            fprintf('  ↳ NEW BEST for %s\n', desc);
        end
    end
end

function net = train_dnn_classifier(X, y, layers)
    fprintf('  → Training final DNN (layers = [%s])...\n', num2str(layers));
    T = full(ind2vec(y' + 1));  
    net = patternnet(layers);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;
    net.trainParam.epochs = 400;
    net.divideFcn = 'dividetrain';  
    net = train(net, X', T);
end

function printBest(name, best)
    fprintf('  %-9s: layers=[%s] → %.2f%%\n', ...
        name, num2str(best.layers), best.acc);
end