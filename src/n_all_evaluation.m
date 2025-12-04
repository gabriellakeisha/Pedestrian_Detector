%% n_all_evaluation_FIXED.m – Complete testing with ALL 7 features × 8 classifiers

clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf('COMPREHENSIVE PEDESTRIAN DETECTION EVALUATION\n');
fprintf('All Features × All Classifiers × All Validation Methods\n');

%% 1. Load ALL feature data for different validation methods
fprintf('Loading datasets for different validation methods...\n');

% 70/30 split
S70 = load('splits/splits.mat');
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% Try to load additional features
try B = load('features/lbp/features_lbp.mat'); has_lbp = true; catch; has_lbp = false; end
try E = load('features/edge/features_edgehog.mat'); has_edge = true; catch; has_edge = false; end
try F = load('features/fusion/features_fusion_hog_lbp.mat'); has_fusion = true; catch; has_fusion = false; end

% 50/50 split  
try
    S50 = load('splits/splits_50_50.mat');
    has_50 = true;
catch
    has_50 = false;
    fprintf('50/50 split not found, skipping\n');
end

fprintf('Dataset loaded: %d total samples\n', numel(R.y));

%% 2. Define ALL model combinations to test
models = {
    'NN-RAW', 'NN-HOG', ...
    'KNN-RAW', 'KNN-HOG', ...
    'SVM-Linear-RAW', 'SVM-Linear-HOG', 'SVM-Linear-PCA', 'SVM-Linear-LDA', ...
    'SVM-RBF-RAW', 'SVM-RBF-HOG', 'SVM-RBF-PCA', 'SVM-RBF-LDA', ...
    'Neural-RAW', 'Neural-HOG', 'Neural-PCA', 'Neural-LDA', ...
    'DNN-RAW', 'DNN-HOG', 'DNN-PCA', 'DNN-LDA', ...
    'CNN-RAW', 'CNN-HOG', 'CNN-PCA', 'CNN-LDA', ...
    'RF-RAW', 'RF-HOG', 'RF-PCA', 'RF-LDA', ...
    'Bagging-RAW', 'Bagging-HOG', 'Bagging-PCA', 'Bagging-LDA'
};

% Add LBP models if available
if has_lbp
    models = [models, {'SVM-Linear-LBP', 'SVM-RBF-LBP', 'Neural-LBP', ...
                       'DNN-LBP', 'CNN-LBP', 'RF-LBP', 'Bagging-LBP'}];
end

% Add EDGE models if available
if has_edge
    models = [models, {'SVM-Linear-EDGE', 'SVM-RBF-EDGE', 'Neural-EDGE', ...
                       'DNN-EDGE', 'CNN-EDGE', 'RF-EDGE', 'Bagging-EDGE'}];
end

% Add FUSION models if available
if has_fusion
    models = [models, {'SVM-Linear-LBP-HOG', 'SVM-RBF-LBP-HOG', 'Neural-LBP-HOG', ...
                       'DNN-LBP-HOG', 'CNN-LBP-HOG', 'RF-LBP-HOG', 'Bagging-LBP-HOG'}];
end

fprintf('Total models to evaluate: %d\n\n', numel(models));

%% 3. Test with 70/30 split
fprintf('TESTING WITH 70/30 HOLDOUT SPLIT (%d test samples)\n', numel(S70.testIdx));

if has_lbp && has_edge && has_fusion
    results_70 = test_all_models(models, R, H, B, E, F, S70.trainIdx, S70.testIdx, '70/30 Holdout');
elseif has_lbp
    results_70 = test_all_models(models, R, H, B, [], [], S70.trainIdx, S70.testIdx, '70/30 Holdout');
else
    results_70 = test_all_models(models, R, H, [], [], [], S70.trainIdx, S70.testIdx, '70/30 Holdout');
end

%% 4. Test with 50/50 split if available
if has_50
    fprintf('\n%s\n', repmat('=', 60, 1));
    fprintf('TESTING WITH 50/50 HOLDOUT SPLIT (%d test samples)\n', numel(S50.testIdx));
    fprintf('%s\n', repmat('=', 60, 1));
    
    if has_lbp && has_edge && has_fusion
        results_50 = test_all_models(models, R, H, B, E, F, S50.trainIdx, S50.testIdx, '50/50 Holdout');
    elseif has_lbp
        results_50 = test_all_models(models, R, H, B, [], [], S50.trainIdx, S50.testIdx, '50/50 Holdout');
    else
        results_50 = test_all_models(models, R, H, [], [], [], S50.trainIdx, S50.testIdx, '50/50 Holdout');
    end
else
    results_50 = [];
end

%% 5. Test with 5-Fold Cross Validation
fprintf('\n%s\n', repmat('=', 60, 1));
fprintf('TESTING WITH 5-FOLD CROSS VALIDATION (Full dataset)\n');
fprintf('%s\n', repmat('=', 60, 1));
results_cv = run_cross_validation(models, R, H, B, E, F, has_lbp, has_edge, has_fusion);

%% 6. Display comprehensive results tables for all methods
display_all_results(results_70, results_50, results_cv, has_50);

%% 7. Create comprehensive visualizations for all methods
create_comprehensive_plots_all_methods(results_70, results_50, results_cv, has_50);

%% 8. Statistical analysis and comparison
perform_comprehensive_statistical_analysis(results_70, results_50, results_cv, has_50);

%% 9. Save all results
if ~exist('results', 'dir'), mkdir('results'); end
save('results/all_evaluation_results.mat', 'results_70', 'results_50', 'results_cv', 'models');
fprintf('\nAll results saved to: results/all_evaluation_results.mat\n');


%% HELPER FUNCTIONS 

function results = test_all_models(models, R, H, B, E, F, trainIdx, testIdx, split_name)
    fprintf('Testing %d models on %s split (%d test samples)...\n\n', ...
        numel(models), split_name, numel(testIdx));

    % Prepare data
    Xtr_raw = double(R.X_raw(trainIdx,:));   Xte_raw = double(R.X_raw(testIdx,:));
    Xtr_hog = double(H.X_hog(trainIdx,:));   Xte_hog = double(H.X_hog(testIdx,:));
    
    if ~isempty(B)
        Xtr_lbp = double(B.X_lbp(trainIdx,:));   Xte_lbp = double(B.X_lbp(testIdx,:));
    else
        Xtr_lbp = []; Xte_lbp = [];
    end
    
    if ~isempty(E)
        Xtr_edge = double(E.X_edge(trainIdx,:)); Xte_edge = double(E.X_edge(testIdx,:));
    else
        Xtr_edge = []; Xte_edge = [];
    end
    
    if ~isempty(F)
        Xtr_fusion = double(F.X_fusion(trainIdx,:)); Xte_fusion = double(F.X_fusion(testIdx,:));
    else
        Xtr_fusion = []; Xte_fusion = [];
    end
    
    ytr = double(R.y(trainIdx)); yte = double(R.y(testIdx));

    % Load PCA parameters
    P = load('features/pca/features_pca.mat');
    targetVar = 0.95;
    if isfield(P,'k'), default_k = P.k; else, default_k = []; end

    results = struct();
    all_predictions = zeros(numel(yte), numel(models));

    for i = 1:numel(models)
        model_name = models{i};
        fprintf('[%2d/%2d] Testing: %-25s ... ', i, numel(models), model_name);

        try
            % Determine feature type and prepare data
            [X_tr, X_te] = prepare_features(model_name, Xtr_raw, Xte_raw, Xtr_hog, Xte_hog, ...
                                           Xtr_lbp, Xte_lbp, Xtr_edge, Xte_edge, ...
                                           Xtr_fusion, Xte_fusion, ytr, default_k, targetVar);
            
            % Normalise features
            mu = mean(X_tr); sg = std(X_tr); sg(sg<eps) = 1;
            Xtr_norm = (X_tr - mu) ./ sg;
            Xte_norm = (X_te - mu) ./ sg;
            
            % Train and predict based on classifier type
            tic;
            y_pred = train_and_predict(model_name, Xtr_norm, ytr, Xte_norm);
            time_per_sample = toc / numel(yte);

            % Calculate metrics
            [acc,prec,rec,f1,tp,fp,tn,fn] = calculate_comprehensive_metrics(yte, y_pred);
            
            results(i).name = model_name;
            results(i).accuracy = acc; results(i).precision = prec; 
            results(i).recall = rec; results(i).f1 = f1;
            results(i).tp = tp; results(i).fp = fp; 
            results(i).tn = tn; results(i).fn = fn;
            results(i).time_per_sample = time_per_sample;
            results(i).split = split_name;
            all_predictions(:,i) = y_pred;

            fprintf('Acc: %5.2f%% | Prec: %.3f | Rec: %.3f | F1: %.3f | Time: %.4fs\n', ...
                acc, prec, rec, f1, time_per_sample);

        catch ME
            fprintf('ERROR: %s\n', ME.message);
            results(i).name = model_name;
            results(i).accuracy = NaN; results(i).precision = NaN; 
            results(i).recall = NaN; results(i).f1 = NaN;
            results(i).tp = NaN; results(i).fp = NaN; 
            results(i).tn = NaN; results(i).fn = NaN;
            results(i).time_per_sample = NaN; results(i).split = split_name;
            all_predictions(:,i) = zeros(numel(yte),1);
        end
    end

    fprintf('\n');
    create_detailed_confusion_matrices(results, yte, all_predictions, split_name);
end


function [X_tr, X_te] = prepare_features(model_name, Xtr_raw, Xte_raw, Xtr_hog, Xte_hog, ...
                                         Xtr_lbp, Xte_lbp, Xtr_edge, Xte_edge, ...
                                         Xtr_fusion, Xte_fusion, ytr, default_k, targetVar)
    % Determine which features to use based on model name
    if contains(model_name, 'LBP-HOG') || contains(model_name, 'FUSION')
        X_tr = Xtr_fusion; X_te = Xte_fusion;
        
    elseif contains(model_name, '-LBP') && ~contains(model_name, 'LBP-HOG')
        X_tr = Xtr_lbp; X_te = Xte_lbp;
        
    elseif contains(model_name, '-EDGE')
        X_tr = Xtr_edge; X_te = Xte_edge;
        
    elseif contains(model_name, '-HOG') && ~contains(model_name, 'LBP-HOG')
        X_tr = Xtr_hog; X_te = Xte_hog;
        
    elseif contains(model_name, '-PCA')
        % RAW -> Standardise -> PCA
        mu_r = mean(Xtr_raw); sg_r = std(Xtr_raw); sg_r(sg_r<eps)=1;
        Ztr = (Xtr_raw - mu_r) ./ sg_r; 
        Zte = (Xte_raw - mu_r) ./ sg_r;
        
        [coeff, scoreTr, ~, ~, explained] = pca(Ztr);
        cumvar = cumsum(explained);
        if isempty(default_k)
            k = find(cumvar >= targetVar*100, 1, 'first');
        else
            k = default_k;
        end
        X_tr = scoreTr(:,1:k); 
        X_te = Zte * coeff(:,1:k);
        
    elseif contains(model_name, '-LDA')
        % RAW -> Standardise -> PCA -> LDA
        mu_r = mean(Xtr_raw); sg_r = std(Xtr_raw); sg_r(sg_r<eps)=1;
        Ztr = (Xtr_raw - mu_r) ./ sg_r; 
        Zte = (Xte_raw - mu_r) ./ sg_r;
        
        [coeff, scoreTr, ~, ~, explained] = pca(Ztr);
        cumvar = cumsum(explained);
        if isempty(default_k)
            k = find(cumvar >= targetVar*100, 1, 'first');
        else
            k = default_k;
        end
        Xtr_pca = scoreTr(:,1:k); 
        Xte_pca = Zte * coeff(:,1:k);
        
        % Apply LDA
        lda = fitcdiscr(Xtr_pca, ytr, 'DiscrimType','linear');
        if size(lda.Coeffs,1)>=2
            W = lda.Coeffs(1,2).Linear;
        else
            W = ones(size(Xtr_pca,2),1);
        end
        X_tr = Xtr_pca * W; 
        X_te = Xte_pca * W;
        
    else  % RAW features
        X_tr = Xtr_raw; 
        X_te = Xte_raw;
    end
end


function y_pred = train_and_predict(model_name, Xtr_norm, ytr, Xte_norm)
    % Train and predict based on classifier type
    
    if startsWith(model_name, 'NN-')
        % Nearest Neighbor (1-NN)
        mdl = fitcknn(Xtr_norm, ytr, 'NumNeighbors', 1, 'Distance', 'euclidean');
        y_pred = predict(mdl, Xte_norm);
        
    elseif startsWith(model_name, 'KNN-')
        % K-Nearest Neighbors (optimized K)
        K = get_best_K(model_name);
        mdl = fitcknn(Xtr_norm, ytr, 'NumNeighbors', K, 'Distance', 'euclidean');
        y_pred = predict(mdl, Xte_norm);
        
    elseif contains(model_name, 'SVM-Linear')
        % Linear SVM
        C = get_best_C(model_name);
        mdl = fitcsvm(Xtr_norm, ytr, 'KernelFunction', 'linear', ...
                      'BoxConstraint', C, 'ClassNames', [0 1], 'Standardize', false);
        y_pred = predict(mdl, Xte_norm);
        
    elseif contains(model_name, 'SVM-RBF')
        % RBF SVM
        [C, ks] = get_best_C_and_KS(model_name);
        mdl = fitcsvm(Xtr_norm, ytr, 'KernelFunction', 'rbf', ...
                      'BoxConstraint', C, 'KernelScale', ks, ...
                      'ClassNames', [0 1], 'Standardize', false);
        y_pred = predict(mdl, Xte_norm);
        
    elseif startsWith(model_name, 'Neural-')
        % Single hidden layer neural network
        net = patternnet(10);
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 100;
        net = train(net, Xtr_norm', ytr');
        y_pred = double(net(Xte_norm')' > 0.5);
        
    elseif startsWith(model_name, 'DNN-')
        % Deep neural network (2 hidden layers)
        net = patternnet([20, 10]);
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 100;
        net = train(net, Xtr_norm', ytr');
        y_pred = double(net(Xte_norm')' > 0.5);
        
    elseif startsWith(model_name, 'CNN-')
        % CNN-like architecture (3 layers)
        net = patternnet([30, 20, 10]);
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 100;
        net = train(net, Xtr_norm', ytr');
        y_pred = double(net(Xte_norm')' > 0.5);
        
    elseif startsWith(model_name, 'RF-')
        % Random Forest
        mdl = TreeBagger(50, Xtr_norm, ytr, 'Method', 'classification');
        y_pred = str2double(predict(mdl, Xte_norm));
        
    elseif startsWith(model_name, 'Bagging-')
        % Bagging ensemble
        mdl = TreeBagger(30, Xtr_norm, ytr, 'Method', 'classification');
        y_pred = str2double(predict(mdl, Xte_norm));
        
    else
        error('Unknown classifier type: %s', model_name);
    end
end


function K = get_best_K(model_name)
    % Get best K for KNN based on feature type
    try
        if contains(model_name, 'HOG')
            M = load('models/knn/modelKNN_hog.mat');
            K = M.bestHog.K;
        else
            M = load('models/knn/modelKNN_raw.mat');
            K = M.bestRaw.K;
        end
    catch
        K = 5;  % Default
    end
end


function C = get_best_C(model_name)
    % Get best C for Linear SVM
    try
        if contains(model_name, 'HOG')
            M = load('models/svm/modelSVM_linear_hog.mat');
            C = M.bestHog.C;
        else
            M = load('models/svm/modelSVM_linear_raw.mat');
            C = M.bestRaw.C;
        end
    catch
        C = 1;  %
    end
end


function [C, ks] = get_best_C_and_KS(model_name)
    % Get best C and KernelScale for RBF SVM
    try
        if contains(model_name, 'LBP-HOG') || contains(model_name, 'FUSION')
            M = load('models/svm/modelSVM_rbf_fusion.mat');
            C = M.bestFusion.C;
            ks = M.bestFusion.ks;
        elseif contains(model_name, 'LBP')
            M = load('models/svm/modelSVM_rbf_lbp.mat');
            C = M.bestLbp.C;
            ks = M.bestLbp.ks;
        elseif contains(model_name, 'EDGE')
            M = load('models/svm/modelSVM_rbf_edge.mat');
            C = M.bestEdge.C;
            ks = M.bestEdge.ks;
        elseif contains(model_name, 'PCA')
            M = load('models/svm/modelSVM_rbf_pca.mat');
            C = M.bestPca.C;
            ks = M.bestPca.ks;
        elseif contains(model_name, 'LDA')
            M = load('models/svm/modelSVM_rbf_lda.mat');
            C = M.bestLda.C;
            ks = M.bestLda.ks;
        elseif contains(model_name, 'HOG')
            M = load('models/svm/modelSVM_rbf_hog.mat');
            C = M.bestHog.C;
            ks = M.bestHog.ks;
        else
            M = load('models/svm/modelSVM_rbf_raw.mat');
            C = M.bestRaw.C;
            ks = M.bestRaw.ks;
        end
    catch
        C = 1;      
        ks = 'auto'; 
    end
end


function results_cv = run_cross_validation(models, R, H, B, E, F, has_lbp, has_edge, has_fusion)
    X_raw = double(R.X_raw);
    X_hog = double(H.X_hog);
    if has_lbp, X_lbp = double(B.X_lbp); else, X_lbp = []; end
    if has_edge, X_edge = double(E.X_edge); else, X_edge = []; end
    if has_fusion, X_fusion = double(F.X_fusion); else, X_fusion = []; end
    y = double(R.y);

    N = numel(y);
    k_folds = 5;
    indices = crossvalind('Kfold', N, k_folds);

    % PCA parameters
    targetVar = 0.95;
    P = load('features/pca/features_pca.mat');
    if isfield(P,'k'), default_k = P.k; else, default_k = []; end

    results_cv = struct();
    fprintf('Running %d-fold cross-validation on %d samples...\n\n', k_folds, N);

    for m = 1:numel(models)
        model_name = models{m};
        fprintf('[%2d/%2d] CV: %-25s ... ', m, numel(models), model_name);

        accs = zeros(k_folds,1); precs = zeros(k_folds,1);
        recs = zeros(k_folds,1); f1s = zeros(k_folds,1);
        times = zeros(k_folds,1);

        try
            for fold = 1:k_folds
                test_idx = (indices == fold);
                train_idx = ~test_idx;

                y_train = y(train_idx);
                y_test  = y(test_idx);

                % Prepare features for this fold
                [X_train, X_test] = prepare_features_cv(model_name, X_raw, X_hog, X_lbp, X_edge, X_fusion, ...
                                                       train_idx, test_idx, y_train, default_k, targetVar);
                
                % Normalise
                mu = mean(X_train); sg = std(X_train); sg(sg<eps)=1;
                Xtr = (X_train - mu) ./ sg; 
                Xte = (X_test - mu) ./ sg;

                % Train and predict
                tic;
                y_pred = train_and_predict(model_name, Xtr, y_train, Xte);
                times(fold) = toc/numel(y_test);

                [accs(fold),precs(fold),recs(fold),f1s(fold)] = calculate_comprehensive_metrics(y_test,y_pred);
            end

            results_cv(m).name = model_name;
            results_cv(m).mean_acc = nanmean(accs); results_cv(m).std_acc = nanstd(accs);
            results_cv(m).mean_prec = nanmean(precs); results_cv(m).std_prec = nanstd(precs);
            results_cv(m).mean_rec = nanmean(recs); results_cv(m).std_rec = nanstd(recs);
            results_cv(m).mean_f1 = nanmean(f1s); results_cv(m).std_f1 = nanstd(f1s);
            results_cv(m).mean_time = nanmean(times);

            fprintf('Mean: %5.2f%% (±%.2f%%)\n', results_cv(m).mean_acc, results_cv(m).std_acc);

        catch ME
            fprintf('ERROR: %s\n', ME.message);
            results_cv(m).name = model_name;
            results_cv(m).mean_acc = NaN; results_cv(m).std_acc = NaN;
            results_cv(m).mean_prec = NaN; results_cv(m).std_prec = NaN;
            results_cv(m).mean_rec = NaN; results_cv(m).std_rec = NaN;
            results_cv(m).mean_f1 = NaN; results_cv(m).std_f1 = NaN;
            results_cv(m).mean_time = NaN;
        end
    end
    fprintf('\n');
end


function [X_train, X_test] = prepare_features_cv(model_name, X_raw, X_hog, X_lbp, X_edge, X_fusion, ...
                                                 train_idx, test_idx, y_train, default_k, targetVar)
    % Prepare features for cross-validation (similar to prepare_features but with indexing)
    if contains(model_name, 'LBP-HOG') || contains(model_name, 'FUSION')
        X_train = X_fusion(train_idx,:); X_test = X_fusion(test_idx,:);
        
    elseif contains(model_name, '-LBP') && ~contains(model_name, 'LBP-HOG')
        X_train = X_lbp(train_idx,:); X_test = X_lbp(test_idx,:);
        
    elseif contains(model_name, '-EDGE')
        X_train = X_edge(train_idx,:); X_test = X_edge(test_idx,:);
        
    elseif contains(model_name, '-HOG') && ~contains(model_name, 'LBP-HOG')
        X_train = X_hog(train_idx,:); X_test = X_hog(test_idx,:);
        
    elseif contains(model_name, '-PCA')
        Xtr_raw = X_raw(train_idx,:); Xte_raw = X_raw(test_idx,:);
        mu_r = mean(Xtr_raw); sg_r = std(Xtr_raw); sg_r(sg_r<eps)=1;
        Ztr = (Xtr_raw - mu_r) ./ sg_r; Zte = (Xte_raw - mu_r) ./ sg_r;
        
        [coeff, scoreTr, ~, ~, explained] = pca(Ztr);
        cumvar = cumsum(explained);
        if isempty(default_k)
            k = find(cumvar >= targetVar*100, 1, 'first');
        else
            k = default_k;
        end
        X_train = scoreTr(:,1:k); X_test = Zte * coeff(:,1:k);
        
    elseif contains(model_name, '-LDA')
        Xtr_raw = X_raw(train_idx,:); Xte_raw = X_raw(test_idx,:);
        mu_r = mean(Xtr_raw); sg_r = std(Xtr_raw); sg_r(sg_r<eps)=1;
        Ztr = (Xtr_raw - mu_r) ./ sg_r; Zte = (Xte_raw - mu_r) ./ sg_r;
        
        [coeff, scoreTr, ~, ~, explained] = pca(Ztr);
        cumvar = cumsum(explained);
        if isempty(default_k)
            k = find(cumvar >= targetVar*100, 1, 'first');
        else
            k = default_k;
        end
        Xtr_pca = scoreTr(:,1:k); Xte_pca = Zte * coeff(:,1:k);
        
        lda = fitcdiscr(Xtr_pca, y_train, 'DiscrimType','linear');
        if size(lda.Coeffs,1)>=2
            W = lda.Coeffs(1,2).Linear;
        else
            W = ones(size(Xtr_pca,2),1);
        end
        X_train = Xtr_pca * W; X_test = Xte_pca * W;
        
    else  % RAW
        X_train = X_raw(train_idx,:); X_test = X_raw(test_idx,:);
    end
end


function [accuracy, precision, recall, f1, tp, fp, tn, fn] = calculate_comprehensive_metrics(y_true, y_pred)
    tp = sum((y_pred == 1) & (y_true == 1));
    fp = sum((y_pred == 1) & (y_true == 0));
    tn = sum((y_pred == 0) & (y_true == 0));
    fn = sum((y_pred == 0) & (y_true == 1));
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) * 100;
    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
end


function create_detailed_confusion_matrices(results, y_true, all_predictions, split_name)
    n_models = numel(results);

    % Automatically choose grid size (approximately square)
    cols = ceil(sqrt(n_models));
    rows = ceil(n_models / cols);

    fig = figure('Position', [100 100 1800 1200], ...
        'Name', ['Confusion Matrices - ' split_name], ...
        'Visible', 'off');

    for i = 1:n_models
        subplot(rows, cols, i);
        if ~isnan(results(i).accuracy)
            cm = confusionmat(y_true, all_predictions(:, i));
            confusionchart(cm, {'Non-Ped', 'Ped'}, ...
                'Title', sprintf('%s (%.1f%%)', results(i).name, results(i).accuracy), ...
                'FontSize', 8);
        else
            text(0.5, 0.5, 'FAILED', ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'Color', 'red');
            title(sprintf('%s\nFAILED', results(i).name), 'Color', 'red', 'FontSize', 8);
            axis off;
        end
    end

    sgtitle(sprintf('Confusion Matrices - %s', split_name), ...
        'FontSize', 14, 'FontWeight', 'bold');

    if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
    saveas(fig, sprintf('results/figures/Fig_Confusion_Matrices_%s.png', ...
        strrep(split_name, '/', '_')));
    fprintf('Confusion matrices saved for %s\n', split_name);
    close(fig);
end


function display_all_results(results_70, results_50, results_cv, has_50)
    % Display 70/30 results
    fprintf('\n%s\n', repmat('=', 110, 1));
    fprintf('%-35s RESULTS SUMMARY - 70/30 HOLDOUT\n', '');
    fprintf('%s\n', repmat('=', 110, 1));
    
    fprintf('%-25s | %8s | %7s | %7s | %7s | %5s | %5s | %5s | %5s | %10s\n', ...
        'Model', 'Accuracy', 'Prec', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN', 'Time(ms)');
    fprintf('%s\n', repmat('-', 110, 1));
    
    for i = 1:numel(results_70)
        r = results_70(i);
        if ~isnan(r.accuracy)
            fprintf('%-25s | %7.2f%% | %7.3f | %7.3f | %7.3f | %5d | %5d | %5d | %5d | %10.2f\n', ...
                r.name, r.accuracy, r.precision, r.recall, r.f1, ...
                r.tp, r.fp, r.tn, r.fn, r.time_per_sample*1000);
        else
            fprintf('%-25s | %8s | %7s | %7s | %7s | %5s | %5s | %5s | %5s | %10s\n', ...
                r.name, 'FAILED', '-', '-', '-', '-', '-', '-', '-', '-');
        end
    end
    
    % Display 50/50 results if available
    if has_50 && ~isempty(results_50)
        fprintf('\n%s\n', repmat('=', 110, 1));
        fprintf('%-35s RESULTS SUMMARY - 50/50 HOLDOUT\n', '');
        fprintf('%s\n', repmat('=', 110, 1));
        
        for i = 1:numel(results_50)
            r = results_50(i);
            if ~isnan(r.accuracy)
                fprintf('%-25s | %7.2f%% | %7.3f | %7.3f | %7.3f | %5d | %5d | %5d | %5d | %10.2f\n', ...
                    r.name, r.accuracy, r.precision, r.recall, r.f1, ...
                    r.tp, r.fp, r.tn, r.fn, r.time_per_sample*1000);
            else
                fprintf('%-25s | %8s | %7s | %7s | %7s | %5s | %5s | %5s | %5s | %10s\n', ...
                    r.name, 'FAILED', '-', '-', '-', '-', '-', '-', '-', '-');
            end
        end
    end
    
    % Display Cross-Validation results
    fprintf('\n%s\n', repmat('=', 110, 1));
    fprintf('%-30s RESULTS SUMMARY - 5-FOLD CROSS VALIDATION\n', '');
    fprintf('%s\n', repmat('=', 110, 1));
    
    fprintf('%-25s | %16s | %16s | %16s | %16s | %10s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time(ms)');
    fprintf('%s\n', repmat('-', 110, 1));
    
    for m = 1:numel(results_cv)
        r = results_cv(m);
        if ~isnan(r.mean_acc)
            fprintf('%-25s | %6.2f±%-6.2f%% | %6.3f±%-6.3f | %6.3f±%-6.3f | %6.3f±%-6.3f | %10.2f\n', ...
                r.name, r.mean_acc, r.std_acc, r.mean_prec, r.std_prec, ...
                r.mean_rec, r.std_rec, r.mean_f1, r.std_f1, r.mean_time*1000);
        else
            fprintf('%-25s | %16s | %16s | %16s | %16s | %10s\n', ...
                r.name, 'FAILED', '-', '-', '-', '-');
        end
    end
    fprintf('%s\n', repmat('=', 110, 1));
end


function create_comprehensive_plots_all_methods(results_70, results_50, results_cv, has_50)
    fig = figure('Position', [50 50 1800 1200], 'Name', 'All Validation Methods Comparison');
    
    models = {results_70.name};
    acc_70 = [results_70.accuracy];
    
    if has_50 && ~isempty(results_50)
        acc_50 = [results_50.accuracy];
    else
        acc_50 = nan(size(acc_70));
    end
    acc_cv = [results_cv.mean_acc];
    
    % Plot 1: Accuracy comparison
    subplot(2, 3, 1);
    x = 1:numel(models);
    hold on;
    plot(x, acc_70, 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '70/30');
    if has_50, plot(x, acc_50, 's-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '50/50'); end
    plot(x, acc_cv, '^-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '5-Fold CV');
    xlabel('Model Index'); ylabel('Accuracy (%)');
    title('Accuracy Comparison - All Methods');
    legend('Location', 'southeast'); grid on;
    ylim([70, 100]);
    
    % Plot 2: CV Reliability
    subplot(2, 3, 2);
    cv_std = [results_cv.std_acc];
    bar(cv_std); xlabel('Model Index'); ylabel('Std Dev (%)');
    title('5-Fold CV Reliability (Lower = Better)'); grid on;
    
    % Plot 3: Best models
    subplot(2, 3, 3);
    [best_70, idx70] = max(acc_70);
    [best_cv, idxcv] = max(acc_cv);
    methods = {'70/30', '5-Fold CV'};
    best_vals = [best_70, best_cv];
    bar(best_vals); set(gca, 'XTickLabel', methods);
    ylabel('Best Accuracy (%)'); title('Best Model Each Method');
    ylim([95, 100]); grid on;
    for i=1:numel(best_vals)
        text(i, best_vals(i)+0.2, sprintf('%.2f%%', best_vals(i)), ...
            'HorizontalAlignment', 'center');
    end
    
    % Plot 4: Computational efficiency
    subplot(2, 3, 4);
    times_70 = [results_70.time_per_sample] * 1000;
    times_cv = [results_cv.mean_time] * 1000;
    bar([times_70; times_cv]'); xlabel('Model Index'); ylabel('Time (ms)');
    title('Computational Efficiency'); legend('70/30', '5-Fold CV'); grid on;
    
    % Plot 5: Feature performance
    subplot(2, 3, 5);
    feat_types = {'RAW', 'HOG', 'PCA', 'LDA', 'LBP', 'EDGE', 'FUSION'};
    avg_70 = []; avg_cv = [];
    for f = 1:numel(feat_types)
        idx = contains(models, feat_types{f});
        if any(idx)
            avg_70(end+1) = mean(acc_70(idx), 'omitnan');
            avg_cv(end+1) = mean(acc_cv(idx), 'omitnan');
        end
    end
    bar([avg_70; avg_cv]'); set(gca, 'XTickLabel', feat_types(1:numel(avg_70)));
    ylabel('Average Accuracy (%)'); title('Feature Type Performance');
    legend({'70/30', '5-Fold CV'}); grid on;
    
    % Plot 6: Classifier performance
    subplot(2, 3, 6);
    class_types = {'NN', 'KNN', 'SVM-Linear', 'SVM-RBF', 'Neural', 'DNN', 'CNN', 'RF', 'Bagging'};
    avg_class_70 = []; avg_class_cv = [];
    for c = 1:numel(class_types)
        idx = startsWith(models, class_types{c});
        if any(idx)
            avg_class_70(end+1) = mean(acc_70(idx), 'omitnan');
            avg_class_cv(end+1) = mean(acc_cv(idx), 'omitnan');
        end
    end
    bar([avg_class_70; avg_class_cv]'); 
    set(gca, 'XTickLabel', class_types(1:numel(avg_class_70)), 'XTickLabelRotation', 45);
    ylabel('Average Accuracy (%)'); title('Classifier Performance');
    legend({'70/30', '5-Fold CV'}); grid on;
    
    if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
    saveas(fig, 'results/figures/Fig_All_Methods_Comparison.png');
    fprintf('Comparison figure saved\n');
    close(fig);
end


function perform_comprehensive_statistical_analysis(results_70, results_50, results_cv, has_50)
    fprintf('\n%s\n', repmat('=', 80, 1));
    fprintf('%-20s COMPREHENSIVE STATISTICAL ANALYSIS\n', '');
    fprintf('%s\n', repmat('=', 80, 1));
    
    % Best models
    accs_70 = [results_70.accuracy];
    [best_70, idx70] = max(accs_70);
    name_70 = results_70(idx70).name;
    
    accs_cv = [results_cv.mean_acc];
    [best_cv, idxcv] = max(accs_cv);
    name_cv = results_cv(idxcv).name;
    std_cv = results_cv(idxcv).std_acc;
    
    fprintf('\nBEST MODELS:\n');
    fprintf('  70/30 Holdout:  %-25s %.2f%%\n', name_70, best_70);
    if has_50 && ~isempty(results_50)
        accs_50 = [results_50.accuracy];
        [best_50, idx50] = max(accs_50);
        name_50 = results_50(idx50).name;
        fprintf('  50/50 Holdout:  %-25s %.2f%%\n', name_50, best_50);
    end
    fprintf('  5-Fold CV:      %-25s %.2f%% ± %.2f%%\n', name_cv, best_cv, std_cv);
    
    % Feature analysis
    fprintf('\nFEATURE ANALYSIS (70/30 Holdout):\n');
    models = {results_70.name};
    feat_types = {'RAW', 'HOG', 'PCA', 'LDA', 'LBP', 'EDGE', 'LBP-HOG'};
    for f = 1:numel(feat_types)
        idx = contains(models, feat_types{f});
        if any(idx)
            avg_acc = mean(accs_70(idx), 'omitnan');
            fprintf('  %-15s: %.2f%% (avg across %d classifiers)\n', ...
                feat_types{f}, avg_acc, sum(idx));
        end
    end
    
    % Classifier analysis
    fprintf('\nCLASSIFIER ANALYSIS (70/30 Holdout):\n');
    class_types = {'NN-', 'KNN-', 'SVM-Linear', 'SVM-RBF', 'Neural-', 'DNN-', 'CNN-', 'RF-', 'Bagging-'};
    for c = 1:numel(class_types)
        idx = startsWith(models, class_types{c});
        if any(idx)
            avg_acc = mean(accs_70(idx), 'omitnan');
            fprintf('  %-15s: %.2f%% (avg across %d features)\n', ...
                strrep(class_types{c}, '-', ''), avg_acc, sum(idx));
        end
    end
    
    % Validation method reliability
    fprintf('\nVALIDATION METHOD RELIABILITY:\n');
    cv_stds = [results_cv.std_acc];
    valid_stds = cv_stds(~isnan(cv_stds));
    if ~isempty(valid_stds)
        fprintf('  5-Fold CV average std: %.3f%%\n', mean(valid_stds));
        [min_std, min_idx] = min(valid_stds);
        fprintf('  Most stable model: %s (std = %.3f%%)\n', results_cv(min_idx).name, min_std);
    end
    
    fprintf('\n%s\n', repmat('=', 80, 1));

end