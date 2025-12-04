%% cross_validation.m 
% Tests using other parameters and the best models using 5-fold since time
% running still fast enough with accuracy stable and good 
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf('CROSS-VALIDATION Test\n');

% Load features
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

X_raw = double(R.X_raw);
X_hog = double(H.X_hog);
y = double(R.y);

N = numel(y);
k_folds = 5; % here where we change for each fold update 

fprintf('Dataset: %d samples\n', N);
fprintf('Using %d-fold cross-validation\n\n', k_folds);

% Create fold indices
indices = crossvalind('Kfold', N, k_folds);

% Models to test 
models_to_test = {
    'KNN-HOG', 'hog', 'knn';
    'SVM-Linear-HOG', 'hog', 'svm_linear';
    'SVM-RBF-HOG', 'hog', 'svm_rbf'
};

results = struct();

%% Run Cross-Validation for Each Model
for m = 1:size(models_to_test, 1)
    model_name = models_to_test{m, 1};
    feature_type = models_to_test{m, 2};
    classifier_type = models_to_test{m, 3};
    
    fprintf('--- %s ---\n', model_name);
    
    % Select features
    if strcmp(feature_type, 'hog')
        X = X_hog;
    else
        X = X_raw;
    end
    
    % Storage for fold results
    fold_acc = zeros(k_folds, 1);
    fold_prec = zeros(k_folds, 1);
    fold_rec = zeros(k_folds, 1);
    fold_f1 = zeros(k_folds, 1);
    
    % Run each fold
    for fold = 1:k_folds
        fprintf('  Fold %d/%d... ', fold, k_folds);
        
        % Split data
        test_idx = (indices == fold);
        train_idx = ~test_idx;
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        % Normalize
        mu = mean(X_train);
        sigma = std(X_train);
        sigma(sigma < eps) = 1;
        
        X_train_norm = (X_train - mu) ./ sigma;
        X_test_norm = (X_test - mu) ./ sigma;
        
        % Train and predict based on classifier type
        try
            switch classifier_type
                case 'knn'
                    % Load best K from your trained model
                    try
                        if strcmp(feature_type, 'hog')
                            M = load('models/knn/modelKNN_hog.mat');
                            K = M.bestHog.K;
                        else
                            M = load('models/knn/modelKNN_raw.mat');
                            K = M.bestRaw.K;
                        end
                    catch
                        K = 1; % Default
                    end
                    
                    mdl = fitcknn(X_train_norm, y_train, ...
                        'NumNeighbors', K, ...
                        'Distance', 'euclidean', ...
                        'Standardize', false);
                    y_pred = predict(mdl, X_test_norm);
                    
                case 'svm_linear'
                    % Load best C from your trained model
                    try
                        if strcmp(feature_type, 'hog')
                            M = load('models/svm/modelSVM_linear_hog.mat');
                            C = M.bestHog.C;
                        else
                            M = load('models/svm/modelSVM_linear_raw.mat');
                            C = M.bestRaw.C;
                        end
                    catch
                        C = 1; % Default
                    end
                    
                    mdl = fitcsvm(X_train_norm, y_train, ...
                        'KernelFunction', 'linear', ...
                        'BoxConstraint', C, ...
                        'ClassNames', [0 1], ...
                        'Standardize', false);
                    y_pred = predict(mdl, X_test_norm);
                    
                case 'svm_rbf'
                    % Load best C and KernelScale from your trained model
                    try
                        if strcmp(feature_type, 'hog')
                            M = load('models/svm/modelSVM_rbf_hog.mat');
                            C = M.bestHog.C;
                            ks = M.bestHog.ks;
                        else
                            M = load('models/svm/modelSVM_rbf_raw.mat');
                            C = M.bestRaw.C;
                            ks = M.bestRaw.ks;
                        end
                    catch
                        C = 10;
                        ks = 'auto';
                    end
                    
                    mdl = fitcsvm(X_train_norm, y_train, ...
                        'KernelFunction', 'rbf', ...
                        'KernelScale', ks, ...
                        'BoxConstraint', C, ...
                        'ClassNames', [0 1], ...
                        'Standardize', false);
                    y_pred = predict(mdl, X_test_norm);
            end
            
            % Calculate metrics
            tp = sum((y_pred==1) & (y_test==1));
            fp = sum((y_pred==1) & (y_test==0));
            tn = sum((y_pred==0) & (y_test==0));
            fn = sum((y_pred==0) & (y_test==1));
            
            fold_acc(fold) = (tp + tn) / numel(y_test) * 100;
            fold_prec(fold) = tp / (tp + fp + eps);
            fold_rec(fold) = tp / (tp + fn + eps);
            fold_f1(fold) = 2 * fold_prec(fold) * fold_rec(fold) / ...
                           (fold_prec(fold) + fold_rec(fold) + eps);
            
            fprintf('Acc: %.2f%%\n', fold_acc(fold));
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            fold_acc(fold) = NaN;
            fold_prec(fold) = NaN;
            fold_rec(fold) = NaN;
            fold_f1(fold) = NaN;
        end
    end
    
    % Calculate statistics
    results(m).name = model_name;
    results(m).mean_acc = mean(fold_acc);
    results(m).std_acc = std(fold_acc);
    results(m).mean_prec = mean(fold_prec);
    results(m).std_prec = std(fold_prec);
    results(m).mean_rec = mean(fold_rec);
    results(m).std_rec = std(fold_rec);
    results(m).mean_f1 = mean(fold_f1);
    results(m).std_f1 = std(fold_f1);
    results(m).fold_acc = fold_acc;
    
    fprintf('  → Mean: %.2f%% (±%.2f%%)\n\n', ...
        results(m).mean_acc, results(m).std_acc);
end

%% Display Results
fprintf('   CROSS-VALIDATION RESULTS SUMMARY\n');

fprintf('%-20s | %12s | %12s | %12s | %12s\n', ...
    'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 85, 1));

for m = 1:numel(results)
    fprintf('%-20s | %5.2f±%.2f%% | %5.3f±%.3f | %5.3f±%.3f | %5.3f±%.3f\n', ...
        results(m).name, ...
        results(m).mean_acc, results(m).std_acc, ...
        results(m).mean_prec, results(m).std_prec, ...
        results(m).mean_rec, results(m).std_rec, ...
        results(m).mean_f1, results(m).std_f1);
end

fprintf('\n');

%% Visualization
fig = figure('Position', [100 100 1000 400], 'Name', 'Cross-Validation Results');

subplot(1,2,1);
% Box plot of accuracies
data_matrix = zeros(k_folds, numel(results));
for m = 1:numel(results)
    data_matrix(:, m) = results(m).fold_acc;
end
boxplot(data_matrix, 'Labels', {results.name});
ylabel('Accuracy (%)');
title('Cross-Validation: Accuracy Distribution');
grid on;
xtickangle(45);
ylim([90 100]);

subplot(1,2,2);
% Bar chart with error bars
means = [results.mean_acc];
stds = [results.std_acc];
b = bar(means);
hold on;
errorbar(1:numel(results), means, stds, 'k.', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {results.name});
xtickangle(45);
ylabel('Accuracy (%)');
title('Mean Accuracy with Standard Deviation');
grid on;
ylim([90 100]);

% Save figure
if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
saveas(fig, 'results/figures/Fig_CrossValidation.png');

fprintf('Cross-validation complete!\n');

%% Save results
save('results/cv_results.mat', 'results');
