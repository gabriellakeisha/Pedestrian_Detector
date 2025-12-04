%% b_loocv_fast.m - Leave-One-Out Cross-Validation 
clearvars; close all; clc; rng(1234,'twister'); 
addpath(genpath('src'));

fprintf('LEAVE-ONE-OUT CV\n');


%% Load Features and Splits
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');
S = load('splits/splits.mat');

X_raw = double(R.X_raw);
X_hog = double(H.X_hog);
y = double(R.y);

% Use only test set for LOOCV
test_idx_all = S.testIdx;
train_idx_all = S.trainIdx;

N_test = numel(test_idx_all);
N_total = numel(y);

fprintf('Total dataset: %d samples\n', N_total);
fprintf('Training set: %d samples (will be used as base)\n', numel(train_idx_all));
fprintf('Test set: %d samples (LOOCV will be performed on these)\n', N_test);
fprintf('Estimated time: %.1f - %.1f minutes\n\n', N_test*0.01, N_test*0.03);

%%Models to test 
models_to_test = {
    'KNN-HOG', 'hog', 'knn';
    'SVM-Linear-HOG', 'hog', 'svm_linear';
    'SVM-RBF-HOG', 'hog', 'svm_rbf'
};

results = struct();

%% Run LOOCV for Each Model
for m = 1:size(models_to_test, 1)
    model_name = models_to_test{m, 1};
    feature_type = models_to_test{m, 2};
    classifier_type = models_to_test{m, 3};
    
    fprintf('\n%s\n', repmat('=', 80, 1));
    fprintf('Testing: %s\n', model_name);
    fprintf('%s\n', repmat('=', 80, 1));
    
    % Select features
    if strcmp(feature_type, 'hog')
        X = X_hog;
    else
        X = X_raw;
    end
    
    % Extract test set
    X_test_set = X(test_idx_all, :);
    y_test_set = y(test_idx_all);
    
    % Extract training set (will be combined with N-1 test samples)
    X_train_base = X(train_idx_all, :);
    y_train_base = y(train_idx_all);
    
    % Storage for predictions
    y_pred = zeros(N_test, 1);
    
    % Progress tracking
    total_start = tic;
    update_interval = max(1, floor(N_test / 20));
    
    % LOOCV on test set
    for i = 1:N_test
        % Progress indicator
        if mod(i, update_interval) == 0 || i == N_test
            elapsed = toc(total_start);
            pct = (i/N_test)*100;
            eta = (elapsed/i) * (N_test-i);
            fprintf('  [%5.1f%%] Sample %d/%d | Elapsed: %s | ETA: %s\n', ...
                pct, i, N_test, format_time(elapsed), format_time(eta));
        end
        
        % Leave one test sample out
        test_sample_idx = i;
        train_from_test_idx = setdiff(1:N_test, i);
        
        % Combine: full training set + (N-1) test samples
        X_train = [X_train_base; X_test_set(train_from_test_idx, :)];
        y_train = [y_train_base; y_test_set(train_from_test_idx)];
        
        X_test = X_test_set(test_sample_idx, :);
        
        % Normalise
        mu = mean(X_train);
        sigma = std(X_train);
        sigma(sigma < eps) = 1;
        
        X_train_norm = (X_train - mu) ./ sigma;
        X_test_norm = (X_test - mu) ./ sigma;
        
        % Train and predict
        try
            switch classifier_type
                case 'knn'
                    mdl = fitcknn(X_train_norm, y_train, ...
                        'NumNeighbors', 5, ...
                        'Distance', 'euclidean', ...
                        'Standardize', false);
                    
                case 'svm_linear'
                    mdl = fitcsvm(X_train_norm, y_train, ...
                        'KernelFunction', 'linear', ...
                        'BoxConstraint', 1, ...
                        'Standardize', false);
                    
                case 'svm_rbf'
                    mdl = fitcsvm(X_train_norm, y_train, ...
                        'KernelFunction', 'rbf', ...
                        'BoxConstraint', 10, ...
                        'KernelScale', 'auto', ...
                        'Standardize', false);
            end
            
            y_pred(i) = predict(mdl, X_test_norm);
            
        catch ME
            fprintf('    ERROR at sample %d: %s\n', i, ME.message);
            y_pred(i) = NaN;
        end
    end
    
    total_time = toc(total_start);
    
    % Calculate metrics
    [acc, prec, rec, f1, tp, fp, tn, fn] = calculate_loocv_metrics(y_test_set, y_pred);
    
    % Store results
    results(m).name = model_name;
    results(m).accuracy = acc;
    results(m).precision = prec;
    results(m).recall = rec;
    results(m).f1 = f1;
    results(m).tp = tp;
    results(m).fp = fp;
    results(m).tn = tn;
    results(m).fn = fn;
    results(m).total_time = total_time;
    results(m).time_per_sample = total_time / N_test;
    results(m).y_true = y_test_set;
    results(m).y_pred = y_pred;
    results(m).n_samples = N_test;
    
    fprintf('\n  LOOCV Results (on %d test samples):\n', N_test);
    fprintf('Accuracy:  %.2f%%\n', acc);
    fprintf('Precision: %.4f\n', prec);
    fprintf('Recall:    %.4f\n', rec);
    fprintf('F1-Score:  %.4f\n', f1);
    fprintf('TP: %d | FP: %d | TN: %d | FN: %d\n', tp, fp, tn, fn);
    fprintf('Total Time: %s (%.3f sec/sample)\n', ...
        format_time(total_time), total_time/N_test);
end

%% Display Results
display_loocv_results(results);

%% Create Visualizations
plot_loocv_results(results);

%% Save Results
save_loocv_results(results);

fprintf('\nLOOCV Complete (Fast Version - Test Set Only)!\n');
fprintf('Best model: %s (%.2f%% accuracy)\n', get_best_model(results));

%% Helper Function
function [acc, prec, rec, f1, tp, fp, tn, fn] = calculate_loocv_metrics(y_true, y_pred)
    valid = ~isnan(y_pred);
    y_true = y_true(valid);
    y_pred = y_pred(valid);
    
    tp = sum((y_pred == 1) & (y_true == 1));
    fp = sum((y_pred == 1) & (y_true == 0));
    tn = sum((y_pred == 0) & (y_true == 0));
    fn = sum((y_pred == 0) & (y_true == 1));
    
    acc = (tp + tn) / (tp + fp + tn + fn) * 100;
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps);
    f1 = 2 * (prec * rec) / (prec + rec + eps);
end

function display_loocv_results(results)
    N = results(1).n_samples;
    fprintf('\n%s\n', repmat('=', 100, 1));
    fprintf('LEAVE-ONE-OUT CROSS-VALIDATION RESULTS (N=%d test samples)\n', N);
    fprintf('%s\n', repmat('=', 100, 1));
    
    fprintf('%-20s | %8s | %9s | %9s | %9s | %9s | %12s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time(s)', 'Sec/Sample');
    fprintf('%s\n', repmat('-', 100, 1));
    
    for m = 1:numel(results)
        fprintf('%-20s | %7.2f%% | %9.4f | %9.4f | %9.4f | %9.1f | %12.4f\n', ...
            results(m).name, results(m).accuracy, results(m).precision, ...
            results(m).recall, results(m).f1, ...
            results(m).total_time, results(m).time_per_sample);
    end
    fprintf('%s\n', repmat('-', 100, 1));
    
    fprintf('\nConfusion Matrix Details:\n');
    fprintf('%-20s | %6s | %6s | %6s | %6s\n', 'Model', 'TP', 'FP', 'TN', 'FN');
    fprintf('%s\n', repmat('-', 60, 1));
    for m = 1:numel(results)
        fprintf('%-20s | %6d | %6d | %6d | %6d\n', ...
            results(m).name, results(m).tp, results(m).fp, ...
            results(m).tn, results(m).fn);
    end
    fprintf('%s\n\n', repmat('-', 60, 1));
end

function plot_loocv_results(results)
    N = results(1).n_samples;
    fig = figure('Position', [100, 100, 1400, 900], ...
        'Name', sprintf('LOOCV Results - Test Set (N=%d)', N));
    
    % Performance metrics
    subplot(2, 3, 1);
    metrics_data = [[results.accuracy]/100; [results.precision]; ...
                    [results.recall]; [results.f1]]';
    b = bar(metrics_data);
    set(gca, 'XTickLabel', {results.name});
    legend({'Accuracy', 'Precision', 'Recall', 'F1-Score'}, 'Location', 'best');
    ylabel('Score');
    title(sprintf('LOOCV Performance (N=%d)', N));
    grid on;
    ylim([0, 1.05]);
    xtickangle(45);
    
    % Confusion matrix
    [~, best_idx] = max([results.accuracy]);
    subplot(2, 3, 2);
    cm = [results(best_idx).tp, results(best_idx).fp; ...
          results(best_idx).fn, results(best_idx).tn];
    imagesc(cm);
    colormap(flipud(gray));
    colorbar;
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Pred+', 'Pred-'}, ...
        'YTick', 1:2, 'YTickLabel', {'True+', 'True-'});
    title(sprintf('Confusion Matrix: %s', results(best_idx).name));
    for i = 1:2
        for j = 1:2
            text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
                'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    
    % Execution time
    subplot(2, 3, 3);
    times = [results.total_time] / 60;
    bar(times);
    set(gca, 'XTickLabel', {results.name});
    ylabel('Time (minutes)');
    title('LOOCV Execution Time');
    grid on;
    xtickangle(45);
    
    % Accuracy comparison
    subplot(2, 3, 4);
    bar([results.accuracy]);
    set(gca, 'XTickLabel', {results.name});
    ylabel('Accuracy (%)');
    title('LOOCV Accuracy');
    ylim([min([results.accuracy])-5, 100]);
    grid on;
    xtickangle(45);
    
    % Time per sample
    subplot(2, 3, 5);
    time_per_sample = [results.time_per_sample] * 1000;
    bar(time_per_sample);
    set(gca, 'XTickLabel', {results.name});
    ylabel('Time (milliseconds)');
    title('Time per Sample');
    grid on;
    xtickangle(45);
    
    % Precision vs Recall
    subplot(2, 3, 6);
    prec = [results.precision];
    rec = [results.recall];
    scatter(rec, prec, 200, 'filled');
    hold on;
    for i = 1:numel(results)
        text(rec(i) + 0.01, prec(i), results(i).name, 'FontSize', 9);
    end
    xlabel('Recall');
    ylabel('Precision');
    title('Precision vs Recall');
    grid on;
    xlim([0, 1.05]);
    ylim([0, 1.05]);
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.5);
    
    if ~exist('results/figures', 'dir')
        mkdir('results/figures');
    end
    saveas(fig, 'results/figures/loocv_fast_results.png');
    fprintf('Figure saved to results/figures/loocv_fast_results.png\n');
end

function save_loocv_results(results)
    if ~exist('results', 'dir')
        mkdir('results');
    end
    
    save('results/loocv_fast_results.mat', 'results');
    fprintf('Results saved to results/loocv_fast_results.mat\n');
    
    fid = fopen('results/loocv_fast_summary.txt', 'w');
    N = results(1).n_samples;
    fprintf(fid, 'LEAVE-ONE-OUT CROSS-VALIDATION RESULTS (Fast Version - Test Set)\n');
    fprintf(fid, 'N = %d samples\n', N);
    fprintf(fid, '%s\n\n', repmat('=', 80, 1));
    fprintf(fid, '%-20s | %8s | %9s | %9s | %9s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
    fprintf(fid, '%s\n', repmat('-', 80, 1));
    for m = 1:numel(results)
        fprintf(fid, '%-20s | %7.2f%% | %9.4f | %9.4f | %9.4f\n', ...
            results(m).name, results(m).accuracy, results(m).precision, ...
            results(m).recall, results(m).f1);
    end
    fprintf(fid, '%s\n', repmat('-', 80, 1));
    fclose(fid);
    fprintf('Summary saved to results/loocv_fast_summary.txt\n');
end

function best_model_name = get_best_model(results)
    [~, best_idx] = max([results.accuracy]);
    best_model_name = results(best_idx).name;
end

function time_str = format_time(seconds)
    if seconds < 60
        time_str = sprintf('%.1f sec', seconds);
    elseif seconds < 3600
        time_str = sprintf('%.1f min', seconds/60);
    else
        hours = floor(seconds/3600);
        minutes = floor(mod(seconds, 3600)/60);
        time_str = sprintf('%dh %dm', hours, minutes);
    end
end
