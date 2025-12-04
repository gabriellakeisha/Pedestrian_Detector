%% s_train_cnn.m — Train CNNs on RAW and HOG
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

if ~exist('models/cnn','dir'), mkdir('models/cnn'); end

%% Load splits & features  
S = load('splits/splits.mat'); 
trainIdx = S.trainIdx;

R = load('features/raw/features_raw.mat');       
H = load('features/hog/features_hog.mat');        

y_all = double(R.y(:));   

%% =====================================================================
%  CNN 1: on RAW image patches (128 x 64)

fprintf('=== CNN on RAW patches ===\n');

rawHW = [128 64];                 
X_raw_all = double(R.X_raw);

Xtr_raw_flat = X_raw_all(trainIdx,:);   
ytr_raw      = y_all(trainIdx);

Ntrain_raw = numel(ytr_raw);
fprintf('RAW training: %d samples, image size %dx%d\n', Ntrain_raw, rawHW(1), rawHW(2));

% Reshape to 4-D 
Xtr_raw_4d = reshape(Xtr_raw_flat', rawHW(1), rawHW(2), 1, Ntrain_raw);
Xtr_raw_4d = mat2gray(Xtr_raw_4d);      

Ytr_raw_cat = categorical(ytr_raw);    

% Internal validation split (80/20)
cv_raw = cvpartition(Ytr_raw_cat, 'Holdout', 0.2);
idxTR_raw  = training(cv_raw);
idxVAL_raw = test(cv_raw);

Xtrain_raw = Xtr_raw_4d(:,:,:,idxTR_raw);
Ytrain_raw = Ytr_raw_cat(idxTR_raw);

Xval_raw   = Xtr_raw_4d(:,:,:,idxVAL_raw);
Yval_raw   = Ytr_raw_cat(idxVAL_raw);

fprintf('  RAW internal split: %d train / %d val\n', numel(Ytrain_raw), numel(Yval_raw));

% CNN architecture for RAW
layers_raw = [
    imageInputLayer([rawHW(1) rawHW(2) 1], ...
        'Name','input_raw', ...
        'Normalization','zerocenter')

    convolution2dLayer(3, 16, 'Padding','same', 'Name','conv1_raw')
    batchNormalizationLayer('Name','bn1_raw')
    reluLayer('Name','relu1_raw')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool1_raw')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv2_raw')
    batchNormalizationLayer('Name','bn2_raw')
    reluLayer('Name','relu2_raw')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool2_raw')

    convolution2dLayer(3, 64, 'Padding','same', 'Name','conv3_raw')
    batchNormalizationLayer('Name','bn3_raw')
    reluLayer('Name','relu3_raw')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool3_raw')

    fullyConnectedLayer(128, 'Name','fc1_raw')
    reluLayer('Name','relu4_raw')
    dropoutLayer(0.5, 'Name','drop1_raw')

    fullyConnectedLayer(2, 'Name','fc_out_raw')
    softmaxLayer('Name','softmax_raw')
    classificationLayer('Name','output_raw')
];

opts_raw = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xval_raw, Yval_raw}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress'); 

fprintf('\nTraining CNN_RAW...\n');
netCNN_raw = trainNetwork(Xtrain_raw, Ytrain_raw, layers_raw, opts_raw);

save('models/cnn/modelCNN_raw.mat', 'netCNN_raw', 'rawHW', '-v7.3');
fprintf(' Saved models/cnn/modelCNN_raw.mat\n');

%% =====================================================================
%  CNN 2: on HOG block-grid

fprintf('\n=== CNN on HOG block-grid ===\n');

X_hog_all = double(H.X_hog);
y_all_hog = double(H.y(:));   

targetSize = H.targetSize;  
cellSize   = H.cellSize;   
blockSize  = H.blockSize;   
numBins    = H.numBins;      

% Infer HOG grid shape
nBlocksY = targetSize(1)/cellSize(1) - blockSize(1) + 1;
nBlocksX = targetSize(2)/cellSize(2) - blockSize(2) + 1;
channels = blockSize(1)*blockSize(2)*numBins;

Xtr_hog_flat = X_hog_all(trainIdx,:); 
ytr_hog      = y_all_hog(trainIdx);

Ntrain_hog = numel(ytr_hog);
nFeat_hog  = size(Xtr_hog_flat, 2);

assert(nBlocksY*nBlocksX*channels == nFeat_hog, ...
    'HOG dimension mismatch: grid %dx%dx%d vs %d features', ...
    nBlocksY, nBlocksX, channels, nFeat_hog);

fprintf('HOG training: %d samples, grid %d x %d x %d (feat=%d)\n', ...
    Ntrain_hog, nBlocksY, nBlocksX, channels, nFeat_hog);

% Reshape HOG to [H W C N]
Xtr_hog_4d = reshape(Xtr_hog_flat', channels, nBlocksX, nBlocksY, Ntrain_hog);
Xtr_hog_4d = permute(Xtr_hog_4d, [3 2 1 4]);  
Xtr_hog_4d = mat2gray(Xtr_hog_4d);

Ytr_hog_cat = categorical(ytr_hog);

% Internal validation split (80/20)
cv_hog = cvpartition(Ytr_hog_cat, 'Holdout', 0.2);
idxTR_hog  = training(cv_hog);
idxVAL_hog = test(cv_hog);

Xtrain_hog = Xtr_hog_4d(:,:,:,idxTR_hog);
Ytrain_hog = Ytr_hog_cat(idxTR_hog);

Xval_hog   = Xtr_hog_4d(:,:,:,idxVAL_hog);
Yval_hog   = Ytr_hog_cat(idxVAL_hog);

fprintf('  HOG internal split: %d train / %d val\n', numel(Ytrain_hog), numel(Yval_hog));

% CNN architecture for HOG grid
layers_hog = [
    imageInputLayer([nBlocksY nBlocksX channels], ...
        'Name','input_hog', ...
        'Normalization','none')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv1_hog')
    batchNormalizationLayer('Name','bn1_hog')
    reluLayer('Name','relu1_hog')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool1_hog')

    convolution2dLayer(3, 64, 'Padding','same', 'Name','conv2_hog')
    batchNormalizationLayer('Name','bn2_hog')
    reluLayer('Name','relu2_hog')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool2_hog')

    convolution2dLayer(3, 128, 'Padding','same', 'Name','conv3_hog')
    batchNormalizationLayer('Name','bn3_hog')
    reluLayer('Name','relu3_hog')

    fullyConnectedLayer(128, 'Name','fc1_hog')
    reluLayer('Name','relu4_hog')
    dropoutLayer(0.5, 'Name','drop1_hog')

    fullyConnectedLayer(2, 'Name','fc_out_hog')
    softmaxLayer('Name','softmax_hog')
    classificationLayer('Name','output_hog')
];

opts_hog = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xval_hog, Yval_hog}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress'); 

fprintf('\nTraining CNN_HOG...\n');
netCNN_hog = trainNetwork(Xtrain_hog, Ytrain_hog, layers_hog, opts_hog);

save('models/cnn/modelCNN_hog.mat', 'netCNN_hog', ...
     'nBlocksY','nBlocksX','channels', ...
     'targetSize','cellSize','blockSize','numBins', '-v7.3');

fprintf('Saved models/cnn/modelCNN_hog.mat\n');
fprintf('\n All CNN models (RAW & HOG) trained and saved.\n');