% b_halfhalf_splits.m — create 50/50 training/testing splits 
clearvars; close all; clc; rng(1234,'twister');

% Get project root based on this file location
thisFileFull = mfilename('fullpath');
thisFileDir  = fileparts(thisFileFull);
projectRoot  = fileparts(thisFileDir);
cd(projectRoot);

posDir = fullfile(projectRoot,'images_preprocessed','pos');
negDir = fullfile(projectRoot,'images_preprocessed','neg');

% Collect positive files
P = [dir(fullfile(posDir,'*.png'));  dir(fullfile(posDir,'*.jpg'));  dir(fullfile(posDir,'*.jpeg')); ...
     dir(fullfile(posDir,'*.PNG'));  dir(fullfile(posDir,'*.JPG'));  dir(fullfile(posDir,'*.JPEG'))];

% Collect negative files
N = [dir(fullfile(negDir,'*.png'));  dir(fullfile(negDir,'*.jpg'));  dir(fullfile(negDir,'*.jpeg')); ...
     dir(fullfile(negDir,'*.PNG'));  dir(fullfile(negDir,'*.JPG'));  dir(fullfile(negDir,'*.JPEG'))];

Npos = numel(P);
Nneg = numel(N);

assert(Npos > 0 && Nneg > 0, ...
    sprintf('Found Npos=%d, Nneg=%d. Check images/pos & images/neg.', Npos, Nneg));

total = Npos + Nneg;

posIdxGlobal = (1:Npos).'; 
negIdxGlobal = (Npos+1 : Npos+Nneg).';

Ntrain_pos = round(0.5 * Npos);
Ntrain_neg = round(0.5 * Nneg);

posOrder = posIdxGlobal(randperm(Npos));
negOrder = negIdxGlobal(randperm(Nneg));

trainIdx_pos = posOrder(1:Ntrain_pos);
testIdx_pos  = posOrder(Ntrain_pos+1:end);

trainIdx_neg = negOrder(1:Ntrain_neg);
testIdx_neg  = negOrder(Ntrain_neg+1:end);

% Merge into single training/testing index sets
trainIdx = [trainIdx_pos; trainIdx_neg];
testIdx  = [testIdx_pos;  testIdx_neg];

% Shuffle overall, just to avoid any ordering bias
trainIdx = trainIdx(randperm(numel(trainIdx)));
testIdx  = testIdx(randperm(numel(testIdx)));

% Create splits directory if needed
splitsDir = fullfile(projectRoot,'splits');
if ~exist(splitsDir,'dir')
    mkdir(splitsDir);
end

% Save with 50/50 filename
save(fullfile(splitsDir,'splits_50_50.mat'), ...
    'trainIdx','testIdx','Npos','Nneg','trainIdx_pos','testIdx_pos','trainIdx_neg','testIdx_neg');

% Also save as splits.mat for compatibility with other scripts
save(fullfile(splitsDir,'splits.mat'), ...
    'trainIdx','testIdx','Npos','Nneg','trainIdx_pos','testIdx_pos','trainIdx_neg','testIdx_neg');

fprintf('\n50/50 SPLIT CREATED\n');
fprintf('Total samples: %d (Npos=%d, Nneg=%d)\n', total, Npos, Nneg);
fprintf('Training: %d samples (50%% approx)\n', numel(trainIdx));
fprintf('Testing:  %d samples (50%% approx)\n', numel(testIdx));
fprintf('Pos train/test: %d / %d\n', numel(trainIdx_pos), numel(testIdx_pos));
fprintf('Neg train/test: %d / %d\n', numel(trainIdx_neg), numel(testIdx_neg));
