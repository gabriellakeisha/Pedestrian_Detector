%% data_preprocessing.m - Image Preprocessing for Pedestrian Dataset
clearvars; close all; clc; rng(1234,'twister');
addpath(genpath('src'));

fprintf('Start data preprocessing setup\n');

inputDir = 'images/';
outputDir = 'images_preprocessed/';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

imgFiles = dir(fullfile(inputDir, '**/*.jpg'));
fprintf('Found %d images\n', numel(imgFiles));

targetSize = [128 64];  % INRIA standard window size

for i = 1:numel(imgFiles)

    inPath = fullfile(imgFiles(i).folder, imgFiles(i).name);

    % extract only the final folder: "neg" or "pos"
    [~, parentFolderName] = fileparts(imgFiles(i).folder);

    % create output folder for pos/neg
    outSubDir = fullfile(outputDir, parentFolderName);
    if ~exist(outSubDir, 'dir')
        mkdir(outSubDir);
    end

    % output full path
    outPath = fullfile(outSubDir, imgFiles(i).name);

    % preprocessing list
    I = imread(inPath);

    if size(I,3) == 3
        I = rgb2gray(I);
    end

    I = imresize(I, targetSize);
    I = adapthisteq(I);
    I = imgaussfilt(I, 0.5);

    % save processed images
    imwrite(I, outPath);
end

fprintf('Preprocessing complete! Saved to %s\n', outputDir);
