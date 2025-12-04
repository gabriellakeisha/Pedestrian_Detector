# VAML Coursework Group 29 #

**Members involved: Gabriella Keisha Andini(40392749), Su Thinzar Thaw(40392455), Yu-Zhi Wong(40374472)**

## Overview

The project implements a sliding window and part-based pedestrian detector. The detector rescales each input image on a set of predefined `winScales` and scans it with a fixed [128 x 64] detection window. 

For every window, the corresponding feature descriptor (HOG, LBP, RAW, EdgeHOG, or Fusion HOG+LBP) is extracted and passed to the selected classifier (SVM, Neural Network, Random Forest). If the classifier output exceeds the `scoreThresh`, the window is recorded as a potential detection. 

## Source Files 

This project uses the following directory and datasets: 
- `images/pos/` - original positive training samples.
- `images/neg/` - original negative training samples. 
- `images_preprocessed/pos/` - positive samples after preprocessing. 
- `images_preprocessed/neg/` - negative samples after preprocessing. 
- `pedestrian/` - full-sized test images used during the detection stage.
- `test_dataset.txt` - ground-truth bounding box annotations for each test image.

Preprocessed sets are the output of `b0_data_preprocessed.m`. These are already included, so you do not need to re-run the preprocessing script unless you want to regenerate them.

## Classifier and Feature Extractor Summary 

This project includes a full set of 9 classifiers implemented for the training and evaluation stage: 
- Nearest Neighbour 
- K-Nearest Neighbour 
- SVM-RBF 
- SVM-Linear 
- Random Forest 
- Bagging 
- Neural Network (NN)
- Convolutional Neural Network (CNN)
- Deep Neural Network (DNN)

In addition, 7 feature extraction methods were implemented: 
- RAW 
- HOG
- PCA 
- LDA
- LBP
- Edge HOG
- Fusion (HOG+LBP)

Although all classifiers and feature desciptors were trained and evaluated, only a selected subset is used in the final detector scripts.

## How to Run 

All scripts are located inside the `src/` directory.
Follow the pipeline in order to reproduce the full pedestrian-detection system.

1. `a_setup.m`
2. `b0_data_preprocessing.m`
3. run `b_make_splits.m` for 70/30 split or `b_halfhalf_splits.m` for 50/50 split
    - `b_cross_validation.m` the splitting script supports cross validation, and is currently set to k=5. You can change this by modifying the parameter inside the script. 
4. run feature extractors 
    - `c_features_raw.m`
    - `d_features_hog.m`
    - `d1_features_pca_split.m`
    - `d2_features_lda.m`
    - `d3_features_lbp.m`
    - `d4_features_edgehog_canny.m`
    - `d5_features_hog_lbp.m`
    all feature extractors generate `.mat` files under `features/<feature_name>/`
5. each model follows the naming pattern:
    - `<letter>_train_<classifier>.m`
    - `<letter>_eval_<classifier>.m`, evaluation scripts produce performance metrics (accuracy, precision, recall, F1) and confusion matrices
6. run sliding window detector `sliding_window_detector/sliding_window_detector.m`
    - configure inside the script: 
        - model_type: `SVM`, `RF`, `NEURAL`
        - feature_type: `RAW`, `HOG`, `LBP`, `FUSION`
7. run part-based detector `partbaseddetector/`. each model follows the naming patter: 
    - training `train_part_<classifier>.m`
    - evaluation `eval_part_<classifier>.m`
    - detect `detect_part_<classifier>.m`
    - detection by tracking (only available for Neural Network classifier) `detect_nn_part_track.m`
    - available classifiers include `SVM` and `Neural`.  

## Recommended Sliding Window Detector Configuration

These parameters are pre-tested and provide the most stable performance for running the detector out-of-the-box. 

```bash
winScales = [1.0 1.3 1.6 1.9 2.2 2.5]
scoreThreesh = 0.40
overlapThresh = 0.30
iouThresh = 0.20
```

These values are used in our final detector scripts and are suggested as default values for anyone running the system. 

### Aditional Notes 

All scripts have already been executed, and all resulting `.mat` feature files have been included in the GitLab respository. 

This means you do NOT need to run the feature extractor scripts again in order to run our, training, testing or detector code. 

Full justification and experimental analysis for these parameters choices can be found in the final report. 
