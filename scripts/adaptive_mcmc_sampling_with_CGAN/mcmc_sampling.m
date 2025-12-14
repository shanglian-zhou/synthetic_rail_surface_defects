function mcmc_sampling()
% ========================================================================
% Adaptive MCMC Sampling for CGAN-based Conditional Image Refinement
% ========================================================================
% Toolboxes required:
%   - Deep Learning Toolbox
%   - Image Processing Toolbox
%
% GPU:
%   - If a supported GPU is available, this script runs inference on GPU.
%     (Generator + Discriminators + dlarray inputs are moved to GPU.)
%
% Notes:
%   - Uses relative paths only. Users should place their own data/checkpoints
%     under the project directory (or update CONFIG paths below).
%   - No parallel processing is used.
% ========================================================================

clc; clear; close all;

%% usage
% In MATLAB command window, run:
% >> mcmc_sampling

% ------------------------------------------------------------------------
% Timestamp for output folder (YYYYMMDD_HHMMSS)
% ------------------------------------------------------------------------
timestamp = datestr(datetime("now"), "yyyymmdd_HHMMSS");

%% ------------------------------
% CONFIG (edit as needed)
% ------------------------------

root_dir = '/Volumes/Sandisk/data/RSDDs/Synthetic'; % replace this to your own data path
addpath(root_dir)

CONFIG.projectRoot   = root_dir;

% Example expected structure (customize as needed):
%   ./custom_data_20250121/custom_labels_20250121/cropped/...
%   ./checkpoint_20250121/epoch_500.mat
CONFIG.labelDirs = {
    fullfile(CONFIG.projectRoot, "custom_data_20250121/custom_labels_20250121/cropped/")
};

% load model checkpoint; training your model using the script from: https://www.mathworks.com/help/deeplearning/ug/train-conditional-generative-adversarial-network.html
CONFIG.checkpointFile = fullfile(CONFIG.projectRoot, "checkpoint_20250121", "epoch_500.mat");

CONFIG.saveFolder = fullfile( ...
    CONFIG.projectRoot, ...
    "outputs", ...
    "mcmc_results_" + timestamp );

CONFIG.saveImages = fullfile(CONFIG.saveFolder, "images");
CONFIG.saveLabels = fullfile(CONFIG.saveFolder, "labels");
CONFIG.saveMeta   = fullfile(CONFIG.saveFolder, "metadata");

CONFIG.imageSize = [160 160];          % [H W]
CONFIG.startIdx  = 1;
CONFIG.endIdx    = 10;               % number of labels to process (after shuffle)

% MCMC parameters
CONFIG.burnIn            = 10;
CONFIG.refineIterations  = 20;
CONFIG.totalIterations   = CONFIG.burnIn + CONFIG.refineIterations;

% Label encoding (RGB PNG):
%   defects    = red   [255 0 0]
%   background = green [0 255 0]
CONFIG.defectRGB    = uint8([255 0 0]);
CONFIG.backgroundRGB= uint8([0 255 0]);

% Normalization bounds for discriminator score -> [0,1]
% Adjustr these bounds based on your dataset distribution
CONFIG.b_max = 0.2767;
CONFIG.b_min = 0.2207;

% MH acceptance function parameters
CONFIG.beta1      = 3;
CONFIG.x_skewness = 0.685;
CONFIG.k_gamma    = 0.5;
CONFIG.lambda1    = 0.5;
CONFIG.iou_target = 0.6;
CONFIG.k_reg      = 0.3;

% Adaptive perturbation thresholds (pixel counts) and initial strength
% These thresholds control structuring-element size selection
CONFIG.smallAreaThresh = 200;
CONFIG.medAreaThresh   = 500;
CONFIG.perturbationStrength0 = 2;
CONFIG.adjust_period = 2;  % adjust every N iterations

rng(666);

%% ------------------------------
% Create output folders
% ------------------------------
if ~exist(CONFIG.saveImages, "dir"), mkdir(CONFIG.saveImages); end
if ~exist(CONFIG.saveLabels, "dir"), mkdir(CONFIG.saveLabels); end
if ~exist(CONFIG.saveMeta,   "dir"), mkdir(CONFIG.saveMeta);   end

%% ------------------------------
% Collect label files and shuffle
% ------------------------------
labelFiles = collect_png_files(CONFIG.labelDirs);
if isempty(labelFiles)
    error("No PNG label files found under configured labelDirs.");
end

perm = randperm(numel(labelFiles));
labelFiles = labelFiles(perm);

CONFIG.endIdx = min(CONFIG.endIdx, numel(labelFiles));
labelFiles = labelFiles(CONFIG.startIdx:CONFIG.endIdx);

fprintf("Total labels to process: %d\n", numel(labelFiles));

%% ------------------------------
% Load networks (Generator + Discriminators)
% ------------------------------
S = load(CONFIG.checkpointFile, ...
    "dlnetGenerator", "dlnetDiscriminatorScale1", "dlnetDiscriminatorScale2");

dlnetG  = S.dlnetGenerator;
dlnetD1 = S.dlnetDiscriminatorScale1;
dlnetD2 = S.dlnetDiscriminatorScale2;

%% ------------------------------
% GPU setup
% ------------------------------
useGPU = canUseGPU();
if useGPU
    dlnetG  = dlupdate(@gpuArray, dlnetG);
    dlnetD1 = dlupdate(@gpuArray, dlnetD1);
    dlnetD2 = dlupdate(@gpuArray, dlnetD2);
end

%% ------------------------------
% Helpers
% ------------------------------
b_norm = @(x) min(1, max(0, (x - CONFIG.b_min) / (CONFIG.b_max - CONFIG.b_min)));

gamma1 = ((1 - CONFIG.k_gamma) * CONFIG.x_skewness + 1) / (CONFIG.x_skewness + 1);

acceptance_prob = @(proposed_norm, current_norm, iou) ...
    min(max(0, exp(CONFIG.beta1 * (proposed_norm - current_norm)) * gamma1 ...
    - CONFIG.lambda1 * abs(iou - CONFIG.iou_target) ...
    - CONFIG.k_reg), 1);

%% ------------------------------
% Main loop (no parallel)
% ------------------------------
for idx = 1:numel(labelFiles)

    labelPath = labelFiles{idx};
    [~, baseName, ext] = fileparts(labelPath);

    outImgName = [baseName ext];
    outMetaName = baseName + ".mat";

    outImgPath  = fullfile(CONFIG.saveImages, outImgName);
    outLabPath  = fullfile(CONFIG.saveLabels, outImgName);
    outMetaPath = fullfile(CONFIG.saveMeta,   outMetaName);

    if exist(outImgPath, "file")
        fprintf("[Skip] exists: %s\n", outImgName);
        continue;
    end

    perturbationStrength = CONFIG.perturbationStrength0;

    % --- Load label and generate initial synthetic image
    currentLabel = read_label_rgb_to_onehot(labelPath, CONFIG.imageSize, ...
        CONFIG.defectRGB, CONFIG.backgroundRGB, useGPU);

    currentImage = generate_from_label(currentLabel, dlnetG, useGPU);
    currentScore = evaluate_image_discriminator(currentImage, currentLabel, dlnetD1, dlnetD2, useGPU);

    bestLabel = currentLabel;
    bestImage = currentImage;
    bestScore = currentScore;

    % --- Pre-allocate metadata
    T = CONFIG.totalIterations;

    record_acceptance = zeros(T,4);   % [alpha, rand_u, acceptFlag, acceptRate]
    record_perturb    = zeros(T,2);   % [strength, area_change]
    record_score      = zeros(T,5);   % [curScore, propScore, diff, bestSoFar, absPctDiff]
    record_iou        = zeros(T,1);
    record_time       = zeros(T,1);

    % --- MCMC loop
    tStart = tic;
    for t = 1:T

        tIterStart = tic;

        proposedLabel = adaptive_perturbation( ...
            gather_if_needed(currentLabel, useGPU), ... % adaptive_perturbation usually expects CPU logical/numeric
            CONFIG.smallAreaThresh, CONFIG.medAreaThresh, perturbationStrength);

        % move proposedLabel back to GPU if needed
        proposedLabel = to_device(single(proposedLabel), useGPU);
        proposedLabel = dlarray(proposedLabel, "SSCB");

        proposedImage = generate_from_label(proposedLabel, dlnetG, useGPU);
        proposedScore = evaluate_image_discriminator(proposedImage, proposedLabel, dlnetD1, dlnetD2, useGPU);

        % area change (absolute pixel difference on defect channel)
        area_dif = proposedLabel(:,:,2,:) - currentLabel(:,:,2,:);
        area_change = sum(abs(gather(extractdata(area_dif))), "all");

        % IoU computed on DEFECT channel (channel 2)
        propMask = gather(extractdata(proposedLabel(:,:,2,:) > 0.5));
        currMask = gather(extractdata(currentLabel(:,:,2,:)  > 0.5));
        iou = compute_iou(propMask, currMask);

        % acceptance
        p_norm = b_norm(proposedScore);
        c_norm = b_norm(currentScore);

        alpha = acceptance_prob(p_norm, c_norm, iou);
        u = rand();
        acceptFlag = (u < alpha);

        if acceptFlag
            currentLabel = proposedLabel;
            currentImage = proposedImage;
            currentScore = proposedScore;
        end

        % Track best sample after burn-in using discriminator score
        if t > CONFIG.burnIn && proposedScore > bestScore
            bestScore = proposedScore;
            bestLabel = proposedLabel;
            bestImage = proposedImage;
        end

        % record
        record_acceptance(t,:) = [alpha, u, acceptFlag, mean(record_acceptance(1:t,3))];
        record_perturb(t,:)    = [perturbationStrength, area_change];
        record_score(t,:)      = [currentScore, proposedScore, proposedScore - currentScore, bestScore, ...
                                  abs((proposedScore - currentScore) / max(eps, currentScore))];
        record_iou(t)          = iou;
        record_time(t)         = toc(tIterStart);

        % ------------------------------------------------------------
        % Adaptive Perturbation Strength Control (Post Burn-in)
        % ------------------------------------------------------------
        
        if t > CONFIG.burnIn && mod(t - CONFIG.burnIn, CONFIG.adjust_period) == 0
        
            % Current acceptance rate
            acceptanceRate = mean(record_acceptance(1:t,3));
        
            % Dynamically adjust perturbation strength
            if acceptanceRate < 0.3
                perturbationStrength = min( ...
                    2.0, ...
                    perturbationStrength * (1 + (2/3) * (0.3 - acceptanceRate)) );
            elseif acceptanceRate > 0.5
                perturbationStrength = max( ...
                    1.0, ...
                    perturbationStrength * (1 - (2/5) * (acceptanceRate - 0.5)) );
            end
        
            % Optional debug print (disabled by default)
            % fprintf('[Iter %d] AcceptRate=%.2f | PerturbStrength=%.2f\n', ...
            %     t, acceptanceRate, perturbationStrength);
        
        end

    end
    totalTime = toc(tStart);

    % --- Save best image + label + metadata
    rgbBestLabel = onehot_to_rgb_label(bestLabel, useGPU);

    imwrite(gather(extractdata(bestImage)), outImgPath);
    imwrite(rgbBestLabel, outLabPath);

    tb_metadata = table( ...
    record_acceptance(:,1), record_acceptance(:,2), record_acceptance(:,3), record_acceptance(:,4), ...
    record_perturb(:,1), record_perturb(:,2), ...
    record_score(:,1), record_score(:,2), record_score(:,3), record_score(:,4), record_score(:,5), ...
    record_iou, record_time, ...
    'VariableNames', { ...
        'alpha','rand_u','accept_flag','accept_rate', ...
        'perturb_strength','area_change', ...
        'current_score','proposed_score','score_diff','best_score','abs_pct_diff', ...
        'iou_score','iter_time_sec'} );

    save(outMetaPath, "tb_metadata", "labelPath", "totalTime");

    fprintf("[%d/%d] saved: %s (%.2fs)\n", idx, numel(labelFiles), outImgName, totalTime);
end

fprintf("Done.\n");

end

%% ========================================================================
% Utilities
% ========================================================================

function files = collect_png_files(dirList)
files = {};
for i = 1:numel(dirList)
    if ~exist(dirList{i}, "dir"), continue; end
    D = dir(fullfile(dirList{i}, "*.png"));
    for k = 1:numel(D)
        files{end+1,1} = fullfile(D(k).folder, D(k).name); %#ok<AGROW>
    end
end
end

function x = to_device(x, useGPU)
if useGPU
    x = gpuArray(x);
end
end

function x = gather_if_needed(x, useGPU)
% adaptive_perturbation uses Image Processing Toolbox ops which typically
% run on CPU for logical masks; keep it CPU-safe.
x = gather(extractdata(x));
if useGPU
    % return CPU array (do NOT wrap gpuArray here)
end
end

function onehot = read_label_rgb_to_onehot(labelPath, imageSize, defectRGB, backgroundRGB, useGPU)
rgb = imread(labelPath);
rgb = imresize(rgb, imageSize, "nearest");

% Create masks
isDefect = (rgb(:,:,1)==defectRGB(1)) & (rgb(:,:,2)==defectRGB(2)) & (rgb(:,:,3)==defectRGB(3));
isBack   = (rgb(:,:,1)==backgroundRGB(1)) & (rgb(:,:,2)==backgroundRGB(2)) & (rgb(:,:,3)==backgroundRGB(3));

% If any pixels are neither red nor green, treat as background (safe default)
isBack = isBack | ~(isDefect | isBack);

H = size(rgb,1); W = size(rgb,2);
onehotCPU = zeros(H,W,2,"single");
onehotCPU(:,:,1) = single(isBack);    % channel 1 = background
onehotCPU(:,:,2) = single(isDefect);  % channel 2 = defect

onehot = dlarray(to_device(onehotCPU, useGPU), "SSCB");
end

function img = generate_from_label(segOneHot, generatorNet, useGPU)
segOneHot = dlarray(segOneHot, "SSCB");
img = predict(generatorNet, segOneHot);
img = rescale(img); % keep [0,1]
img = dlarray(img, "SSCB");
if ~useGPU
    % keep as dlarray on CPU
end
end

function score = evaluate_image_discriminator(image, segMap, D1, D2, useGPU)
dlSeg = dlarray(single(segMap), "SSCB");
dlImg = dlarray(single(image),  "SSCB");

if useGPU
    dlSeg = gpuArray(dlSeg);
    dlImg = gpuArray(dlImg);
end

inp1 = cat(3, dlSeg, dlImg);

seg2 = dlresize(dlSeg, Scale=0.5, Method="nearest");
img2 = dlresize(dlImg, Scale=0.5, Method="linear");
inp2 = cat(3, seg2, img2);

featureNames = ["act_top","act_mid_1","act_mid_2","act_tail","conv2d_final"];

pred1 = cell(size(featureNames));
[pred1{:}] = forward(D1, inp1, Outputs=featureNames);
pred2 = cell(size(featureNames));
[pred2{:}] = forward(D2, inp2, Outputs=featureNames);

PredScale1 = pred1{end};
PredScale2 = pred2{end};

DLossScale1 = (PredScale1).^2;
DLossScale2 = (PredScale2).^2;

DLoss = 0.5 * (mean(DLossScale1,[1 2 3]) + mean(DLossScale2,[1 2 3]));
score = double(gather(extractdata(squeeze(DLoss))));
end

function iou = compute_iou(maskA, maskB)
intersection = sum(maskA & maskB, "all");
union = sum(maskA | maskB, "all");
iou = intersection / max(1, union);
end

function rgb = onehot_to_rgb_label(onehotLabel, useGPU)
L = gather(extractdata(onehotLabel));
defect = L(:,:,2) > 0.5;
back   = ~defect;

rgb = zeros(size(L,1), size(L,2), 3, "uint8");
rgb(:,:,2) = uint8(back)   * 255;  % green background
rgb(:,:,1) = uint8(defect) * 255;  % red defect
end


% ========================================================================
% Adaptive Perturbation Function for MCMC-based Label Refinement
% ------------------------------------------------------------------------
% This function is implemented in MATLAB and is part of the adaptive MCMC
% sampling framework proposed in:
%
%   "Markov Chain Monte Carlo-driven Exploration and Refinement for
%    CGAN-based Synthetic Image Generation for Rail Surface Defect
%    Segmentation"
%
% The function applies adaptive morphological perturbations to segmentation
% masks to regulate exploration in the image–label space during MCMC
% sampling.
% ========================================================================

function perturbedOneHot = adaptive_perturbation( ...
    segMap, small_thresh, med_thresh, perturbationStrength)
% ========================================================================
% Adaptive Perturbation Function (Single-step Morphology)
% ========================================================================
% Applies a single adaptive morphological perturbation to regulate
% exploration in the image–label space during MCMC sampling.
%
% Inputs:
%   - segMap: One-hot encoded segmentation mask (H × W × 2)
%   - small_thresh: Area threshold for small defects
%   - med_thresh: Area threshold for medium defects
%   - perturbationStrength: Controls structuring-element size
%
% Output:
%   - perturbedOneHot: Perturbed one-hot encoded segmentation mask
% ========================================================================

    % Extract defect channel
    defectMask = segMap(:,:,2) > 0;

    % Compute defect area
    defectArea = sum(defectMask(:));

    % ------------------------------------------------------------
    % Adaptive structuring element size
    % ------------------------------------------------------------
    baseSize = max(2, round(perturbationStrength));  % ensure size ≥ 2

    if defectArea < small_thresh
        seSize = randi([max(1, baseSize - 1), baseSize]);
    elseif defectArea < med_thresh
        seSize = randi([baseSize, baseSize + 1]);
    else
        seSize = randi([baseSize + 1, baseSize + 2]);
    end

    se = strel('disk', seSize);

    % ------------------------------------------------------------
    % Single-step dilation or erosion (no iteration)
    % ------------------------------------------------------------
    if rand < 0.5
        % Dilation
        perturbedMask = imdilate(defectMask, se);
    else
        % Erosion
        perturbedMask = imerode(defectMask, se);

        % Ensure defect is not fully removed
        if ~any(perturbedMask(:))
            perturbedMask = imdilate(defectMask, se);
        end
    end

    % ------------------------------------------------------------
    % Add adaptive boundary noise
    % ------------------------------------------------------------
    baseNoise = 0.02 + 0.03 * rand();      % [0.02, 0.05]
    noiseProbability = min(0.05, baseNoise * perturbationStrength);

    perturbedMask = add_boundary_noise(perturbedMask, noiseProbability);

    % ------------------------------------------------------------
    % Convert back to one-hot encoding
    % ------------------------------------------------------------
    perturbedOneHot = zeros(size(segMap), 'like', segMap);
    perturbedOneHot(:,:,1) = ~perturbedMask;  % background
    perturbedOneHot(:,:,2) =  perturbedMask;  % defect

end

% ========================================================================
% Helper Function: Boundary Noise Injection
% ========================================================================
function noisyMask = add_boundary_noise(mask, probability)
% Adds localized boundary noise to a binary defect mask.
%
% Inputs:
%   - mask: Binary defect mask (H × W)
%   - probability: Probability of boundary perturbation
%
% Output:
%   - noisyMask: Binary mask with boundary noise applied

    boundary = edge(mask, 'sobel');
    noise = rand(size(mask)) < probability;

    noisyMask = mask | (boundary & noise);
end
