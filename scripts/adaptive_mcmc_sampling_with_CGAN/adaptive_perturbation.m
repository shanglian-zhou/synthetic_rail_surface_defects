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

% Applies adaptive perturbation for MCMC sampling with improved realism.
%
% Inputs:
%   - segMap: One-hot encoded segmentation mask (H × W × C)
%   - small_thresh: Threshold to classify defects as small
%   - med_thresh: Threshold to classify defects as medium
%   - perturbationStrength: Controls perturbation intensity
%
% Output:
%   - perturbedOneHot: Perturbed one-hot encoded segmentation mask (H × W × C)

    % ------------------------------------------------------------
    % Extract defect channel from one-hot encoding
    % ------------------------------------------------------------
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
    % Adaptive number of morphological iterations
    % NOTE: MATLAB imdilate / imerode do NOT support iteration
    % arguments — iterations must be explicitly looped.
    % ------------------------------------------------------------
    numIterations = max(1, round(perturbationStrength * randi([1, 2])));
    if defectArea > med_thresh
        numIterations = max(1, round(perturbationStrength * randi([2, 3])));
    end

    perturbedMask = defectMask;

    % ------------------------------------------------------------
    % Randomly choose dilation or erosion
    % ------------------------------------------------------------
    if rand < 0.5
        % --- Dilation ---
        for k = 1:numIterations
            perturbedMask = imdilate(perturbedMask, se);
        end
    else
        % --- Erosion ---
        for k = 1:numIterations
            perturbedMask = imerode(perturbedMask, se);
        end

        % Ensure defect is not fully removed
        if sum(perturbedMask(:)) == 0
            perturbedMask = defectMask;
            for k = 1:numIterations
                perturbedMask = imdilate(perturbedMask, se);
            end
        end
    end

    % ------------------------------------------------------------
    % Add adaptive boundary noise
    % ------------------------------------------------------------
    baseNoise = 0.02 + 0.03 * rand();   % [0.02, 0.05]
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
