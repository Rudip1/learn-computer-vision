% Lab 2 Section 3 Analysis Script
% This script analyzes the projection error between matched SIFT features
% in two images over varying distance ratios. It calculates and plots key
% metrics for each distance ratio, including average and maximum error, 
% number of feature matches, and number of matches with high error.

clear all;
close all;
clc;

% Load images and homography matrix
image1filename = 'imgl01311.jpg';
image2filename = 'imgl01396.jpg';
H12 = [0.923963035362317 0.105144470821343 -41.6461418047369; 
      -0.105144470821343 0.923963035362317 -224.712186142836; 
       0 0 1];

% Read images
I1 = imread(image1filename);
I2 = imread(image2filename);

% Define range for distance ratios to be tested
distRatios = 0.3:0.05:0.9;
distanceThreshold = 50; % Define threshold for high-error features (50 pixels)

% Initialize stacks to store analysis data for each distance ratio
avgErrorStack = [];               % To store average projection error
maxErrorStack = [];               % To store maximum projection error
numFeaturesStack = [];            % To store the number of matched features
numFeaturesAboveThresholdStack = []; % To store count of high-error matches

% Loop through each distance ratio and perform feature matching and error analysis
for distRatio = distRatios
    % Perform SIFT feature matching with the current distance ratio
    drawMatches = false;  % Set to true if you want to visualize matches (optional)
    [CL1uv, CL2uv, ~, ~, ~, ~] = matchsiftmodif(image1filename, image2filename, distRatio, drawMatches);

    % If matching points are found, calculate the projection error vector
    if ~isempty(CL1uv) && ~isempty(CL2uv)
        % Calculate projection error between matched points using homography
        errorVec = projectionerrorvec(H12, CL1uv, CL2uv);
        
        % Store computed metrics for the current distance ratio
        avgErrorStack = [avgErrorStack; mean(errorVec)];            % Average error
        maxErrorStack = [maxErrorStack; max(errorVec)];             % Maximum error
        numFeaturesStack = [numFeaturesStack; size(CL1uv, 1)];      % Number of matches
        numFeaturesAboveThresholdStack = [numFeaturesAboveThresholdStack; sum(errorVec > distanceThreshold)];
                                                                  % Matches with error > threshold
    else
        % If no matches found, store NaN or zero for consistency in stack lengths
        avgErrorStack = [avgErrorStack; NaN];
        maxErrorStack = [maxErrorStack; NaN];
        numFeaturesStack = [numFeaturesStack; 0];
        numFeaturesAboveThresholdStack = [numFeaturesAboveThresholdStack; 0];
    end
end

% Plot results: Average and maximum errors, number of matches, and high-error count
figure;

% Plot average projection error as a function of distance ratio
subplot(2, 2, 1);
plot(distRatios, avgErrorStack, '-o');
xlabel('Distance Ratio');
ylabel('Average Error (pixels)');
title('Average Projection Error');

% Plot maximum projection error as a function of distance ratio
subplot(2, 2, 2);
plot(distRatios, maxErrorStack, '-o');
xlabel('Distance Ratio');
ylabel('Maximum Error (pixels)');
title('Maximum Projection Error');

% Plot number of matched features for each distance ratio
subplot(2, 2, 3);
plot(distRatios, numFeaturesStack, '-o');
xlabel('Distance Ratio');
ylabel('Number of Matches');
title('Number of Associated Features');

% Plot count of features with error above the defined threshold (50 pixels)
subplot(2, 2, 4);
plot(distRatios, numFeaturesAboveThresholdStack, '-o');
xlabel('Distance Ratio');
ylabel('Matches with Error > 50 pixels');
title('High Error Feature Count');

% Display consolidated plots with both linear and logarithmic scaling for deeper insights
figure;

% Linear scale plot for all metrics
subplot(1, 2, 1);
plot(distRatios, avgErrorStack, distRatios, maxErrorStack, distRatios, numFeaturesStack, distRatios, numFeaturesAboveThresholdStack);
xlabel('Distance Ratio');
ylabel('Metric Value (linear scale)');
legend('Average Error', 'Max Error', 'Num Matches', 'Error > 50px');
title('Error and Matches (Linear Scale)');

% Log scale plot for metrics to observe variations over small and large ranges
subplot(1, 2, 2);
semilogy(distRatios, avgErrorStack, distRatios, maxErrorStack, distRatios, numFeaturesStack, distRatios, numFeaturesAboveThresholdStack);
xlabel('Distance Ratio');
ylabel('Metric Value (log scale)');
legend('Average Error', 'Max Error', 'Num Matches', 'Error > 50px');
title('Error and Matches (Log Scale)');

% End of script
