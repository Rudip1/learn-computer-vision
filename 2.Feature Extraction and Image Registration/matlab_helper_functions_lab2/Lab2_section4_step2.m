% Test for helper script for lab 2 of MVG section 4

clear all;
close all;
clc;

% Read the first pair of synthetic images and the given coordinates of
% matched points. These matched points are error free thus ideal for
% testing
image1filename = 'DataSet01/00.png';
image2filename = 'DataSet01/01.png';
load('DataSet01/Features.mat');
I1 = imread(image1filename);
CL1uv = Features(1).xy;
image_files = {'DataSet01/01.png' , 'DataSet01/02.png' , 'DataSet01/03.png'};
Models ={'Translation','Similarity','Affine','Projective'};
% Create a figure
figure;
% Loop to create subplots
avg_mean_error = zeros(length(image_files),length(Models)); % 3- for image 4 -for model types


for i = 1:3
    for j = 1:length(Models)
        CL2uv = Features(i+1).xy;
        I2 = imread(image_files{i});
        H12 = computeHomography(CL1uv, CL2uv, Models{j});
        errorVec = projectionerrorvec(H12, CL1uv, CL2uv);
        mean_error = mean(errorVec);
        avg_mean_error(i, j) = mean_error;

        % This is a function that warps image I2 into the frame of image I1
        overlay = showwarpedimages(I1, I2, H12);

        % Calculate the subplot index
        subplotIndex = (i - 1) * length(Models) + j;

        % Create the subplot
        subplot(3, 4, subplotIndex);

        % Display the warped image with a title
        imshow(overlay);
        title(Models{j});

        % Add labels for image name and model name
        xlabel(['Image: ' image_files{i}], 'Interpreter', 'none', 'FontSize', 8);
        ylabel(['Model: ' Models{j}], 'Interpreter', 'none', 'FontSize', 8);
    end
end

% Adjust the overall layout
sgtitle('Warped Images with Model Overlays');

% Set a larger figure size for better visibility
set(gcf, 'Position', [100, 100, 1200, 800]);


% Loop over model types
% Plot the average mean error for each image

figure;
% Loop over images
for i = 1:3
    % Create the subplot for each image
    subplot(3, 1, i);
    
    % Bar plot for average mean error for all model types with reduced bar width
    bar(avg_mean_error(i, :), 'BarWidth', 0.1); % You can adjust the BarWidth value as needed
    
    % Set axis labels and title
    xlabel('Model Type', 'FontWeight', 'normal', 'FontSize', 10);
    ylabel('Average Mean Error', 'FontWeight', 'normal', 'FontSize', 10);
    title(['Im ' image_files{i}], 'FontWeight', 'normal', 'FontSize', 12);
    
    % Customize the x-axis ticks and labels for better visibility
    xticks(1:length(Models));
    xticklabels(Models);
    
    % Rotate x-axis labels for better readability
    % xtickangle(45);
    
    % Adjust the width of the histogram labels
    set(gca, 'TickLabelInterpreter', 'none');
end

disp(avg_mean_error)

