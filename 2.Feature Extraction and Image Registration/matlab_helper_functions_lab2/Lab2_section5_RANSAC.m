% Test for helper script for lab 2 of MVG section 4 - under water image
clear all;
close all;
clc;

% Read the first pair of synthetic images and the given coordinates of
% matched points. These matched points are error free thus ideal for
% testing
image1filename = 'MiamiSet00/imgl01311.jpg';
image2filename = 'imgl01366.jpg';


% This is a function that you have to populate. 
% Change the name to H12 = computeHomography(CL1uv,CL2uv, Model) after you added your own code inside
% Model = 'Projective';
% Model = 'Affine'
image_files = {'MiamiSet00/imgl01311.jpg','MiamiSet00/imgl01366.jpg' , 'MiamiSet00/imgl01386.jpg' , 'MiamiSet00/imgl01396.jpg'};
Models ={'Translation','Similarity','Affine','Projective'};
 % Create a figure
figure;
I1 = imread(image1filename);
% Loop to create subplots
avg_mean_error = zeros(length(image_files),length(Models)); % 3- for image 4 -for model types
for i = 2:4
    for j = 1:length(Models)
        % Do feature association with the modified match.m function
        distRatio = 0.8;
        drawMatches = false;        
        [CL1uv,CL2uv,loc1,des1,loc2,des2] = matchsiftmodif(image1filename, image_files{i}, distRatio, drawMatches);

        
        I2 = imread(image_files{i});

        % compute the homography 
        H12 = computeHomographyRANSAC(CL1uv, CL2uv, Models{j});
        errorVec = projectionerrorvec(H12, CL1uv, CL2uv);
        mean_error = mean(errorVec);
        avg_mean_error(i, j) = mean_error;

        % This is a function that warps image I2 into the frame of image I1
        overlay = showwarpedimages(I1, I2, H12);

        % Calculate the subplot index
        subplotIndex = (i - 2) * length(Models) + j;

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
sgtitle('Warped Images with Model Overlays Adding Ransac');

% Set a larger figure size for better visibility
set(gcf, 'Position', [100, 100, 1200, 800]);


% Loop over model types
% Plot the average mean error for each image
disp(avg_mean_error)
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



