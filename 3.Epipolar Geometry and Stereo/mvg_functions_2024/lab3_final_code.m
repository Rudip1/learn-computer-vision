% This a empty script to help you move faster on you lab work.
clear all;
close all;
clc;

%% Step 1
% Camera 1
au1 = 100; av1 = 120; uo1 = 128; vo1 = 128;
imageSize = [256 256];

%% Step 2
% Camera 2
au2 = 90; av2 = 110; uo2 = 128; vo2 = 128; 
ax = 0.1; by = pi/4; cz = 0.2; % XYZ EULER 
tx = -1000; ty = 190; tz = 230; 

%% STEP 3
% Compute intrinsic matrices and projection matrices

K1 = [au1 0 uo1; 0 av1 vo1; 0 0 1]; % Intrisics matrix for camera 1
wR1c = eye(3);   % rotation of camera 1, from the camera to the world coordinate frame
wt1c = [0 0 0]'; % translation of camera 1, from the camera to the world coordinate frame

% Note: ************** You have to add your own code from here onward ************
K2 = [au2 0 uo2; 0 av2 vo2; 0 0 1];%Intrinsic matrix for camera 2

%computing rotaional matrix for camera 2 from euler angles
Rotx=[1 0 0;0 cos(ax) -sin(ax); 0 sin(ax) cos(ax)];
Roty=[cos(by) 0 sin(by); 0 1 0;-sin(by) 0 cos(by)];
Rotz=[cos(cz) -sin(cz) 0;sin(cz) cos(cz) 0; 0 0 1];
wR2c = Rotx*Roty*Rotz;%combined rotation for camera2
wt2c = [tx;ty;tz;];%Translation for camera 2


P1 = K1 * [wR1c' -wR1c' * wt1c];

P2 = K2 * [wR2c' -wR2c' * wt2c]; % Projection matrix for Camera 2


%% STEP 4

% Attention: This is an invented matrix just to have some input for the drawing
% functions. You have to compute it properly 
%F = [3e-05 7e-05 -0.006;2e-05 -2.e-05 0.01;-0.009 -0.01 1];
% Define relative rotation from Camera 2 to Camera 1
c1Rc2 = wR1c' * wR2c;

% Define skew-symmetric matrix for translation vector from Camera 1 to Camera 2
c1tc2x = [0, -tz, ty; tz, 0, -tx; -ty, tx, 0];

% Compute the fundamental matrix
F = inv(K2)' *  c1Rc2' * c1tc2x  * inv(K1);

F = F./F(3,3);

fprintf('Step 4:\n\tAnalytically obtained F:\n');
disp(F);
%% STEP 5
V(:,1) = [100;-400;2000];
V(:,2) = [300;-400;3000];
V(:,3) = [500;-400;4000];
V(:,4) = [700;-400;2000];
V(:,5) = [900;-400;3000];
V(:,6) = [100;-40;4000];
V(:,7) = [300;-40;2000];
V(:,8) = [500;-40;3000];
V(:,9) = [700;-40;4000];
V(:,10) = [900;-40;2000];
V(:,11) = [100;40;3000];
V(:,12) = [300;40;4000];
V(:,13) = [500;40;2000];
V(:,14) = [700;40;3000];
V(:,15) = [900;40;4000];
V(:,16) = [100;400;2000];
V(:,17) = [300;400;3000];
V(:,18) = [500;400;4000];
V(:,19) = [700;400;2000];
V(:,20) = [900;400;3000];

%% STEP 6
% Projection on image planes
cam1_p2d = mvg_projectPointToImagePlane(V,P1);
cam2_p2d = mvg_projectPointToImagePlane(V,P2);

%% STEP 7
% example of the plotting functions 
% Draw 2D projections on image planes
cam1_fig = mvg_show_projected_points(cam1_p2d(1:2,:),imageSize,'Projected points on image plane 1 for Analytical F');
cam2_fig = mvg_show_projected_points(cam2_p2d(1:2,:),imageSize,'Projected points on image plane 2 for Analytical F');
                                                                                 
% Draw epipolar lines                                                                                       
[~,~,c1_l_coeff,c2_l_coeff] = mvg_compute_epipolar_geom_modif(cam1_p2d,cam2_p2d,F);                             
[cam1_fig,cam2_fig] = mvg_show_epipolar_lines(cam1_fig, cam2_fig, c1_l_coeff,c2_l_coeff, [-400,1;300,400],'b');                 
                                                                                                                                                        
% Draw epipoles
% Draw epipoles Using Null space
e1 = null(F);
e2 = null(F');
% Normalize
ep_1 = e1/e1(3);
ep_2 = e2/e2(3);
[~,~] = mvg_show_epipoles(cam1_fig, cam2_fig,ep_1,ep_2);

%% Step8
%Fundamental matrix with 8-points
enforce_rank2 = 0;
New_F = compute_fundamental_m(cam1_p2d,cam2_p2d);
New_F = New_F./New_F(3,3);%normalizing
fprintf('Funadamental_with_8_point:Step8:\n');
disp(New_F);

%% Step9
% sum of absolute differencees 
sad= sum(abs(F(:) - New_F(:)));
disp(['Sum of Absolute Differences of F ana New_F: ', num2str(sad)]);


%% Step10
% Display projected points on image planes
img1_proj = mvg_show_projected_points(cam1_p2d(1:2, :), imageSize, 'Projected points on Image Plane 1 (8-Point, STEP10)');
img2_proj = mvg_show_projected_points(cam2_p2d(1:2, :), imageSize, 'Projected points on Image Plane 2 (8-Point, STEP10)');

% Compute and draw epipolar lines
[~, ~, epiline_coeffs_img1, epiline_coeffs_img2] = mvg_compute_epipolar_geom_modif(cam1_p2d, cam2_p2d, New_F);
[img1_with_lines, img2_with_lines] = mvg_show_epipolar_lines(img1_proj, img2_proj, epiline_coeffs_img1, epiline_coeffs_img2, [-400, 1; 300, 400], 'k');

% Compute epipoles
epipole_img1 = null(New_F);
epipole_img2 = null(New_F');

% Normalize epipoles
epipole_img1 = epipole_img1 / epipole_img1(3);
epipole_img2 = epipole_img2 / epipole_img2(3);

% Show epipoles on images
[~, ~] = mvg_show_epipoles(img1_with_lines, img2_with_lines, epipole_img1, epipole_img2);


%% Step 11: Adding noise to points
% standard deviation for noise
noise_std_dev = 0.5;

% Generate random noise with normal distribution 
cam1_points_noisy = cam1_p2d + normrnd(0, noise_std_dev, size(cam1_p2d));
cam2_points_noisy = cam2_p2d + normrnd(0, noise_std_dev, size(cam2_p2d));

% Displaying the original and noisy points for comparison
fprintf('Step 11: Noise-free points (Camera 1):\n');
disp(cam1_p2d);

fprintf('Step 11: Noisy points (Camera 1):\n');
disp(cam1_points_noisy);


%% Step12 
%Step8 with noisy points
% Fundamental with 8 point for noisy Points
enforce_rank2 = 0;
Noisy_F12 = compute_fundamental_m(cam1_points_noisy,cam2_points_noisy);
% normalize
Noisy_F12 = Noisy_F12./Noisy_F12(end, end);
fprintf('Step 12: Funadamental_M_for_noise_point:\n');
disp(Noisy_F12);

%Step9 for noisy
% error difference with noise
% sum of absolute differences(SAD)
sad_noisy = sum(abs(F(:) - Noisy_F12(:)));
disp(['Sum of Absolute Differences of F ana Noisy_F: ', num2str(sad_noisy)]);

%Step10 for noisy
% Drawing Epipolar lines and epipole for noised one
cam1_fig = mvg_show_projected_points(cam1_points_noisy(1:2,:),imageSize,'Projected points on image plane 1 for noisy Step12');
cam2_fig = mvg_show_projected_points(cam2_points_noisy(1:2,:),imageSize,'Projected points on image plane 2 for noisy Step12');

% Draw epipolar lines
[~,~,c1_l_coeff_s12_noise,c2_l_coeff_s12_noise] = mvg_compute_epipolar_geom_modif(cam1_points_noisy,cam2_points_noisy,Noisy_F12);
[cam1_fig,cam2_fig] = mvg_show_epipolar_lines(cam1_fig, cam2_fig, c1_l_coeff_s12_noise,c2_l_coeff_s12_noise, [-400,1;300,400],'c');

% Draw epipoles  
% Epipoles Investigation with SVD
[U,~, V] = svd(Noisy_F12);
ep1_12_noise= V(:, end);
ep2_12_noise= U(:, end);

% Normalization
ep1_12_noise = ep1_12_noise / ep1_12_noise(3);
ep2_12_noise = ep2_12_noise / ep2_12_noise(3);

[~,~] = mvg_show_epipoles(cam1_fig, cam2_fig,ep1_12_noise,ep2_12_noise);


%% Step 13: Adding Noise and Recomputing

% Add Gaussian noise to points (std = 1 for 95% of values in [-2, +2])
noise_std_dev = 1;
cam1_points_noisy_13 = cam1_p2d + normrnd(0, noise_std_dev, size(cam1_p2d));
cam2_points_noisy_13 = cam2_p2d + normrnd(0, noise_std_dev, size(cam2_p2d));

% Compute fundamental matrix with noisy points and normalize
Noisy_F13 = compute_fundamental_m(cam1_points_noisy_13, cam2_points_noisy_13);
Noisy_F13 = Noisy_F13 ./ Noisy_F13(end, end);
fprintf('Step 13: Fundamental Matrix with Noisy Points:\n');
disp(Noisy_F13);

% Compute error (Sum of Absolute Differences) between F and noisy F
sad_noisy_13 = sum(abs(F(:) - Noisy_F13(:)));
disp(['Sum of Absolute Differences (F vs Noisy_F13): ', num2str(sad_noisy_13)]);

% Visualize noisy points projection on image planes
cam1_fig = mvg_show_projected_points(cam1_points_noisy_13(1:2, :), imageSize, 'Image 1: Noisy Points Step13');
cam2_fig = mvg_show_projected_points(cam2_points_noisy_13(1:2, :), imageSize, 'Image 2: Noisy Points Step13');

% Compute and draw epipolar lines with noisy points
[~, ~, c1_l_coeff_s13_noise, c2_l_coeff_s13_noise] = mvg_compute_epipolar_geom_modif(cam1_points_noisy_13, cam2_points_noisy_13, Noisy_F13);
[cam1_fig, cam2_fig] = mvg_show_epipolar_lines(cam1_fig, cam2_fig, c1_l_coeff_s13_noise, c2_l_coeff_s13_noise, [-400, 1; 300, 400], 'r');

% Compute epipoles using SVD and normalize
[U, ~, V] = svd(Noisy_F13);
ep1_13_noise = V(:, end) / V(3, end);
ep2_13_noise = U(:, end) / U(3, end);

% Display epipoles on images
[~, ~] = mvg_show_epipoles(cam1_fig, cam2_fig, ep1_13_noise, ep2_13_noise);

%% Step 14: Coordinate Normalization and Fundamental Matrix Estimation

% Normalize Points for Camera 1 
x_mean = mean(cam1_p2d(1, :));
y_mean = mean(cam1_p2d(2, :));
translated_points = cam1_p2d - [x_mean; y_mean];
Std_x = std(translated_points(1, :));
Std_y = std(translated_points(2, :));
scaled_points_cam1 = [translated_points(1, :) / Std_x; translated_points(2, :) / Std_y];

% T1 Transformation Matrix
T1_matrix = [1 / Std_x, 0, -x_mean / Std_x;
             0, 1 / Std_y, -y_mean / Std_y;
             0, 0, 1];

% Convert to Homogeneous Coordinates
Scaled_homo_cam1 = [scaled_points_cam1; ones(1, size(scaled_points_cam1, 2))];
WED_cam1 = T1_matrix * Scaled_homo_cam1;

%Normalizing Points for Camera 2
x1_mean = mean(cam2_p2d(1, :));
y1_mean = mean(cam2_p2d(2, :));
translated_points_2 = cam2_p2d - [x1_mean; y1_mean];
Std_x_2 = std(translated_points_2(1, :));
Std_y_2 = std(translated_points_2(2, :));
scaled_points_cam2 = [translated_points_2(1, :) / Std_x_2; translated_points_2(2, :) / Std_y_2];

% T2 Transformation Matrix
T2_matrix = [1 / Std_x_2, 0, -x1_mean / Std_x_2;
             0, 1 / Std_y_2, -y1_mean / Std_y_2;
             0, 0, 1];

% Convert to Homogeneous Coordinates
Scaled_homo_cam2 = [scaled_points_cam2; ones(1, size(scaled_points_cam2, 2))];
WED_cam2 = T2_matrix * Scaled_homo_cam2;


% Add Gaussian Noise 
noise_std_dev = 1;
cam1_p2d_14 = WED_cam1(1:2, :) + normrnd(0, noise_std_dev, size(cam1_p2d));
cam2_p2d_14 = WED_cam2(1:2, :) + normrnd(0, noise_std_dev, size(cam2_p2d));
%  Fundamental Matrix Estimation without Normalization 
U_n = [];
for i = 1:size(cam1_p2d, 2)
    U_row = [
        cam1_p2d(1, i) * cam2_p2d(1, i), ...
        cam1_p2d(2, i) * cam2_p2d(1, i), ...
        cam2_p2d(1, i), ...
        cam1_p2d(1, i) * cam2_p2d(2, i), ...
        cam1_p2d(2, i) * cam2_p2d(2, i), ...
        cam2_p2d(2, i), ...
        cam1_p2d(1, i), cam1_p2d(2, i), 1
    ];
    U_n = [U_n; U_row];
end
% Fundamental Matrix Estimation
U_n_14 = [];
for i = 1:size(cam1_p2d_14, 2)
    U_row = [
        cam1_p2d_14(1, i) * cam2_p2d_14(1, i), ...
        cam1_p2d_14(2, i) * cam2_p2d_14(1, i), ...
        cam2_p2d_14(1, i), ...
        cam1_p2d_14(1, i) * cam2_p2d_14(2, i), ...
        cam1_p2d_14(2, i) * cam2_p2d_14(2, i), ...
        cam2_p2d_14(2, i), ...
        cam1_p2d_14(1, i), cam1_p2d_14(2, i), 1
    ];
    U_n_14 = [U_n_14; U_row];
end

[~, ~, V] = svd(U_n_14);
F_14 = reshape(V(:, end), 3, 3)';
[U, D, V] = svd(F_14);
F_14_noised = U * diag([D(1, 1), D(2, 2), 0]) * V';
F_14_noised = F_14_noised / F_14_noised(end, end);

%Condition Numbers
condition_number_U_n = cond(U_n); % Non-normalized
condition_number_U_n_14 = cond(U_n_14); % Normalized
fprintf('Condition Number_Non-Normalized: %.2f\n', condition_number_U_n);
fprintf('Condition Number_Normalized: %.2f\n', condition_number_U_n_14);

% Sum of Absolute Differences
SAD_noised_14 = sum(abs(F(:) - F_14_noised(:)));
fprintf('Sum of Absolute Differences step14: %.4f\n', SAD_noised_14);

% Epipolar Geometry Visualization
cam1_fig = mvg_show_projected_points(cam1_p2d_14, imageSize, 'Projected points on image plane 1 STEP 14');
cam2_fig = mvg_show_projected_points(cam2_p2d_14, imageSize, 'Projected points on image plane 2 STEP 14');

[~, ~, c1_l_coeff_s14_noise, c2_l_coeff_s14_noise] = mvg_compute_epipolar_geom_modif(cam1_p2d_14, cam2_p2d_14, F_14_noised);
[cam1_fig, cam2_fig] = mvg_show_epipolar_lines(cam1_fig, cam2_fig, c1_l_coeff_s14_noise, c2_l_coeff_s14_noise, [-400, 1; 300, 400], 'y');

% Compute Epipoles
[U, ~, V] = svd(F_14_noised);
ep1_14_noise = V(:, end) / V(end, end);
ep2_14_noise = U(:, end) / U(end, end);
[~, ~] = mvg_show_epipoles(cam1_fig, cam2_fig, ep1_14_noise, ep2_14_noise);
