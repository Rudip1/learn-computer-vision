% Step 1: Initializating of intrinsic and extrinsic parameters
% Intrinsic parameters Initializations
au = 557.0943; 
av = 712.9824; 
u0 = 326.3819; 
v0 = 298.6679; 

% Location of the world reference frame in camera coordinates in mm
trans_x = 100; 
trans_y = 0; 
trans_z = 1500;

% World rotation w.r.t. camera coordinates
% Euler XYX1 angles
rot_x_angle = 0.8 * pi / 2;
rot_y_angle = -1.8 * pi / 2;
rot_x1_angle = pi / 5;

% Step 2: Calculating the camera intrinsic and extrinsic transformation matrix
% Intrinsic matrix
intrinsicMatrix = [au, 0, u0; 0, av, v0; 0, 0, 1]; 

% Rotation matrices using Euler angles XYX1
rot_x_matrix = [1, 0, 0; 0, cos(rot_x_angle), -sin(rot_x_angle); 0, sin(rot_x_angle), cos(rot_x_angle)];
rot_y_matrix = [cos(rot_y_angle), 0, sin(rot_y_angle); 0, 1, 0; -sin(rot_y_angle), 0, cos(rot_y_angle)];
rot_x1_matrix = [1, 0, 0; 0, cos(rot_x1_angle), -sin(rot_x1_angle); 0, sin(rot_x1_angle), cos(rot_x1_angle)];

% Combining the rotation matrices
rot_combined_matrix = rot_x_matrix * rot_y_matrix * rot_x1_matrix;

% Creating the rotation matrix in homogeneous form
rotationMatrix = [rot_combined_matrix, [0; 0; 0]; 0, 0, 0, 1];

% Translation matrix (4x1)
translationMatrix = [trans_x; trans_y; trans_z; 1]; 

% Combining rotation and translation to form the camera-to-world transformation matrix
camWorldTransform = [rotationMatrix(:, 1:3), translationMatrix]; 

% Projection Matrix
projMatrix = intrinsicMatrix * camWorldTransform(1:3, :); 
disp('Camera Projection Matrix');
disp(projMatrix);

% Step 3: Create 6 random 3D points
% Function to generate 3D points
function random_points = generate_3d_points(min_val, max_val, num_points)
    random_points = min_val + (max_val - min_val) * rand(num_points, 3);
end

% Input to function above
min_val = -480;
max_val = 480;
num_points_3d = 6;
points_3d = generate_3d_points(min_val, max_val, num_points_3d);
disp('Random 3D Points:');
disp(points_3d);

% 3D plot of the 3D Points
figure('Name', '3D Points Distribution', 'NumberTitle', 'off');
scatter3(points_3d(:,1), points_3d(:,2), points_3d(:,3), 100, 'filled', 'MarkerFaceColor', 'cyan');
grid on;
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('3D Points Distribution');
axis equal;
hold on;

% Step 4: Project the 3D points onto the image plane
function points_2d = project_3d_to_2d(num_points_3d, points_3d, projMatrix)
    % Convert 3D points to homogeneous coordinates (add a row of ones)
    points_3d_homogeneous = [points_3d.'; ones(1, num_points_3d)];
    
    % Projecting 3D points onto the image plane using the projection matrix
    projected_points_homogeneous = projMatrix * points_3d_homogeneous;
    
    % Normalizing
    points_2d = projected_points_homogeneous(1:2, :) ./ projected_points_homogeneous(3, :);
end
points_2d = project_3d_to_2d(num_points_3d, points_3d, projMatrix);

% Display the projected points
disp('Projection of the 3D points on the 2D image plane:');
disp(points_2d);

% Step 5: Plot the 2D points in a separate window
figure('Name', 'Projected 2D Points on the Image Plane', 'NumberTitle', 'off');
scatter(points_2d(1, :), points_2d(2, :), 100, 'filled', 'MarkerFaceColor', 'blue');
grid on;
xlabel('u (pixels)');
ylabel('v (pixels)');
title('2D Projection of 3D Points on the Image Plane');
axis equal;

% Step 6: Estimating projection matrix using Hall's method
estimated_P = estimate_proj_matrix(points_2d, points_3d, num_points_3d, projMatrix);
disp('The projection matrix using Hall''s method is:');
disp(estimated_P);

function estimated_P = estimate_proj_matrix(points_2d, points_3d, num_points, original_P)
    % Initializing the Q matrix and B vector
    Q_matrix = zeros(2 * num_points, 11);
    B_vector = zeros(2 * num_points, 1);

    for i = 1:num_points
        % Building the Q matrix
        Q_matrix(2 * i - 1, :) = [points_3d(i, :), 1, 0, 0, 0, 0, ...
            -points_2d(1, i) * points_3d(i, 1), -points_2d(1, i) * points_3d(i, 2), -points_2d(1, i) * points_3d(i, 3)];
        Q_matrix(2 * i, :) = [0, 0, 0, 0, points_3d(i, :), 1, ...
            -points_2d(2, i) * points_3d(i, 1), -points_2d(2, i) * points_3d(i, 2), -points_2d(2, i) * points_3d(i, 3)];
        
        % Building the B vector (2D image points)
        B_vector(2 * i - 1) = points_2d(1, i);  % u-coordinate
        B_vector(2 * i) = points_2d(2, i);      % v-coordinate
    end

    % Solving
    A_vector = pinv(Q_matrix) * B_vector;

    % Adding 1 for homogeneous scale
    A_vector = [A_vector; 1];

    % Reshape the vector into a 3x4 projection matrix
    estimated_P = reshape(A_vector, 4, 3)';
end

% Step 7: Compare estimated matrix to the original one and extract K and R
%Comparing the matrix obtained in Step 6 (estimated_P) to the one defined in step 2 (projMatrix).
% Difference betn (estimated_P) to the (projMatrix).
matrix_difference = projMatrix - estimated_P;
disp('Difference between Original and Estimated Projection Matrices:');
disp(matrix_difference);

% Function to extract intrinsic parameters and rotation matrix from projection matrix
[K, cRw] = get_intrinsics_from_proj_matrix(estimated_P);
disp('Intrinsic Parameter Matrix K:');
disp(K);
disp('Camera Rotation Matrix R_w to c:');
disp(cRw);

function [K, cRw] = get_intrinsics_from_proj_matrix(P)
    % Get left-side 3x3 block of P
    M = P(:,1:3);
    [Q,R] = qr(rot90(M,3));
    R = rot90(R,2)';
    Q = rot90(Q);
    
    % Determinant checking of Q to make cRw
    if det(Q) < 0
        cRw = -Q;
    else
        cRw = Q;
    end
    
    %Normalized intrinsics
    K = R / R(3,3);
end


% Step 8: Adding Gaussian noise to all the 2D points
% Generate Gaussian noise
mean_noise = 0; % mean of the noise
std_noise = 0.5; % standard deviation of the noise for 95% within ±1 pixel
noise = normrnd(mean_noise, std_noise, size(points_2d)); % Generate noise

% Add noise to the projected 2D points
noisy_points_2d = points_2d + noise;

% Display the noisy projected points
disp('Noisy Projection of the 3D points on the 2D image plane:');
disp(noisy_points_2d);

% Plot the noisy 2D points in a separate window
figure('Name', 'Noisy Projected 2D Points on the Image Plane', 'NumberTitle', 'off');
scatter(noisy_points_2d(1, :), noisy_points_2d(2, :), 100, 'filled', 'MarkerFaceColor', 'red');
grid on;
xlabel('u (pixels)');
ylabel('v (pixels)');
title('Noisy 2D Projection of 3D Points on the Image Plane');
axis equal;

% Repeating  Step 6 to  Estimate noisy projection matrix using Hall's method
estimated_P_noisy = estimate_proj_matrix(noisy_points_2d, points_3d, num_points_3d, projMatrix);
disp('The projection matrix using Hall''s method with noisy points is:');
disp(estimated_P_noisy);

%Comparing the projection matrix obtained in  step 6, with the noise-free points.
% Difference betn (estimated_P_noisy) to the (estimated_P).
matrix_difference2 = estimated_P-estimated_P_noisy;
disp('Difference between estimated Projection Matrices without noisa and with noise:');
disp(matrix_difference2);

%Extracting K and R from the noisy projection matrix
[K_noisy, cRw_noisy] = get_intrinsics_from_proj_matrix(estimated_P_noisy);

% Display the results
disp('Intrinsic Parameter Matrix K (from noisy projection):');
disp(K_noisy);

disp('Camera Rotation Matrix R_w to c (from noisy projection):');
disp(cRw_noisy);


% Step 9: Compute the 2D points using noisy estimated projection matrix and
% compare with orginal projection matrix
% Project the 3D points using the noisy projection matrix
noisy_proj_matrix = estimated_P_noisy; % Use the projection matrix from step 8
new_points_2d = project_3d_to_2d(num_points_3d, points_3d, noisy_proj_matrix);

% Display the newly projected points
disp('Noisy Projection of 3D points on the 2D image plane with noise:');
disp(noisy_points_2d);

% Euclidean distances between the original 2D points and noisy projected points
euclidean_distances = sqrt(sum((points_2d - noisy_points_2d).^2, 1));

%Average projection error
average_projection_error = mean(euclidean_distances);
disp('Average Projection Error between noise-free and noisy points:');
disp(average_projection_error);


% Step 10: Increase the number of 3D points and repeat steps 8 and 9
num_points_array = 1:50;
average_errors = zeros(length(num_points_array), 1);

for idx = 1:length(num_points_array)
    num_points_3d = num_points_array(idx); % Current number of points
    points_3d = generate_3d_points(min_val, max_val, num_points_3d); % Generate new random 3D points

    % Repeating Step 4: Project the 3D points onto the image plane (using original projection matrix)
    points_2d = project_3d_to_2d(num_points_3d, points_3d, projMatrix);

    % Repeating Step 8: Add Gaussian noise to the 2D points and repeat the projection
    mean_noise = 0; % mean of the noise
    std_noise = 0.5; % standard deviation of the noise for 95% within ±1 pixel
    noise = normrnd(mean_noise, std_noise, size(points_2d)); % Generate noise

    % Adding noise to the projected 2D points
    noisy_points_2d = points_2d + noise;

    % Repeating Step 6: Projection matrix using noisy 2D points
    estimated_P_noisy = estimate_proj_matrix(noisy_points_2d, points_3d, num_points_3d, projMatrix);

    % Repeating Step 9: New 2D points using the noisy estimated projection matrix
    new_points_2d = project_3d_to_2d(num_points_3d, points_3d, estimated_P_noisy);

    %Average projection error
    euclidean_distances = sqrt(sum((points_2d - new_points_2d).^2, 1)); 
    average_errors(idx) = mean(euclidean_distances);
end

% Step 10: Ploting the average projection error as a function of the number of points
figure('Name', 'Average Projection Error vs Number of Points', 'NumberTitle', 'off');
plot(num_points_array, average_errors, '-o', 'MarkerFaceColor', 'cyan', 'LineWidth', 2);
grid on;
xlabel('Number of 3D Points');
ylabel('Average Projection Error (pixels)');
title('Effect of Number of Points on Average Projection Error');
axis tight;
