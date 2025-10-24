function F = compute_fundamental_m(cam1_p2d, cam2_p2d )
    %matrix to hold coefficients of solving the equation U_n * F = 0
    U_n = [];
    %constructing U_n
    for i = 1:size(cam1_p2d,2)
       U = [cam1_p2d(1,i).*cam2_p2d(1,i), cam1_p2d(2,i).*cam2_p2d(1,i), cam2_p2d(1,i),...
           cam1_p2d(1,i).*cam2_p2d(2,i), cam1_p2d(2,i).*cam2_p2d(2,i), cam2_p2d(2,i),...
           cam1_p2d(1,i),cam1_p2d(2,i)  , 1];
       %appending the row
       U_n = [U_n;U];
    
    end
    %singular value decomposition to solve for fundamentals matrix
    [~,~,V] = svd(U_n);
    
    % Extract fundamental matrix (last column of v 
    F= reshape(V(:,9),3,3)';
    % Enforcing Rank2 constraint
    [U,D,V] = svd(F);
    F = U*diag([D(1,1) D(2,2) 0])*V'; 
   
end