

disp(H12)

c1 = [2,4;6,8]

c2 = [2,4 ; 5,6]

err_vec = c1 -c2
err_ve = sqrt(sum(err_vec,2))
size(c1);

c1_h = [ c1 , ones(size(c1,1),1)]';

c1_est = H12*c1_h;

% function errorVec = projectionerrorvec(H12, CL1uv, CL2uv)
%     % Apply homography to CL1uv
%     CL1uv_hom = [CL1uv, ones(size(CL1uv, 1), 1)]';
%     CL2uv_est_hom = H12 * CL1uv_hom;
%     CL2uv_est_hom = CL2uv_est_hom ./ CL2uv_est_hom(3, :);
%     CL2uv_est = CL2uv_est_hom(1:2, :)';
% 
%     % Compute Euclidean distance between CL2uv and CL2uv_est
%     errorVec = sqrt(sum((CL2uv - CL2uv_est).^2, 2));
% end