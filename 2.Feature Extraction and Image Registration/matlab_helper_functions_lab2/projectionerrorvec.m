
function [errorVec] = projectionerrorvec(H12,CL1uv,CL2uv)
    % Calaculate the projection error vector 
    CL2uv_hom= [CL2uv , ones(size(CL2uv,1),1)]'; % change to homogenous
    CL1uv_est_hom= H12 * CL2uv_hom;  % estimate the corresponding projection points 
    CL1uv_est_hom = CL1uv_est_hom ./ CL1uv_est_hom(3, :);
    CL1uv_est = CL1uv_est_hom(1:2,:)'; % change to cartsian 
    
    errorVec = sqrt(sum((CL1uv - CL1uv_est).^2, 2)); % calaculate the error difference 

    

    