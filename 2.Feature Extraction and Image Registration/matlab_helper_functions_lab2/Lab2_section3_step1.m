% Define image file names
image1name = 'imgl01311.jpg'; 
image2name = 'imgl01396.jpg';

% Test with distRatio = 0.4
distRatio = 0.4;
drawMatches = true;  % Set to true to visualize the matches
[CL1uv, CL2uv] = matchsiftmodif(image1name, image2name, distRatio, drawMatches);

% Test with distRatio = 0.6
distRatio = 0.6;
[CL1uv, CL2uv] = matchsiftmodif(image1name, image2name, distRatio, drawMatches);

% Test with distRatio = 0.8
distRatio = 0.8;
[CL1uv, CL2uv] = matchsiftmodif(image1name, image2name, distRatio, drawMatches);
