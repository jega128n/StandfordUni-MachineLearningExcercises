function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%temp variables
temp_sqroot_distance = 0.0;
min_tmp_sqroot = intmax(); %init to a really big value
temp_centriod = 0;


for i = 1 : length(X) 

  for j = 1 : K
   
     % X and centriods are matrixes
     temp_sqroot_distance = sum( power((X(i,:) - centroids(j,:)), 2) );
     
     
     if temp_sqroot_distance < min_tmp_sqroot
        
        min_tmp_sqroot = temp_sqroot_distance;
        temp_centriod = j;
       
     end
     
  end

  %assign the closest centriod
  idx(i) = temp_centriod;
  
  %reset temp
  min_tmp_sqroot = intmax();
  temp_centriod = 0;
  

end




% =============================================================

end

