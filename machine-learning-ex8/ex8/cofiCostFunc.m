function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

 % multipling with R (binary matrix) to keep only the non-zero values
 product =  power( (X*Theta' - Y).*R, 2);
 
 % Calculate the J now 
 J = 0.5 * sum( product(:) );
 
 REG_1 = 0.5*lambda*sum( power(Theta, 2) (:));
 
 REG_2 = 0.5*lambda*sum( power(X, 2) (:));
 
 % Added with regularizations
 J = J + REG_1 + REG_2;
 
 
 [mX, nX] = size(X);
 [mTh, nTh] = size (Theta);
 
 
 %calculate for every X value
 for i = 1:mX 
  %find the users who rated this movie
  idx = find ( R (i,:) == 1);
  Theta_temp = Theta(idx, :);
  Y_temp = Y(i, idx);
  X_grad(i,:) = ( X(i,:)*Theta_temp' - Y_temp)*Theta_temp;
  
  %Add regularization
  X_grad(i,:) = X_grad(i,:) + lambda*X(i,:);
  
 end
 
 
 %num_users = 4; num_movies = 5; num_features = 3;
%X = X(1:num_movies, 1:num_features);
%Theta = Theta(1:num_users, 1:num_features);
%Y = Y(1:num_movies, 1:num_users);
%R = R(1:num_movies, 1:num_users);
 
 %calculate for every theta value
 for k = 1:mTh
  
  %find the movies rated by this user
  idx = find ( R(:,k) == 1);  
  X_temp = X ( idx, :);
  Y_temp = Y (idx,k);
 
  Theta_grad(k,:) = ( Theta(k,:)*X_temp' - Y_temp')*X_temp;
  
  %Add regularization
  Theta_grad(k,:) = Theta_grad(k,:) + lambda*Theta(k,:);
 
 end
 
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
