function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


for I = 1 : m
 
      % theta' = 1*2 matrix
      % X ( I,:) - 2*1 matrix  
      J = J + power( theta'*X(I,:)' - y(I,:) , 2 ) ;
      
      % Alternative equation
      %J = J + power( X(I,:)*theta - y(I,:) , 2 ) ; 

  
endfor

      % calculate (1/2m) of the summazation of squares
      J = (J/m)*0.5;


% =========================================================================

end
