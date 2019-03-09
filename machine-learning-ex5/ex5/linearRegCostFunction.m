function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% X = MxN
% y = Mx1
% h_of_x = Mx1
% theta = N*1


h_of_x = X*theta;


J = 0.5*(1/m)*sum(power((h_of_x - y), 2)) + 0.5*(lambda/m)*sum( power(theta(2:end), 2));

%fprintf("J VALUE CALCULATED %f \n", J);

grad = (1/m)*(X'*(h_of_x - y)) +  [ 0; (lambda/m)*(theta(2:end))];


% =========================================================================


grad = grad(:);

end
