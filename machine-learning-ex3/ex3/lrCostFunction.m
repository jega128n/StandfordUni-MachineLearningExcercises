function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


temp = sigmoid(X*theta);

%calculate the cost function without the regulaized part using the vectorized format
%NOTE: The element wise operator is used to calculate the y*log(g(x)) etc as this is an scalar multiplication of the vector.
J = (-1/m)*( y.*log(temp) + (1-y).*log(1-temp) );


% calculate the regularization value using the vector format - REMEMBER : Regularization does not incude Biasing term ( theta (0))
reg = lambda*0.5*(1/m)*sum(power(theta(2:end),2));


% SUM up the individual vectors to give a final value for the Cost function
J = sum(J) + sum(reg);



% Calculate the gradiation using the vector format ( without the regularizatio part)
grad = (1/m)*X'*(temp-y);

% regularization of the gradient -- REMEMBER : Regularization does not incude Biasing term ( theta (0))
grad_reg = lambda*(1/m).*theta(2:end);

grad_reg_comp = [0;grad_reg];

grad = grad+grad_reg_comp;


% =============================================================

%grad = grad(:);

end
