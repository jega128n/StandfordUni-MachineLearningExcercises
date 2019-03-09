function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


%n number of features + 1
n=length(theta);


%Call the function already implementated to calculate for standard logistic regression
[J, grad] = costFunction(theta, X, y);

% Theta_J Square value to add for regularization
theta_J = 0;

% PLEASE NOTE: The Theta_0 is omitted in this calculatation by definition, so starting from 2 ( as Octave is 1 indexed)
for j = 2 : n
 theta_J  = theta_J + power(theta(j,:), 2);
endfor

%theta_J

%Add to the J as per the definition of the regularized equation
J = J + (1/ (2*m))*lambda*theta_J;

grad_j = zeros(size(theta));

% PLEASE NOTE: The Theta_0 is omitted in this calculatation by definition, so starting from 2 ( as Octave is 1 indexed)
for j = 2 : n
  grad_j(j) = (1/m)*lambda*theta(j,:);
endfor

%grad_j

%Add to simplified grad to calculate the regulalized amount
grad = grad+grad_j;



% =============================================================

end
