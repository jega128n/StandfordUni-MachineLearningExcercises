function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).




% Functon to implement
% g(z) = 1 / ( 1 + power(e, -z)



% Declare ons matrix of Size z to represent Numerator 1

N = ones ( size(z));


% Declare D to represent the deominator
D = 1+power(e, -z);

% Use dot (.) for matrix element-wise operation

g = N./D;



% =============================================================

end
