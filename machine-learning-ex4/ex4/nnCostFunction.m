function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%Calculate A1, A2 and A3

%A1 = add bias term to X
A1 = [ones(size(X,1), 1) X];   % 5000x401   401x25

A2 = sigmoid(A1*Theta1');      % 5000X25

%Add bias term to A2
A2 = [ones(size(A2,1), 1) A2];  % 5000%26


A3 = sigmoid(A2*Theta2');       % 5000x26 26x10  = 5000x10

%use the temp variables
temp_value = 0;
temp_lable = 0;


%Below part is bit tricky. For every multiplication of y_of_j * log (G(X)), the y_of_j should be a vector of size num_label having 1s and 0. 1 should be only at the position of digit. 
% See the ex4.pdf page 5 for the example.

for i=1:m

     temp_vector = reshape( (1:num_labels), num_labels, 1);
     temp_vector = (temp_vector == y(i));  % The logical array - having only 0s and 1. 1 denote the output result label
     temp_value = temp_value + log(A3(i,:))*temp_vector + log(1-A3(i,:))*(1-temp_vector);

endfor


J = temp_value*(-1/m);


%add regularization - NOTE: SHOULD NOT INCLUDE THE BIAS TERM (MEANS IGNORE THE COLUMN 1 in both THETA)

reg=0;
reg=sum(power(Theta1(:,2:end),2)(:)) + sum ( power(Theta2(:,2:end),2)(:));
reg = reg*(lambda/2)*(1/m);


%Minimum cost function
J = J+reg;



%%%% ================ CALCULATE TEH BACKPROPAGATION
  %ACCUMULATORS 
  DELTA_1 = zeros(size(Theta1));
  DELTA_2 = zeros(size(Theta2));


% for every t of the input sample data and output (xt, yt)
for t = 1:m

% ------------ PART 1 - %Calculate computing the activations (z(2); a(2); z(3); a(3))  for layers 2 and 3

  %A1 = add bias term to X_T
  A1_T = [1 X(t,:)];                % 1+ 1x400 = 1x401
  
  Z2_T = A1_T*Theta1';              %1x401 401x25  = 1x25

  A2_T = [1 sigmoid(Z2_T)];     %1 x 26

  Z3_T = A2_T*Theta2';        % 1x26  26x10 = 1x10
  
  % Calculate the A3
  A3_T = sigmoid(Z3_T);     % 1x10

  
  
% ------------ PART 2 - Calculate the delta_3 = a3 - y(t)
  y_vector = (1:num_labels) ; 
  
  delta_3 = A3_T-(y_vector==y(t));   % logical array - 1x10
  
 
% ------------ PART 3 - Calculate the delta2

  delta_2 = (delta_3*Theta2).*sigmoidGradient([1 Z2_T]);  % 1x10 * 10x26 = 1x26


% ------------ PART 4  
   DELTA_1 = DELTA_1 + delta_2(2:end)'*A1_T;   % 25x1 1x401 = 25x401
   %Theta1_grad = Theta1_grad + delta_2'* A1_T;   % 25x1 1x401 = 25x401
   
   DELTA_2 = DELTA_2 + delta_3'*A2_T;      % 10x1 1x26 = 10x26
   %Theta2_grad = Theta2_grad + delta_3'*[zeros(size(A2_T,1), 1) A2_T];      % 10x1 1x26 = 10x26
  

 endfor

  
 % STEP 5- Divide by m as per the equation
 
 % Add the regularization for the gradient - ignore the first column of theta from the bias term
 Theta1_grad = DELTA_1./m + (lambda/m)*[zeros(size(Theta1,1), 1)  Theta1(:, 2:end)];
 Theta2_grad = DELTA_2./m + (lambda/m)*[zeros(size(Theta2,1), 1)  Theta2(:, 2:end)];
 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
