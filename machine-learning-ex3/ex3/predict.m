function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%




%intermediate variables


% Add a0 to the matrix
A1 = [ones(size(X,1), 1) X];

%calculate the G_of_Z2
A2 = sigmoid(A1*Theta1');

%Add bias term to the value
A2 = [ones(size(A2,1), 1) A2];


%calculate the G_of_Z3
A3 = sigmoid(A2*Theta2');

%intermediate variable - dont need to use as it points to the maximum values per each rows in the matrix A3
value_returned = zeros(size(p));

% the max function will return the maximum value of every row in the matrix - which is the highest possibility for a digit - The index of the highest value is the label itself. :)

[value_returned, p] = max(A3, [], 2);



% =========================================================================


end
