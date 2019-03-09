function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%temp variables
error = 100000;  % set a massive number
error_temp=0; 


% The values to be tried for C and Sigma
value_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
count = 0;

%iterate over the combination of C and Sigma values
for ic = value_set

  %loop over again for sigma values  
  for jsigma = value_set
    
    %Train the model using the GuassianKernel 
    model = svmTrain(X, y, ic, @(x1, x2) gaussianKernel(x1, x2, jsigma));
    
    %Use the model to get the prediction
    predictions = svmPredict(model, Xval);

    %Calculate the error
    error_temp = mean(double(predictions ~= yval));
    
    %retain the value set of C and Sigma based on minimal error
   if error_temp < error
    %fprintf('UPDATE current Error : %f to latest lower value : %f', error, error_temp);
    %fprintf('Current optima for C is %f and Sigma is %f' , ic, jsigma);
    error = error_temp;
    C = ic;
    sigma = jsigma;
   %else 
    %fprintf('Dont update error : %f and error temp : %f', error, error_temp);
    %fprintf('Current optima for C is %f and Sigma is %f' , C_temp, sigma_temp);
   end
   
   count = count +1;
      
   fprintf('Count number :%f Current optima for C is %f and Sigma is %f' , count, C, sigma);
    
  end
end


% =========================================================================


end
