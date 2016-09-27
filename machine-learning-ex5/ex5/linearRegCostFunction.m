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
grad = zeros(size(theta));  % [ 2 x 1]


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X [ 12 x 2 ]
% y [ 12 x 1 ]
% theta  [ 2 x 1 ]

J = sum( (X*theta-y).^2)/(2*m);


temp = theta;
temp(1) = 0;

J = J + sum(temp.^2)*lambda/(2*m);


grad = sum( (X*theta - y).*X)/m
grad = grad';

grad = grad + (lambda/m)*temp;


%for j = 1:size(grad,1)
 % grad(j) = (1/m)*sum(( X*theta - y).*X(:,j)) + (lambda*temp(j)/m);
%endfor



% =========================================================================

grad = grad(:);

end
