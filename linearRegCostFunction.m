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

hypo = X*theta; % Computes the hypothesis vector
theta_reg = theta; 
theta_reg(1) = 0; % Excludes the bias parameter for regularization

J = (1/(2*m))*(sum((hypo-y).^2)) + (lambda/(2*m))*sum(sum(theta_reg.^2)); % Regularized cost function

grad = (1/m)*(X')*(hypo-y) + (lambda/m)*theta_reg; % Computes the gradient of the cost 

%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
