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

Y = zeros(size(y, 1), num_labels); % Initializes the expanded data in vector format

for i = 1:size(y, 1) % Loop that recodes the labels to vector format
    label = y(i);
    Y(i, label) = 1; % Assigns 1 to the position corresponding to the label
end

% Initialize the error matrices
Theta1_grad = zeros(size(Theta1, 1), size(X, 2)+1); 
Theta2_grad = zeros(size(Theta2, 1), size(Theta2, 2));

for i = 1:size(X, 1) % Loop that computes the cost function and gradient over all training examples
    
    % Forward propagation
    a1 = [X(i, :)]';
    a1_bias = [1 X(i, :)]';
    z2 = Theta1*a1_bias;
    a2 = sigmoid(z2);
    a2_bias = [1; a2];
    z3 = Theta2*a2_bias;
    a3 = sigmoid(z3);
    
    y_rel = Y(i, :)'; % Extracts the label vector for one example
    
    % Backpropagation
    del_3 = a3 - y_rel;
    del_2 = (Theta2')*del_3;
    del_2 = del_2(2:end); % Excludes the error in the bias unit
    del_2 = del_2.*sigmoidGradient(z2);
    
    
    % Accumulates the gradients
    Theta1_grad = Theta1_grad + del_2*(a1_bias'); 
    Theta2_grad = Theta2_grad + del_3*(a2_bias');
    
    % Accumulates the cost
    add_cost = sum(-y_rel.*log(a3)-(1-y_rel).*log(1-a3));
    J = J + add_cost;
end

J = (1/m)*J; % !! This cost function is unregularized
Theta1_grad = (1/m)*Theta1_grad; % !! This gradient matrix is unregularized
Theta2_grad = (1/m)*Theta2_grad; % !! This gradient matrix is unregularized

Theta1_reg = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]; % Excludes the bias column
Theta2_reg = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]; % Excludes the bias column

J = J + (lambda/(2*m))*(sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2))); % Regularizes the cost function

Theta1_grad = Theta1_grad + (lambda/m)*Theta1_reg;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2_reg;

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
