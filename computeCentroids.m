function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%


for i = 1:K % Loops over values of K
    a = find(idx==i)'; % Returns the position of every example in cluster i
    for j = a % Loops over training examples
        centroids(i, :) = centroids(i, :) + X(j, :); % Adds examples to each other
    end
    centroids(i, :) = centroids(i, :)/length(a); % Divides by the number of examples in that cluster, obtaining an average
end




        


% Note: You can use a for-loop over the centroids to compute this.
%








% =============================================================


end

