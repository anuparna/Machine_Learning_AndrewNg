function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;




% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
positives = find(y==1);%will return the indices
negatives = find(y==0);

plot(X(positives,1),X(positives,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(negatives,1),X(negatives,2),'ko','LineWidth',2,'MarkerSize',7);








% =========================================================================



hold off;

end
