function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

divfactor=1/m;
modFactor = (theta')*(X');
factorVal1=log(sigmoid(modFactor));
factorVal2=log(1-sigmoid(modFactor));
factorVal3=(-1).*y;
factorVal4=(1-y);
factorVal5=factorVal1.*(factorVal3');
factorVal6=(factorVal4').*factorVal2;
J=divfactor.* (sum(factorVal5-factorVal6));
error = (sigmoid(modFactor))-(y');
for matSize=1:size(theta) 
        grad(matSize,:)=divfactor.*(sum((error').*X(:,matSize)));
end
% =============================================================

end
