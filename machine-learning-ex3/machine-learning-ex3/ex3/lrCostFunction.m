function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
costFunctionFac = lambda/(2*m);
temp=theta;
temp(1)=0;
regularizationFactors = (temp).^2;
regularization=costFunctionFac*sum(regularizationFactors);
regularizationGrad=lambda/m;
%--------Cost Function-----------

z=X*theta;
hypothesis = sigmoid(z);
mulFactor=(1/m);
logXMat=log(hypothesis);
logAnotherFactor=log(1-hypothesis);
mul1=(-1)*y;
mul2=(1-y);
product1=mul1.*logXMat;
product2=mul2.*logAnotherFactor;
tempMat = product1-product2;
J=(mulFactor*sum(tempMat))+regularization;

%-------------------------------------------
%--------Gradient Calculation----------------
gradFactor1 = hypothesis-y;
grad=((mulFactor).*(X'))*(gradFactor1);
temp=theta;%[(ones((size(theta,1)),1));theta];
temp(1)=0;
regularGradFactor=regularizationGrad.*temp;
grad=grad+regularGradFactor;
%--------Gradient Calculation----------------

% =============================================================
end
