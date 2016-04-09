% Data for binary logistic regression
% load('log_reg_data.mat');

% Data for multinomial logistic regression
load('multi_log_reg_data.mat');

% Invoke constant term
tX = [ones(length(y), 1) X];

% Create model
m = logisticRegression(tX, y);
% m = multiLogisticRegression(tX, y);

maxIters = 10000;
iters = 0;
tol = 0.00001;
grad = m.gradient;

while iters < maxIters && grad' * grad > tol
    grad = m.gradient;
    % m.w = gradientDescent(0.01, m.w, grad);
    % m.W = reshape(gradientDescent(0.1, m.getWeights, grad), size(m.W));
    m.w = newtonsMethod(1, m.w, grad, m.hessian);
    % m.W = reshape(newtonsMethod(0.0001, m.getWeights, grad, m.hessian), size(m.W));
    iters = iters + 1;
    m.cost
end
