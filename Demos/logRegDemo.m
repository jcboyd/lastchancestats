% Data for binary logistic regression
load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
y = gender;
X = [height(:) weight(:)];
% Randomly permute data
N = length(y);
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% Subsample
y = y(1:200);
X = X(1:200,:);
% Normalise datas
X (:,1) = X (:,1) - mean(X(:, 1));
X (:,2) = X (:,2) - mean(X(:, 2));
X(:,1) = X(:,1) ./ std(X(:,1));
X(:,2) = X(:,2) ./ std(X(:,2));
% Invoke constant term
tX = [ones(length(y), 1) X];

% % Data for multinomial logistic regression
% N = 10;
% C = 3;
% 
% X = [rand(N, 1), rand(N, 1);
%     rand(N, 1), 3 * rand(N, 1) * 0.5;
%     3 * rand(N, 1) + 0.5, 3 * rand(N, 1) * 0.5];
% I = eye(C);
% y = [repmat(I(1, :), N, 1); repmat(I(2, :), N, 1); repmat(I(3, :), N, 1)];
% 
% idx = randperm(N * C);
% y = y(idx,:);
% X = X(idx,:);
% % Invoke constant term
% tX = [ones(length(y), 1) X];

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
