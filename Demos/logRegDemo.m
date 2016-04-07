% Load and prepare data
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

% Create model
l = logisticRegression(tX, y);

maxIters = 10000;
iters = 0;
tol = 0.00001;
grad = l.gradient;

while iters < maxIters && grad' * grad > tol
    grad = l.gradient;
    % l.w = gradientDescent(0.01, l.w, grad);
    l.w = newtonsMethod(1, l.w, grad, l.hessian);
    iters = iters + 1;
    l.cost
end
