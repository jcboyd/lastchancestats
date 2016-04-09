load('regression_data');

% Invoke constant term
tX = [ones(length(y), 1) X];
% Record data dimensions
[N, D] = size(tX);

% Initialise parameters
theta = zeros(D, 1);
alpha = 0.01;
% Initialise gradient
grad = -tX' * y;

% Set stopping conditions
maxIters = 1000;
iters = 0;
tol = 0.0001;

% Start timer
tic;

while iters < maxIters && grad' * grad > tol
    % LEAST SQUARES GRADIENT DESCENT
    % grad = tX' * (tX * theta - y);
    % theta = gradientDescent(alpha, theta, grad);

    % STOCHASTIC GRADIENT DESCENT
    idx = randperm(N);
    % Randomly permute data
    tX = tX(idx, :);
    y = y(idx, :);
    % Run over epoch
    for i = 1 : N / B
        grad = ((tX(a:b, :) * theta - y(a:b, :))' * tX(a:b, :))';
        theta = gradientDescent(alpha, theta, grad);
    end
    % Report cost
    cost = 1 / N * (y - tX * theta)' * (y - tX * theta);
    fprintf('Cost: %f\n', cost);
    iters = iters + 1;
end

elapsedTime = toc;
fprintf('Run time: %f s\n', elapsedTime);
