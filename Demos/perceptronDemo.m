% PERCEPTRONDEMO learns the logical NAND function
% inspired by the example at https://en.wikipedia.org/wiki/Perceptron

% Construct data
D = 2;
N = 2^D;
X = dec2bin(0:N-1)-'0';
tX = [ones(N, 1), X];
y = ~(b(:, 1) & b(:, 1) & b(:, 2));

% Initialise weights
w = zeros(D + 1, 1);

% Set learning parameters
threshold = 0.5;
rate = 0.1;

maxIters = 1000;
iters = 0;

while iters < maxIters
    % Run over data
    for i = 1 : N
        y_hat = tX(i, :) * w > threshold;
        w = w + rate * (y(i) - y_hat) * tX(i, :)';
    end
    if tX * w > threshold == y
        break; % No more errors--convergence
    end
    iters = iters + 1;
end
