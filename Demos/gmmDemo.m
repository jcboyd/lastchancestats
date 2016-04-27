% Gaussian Mixture Models (GMM) with expectation maximisation on randomly
% generated data. Note pause (line 53) to observe visualisation progress.
clear all;
load('gmm_data.mat');
[N, ~] = size(X);

% EM algorithm
K = 3;

% Initialise parameters
Mu = cell(K, 1); Sigma = cell(K, 1); Pi = cell(K, 1);

for k = 1 : K
    Mu{k} = min(X) + (max(X) - min(X)) .* rand(1, 2);
    Sigma{k} = eye(2);
    Pi{k} = 1 / K;
end

% Initialise marginals
P = zeros(N, K);

numiters = 0; maxiters = 10;

while numiters < maxiters
    hold off;
    % E-step
    for n = 1 : N
        for k = 1 : K
            P(n, k) = Pi{k} * mvnpdf(X(n, :), Mu{k}, Sigma{k});
        end
        P(n, :) = P(n, :) ./ sum(P(n, :));
        % Plot observations
        plot(X(n, 1), X(n, 2), 'o', 'MarkerSize', 8, 'LineWidth', 1, ...
            'Color', [P(n, 1), P(n, 2), P(n, 3)], ...
            'MarkerFaceColor', [P(n, 1), P(n, 2), P(n, 3)], ...
            'MarkerEdgeColor', [0.5 * P(n, 1), 0.5 * P(n, 2), 0.5 * P(n, 3)]);
        hold on; grid on;
    end
    drawnow; pause;
    % M-step
    for k = 1 : K
        % Optimise each parameter
        Mu{k} = sum(diag(P(:, k)) * X, 1) / sum(P(:, k));
        dev = X - repmat(Mu{k}, N, 1);
        Sigma{k} = dev' * diag(P(:, k)) * dev / sum(P(:, k));
        Pi{k} = sum(P(:, k)) / N;
    end
    numiters = numiters + 1;
end

% Plot contours
x1 = -5:.05:5; x2 = -5:.05:5;
[X1, X2] = meshgrid(x1, x2);

for k = 1:K
    hold on;
    F = mvnpdf([X1(:) X2(:)], Mu{k}, Sigma{k});
    F = reshape(F, length(x2), length(x1));
    contour(x1, x2, F, .25, 'color', 'black', 'LineWidth', 1);
end
