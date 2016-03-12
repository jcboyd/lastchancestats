% Gaussian Mixture Models (GMM) with expectation maximisation on randomly generated data
% Note pause (line 53) to observe visualisation progress
%

% Generate data

mu1 = [1 -1];
mu2 = [2, 1];
mu3 = [3, 2];
sigma1 = [.17 .34; .34 .85];
sigma2 = [.90 .40; .40 .30];
sigma3 = [.34 -.30; -.30 .36];

N = 180;

X1 = mvnrnd(mu1,sigma1,N/3);
X2 = mvnrnd(mu2,sigma2,N/3);
X3 = mvnrnd(mu3,sigma3,N/3);

X = [X1; X2; X3];

% plot(X1(:,1), X1(:,2), 'o', 'red', col(1), 'MarkerSize', 6, 'MarkerFaceColor', 'red');
% hold on;
% plot(X2(:,1), X2(:,2), 'o', 'green', col(2), 'MarkerSize', 6, 'MarkerFaceColor', 'green');
% hold on;
% plot(X3(:,1), X3(:,2), 'o', 'blue', col(3), 'MarkerSize', 6, 'MarkerFaceColor', 'blue);
% hold on;

grid on;

% EM algorithm

K = 3;

I = eye(2);
Pi =  ones(K, 1) / K; Sigma = repmat(I, K, 1); Mu = rand(K, 2);
P = zeros(N, K);

numiters = 1; maxiters = 30;

while numiters < maxiters
    % E-step
    for n = 1:N
        for k = 1:K
            P(n, k) = Pi(k) * mvnpdf(X(n, :), Mu(k, :), Sigma((2*(k - 1) + 1):2*k, :));
        end
        P(n, :) = P(n, :) ./ sum(P(n, :));
        % Plot observations
        hold on;
        plot(X(n, 1), X(n, 2), 'o', 'color', [P(n, 1), P(n, 2), P(n, 3)], 'MarkerSize', 6, 'MarkerFaceColor', [P(n, 1), P(n, 2), P(n, 3)]);
    end
    hold off;
    pause
    % M-step
    for k = 1:K
        Mu(k, :) = sum(diag(P(:, k)) * X, 1) / sum(P(:, k));
        dev = X - repmat(Mu(k, :), N, 1);
        Sigma((2*(k - 1) + 1):2*k, :) = transpose(dev) * diag(P(:, k)) * dev / sum(P(:, k));
        Pi(k) = sum(P(:, k)) / N;
    end
    numiters = numiters + 1;
end

% Plot contours
x1 = -5:.05:5; x2 = -5:.05:5;
[X1,X2] = meshgrid(x1,x2);
for k = 1:K
    hold on;
    F = mvnpdf([X1(:) X2(:)], Mu(k, :), Sigma((2*(k - 1) + 1):2*k, :));
    F = reshape(F,length(x2),length(x1));
    contour(x1,x2,F,.25,'color', 'black', 'LineWidth', 2);
end
