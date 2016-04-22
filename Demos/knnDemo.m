clear all;
rng default;

% Set model parameters
k = 7;

N1 = 40;
N2 = 30;

% Create training data
r1 = mvnrnd([-1.0, -0.5], eye(2), N1);
r2 = mvnrnd([+1.5, +1.0], 2*eye(2), N2);

hold on;
xlim([-4, 4]);
ylim([-4, 4]);
plot(r1(:,1), r1(:, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1.5, 'color', 'red');
plot(r2(:,1), r2(:, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1.5, 'color', 'blue');

% Create test data
[X, Y] = meshgrid(-3.75:0.25:3.75, -3.75:0.25:3.75);
Z = zeros(size(X));

% Create class column
r = [[r1, zeros(N1, 1); r2, ones(N2, 1)], zeros(N1 + N2, 1)];

for i = 1 : size(X)
    for j = 1 : size(Y)
        % Create distance column
        r(:, 4) = pdist2(r(:,1:2), [X(i, j), Y(i, j)]);
        [~, idx] = sort(r(:, 4));
        classes = r(idx, 3);
        Z(i, j) = round(mean(classes(1:k)));
    end
end

contour(X, Y, Z, [1,1], 'black');

cl1 = find(Z == 0);
cl2 = find(Z == 1);

X0 = X(cl1); Y0 = Y(cl1); X1 = X(cl2); Y1 = Y(cl2);

plot(X0(:), Y0(:), '+', 'MarkerSize', 2, 'LineWidth', 0.1, 'color', 'red');
plot(X1(:), Y1(:), '+', 'MarkerSize', 2, 'LineWidth', 0.1, 'color', 'blue');
