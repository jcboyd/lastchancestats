clear all;
load('knn_data.mat');

% set model parameters
k = 7;

% create test data
[teX1, teX2] = meshgrid(-3.75:0.25:3.75, -3.75:0.25:3.75);
teY = zeros(size(teX1));

for i = 1 : size(teX1)
    for j = 1 : size(teX2)
        % calculate distances
        dists = pdist2(X, [teX1(i, j), teX2(i, j)]);
        [~, idx] = sort(dists);
        classes = y(idx);
        % assign class on majority vote
        teY(i, j) = round(1/k * sum(classes(1:k)));
    end
end

%% Plot data
hold on; xlim([-4, 4]); ylim([-4, 4]);

% Plot training data
plot(X(y==0, 1), X(y==0, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1.5, ...
    'Color', 'red');
plot(X(y==1, 1), X(y==1, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1.5, ...
    'Color', 'blue');

% Plot decision boundaries
contour(teX1, teX2, teY, [1,1], 'black');

% Plot predictions
plot(teX1(teY==0), teX2(teY==0), '+', 'MarkerSize', 2, 'LineWidth', 0.1, ...
    'Color', 'red');
plot(teX1(teY==1), teX2(teY==1), '+', 'MarkerSize', 2, 'LineWidth', 0.1, ...
    'Color', 'blue');
