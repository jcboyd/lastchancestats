clear all;

load('regression_data');
% X = linspace(-1, 1, 20)';
% y = abs(X) + 1;
tX = [ones(length(X), 1), X];
m = multiLayerPerceptron(tX, y, [2, 2], 'tanh', 'regression');
m.train(200, 0.01);
