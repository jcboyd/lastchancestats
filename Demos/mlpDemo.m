clear all;

load('regression_data');
m = multiLayerPerceptron(X, y, 2, 'tanh', 'regression');
m.train(1000, 0.001);
