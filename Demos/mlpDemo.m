clear all;

% generate data
X = linspace(-1, 1, 50)';
y = abs(X);

% initialise model
m = multiLayerPerceptron(X, y, [2, 3, 3, 1], 'tanh', 'regression');

% train
m.train(2000, 0.2, 0);

% makep predictions
predictions = zeros(length(X), 1);
for i = 1 : length(X)
    predictions(i) = m.predict(X(i, :));
end

% plot comparison
plot(X, y, '.', 'color', 'blue');
hold on;
plot(X, predictions, '*', 'color', 'red');
