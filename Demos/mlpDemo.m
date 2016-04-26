%% Regression demo
clear all;
load('mlp_regression_data.mat');

% initialise model
m = multiLayerPerceptron(X, y, [1, 3, 3, 1], 'tanh', 'regression');

% train
m.train(2000, 0.3, 0);

% make predictions
predictions = zeros(length(X), 1);
for i = 1 : length(X)
    predictions(i) = m.predict(X(i, :));
end

% plot comparison
figure
grid on;
plot(X, y, '.', 'color', 'blue');
hold on;
plot(X, predictions, 'color', 'red');

%% Classification demo
clear all;
load('mlp_classification_data.mat');

% initialise model
m = multiLayerPerceptron(X, y, [2, 5, 5, 1], 'tanh', 'classification');

% train
m.train(100, 0.1, 0);

% plot comparison
figure
h = min(X(:,1)):.1:max(X(:,1));
w = min(X(:,2)):.1:max(X(:,2));
[hx,wx] = meshgrid(h,w);

% make predictions
predictions = zeros(size(hx));
for i = 1 : length(w)
    for j = 1 : length(h)
        predictions(i, j) = m.predict([hx(i, j), wx(i, j)]);
    end
end

contourf(hx, wx, predictions, 1);
colormap(winter);
% plot indiviual data points
hold on;
blue = [0.05 0.05 1];
red = [1 0.05 0.05];
plot(X(y==0, 1), X(y==0, 2), 'xr', 'color', blue, 'linewidth', ...
   2, 'markerfacecolor', blue);
hold on;
plot(X(y==1, 1), X(y==1, 2),'or','color', ... 
    red,'linewidth', 2, 'markerfacecolor', red);
