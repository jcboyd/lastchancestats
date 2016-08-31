clear all;
rng(0,'twister');

load('svm_data.mat');

m = supportVectorMachine(X, y, 'rbf');
m.train(1, 1e-5, 10);

% plot comparison
figure
h = min(X(:,1)):.1:max(X(:,1));
w = min(X(:,2)):.1:max(X(:,2));
[hx,wx] = meshgrid(h,w);

% make predictions
fprintf('\nPlotting predictions...\n');
predictions = zeros(size(hx));
for i = 1 : length(w)
    for j = 1 : length(h)
        predictions(i, j) = m.predict([hx(i, j), wx(i, j)]);
    end
end

contourf(hx, wx, predictions, 1);
colormap(gray);
% plot indiviual data points
hold on;
blue = [0.05 0.05 1];
red = [1 0.05 0.05];
plot(X(y==-1, 1), X(y==-1, 2), 'xr', 'color', blue, 'linewidth', ...
   2, 'markerfacecolor', blue);
hold on;
plot(X(y==1, 1), X(y==1, 2),'or','color', ... 
    red,'linewidth', 2, 'markerfacecolor', red);
